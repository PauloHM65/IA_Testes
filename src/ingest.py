"""Módulo de ingestão: carrega documentos, divide em chunks e indexa no Redis."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredPDFLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


_embeddings_cache: HuggingFaceEmbeddings | None = None


def get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache

    from src.logger import log_embeddings_carregado

    model_name = model_name or os.getenv(
        "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
    )

    _embeddings_cache = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )

    # Modelos E5 precisam de prefixo "query: " / "passage: " para funcionar bem.
    # Pydantic não permite sobrescrever métodos, então usamos object.__setattr__.
    if "e5" in model_name.lower():
        _original_embed_query = _embeddings_cache.embed_query
        _original_embed_documents = _embeddings_cache.embed_documents

        def _e5_embed_query(text: str) -> list[float]:
            return _original_embed_query(f"query: {text}")

        def _e5_embed_documents(texts: list[str]) -> list[list[float]]:
            return _original_embed_documents([f"passage: {t}" for t in texts])

        object.__setattr__(_embeddings_cache, "embed_query", _e5_embed_query)
        object.__setattr__(_embeddings_cache, "embed_documents", _e5_embed_documents)

    log_embeddings_carregado(model_name)
    return _embeddings_cache


def load_documents(docs_dir: str = "fontes") -> list:
    """Carrega PDFs e arquivos .txt do diretório informado."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Diretório '{docs_dir}' não encontrado.")

    documents = []
    failed_files = []

    # PDFs
    pdf_loader = DirectoryLoader(
        str(docs_path), glob="**/*.pdf", loader_cls=UnstructuredPDFLoader,
        silent_errors=True
    )
    pdf_docs = pdf_loader.load()
    # Verifica PDFs vazios/corrompidos
    for doc in pdf_docs:
        if doc.page_content.strip():
            documents.append(doc)
        else:
            failed_files.append(doc.metadata.get("source", "?"))

    # Texto
    txt_loader = DirectoryLoader(
        str(docs_path), glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}, silent_errors=True
    )
    documents.extend(txt_loader.load())

    if failed_files:
        from src.logger import log_alerta
        log_alerta(f"PDFs vazios ou corrompidos ignorados: {', '.join(failed_files)}")

    if not documents:
        raise ValueError(f"Nenhum documento encontrado em '{docs_dir}'.")

    from src.logger import log_documentos_carregados
    fontes = list(set(doc.metadata.get("source", "?") for doc in documents))
    log_documentos_carregados(len(documents), sorted(fontes))

    return documents


def split_documents(documents: list) -> list:
    """Divide documentos em chunks menores e atribui chunk_index por fonte."""
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    # Normaliza espaços em branco: colapsa tabs/espaços múltiplos e linhas vazias
    for chunk in chunks:
        text = chunk.page_content
        text = re.sub(r"[^\S\n]+", " ", text)       # tabs e espaços múltiplos -> 1 espaço
        text = re.sub(r" *\n *", "\n", text)         # espaços ao redor de quebras de linha
        text = re.sub(r"\n{3,}", "\n\n", text)       # 3+ quebras de linha -> 2
        chunk.page_content = text.strip()

    # Atribui chunk_index sequencial por fonte (para busca de vizinhos)
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "desconhecido")
        idx = source_counters.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        source_counters[source] = idx + 1

    from src.logger import log_chunks_gerados
    log_chunks_gerados(chunks, chunk_size, chunk_overlap)

    return chunks


def _compute_docs_hash(docs_dir: str) -> str:
    """Calcula hash MD5 combinado de todos os arquivos do diretório de documentos."""
    docs_path = Path(docs_dir)
    hasher = hashlib.md5()
    for filepath in sorted(docs_path.rglob("*")):
        if filepath.is_file() and filepath.suffix.lower() in (".pdf", ".txt"):
            hasher.update(filepath.name.encode())
            hasher.update(str(filepath.stat().st_size).encode())
            hasher.update(str(filepath.stat().st_mtime_ns).encode())
    return hasher.hexdigest()


def _get_stored_hash(redis_url: str) -> str | None:
    """Recupera o hash da última ingestão armazenado no Redis."""
    import redis
    try:
        client = redis.from_url(redis_url)
        stored = client.get("rag_docs_hash")
        return stored.decode("utf-8") if stored else None
    except Exception:
        return None


def _store_hash(redis_url: str, docs_hash: str):
    """Armazena o hash da ingestão atual no Redis."""
    import redis
    client = redis.from_url(redis_url)
    client.set("rag_docs_hash", docs_hash)


def _drop_existing_index(redis_url: str, index_name: str):
    """Remove índice e documentos antigos do Redis para evitar duplicatas."""
    import redis
    try:
        client = redis.from_url(redis_url)
        client.ft(index_name).dropindex(delete_documents=True)
    except Exception:
        pass  # índice não existe ainda


def index_documents(chunks: list, embeddings: HuggingFaceEmbeddings) -> RedisVectorStore:
    """Indexa os chunks no Redis como banco vetorial + hash para vizinhos."""
    import redis as redis_lib

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    _drop_existing_index(redis_url, "rag_docs")

    vectorstore = RedisVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        redis_url=redis_url,
        index_name="rag_docs",
    )

    # Salva chunks num hash Redis para busca de vizinhos: chave = "chunk:{source}:{index}"
    client = redis_lib.from_url(redis_url)
    client.delete("rag_chunks_map")
    for chunk in chunks:
        source = chunk.metadata.get("source", "desconhecido")
        idx = chunk.metadata.get("chunk_index", 0)
        key = f"{source}:{idx}"
        client.hset("rag_chunks_map", key, chunk.page_content)

    from src.logger import log_indexacao_redis
    log_indexacao_redis(len(chunks), redis_url)

    return vectorstore


def run_ingest(docs_dir: str = "fontes", force: bool = False) -> RedisVectorStore | None:
    """Pipeline completo de ingestão. Pula se os documentos não mudaram."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Validação de conexão Redis
    import redis
    try:
        client = redis.from_url(redis_url)
        client.ping()
    except redis.exceptions.ConnectionError:
        raise ConnectionError(
            f"Não foi possível conectar ao Redis em '{redis_url}'. "
            f"Verifique se o container está rodando: docker compose up -d"
        )

    # Ingestão incremental: verifica se docs mudaram
    if not force:
        current_hash = _compute_docs_hash(docs_dir)
        stored_hash = _get_stored_hash(redis_url)
        if current_hash == stored_hash:
            from src.logger import log_alerta
            log_alerta("Documentos não mudaram desde a última ingestão. Pulando.")
            return None

    embeddings = get_embeddings()
    documents = load_documents(docs_dir)
    chunks = split_documents(documents)
    vectorstore = index_documents(chunks, embeddings)

    # Salva hash para próxima verificação
    current_hash = _compute_docs_hash(docs_dir)
    _store_hash(redis_url, current_hash)

    return vectorstore


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    run_ingest()
