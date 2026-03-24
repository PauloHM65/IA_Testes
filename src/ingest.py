"""Módulo de ingestão: carrega documentos, divide em chunks e indexa no Redis."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import redis as redis_lib
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredPDFLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import cfg


class E5Embeddings(HuggingFaceEmbeddings):
    """Wrapper que adiciona os prefixos 'query: ' / 'passage: ' exigidos por modelos E5."""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])


_embeddings_cache: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache

    from src.logger import log_embeddings_carregado

    model_name = cfg.EMBEDDING_MODEL
    cls = E5Embeddings if "e5" in model_name.lower() else HuggingFaceEmbeddings

    _embeddings_cache = cls(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )

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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    # Normaliza espaços em branco
    for chunk in chunks:
        text = chunk.page_content
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        chunk.page_content = text.strip()

    # Atribui chunk_index sequencial por fonte (para busca de vizinhos)
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "desconhecido")
        idx = source_counters.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        source_counters[source] = idx + 1

    from src.logger import log_chunks_gerados
    log_chunks_gerados(chunks, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)

    return chunks


def _compute_docs_hash(docs_dir: str) -> str:
    """Calcula hash SHA-256 dos arquivos incluindo amostra do conteúdo."""
    docs_path = Path(docs_dir)
    hasher = hashlib.sha256()
    for filepath in sorted(docs_path.rglob("*")):
        if filepath.is_file() and filepath.suffix.lower() in (".pdf", ".txt"):
            hasher.update(filepath.name.encode())
            hasher.update(str(filepath.stat().st_size).encode())
            # Lê primeiros 8KB do conteúdo para detectar alterações reais
            with open(filepath, "rb") as f:
                hasher.update(f.read(8192))
    return hasher.hexdigest()


def _get_stored_hash() -> str | None:
    """Recupera o hash da última ingestão armazenado no Redis."""
    try:
        client = redis_lib.from_url(cfg.REDIS_URL)
        stored = client.get("rag_docs_hash")
        return stored.decode("utf-8") if stored else None
    except Exception:
        return None


def _store_hash(docs_hash: str):
    """Armazena o hash da ingestão atual no Redis."""
    client = redis_lib.from_url(cfg.REDIS_URL)
    client.set("rag_docs_hash", docs_hash)


def _drop_existing_index(index_name: str):
    """Remove índice e documentos antigos do Redis para evitar duplicatas."""
    try:
        client = redis_lib.from_url(cfg.REDIS_URL)
        client.ft(index_name).dropindex(delete_documents=True)
    except Exception:
        pass


def index_documents(chunks: list, embeddings: HuggingFaceEmbeddings) -> RedisVectorStore:
    """Indexa os chunks no Redis como banco vetorial + hash para vizinhos."""
    _drop_existing_index("rag_docs")

    vectorstore = RedisVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        redis_url=cfg.REDIS_URL,
        index_name="rag_docs",
    )

    # Salva chunks num hash Redis para busca de vizinhos
    client = redis_lib.from_url(cfg.REDIS_URL)
    client.delete("rag_chunks_map")
    for chunk in chunks:
        source = chunk.metadata.get("source", "desconhecido")
        idx = chunk.metadata.get("chunk_index", 0)
        key = f"{source}:{idx}"
        client.hset("rag_chunks_map", key, chunk.page_content)

    from src.logger import log_indexacao_redis
    log_indexacao_redis(len(chunks), cfg.REDIS_URL)

    return vectorstore


def run_ingest(docs_dir: str = "fontes", force: bool = False) -> RedisVectorStore | None:
    """Pipeline completo de ingestão. Pula se os documentos não mudaram."""
    # Validação de conexão Redis
    try:
        client = redis_lib.from_url(cfg.REDIS_URL)
        client.ping()
    except redis_lib.exceptions.ConnectionError:
        raise ConnectionError(
            f"Não foi possível conectar ao Redis em '{cfg.REDIS_URL}'. "
            f"Verifique se o container está rodando: docker compose up -d"
        )

    # Ingestão incremental
    if not force:
        current_hash = _compute_docs_hash(docs_dir)
        stored_hash = _get_stored_hash()
        if current_hash == stored_hash:
            from src.logger import log_alerta
            log_alerta("Documentos não mudaram desde a última ingestão. Pulando.")
            return None

    embeddings = get_embeddings()
    documents = load_documents(docs_dir)
    chunks = split_documents(documents)
    vectorstore = index_documents(chunks, embeddings)

    _store_hash(_compute_docs_hash(docs_dir))

    return vectorstore


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    run_ingest()
