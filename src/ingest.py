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
from langchain_redis import RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ServiceConfig, env
from src.embeddings import get_embeddings


def load_documents(docs_dir: str) -> list:
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


def split_documents(documents: list, config: ServiceConfig) -> list:
    """Divide documentos em chunks menores e atribui chunk_index por fonte."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
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
    log_chunks_gerados(chunks, config.chunk_size, config.chunk_overlap)

    return chunks


def _compute_docs_hash(docs_dir: str) -> str:
    """Calcula hash SHA-256 dos arquivos incluindo amostra do conteúdo."""
    docs_path = Path(docs_dir)
    hasher = hashlib.sha256()
    for filepath in sorted(docs_path.rglob("*")):
        if filepath.is_file() and filepath.suffix.lower() in (".pdf", ".txt"):
            hasher.update(filepath.name.encode())
            hasher.update(str(filepath.stat().st_size).encode())
            with open(filepath, "rb") as f:
                hasher.update(f.read(8192))
    return hasher.hexdigest()


def _get_stored_hash(config: ServiceConfig) -> str | None:
    """Recupera o hash da última ingestão armazenado no Redis."""
    try:
        client = redis_lib.from_url(env.REDIS_URL)
        stored = client.get(config.hash_key)
        return stored.decode("utf-8") if stored else None
    except Exception:
        return None


def _store_hash(config: ServiceConfig, docs_hash: str):
    """Armazena o hash da ingestão atual no Redis."""
    client = redis_lib.from_url(env.REDIS_URL)
    client.set(config.hash_key, docs_hash)


def _drop_existing_index(index_name: str):
    """Remove índice e documentos antigos do Redis para evitar duplicatas."""
    try:
        client = redis_lib.from_url(env.REDIS_URL)
        client.ft(index_name).dropindex(delete_documents=True)
    except Exception:
        pass


def index_documents(chunks: list, config: ServiceConfig) -> RedisVectorStore:
    """Indexa os chunks no Redis como banco vetorial + hash para vizinhos."""
    embeddings = get_embeddings(config.embedding_model)

    _drop_existing_index(config.index_name)

    vectorstore = RedisVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        redis_url=env.REDIS_URL,
        index_name=config.index_name,
    )

    # Salva chunks num hash Redis para busca de vizinhos
    client = redis_lib.from_url(env.REDIS_URL)
    client.delete(config.chunks_map_key)
    for chunk in chunks:
        source = chunk.metadata.get("source", "desconhecido")
        idx = chunk.metadata.get("chunk_index", 0)
        key = f"{source}:{idx}"
        client.hset(config.chunks_map_key, key, chunk.page_content)

    from src.logger import log_indexacao_redis
    log_indexacao_redis(len(chunks), env.REDIS_URL, config.index_name)

    return vectorstore


def run_ingest(config: ServiceConfig, force: bool = False) -> RedisVectorStore | None:
    """Pipeline completo de ingestão. Pula se os documentos não mudaram."""
    # Validação de conexão Redis
    try:
        client = redis_lib.from_url(env.REDIS_URL)
        client.ping()
    except redis_lib.exceptions.ConnectionError:
        raise ConnectionError(
            f"Não foi possível conectar ao Redis em '{env.REDIS_URL}'. "
            f"Verifique se o container está rodando: docker compose up -d"
        )

    # Ingestão incremental
    if not force:
        current_hash = _compute_docs_hash(config.docs_dir)
        stored_hash = _get_stored_hash(config)
        if current_hash == stored_hash:
            from src.logger import log_alerta
            log_alerta("Documentos não mudaram desde a última ingestão. Pulando.")
            return None

    from src.logger import log_embeddings_carregado
    embeddings = get_embeddings(config.embedding_model)
    log_embeddings_carregado(config.embedding_model)

    documents = load_documents(config.docs_dir)
    chunks = split_documents(documents, config)
    vectorstore = index_documents(chunks, config)

    _store_hash(config, _compute_docs_hash(config.docs_dir))

    return vectorstore
