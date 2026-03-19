"""Módulo de ingestão: carrega documentos, divide em chunks e indexa no Redis."""

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


def get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    from src.logger import log_embeddings_carregado

    model_name = model_name or os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )
    log_embeddings_carregado(model_name)
    return embeddings


def load_documents(docs_dir: str = "fontes") -> list:
    """Carrega PDFs e arquivos .txt do diretório informado."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Diretório '{docs_dir}' não encontrado.")

    documents = []

    # PDFs
    pdf_loader = DirectoryLoader(
        str(docs_path), glob="**/*.pdf", loader_cls=UnstructuredPDFLoader,
        silent_errors=True
    )
    documents.extend(pdf_loader.load())

    # Texto
    txt_loader = DirectoryLoader(
        str(docs_path), glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}, silent_errors=True
    )
    documents.extend(txt_loader.load())

    if not documents:
        raise ValueError(f"Nenhum documento encontrado em '{docs_dir}'.")

    from src.logger import log_documentos_carregados
    fontes = list(set(doc.metadata.get("source", "?") for doc in documents))
    log_documentos_carregados(len(documents), sorted(fontes))

    return documents


def split_documents(documents: list) -> list:
    """Divide documentos em chunks menores."""
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

    from src.logger import log_chunks_gerados
    log_chunks_gerados(chunks, chunk_size, chunk_overlap)

    return chunks


def _drop_existing_index(redis_url: str, index_name: str):
    """Remove índice e documentos antigos do Redis para evitar duplicatas."""
    import redis
    try:
        client = redis.from_url(redis_url)
        client.ft(index_name).dropindex(delete_documents=True)
    except Exception:
        pass  # índice não existe ainda


def index_documents(chunks: list, embeddings: HuggingFaceEmbeddings) -> RedisVectorStore:
    """Indexa os chunks no Redis como banco vetorial."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    _drop_existing_index(redis_url, "rag_docs")

    vectorstore = RedisVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        redis_url=redis_url,
        index_name="rag_docs",
    )
    from src.logger import log_indexacao_redis
    log_indexacao_redis(len(chunks), redis_url)

    return vectorstore


def run_ingest(docs_dir: str = "fontes") -> RedisVectorStore:
    """Pipeline completo de ingestão."""
    embeddings = get_embeddings()
    documents = load_documents(docs_dir)
    chunks = split_documents(documents)
    vectorstore = index_documents(chunks, embeddings)
    return vectorstore


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    run_ingest()
