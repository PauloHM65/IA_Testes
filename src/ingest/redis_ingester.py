"""Ingester que indexa documentos no Redis Stack."""

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
from src.ingest.base import BaseIngester
from src.logging.base import IngestLogger


class RedisIngester(BaseIngester):
    """Ingere documentos em indices Redis (materias + exercicios)."""

    def __init__(self, config: ServiceConfig, logger: IngestLogger):
        super().__init__(config, logger)
        self._redis = redis_lib.from_url(env.REDIS_URL)

    def run(self, force: bool = False) -> bool | None:
        # Valida conexao
        try:
            self._redis.ping()
        except redis_lib.exceptions.ConnectionError:
            raise ConnectionError(
                f"Nao foi possivel conectar ao Redis em '{env.REDIS_URL}'. "
                f"Verifique se o container esta rodando: docker compose up -d"
            )

        # Ingestao incremental
        if not force:
            current_hash = self._compute_docs_hash(self.config.docs_dir)
            stored_hash = self._get_stored_hash()
            if current_hash == stored_hash:
                self.logger.alerta("Documentos nao mudaram desde a ultima ingestao. Pulando.")
                return None

        # Ingere materias
        self._ingest_directory(
            docs_dir=self.config.materias_dir,
            index_name=self.config.index_name,
            chunks_map_key=self.config.chunks_map_key,
            label="Materias",
        )

        # Ingere exercicios
        self._ingest_directory(
            docs_dir=self.config.exercicios_dir,
            index_name=self.config.exercicios_index_name,
            chunks_map_key=self.config.exercicios_chunks_map_key,
            label="Exercicios",
        )

        self._store_hash(self._compute_docs_hash(self.config.docs_dir))
        return True

    # --- Metodos internos ---

    def _ingest_directory(self, docs_dir: str, index_name: str,
                          chunks_map_key: str, label: str) -> RedisVectorStore | None:
        docs_path = Path(docs_dir)
        if not docs_path.exists() or not any(docs_path.rglob("*")):
            self.logger.alerta(f"{label}: diretorio '{docs_dir}' vazio ou nao existe. Pulando.")
            return None

        embeddings = get_embeddings(self.config.embedding_model)
        self.logger.embeddings_carregado(self.config.embedding_model)

        documents = self._load_documents(docs_dir)
        chunks = self._split_documents(documents)

        self._drop_index(index_name)

        vectorstore = RedisVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            redis_url=env.REDIS_URL,
            index_name=index_name,
        )

        self._redis.delete(chunks_map_key)
        for chunk in chunks:
            source = chunk.metadata.get("source", "desconhecido")
            idx = chunk.metadata.get("chunk_index", 0)
            key = f"{source}:{idx}"
            self._redis.hset(chunks_map_key, key, chunk.page_content)

        self.logger.indexacao_redis(len(chunks), env.REDIS_URL, index_name)
        return vectorstore

    def _load_documents(self, docs_dir: str) -> list:
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Diretorio '{docs_dir}' nao encontrado.")

        documents = []
        failed_files = []

        pdf_loader = DirectoryLoader(
            str(docs_path), glob="**/*.pdf", loader_cls=UnstructuredPDFLoader,
            silent_errors=True
        )
        for doc in pdf_loader.load():
            if doc.page_content.strip():
                documents.append(doc)
            else:
                failed_files.append(doc.metadata.get("source", "?"))

        txt_loader = DirectoryLoader(
            str(docs_path), glob="**/*.txt", loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True}, silent_errors=True
        )
        documents.extend(txt_loader.load())

        if failed_files:
            self.logger.alerta(f"PDFs vazios ou corrompidos ignorados: {', '.join(failed_files)}")

        if not documents:
            raise ValueError(f"Nenhum documento encontrado em '{docs_dir}'.")

        fontes = sorted(set(doc.metadata.get("source", "?") for doc in documents))
        self.logger.documentos_carregados(len(documents), fontes)
        return documents

    def _split_documents(self, documents: list) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            text = chunk.page_content
            text = re.sub(r"[^\S\n]+", " ", text)
            text = re.sub(r" *\n *", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            chunk.page_content = text.strip()

        source_counters: dict[str, int] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "desconhecido")
            idx = source_counters.get(source, 0)
            chunk.metadata["chunk_index"] = idx
            source_counters[source] = idx + 1

        self.logger.chunks_gerados(chunks, self.config.chunk_size, self.config.chunk_overlap)
        return chunks

    def _drop_index(self, index_name: str):
        try:
            self._redis.ft(index_name).dropindex(delete_documents=True)
        except Exception:
            pass

    @staticmethod
    def _compute_docs_hash(docs_dir: str) -> str:
        docs_path = Path(docs_dir)
        hasher = hashlib.sha256()
        for filepath in sorted(docs_path.rglob("*")):
            if filepath.is_file() and filepath.suffix.lower() in (".pdf", ".txt"):
                hasher.update(filepath.name.encode())
                hasher.update(str(filepath.stat().st_size).encode())
                with open(filepath, "rb") as f:
                    hasher.update(f.read(8192))
        return hasher.hexdigest()

    def _get_stored_hash(self) -> str | None:
        try:
            stored = self._redis.get(self.config.hash_key)
            return stored.decode("utf-8") if stored else None
        except Exception:
            return None

    def _store_hash(self, docs_hash: str):
        self._redis.set(self.config.hash_key, docs_hash)
