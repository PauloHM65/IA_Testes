"""Configuração centralizada: lê .env via os.getenv com acesso lazy."""

from __future__ import annotations

import os
from functools import lru_cache


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _str(key: str, default: str) -> str:
    return os.getenv(key, default)


# Propriedades de módulo com acesso lazy (lê os.getenv no momento do acesso,
# não no momento do import — assim load_dotenv() no main.py tem efeito).

class _Config:
    """Proxy que lê variáveis de ambiente sob demanda."""

    @property
    def REDIS_URL(self) -> str:
        return _str("REDIS_URL", "redis://localhost:6379")

    @property
    def OLLAMA_BASE_URL(self) -> str:
        return _str("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def LLM_MODEL(self) -> str:
        return _str("LLM_MODEL", "qwen2.5:14b")

    @property
    def EMBEDDING_MODEL(self) -> str:
        return _str("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

    @property
    def RERANK_MODEL(self) -> str:
        return _str("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    @property
    def RERANK_TOP_N(self) -> int:
        return _int("RERANK_TOP_N", 5)

    @property
    def RETRIEVER_K(self) -> int:
        return _int("RETRIEVER_K", 20)

    @property
    def MULTI_QUERY_N(self) -> int:
        return _int("MULTI_QUERY_N", 3)

    @property
    def NEIGHBOR_WINDOW(self) -> int:
        return _int("NEIGHBOR_WINDOW", 2)

    @property
    def CHUNK_SIZE(self) -> int:
        return _int("CHUNK_SIZE", 1500)

    @property
    def CHUNK_OVERLAP(self) -> int:
        return _int("CHUNK_OVERLAP", 300)

    def as_dict(self) -> dict[str, str]:
        """Retorna todas as configurações como dict para logging."""
        return {
            "REDIS_URL": self.REDIS_URL,
            "OLLAMA_BASE_URL": self.OLLAMA_BASE_URL,
            "LLM_MODEL": self.LLM_MODEL,
            "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
            "RERANK_MODEL": self.RERANK_MODEL,
            "RERANK_TOP_N": str(self.RERANK_TOP_N),
            "RETRIEVER_K": str(self.RETRIEVER_K),
            "MULTI_QUERY_N": str(self.MULTI_QUERY_N),
            "NEIGHBOR_WINDOW": str(self.NEIGHBOR_WINDOW),
            "CHUNK_SIZE": str(self.CHUNK_SIZE),
            "CHUNK_OVERLAP": str(self.CHUNK_OVERLAP),
        }


# Instância singleton — importar como: from src.config import cfg
cfg = _Config()
