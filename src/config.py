"""Configuração centralizada: globals do .env + ServiceConfig por YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SERVICES_DIR = Path(__file__).parent.parent / "services"


# ---------------------------------------------------------------------------
# Globals (compartilhados entre todos os serviços — lidos do .env)
# ---------------------------------------------------------------------------

class _Globals:
    """Variáveis de ambiente globais (Redis, Ollama). Leitura lazy."""

    @property
    def REDIS_URL(self) -> str:
        return os.getenv("REDIS_URL", "redis://localhost:6379")

    @property
    def OLLAMA_BASE_URL(self) -> str:
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


env = _Globals()


# ---------------------------------------------------------------------------
# ServiceConfig (específico por serviço — lido do YAML)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServiceConfig:
    """Configuração de um serviço DRAG, carregada de um arquivo YAML."""

    # Identidade
    name: str
    display_name: str
    docs_dir: str

    # Modelos
    llm_model: str
    embedding_model: str
    rerank_model: str

    # Parâmetros de ingestão
    chunk_size: int
    chunk_overlap: int

    # Parâmetros de busca
    retriever_k: int
    rerank_top_n: int
    multi_query_n: int
    neighbor_window: int

    # Pipeline
    pipeline_steps: tuple[str, ...] = ()

    # Prompts
    system_prompt: str = ""
    human_prompt: str = "{question}"

    # Derivados (calculados a partir do name)
    index_name: str = ""
    chunks_map_key: str = ""
    hash_key: str = ""

    def __post_init__(self):
        # frozen=True exige object.__setattr__ para derivados
        if not self.index_name:
            object.__setattr__(self, "index_name", f"rag_{self.name}")
        if not self.chunks_map_key:
            object.__setattr__(self, "chunks_map_key", f"rag_chunks_map_{self.name}")
        if not self.hash_key:
            object.__setattr__(self, "hash_key", f"rag_docs_hash_{self.name}")

    @classmethod
    def load(cls, name: str) -> ServiceConfig:
        """Carrega config de services/{name}.yaml."""
        path = SERVICES_DIR / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Serviço '{name}' não encontrado. "
                f"Crie o arquivo: {path}"
            )

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            docs_dir=data.get("docs_dir", "fontes"),
            llm_model=data.get("llm_model", "qwen2.5:14b"),
            embedding_model=data.get("embedding_model", "intfloat/multilingual-e5-base"),
            rerank_model=data.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            chunk_size=int(data.get("chunk_size", 1500)),
            chunk_overlap=int(data.get("chunk_overlap", 300)),
            retriever_k=int(data.get("retriever_k", 20)),
            rerank_top_n=int(data.get("rerank_top_n", 5)),
            multi_query_n=int(data.get("multi_query_n", 3)),
            neighbor_window=int(data.get("neighbor_window", 2)),
            pipeline_steps=tuple(data.get("pipeline_steps", [
                "multi_query", "retrieve", "neighbors", "rerank", "generate", "latex_to_unicode",
            ])),
            system_prompt=data.get("system_prompt", ""),
            human_prompt=data.get("human_prompt", "{question}"),
        )

    def as_dict(self) -> dict[str, str]:
        """Retorna configuração completa para logging."""
        return {
            "service": self.name,
            "display_name": self.display_name,
            "docs_dir": self.docs_dir,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "rerank_model": self.rerank_model,
            "chunk_size": str(self.chunk_size),
            "chunk_overlap": str(self.chunk_overlap),
            "retriever_k": str(self.retriever_k),
            "rerank_top_n": str(self.rerank_top_n),
            "multi_query_n": str(self.multi_query_n),
            "neighbor_window": str(self.neighbor_window),
            "pipeline_steps": ", ".join(self.pipeline_steps),
            "index_name": self.index_name,
        }


def list_services() -> list[str]:
    """Retorna nomes dos serviços disponíveis (arquivos YAML em services/)."""
    if not SERVICES_DIR.exists():
        return []
    return sorted(p.stem for p in SERVICES_DIR.glob("*.yaml"))


def get_default_service() -> str:
    """Retorna o serviço padrão. Se só há 1, usa ele. Senão, 'default'."""
    services = list_services()
    if len(services) == 1:
        return services[0]
    return "default"
