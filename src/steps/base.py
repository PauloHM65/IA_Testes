"""Classes base do pipeline: BaseStep (ABC), PipelineData, PipelineContext."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis as redis_lib
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable
    from langchain_redis import RedisVectorStore
    from sentence_transformers import CrossEncoder

    from src.config import ServiceConfig


# ---------------------------------------------------------------------------
# Dados transitorios entre etapas
# ---------------------------------------------------------------------------

@dataclass
class PipelineData:
    """Estado passado entre etapas do pipeline."""

    pergunta: str
    raw_chunks: list = field(default_factory=list)
    generated_queries: list[str] = field(default_factory=list)
    raw_count: int = 0
    chunks: list = field(default_factory=list)
    contexto: str = ""
    resposta: str = ""
    rerank_count: int = 0
    # Campos extras para pipeline de exercicios
    materia_identificada: str = ""
    documento_filtro: str = ""
    fontes_selecionadas: list[str] = field(default_factory=list)
    exercicio_texto: str = ""
    exercicio_chunks: list = field(default_factory=list)
    materia_chunks: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Contexto compartilhado (injetado nos steps)
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """Recursos compartilhados injetados nos steps via construtor."""

    config: ServiceConfig
    redis: redis_lib.Redis
    llm: BaseChatModel
    reranker: CrossEncoder
    base_retriever: object
    multi_retriever: object
    vectorstore: RedisVectorStore
    answer_chain: Runnable


# ---------------------------------------------------------------------------
# Classe base abstrata para steps
# ---------------------------------------------------------------------------

class BaseStep(ABC):
    """Step plugavel do pipeline. Subclasses definem name, label e execute()."""

    name: str = ""
    label: str = ""

    def __init__(self, ctx: PipelineContext):
        self.ctx = ctx

    @abstractmethod
    def execute(self, data: PipelineData) -> PipelineData:
        ...
