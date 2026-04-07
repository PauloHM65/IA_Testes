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
    # Tempos de execucao por etapa (preenchido pelo decorator @timed_step)
    timings: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Contextos segregados (ISP) — cada step usa apenas o que precisa
# ---------------------------------------------------------------------------

@dataclass
class LLMContext:
    """Recursos de LLM: modelo e config."""
    config: ServiceConfig
    llm: BaseChatModel


@dataclass
class RedisContext:
    """Recursos de Redis: conexao e config."""
    config: ServiceConfig
    redis: redis_lib.Redis


@dataclass
class RetrieverContext:
    """Recursos de retrieval: retrievers e vectorstores."""
    base_retriever: object
    multi_retriever: object
    vectorstore: RedisVectorStore
    exercicios_vectorstore: RedisVectorStore | None = None


@dataclass
class RankerContext:
    """Recursos de reranking."""
    config: ServiceConfig
    reranker: CrossEncoder


@dataclass
class ChainContext:
    """Recursos de chain (prompt + LLM)."""
    config: ServiceConfig
    answer_chain: Runnable


# ---------------------------------------------------------------------------
# Contexto completo (compoe todos os sub-contextos)
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """Recursos compartilhados injetados nos steps via construtor.

    Compoe todos os sub-contextos. Steps que precisam de apenas uma fatia
    podem acessar via propriedades tipadas (ex: ctx.llm_ctx).
    """

    config: ServiceConfig
    redis: redis_lib.Redis
    llm: BaseChatModel
    reranker: CrossEncoder
    base_retriever: object
    multi_retriever: object
    vectorstore: RedisVectorStore
    answer_chain: Runnable
    exercicios_vectorstore: RedisVectorStore | None = None

    @property
    def llm_ctx(self) -> LLMContext:
        return LLMContext(config=self.config, llm=self.llm)

    @property
    def redis_ctx(self) -> RedisContext:
        return RedisContext(config=self.config, redis=self.redis)

    @property
    def retriever_ctx(self) -> RetrieverContext:
        return RetrieverContext(
            base_retriever=self.base_retriever,
            multi_retriever=self.multi_retriever,
            vectorstore=self.vectorstore,
            exercicios_vectorstore=self.exercicios_vectorstore,
        )

    @property
    def ranker_ctx(self) -> RankerContext:
        return RankerContext(config=self.config, reranker=self.reranker)

    @property
    def chain_ctx(self) -> ChainContext:
        return ChainContext(config=self.config, answer_chain=self.answer_chain)


# ---------------------------------------------------------------------------
# Classe base abstrata para steps
# ---------------------------------------------------------------------------

class BaseStep(ABC):
    """Step plugavel do pipeline. Subclasses definem name, label e execute()."""

    name: str = ""
    label: str = ""

    def __init__(self, ctx: PipelineContext):
        self.ctx = ctx

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Aplica @timed_step automaticamente em toda subclasse concreta
        if "execute" in cls.__dict__ and not getattr(cls.__dict__["execute"], "__isabstractmethod__", False):
            from src.logging.performance import timed_step
            cls.execute = timed_step(cls.execute)

    @abstractmethod
    def execute(self, data: PipelineData) -> PipelineData:
        ...
