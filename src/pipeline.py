"""Pipeline DRAG plugavel: etapas OO configuraveis por servico via YAML."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import redis as redis_lib
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_redis import RedisConfig, RedisVectorStore

from src.config import ServiceConfig, env
from src.embeddings import get_embeddings
from src.llm.provider import LLMProvider, is_api_exhausted
from src.steps import STEP_REGISTRY, _load_all_steps
from src.steps.base import BaseStep, PipelineContext, PipelineData

log = logging.getLogger("drag.pipeline")


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class DragResult:
    """Resultado completo de uma invocacao do pipeline DRAG."""
    resposta: str
    chunks: list
    contexto: str
    generated_queries: list[str] = field(default_factory=list)
    raw_count: int = 0
    rerank_count: int = 0
    timings: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt padrao do MultiQuery
# ---------------------------------------------------------------------------

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Voce e um assistente que gera variacoes de perguntas para busca semantica. "
     "Dado a pergunta original, gere {n} variacoes alternativas que capturem "
     "diferentes perspectivas ou reformulacoes da mesma intencao. "
     "Retorne APENAS as variacoes, uma por linha, sem numeracao."),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# PipelineBuilder — constroi todos os componentes do pipeline (SRP)
# ---------------------------------------------------------------------------

class PipelineBuilder:
    """Constroi componentes (vectorstores, retrievers, chains) para o pipeline."""

    def __init__(self, config: ServiceConfig, redis_conn: redis_lib.Redis,
                 llm_provider: LLMProvider):
        self.config = config
        self.redis = redis_conn
        self.llm_provider = llm_provider

    def build_vectorstore(self, index_name: str) -> RedisVectorStore:
        try:
            self.redis.ping()
        except redis_lib.exceptions.ConnectionError:
            raise ConnectionError(
                f"Nao foi possivel conectar ao Redis em '{env.REDIS_URL}'. "
                f"Verifique se o container esta rodando: docker compose up -d"
            )

        embeddings = get_embeddings(self.config.embedding_model)
        return RedisVectorStore(
            embeddings=embeddings,
            config=RedisConfig(
                index_name=index_name,
                redis_url=env.REDIS_URL,
                from_existing=True,
            ),
        )

    def build_exercicios_vectorstore(self) -> RedisVectorStore | None:
        try:
            return self.build_vectorstore(self.config.exercicios_index_name)
        except Exception:
            return None

    def build_multi_retriever(self, base_retriever) -> MultiQueryRetriever:
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm_provider.llm,
            prompt=MULTI_QUERY_PROMPT.partial(n=str(self.config.multi_query_n)),
        )

    def build_answer_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            ("human", self.config.human_prompt),
        ])
        return prompt | self.llm_provider.llm | StrOutputParser()

    def build_reranker(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(self.config.rerank_model)

    def build_context(self, *, reranker, base_retriever, multi_retriever,
                      vectorstore, exercicios_vectorstore,
                      answer_chain) -> PipelineContext:
        return PipelineContext(
            config=self.config,
            redis=self.redis,
            llm=self.llm_provider.llm,
            reranker=reranker,
            base_retriever=base_retriever,
            multi_retriever=multi_retriever,
            vectorstore=vectorstore,
            answer_chain=answer_chain,
            exercicios_vectorstore=exercicios_vectorstore,
        )

    def build_steps(self, ctx: PipelineContext) -> list[BaseStep]:
        instances = []
        for step_name in self.config.pipeline_steps:
            step_cls = STEP_REGISTRY.get(step_name)
            if step_cls is None:
                log.warning("Step '%s' nao encontrado no STEP_REGISTRY.", step_name)
                continue
            instances.append(step_cls(ctx))
        return instances


# ---------------------------------------------------------------------------
# Labels de status para o spinner
# ---------------------------------------------------------------------------

def _get_step_label(step: BaseStep) -> str:
    return step.label or step.name


# ---------------------------------------------------------------------------
# DragPipeline — orquestrador (apenas executa steps)
# ---------------------------------------------------------------------------

class DragPipeline:
    """Pipeline DRAG: orquestra a execucao de steps na ordem do YAML."""

    def __init__(self, config: ServiceConfig):
        self.config = config

        _load_all_steps()

        # Dependencias base
        self._llm_provider = LLMProvider(config.llm_provider, config.llm_model)
        self._redis = redis_lib.from_url(env.REDIS_URL)
        self._builder = PipelineBuilder(config, self._redis, self._llm_provider)

        # Constroi componentes via builder
        self._reranker = self._builder.build_reranker()
        self._vectorstore = self._builder.build_vectorstore(config.index_name)
        self._exercicios_vectorstore = self._builder.build_exercicios_vectorstore()
        self._base_retriever = self._vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": config.retriever_k}
        )
        self._multi_retriever = self._builder.build_multi_retriever(self._base_retriever)
        self._answer_chain = self._builder.build_answer_chain()

        # Monta steps
        self._step_instances = self._rebuild_steps()

    @property
    def active_provider(self) -> str:
        return self._llm_provider.active_provider

    @property
    def active_model(self) -> str:
        return self._llm_provider.active_model

    def _rebuild_steps(self) -> list[BaseStep]:
        ctx = self._builder.build_context(
            reranker=self._reranker,
            base_retriever=self._base_retriever,
            multi_retriever=self._multi_retriever,
            vectorstore=self._vectorstore,
            exercicios_vectorstore=self._exercicios_vectorstore,
            answer_chain=self._answer_chain,
        )
        return self._builder.build_steps(ctx)

    # --- Switch de modelo ---

    def switch_model(self, provider: str, model: str):
        """Troca o LLM ativo em runtime e recria dependencias."""
        self._llm_provider.switch(provider, model)
        self._multi_retriever = self._builder.build_multi_retriever(self._base_retriever)
        self._answer_chain = self._builder.build_answer_chain()
        self._step_instances = self._rebuild_steps()

    def _fallback_to_ollama(self):
        self._llm_provider.fallback_to_local()
        self._multi_retriever = self._builder.build_multi_retriever(self._base_retriever)
        self._answer_chain = self._builder.build_answer_chain()
        self._step_instances = self._rebuild_steps()

    # --- Invocacao publica ---

    def invoke(self, pergunta: str, status: list | None = None,
               fontes_selecionadas: list[str] | None = None) -> DragResult:
        """Executa as etapas definidas no config.pipeline_steps."""
        import time

        total = len(self._step_instances)
        pipeline_start = time.perf_counter()

        data = PipelineData(
            pergunta=pergunta,
            fontes_selecionadas=fontes_selecionadas or [],
        )

        for i, step in enumerate(self._step_instances, 1):
            if status is not None:
                status[0] = f"[{i}/{total}] {_get_step_label(step)}..."

            try:
                data = step.execute(data)
            except Exception as e:
                log.error("Erro no step '%s': [%s] %s", step.name, type(e).__name__, e)
                if self.active_provider != "ollama" and is_api_exhausted(e):
                    log.warning("Fallback ativado para Ollama.")
                    if status is not None:
                        status[0] = "API esgotada — alternando para modelo local..."
                    self._fallback_to_ollama()
                    data = self._step_instances[i - 1].execute(data)
                else:
                    raise

        total_time = time.perf_counter() - pipeline_start
        data.timings["pipeline_total"] = total_time

        return DragResult(
            resposta=data.resposta,
            chunks=data.chunks,
            contexto=data.contexto,
            generated_queries=data.generated_queries,
            raw_count=data.raw_count,
            rerank_count=data.rerank_count,
            timings=data.timings,
        )


# ---------------------------------------------------------------------------
# Cache e atalhos
# ---------------------------------------------------------------------------

_pipeline_cache: dict[str, DragPipeline] = {}


def get_pipeline(config: ServiceConfig) -> DragPipeline:
    if config.name not in _pipeline_cache:
        _pipeline_cache[config.name] = DragPipeline(config)
    return _pipeline_cache[config.name]


def invalidate_pipeline(service_name: str):
    """Remove pipeline do cache (ex: apos re-ingestao)."""
    _pipeline_cache.pop(service_name, None)


def ask(config: ServiceConfig, question: str) -> str:
    result = get_pipeline(config).invoke(question)
    return result.resposta
