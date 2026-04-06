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
# Captura de queries geradas pelo MultiQueryRetriever
# ---------------------------------------------------------------------------

class _capture_multiquery:
    """Context manager que intercepta as queries geradas pelo MultiQueryRetriever."""

    def __init__(self, retriever: MultiQueryRetriever, output: list[str]):
        self._retriever = retriever
        self._output = output
        self._original = None

    def __enter__(self):
        original_generate = self._retriever.generate_queries

        def _capturing_generate(question, run_manager):
            queries = original_generate(question, run_manager)
            self._output.extend(queries)
            return queries

        self._original = original_generate
        self._retriever.generate_queries = _capturing_generate
        return self

    def __exit__(self, *exc):
        self._retriever.generate_queries = self._original


# ---------------------------------------------------------------------------
# Labels de status para o spinner
# ---------------------------------------------------------------------------

def _get_step_label(step: BaseStep) -> str:
    return step.label or step.name


# ---------------------------------------------------------------------------
# DragPipeline — executa steps OO na ordem do YAML
# ---------------------------------------------------------------------------

class DragPipeline:
    """Pipeline DRAG com etapas OO plugaveis definidas no ServiceConfig."""

    def __init__(self, config: ServiceConfig):
        self.config = config

        # Garante que todos os steps estao registrados
        _load_all_steps()

        # LLM Provider (multi-provider com fallback)
        self._llm_provider = LLMProvider(config.llm_provider, config.llm_model)

        # Componentes compartilhados
        self._redis = redis_lib.from_url(env.REDIS_URL)
        from sentence_transformers import CrossEncoder
        self._reranker = CrossEncoder(config.rerank_model)

        # Vectorstore + retrievers
        self._vectorstore = self._build_vectorstore()
        self._base_retriever = self._vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": config.retriever_k}
        )
        self._multi_retriever = self._build_multi_retriever()

        # Answer chain
        self._answer_chain = self._build_answer_chain()

        # Instancia os steps OO
        self._step_instances = self._build_steps()

    @property
    def active_provider(self) -> str:
        return self._llm_provider.active_provider

    @property
    def active_model(self) -> str:
        return self._llm_provider.active_model

    # --- Builders ---

    def _build_vectorstore(self) -> RedisVectorStore:
        try:
            self._redis.ping()
        except redis_lib.exceptions.ConnectionError:
            raise ConnectionError(
                f"Nao foi possivel conectar ao Redis em '{env.REDIS_URL}'. "
                f"Verifique se o container esta rodando: docker compose up -d"
            )

        embeddings = get_embeddings(self.config.embedding_model)
        return RedisVectorStore(
            embeddings=embeddings,
            config=RedisConfig(
                index_name=self.config.index_name,
                redis_url=env.REDIS_URL,
                from_existing=True,
            ),
        )

    def _build_multi_retriever(self) -> MultiQueryRetriever:
        return MultiQueryRetriever.from_llm(
            retriever=self._base_retriever,
            llm=self._llm_provider.llm,
            prompt=MULTI_QUERY_PROMPT.partial(n=str(self.config.multi_query_n)),
        )

    def _build_answer_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            ("human", self.config.human_prompt),
        ])
        return prompt | self._llm_provider.llm | StrOutputParser()

    def _build_context(self) -> PipelineContext:
        return PipelineContext(
            config=self.config,
            redis=self._redis,
            llm=self._llm_provider.llm,
            reranker=self._reranker,
            base_retriever=self._base_retriever,
            multi_retriever=self._multi_retriever,
            vectorstore=self._vectorstore,
            answer_chain=self._answer_chain,
        )

    def _build_steps(self) -> list[BaseStep]:
        ctx = self._build_context()
        instances = []
        for step_name in self.config.pipeline_steps:
            step_cls = STEP_REGISTRY.get(step_name)
            if step_cls is None:
                log.warning("Step '%s' nao encontrado no STEP_REGISTRY.", step_name)
                continue
            instances.append(step_cls(ctx))
        return instances

    # --- Switch de modelo ---

    def switch_model(self, provider: str, model: str):
        """Troca o LLM ativo em runtime e recria dependencias."""
        self._llm_provider.switch(provider, model)
        self._multi_retriever = self._build_multi_retriever()
        self._answer_chain = self._build_answer_chain()
        # Recria contexto nos steps
        self._step_instances = self._build_steps()

    def _fallback_to_ollama(self):
        self._llm_provider.fallback_to_local()
        self._multi_retriever = self._build_multi_retriever()
        self._answer_chain = self._build_answer_chain()
        self._step_instances = self._build_steps()

    # --- Helpers publicos (usados pelos steps via ctx) ---

    @staticmethod
    def format_docs(docs: list) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "desconhecido")
            parts.append(f"[Fonte: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # --- Invocacao publica ---

    def invoke(self, pergunta: str, status: list | None = None,
               fontes_selecionadas: list[str] | None = None) -> DragResult:
        """Executa as etapas definidas no config.pipeline_steps."""
        total = len(self._step_instances)

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
                    # Retry com novo LLM (steps ja foram recriados)
                    data = self._step_instances[i - 1].execute(data)
                else:
                    raise

        return DragResult(
            resposta=data.resposta,
            chunks=data.chunks,
            contexto=data.contexto,
            generated_queries=data.generated_queries,
            raw_count=data.raw_count,
            rerank_count=data.rerank_count,
        )


# ---------------------------------------------------------------------------
# Atalho para uso direto (cmd_ask, api.py)
# ---------------------------------------------------------------------------

_pipeline_cache: dict[str, DragPipeline] = {}


def get_pipeline(config: ServiceConfig) -> DragPipeline:
    if config.name not in _pipeline_cache:
        _pipeline_cache[config.name] = DragPipeline(config)
    return _pipeline_cache[config.name]


def ask(config: ServiceConfig, question: str) -> str:
    result = get_pipeline(config).invoke(question)
    return result.resposta
