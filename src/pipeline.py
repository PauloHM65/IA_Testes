"""Pipeline DRAG plugavel: etapas configuraveis por servico via YAML."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import redis as redis_lib
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_redis import RedisConfig, RedisVectorStore
from sentence_transformers import CrossEncoder

from src.config import ServiceConfig, env
from src.embeddings import get_embeddings

logger = logging.getLogger("drag.pipeline")


def _is_api_exhausted(exc: Exception) -> bool:
    """Detecta se a excecao indica quota/rate-limit esgotado da API."""
    # Checa pelo tipo da exceção (google.api_core.exceptions.ResourceExhausted)
    exc_type = type(exc).__name__.lower()
    if "resourceexhausted" in exc_type or "ratelimit" in exc_type:
        return True
    # Checa toda a cadeia de exceções (causa raiz)
    cause = exc.__cause__ or exc.__context__
    if cause:
        cause_type = type(cause).__name__.lower()
        if "resourceexhausted" in cause_type or "ratelimit" in cause_type:
            return True
    # Checa pela mensagem de erro
    err_msg = str(exc).lower()
    return any(term in err_msg for term in (
        "429", "quota", "resource_exhausted", "resourceexhausted",
        "rate limit", "rate_limit", "too many requests",
    ))


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
# Dados transitorios entre etapas do pipeline
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
    documento_filtro: str = ""  # nome do documento para filtrar busca (ex: "guia_02")
    fontes_selecionadas: list[str] = field(default_factory=list)  # docs selecionados pelo usuario
    exercicio_texto: str = ""
    exercicio_chunks: list = field(default_factory=list)
    materia_chunks: list = field(default_factory=list)


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

STEP_LABELS = {
    "multi_query": "Gerando variacoes da pergunta (MultiQuery)",
    "retrieve": "Buscando chunks relevantes",
    "neighbors": "Expandindo com vizinhos adjacentes",
    "rerank": "Reranking com cross-encoder",
    "generate": "Gerando resposta com LLM",
    "classificar": "Identificando materia e exercicio",
    "buscar_exercicio": "Buscando exercicio completo",
    "buscar_materia": "Buscando teoria relevante",
    "resolver": "Resolvendo passo a passo",
}


# ---------------------------------------------------------------------------
# DragPipeline — executa steps na ordem do YAML
# ---------------------------------------------------------------------------

class DragPipeline:
    """Pipeline DRAG com etapas plugaveis definidas no ServiceConfig."""

    # Modelo local de fallback (usado quando a API esgota)
    FALLBACK_LLM_MODEL = "qwen2.5:14b"

    def __init__(self, config: ServiceConfig):
        self.config = config

        # Provider ativo (pode mudar em runtime via fallback)
        self.active_provider: str = config.llm_provider
        self.active_model: str = config.llm_model

        # Componentes compartilhados (usados pelos steps)
        self._redis = redis_lib.from_url(env.REDIS_URL)
        if config.llm_provider == "gemini":
            self._llm = ChatGoogleGenerativeAI(
                model=config.llm_model,
                google_api_key=env.GOOGLE_API_KEY,
                temperature=0.55,
                max_retries=0,
            )
        else:
            self._llm = ChatOllama(
                model=config.llm_model,
                base_url=env.OLLAMA_BASE_URL,
                temperature=0.55,
            )
        self._reranker = CrossEncoder(config.rerank_model)

        # Vectorstore + retrievers
        self._vectorstore = self._build_vectorstore()
        self._base_retriever = self._vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": config.retriever_k}
        )
        self._multi_retriever = self._build_multi_retriever()

        # Answer chain (prompt do servico — usado pelo step generate)
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.system_prompt),
            ("human", config.human_prompt),
        ])
        self._answer_chain = prompt | self._llm | StrOutputParser()

        # Carrega steps do registro
        from src.steps import STEP_REGISTRY
        self._steps: dict[str, callable] = {}
        for step_name in config.pipeline_steps:
            if step_name in STEP_REGISTRY:
                self._steps[step_name] = STEP_REGISTRY[step_name]
            else:
                logger.warning(f"Step '{step_name}' nao encontrado no STEP_REGISTRY.")

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
            llm=self._llm,
            prompt=MULTI_QUERY_PROMPT.partial(n=str(self.config.multi_query_n)),
        )

    # --- Helpers publicos (usados pelos steps) ---

    def fetch_neighbors(self, chunks: list, window: int | None = None,
                        chunks_map_key: str | None = None) -> list:
        """Para cada chunk, busca vizinhos adjacentes no Redis."""
        window = window if window is not None else self.config.neighbor_window
        map_key = chunks_map_key or self.config.chunks_map_key

        all_keys = self._redis.hkeys(map_key)
        max_index_per_source: dict[str, int] = {}
        for raw_key in all_keys:
            key_str = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
            last_colon = key_str.rfind(":")
            if last_colon == -1:
                continue
            source = key_str[:last_colon]
            try:
                idx = int(key_str[last_colon + 1:])
            except ValueError:
                continue
            if source not in max_index_per_source or idx > max_index_per_source[source]:
                max_index_per_source[source] = idx

        seen = set()
        keys_to_fetch: list[str] = []
        fetch_meta: list[tuple[str, int]] = []

        for chunk in chunks:
            source = chunk.metadata.get("source", "desconhecido")
            idx = chunk.metadata.get("chunk_index")
            if idx is None:
                key = (source, id(chunk))
                if key not in seen:
                    seen.add(key)
                continue

            max_idx = max_index_per_source.get(source, idx)
            for offset in range(-window, window + 1):
                neighbor_idx = idx + offset
                if neighbor_idx < 0 or neighbor_idx > max_idx:
                    continue
                neighbor_key = (source, neighbor_idx)
                if neighbor_key in seen:
                    continue
                seen.add(neighbor_key)

                if offset != 0:
                    redis_key = f"{source}:{neighbor_idx}"
                    keys_to_fetch.append(redis_key)
                    fetch_meta.append((source, neighbor_idx))

        neighbor_docs: dict[tuple[str, int], Document] = {}
        if keys_to_fetch:
            values = self._redis.hmget(map_key, keys_to_fetch)
            for (source, idx), content in zip(fetch_meta, values):
                if content:
                    neighbor_docs[(source, idx)] = Document(
                        page_content=content.decode("utf-8"),
                        metadata={"source": source, "chunk_index": idx},
                    )

        expanded = []
        added = set()
        for chunk in chunks:
            source = chunk.metadata.get("source", "desconhecido")
            idx = chunk.metadata.get("chunk_index")
            if idx is None:
                expanded.append(chunk)
                continue

            max_idx = max_index_per_source.get(source, idx)
            for offset in range(-window, window + 1):
                neighbor_idx = idx + offset
                if neighbor_idx < 0 or neighbor_idx > max_idx:
                    continue
                add_key = (source, neighbor_idx)
                if add_key in added:
                    continue
                added.add(add_key)

                if offset == 0:
                    expanded.append(chunk)
                elif add_key in neighbor_docs:
                    expanded.append(neighbor_docs[add_key])

        expanded.sort(key=lambda d: (
            d.metadata.get("source", ""),
            d.metadata.get("chunk_index", 0),
        ))
        return expanded

    @staticmethod
    def format_docs(docs: list) -> str:
        """Formata chunks com tag [Fonte: ...]."""
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "desconhecido")
            parts.append(f"[Fonte: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # --- Fallback para modelo local ---

    def _fallback_to_ollama(self):
        """Troca o LLM para Ollama local quando a API esgota."""
        logger.warning("API esgotada — alternando para modelo local (Ollama).")
        self.active_provider = "ollama"
        self.active_model = self.FALLBACK_LLM_MODEL
        self._llm = ChatOllama(
            model=self.FALLBACK_LLM_MODEL,
            base_url=env.OLLAMA_BASE_URL,
            temperature=0.55,
        )
        # Recria chains que dependem do LLM
        self._multi_retriever = self._build_multi_retriever()
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            ("human", self.config.human_prompt),
        ])
        self._answer_chain = prompt | self._llm | StrOutputParser()

    # --- Invocacao publica ---

    def invoke(self, pergunta: str, status: list | None = None,
               fontes_selecionadas: list[str] | None = None) -> DragResult:
        """Executa as etapas definidas no config.pipeline_steps."""
        steps = self.config.pipeline_steps
        total = len(steps)

        data = PipelineData(
            pergunta=pergunta,
            fontes_selecionadas=fontes_selecionadas or [],
        )

        for i, step_name in enumerate(steps, 1):
            if status is not None:
                label = STEP_LABELS.get(step_name, step_name)
                status[0] = f"[{i}/{total}] {label}..."

            step_fn = self._steps.get(step_name)
            if step_fn is None:
                logger.warning(f"Etapa desconhecida ignorada: {step_name}")
                continue

            try:
                data = step_fn(data, self)
            except Exception as e:
                logger.error("Erro no step '%s': [%s] %s", step_name, type(e).__name__, e)
                if self.active_provider != "ollama" and _is_api_exhausted(e):
                    logger.warning("Fallback ativado para Ollama.")
                    if status is not None:
                        status[0] = "API esgotada — alternando para modelo local..."
                    self._fallback_to_ollama()
                    data = step_fn(data, self)
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
# Atalho para uso direto (cmd_ask)
# ---------------------------------------------------------------------------

_pipeline_cache: dict[str, DragPipeline] = {}


def get_pipeline(config: ServiceConfig) -> DragPipeline:
    if config.name not in _pipeline_cache:
        _pipeline_cache[config.name] = DragPipeline(config)
    return _pipeline_cache[config.name]


def ask(config: ServiceConfig, question: str) -> str:
    """Faz uma pergunta ao DRAG e retorna a resposta."""
    result = get_pipeline(config).invoke(question)
    return result.resposta
