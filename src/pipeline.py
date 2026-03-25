"""Pipeline DRAG plugável: etapas configuráveis por serviço via YAML."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import redis as redis_lib
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_redis import RedisConfig, RedisVectorStore
from sentence_transformers import CrossEncoder

from src.config import ServiceConfig, env
from src.embeddings import get_embeddings

logger = logging.getLogger("drag.pipeline")


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class DragResult:
    """Resultado completo de uma invocação do pipeline DRAG."""
    resposta: str
    chunks: list
    contexto: str
    generated_queries: list[str] = field(default_factory=list)
    raw_count: int = 0
    rerank_count: int = 0


# ---------------------------------------------------------------------------
# Dados transitórios entre etapas do pipeline
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


# ---------------------------------------------------------------------------
# Prompt padrão do MultiQuery
# ---------------------------------------------------------------------------

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente que gera variações de perguntas para busca semântica. "
     "Dado a pergunta original, gere {n} variações alternativas que capturem "
     "diferentes perspectivas ou reformulações da mesma intenção. "
     "Retorne APENAS as variações, uma por linha, sem numeração."),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# Conversão LaTeX → Unicode
# ---------------------------------------------------------------------------

LATEX_TO_UNICODE = {
    r"\sum": "Σ", r"\prod": "Π", r"\int": "∫",
    r"\infty": "∞", r"\pi": "π", r"\alpha": "α", r"\beta": "β",
    r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε", r"\theta": "θ",
    r"\lambda": "λ", r"\mu": "μ", r"\sigma": "σ", r"\omega": "ω",
    r"\phi": "φ", r"\psi": "ψ", r"\tau": "τ",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\times": "×", r"\div": "÷", r"\cdot": "·", r"\pm": "±",
    r"\sqrt": "√", r"\in": "∈", r"\notin": "∉",
    r"\subset": "⊂", r"\supset": "⊃", r"\cup": "∪", r"\cap": "∩",
    r"\forall": "∀", r"\exists": "∃", r"\emptyset": "∅",
    r"\rightarrow": "→", r"\leftarrow": "←", r"\Rightarrow": "⇒",
    r"\Leftarrow": "⇐", r"\leftrightarrow": "↔",
    r"\partial": "∂", r"\nabla": "∇",
}


def latex_to_unicode(text: str) -> str:
    """Converte notação LaTeX residual para símbolos Unicode."""
    if "\\" not in text and "$" not in text:
        return text

    text = re.sub(r"\\\[|\\\]", "", text)
    text = re.sub(r"\\\(|\\\)", "", text)
    text = re.sub(r"\$\$?", "", text)
    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)
    text = re.sub(r"\\left\s*", "", text)
    text = re.sub(r"\\right\s*", "", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\sum_\{([^}]*)\}\^\{([^}]*)\}", r"Σ(\1 até \2)", text)
    text = re.sub(r"\\sum_\{([^}]*)\}", r"Σ(\1)", text)
    text = re.sub(r"\\prod_\{([^}]*)\}\^\{([^}]*)\}", r"Π(\1 até \2)", text)

    def _superscript(m):
        exp = m.group(1)
        sup_map = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
                    "n": "ⁿ", "i": "ⁱ"}
        return sup_map.get(exp, f"^{exp}")
    text = re.sub(r"\^\{([^}]*)\}", _superscript, text)

    text = re.sub(r"_\{([^}]*)\}", r"_\1", text)

    for latex_cmd, unicode_char in LATEX_TO_UNICODE.items():
        text = text.replace(latex_cmd, unicode_char)

    text = text.replace("{", "").replace("}", "")
    return text


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
# DragPipeline — etapas plugáveis configuradas por YAML
# ---------------------------------------------------------------------------

class DragPipeline:
    """Pipeline DRAG com etapas plugáveis definidas no ServiceConfig."""

    def __init__(self, config: ServiceConfig):
        self.config = config

        # Componentes compartilhados
        self._redis = redis_lib.from_url(env.REDIS_URL)
        self._llm = ChatOllama(
            model=config.llm_model,
            base_url=env.OLLAMA_BASE_URL,
            temperature=0.0,
        )
        self._reranker = CrossEncoder(config.rerank_model)

        # Vectorstore + retrievers
        self._vectorstore = self._build_vectorstore()
        self._base_retriever = self._vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": config.retriever_k}
        )
        self._multi_retriever = self._build_multi_retriever()

        # Answer chain (prompt do serviço)
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.system_prompt),
            ("human", config.human_prompt),
        ])
        self._answer_chain = prompt | self._llm | StrOutputParser()

        # Registro de etapas disponíveis
        self._steps: dict[str, callable] = {
            "multi_query": self._step_multi_query,
            "retrieve": self._step_retrieve,
            "neighbors": self._step_neighbors,
            "rerank": self._step_rerank,
            "generate": self._step_generate,
            "latex_to_unicode": self._step_latex_to_unicode,
        }

    # --- Builders ---

    def _build_vectorstore(self) -> RedisVectorStore:
        try:
            self._redis.ping()
        except redis_lib.exceptions.ConnectionError:
            raise ConnectionError(
                f"Não foi possível conectar ao Redis em '{env.REDIS_URL}'. "
                f"Verifique se o container está rodando: docker compose up -d"
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

    # --- Etapas do pipeline ---

    def _step_multi_query(self, data: PipelineData) -> PipelineData:
        """Gera variações da pergunta e busca para cada uma."""
        with _capture_multiquery(self._multi_retriever, data.generated_queries):
            data.raw_chunks = self._multi_retriever.invoke(data.pergunta)
        data.raw_count = len(data.raw_chunks)
        return data

    def _step_retrieve(self, data: PipelineData) -> PipelineData:
        """Busca vetorial direta (sem MultiQuery). Usado quando multi_query não está no pipeline."""
        if not data.raw_chunks:
            data.raw_chunks = self._base_retriever.invoke(data.pergunta)
            data.raw_count = len(data.raw_chunks)
        return data

    def _step_neighbors(self, data: PipelineData) -> PipelineData:
        """Expande cada chunk com vizinhos adjacentes do Redis."""
        data.chunks = self._fetch_neighbors(data.raw_chunks)
        return data

    def _step_rerank(self, data: PipelineData) -> PipelineData:
        """Reordena com cross-encoder e mantém top_n."""
        source = data.chunks if data.chunks else data.raw_chunks
        if not source:
            return data
        pairs = [[data.pergunta, doc.page_content] for doc in source]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(scores, source), key=lambda x: x[0], reverse=True)
        data.chunks = [doc for _, doc in ranked[:self.config.rerank_top_n]]
        data.rerank_count = len(data.chunks)
        return data

    def _step_generate(self, data: PipelineData) -> PipelineData:
        """Formata contexto e gera resposta com LLM."""
        data.contexto = self._format_docs(data.chunks)
        data.resposta = self._answer_chain.invoke({
            "context": data.contexto,
            "question": data.pergunta,
        })
        return data

    def _step_latex_to_unicode(self, data: PipelineData) -> PipelineData:
        """Converte LaTeX residual para Unicode."""
        data.resposta = latex_to_unicode(data.resposta)
        return data

    # --- Helpers ---

    def _fetch_neighbors(self, chunks: list) -> list:
        """Para cada chunk, busca vizinhos adjacentes no Redis com hmget."""
        window = self.config.neighbor_window

        all_keys = self._redis.hkeys(self.config.chunks_map_key)
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
            values = self._redis.hmget(self.config.chunks_map_key, keys_to_fetch)
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
    def _format_docs(docs: list) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "desconhecido")
            parts.append(f"[Fonte: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # --- Invocação pública ---

    def invoke(self, pergunta: str, status: list | None = None) -> DragResult:
        """Executa as etapas definidas no config.pipeline_steps."""
        steps = self.config.pipeline_steps
        total = len(steps)

        step_labels = {
            "multi_query": "Gerando variações da pergunta (MultiQuery)",
            "retrieve": "Buscando chunks relevantes",
            "neighbors": "Expandindo com vizinhos adjacentes",
            "rerank": "Reranking com cross-encoder",
            "generate": "Gerando resposta com LLM",
            "latex_to_unicode": "Formatando resposta",
        }

        data = PipelineData(pergunta=pergunta)

        for i, step_name in enumerate(steps, 1):
            if status is not None:
                label = step_labels.get(step_name, step_name)
                status[0] = f"[{i}/{total}] {label}..."

            step_fn = self._steps.get(step_name)
            if step_fn is None:
                logger.warning(f"Etapa desconhecida ignorada: {step_name}")
                continue
            data = step_fn(data)

        return DragResult(
            resposta=data.resposta,
            chunks=data.chunks,
            contexto=data.contexto,
            generated_queries=data.generated_queries,
            raw_count=data.raw_count,
            rerank_count=data.rerank_count,
        )

    # --- Registro de etapas customizadas ---

    def register_step(self, name: str, fn: callable):
        """Registra uma etapa customizada no pipeline."""
        self._steps[name] = fn


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
