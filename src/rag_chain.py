"""Módulo DRAG: Retrieval-Augmented Generation com MultiQuery, reranking e vizinhos."""

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
from langchain_redis import RedisVectorStore
from sentence_transformers import CrossEncoder

from src.config import cfg
from src.ingest import get_embeddings

logger = logging.getLogger("drag.multiquery")


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
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
REGRA OBRIGATÓRIA DE FORMATAÇÃO:
- PROIBIDO usar LaTeX, \\sum, \\frac, \\(, \\), \\[, \\], $, $$.
- Use APENAS símbolos Unicode: Σ para somatório, ∈ para pertence, ≤ ≥ ≠ × ÷ √ ∞ π ² ³ → ⇒ ∀ ∃.
- Frações: escreva como n(n+1)/2, nunca como \\frac.
- Exemplo correto: Σ(i=1 até n) i = n(n+1)/2
- Exemplo ERRADO: \\sum_{{i=1}}^{{n}} i = \\frac{{n(n+1)}}{{2}}

Você é um professor universitário com doutorado em todas as áreas de exatas. Use SOMENTE o contexto abaixo para responder.
Se o contexto tiver informação relacionada à pergunta, use-a para responder, \
mesmo que não cubra 100% da pergunta.
Se o contexto não tiver NADA relacionado ao tema da pergunta, diga que não tem \
informação suficiente. Nunca invente informação que não esteja no contexto.
Cada trecho do contexto tem uma tag [Fonte: ...] com o nome do arquivo de origem.
Quando citar informações, mencione a fonte correta.

Contexto:
{context}
"""

HUMAN_PROMPT = "{question}\n\nLembre-se: responda usando símbolos Unicode (Σ, ∈, ≤, ², →), NUNCA LaTeX."

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
# DragPipeline — encapsula todo o pipeline DRAG
# ---------------------------------------------------------------------------

class DragPipeline:
    """Pipeline DRAG completo: MultiQuery → Retrieve → Neighbors → Rerank → LLM."""

    def __init__(self):
        self._vectorstore = self._build_vectorstore()
        self._llm = self._build_llm()
        self._base_retriever = self._vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": cfg.RETRIEVER_K}
        )
        self._multi_retriever = self._build_multi_retriever()
        self._answer_chain = self._build_answer_chain()
        self._reranker = CrossEncoder(cfg.RERANK_MODEL)
        self._redis = redis_lib.from_url(cfg.REDIS_URL)

    # --- Builders privados ---

    @staticmethod
    def _build_vectorstore() -> RedisVectorStore:
        from langchain_redis import RedisConfig

        client = redis_lib.from_url(cfg.REDIS_URL)
        try:
            client.ping()
        except redis_lib.exceptions.ConnectionError:
            raise ConnectionError(
                f"Não foi possível conectar ao Redis em '{cfg.REDIS_URL}'. "
                f"Verifique se o container está rodando: docker compose up -d"
            )

        return RedisVectorStore(
            embeddings=get_embeddings(),
            config=RedisConfig(
                index_name="rag_docs",
                redis_url=cfg.REDIS_URL,
                from_existing=True,
            ),
        )

    @staticmethod
    def _build_llm() -> ChatOllama:
        return ChatOllama(
            model=cfg.LLM_MODEL,
            base_url=cfg.OLLAMA_BASE_URL,
            temperature=0.0,
        )

    def _build_multi_retriever(self) -> MultiQueryRetriever:
        return MultiQueryRetriever.from_llm(
            retriever=self._base_retriever,
            llm=self._llm,
            prompt=MULTI_QUERY_PROMPT.partial(n=str(cfg.MULTI_QUERY_N)),
        )

    def _build_answer_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        return prompt | self._llm | StrOutputParser()

    # --- Etapas do pipeline ---

    def _retrieve(self, pergunta: str) -> tuple[list, list[str]]:
        """MultiQuery → Retrieve. Retorna (chunks_únicos, queries_geradas)."""
        generated_queries: list[str] = []

        def _capture_queries(info: dict):
            queries = info.get("queries", info.get("result", []))
            generated_queries.extend(queries)

        handler = type("CaptureHandler", (), {
            "on_retriever_end": lambda self, *a, **kw: None,
            "on_retriever_error": lambda self, *a, **kw: None,
        })()

        # MultiQueryRetriever gera variações e busca para cada uma
        with _capture_multiquery(self._multi_retriever, generated_queries):
            raw_chunks = self._multi_retriever.invoke(pergunta)

        return raw_chunks, generated_queries

    def _fetch_neighbors(self, chunks: list) -> list:
        """Para cada chunk, busca vizinhos adjacentes no Redis com hmget."""
        window = cfg.NEIGHBOR_WINDOW

        # Descobre limites por fonte
        all_keys = self._redis.hkeys("rag_chunks_map")
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

        # Coleta todas as chaves de vizinhos necessárias
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

        # Busca todos os vizinhos de uma vez com hmget
        neighbor_docs: dict[tuple[str, int], Document] = {}
        if keys_to_fetch:
            values = self._redis.hmget("rag_chunks_map", keys_to_fetch)
            for (source, idx), content in zip(fetch_meta, values):
                if content:
                    neighbor_docs[(source, idx)] = Document(
                        page_content=content.decode("utf-8"),
                        metadata={"source": source, "chunk_index": idx},
                    )

        # Monta lista expandida: chunks originais + vizinhos
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

    def _rerank(self, query: str, docs: list) -> list:
        """Reordena documentos usando cross-encoder e retorna os top_n."""
        if not docs:
            return docs
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:cfg.RERANK_TOP_N]]

    @staticmethod
    def _format_docs(docs: list) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "desconhecido")
            parts.append(f"[Fonte: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # --- Invocação pública ---

    def invoke(self, pergunta: str, status: list | None = None) -> DragResult:
        """Executa o pipeline completo: MultiQuery → Retrieve → Neighbors → Rerank → LLM."""

        def _set(msg: str):
            if status is not None:
                status[0] = msg

        _set("[1/6] Gerando variações da pergunta (MultiQuery)...")
        raw_chunks, generated_queries = self._retrieve(pergunta)
        raw_count = len(raw_chunks)

        _set("[2/6] Expandindo com vizinhos adjacentes...")
        expanded_chunks = self._fetch_neighbors(raw_chunks)

        _set("[3/6] Reranking com cross-encoder...")
        ranked_chunks = self._rerank(pergunta, expanded_chunks)
        rerank_count = len(ranked_chunks)

        _set("[4/6] Formatando contexto...")
        contexto = self._format_docs(ranked_chunks)

        _set("[5/6] Gerando resposta com LLM...")
        resposta = self._answer_chain.invoke({"context": contexto, "question": pergunta})

        _set("[6/6] Formatando resposta...")
        resposta = latex_to_unicode(resposta)

        return DragResult(
            resposta=resposta,
            chunks=ranked_chunks,
            contexto=contexto,
            generated_queries=generated_queries,
            raw_count=raw_count,
            rerank_count=rerank_count,
        )


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
# Atalho para uso direto (cmd_ask)
# ---------------------------------------------------------------------------

_pipeline_cache: DragPipeline | None = None


def get_pipeline() -> DragPipeline:
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = DragPipeline()
    return _pipeline_cache


def ask(question: str) -> str:
    """Faz uma pergunta ao DRAG e retorna a resposta."""
    result = get_pipeline().invoke(question)
    return result.resposta
