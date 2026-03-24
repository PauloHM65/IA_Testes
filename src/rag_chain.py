"""Módulo DRAG: Retrieval-Augmented Generation com reranking via cross-encoder."""

from __future__ import annotations

# import ast  # MultiQuery desativado
# import logging  # MultiQuery desativado
import os
import re

import redis as redis_lib
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_redis import RedisVectorStore
# from langchain_classic.retrievers import MultiQueryRetriever  # MultiQuery desativado
from sentence_transformers import CrossEncoder

from src.ingest import get_embeddings


RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
MULTI_QUERY_N = int(os.getenv("MULTI_QUERY_N", "3"))
NEIGHBOR_WINDOW = int(os.getenv("NEIGHBOR_WINDOW", "2"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "20"))

_reranker_cache: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker_cache
    if _reranker_cache is None:
        _reranker_cache = CrossEncoder(RERANK_MODEL)
    return _reranker_cache


# Cache dos componentes DRAG para reutilização (cmd_ask)
_drag_components_cache = None


def _get_drag_components():
    global _drag_components_cache
    if _drag_components_cache is None:
        _drag_components_cache = build_drag_components()
    return _drag_components_cache


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


def check_redis_connection() -> redis_lib.Redis:
    """Valida conexão com o Redis e retorna o client. Levanta erro claro se falhar."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        client = redis_lib.from_url(redis_url)
        client.ping()
        return client
    except redis_lib.exceptions.ConnectionError:
        raise ConnectionError(
            f"Não foi possível conectar ao Redis em '{redis_url}'. "
            f"Verifique se o container está rodando: docker compose up -d"
        )


def get_llm() -> ChatOllama:
    """Retorna o LLM configurado via Ollama."""
    return ChatOllama(
        model=os.getenv("LLM_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.0,
    )


def get_vectorstore() -> RedisVectorStore:
    """Conecta ao índice existente no Redis."""
    from langchain_redis import RedisConfig

    check_redis_connection()

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    embeddings = get_embeddings()

    config = RedisConfig(
        index_name="rag_docs",
        redis_url=redis_url,
        from_existing=True,
    )

    return RedisVectorStore(embeddings=embeddings, config=config)


def format_docs(docs: list) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "desconhecido")
        parts.append(f"[Fonte: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def rerank(query: str, docs: list, top_n: int = RERANK_TOP_N) -> list:
    """Reordena documentos usando cross-encoder e retorna os top_n mais relevantes."""
    if not docs:
        return docs
    model = get_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def fetch_neighbors(chunks: list, window: int = NEIGHBOR_WINDOW) -> list:
    """Para cada chunk, busca os vizinhos (antes e depois) no Redis e retorna lista expandida."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    client = redis_lib.from_url(redis_url)

    # Descobre o total de chunks por fonte para respeitar fronteiras
    all_keys = client.hkeys("rag_chunks_map")
    max_index_per_source: dict[str, int] = {}
    for raw_key in all_keys:
        key_str = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
        # Formato: "fonte/caminho.pdf:index"
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
    expanded = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "desconhecido")
        idx = chunk.metadata.get("chunk_index")
        if idx is None:
            key = (source, id(chunk))
            if key not in seen:
                seen.add(key)
                expanded.append(chunk)
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

            if offset == 0:
                expanded.append(chunk)
            else:
                redis_key = f"{source}:{neighbor_idx}"
                content = client.hget("rag_chunks_map", redis_key)
                if content:
                    neighbor_doc = Document(
                        page_content=content.decode("utf-8"),
                        metadata={"source": source, "chunk_index": neighbor_idx},
                    )
                    expanded.append(neighbor_doc)

    expanded.sort(key=lambda d: (
        d.metadata.get("source", ""),
        d.metadata.get("chunk_index", 0),
    ))

    return expanded


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
    # Guard: só processa se houver possíveis comandos LaTeX
    if "\\" not in text and "$" not in text:
        return text

    # Remove delimitadores LaTeX: \[ \] \( \) $$ $
    text = re.sub(r"\\\[|\\\]", "", text)
    text = re.sub(r"\\\(|\\\)", "", text)
    text = re.sub(r"\$\$?", "", text)

    # \frac{a}{b} → a/b
    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)

    # \left e \right (decorativos)
    text = re.sub(r"\\left\s*", "", text)
    text = re.sub(r"\\right\s*", "", text)

    # \text{...} → conteúdo
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)

    # \sum_{i=a}^{b} → Σ(i=a até b)
    text = re.sub(r"\\sum_\{([^}]*)\}\^\{([^}]*)\}", r"Σ(\1 até \2)", text)
    text = re.sub(r"\\sum_\{([^}]*)\}", r"Σ(\1)", text)

    # \prod_{i=a}^{b} → Π(i=a até b)
    text = re.sub(r"\\prod_\{([^}]*)\}\^\{([^}]*)\}", r"Π(\1 até \2)", text)

    # x^{2} → x² , x^{3} → x³, x^{n} → x^n
    def _superscript(m):
        exp = m.group(1)
        sup_map = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
                    "n": "ⁿ", "i": "ⁱ"}
        if exp in sup_map:
            return sup_map[exp]
        return f"^{exp}"
    text = re.sub(r"\^\{([^}]*)\}", _superscript, text)

    # _{...} → subscrito simples (remove chaves)
    text = re.sub(r"_\{([^}]*)\}", r"_\1", text)

    # Substituições diretas de comandos LaTeX
    for latex_cmd, unicode_char in LATEX_TO_UNICODE.items():
        text = text.replace(latex_cmd, unicode_char)

    # Limpa chaves remanescentes soltas { }
    text = text.replace("{", "").replace("}", "")

    return text


def build_drag_components():
    """Inicializa os componentes do DRAG: retriever, prompt, LLM."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
    )
    llm = get_llm()

    # MultiQuery desativado por enquanto — usar retriever base direto.
    # Para reativar, descomentar abaixo e comentar o retriever acima:
    # from langchain_classic.retrievers import MultiQueryRetriever
    # multi_query_prompt = ChatPromptTemplate.from_messages([
    #     ("system",
    #      "Você é um assistente que gera variações de perguntas para busca semântica. "
    #      "Dado a pergunta original, gere {n} variações alternativas que capturem "
    #      "diferentes perspectivas ou reformulações da mesma intenção. "
    #      "Retorne APENAS as variações, uma por linha, sem numeração."),
    #     ("human", "{question}"),
    # ])
    # retriever = MultiQueryRetriever.from_llm(
    #     retriever=retriever,
    #     llm=llm,
    #     prompt=multi_query_prompt.partial(n=str(MULTI_QUERY_N)),
    # )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])
    answer_chain = prompt | llm | StrOutputParser()
    return retriever, answer_chain


def invoke_with_log(retriever, answer_chain, pergunta: str, status: list | None = None) -> tuple:
    """Retriever → Rerank → Vizinhos → LLM. Atualiza status[0] a cada etapa."""
    def _set(msg: str):
        if status is not None:
            status[0] = msg

    _set("[1/5] Buscando chunks relevantes...")
    raw_chunks = retriever.invoke(pergunta)

    _set("[2/5] Reranking com cross-encoder...")
    ranked_chunks = rerank(pergunta, raw_chunks, top_n=RERANK_TOP_N)
    rerank_count = len(ranked_chunks)

    _set("[3/5] Expandindo com vizinhos...")
    chunks = fetch_neighbors(ranked_chunks, window=NEIGHBOR_WINDOW)
    contexto = format_docs(chunks)

    _set("[4/5] Gerando resposta com LLM...")
    resposta = answer_chain.invoke({"context": contexto, "question": pergunta})

    _set("[5/5] Formatando resposta...")
    resposta = latex_to_unicode(resposta)

    return resposta, chunks, contexto, [], len(raw_chunks), rerank_count


def ask(question: str) -> str:
    """Faz uma pergunta ao DRAG e retorna a resposta (reutiliza componentes)."""
    retriever, answer_chain = _get_drag_components()
    resposta, _, _, _, _, _ = invoke_with_log(retriever, answer_chain, question)
    return resposta
