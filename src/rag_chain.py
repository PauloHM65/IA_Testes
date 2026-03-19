"""Módulo DRAG: Retrieval-Augmented Generation com reranking via cross-encoder."""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_redis import RedisVectorStore
from sentence_transformers import CrossEncoder

from src.ingest import get_embeddings


RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

SYSTEM_PROMPT = """\
Você é um profressor universitario com doutorado em todas as áreas de exatas. Use SOMENTE o contexto abaixo para responder.
Se o contexto tiver informação relacionada à pergunta, use-a para responder, \
mesmo que não cubra 100% da pergunta.
Se o contexto não tiver NADA relacionado ao tema da pergunta, diga que não tem \
informação suficiente. Nunca invente informação que não esteja no contexto.
Cada trecho do contexto tem uma tag [Fonte: ...] com o nome do arquivo de origem.
Quando citar informações, mencione a fonte correta.

Contexto:
{context}
"""

HUMAN_PROMPT = "{question}"


def get_llm() -> ChatOllama:
    """Retorna o LLM configurado via Ollama."""
    return ChatOllama(
        model=os.getenv("LLM_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1,
    )


def get_vectorstore() -> RedisVectorStore:
    """Conecta ao índice existente no Redis."""
    from langchain_redis import RedisConfig

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
    model = CrossEncoder(RERANK_MODEL)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def build_drag_components():
    """Inicializa os componentes do DRAG: retriever, prompt, LLM."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])
    llm = get_llm()
    answer_chain = prompt | llm | StrOutputParser()
    return retriever, answer_chain


def invoke_with_log(retriever, answer_chain, pergunta: str) -> tuple:
    """Retriever → Rerank → LLM. Retorna (resposta, chunks, contexto)."""
    raw_chunks = retriever.invoke(pergunta)
    chunks = rerank(pergunta, raw_chunks)
    contexto = format_docs(chunks)
    resposta = answer_chain.invoke({"context": contexto, "question": pergunta})
    return resposta, chunks, contexto


def ask(question: str) -> str:
    """Faz uma pergunta ao DRAG e retorna a resposta."""
    retriever, answer_chain = build_drag_components()
    resposta, _, _ = invoke_with_log(retriever, answer_chain, question)
    return resposta
