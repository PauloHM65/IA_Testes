"""Repositorio unico de steps do pipeline DRAG.

Cada step e uma funcao: (data, pipeline) -> data
O YAML define a ordem, este arquivo define o comportamento.
Para adicionar um step novo: escreva a funcao + adicione ao STEP_REGISTRY.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_redis import RedisConfig, RedisVectorStore

from src.config import env
from src.embeddings import get_embeddings

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.pipeline import DragPipeline, PipelineData


# ===========================================================================
# Steps padrao (fluxo linear)
# ===========================================================================

def step_multi_query(data: PipelineData, pipeline: DragPipeline) -> PipelineData:
    """Gera variacoes da pergunta e busca para cada uma."""
    from src.pipeline import _capture_multiquery
    with _capture_multiquery(pipeline._multi_retriever, data.generated_queries):
        data.raw_chunks = pipeline._multi_retriever.invoke(data.pergunta)
    data.raw_count = len(data.raw_chunks)
    return data


def step_retrieve(data: PipelineData, pipeline: DragPipeline) -> PipelineData:
    """Busca vetorial direta (sem MultiQuery)."""
    if not data.raw_chunks:
        data.raw_chunks = pipeline._base_retriever.invoke(data.pergunta)
        data.raw_count = len(data.raw_chunks)
    return data


def step_neighbors(data: PipelineData, pipeline: DragPipeline) -> PipelineData:
    """Expande cada chunk com vizinhos adjacentes do Redis."""
    data.chunks = pipeline.fetch_neighbors(data.raw_chunks)
    return data


def step_rerank(data: PipelineData, pipeline: DragPipeline) -> PipelineData:
    """Reordena com cross-encoder e mantem top_n."""
    source = data.chunks if data.chunks else data.raw_chunks
    if not source:
        return data
    pairs = [[data.pergunta, doc.page_content] for doc in source]
    scores = pipeline._reranker.predict(pairs)
    ranked = sorted(zip(scores, source), key=lambda x: x[0], reverse=True)
    data.chunks = [doc for _, doc in ranked[:pipeline.config.rerank_top_n]]
    data.rerank_count = len(data.chunks)
    return data


def step_generate(data: PipelineData, pipeline: DragPipeline) -> PipelineData:
    """Formata contexto e gera resposta com LLM."""
    data.contexto = pipeline.format_docs(data.chunks)
    data.resposta = pipeline._answer_chain.invoke({
        "context": data.contexto,
        "question": data.pergunta,
    })
    return data


# ===========================================================================
# Steps do agente inteligente (classificar + exercicios + materia)
# ===========================================================================

# Quantidade de chunks sequenciais a pegar apos o melhor match
EXERCICIO_SEQUENCIAL_CHUNKS = 20

# --- Prompts ---

_CLASSIFICAR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Voce e um classificador academico. Dada a pergunta do aluno, identifique:\n"
     "1. Se e um pedido de RESOLUCAO DE EXERCICIO ou uma PERGUNTA TEORICA\n"
     "2. Qual materia/assunto trata\n"
     "3. Palavras-chave para buscar a teoria\n"
     "4. Se o aluno mencionou um documento/guia/lista especifico (ex: guia_02, lista_3)\n\n"
     "Responda APENAS neste formato (sem mais nada):\n"
     "TIPO: exercicio ou teoria\n"
     "MATERIA: <nome da materia/assunto>\n"
     "DOCUMENTO: <nome do documento mencionado, ou vazio se nao mencionou>\n"
     "BUSCA: <palavras-chave para buscar teoria, separadas por virgula>"),
    ("human", "{question}"),
])


# --- Steps ---

def step_classificar(data: PipelineData, pipeline: DragPipeline) -> PipelineData:
    """Identifica se e exercicio ou teoria, materia, documento e palavras-chave."""
    chain = _CLASSIFICAR_PROMPT | pipeline._llm | StrOutputParser()
    resultado = chain.invoke({"question": data.pergunta})

    tipo = "teoria"
    materia = ""
    documento = ""
    busca = ""
    for linha in resultado.strip().split("\n"):
        upper = linha.upper()
        if upper.startswith("TIPO:"):
            tipo = linha.split(":", 1)[1].strip().lower()
        elif upper.startswith("MATERIA:"):
            materia = linha.split(":", 1)[1].strip()
        elif upper.startswith("DOCUMENTO:"):
            documento = linha.split(":", 1)[1].strip()
        elif upper.startswith("BUSCA:"):
            busca = linha.split(":", 1)[1].strip()

    data.materia_identificada = materia
    data.documento_filtro = documento
    if busca:
        data.generated_queries = [busca]

    data.exercicio_texto = "" if tipo == "teoria" else "__EXERCICIO__"

    return data


def step_buscar_exercicio(data: PipelineData, pipeline: DragPipeline) -> PipelineData:
    """Acha o chunk mais relevante do exercicio e pega os N sequenciais via hmget."""
    # Se classificado como teoria, pula
    if data.exercicio_texto != "__EXERCICIO__":
        data.exercicio_texto = ""
        return data

    # Conecta ao indice de exercicios
    exercicios_vs = _get_exercicios_vectorstore(pipeline)
    if exercicios_vs is None:
        data.exercicio_texto = ""
        return data

    # Monta filtro por documento se identificado
    search_kwargs = {"k": 1}  # so precisa do melhor match
    if data.documento_filtro:
        from redisvl.query.filter import Text
        search_kwargs["filter"] = Text("source") % data.documento_filtro

    # Busca o chunk mais relevante
    results = exercicios_vs.similarity_search(data.pergunta, **search_kwargs)
    if not results:
        data.exercicio_texto = ""
        return data

    best = results[0]
    source = best.metadata.get("source", "desconhecido")
    start_idx = best.metadata.get("chunk_index", 0)
    data.raw_count = 1

    # Pega os N chunks sequenciais a partir do melhor match via Redis hash
    map_key = pipeline.config.exercicios_chunks_map_key
    keys = [f"{source}:{start_idx + i}" for i in range(EXERCICIO_SEQUENCIAL_CHUNKS)]
    values = pipeline._redis.hmget(map_key, keys)

    exercicio_chunks = []
    for i, content in enumerate(values):
        if content is None:
            break  # acabaram os chunks desse documento
        exercicio_chunks.append(Document(
            page_content=content.decode("utf-8") if isinstance(content, bytes) else content,
            metadata={"source": source, "chunk_index": start_idx + i},
        ))

    data.exercicio_chunks = exercicio_chunks
    data.exercicio_texto = "\n\n".join(c.page_content for c in exercicio_chunks)

    return data


def step_buscar_materia(data: PipelineData, pipeline: DragPipeline) -> PipelineData:
    """Busca teoria relevante no indice de materias do proprio servico."""
    # Se tem exercicio, usa o conteudo dele como query (mais preciso)
    # Se e teoria, usa a pergunta original
    if data.exercicio_texto and data.exercicio_texto != "__EXERCICIO__":
        query = data.exercicio_texto[:500]
    else:
        query = data.pergunta

    materia_chunks = pipeline._base_retriever.invoke(query)

    # Filtra por fontes selecionadas pelo usuario (se houver)
    if data.fontes_selecionadas:
        materia_chunks = [
            c for c in materia_chunks
            if any(sel in c.metadata.get("source", "") for sel in data.fontes_selecionadas)
        ]

    # Rerank
    if materia_chunks:
        pairs = [[query, doc.page_content] for doc in materia_chunks]
        scores = pipeline._reranker.predict(pairs)
        ranked = sorted(zip(scores, materia_chunks), key=lambda x: x[0], reverse=True)
        data.materia_chunks = [doc for _, doc in ranked[:pipeline.config.rerank_top_n]]
    else:
        data.materia_chunks = []

    return data


def step_resolver(data: PipelineData, pipeline: DragPipeline) -> PipelineData:
    """Combina exercicio + materia e gera resolucao/resposta."""
    materia_context = _format_docs(data.materia_chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", pipeline.config.system_prompt),
        ("human", pipeline.config.human_prompt),
    ])
    chain = prompt | pipeline._llm | StrOutputParser()

    data.resposta = chain.invoke({
        "exercicio": data.exercicio_texto or "(Sem exercicio — pergunta teorica)",
        "context": materia_context,
        "question": data.pergunta,
    })

    # Junta chunks para logging
    data.chunks = data.exercicio_chunks + data.materia_chunks
    data.contexto = (
        f"=== EXERCICIO ===\n{data.exercicio_texto}\n\n"
        f"=== MATERIA ===\n{materia_context}"
    )
    data.rerank_count = len(data.chunks)

    return data


# ===========================================================================
# Helpers (privados)
# ===========================================================================

def _get_exercicios_vectorstore(pipeline: DragPipeline) -> RedisVectorStore | None:
    """Conecta ao indice de exercicios do servico. Retorna None se nao existe."""
    try:
        embeddings = get_embeddings(pipeline.config.embedding_model)
        return RedisVectorStore(
            embeddings=embeddings,
            config=RedisConfig(
                index_name=pipeline.config.exercicios_index_name,
                redis_url=env.REDIS_URL,
                from_existing=True,
            ),
        )
    except Exception:
        return None


def _format_docs(docs: list) -> str:
    """Formata chunks com tag de fonte."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "desconhecido")
        parts.append(f"[Fonte: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ===========================================================================
# STEP_REGISTRY — nome no YAML -> funcao
# ===========================================================================

STEP_REGISTRY: dict[str, callable] = {
    # Steps padrao (fluxo linear)
    "multi_query": step_multi_query,
    "retrieve": step_retrieve,
    "neighbors": step_neighbors,
    "rerank": step_rerank,
    "generate": step_generate,
    # Steps do agente inteligente
    "classificar": step_classificar,
    "buscar_exercicio": step_buscar_exercicio,
    "buscar_materia": step_buscar_materia,
    "resolver": step_resolver,
}
