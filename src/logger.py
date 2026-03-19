"""Módulo de logging: registra cada etapa do pipeline RAG em arquivos."""

from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

_session_file = None
_step_counter = 0


def _get_session_file() -> Path:
    global _session_file
    if _session_file is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _session_file = LOGS_DIR / f"sessao_{timestamp}.log"
    return _session_file


def _write(text: str):
    filepath = _get_session_file()
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(text)


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _next_step(descricao: str):
    global _step_counter
    _step_counter += 1
    _write(f"  [{_timestamp()}] PASSO {_step_counter} — {descricao}\n")


# ==================== SEÇÃO 1: INICIALIZAÇÃO ====================

def log_cabecalho():
    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write(
        "╔════════════════════════════════════════════════════════════╗\n"
        "║          DRAG Qwen 2.5 + LangChain + Redis              ║\n"
        f"║           Sessão: {agora}                    ║\n"
        "╚════════════════════════════════════════════════════════════╝\n\n"
    )


def log_inicio_inicializacao():
    _write(
        "┌────────────────────────────────────────────────────────────┐\n"
        "│              FASE 1: INICIALIZAÇÃO DO SISTEMA              │\n"
        "└────────────────────────────────────────────────────────────┘\n\n"
    )


def log_redis_pronto():
    _next_step("REDIS STACK — Container Docker iniciado, porta 6379 disponível.")


def log_dotenv_carregado(variaveis: dict):
    _next_step("DOTENV — Variáveis de ambiente carregadas:")
    for k, v in variaveis.items():
        _write(f"             {k} = {v}\n")
    _write("\n")


def log_documentos_carregados(num_docs: int, fontes: list):
    _next_step(f"LOADER — {num_docs} documento(s) carregado(s) de:")
    for fonte in fontes:
        _write(f"             - {fonte}\n")
    _write("\n")


def log_chunks_gerados(chunks: list, chunk_size: int, chunk_overlap: int):
    num_chunks = len(chunks)
    _next_step(
        f"SPLITTER — {num_chunks} chunk(s) gerados "
        f"(tamanho={chunk_size}, overlap={chunk_overlap})"
    )
    _write("\n")
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "?")
        page = chunk.metadata.get("page", "?")
        tamanho = len(chunk.page_content)
        preview = chunk.page_content[:150].replace("\n", " ")
        _write(f"    ┌─ Chunk {i+1}/{num_chunks} ──────────────────────────────────\n")
        _write(f"    │ Fonte: {source}\n")
        _write(f"    │ Página: {page}\n")
        _write(f"    │ Tamanho: {tamanho} chars\n")
        _write(f"    │ Conteúdo: \"{preview}...\"\n")
        _write(f"    └──────────────────────────────────────────────────\n\n")
    _write("\n")


def log_embeddings_carregado(model_name: str):
    _next_step(f"EMBEDDINGS — Modelo carregado: {model_name}")
    _write("             Converte texto → vetor de 768 dimensões (CPU)\n\n")


def log_indexacao_redis(num_chunks: int, redis_url: str):
    _next_step(
        f"INDEXAÇÃO — {num_chunks} chunks vetorizados e salvos no Redis "
        f"({redis_url}, índice: rag_docs)"
    )
    _write("\n")


def log_chain_montada():
    _next_step("CHAIN DRAG — Pipeline montado:")
    _write(
        "             Pergunta → Retriever (top-20) → Rerank (top-5)\n"
        "                → format_docs → Prompt (system + human)\n"
        "                → LLM → StrOutputParser → Resposta\n\n"
    )


def log_pronto():
    _next_step("SISTEMA PRONTO — Aguardando perguntas do usuário.")
    _write("\n")


# ==================== SEPARADOR ====================

def log_separador_conversas():
    _write(
        "\n"
        "╔════════════════════════════════════════════════════════════╗\n"
        "║                 FASE 2: CONVERSAS (CHAT)                  ║\n"
        "╚════════════════════════════════════════════════════════════╝\n\n"
    )


# ==================== SEÇÃO 2: INTERAÇÕES ====================

def log_interacao(numero: int, pergunta: str, chunks: list, contexto: str, resposta: str):
    _write(
        f"┌──────────────────────────────────────────────────────────┐\n"
        f"│  INTERAÇÃO #{numero:<3}                          {_timestamp()}  │\n"
        f"└──────────────────────────────────────────────────────────┘\n\n"
    )

    # Passo 1 — Input
    _write(f"  [ENTRADA] Pergunta do usuário:\n")
    _write(f"    \"{pergunta}\"\n\n")

    # Passo 2 — Embedding da pergunta
    _write(f"  [EMBEDDING] Pergunta convertida em vetor de 768 dimensões\n")
    _write(f"    Para busca por similaridade no Redis.\n\n")

    # Passo 3 — Retriever + Rerank
    _write(f"  [RETRIEVER → RERANK] 20 chunks buscados, {len(chunks)} mantidos após reranking:\n\n")
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "?")
        page = chunk.metadata.get("page", "?")
        preview = chunk.page_content[:200].replace("\n", " ")
        _write(f"    ┌─ Chunk {i+1} ─────────────────────────────────────────\n")
        _write(f"    │ Fonte: {source}\n")
        _write(f"    │ Página: {page}\n")
        _write(f"    │ Preview: \"{preview}...\"\n")
        _write(f"    └─────────────────────────────────────────────────────\n\n")

    # Passo 4 — Contexto formatado
    _write(f"  [FORMAT_DOCS] Contexto montado ({len(contexto)} caracteres):\n")
    _write(f"    Cada chunk recebe tag [Fonte: ...] e são separados por ---\n\n")

    # Passo 5 — Prompt
    _write(f"  [PROMPT] Mensagem enviada ao LLM:\n")
    _write(f"    SYSTEM: instruções + contexto ({len(contexto)} chars)\n")
    _write(f"    HUMAN: \"{pergunta}\"\n\n")

    # Passo 6 — LLM
    _write(f"  [LLM] Resposta do modelo (via Ollama):\n")
    _write(f"    \"{resposta}\"\n\n")

    # Separador entre interações
    _write("  " + "─" * 56 + "\n\n")


def log_encerramento(motivo: str):
    _write(
        f"\n"
        f"╔════════════════════════════════════════════════════════════╗\n"
        f"║  SESSÃO ENCERRADA — {_timestamp()}                             ║\n"
        f"║  Motivo: {motivo:<49}║\n"
        f"╚════════════════════════════════════════════════════════════╝\n"
    )
