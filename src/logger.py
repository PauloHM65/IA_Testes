"""Módulo de logging: registra cada etapa do pipeline DRAG em arquivos."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class SessionLogger:
    """Logger de sessão thread-safe, sem estado global mutável."""

    def __init__(self):
        self._lock = threading.Lock()
        self._session_file: Path | None = None
        self._step_counter = 0

    def _get_session_file(self) -> Path:
        if self._session_file is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._session_file = LOGS_DIR / f"sessao_{timestamp}.log"
        return self._session_file

    def _write(self, text: str):
        filepath = self._get_session_file()
        with self._lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(text)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _next_step(self, descricao: str):
        self._step_counter += 1
        self._write(f"  [{self._timestamp()}] PASSO {self._step_counter} — {descricao}\n")


# Instância singleton usada pelas funções de módulo
_logger = SessionLogger()


# ==================== SEÇÃO 1: INICIALIZAÇÃO ====================

def log_cabecalho(service_display_name: str = ""):
    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _logger._write(
        "╔════════════════════════════════════════════════════════════╗\n"
        "║          DRAG Multi-Serviço + LangChain + Redis           ║\n"
        f"║  Sessão: {agora}                               ║\n"
    )
    if service_display_name:
        _logger._write(
            f"║  Serviço: {service_display_name:<48}║\n"
        )
    _logger._write(
        "╚════════════════════════════════════════════════════════════╝\n\n"
    )


def log_inicio_inicializacao():
    _logger._write(
        "┌────────────────────────────────────────────────────────────┐\n"
        "│              FASE 1: INICIALIZAÇÃO DO SISTEMA              │\n"
        "└────────────────────────────────────────────────────────────┘\n\n"
    )


def log_redis_pronto():
    _logger._next_step("REDIS STACK — Container Docker iniciado.")


def log_config_carregado(variaveis: dict):
    _logger._next_step("CONFIG — Configuração do serviço carregada:")
    for k, v in variaveis.items():
        _logger._write(f"             {k} = {v}\n")
    _logger._write("\n")


def log_documentos_carregados(num_docs: int, fontes: list):
    _logger._next_step(f"LOADER — {num_docs} documento(s) carregado(s) de:")
    for fonte in fontes:
        _logger._write(f"             - {fonte}\n")
    _logger._write("\n")


def log_chunks_gerados(chunks: list, chunk_size: int, chunk_overlap: int):
    num_chunks = len(chunks)
    _logger._next_step(
        f"SPLITTER — {num_chunks} chunk(s) gerados "
        f"(tamanho={chunk_size}, overlap={chunk_overlap})"
    )
    _logger._write("\n")
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "?")
        page = chunk.metadata.get("page", "?")
        tamanho = len(chunk.page_content)
        preview = chunk.page_content[:150].replace("\n", " ")
        _logger._write(f"    ┌─ Chunk {i+1}/{num_chunks} ──────────────────────────────────\n")
        _logger._write(f"    │ Fonte: {source}\n")
        _logger._write(f"    │ Página: {page}\n")
        _logger._write(f"    │ Tamanho: {tamanho} chars\n")
        _logger._write(f"    │ Conteúdo: \"{preview}...\"\n")
        _logger._write(f"    └──────────────────────────────────────────────────\n\n")
    _logger._write("\n")


def log_embeddings_carregado(model_name: str):
    _logger._next_step(f"EMBEDDINGS — Modelo carregado: {model_name}")
    _logger._write("             Converte texto → vetor numérico (CPU)\n\n")


def log_indexacao_redis(num_chunks: int, redis_url: str, index_name: str = ""):
    extra = f", índice: {index_name}" if index_name else ""
    _logger._next_step(
        f"INDEXAÇÃO — {num_chunks} chunks vetorizados e salvos no Redis "
        f"({redis_url}{extra})"
    )
    _logger._write("\n")


def log_chain_montada(pipeline_steps: list[str]):
    _logger._next_step("PIPELINE — Etapas configuradas:")
    for i, step in enumerate(pipeline_steps, 1):
        _logger._write(f"             {i}. {step}\n")
    _logger._write("\n")


def log_pronto():
    _logger._next_step("SISTEMA PRONTO — Aguardando perguntas do usuário.")
    _logger._write("\n")


def log_alerta(mensagem: str):
    _logger._write(f"  [{_logger._timestamp()}] ALERTA — {mensagem}\n\n")


# ==================== SEPARADOR ====================

def log_separador_conversas():
    _logger._write(
        "\n"
        "╔════════════════════════════════════════════════════════════╗\n"
        "║                 FASE 2: CONVERSAS (CHAT)                  ║\n"
        "╚════════════════════════════════════════════════════════════╝\n\n"
    )


# ==================== SEÇÃO 2: INTERAÇÕES ====================

def log_interacao(
    numero: int,
    pergunta: str,
    chunks: list,
    contexto: str,
    resposta: str,
    generated_queries: list[str],
    raw_count: int,
    rerank_count: int,
):
    """Registra uma interação completa. Recebe todos os dados prontos."""
    _logger._write(
        f"┌──────────────────────────────────────────────────────────┐\n"
        f"│  INTERAÇÃO #{numero:<3}                          {_logger._timestamp()}  │\n"
        f"└──────────────────────────────────────────────────────────┘\n\n"
    )

    # Entrada
    _logger._write(f"  [ENTRADA] Pergunta do usuário:\n")
    _logger._write(f"    \"{pergunta}\"\n\n")

    # MultiQuery
    if generated_queries:
        total_queries = len(generated_queries) + 1
        _logger._write(f"  [MULTIQUERY] {total_queries} queries ({len(generated_queries)} geradas + original):\n")
        _logger._write(f"    1. \"{pergunta}\" (original)\n")
        for i, q in enumerate(generated_queries, start=2):
            _logger._write(f"    {i}. \"{q}\"\n")
        _logger._write("\n")

    # Retriever
    _logger._write(f"  [RETRIEVER] {raw_count} chunks recuperados\n\n")

    # Chunks finais
    _logger._write(f"  [RESULTADO] {rerank_count} chunks finais:\n\n")
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "?")
        page = chunk.metadata.get("page", "?")
        preview = chunk.page_content[:200].replace("\n", " ")
        _logger._write(f"    ┌─ Chunk {i+1} ─────────────────────────────────────────\n")
        _logger._write(f"    │ Fonte: {source}\n")
        _logger._write(f"    │ Página: {page}\n")
        _logger._write(f"    │ Preview: \"{preview}...\"\n")
        _logger._write(f"    └─────────────────────────────────────────────────────\n\n")

    # Contexto
    _logger._write(f"  [FORMAT_DOCS] Contexto montado ({len(contexto)} caracteres)\n\n")

    # Prompt
    _logger._write(f"  [PROMPT] Mensagem enviada ao LLM:\n")
    _logger._write(f"    SYSTEM: instruções + contexto ({len(contexto)} chars)\n")
    _logger._write(f"    HUMAN: \"{pergunta}\"\n\n")

    # LLM
    _logger._write(f"  [LLM] Resposta do modelo (via Ollama):\n")
    _logger._write(f"    \"{resposta}\"\n\n")

    _logger._write("  " + "─" * 56 + "\n\n")


def log_encerramento(motivo: str):
    _logger._write(
        f"\n"
        f"╔════════════════════════════════════════════════════════════╗\n"
        f"║  SESSÃO ENCERRADA — {_logger._timestamp()}                             ║\n"
        f"║  Motivo: {motivo:<49}║\n"
        f"╚════════════════════════════════════════════════════════════╝\n"
    )
