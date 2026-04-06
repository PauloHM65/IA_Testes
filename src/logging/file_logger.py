"""Logger que escreve em arquivos de sessao (logs/sessao_*.log)."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from src.logging.base import BaseLogger

LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class FileLogger(BaseLogger):
    """Logger de sessao thread-safe que escreve em arquivo."""

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
        self._write(f"  [{self._timestamp()}] PASSO {self._step_counter} \u2014 {descricao}\n")

    # --- Interface BaseLogger ---

    def cabecalho(self, service_name: str = ""):
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(
            "\u2554" + "\u2550" * 58 + "\u2557\n"
            "\u2551          DRAG Multi-Servico + LangChain + Redis           \u2551\n"
            f"\u2551  Sess\u00e3o: {agora}                               \u2551\n"
        )
        if service_name:
            self._write(f"\u2551  Servico: {service_name:<48}\u2551\n")
        self._write("\u255a" + "\u2550" * 58 + "\u255d\n\n")

    def section(self, titulo: str):
        self._write(
            "\u250c" + "\u2500" * 58 + "\u2510\n"
            f"\u2502  {titulo:<56}\u2502\n"
            "\u2514" + "\u2500" * 58 + "\u2518\n\n"
        )

    def step(self, descricao: str):
        self._next_step(descricao)

    def alerta(self, mensagem: str):
        self._write(f"  [{self._timestamp()}] ALERTA \u2014 {mensagem}\n\n")

    def config_carregado(self, variaveis: dict):
        self._next_step("CONFIG \u2014 Configuracao do servico carregada:")
        for k, v in variaveis.items():
            self._write(f"             {k} = {v}\n")
        self._write("\n")

    def documentos_carregados(self, num_docs: int, fontes: list):
        self._next_step(f"LOADER \u2014 {num_docs} documento(s) carregado(s) de:")
        for fonte in fontes:
            self._write(f"             - {fonte}\n")
        self._write("\n")

    def chunks_gerados(self, chunks: list, chunk_size: int, chunk_overlap: int):
        num_chunks = len(chunks)
        self._next_step(
            f"SPLITTER \u2014 {num_chunks} chunk(s) gerados "
            f"(tamanho={chunk_size}, overlap={chunk_overlap})"
        )
        self._write("\n")
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "?")
            page = chunk.metadata.get("page", "?")
            tamanho = len(chunk.page_content)
            preview = chunk.page_content[:150].replace("\n", " ")
            self._write(f"    \u250c\u2500 Chunk {i+1}/{num_chunks} " + "\u2500" * 30 + "\n")
            self._write(f"    \u2502 Fonte: {source}\n")
            self._write(f"    \u2502 P\u00e1gina: {page}\n")
            self._write(f"    \u2502 Tamanho: {tamanho} chars\n")
            self._write(f"    \u2502 Conte\u00fado: \"{preview}...\"\n")
            self._write(f"    \u2514" + "\u2500" * 50 + "\n\n")
        self._write("\n")

    def embeddings_carregado(self, model_name: str):
        self._next_step(f"EMBEDDINGS \u2014 Modelo carregado: {model_name}")
        self._write("             Converte texto \u2192 vetor num\u00e9rico (CPU)\n\n")

    def indexacao_redis(self, num_chunks: int, redis_url: str, index_name: str = ""):
        extra = f", \u00edndice: {index_name}" if index_name else ""
        self._next_step(
            f"INDEXA\u00c7\u00c3O \u2014 {num_chunks} chunks vetorizados e salvos no Redis "
            f"({redis_url}{extra})"
        )
        self._write("\n")

    def pipeline_montado(self, steps: list[str]):
        self._next_step("PIPELINE \u2014 Etapas configuradas:")
        for i, s in enumerate(steps, 1):
            self._write(f"             {i}. {s}\n")
        self._write("\n")

    def pronto(self):
        self._next_step("SISTEMA PRONTO \u2014 Aguardando perguntas do usu\u00e1rio.")
        self._write("\n")

    def separador_conversas(self):
        self._write(
            "\n"
            "\u2554" + "\u2550" * 58 + "\u2557\n"
            "\u2551                 FASE 2: CONVERSAS (CHAT)                  \u2551\n"
            "\u255a" + "\u2550" * 58 + "\u255d\n\n"
        )

    def interacao(self, numero: int, pergunta: str, chunks: list, contexto: str,
                  resposta: str, generated_queries: list[str], raw_count: int,
                  rerank_count: int):
        self._write(
            f"\u250c" + "\u2500" * 56 + "\u2510\n"
            f"\u2502  INTERA\u00c7\u00c3O #{numero:<3}                          {self._timestamp()}  \u2502\n"
            f"\u2514" + "\u2500" * 56 + "\u2518\n\n"
        )

        self._write(f"  [ENTRADA] Pergunta do usu\u00e1rio:\n")
        self._write(f"    \"{pergunta}\"\n\n")

        if generated_queries:
            total = len(generated_queries) + 1
            self._write(f"  [MULTIQUERY] {total} queries ({len(generated_queries)} geradas + original):\n")
            self._write(f"    1. \"{pergunta}\" (original)\n")
            for i, q in enumerate(generated_queries, start=2):
                self._write(f"    {i}. \"{q}\"\n")
            self._write("\n")

        self._write(f"  [RETRIEVER] {raw_count} chunks recuperados\n\n")

        self._write(f"  [RESULTADO] {rerank_count} chunks finais:\n\n")
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "?")
            page = chunk.metadata.get("page", "?")
            preview = chunk.page_content[:200].replace("\n", " ")
            self._write(f"    \u250c\u2500 Chunk {i+1} " + "\u2500" * 45 + "\n")
            self._write(f"    \u2502 Fonte: {source}\n")
            self._write(f"    \u2502 P\u00e1gina: {page}\n")
            self._write(f"    \u2502 Preview: \"{preview}...\"\n")
            self._write(f"    \u2514" + "\u2500" * 51 + "\n\n")

        self._write(f"  [FORMAT_DOCS] Contexto montado ({len(contexto)} caracteres)\n\n")

        self._write(f"  [PROMPT] Mensagem enviada ao LLM:\n")
        self._write(f"    SYSTEM: instru\u00e7\u00f5es + contexto ({len(contexto)} chars)\n")
        self._write(f"    HUMAN: \"{pergunta}\"\n\n")

        self._write(f"  [LLM] Resposta do modelo (via Ollama):\n")
        self._write(f"    \"{resposta}\"\n\n")

        self._write("  " + "\u2500" * 56 + "\n\n")

    def encerramento(self, motivo: str):
        self._write(
            f"\n"
            f"\u2554" + "\u2550" * 58 + "\u2557\n"
            f"\u2551  SESS\u00c3O ENCERRADA \u2014 {self._timestamp()}                             \u2551\n"
            f"\u2551  Motivo: {motivo:<49}\u2551\n"
            f"\u255a" + "\u2550" * 58 + "\u255d\n"
        )


# Singleton para uso global (compatibilidade)
_default_logger = FileLogger()


def get_logger() -> FileLogger:
    return _default_logger
