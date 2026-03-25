"""Ponto de entrada: CLI interativo para o DRAG Multi-Serviço."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import warnings

from dotenv import load_dotenv

# Suprime warnings do terminal — vão para o log
warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)

# Cores ANSI
AZUL = "\033[34m"
LARANJA = "\033[38;5;208m"
RESET = "\033[0m"

SPINNER_FRAMES = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]

load_dotenv()


def _load_service_config(service_name: str | None):
    """Carrega a config do serviço (pelo nome ou auto-detecta)."""
    from src.config import ServiceConfig, get_default_service

    name = service_name or get_default_service()
    return ServiceConfig.load(name)


def cmd_ingest(args):
    from src.ingest import run_ingest
    from src.logger import (
        log_cabecalho, log_inicio_inicializacao, log_redis_pronto,
        log_config_carregado,
    )

    config = _load_service_config(args.service)

    log_cabecalho(config.display_name)
    log_inicio_inicializacao()
    log_redis_pronto()
    log_config_carregado(config.as_dict())

    print(f"{LARANJA}Serviço: {config.display_name}{RESET}")

    stop = threading.Event()
    t = threading.Thread(target=lambda: _ingest_spinner(stop), daemon=True)
    t.start()
    result = run_ingest(config, force=args.force)
    stop.set()
    t.join()
    if result is None:
        print(f"\r{LARANJA}Documentos não mudaram. Ingestão pulada.{RESET}          ")
    else:
        print(f"\r{LARANJA}Ingestão concluída com sucesso!{RESET}          ")


def _ingest_spinner(stop_event):
    etapas = ["Carregando modelo de embeddings", "Lendo documentos", "Gerando chunks", "Indexando no Redis"]
    i = 0
    while not stop_event.is_set():
        etapa = etapas[min(i // 30, len(etapas) - 1)]
        frame = SPINNER_FRAMES[i % len(SPINNER_FRAMES)]
        print(f"\r{LARANJA}{frame} {etapa}...{RESET}          ", end="", flush=True)
        i += 1
        time.sleep(0.1)


def spinner_loop(stop_event, status=None):
    i = 0
    while not stop_event.is_set():
        msg = status[0] if status else "Pensando..."
        print(f"\r{LARANJA}{SPINNER_FRAMES[i % len(SPINNER_FRAMES)]} {msg}{RESET}          ", end="", flush=True)
        i += 1
        time.sleep(0.1)
    print("\r" + " " * 60 + "\r", end="", flush=True)


def cmd_ask(args):
    from src.pipeline import ask

    config = _load_service_config(args.service)
    resposta = ask(config, args.question)
    print(f"\n{LARANJA}{resposta}{RESET}")


def cmd_chat(args):
    from src.pipeline import DragPipeline
    from src.logger import (
        log_chain_montada, log_pronto,
        log_separador_conversas, log_interacao, log_encerramento,
    )

    config = _load_service_config(args.service)
    pipeline = DragPipeline(config)

    log_chain_montada(list(config.pipeline_steps))
    log_pronto()
    log_separador_conversas()
    print(f"Chat DRAG iniciado ({config.display_name}). Digite 'sair' para encerrar.\n")

    interacao_num = 0
    while True:
        try:
            pergunta = input(f"{AZUL}Você: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            log_encerramento("Ctrl+C pelo usuário")
            print("\nAté logo!")
            break

        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "exit", "quit"):
            log_encerramento("Comando sair")
            print("Até logo!")
            break

        interacao_num += 1

        status = ["Pensando..."]
        stop = threading.Event()
        t = threading.Thread(target=spinner_loop, args=(stop, status), daemon=True)
        t.start()
        result = pipeline.invoke(pergunta, status=status)
        stop.set()
        t.join()

        log_interacao(
            interacao_num,
            pergunta,
            result.chunks,
            result.contexto,
            result.resposta,
            result.generated_queries,
            result.raw_count,
            result.rerank_count,
        )
        print(f"{LARANJA}IA: {result.resposta}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description="DRAG Multi-Serviço + LangChain + Redis"
    )
    parser.add_argument(
        "--service", "-s", default=None,
        help="Nome do serviço (default: auto-detecta)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Comando: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Indexar documentos no Redis")
    ingest_parser.add_argument(
        "--docs-dir", default=None, help="Diretório com os documentos (sobrescreve o YAML)"
    )
    ingest_parser.add_argument(
        "--force", action="store_true", help="Forçar reingestão mesmo sem mudanças nos documentos"
    )
    ingest_parser.add_argument(
        "--service", "-s", default=None, help="Nome do serviço"
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    # Comando: ask
    ask_parser = subparsers.add_parser("ask", help="Fazer uma pergunta ao DRAG")
    ask_parser.add_argument("question", help="Pergunta para o DRAG")
    ask_parser.add_argument("--service", "-s", default=None, help="Nome do serviço")
    ask_parser.set_defaults(func=cmd_ask)

    # Comando: chat
    chat_parser = subparsers.add_parser("chat", help="Chat interativo com o DRAG")
    chat_parser.add_argument("--service", "-s", default=None, help="Nome do serviço")
    chat_parser.set_defaults(func=cmd_chat)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
