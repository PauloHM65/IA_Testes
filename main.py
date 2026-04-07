"""Ponto de entrada: CLI interativo para o DRAG Multi-Servico."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)

load_dotenv()


def _load_service_config(service_name: str | None):
    from src.config import ServiceConfig, get_default_service
    name = service_name or get_default_service()
    return ServiceConfig.load(name)


def cmd_ingest(args):
    from src.ingest import RedisIngester
    from src.logging import FileLogger

    config = _load_service_config(args.service)
    logger = FileLogger()

    logger.cabecalho(config.display_name)
    logger.section("FASE 1: INICIALIZACAO DO SISTEMA")
    logger.step("REDIS STACK — Container Docker iniciado.")
    logger.config_carregado(config.as_dict())

    from src.ui import TerminalDisplay, SpinnerAnimation
    display = TerminalDisplay()
    display.info(f"Servico: {config.display_name}")

    spinner = SpinnerAnimation()
    spinner.update("Carregando modelo de embeddings...")
    spinner.start()

    ingester = RedisIngester(config, logger)
    result = ingester.run(force=args.force)

    spinner.stop()

    if result is None:
        display.success("Documentos nao mudaram. Ingestao pulada.")
    else:
        display.success("Ingestao concluida com sucesso!")


def cmd_ask(args):
    from src.pipeline import ask

    config = _load_service_config(args.service)
    resposta = ask(config, args.question)

    from src.ui import TerminalDisplay
    TerminalDisplay().ai_response(resposta)


def cmd_chat(args):
    from src.pipeline import DragPipeline
    from src.logging import FileLogger
    from src.ui import TerminalDisplay, SpinnerAnimation

    config = _load_service_config(args.service)
    pipeline = DragPipeline(config)

    logger = FileLogger()
    display = TerminalDisplay()

    logger.pipeline_montado(list(config.pipeline_steps))
    logger.pronto()
    logger.separador_conversas()

    print(f"Chat DRAG iniciado ({config.display_name}). Digite 'sair' para encerrar.\n")

    interacao_num = 0
    while True:
        try:
            pergunta = display.prompt("Voce")
        except (KeyboardInterrupt, EOFError):
            logger.encerramento("Ctrl+C pelo usuario")
            print("\nAte logo!")
            break

        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "exit", "quit"):
            logger.encerramento("Comando sair")
            print("Ate logo!")
            break

        interacao_num += 1

        status = ["Pensando..."]
        spinner = SpinnerAnimation()
        spinner.update(status[0])
        spinner.start()

        # O pipeline atualiza status[0], o spinner le via update
        import threading
        stop = threading.Event()

        def _sync_status():
            last = ""
            while not stop.is_set():
                if status[0] != last:
                    last = status[0]
                    spinner.update(last)
                stop.wait(0.1)

        sync_thread = threading.Thread(target=_sync_status, daemon=True)
        sync_thread.start()

        result = pipeline.invoke(pergunta, status=status)

        stop.set()
        sync_thread.join()
        spinner.stop()

        logger.interacao(
            interacao_num,
            pergunta,
            result.chunks,
            result.contexto,
            result.resposta,
            result.generated_queries,
            result.raw_count,
            result.rerank_count,
        )
        display.ai_response(result.resposta)


def main():
    parser = argparse.ArgumentParser(
        description="DRAG Multi-Servico + LangChain + Redis"
    )
    parser.add_argument(
        "--service", "-s", default=None,
        help="Nome do servico (default: auto-detecta)"
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Indexar documentos no Redis")
    ingest_parser.add_argument("--docs-dir", default=None)
    ingest_parser.add_argument("--force", action="store_true")
    ingest_parser.add_argument("--service", "-s", default=None)
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="Fazer uma pergunta ao DRAG")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--service", "-s", default=None)
    ask_parser.set_defaults(func=cmd_ask)

    chat_parser = subparsers.add_parser("chat", help="Chat interativo com o DRAG")
    chat_parser.add_argument("--service", "-s", default=None)
    chat_parser.set_defaults(func=cmd_chat)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
