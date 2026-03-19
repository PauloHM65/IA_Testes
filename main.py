"""Ponto de entrada: CLI interativo para o DRAG com Qwen 2.5 + Redis."""

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


def cmd_ingest(args):
    import os
    from src.ingest import run_ingest
    from src.logger import (
        log_cabecalho, log_inicio_inicializacao, log_redis_pronto,
        log_dotenv_carregado,
    )

    log_cabecalho()
    log_inicio_inicializacao()
    log_redis_pronto()
    log_dotenv_carregado({
        "REDIS_URL": os.getenv("REDIS_URL", ""),
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", ""),
        "LLM_MODEL": os.getenv("LLM_MODEL", ""),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", ""),
        "RERANK_MODEL": os.getenv("RERANK_MODEL", ""),
        "RERANK_TOP_N": os.getenv("RERANK_TOP_N", ""),
        "CHUNK_SIZE": os.getenv("CHUNK_SIZE", ""),
        "CHUNK_OVERLAP": os.getenv("CHUNK_OVERLAP", ""),
    })

    stop = threading.Event()
    t = threading.Thread(target=lambda: _ingest_spinner(stop), daemon=True)
    t.start()
    run_ingest(args.docs_dir)
    stop.set()
    t.join()
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


def spinner_loop(stop_event):
    i = 0
    while not stop_event.is_set():
        print(f"\r{LARANJA}{SPINNER_FRAMES[i % len(SPINNER_FRAMES)]} Pensando...{RESET}", end="", flush=True)
        i += 1
        time.sleep(0.1)
    print("\r" + " " * 20 + "\r", end="", flush=True)


def invoke_with_spinner(chain, pergunta):
    stop = threading.Event()
    t = threading.Thread(target=spinner_loop, args=(stop,), daemon=True)
    t.start()
    resposta = chain.invoke(pergunta)
    stop.set()
    t.join()
    return resposta


def cmd_ask(args):
    from src.rag_chain import ask

    resposta = ask(args.question)
    print(f"\n{LARANJA}{resposta}{RESET}")


def cmd_chat(_args):
    from src.rag_chain import build_drag_components, invoke_with_log
    from src.logger import (
        log_chain_montada, log_pronto,
        log_separador_conversas, log_interacao, log_encerramento,
    )

    retriever, answer_chain = build_drag_components()
    log_chain_montada()
    log_pronto()
    log_separador_conversas()
    print("Chat DRAG iniciado. Digite 'sair' para encerrar.\n")

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

        stop = threading.Event()
        t = threading.Thread(target=spinner_loop, args=(stop,), daemon=True)
        t.start()
        resposta, chunks, contexto = invoke_with_log(retriever, answer_chain, pergunta)
        stop.set()
        t.join()

        log_interacao(interacao_num, pergunta, chunks, contexto, resposta)
        print(f"{LARANJA}IA: {resposta}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description="DRAG com Qwen 2.5 + LangChain + Redis"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Comando: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Indexar documentos no Redis")
    ingest_parser.add_argument(
        "--docs-dir", default="fontes", help="Diretório com os documentos (default: fontes)"
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    # Comando: ask
    ask_parser = subparsers.add_parser("ask", help="Fazer uma pergunta ao RAG")
    ask_parser.add_argument("question", help="Pergunta para o RAG")
    ask_parser.set_defaults(func=cmd_ask)

    # Comando: chat
    chat_parser = subparsers.add_parser("chat", help="Chat interativo com o RAG")
    chat_parser.set_defaults(func=cmd_chat)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
