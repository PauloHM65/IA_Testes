"""Display para terminal com cores ANSI."""

from src.ui.base_display import BaseDisplay

AZUL = "\033[34m"
LARANJA = "\033[38;5;208m"
RESET = "\033[0m"


class TerminalDisplay(BaseDisplay):

    def info(self, msg: str):
        print(f"{LARANJA}{msg}{RESET}")

    def success(self, msg: str):
        print(f"\r{LARANJA}{msg}{RESET}          ")

    def error(self, msg: str):
        print(f"\r{LARANJA}{msg}{RESET}")

    def prompt(self, label: str = "Voce") -> str:
        try:
            return input(f"{AZUL}{label}: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            raise

    def ai_response(self, text: str):
        print(f"{LARANJA}IA: {text}{RESET}\n")

    def banner(self, title: str):
        print(f"{AZUL}{'━' * 43}{RESET}")
        print(f"{AZUL}  {title}{RESET}")
        print(f"{AZUL}{'━' * 43}{RESET}")
        print()
