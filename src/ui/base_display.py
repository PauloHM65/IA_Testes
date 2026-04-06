"""Base para exibicao de mensagens ao usuario."""

from abc import ABC, abstractmethod


class BaseDisplay(ABC):
    """Interface de I/O com o usuario."""

    @abstractmethod
    def info(self, msg: str): ...

    @abstractmethod
    def success(self, msg: str): ...

    @abstractmethod
    def error(self, msg: str): ...

    @abstractmethod
    def prompt(self, label: str) -> str: ...

    @abstractmethod
    def ai_response(self, text: str): ...

    @abstractmethod
    def banner(self, title: str): ...
