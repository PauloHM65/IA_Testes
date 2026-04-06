"""Base para pipelines de ingestao."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import ServiceConfig


class BaseIngester(ABC):
    """Pipeline de ingestao de documentos."""

    def __init__(self, config: ServiceConfig):
        self.config = config

    @abstractmethod
    def run(self, force: bool = False) -> bool | None:
        """Executa ingestao. Retorna None se pulou (sem mudancas), True se ingeriu."""
        ...
