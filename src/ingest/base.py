"""Base para pipelines de ingestao."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import ServiceConfig
    from src.logging.base import IngestLogger


class BaseIngester(ABC):
    """Pipeline de ingestao de documentos."""

    def __init__(self, config: ServiceConfig, logger: IngestLogger):
        self.config = config
        self.logger = logger

    @abstractmethod
    def run(self, force: bool = False) -> bool | None:
        """Executa ingestao. Retorna None se pulou (sem mudancas), True se ingeriu."""
        ...
