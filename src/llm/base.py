"""Base para fabricas de LLM."""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class BaseLLMFactory(ABC):
    """Fabrica que cria instancias de LLM para um provider especifico."""

    provider_name: str = ""

    @abstractmethod
    def create(self, model: str, **kwargs) -> BaseChatModel:
        ...
