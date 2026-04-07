"""Base para fabricas de LLM com auto-registro."""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

# Registry populado automaticamente via @register_llm
LLM_REGISTRY: dict[str, object] = {}


class BaseLLMFactory(ABC):
    """Fabrica que cria instancias de LLM para um provider especifico."""

    provider_name: str = ""

    @abstractmethod
    def create(self, model: str, **kwargs) -> BaseChatModel:
        ...


def register_llm(cls: type[BaseLLMFactory]) -> type[BaseLLMFactory]:
    """Decorator que registra uma factory no LLM_REGISTRY pelo provider_name."""
    if cls.provider_name:
        LLM_REGISTRY[cls.provider_name] = cls()
    return cls
