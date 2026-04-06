"""LLMProvider: gerencia LLM ativo com suporte a switch e fallback."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from src.llm.base import BaseLLMFactory
from src.llm.ollama import OllamaFactory
from src.llm.gemini import GeminiFactory

logger = logging.getLogger("drag.llm")

# Fabricas disponiveis (registrar novas aqui)
_FACTORIES: dict[str, BaseLLMFactory] = {
    "ollama": OllamaFactory(),
    "gemini": GeminiFactory(),
}

FALLBACK_PROVIDER = "ollama"
FALLBACK_MODEL = "qwen2.5:14b"


class LLMProvider:
    """Gerencia o LLM ativo, com switch em runtime e fallback automatico."""

    def __init__(self, provider: str, model: str, **kwargs):
        self.active_provider = provider
        self.active_model = model
        self._kwargs = kwargs
        self._llm = self._create(provider, model)

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    def switch(self, provider: str, model: str):
        """Troca o LLM ativo em runtime."""
        if provider == self.active_provider and model == self.active_model:
            return
        logger.info("Trocando modelo: %s/%s -> %s/%s",
                     self.active_provider, self.active_model, provider, model)
        self.active_provider = provider
        self.active_model = model
        self._llm = self._create(provider, model)

    def fallback_to_local(self):
        """Troca para modelo local (Ollama) quando API esgota."""
        logger.warning("API esgotada — alternando para modelo local (Ollama).")
        self.switch(FALLBACK_PROVIDER, FALLBACK_MODEL)

    def _create(self, provider: str, model: str) -> BaseChatModel:
        factory = _FACTORIES.get(provider)
        if factory is None:
            raise ValueError(
                f"Provider '{provider}' nao registrado. "
                f"Disponiveis: {list(_FACTORIES.keys())}"
            )
        return factory.create(model, **self._kwargs)


def is_api_exhausted(exc: Exception) -> bool:
    """Detecta se a excecao indica quota/rate-limit esgotado da API."""
    exc_type = type(exc).__name__.lower()
    if "resourceexhausted" in exc_type or "ratelimit" in exc_type:
        return True
    cause = exc.__cause__ or exc.__context__
    if cause:
        cause_type = type(cause).__name__.lower()
        if "resourceexhausted" in cause_type or "ratelimit" in cause_type:
            return True
    err_msg = str(exc).lower()
    return any(term in err_msg for term in (
        "429", "quota", "resource_exhausted", "resourceexhausted",
        "rate limit", "rate_limit", "too many requests",
    ))
