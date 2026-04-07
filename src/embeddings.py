"""Fabrica de embeddings com registry e cache."""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings


# ---------------------------------------------------------------------------
# Registry de fabricas de embeddings
# ---------------------------------------------------------------------------

EMBEDDINGS_REGISTRY: dict[str, object] = {}


class BaseEmbeddingsFactory(ABC):
    """Fabrica abstrata para criar instancias de embeddings."""

    provider_name: str = ""

    @abstractmethod
    def create(self, model_name: str) -> Embeddings:
        ...


def register_embeddings(cls: type[BaseEmbeddingsFactory]) -> type[BaseEmbeddingsFactory]:
    """Decorator que registra uma factory no EMBEDDINGS_REGISTRY."""
    if cls.provider_name:
        EMBEDDINGS_REGISTRY[cls.provider_name] = cls()
    return cls


# ---------------------------------------------------------------------------
# Implementacao: HuggingFace (default)
# ---------------------------------------------------------------------------

@register_embeddings
class HuggingFaceEmbeddingsFactory(BaseEmbeddingsFactory):
    provider_name = "huggingface"

    def create(self, model_name: str) -> Embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
        )


# ---------------------------------------------------------------------------
# Cache e funcao de acesso
# ---------------------------------------------------------------------------

_cache: dict[str, Embeddings] = {}

# Provider padrao (pode ser sobrescrito via config no futuro)
_default_provider = "huggingface"


def get_embeddings(model_name: str, provider: str | None = None) -> Embeddings:
    """Retorna instancia de embeddings (cacheada por model_name)."""
    cache_key = f"{provider or _default_provider}:{model_name}"
    if cache_key in _cache:
        return _cache[cache_key]

    factory = EMBEDDINGS_REGISTRY.get(provider or _default_provider)
    if factory is None:
        raise ValueError(
            f"Embeddings provider '{provider}' nao registrado. "
            f"Disponiveis: {list(EMBEDDINGS_REGISTRY.keys())}"
        )

    instance = factory.create(model_name)
    _cache[cache_key] = instance
    return instance
