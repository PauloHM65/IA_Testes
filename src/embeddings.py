"""Fábrica de embeddings: seleciona o modelo correto baseado no nome."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings


# Wrapper para modelos E5 (desativado — bge-m3 nao precisa de prefixos)
# Manter caso volte para um modelo E5 no futuro.
# class E5Embeddings(HuggingFaceEmbeddings):
#     """Wrapper que adiciona os prefixos 'query: ' / 'passage: ' exigidos por modelos E5."""
#
#     def embed_query(self, text: str) -> list[float]:
#         return super().embed_query(f"query: {text}")
#
#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         return super().embed_documents([f"passage: {t}" for t in texts])


# Cache por model_name para evitar carregar o mesmo modelo 2x
_cache: dict[str, HuggingFaceEmbeddings] = {}


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Retorna instância de embeddings (cacheada por model_name)."""
    if model_name in _cache:
        return _cache[model_name]

    # Selecao de classe: E5 precisa de prefixos, outros modelos (bge-m3, etc) nao
    # cls = E5Embeddings if "e5" in model_name.lower() else HuggingFaceEmbeddings
    cls = HuggingFaceEmbeddings

    instance = cls(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )

    _cache[model_name] = instance
    return instance
