"""Fabrica de LLMs via Ollama (local)."""

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.config import env
from src.llm.base import BaseLLMFactory


class OllamaFactory(BaseLLMFactory):
    provider_name = "ollama"

    def create(self, model: str, **kwargs) -> BaseChatModel:
        return ChatOllama(
            model=model,
            base_url=env.OLLAMA_BASE_URL,
            temperature=kwargs.get("temperature", 0.55),
        )
