"""Fabrica de LLMs via Google Gemini (API)."""

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import env
from src.llm.base import BaseLLMFactory, register_llm


@register_llm
class GeminiFactory(BaseLLMFactory):
    provider_name = "gemini"

    def create(self, model: str, **kwargs) -> BaseChatModel:
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=env.GOOGLE_API_KEY,
            temperature=kwargs.get("temperature", 0.55),
            max_retries=kwargs.get("max_retries", 0),
        )
