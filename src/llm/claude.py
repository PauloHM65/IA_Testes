"""Fabrica de LLMs via Anthropic Claude (API)."""

from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic

from src.config import env
from src.llm.base import BaseLLMFactory, register_llm


@register_llm
class ClaudeFactory(BaseLLMFactory):
    provider_name = "claude"

    def create(self, model: str, **kwargs) -> BaseChatModel:
        return ChatAnthropic(
            model=model,
            api_key=env.ANTHROPIC_API_KEY,
            temperature=kwargs.get("temperature", 0.55),
            max_retries=kwargs.get("max_retries", 0),
        )