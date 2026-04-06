"""Base para steps de geracao de resposta."""

from abc import abstractmethod

from src.steps.base import BaseStep, PipelineData


class BaseGenerator(BaseStep):
    """Step que gera resposta usando LLM."""

    @abstractmethod
    def execute(self, data: PipelineData) -> PipelineData:
        ...

    @staticmethod
    def format_docs(docs: list) -> str:
        """Formata chunks com tag [Fonte: ...]."""
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "desconhecido")
            parts.append(f"[Fonte: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)
