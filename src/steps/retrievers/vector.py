"""Busca vetorial direta (sem MultiQuery)."""

from src.steps import register
from src.steps.base import PipelineData
from src.steps.retrievers.base import BaseRetriever


@register
class VectorRetriever(BaseRetriever):
    name = "retrieve"
    label = "Buscando chunks relevantes"

    def execute(self, data: PipelineData) -> PipelineData:
        if not data.raw_chunks:
            data.raw_chunks = self.ctx.base_retriever.invoke(data.pergunta)
            data.raw_count = len(data.raw_chunks)
        return data