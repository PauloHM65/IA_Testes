"""MultiQuery: gera variacoes da pergunta e busca para cada uma."""

from __future__ import annotations

from src.steps import register
from src.steps.base import PipelineData
from src.steps.retrievers.base import BaseRetriever


@register
class MultiQueryRetrieverStep(BaseRetriever):
    name = "multi_query"
    label = "Gerando variacoes da pergunta (MultiQuery)"

    def execute(self, data: PipelineData) -> PipelineData:
        from src.pipeline import _capture_multiquery

        with _capture_multiquery(self.ctx.multi_retriever, data.generated_queries):
            data.raw_chunks = self.ctx.multi_retriever.invoke(data.pergunta)
        data.raw_count = len(data.raw_chunks)
        return data
