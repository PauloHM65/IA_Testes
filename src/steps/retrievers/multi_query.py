"""MultiQuery: gera variacoes da pergunta e busca para cada uma."""

from __future__ import annotations

from src.steps import register
from src.steps.base import PipelineData
from src.steps.retrievers.base import BaseRetriever


class _CaptureMultiQuery:
    """Context manager que intercepta as queries geradas pelo MultiQueryRetriever."""

    def __init__(self, retriever, output: list[str]):
        self._retriever = retriever
        self._output = output
        self._original = None

    def __enter__(self):
        original_generate = self._retriever.generate_queries

        def _capturing_generate(question, run_manager):
            queries = original_generate(question, run_manager)
            self._output.extend(queries)
            return queries

        self._original = original_generate
        self._retriever.generate_queries = _capturing_generate
        return self

    def __exit__(self, *exc):
        self._retriever.generate_queries = self._original


@register
class MultiQueryRetrieverStep(BaseRetriever):
    name = "multi_query"
    label = "Gerando variacoes da pergunta (MultiQuery)"

    def execute(self, data: PipelineData) -> PipelineData:
        with _CaptureMultiQuery(self.ctx.multi_retriever, data.generated_queries):
            data.raw_chunks = self.ctx.multi_retriever.invoke(data.pergunta)
        data.raw_count = len(data.raw_chunks)
        return data
