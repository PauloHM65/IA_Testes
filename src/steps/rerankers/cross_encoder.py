"""Reranking via cross-encoder (sentence-transformers)."""

from src.steps import register
from src.steps.base import PipelineData
from src.steps.rerankers.base import BaseReranker


@register
class CrossEncoderReranker(BaseReranker):
    name = "rerank"
    label = "Reranking com cross-encoder"

    def execute(self, data: PipelineData) -> PipelineData:
        source = data.chunks if data.chunks else data.raw_chunks
        if not source:
            return data
        pairs = [[data.pergunta, doc.page_content] for doc in source]
        scores = self.ctx.reranker.predict(pairs)
        ranked = sorted(zip(scores, source), key=lambda x: x[0], reverse=True)
        data.chunks = [doc for _, doc in ranked[:self.ctx.config.rerank_top_n]]
        data.rerank_count = len(data.chunks)
        return data
