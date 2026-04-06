"""Base para steps de reranking."""

from abc import abstractmethod

from src.steps.base import BaseStep, PipelineData


class BaseReranker(BaseStep):
    """Step que reordena chunks por relevancia."""

    @abstractmethod
    def execute(self, data: PipelineData) -> PipelineData:
        ...