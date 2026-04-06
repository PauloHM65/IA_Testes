"""Base para steps de retrieval."""

from abc import abstractmethod

from src.steps.base import BaseStep, PipelineData


class BaseRetriever(BaseStep):
    """Step que popula data.raw_chunks a partir de uma busca."""

    @abstractmethod
    def execute(self, data: PipelineData) -> PipelineData:
        ...