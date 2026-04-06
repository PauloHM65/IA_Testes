"""Base para steps de pos-processamento."""

from abc import abstractmethod

from src.steps.base import BaseStep, PipelineData


class BasePostProcessor(BaseStep):
    """Step que transforma a resposta apos a geracao."""

    @abstractmethod
    def execute(self, data: PipelineData) -> PipelineData:
        ...
