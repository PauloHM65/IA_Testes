"""Generator simples: format_docs -> answer_chain."""

from src.steps import register
from src.steps.base import PipelineData
from src.steps.generators.base import BaseGenerator


@register
class SimpleGenerator(BaseGenerator):
    name = "generate"
    label = "Gerando resposta com LLM"

    def execute(self, data: PipelineData) -> PipelineData:
        data.contexto = self.format_docs(data.chunks)
        data.resposta = self.ctx.answer_chain.invoke({
            "context": data.contexto,
            "question": data.pergunta,
        })
        return data
