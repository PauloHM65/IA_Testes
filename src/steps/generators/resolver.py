"""Resolver: combina exercicio + materia e gera resolucao/resposta."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.steps import register
from src.steps.base import PipelineData
from src.steps.generators.base import BaseGenerator


@register
class ResolverGenerator(BaseGenerator):
    name = "resolver"
    label = "Resolvendo passo a passo"

    def execute(self, data: PipelineData) -> PipelineData:
        materia_context = self.format_docs(data.materia_chunks)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.ctx.config.system_prompt),
            ("human", self.ctx.config.human_prompt),
        ])
        chain = prompt | self.ctx.llm | StrOutputParser()

        data.resposta = chain.invoke({
            "exercicio": data.exercicio_texto or "(Sem exercicio — pergunta teorica)",
            "context": materia_context,
            "question": data.pergunta,
        })

        # Junta chunks para logging
        data.chunks = data.exercicio_chunks + data.materia_chunks
        data.contexto = (
            f"=== EXERCICIO ===\n{data.exercicio_texto}\n\n"
            f"=== MATERIA ===\n{materia_context}"
        )
        data.rerank_count = len(data.chunks)

        return data
