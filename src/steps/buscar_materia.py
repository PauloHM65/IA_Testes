"""Busca teoria relevante no indice de materias com rerank interno."""

from src.steps import register
from src.steps.base import BaseStep, PipelineData


@register
class BuscarMateriaStep(BaseStep):
    name = "buscar_materia"
    label = "Buscando teoria relevante"

    def execute(self, data: PipelineData) -> PipelineData:
        # Se tem exercicio, usa o conteudo dele como query (mais preciso)
        if data.exercicio_texto and data.exercicio_texto != "__EXERCICIO__":
            query = data.exercicio_texto[:500]
        else:
            query = data.pergunta

        materia_chunks = self.ctx.base_retriever.invoke(query)

        # Filtra por fontes selecionadas pelo usuario (se houver)
        if data.fontes_selecionadas:
            materia_chunks = [
                c for c in materia_chunks
                if any(sel in c.metadata.get("source", "") for sel in data.fontes_selecionadas)
            ]

        # Rerank interno
        if materia_chunks:
            pairs = [[query, doc.page_content] for doc in materia_chunks]
            scores = self.ctx.reranker.predict(pairs)
            ranked = sorted(zip(scores, materia_chunks), key=lambda x: x[0], reverse=True)
            data.materia_chunks = [doc for _, doc in ranked[:self.ctx.config.rerank_top_n]]
        else:
            data.materia_chunks = []

        return data
