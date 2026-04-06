"""Busca o exercicio completo: acha melhor chunk e pega N sequenciais."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_redis import RedisConfig, RedisVectorStore

from src.config import env
from src.embeddings import get_embeddings
from src.steps import register
from src.steps.base import BaseStep, PipelineData, PipelineContext

EXERCICIO_SEQUENCIAL_CHUNKS = 20


@register
class BuscarExercicioStep(BaseStep):
    name = "buscar_exercicio"
    label = "Buscando exercicio completo"

    def execute(self, data: PipelineData) -> PipelineData:
        # Se classificado como teoria e usuario nao selecionou fontes especificas, pula
        if data.exercicio_texto != "__EXERCICIO__" and not data.fontes_selecionadas:
            data.exercicio_texto = ""
            return data

        exercicios_vs = self._get_exercicios_vectorstore()
        if exercicios_vs is None:
            data.exercicio_texto = ""
            return data

        # Busca os chunks mais relevantes
        search_kwargs = {"k": 5}
        results = exercicios_vs.similarity_search(data.pergunta, **search_kwargs)

        # Filtra por fonte selecionada pelo usuario ou identificada pelo classificador
        doc_filter = (
            data.fontes_selecionadas
            if data.fontes_selecionadas
            else ([data.documento_filtro] if data.documento_filtro else [])
        )
        if doc_filter:
            results = [
                r for r in results
                if any(f in r.metadata.get("source", "") for f in doc_filter)
            ]

        if not results:
            data.exercicio_texto = ""
            return data

        best = results[0]
        source = best.metadata.get("source", "desconhecido")
        start_idx = best.metadata.get("chunk_index", 0)
        data.raw_count = 1

        # Pega os N chunks sequenciais a partir do melhor match via Redis hash
        map_key = self.ctx.config.exercicios_chunks_map_key
        keys = [f"{source}:{start_idx + i}" for i in range(EXERCICIO_SEQUENCIAL_CHUNKS)]
        values = self.ctx.redis.hmget(map_key, keys)

        exercicio_chunks = []
        for i, content in enumerate(values):
            if content is None:
                break
            exercicio_chunks.append(Document(
                page_content=content.decode("utf-8") if isinstance(content, bytes) else content,
                metadata={"source": source, "chunk_index": start_idx + i},
            ))

        data.exercicio_chunks = exercicio_chunks
        data.exercicio_texto = "\n\n".join(c.page_content for c in exercicio_chunks)

        return data

    def _get_exercicios_vectorstore(self) -> RedisVectorStore | None:
        try:
            embeddings = get_embeddings(self.ctx.config.embedding_model)
            return RedisVectorStore(
                embeddings=embeddings,
                config=RedisConfig(
                    index_name=self.ctx.config.exercicios_index_name,
                    redis_url=env.REDIS_URL,
                    from_existing=True,
                ),
            )
        except Exception:
            return None
