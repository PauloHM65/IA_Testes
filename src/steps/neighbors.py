"""Expande cada chunk com vizinhos adjacentes do Redis."""

from __future__ import annotations

from langchain_core.documents import Document

from src.steps import register
from src.steps.base import BaseStep, PipelineData


@register
class NeighborsStep(BaseStep):
    name = "neighbors"
    label = "Expandindo com vizinhos adjacentes"

    def execute(self, data: PipelineData) -> PipelineData:
        data.chunks = self._fetch_neighbors(data.raw_chunks)
        return data

    def _fetch_neighbors(self, chunks: list) -> list:
        window = self.ctx.config.neighbor_window
        map_key = self.ctx.config.chunks_map_key

        max_per_source = self._parse_max_indices(map_key)
        neighbor_keys = self._collect_neighbor_keys(chunks, window, max_per_source)
        neighbor_docs = self._fetch_from_redis(map_key, neighbor_keys)
        return self._assemble_expanded(chunks, window, max_per_source, neighbor_docs)

    def _parse_max_indices(self, map_key: str) -> dict[str, int]:
        """Descobre o indice maximo de cada source a partir das chaves no Redis."""
        all_keys = self.ctx.redis.hkeys(map_key)
        max_per_source: dict[str, int] = {}
        for raw_key in all_keys:
            key_str = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
            last_colon = key_str.rfind(":")
            if last_colon == -1:
                continue
            source = key_str[:last_colon]
            try:
                idx = int(key_str[last_colon + 1:])
            except ValueError:
                continue
            if source not in max_per_source or idx > max_per_source[source]:
                max_per_source[source] = idx
        return max_per_source

    def _collect_neighbor_keys(
        self, chunks: list, window: int, max_per_source: dict[str, int]
    ) -> list[tuple[str, int]]:
        """Coleta pares (source, idx) dos vizinhos que precisam ser buscados."""
        seen: set[tuple[str, int]] = set()
        keys: list[tuple[str, int]] = []

        for chunk in chunks:
            source = chunk.metadata.get("source", "desconhecido")
            idx = chunk.metadata.get("chunk_index")
            if idx is None:
                continue

            max_idx = max_per_source.get(source, idx)
            for offset in range(-window, window + 1):
                neighbor_idx = idx + offset
                if neighbor_idx < 0 or neighbor_idx > max_idx:
                    continue
                neighbor_key = (source, neighbor_idx)
                if neighbor_key in seen:
                    continue
                seen.add(neighbor_key)
                if offset != 0:
                    keys.append(neighbor_key)

        return keys

    def _fetch_from_redis(
        self, map_key: str, keys: list[tuple[str, int]]
    ) -> dict[tuple[str, int], Document]:
        """Busca conteudo dos vizinhos no Redis hash."""
        if not keys:
            return {}

        redis_keys = [f"{source}:{idx}" for source, idx in keys]
        values = self.ctx.redis.hmget(map_key, redis_keys)

        docs: dict[tuple[str, int], Document] = {}
        for (source, idx), content in zip(keys, values):
            if content:
                docs[(source, idx)] = Document(
                    page_content=content.decode("utf-8") if isinstance(content, bytes) else content,
                    metadata={"source": source, "chunk_index": idx},
                )
        return docs

    def _assemble_expanded(
        self, chunks: list, window: int,
        max_per_source: dict[str, int],
        neighbor_docs: dict[tuple[str, int], Document],
    ) -> list:
        """Monta lista final intercalando chunks originais com vizinhos."""
        expanded = []
        added: set[tuple[str, int]] = set()

        for chunk in chunks:
            source = chunk.metadata.get("source", "desconhecido")
            idx = chunk.metadata.get("chunk_index")
            if idx is None:
                expanded.append(chunk)
                continue

            max_idx = max_per_source.get(source, idx)
            for offset in range(-window, window + 1):
                neighbor_idx = idx + offset
                if neighbor_idx < 0 or neighbor_idx > max_idx:
                    continue
                add_key = (source, neighbor_idx)
                if add_key in added:
                    continue
                added.add(add_key)

                if offset == 0:
                    expanded.append(chunk)
                elif add_key in neighbor_docs:
                    expanded.append(neighbor_docs[add_key])

        expanded.sort(key=lambda d: (
            d.metadata.get("source", ""),
            d.metadata.get("chunk_index", 0),
        ))
        return expanded
