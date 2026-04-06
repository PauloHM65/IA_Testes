"""Expande cada chunk com vizinhos adjacentes do Redis."""

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
        redis = self.ctx.redis

        all_keys = redis.hkeys(map_key)
        max_index_per_source: dict[str, int] = {}
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
            if source not in max_index_per_source or idx > max_index_per_source[source]:
                max_index_per_source[source] = idx

        seen = set()
        keys_to_fetch: list[str] = []
        fetch_meta: list[tuple[str, int]] = []

        for chunk in chunks:
            source = chunk.metadata.get("source", "desconhecido")
            idx = chunk.metadata.get("chunk_index")
            if idx is None:
                key = (source, id(chunk))
                if key not in seen:
                    seen.add(key)
                continue

            max_idx = max_index_per_source.get(source, idx)
            for offset in range(-window, window + 1):
                neighbor_idx = idx + offset
                if neighbor_idx < 0 or neighbor_idx > max_idx:
                    continue
                neighbor_key = (source, neighbor_idx)
                if neighbor_key in seen:
                    continue
                seen.add(neighbor_key)

                if offset != 0:
                    redis_key = f"{source}:{neighbor_idx}"
                    keys_to_fetch.append(redis_key)
                    fetch_meta.append((source, neighbor_idx))

        neighbor_docs: dict[tuple[str, int], Document] = {}
        if keys_to_fetch:
            values = redis.hmget(map_key, keys_to_fetch)
            for (source, idx), content in zip(fetch_meta, values):
                if content:
                    neighbor_docs[(source, idx)] = Document(
                        page_content=content.decode("utf-8"),
                        metadata={"source": source, "chunk_index": idx},
                    )

        expanded = []
        added = set()
        for chunk in chunks:
            source = chunk.metadata.get("source", "desconhecido")
            idx = chunk.metadata.get("chunk_index")
            if idx is None:
                expanded.append(chunk)
                continue

            max_idx = max_index_per_source.get(source, idx)
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
