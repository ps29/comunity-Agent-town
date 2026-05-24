import math


class MemoryService:
    def __init__(self, memory_repo, embeddings_module):
        self.repo = memory_repo
        self.embed = embeddings_module

    async def add(
        self,
        agent_id: int,
        kind: str,
        content: str,
        importance: int,
        sim_tick: int,
        sim_time: str,
        metadata: dict | None = None,
    ) -> int:
        emb = self.embed.embed(content)
        return await self.repo.add(
            agent_id=agent_id,
            kind=kind,
            content=content,
            importance=max(1, min(10, int(importance))),
            embedding=self.embed.embedding_to_blob(emb),
            sim_tick=sim_tick,
            sim_time=sim_time,
            metadata=metadata,
        )

    async def retrieve(
        self,
        agent_id: int,
        query: str,
        current_tick: int,
        top_k: int = 10,
        metadata_boosts: dict[str, str] | None = None,
        recent_n: int = 80,
        important_n: int = 80,
    ) -> list[dict]:
        if hasattr(self.repo, "get_retrieval_candidates"):
            memories = await self.repo.get_retrieval_candidates(agent_id, recent_n, important_n)
        else:
            memories = await self.repo.get_all_for_agent(agent_id)
        if not memories:
            return []
        query_emb = self.embed.embed(query)
        scored = []
        for mem in memories:
            ticks_ago = max(0, current_tick - int(mem["sim_tick"]))
            recency = math.exp(-0.05 * ticks_ago)
            importance = int(mem["importance"]) / 10.0
            relevance = self.embed.cosine_similarity(query_emb, self.embed.blob_to_embedding(mem["embedding"]))
            metadata_bonus = self._metadata_bonus(mem, metadata_boosts or {})
            score = 0.35 * recency + 0.25 * importance + 0.3 * relevance + 0.1 * metadata_bonus
            scored.append((score, mem))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def _metadata_bonus(self, memory: dict, metadata_boosts: dict[str, str]) -> float:
        if not metadata_boosts:
            return 0.0
        import json

        raw = memory.get("metadata_json") or "{}"
        try:
            metadata = raw if isinstance(raw, dict) else json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return 0.0
        hits = 0
        for key, expected in metadata_boosts.items():
            value = metadata.get(key)
            if isinstance(value, list) and expected in value:
                hits += 1
            elif value == expected:
                hits += 1
        return min(1.0, hits / max(1, len(metadata_boosts)))
