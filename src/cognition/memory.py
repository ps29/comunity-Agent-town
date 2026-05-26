import math


class MemoryService:
    def __init__(self, memory_repo, embeddings_module, semantic_repo=None):
        self.repo = memory_repo
        self.embed = embeddings_module
        self.semantic_repo = semantic_repo
        self.embedding_blob_fallbacks = 0

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
        if hasattr(self.repo, "get_candidates"):
            memories = await self.repo.get_candidates(agent_id, max_rows=recent_n + important_n)
        elif hasattr(self.repo, "get_retrieval_candidates"):
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
            relevance = self.embed.cosine_similarity(query_emb, self._memory_embedding(mem, query_emb))
            metadata_bonus = self._metadata_bonus(mem, metadata_boosts or {})
            score = 0.35 * recency + 0.25 * importance + 0.3 * relevance + 0.1 * metadata_bonus
            scored.append((score, mem))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def _memory_embedding(self, memory: dict, query_emb) -> object:
        try:
            embedding = self.embed.blob_to_embedding(memory.get("embedding") or b"")
            if getattr(embedding, "ndim", 1) != 1 or len(embedding) == 0 or len(embedding) != len(query_emb):
                raise ValueError("stored embedding has an unexpected shape")
            return embedding
        except Exception:
            self.embedding_blob_fallbacks += 1
            return self.embed.embed(str(memory.get("content", "")))

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

    async def add_semantic(
        self,
        agent_id: int,
        subject: str,
        fact: str,
        confidence: float,
        source_memory_ids: list[int],
        sim_tick: int,
    ) -> int | None:
        if self.semantic_repo is None:
            return None
        existing = await self.semantic_repo.find_similar(agent_id, subject, fact)
        if existing:
            await self.semantic_repo.reinforce(existing["id"], sim_tick, delta=0.15)
            return int(existing["id"])
        return await self.semantic_repo.add(
            agent_id=agent_id,
            subject=subject,
            fact=fact,
            confidence=confidence,
            source_memory_ids=source_memory_ids,
            sim_tick=sim_tick,
        )

    async def get_semantic(self, agent_id: int, min_confidence: float = 0.0) -> list[dict]:
        if self.semantic_repo is None:
            return []
        return await self.semantic_repo.get_for_agent(agent_id, min_confidence)

    async def get_semantic_about(self, agent_id: int, subject: str, n: int = 10) -> list[dict]:
        if self.semantic_repo is None:
            return []
        return await self.semantic_repo.get_about_subject(agent_id, subject, n)

    async def retrieve_about(self, agent_id: int, subject: str, top_k: int = 5) -> list[dict]:
        if hasattr(self.repo, "retrieve_about"):
            return await self.repo.retrieve_about(agent_id, subject, n=top_k)
        memories = await self.repo.get_all_for_agent(agent_id)
        subject_lower = subject.lower()
        matches = [m for m in memories if subject_lower in str(m.get("content", "")).lower()]
        matches.sort(key=lambda m: (int(m.get("importance", 0)), int(m.get("sim_tick", 0))), reverse=True)
        return matches[:top_k]

    async def recent_dialogue_with(self, agent_id: int, target_name: str, n: int = 3) -> list[str]:
        if hasattr(self.repo, "recent_speech_to"):
            return await self.repo.recent_speech_to(agent_id, target_name, n=n)
        return []

    async def get_today(self, agent_id: int, current_tick: int, window: int = 48) -> list[dict]:
        if hasattr(self.repo, "get_today"):
            return await self.repo.get_today(agent_id, current_tick, window)
        memories = await self.repo.get_recent(agent_id, n=window)
        cutoff = max(0, current_tick - window)
        return [m for m in memories if int(m.get("sim_tick", 0)) > cutoff]

    async def mark_consolidated(self, memory_ids: list[int]) -> None:
        if hasattr(self.repo, "mark_consolidated"):
            await self.repo.mark_consolidated(memory_ids)

    def embedding_diagnostics(self) -> dict:
        diagnostics = {}
        if hasattr(self.embed, "diagnostics"):
            diagnostics = self.embed.diagnostics()
        return {
            "embedding_blob_fallbacks": self.embedding_blob_fallbacks,
            "hash_embedding_fallback_used": bool(diagnostics.get("hash_fallback_used", False)),
            "hash_embedding_fallback_reason": diagnostics.get("hash_fallback_reason"),
        }
