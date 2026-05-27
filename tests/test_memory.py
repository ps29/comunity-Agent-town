import numpy as np
import pytest

from src.cognition import embeddings
from src.cognition.memory import MemoryService
from src.storage.db import get_connection, init_db
from src.storage.repositories import AgentRepository, MemoryRepository, SemanticMemoryRepository


class TinyEmbeddings:
    @staticmethod
    def embed(text):
        vector = np.zeros(3, dtype=np.float32)
        if "coffee" in text.lower():
            vector[0] = 1
        if "novel" in text.lower():
            vector[1] = 1
        if vector.sum() == 0:
            vector[2] = 1
        return vector

    @staticmethod
    def cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    @staticmethod
    def embedding_to_blob(emb):
        return emb.astype(np.float32).tobytes()

    @staticmethod
    def blob_to_embedding(blob):
        return np.frombuffer(blob, dtype=np.float32)


@pytest.mark.asyncio
async def test_memory_retrieval_prefers_relevant_important_recent(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    await init_db(str(db_path))
    conn = await get_connection(str(db_path))
    try:
        await AgentRepository(conn).create("Maria", {"name": "Maria"}, "cafe")
        service = MemoryService(MemoryRepository(conn), TinyEmbeddings)
        for idx in range(20):
            await service.add(1, "observation", f"old neutral memory {idx}", 2, idx, "08:00")
        await service.add(
            1,
            "observation",
            "Maria remembers John's coffee order.",
            9,
            25,
            "08:12",
            metadata={"location": "cafe", "agents": ["John"]},
        )

        results = await service.retrieve(1, "coffee at the cafe", 26, top_k=3, metadata_boosts={"location": "cafe", "agents": "John"})
        assert results[0]["content"] == "Maria remembers John's coffee order."
    finally:
        await conn.close()


class CandidateRepo:
    def __init__(self):
        self.used_candidates = False

    async def get_retrieval_candidates(self, agent_id, recent_n=80, important_n=80):
        self.used_candidates = True
        emb = TinyEmbeddings.embedding_to_blob(TinyEmbeddings.embed("coffee"))
        return [
            {
                "id": 1,
                "agent_id": agent_id,
                "sim_tick": 10,
                "kind": "observation",
                "content": "Coffee memory",
                "importance": 8,
                "embedding": emb,
                "metadata_json": "{}",
            }
        ]

    async def get_all_for_agent(self, agent_id):
        raise AssertionError("retrieve should use bounded candidates")


@pytest.mark.asyncio
async def test_memory_retrieval_uses_bounded_candidates():
    repo = CandidateRepo()
    service = MemoryService(repo, TinyEmbeddings)
    results = await service.retrieve(1, "coffee", 11)
    assert repo.used_candidates
    assert results[0]["content"] == "Coffee memory"


class InvalidEmbeddingRepo:
    async def get_retrieval_candidates(self, agent_id, recent_n=80, important_n=80):
        return [
            {
                "id": 1,
                "agent_id": agent_id,
                "sim_tick": 10,
                "kind": "observation",
                "content": "Coffee memory",
                "importance": 8,
                "embedding": b"",
                "metadata_json": "{}",
            }
        ]


@pytest.mark.asyncio
async def test_memory_retrieval_falls_back_when_stored_embedding_is_invalid():
    service = MemoryService(InvalidEmbeddingRepo(), TinyEmbeddings)
    results = await service.retrieve(1, "coffee", 11)
    assert results[0]["content"] == "Coffee memory"
    assert service.embedding_blob_fallbacks == 1


def test_embedding_hash_fallback_is_visible(monkeypatch, caplog):
    def fail_model_load():
        raise RuntimeError("model cache missing")

    monkeypatch.setattr(embeddings, "get_model", fail_model_load)
    monkeypatch.setattr(embeddings, "_hash_fallback_used", False)
    monkeypatch.setattr(embeddings, "_hash_fallback_reason", None)
    monkeypatch.setattr(embeddings, "_hash_fallback_warned", False)

    with caplog.at_level("WARNING"):
        vector = embeddings.embed("coffee memory")

    assert vector.shape == (384,)
    assert embeddings.diagnostics()["hash_fallback_used"] is True
    assert "model cache missing" in embeddings.diagnostics()["hash_fallback_reason"]
    assert "Falling back to deterministic hash embeddings" in caplog.text


@pytest.mark.asyncio
async def test_semantic_memory_can_be_added_and_retrieved(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    await init_db(str(db_path))
    conn = await get_connection(str(db_path))
    try:
        agent_id = await AgentRepository(conn).create("Maria", {"name": "Maria"}, "cafe")
        service = MemoryService(MemoryRepository(conn), TinyEmbeddings, semantic_repo=SemanticMemoryRepository(conn))

        fact_id = await service.add_semantic(
            agent_id,
            "John",
            "John prefers thoughtful questions about his novel.",
            0.7,
            [1, 2],
            48,
        )

        facts = await service.get_semantic_about(agent_id, "John")
        assert fact_id == facts[0]["id"]
        assert facts[0]["fact"] == "John prefers thoughtful questions about his novel."
    finally:
        await conn.close()
