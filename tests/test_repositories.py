import pytest

from src.storage.db import get_connection, init_db
from src.storage.repositories import AgentRepository, EventRepository, MemoryRepository, RelationshipRepository
from src.world.events import MoveEvent, ObjectStateChangeEvent, RejectedActionEvent, SpeechEvent


@pytest.mark.asyncio
async def test_repositories_create_and_retrieve(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    await init_db(str(db_path))
    conn = await get_connection(str(db_path))
    try:
        agents = AgentRepository(conn)
        memories = MemoryRepository(conn)
        events = EventRepository(conn)

        agent_id = await agents.create("Maria", {"name": "Maria"}, "cafe", state={"mood": "bright"})
        john_id = await agents.create("John", {"name": "John"}, "study_room")
        assert (await agents.get_state(agent_id))["mood"] == "bright"
        await agents.update_state(agent_id, {"mood": "focused"})
        assert (await agents.get_state(agent_id))["mood"] == "focused"

        await memories.add(agent_id, "observation", "Maria made coffee.", 4, b"1234", 1, "08:00", metadata={"location": "cafe"})
        await memories.add(agent_id, "reflection", "Coffee brings people together.", 8, b"5678", 2, "08:01")

        recent = await memories.get_recent(agent_id, 2)
        important = await memories.get_top_by_importance(agent_id, 1)
        assert [m["kind"] for m in recent] == ["reflection", "observation"]
        assert important[0]["content"] == "Coffee brings people together."

        await events.append(SpeechEvent("Maria", None, "cafe", "Good morning!", 2), "08:01")
        tick_events = await events.get_by_tick(2)
        assert tick_events[0]["kind"] == "speech"

        relationships = RelationshipRepository(conn)
        rel_id = await relationships.upsert(agent_id, john_id, affinity_delta=0.2, trust_delta=0.1, familiarity_delta=0.4, summary="Warm chat")
        await relationships.upsert(agent_id, john_id, familiarity_delta=0.1)
        rel = await relationships.get(agent_id, john_id)
        assert rel["id"] == rel_id
        assert rel["familiarity"] == 0.5
        assert "John:" in (await relationships.notes_for_agent(agent_id, {john_id: "John"}))[0]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_agent_upsert_updates_provided_runtime_fields_and_preserves_omitted_fields(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    await init_db(str(db_path))
    conn = await get_connection(str(db_path))
    try:
        agents = AgentRepository(conn)
        agent_id = await agents.upsert(
            "Maria",
            {"name": "Maria", "role": "barista"},
            "cafe",
            state={"mood": "bright"},
            needs={"hunger": 0.2},
        )

        await agents.upsert(
            "Maria",
            {"name": "Maria", "role": "host"},
            "library",
            state={"mood": "focused"},
            needs={"hunger": 0.7},
        )
        row = await agents.get_by_name("Maria")
        assert row["id"] == agent_id
        assert row["current_location"] == "library"
        assert await agents.get_state(agent_id) == {"mood": "focused"}
        assert await agents.get_needs(agent_id) == {"hunger": 0.7}

        await agents.upsert("Maria", {"name": "Maria", "role": "owner"})
        row = await agents.get_by_name("Maria")
        assert row["current_location"] == "library"
        assert await agents.get_state(agent_id) == {"mood": "focused"}
        assert await agents.get_needs(agent_id) == {"hunger": 0.7}
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_event_repository_attributes_agent_sources_when_mapping_is_available(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    await init_db(str(db_path))
    conn = await get_connection(str(db_path))
    try:
        events = EventRepository(conn, {"Maria": 1, "John": 2})
        await events.append(SpeechEvent("Maria", "John", "cafe", "Good morning.", 1), "08:00")
        await events.append(MoveEvent("John", "library", "cafe", "cafe", 1), "08:00")
        await events.append(RejectedActionEvent("Maria", "move", "unknown location", "cafe", 1), "08:00")
        await events.append(ObjectStateChangeEvent("coffee_maker", "ready", "brewing", "cafe", 1), "08:00")

        rows = await events.get_by_tick(1)
        assert [row["source_agent_id"] for row in rows] == [1, 2, 1, None]
    finally:
        await conn.close()
