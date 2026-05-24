import pytest

from src.agents.agent import Agent
from src.agents.files import AgentFiles
from src.agents.proposals import SpeakProposal
from src.agents.proposals import UseObjectProposal
from src.agents.proposals import MoveProposal
from src.cognition.memory import MemoryService
from src.llm.gateway import CallKind
from src.storage.db import get_connection, init_db
from src.storage.repositories import AgentRepository, MemoryRepository
from src.world.events import EventBus
from src.world.state import WorldState
from tests.test_memory import TinyEmbeddings


class FakeGateway:
    def __init__(self):
        self.calls = []

    async def call(self, kind, system, user, agent_name):
        self.calls.append((kind, system, user, agent_name))
        if kind == CallKind.PERCEIVE:
            return {"observations": [f"{agent_name} notices John nearby."]}
        if kind == CallKind.SCORE_IMPORTANCE:
            return {"importance": 5}
        if kind == CallKind.PLAN:
            return {"schedule": {"hour_08": "Talk with John at the cafe."}}
        if kind == CallKind.ACT:
            return {"action": "speak_to", "target": "John", "message": "Good morning, John.", "interaction": "", "reasoning": "John is nearby."}
        if kind == CallKind.REFLECT:
            return {"insights": ["Emma likes calm chats."], "knowledge": ["Emma likes blueberry muffins."]}
        return {}


class FakeTranscript:
    def __init__(self):
        self.lines = []

    def log(self, agent, kind, content):
        self.lines.append((agent, kind, content))


class FakeRelationshipRepo:
    async def notes_for_agent(self, source_agent_id, target_agent_names_by_id):
        return ["John: familiarity 0.8, trust 0.4, affinity 0.5; Shared a quiet coffee."]


@pytest.mark.asyncio
async def test_single_agent_tick_persists_memory_and_proposes_action(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    await init_db(str(db_path))
    conn = await get_connection(str(db_path))
    try:
        await AgentRepository(conn).create("Maria", {"name": "Maria"}, "cafe")
        memory = MemoryService(MemoryRepository(conn), TinyEmbeddings)
        world = WorldState({"cafe": {"objects": ["coffee_maker"]}}, {"coffee_maker": {"state": "ready", "location": "cafe"}})
        world.place_agent("Maria", "cafe")
        world.place_agent("John", "cafe")
        bio = {"id": 1, "name": "Maria", "age": 28, "job": "cafe owner", "personality": "Warm", "goals": ["Welcome people"]}
        agent = Agent(
            bio,
            memory,
            FakeGateway(),
            world,
            EventBus(),
            FakeTranscript(),
            relationship_repo=FakeRelationshipRepo(),
            agent_ids_by_name={"Maria": 1, "John": 2},
        )

        await agent.perceive(1, "08:00")
        await agent.maybe_plan(1, "08:00")
        proposal = await agent.propose_action(1, "08:00")

        assert isinstance(proposal, SpeakProposal)
        assert proposal.target == "John"
        assert "Shared a quiet coffee" in bio["character_capsule"]
        assert (await memory.repo.get_recent(1, 1))[0]["kind"] in {"plan", "observation"}
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_agent_files_are_loaded_and_heartbeats_write_today_and_knowledge(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    await init_db(str(db_path))
    conn = await get_connection(str(db_path))
    try:
        await AgentRepository(conn).create("Maria", {"name": "Maria"}, "cafe")
        memory = MemoryService(MemoryRepository(conn), TinyEmbeddings)
        world = WorldState({"cafe": {"objects": []}}, {})
        world.place_agent("Maria", "cafe")
        bio = {"id": 1, "name": "Maria", "age": 28, "job": "cafe owner", "personality": "Warm", "goals": ["Welcome people"]}
        files = AgentFiles(tmp_path / "agents")
        files.ensure(bio)
        (tmp_path / "agents" / "maria" / "SOUL.md").write_text("Avoid saying divine.", encoding="utf-8")
        gateway = FakeGateway()
        agent = Agent(bio, memory, gateway, world, EventBus(), FakeTranscript(), agent_files=files)

        await agent.daily_heartbeat(1, "08:00")
        assert "hour_08: Talk with John at the cafe." in files.load_context(bio)["today"]
        assert any("Avoid saying divine." in system for _, system, _, _ in gateway.calls)

        await agent.end_of_day_heartbeat(29, "22:00")
        assert "Emma likes blueberry muffins." in files.load_context(bio)["knowledge"]
    finally:
        await conn.close()


def test_normalize_observation_dict_booleans():
    bio = {"id": 1, "name": "Maria", "age": 28, "job": "cafe owner", "personality": "Warm", "goals": []}
    world = WorldState({"cafe": {"objects": []}}, {})
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())
    assert agent._normalize_observation({"desk_is_clear": True, "ignored": False}) == "desk is clear"


def test_agent_normalizes_common_object_action_near_misses():
    bio = {"id": 1, "name": "Maria", "age": 28, "job": "cafe owner", "personality": "Warm", "goals": []}
    world = WorldState(
        {"cafe": {"objects": ["coffee_maker"]}},
        {"coffee_maker": {"state": "ready", "location": "cafe", "affordances": ["brew_coffee"], "allowed_states": ["ready", "brewing"]}},
    )
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())
    proposal = agent._proposal_from_dict(
        {"action": "use_object", "target": "coffee maker", "interaction": "Press the brew button."},
        objects_here=["coffee_maker"],
    )
    assert isinstance(proposal, UseObjectProposal)
    assert proposal.object == "coffee_maker"
    assert proposal.interaction == "brew_coffee"


def test_agent_redirects_non_nearby_object_use_to_object_location():
    bio = {"id": 1, "name": "John", "age": 32, "job": "novelist", "personality": "Quiet", "goals": []}
    world = WorldState(
        {"cafe": {"objects": ["coffee_maker"]}, "study_room": {"objects": ["desk"]}},
        {
            "coffee_maker": {"state": "ready", "location": "cafe", "affordances": ["brew_coffee"], "allowed_states": ["ready", "brewing"]},
            "desk": {"state": "empty", "location": "study_room", "affordances": ["write"], "allowed_states": ["empty", "in_use"]},
        },
    )
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())
    proposal = agent._proposal_from_dict(
        {"action": "use_object", "target": "coffee maker", "interaction": "make coffee"},
        objects_here=["desk"],
        location="study_room",
    )
    assert isinstance(proposal, MoveProposal)
    assert proposal.target_location == "cafe"


def test_agent_sanitizes_afternoon_greeting_and_avoids_repeated_message():
    bio = {"id": 1, "name": "Emma", "age": 26, "job": "student", "personality": "Curious", "goals": []}
    world = WorldState(
        {"cafe": {"objects": ["corner_table"]}},
        {"corner_table": {"state": "empty", "location": "cafe", "affordances": ["sit"], "allowed_states": ["empty", "occupied"]}},
    )
    world.place_agent("Emma", "cafe")
    world.place_agent("John", "cafe")
    world.place_agent("Maria", "cafe")
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())
    first = agent._proposal_from_dict(
        {"action": "speak_to", "target": "John", "message": "Good morning, John. How is your day going?"},
        agents_here=["John"],
        objects_here=["corner_table"],
        location="cafe",
        sim_time="14:00",
    )
    assert first.message.startswith("Good afternoon")
    addressed = agent._sanitize_message_for_target("Maria, did you see this?", "John")
    assert addressed.startswith("John,")
    agent._remember_message(first.message)
    repeated = agent._avoid_repetitive_speech(first, ["corner_table"], "cafe", "")
    assert isinstance(repeated, UseObjectProposal)
    assert repeated.object == "corner_table"
    assert repeated.interaction == "sit"


def test_agent_deterministic_perception_uses_world_state_only():
    bio = {"id": 1, "name": "Maria", "age": 28, "job": "cafe owner", "personality": "Warm", "goals": []}
    world = WorldState(
        {"cafe": {"objects": ["coffee_maker"]}},
        {"coffee_maker": {"state": "ready", "location": "cafe"}},
    )
    world.place_agent("Maria", "cafe")
    world.place_agent("Emma", "cafe")
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())
    observations = agent._deterministic_observations("cafe", ["Emma"], ["coffee_maker"], [])
    assert observations[0] == "Maria is at the cafe with Emma; nearby objects include coffee_maker."
    assert "coffee_maker is ready" in observations[-1]


def test_agent_move_to_nearby_object_becomes_object_use():
    bio = {"id": 1, "name": "Emma", "age": 26, "job": "student", "personality": "Curious", "goals": []}
    world = WorldState(
        {"cafe": {"objects": ["corner_table"]}},
        {"corner_table": {"state": "empty", "location": "cafe", "affordances": ["sit"], "allowed_states": ["empty", "occupied"]}},
    )
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())
    proposal = agent._proposal_from_dict(
        {"action": "move_to", "target": "corner table"},
        objects_here=["corner_table"],
        location="cafe",
    )
    assert isinstance(proposal, UseObjectProposal)
    assert proposal.object == "corner_table"
    assert proposal.interaction == "sit"
