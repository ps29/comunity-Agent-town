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
from src.world.events import EventBus, SpeechEvent
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


def test_agent_reads_clock_style_plan_keys():
    bio = {"id": 1, "name": "Maria", "age": 28, "job": "cafe owner", "personality": "Warm", "goals": []}
    world = WorldState({"cafe": {"objects": []}}, {})
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())
    agent.current_plan = {"08:00": "Open the cafe.", "hour_09": "Welcome guests."}
    assert agent._current_plan_chunk("08:00") == "Open the cafe."
    assert agent._current_plan_chunk("09:30") == "Welcome guests."


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


def test_agent_maps_notebook_to_nearby_writing_surface():
    bio = {"id": 1, "name": "John", "age": 32, "job": "novelist", "personality": "Quiet", "goals": []}
    world = WorldState(
        {"study_room": {"objects": ["desk"]}},
        {"desk": {"state": "empty", "location": "study_room", "affordances": ["write"], "allowed_states": ["empty", "in_use"]}},
    )
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())
    proposal = agent._proposal_from_dict(
        {"action": "use_object", "target": "notebook", "interaction": "I open the notebook and begin writing."},
        objects_here=["desk"],
        location="study_room",
    )
    assert isinstance(proposal, UseObjectProposal)
    assert proposal.object == "desk"
    assert proposal.interaction == "write"


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


def test_agent_fallback_uses_grounded_need_action_before_waiting():
    bio = {"id": 1, "name": "Maria", "age": 28, "job": "cafe owner", "personality": "Warm", "goals": []}
    world = WorldState(
        {"cafe": {"objects": ["pastry_case", "coffee_maker"]}},
        {
            "pastry_case": {"state": "stocked", "location": "cafe", "affordances": ["serve_pastry"], "allowed_states": ["stocked"]},
            "coffee_maker": {"state": "ready", "location": "cafe", "affordances": ["serve_coffee"], "allowed_states": ["ready"]},
        },
    )
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())
    agent.needs.hunger = 0.9
    proposal = agent._proposal_from_dict(
        {"action": "wait", "fallback": True},
        objects_here=["pastry_case", "coffee_maker"],
        location="cafe",
        sim_time="15:00",
    )
    assert isinstance(proposal, UseObjectProposal)
    assert proposal.object == "pastry_case"
    assert proposal.interaction == "serve_pastry"


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
    assert observations[0] == "I am at the cafe with Emma; nearby objects include coffee_maker."
    assert "coffee_maker is ready" in observations[-1]


def test_agent_remembers_incoming_speech_for_repetition_filter():
    bio = {"id": 1, "name": "Maria", "age": 28, "job": "cafe owner", "personality": "Warm", "goals": []}
    world = WorldState({"cafe": {"objects": ["corner_table"]}}, {"corner_table": {"state": "empty", "location": "cafe", "affordances": ["sit"], "allowed_states": ["empty", "occupied"]}})
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())
    event = SpeechEvent("Emma", "Maria", "cafe", "That espresso is fantastic. Do you remember Mrs. Higgins and the old mill?", 1)
    agent._deterministic_observations("cafe", ["Emma"], ["corner_table"], [event])
    repeated = SpeakProposal("Maria", "Emma", "That espresso is fantastic. Do you remember Mrs. Higgins and the old mill?")
    proposal = agent._avoid_repetitive_speech(repeated, ["corner_table"], "cafe", "")
    assert isinstance(proposal, UseObjectProposal)


def test_agent_filters_ungrounded_legacy_file_context():
    bio = {"id": 1, "name": "Maria", "age": 28, "job": "cafe owner", "personality": "Warm", "goals": []}
    world = WorldState({"cafe": {"objects": []}}, {})
    world.place_agent("Maria", "cafe")
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())

    grounded = agent._ground_agent_file_text(
        "# Knowledge\n"
        "- Emma likes coffee.\n"
        "- Silas told stories by the river.\n"
        "- Mr. Henderson owns the bakery near the village square.\n"
        "- The back room contains artisan sketches.\n"
    )

    assert "Emma likes coffee" in grounded
    assert "Silas" not in grounded
    assert "Henderson" not in grounded
    assert "artisan" not in grounded


def test_agent_compacts_action_memories_and_excludes_raw_plans():
    bio = {"id": 1, "name": "Emma", "age": 26, "job": "student", "personality": "Curious", "goals": []}
    world = WorldState({"cafe": {"objects": []}}, {})
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())
    memories = [
        {"kind": "plan", "content": '{"hour_08":"Visit cafe","hour_09":"Repeat"}'},
        {"kind": "dialogue", "content": "Maria said: " + "coffee " * 100},
        {"kind": "dialogue", "content": "Maria said: " + "coffee " * 100},
        {"kind": "reflection", "content": "Emma keeps returning to the cafe for research."},
        {"kind": "observation", "content": "The coffee_maker is ready."},
    ]
    compact = agent._memories_for_action_prompt(memories)
    assert len(compact) == 3
    assert all(memory["kind"] != "plan" for memory in compact)
    assert all(len(memory["content"]) <= 240 for memory in compact)


def test_agent_budgeted_action_prompt_stays_under_cap():
    bio = {"id": 1, "name": "Emma", "age": 26, "job": "student", "personality": "Curious", "goals": ["Research"]}
    world = WorldState({"cafe": {"objects": ["coffee_maker"]}}, {"coffee_maker": {"state": "ready", "location": "cafe"}})
    agent = Agent(bio, None, None, world, EventBus(), FakeTranscript())
    bio["character_capsule"] = "curious"
    bio["file_context"] = "Knowledge:\n" + ("very long context " * 1000)
    memories = [{"content": "memory " * 200, "kind": "observation"} for _ in range(12)]
    system, user, kept = agent._build_budgeted_action_prompt(
        {
            "sim_time": "10:00",
            "location": "cafe",
            "agents_present": [],
            "objects_here": ["coffee_maker"],
            "action_menu": "Valid use_object targets:\n- coffee_maker state=ready affordances=[observe]",
        },
        "Observe the cafe.",
        memories,
    )
    assert len(system) + len(user) <= 12000
    assert len(kept) < len(memories)


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
