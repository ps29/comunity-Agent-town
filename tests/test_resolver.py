import pytest

from src.agents.proposals import MoveProposal, SpeakProposal, UseObjectProposal
from src.world.events import EventBus, MoveEvent, RejectedActionEvent, SpeechEvent
from src.world.resolver import ActionResolver
from src.world.state import WorldState


class FakeEventRepo:
    def __init__(self):
        self.events = []

    async def append(self, event, sim_time=""):
        self.events.append((event, sim_time))


@pytest.mark.asyncio
async def test_resolver_moves_and_speaks():
    world = WorldState({"cafe": {"objects": []}, "park": {"objects": []}}, {})
    world.place_agent("Maria", "cafe")
    world.place_agent("John", "cafe")
    bus = EventBus()
    repo = FakeEventRepo()
    resolver = ActionResolver(world, bus, repo, 1, "08:00")

    move_events = await resolver.resolve(MoveProposal("Maria", "park"))
    assert isinstance(move_events[0], MoveEvent)
    assert world.agent_location("Maria") == "park"

    invalid_speech = await resolver.resolve(SpeakProposal("Maria", "John", "Hello"))
    assert isinstance(invalid_speech[0], RejectedActionEvent)
    invented_speech = await resolver.resolve(SpeakProposal("Maria", "Sarah", "Hello"))
    assert isinstance(invented_speech[0], RejectedActionEvent)

    await resolver.resolve(MoveProposal("John", "park"))
    speech_events = await resolver.resolve(SpeakProposal("Maria", "John", "Hello"))
    assert isinstance(speech_events[0], SpeechEvent)
    assert bus.events_at_location_for_tick("park", 1)[-1].content == "Hello"


@pytest.mark.asyncio
async def test_resolver_validates_object_affordances_and_compact_states():
    world = WorldState(
        {"cafe": {"objects": ["coffee_maker"]}},
        {"coffee_maker": {"state": "ready", "location": "cafe", "allowed_states": ["ready", "brewing"], "affordances": ["brew_coffee"]}},
    )
    world.place_agent("Maria", "cafe")
    bus = EventBus()
    repo = FakeEventRepo()
    resolver = ActionResolver(world, bus, repo, 1, "08:00")

    invalid = await resolver.resolve(UseObjectProposal("Maria", "coffee_maker", "make a delicious latte for the customer"))
    assert isinstance(invalid[0], RejectedActionEvent)
    assert world.get_object_state("coffee_maker") == "ready"

    valid = await resolver.resolve(UseObjectProposal("Maria", "coffee_maker", "brew_coffee"))
    assert valid[0].new_state == "brewing"
    assert world.get_object_state("coffee_maker") == "brewing"

    no_op = await resolver.resolve(UseObjectProposal("Maria", "coffee_maker", "brew_coffee"))
    assert no_op == []
