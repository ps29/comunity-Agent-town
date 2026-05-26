from src.agents.proposals import MoveProposal, SpeakProposal, UseObjectProposal, WaitProposal
from src.engine.simulation import SimulationEngine


def test_first_tick_starts_at_0800_and_48_ticks_cover_day():
    engine = SimulationEngine({"tick_duration_sim_seconds": 1800}, None, None, None, None, None, None)
    engine.sim_tick = 1
    assert engine._format_time() == "08:00"
    engine.sim_tick = 48
    assert engine._format_time() == "07:30"


def test_resolution_order_keeps_perceived_speech_before_moves():
    proposals = [
        MoveProposal("John", "park"),
        WaitProposal("Maria", "pause"),
        SpeakProposal("Emma", "John", "Before you go."),
        UseObjectProposal("Maria", "coffee_maker", "serve_coffee"),
    ]
    ordered = sorted(proposals, key=SimulationEngine._resolution_priority)
    assert isinstance(ordered[0], SpeakProposal)
    assert isinstance(ordered[1], UseObjectProposal)
    assert isinstance(ordered[2], MoveProposal)
    assert isinstance(ordered[3], WaitProposal)
