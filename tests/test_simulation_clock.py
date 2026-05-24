from src.engine.simulation import SimulationEngine


def test_first_tick_starts_at_0800_and_48_ticks_cover_day():
    engine = SimulationEngine({"tick_duration_sim_seconds": 1800}, None, None, None, None, None, None)
    engine.sim_tick = 1
    assert engine._format_time() == "08:00"
    engine.sim_tick = 48
    assert engine._format_time() == "07:30"
