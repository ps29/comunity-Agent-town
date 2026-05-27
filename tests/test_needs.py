from src.cognition.needs import AgentNeeds
from src.prompts import act


def test_needs_decay_and_status_string():
    needs = AgentNeeds(hunger=0.69, energy=0.31, social_satiety=0.21)
    needs.decay_tick()

    assert needs.hunger > 0.7
    assert needs.energy < 0.31
    assert "hungry" in needs.status_string()


def test_action_prompt_includes_needs():
    system, _ = act.build(
        {
            "name": "Maria",
            "age": 28,
            "job": "cafe owner",
            "personality": "Warm.",
            "goals": ["Welcome people"],
        },
        {
            "sim_time": "09:00",
            "location": "cafe",
            "agents_present": [],
            "objects_here": [],
            "action_menu": "Current location: cafe",
        },
        "Open the cafe",
        [],
        "hungry (72%)",
    )

    assert "Current needs:" in system
    assert "hungry (72%)" in system
