from src.world.context import build_action_menu, format_action_menu
from src.world.state import WorldState


def test_grounded_action_menu_lists_only_real_nearby_options():
    world = WorldState(
        {"cafe": {"objects": ["coffee_maker"]}, "park": {"objects": []}},
        {"coffee_maker": {"state": "ready", "location": "cafe", "affordances": ["brew_coffee"], "allowed_states": ["ready", "brewing"]}},
    )
    world.place_agent("Maria", "cafe")
    world.place_agent("Emma", "cafe")
    menu = build_action_menu(world, "Maria")
    rendered = format_action_menu(menu)

    assert menu["speak_targets"] == ["Emma"]
    assert "coffee_maker state=ready affordances=[brew_coffee]" in rendered
    assert "Sarah" not in rendered
