from pathlib import Path

import yaml

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


def test_configured_town_world_has_broader_grounded_mystery_locations():
    config = yaml.safe_load(Path("config/world.yaml").read_text(encoding="utf-8"))
    world = WorldState.from_config(config)
    world.place_agent("Emma", "archive_room")

    expected_locations = {
        "cafe",
        "town_square",
        "library",
        "archive_room",
        "riverside_path",
        "old_mill",
        "market_stalls",
        "community_hall",
        "newspaper_office",
        "park",
    }
    assert expected_locations <= set(world.locations)

    menu = build_action_menu(world, "Emma")
    rendered = format_action_menu(menu)
    assert "archive_boxes state=untouched affordances=[search_records, organize_notes, observe]" in rendered
    assert "map_table state=empty affordances=[study_map, organize_notes, observe]" in rendered
    assert "old_mill" in menu["move_targets"]
