def build_action_menu(world, agent_name: str) -> dict:
    location = world.agent_location(agent_name)
    nearby_agents = [name for name in world.agents_at(location) if name != agent_name]
    return {
        "current_location": location,
        "move_targets": sorted(world.locations.keys()),
        "speak_targets": nearby_agents,
        "objects": world.objects_menu_at(location),
    }


def format_action_menu(menu: dict) -> str:
    lines = [
        f"Current location: {menu['current_location']}",
        "Valid move_to targets: " + (", ".join(menu["move_targets"]) or "none"),
        "Valid speak_to targets: " + (", ".join(menu["speak_targets"]) or "none"),
        "Valid use_object targets:",
    ]
    if not menu["objects"]:
        lines.append("- none")
    for obj in menu["objects"]:
        affordances = ", ".join(obj["affordances"]) or "none"
        lines.append(f"- {obj['name']} state={obj['state']} affordances=[{affordances}]")
    return "\n".join(lines)
