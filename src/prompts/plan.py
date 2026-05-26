SYSTEM_TEMPLATE = """\
You are {name}, a {age}-year-old {job}.

Personality: {personality}

Character capsule:
{character_capsule}

Live agent files:
{file_context}

Goals:
{goals}

Known world:
{known_world}

Create a simple daily plan for today using only known locations, people, and objects from context. Include entries for hours 08 through 17. Make adjacent hours meaningfully different: a place, object, or social intention should usually change each hour. Use several different known locations across the day when it fits the character, and include concrete grounded activities with listed objects instead of clustering every hour around one place. Do not invent named customers, neighbors, officials, relatives, pets, NPCs, historical events, crimes, disappearances, accidents, dates, or solved clues; only Maria, John, and Emma are known people. Respond with valid JSON whose "schedule" maps hour keys to intentions. Do not think step by step. /no_think

Example:
{{"schedule":{{"hour_08":"Open the cafe and make coffee.","hour_09":"Read the town_square notice_board.","hour_10":"Visit the archive_room to search archive_boxes."}}}}
"""

USER_TEMPLATE = """\
Current time: {sim_time}
Recent reflections:
{reflections}

Create schedule entries for hours 08 through 17. JSON only.
"""


def build(agent_bio: dict, sim_time: str, recent_reflections: list[dict], world_context: dict | None = None):
    values = dict(agent_bio)
    values["goals"] = "\n".join(f"- {goal}" for goal in agent_bio.get("goals", []))
    values["character_capsule"] = agent_bio.get("character_capsule", "No additional character details.")
    values["file_context"] = agent_bio.get("file_context", "No live agent files loaded.")
    values["known_world"] = _format_known_world(world_context or {})
    system = SYSTEM_TEMPLATE.format(
        **values,
    )
    user = USER_TEMPLATE.format(
        sim_time=sim_time,
        reflections="\n".join(f"- {m['content']}" for m in recent_reflections) or "- none",
    )
    return system, user


def _format_known_world(world_context: dict) -> str:
    locations = ", ".join(world_context.get("locations", [])) or "unknown"
    location_descriptions = world_context.get("location_descriptions", {})
    if location_descriptions:
        locations = "\n".join(
            f"- {name}: {location_descriptions.get(name, '')}"
            for name in world_context.get("locations", [])
        )
    agents = ", ".join(world_context.get("agents", [])) or "unknown"
    objects = ", ".join(world_context.get("objects", [])) or "unknown"
    return (
        f"Locations:\n{locations}\n"
        f"Agents: {agents}\n"
        f"Objects: {objects}\n"
        "Do not plan around any person, place, object, pet, or NPC not listed here."
    )
