SYSTEM_TEMPLATE = """\
You are {name}, a {age}-year-old {job} in a small village.

Personality: {personality}

Character capsule:
{character_capsule}

Live agent files:
{file_context}

You summarize only concrete things you can observe now from the listed agents, objects, and recent events. Do not invent weather, NPCs, hidden objects, or unseen actions. Return final JSON immediately. Do not think step by step. /no_think

Example:
{{"observations":["Maria notices John entering the cafe and looking tired."]}}
"""

USER_TEMPLATE = """\
Current time: {sim_time}
Current location: {location}
Other agents here: {agents_here}
Objects here: {objects_here}
Recent events here:
{events}

List 1 to 3 concise observations as strings. Respond with JSON only.
"""


def build(agent_bio: dict, location: str, agents_here: list[str], objects_here: list[str], recent_events: list, sim_time: str):
    values = dict(agent_bio)
    values["character_capsule"] = agent_bio.get("character_capsule", "No additional character details.")
    values["file_context"] = agent_bio.get("file_context", "No live agent files loaded.")
    system = SYSTEM_TEMPLATE.format(**values)
    event_lines = []
    for event in recent_events:
        if hasattr(event, "content"):
            event_lines.append(f"- {event.speaker} said: {event.content}")
        elif hasattr(event, "from_location"):
            event_lines.append(f"- {event.agent} moved from {event.from_location} to {event.to_location}")
        elif hasattr(event, "reason"):
            event_lines.append(f"- {event.agent}'s {event.proposal_type} action was rejected: {event.reason}")
        else:
            event_lines.append(f"- {event}")
    user = USER_TEMPLATE.format(
        sim_time=sim_time,
        location=location,
        agents_here=", ".join(agents_here) or "no one",
        objects_here=", ".join(objects_here) or "none",
        events="\n".join(event_lines) or "- none",
    )
    return system, user
