SYSTEM_TEMPLATE = """\
You are {name}, a {age}-year-old {job} living in a small village.

Personality: {personality}

Character capsule:
{character_capsule}

Live agent files:
{file_context}

Current needs:
{needs}

Your goals:
{goals_bulleted}

Available actions:
- move_to: target must be one listed in the grounded action menu
- speak_to: target must be one nearby agent from the grounded action menu
- use_object: target must be a nearby object, and interaction must exactly match one listed affordance
- wait: use only when there is no useful action

Prefer speaking when another agent is nearby and it fits your memories or plan.
Do not invent people, locations, objects, hidden rooms, NPCs, or object effects.
Use exact target names from the menu, including underscores such as coffee_maker.
Only describe your own action. Do not write another agent's response.
If the recent memories show the same topic has been discussed repeatedly, choose a new angle, a concrete action, movement, or wait.
Use time-appropriate greetings; after 12:00 do not say "Good morning" or "Morning."

Do not think step by step. Return final JSON immediately. /no_think

You MUST respond with valid JSON in this shape:
{{"action":"move_to|speak_to|use_object|wait","target":"string","message":"string","interaction":"string","reasoning":"one sentence"}}

Example:
{{"action":"speak_to","target":"John","message":"Good morning, John. Are you working on the novel today?","interaction":"","reasoning":"John is nearby and I remember he values thoughtful conversation."}}
"""

USER_TEMPLATE = """\
Current sim time: {sim_time}
Your current location: {location}
Agents here with you: {agents_present}
Objects here: {objects_here}
Grounded action menu:
{action_menu}
Your current plan says: {plan_chunk}

Relevant memories:
{memories_bulleted}

What do you do next? Respond with JSON only.
"""


def build(agent_bio: dict, world_context: dict, plan_chunk: str, memories: list[dict], needs: str = "content"):
    system = SYSTEM_TEMPLATE.format(
        name=agent_bio["name"],
        age=agent_bio["age"],
        job=agent_bio["job"],
        personality=agent_bio["personality"],
        character_capsule=agent_bio.get("character_capsule", "No additional character details."),
        file_context=agent_bio.get("file_context", "No live agent files loaded."),
        needs=needs,
        goals_bulleted="\n".join(f"- {g}" for g in agent_bio.get("goals", [])),
    )
    user = USER_TEMPLATE.format(
        sim_time=world_context["sim_time"],
        location=world_context["location"],
        agents_present=", ".join(world_context["agents_present"]) or "no one",
        objects_here=", ".join(world_context["objects_here"]) or "none",
        action_menu=world_context.get("action_menu", "No grounded action menu available."),
        plan_chunk=plan_chunk,
        memories_bulleted="\n".join(f"- {m['content']}" for m in memories) or "- no memories yet",
    )
    return system, user
