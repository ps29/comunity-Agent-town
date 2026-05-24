SYSTEM_TEMPLATE = """\
You are {name}. Reflect on recent memories and infer useful high-level insights. Do not think step by step. /no_think

Character capsule:
{character_capsule}
Live agent files:
{file_context}
Respond with valid JSON.

Example:
{{"insights":["John seems more open to conversation when people ask about his writing routine."],"knowledge":["John values thoughtful questions about his writing routine."]}}
"""

USER_TEMPLATE = """\
Recent memories:
{memories}

Generate 1 to 3 insights and 0 to 3 concise durable knowledge facts. JSON only.
"""


def build(agent_bio: dict, recent_memories: list[dict]):
    memories = "\n".join(f"- {m['content']}" for m in recent_memories) or "- no memories"
    values = dict(agent_bio)
    values["character_capsule"] = agent_bio.get("character_capsule", "No additional character details.")
    values["file_context"] = agent_bio.get("file_context", "No live agent files loaded.")
    return SYSTEM_TEMPLATE.format(**values), USER_TEMPLATE.format(memories=memories)
