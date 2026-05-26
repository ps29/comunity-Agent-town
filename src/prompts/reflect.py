SYSTEM_TEMPLATE = """\
You are {name}. Reflect on recent memories and infer useful high-level insights. Stay evidence-bound. Do not think step by step. /no_think

Character capsule:
{character_capsule}
Live agent files:
{file_context}
Respond with valid JSON.

Rules:
- Only write an insight when at least two recent memories support it, or when one concrete event is clearly important.
- Do not turn ordinary repeated object states into mysteries.
- Do not invent historical facts, crimes, disappearances, accidents, dates, or solved clues.
- If memories show repetition or getting stuck, say that plainly and suggest a grounded next kind of action.
- Durable knowledge facts must be modest facts, not theories.

Example:
{{"insights":["John seems more open to conversation when people ask about his writing routine."],"knowledge":["John values thoughtful questions about his writing routine."]}}
"""

USER_TEMPLATE = """\
Recent memories:
{memories}

Generate 0 to 2 insights and 0 to 2 concise durable knowledge facts. JSON only.
"""


def build(agent_bio: dict, recent_memories: list[dict]):
    memories = "\n".join(f"- {m['content']}" for m in recent_memories) or "- no memories"
    values = dict(agent_bio)
    values["character_capsule"] = agent_bio.get("character_capsule", "No additional character details.")
    values["file_context"] = agent_bio.get("file_context", "No live agent files loaded.")
    return SYSTEM_TEMPLATE.format(**values), USER_TEMPLATE.format(memories=memories)
