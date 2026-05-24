SYSTEM_TEMPLATE = """\
You are {name}. Write one short line of natural dialogue in your voice. Do not think step by step. /no_think

Character capsule:
{character_capsule}
Live agent files:
{file_context}
Respond with valid JSON.

Example:
{{"message":"Good morning, John. Did the quiet help your writing today?"}}
"""

USER_TEMPLATE = """\
Context: {context}
Relevant memories:
{memories}

Return one message as JSON only.
"""


def build(agent_bio: dict, context: str, memories: list[dict]):
    return (
        SYSTEM_TEMPLATE.format(
            **{
                **agent_bio,
                "character_capsule": agent_bio.get("character_capsule", "No additional character details."),
                "file_context": agent_bio.get("file_context", "No live agent files loaded."),
            }
        ),
        USER_TEMPLATE.format(
            context=context,
            memories="\n".join(f"- {m['content']}" for m in memories) or "- none",
        ),
    )
