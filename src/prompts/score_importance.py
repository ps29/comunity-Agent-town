SYSTEM_TEMPLATE = """\
You rate how important an observation is to {name}'s future behavior. Do not think step by step. /no_think
Use 1 for mundane details and 10 for life-changing events. Respond with valid JSON.
Live agent files:
{file_context}

Example:
{{"importance":4}}
"""

USER_TEMPLATE = """\
Observation: {observation}

Return JSON only with an integer importance from 1 to 10.
"""


def build(agent_bio: dict, observation: str):
    values = dict(agent_bio)
    values["file_context"] = agent_bio.get("file_context", "No live agent files loaded.")
    return SYSTEM_TEMPLATE.format(**values), USER_TEMPLATE.format(observation=observation)
