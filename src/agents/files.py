from __future__ import annotations

import re
from pathlib import Path


class AgentFiles:
    def __init__(self, root: str | Path = "agents"):
        self.root = Path(root)

    def slug(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug or "agent"

    def folder(self, name: str) -> Path:
        return self.root / self.slug(name)

    def ensure(self, bio: dict) -> None:
        folder = self.folder(bio["name"])
        folder.mkdir(parents=True, exist_ok=True)
        soul = folder / "SOUL.md"
        knowledge = folder / "KNOWLEDGE.md"
        today = folder / "TODAY.md"
        if not soul.exists():
            soul.write_text(self._default_soul(bio), encoding="utf-8")
        if not knowledge.exists():
            knowledge.write_text(f"# Knowledge: {bio['name']}\n\n", encoding="utf-8")
        if not today.exists():
            today.write_text(f"# Today: {bio['name']}\n\n", encoding="utf-8")

    def load_context(self, bio: dict) -> dict[str, str]:
        self.ensure(bio)
        folder = self.folder(bio["name"])
        return {
            "soul": (folder / "SOUL.md").read_text(encoding="utf-8"),
            "knowledge": (folder / "KNOWLEDGE.md").read_text(encoding="utf-8"),
            "today": (folder / "TODAY.md").read_text(encoding="utf-8"),
        }

    def append_knowledge(self, bio: dict, lines: list[str], sim_time: str = "") -> None:
        clean = [line.strip() for line in lines if isinstance(line, str) and line.strip()]
        if not clean:
            return
        self.ensure(bio)
        path = self.folder(bio["name"]) / "KNOWLEDGE.md"
        prefix = f"\n## {sim_time}\n" if sim_time else "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(prefix)
            for line in clean:
                handle.write(f"- {line}\n")

    def write_today(self, bio: dict, plan: dict, sim_time: str = "") -> None:
        self.ensure(bio)
        path = self.folder(bio["name"]) / "TODAY.md"
        lines = [f"# Today: {bio['name']}", ""]
        if sim_time:
            lines.extend([f"Plan generated at {sim_time}.", ""])
        for key in sorted(plan):
            lines.append(f"- {key}: {plan[key]}")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def reset_today(self, bio: dict, sim_time: str = "") -> None:
        self.ensure(bio)
        text = f"# Today: {bio['name']}\n\n"
        if sim_time:
            text += f"Reset at {sim_time}.\n\n"
        (self.folder(bio["name"]) / "TODAY.md").write_text(text, encoding="utf-8")

    def _default_soul(self, bio: dict) -> str:
        goals = "\n".join(f"- {goal}" for goal in bio.get("goals", [])) or "- Live naturally in the village."
        return (
            f"# Soul: {bio['name']}\n\n"
            f"{bio.get('personality', '').strip()}\n\n"
            "## Guardrails\n"
            "- Stay grounded in the current world state.\n"
            "- Do not invent people, places, hidden rooms, or events that are not provided.\n"
            "- Avoid repeating the same phrasing or topic when a conversation has already covered it.\n\n"
            "## Goals\n"
            f"{goals}\n"
        )
