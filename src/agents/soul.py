from __future__ import annotations

from pathlib import Path

from src.agents.files import AgentFiles


def ensure_soul(agent_name: str, bio: dict, root: str | Path = "agents") -> Path:
    files = AgentFiles(root)
    merged_bio = {**bio, "name": agent_name}
    files.ensure(merged_bio)
    return files.folder(agent_name) / "SOUL.md"


def load_soul(agent_name: str, bio: dict | None = None, root: str | Path = "agents") -> str:
    bio = {**(bio or {}), "name": agent_name}
    files = AgentFiles(root)
    files.ensure(bio)
    return (files.folder(agent_name) / "SOUL.md").read_text(encoding="utf-8")
