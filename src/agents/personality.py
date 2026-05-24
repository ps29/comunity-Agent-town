from dataclasses import dataclass, field
from typing import Any


@dataclass
class CharacterProfile:
    name: str
    age: int
    job: str
    personality: str
    goals: list[str] = field(default_factory=list)
    personality_traits: dict[str, Any] = field(default_factory=dict)
    habits: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    speech_style: str = ""
    values: list[str] = field(default_factory=list)
    quirks: list[str] = field(default_factory=list)
    emotional_state: dict[str, Any] = field(default_factory=dict)
    start_location: str = "cafe"

    @classmethod
    def from_bio(cls, bio: dict) -> "CharacterProfile":
        return cls(
            name=bio["name"],
            age=int(bio["age"]),
            job=bio["job"],
            personality=str(bio.get("personality", "")).strip(),
            goals=list(bio.get("goals", [])),
            personality_traits=dict(bio.get("personality_traits", {})),
            habits=list(bio.get("habits", [])),
            preferences=dict(bio.get("preferences", {})),
            speech_style=str(bio.get("speech_style", "")).strip(),
            values=list(bio.get("values", [])),
            quirks=list(bio.get("quirks", [])),
            emotional_state=dict(bio.get("emotional_state", {})),
            start_location=bio.get("start_location", "cafe"),
        )

    def to_prompt_capsule(self, relationship_notes: list[str] | None = None) -> str:
        parts = [
            f"Traits: {_format_mapping(self.personality_traits)}",
            f"Habits: {_format_list(self.habits, 3)}",
            f"Preferences: {_format_preferences(self.preferences)}",
            f"Speech style: {self.speech_style or 'natural and grounded'}",
            f"Values: {_format_list(self.values, 4)}",
            f"Quirks: {_format_list(self.quirks, 2)}",
            f"Current state: {_format_mapping(self.emotional_state)}",
        ]
        if relationship_notes:
            parts.append(f"Relationship notes: {_format_list(relationship_notes, 3)}")
        return "\n".join(parts)


def build_character_capsule(bio: dict, relationship_notes: list[str] | None = None) -> str:
    return CharacterProfile.from_bio(bio).to_prompt_capsule(relationship_notes)


def initial_state_from_bio(bio: dict) -> dict:
    profile = CharacterProfile.from_bio(bio)
    return {
        "emotional_state": profile.emotional_state,
        "habit_streaks": {habit: 0 for habit in profile.habits},
        "recent_concerns": [],
        "active_values": profile.values,
    }


def _format_mapping(values: dict[str, Any]) -> str:
    if not values:
        return "unspecified"
    return ", ".join(f"{key}={value}" for key, value in values.items())


def _format_list(values: list[Any], limit: int) -> str:
    if not values:
        return "none"
    return "; ".join(str(value) for value in values[:limit])


def _format_preferences(preferences: dict[str, Any]) -> str:
    if not preferences:
        return "unspecified"
    chunks = []
    for key, value in preferences.items():
        if isinstance(value, list):
            chunks.append(f"{key}: {', '.join(str(item) for item in value[:4])}")
        else:
            chunks.append(f"{key}: {value}")
    return "; ".join(chunks)
