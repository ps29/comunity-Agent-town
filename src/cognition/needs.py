from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentNeeds:
    hunger: float = 0.2
    energy: float = 0.8
    social_satiety: float = 0.5
    hunger_threshold: float = 0.7
    tiredness_threshold: float = 0.3
    loneliness_threshold: float = 0.2

    def decay_tick(self) -> None:
        self.hunger = min(1.0, self.hunger + 0.02)
        self.energy = max(0.0, self.energy - 0.01)
        self.social_satiety = max(0.0, self.social_satiety - 0.005)

    def satisfy_hunger(self, amount: float = 0.3) -> None:
        self.hunger = max(0.0, self.hunger - amount)

    def rest(self, amount: float = 0.5) -> None:
        self.energy = min(1.0, self.energy + amount)

    def socialize(self, amount: float = 0.2) -> None:
        self.social_satiety = min(1.0, self.social_satiety + amount)

    def status_string(self) -> str:
        parts = []
        if self.hunger > self.hunger_threshold:
            parts.append(f"hungry ({self.hunger:.0%})")
        if self.energy < self.tiredness_threshold:
            parts.append(f"tired ({self.energy:.0%} energy)")
        if self.social_satiety < self.loneliness_threshold:
            parts.append(f"lonely ({self.social_satiety:.0%} social)")
        return "; ".join(parts) if parts else "content"

    def to_dict(self) -> dict:
        return {
            "hunger": self.hunger,
            "energy": self.energy,
            "social_satiety": self.social_satiety,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "AgentNeeds":
        data = data or {}
        return cls(
            hunger=float(data.get("hunger", 0.2)),
            energy=float(data.get("energy", 0.8)),
            social_satiety=float(data.get("social_satiety", 0.5)),
        )
