from dataclasses import dataclass


@dataclass
class SpeechEvent:
    speaker: str
    listener: str | None
    location: str
    content: str
    sim_tick: int


@dataclass
class MoveEvent:
    agent: str
    from_location: str
    to_location: str
    location: str
    sim_tick: int


@dataclass
class ObjectStateChangeEvent:
    object: str
    old_state: str
    new_state: str
    location: str
    sim_tick: int


@dataclass
class RejectedActionEvent:
    agent: str
    proposal_type: str
    reason: str
    location: str
    sim_tick: int


Event = SpeechEvent | MoveEvent | ObjectStateChangeEvent | RejectedActionEvent


class EventBus:
    def __init__(self):
        self._events_by_tick: dict[int, list[Event]] = {}

    def emit(self, event: Event) -> None:
        self._events_by_tick.setdefault(event.sim_tick, []).append(event)

    def events_at_location_for_tick(self, location: str, tick: int) -> list[Event]:
        return [
            event
            for event in self._events_by_tick.get(tick, [])
            if getattr(event, "location", None) == location
        ]
