from copy import deepcopy


class WorldState:
    def __init__(self, locations: dict, objects: dict):
        self.locations = deepcopy(locations)
        self.objects = deepcopy(objects)
        self.agent_positions: dict[str, str] = {}

    @classmethod
    def from_config(cls, config: dict) -> "WorldState":
        return cls(config.get("locations", {}), config.get("objects", {}))

    def place_agent(self, name: str, location: str) -> None:
        if location not in self.locations:
            raise ValueError(f"Unknown location: {location}")
        self.agent_positions[name] = location

    def agent_location(self, name: str) -> str:
        return self.agent_positions[name]

    def has_agent(self, name: str) -> bool:
        return name in self.agent_positions

    def agents_at(self, location: str) -> list[str]:
        return sorted([name for name, loc in self.agent_positions.items() if loc == location])

    def objects_at(self, location: str) -> list[str]:
        return list(self.locations.get(location, {}).get("objects", []))

    def object_info(self, name: str) -> dict:
        return deepcopy(self.objects.get(name, {}))

    def object_affordances(self, name: str) -> list[str]:
        obj = self.objects.get(name, {})
        return list(obj.get("affordances", []))

    def object_allowed_states(self, name: str) -> list[str]:
        obj = self.objects.get(name, {})
        allowed = obj.get("allowed_states")
        if allowed:
            return list(allowed)
        state = obj.get("state")
        return [] if state is None else [state]

    def objects_menu_at(self, location: str) -> list[dict]:
        menu = []
        for name in self.objects_at(location):
            obj = self.objects.get(name, {})
            menu.append(
                {
                    "name": name,
                    "state": obj.get("state", "unknown"),
                    "affordances": list(obj.get("affordances", [])),
                    "allowed_states": self.object_allowed_states(name),
                }
            )
        return menu

    def is_valid_affordance(self, name: str, interaction: str) -> bool:
        return interaction in self.object_affordances(name)

    def move_agent(self, name: str, location: str) -> None:
        if location not in self.locations:
            raise ValueError(f"Unknown location: {location}")
        self.agent_positions[name] = location

    def get_object_state(self, name: str) -> str | None:
        obj = self.objects.get(name)
        return None if obj is None else obj.get("state")

    def set_object_state(self, name: str, state: str) -> None:
        if name not in self.objects:
            raise ValueError(f"Unknown object: {name}")
        self.objects[name]["state"] = state
