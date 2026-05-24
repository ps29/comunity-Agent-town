from src.agents.proposals import MoveProposal, SpeakProposal, UseObjectProposal, WaitProposal
from src.world.events import MoveEvent, ObjectStateChangeEvent, RejectedActionEvent, SpeechEvent


AFFORDANCE_STATE_MAP = {
    "brew_coffee": "brewing",
    "serve_coffee": "ready",
    "serve_pastry": "stocked",
    "check_pastries": "stocked",
    "sit": "occupied",
    "leave": "empty",
    "write": "in_use",
    "organize_notes": "organized",
    "read": "in_use",
    "browse": "in_use",
    "rest": "occupied",
    "observe": "observed",
}


class ActionResolver:
    def __init__(self, world, bus, event_repo, sim_tick: int, sim_time: str = ""):
        self.world = world
        self.bus = bus
        self.event_repo = event_repo
        self.sim_tick = sim_tick
        self.sim_time = sim_time

    async def resolve(self, proposal):
        if isinstance(proposal, MoveProposal):
            if proposal.target_location not in self.world.locations:
                return await self._reject(proposal.agent, "move", f"unknown location: {proposal.target_location}")
            from_loc = self.world.agent_location(proposal.agent)
            self.world.move_agent(proposal.agent, proposal.target_location)
            event = MoveEvent(proposal.agent, from_loc, proposal.target_location, proposal.target_location, self.sim_tick)
            self.bus.emit(event)
            if self.event_repo:
                await self.event_repo.append(event, self.sim_time)
            return [event]

        if isinstance(proposal, SpeakProposal):
            location = self.world.agent_location(proposal.agent)
            if proposal.target and not self.world.has_agent(proposal.target):
                return await self._reject(proposal.agent, "speak", f"unknown agent: {proposal.target}")
            if proposal.target and self.world.agent_location(proposal.target) != location:
                return await self._reject(proposal.agent, "speak", f"agent not nearby: {proposal.target}")
            message = proposal.message.strip()
            if not message:
                return await self._reject(proposal.agent, "speak", "empty message")
            event = SpeechEvent(proposal.agent, proposal.target, location, message, self.sim_tick)
            self.bus.emit(event)
            if self.event_repo:
                await self.event_repo.append(event, self.sim_time)
            return [event]

        if isinstance(proposal, UseObjectProposal):
            location = self.world.agent_location(proposal.agent)
            if proposal.object not in self.world.objects_at(location):
                return await self._reject(proposal.agent, "use_object", f"object not nearby: {proposal.object}")
            interaction = proposal.interaction.strip()
            if not self.world.is_valid_affordance(proposal.object, interaction):
                return await self._reject(
                    proposal.agent,
                    "use_object",
                    f"unsupported interaction for {proposal.object}: {interaction}",
                )
            old_state = self.world.get_object_state(proposal.object) or "unknown"
            new_state = self._state_after_interaction(proposal.object, interaction, old_state)
            if new_state == old_state:
                return []
            self.world.set_object_state(proposal.object, new_state)
            event = ObjectStateChangeEvent(proposal.object, old_state, new_state, location, self.sim_tick)
            self.bus.emit(event)
            if self.event_repo:
                await self.event_repo.append(event, self.sim_time)
            return [event]

        if isinstance(proposal, WaitProposal):
            return []

        return []

    async def _reject(self, agent: str, proposal_type: str, reason: str):
        location = self.world.agent_location(agent) if self.world.has_agent(agent) else ""
        event = RejectedActionEvent(agent, proposal_type, reason, location, self.sim_tick)
        self.bus.emit(event)
        if self.event_repo:
            await self.event_repo.append(event, self.sim_time)
        return [event]

    def _state_after_interaction(self, obj: str, interaction: str, old_state: str) -> str:
        candidate = AFFORDANCE_STATE_MAP.get(interaction, old_state)
        allowed = self.world.object_allowed_states(obj)
        if candidate in allowed:
            return candidate
        return old_state
