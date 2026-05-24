import asyncio

from src.agents.agent import Agent
from src.world.resolver import ActionResolver


class SimulationEngine:
    def __init__(self, config, gateway, memory, world, bus, transcript, event_repo):
        self.config = config
        self.gateway = gateway
        self.memory = memory
        self.world = world
        self.bus = bus
        self.transcript = transcript
        self.event_repo = event_repo
        self.agents: list[Agent] = []
        self.sim_tick = 0

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)

    async def tick(self) -> None:
        self.sim_tick += 1
        sim_time = self._format_time()
        self.transcript.section(f"TICK {self.sim_tick} ({sim_time})")
        for agent in self.agents:
            await agent.daily_heartbeat(self.sim_tick, sim_time)
        self.transcript.log("SYSTEM", "PHASE", "Perceiving surroundings")
        await asyncio.gather(*[agent.perceive(self.sim_tick, sim_time) for agent in self.agents])
        self.transcript.log("SYSTEM", "PHASE", "Checking reflections")
        for agent in self.agents:
            await agent.maybe_reflect(self.sim_tick, sim_time, self.config.get("reflection_threshold", 25))
        self.transcript.log("SYSTEM", "PHASE", "Planning")
        for agent in self.agents:
            self.transcript.log("SYSTEM", "WAIT", f"Planning for {agent.bio['name']}")
            await agent.maybe_plan(self.sim_tick, sim_time)
        for agent in self.agents:
            await agent.end_of_day_heartbeat(self.sim_tick, sim_time)
        self.transcript.log("SYSTEM", "PHASE", "Choosing actions")
        proposals = await asyncio.gather(*[agent.propose_action(self.sim_tick, sim_time) for agent in self.agents])
        self.transcript.log("SYSTEM", "PHASE", "Resolving actions")
        resolver = ActionResolver(self.world, self.bus, self.event_repo, self.sim_tick, sim_time)
        for proposal in proposals:
            events = await resolver.resolve(proposal)
            for event in events:
                self.transcript.log_event(event)

    async def run(self, num_ticks: int) -> None:
        for _ in range(num_ticks):
            await self.tick()

    def _format_time(self) -> str:
        tick_seconds = int(self.config.get("tick_duration_sim_seconds", 30))
        total_min = ((self.sim_tick - 1) * tick_seconds) // 60
        hour = (8 + total_min // 60) % 24
        minute = total_min % 60
        return f"{hour:02d}:{minute:02d}"
