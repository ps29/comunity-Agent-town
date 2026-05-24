import argparse
import asyncio
import os
import random
from pathlib import Path

import yaml

from src.agents.agent import Agent
from src.agents.files import AgentFiles
from src.agents.personality import initial_state_from_bio
from src.cognition import embeddings
from src.cognition.memory import MemoryService
from src.engine.simulation import SimulationEngine
from src.llm.client import LMStudioClient
from src.llm.gateway import LLMGateway
from src.observability.event_log import EventLog
from src.observability.transcript import Transcript
from src.storage.db import get_connection, init_db
from src.storage.repositories import (
    AgentRepository,
    EventRepository,
    MemoryRepository,
    PlanRepository,
    RelationshipRepository,
)
from src.world.events import EventBus
from src.world.state import WorldState


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    random.seed(args.seed)

    sim_cfg = yaml.safe_load(Path("config/simulation.yaml").read_text(encoding="utf-8"))
    agents_cfg = yaml.safe_load(Path("config/agents.yaml").read_text(encoding="utf-8"))
    world_cfg = yaml.safe_load(Path("config/world.yaml").read_text(encoding="utf-8"))

    db_path = args.db or os.environ.get("DATABASE_URL") or "simulation.sqlite3"
    await init_db(db_path)
    conn = await get_connection(db_path)
    try:
        event_log = EventLog("events.jsonl")
        base_url = sim_cfg.get("ollama_url") or sim_cfg["lm_studio_url"]
        primary = LMStudioClient(base_url, sim_cfg["primary_model"])
        cheap = LMStudioClient(base_url, sim_cfg["cheap_model"])
        gateway = LLMGateway(primary, cheap, event_log)

        memory_repo = MemoryRepository(conn)
        agent_repo = AgentRepository(conn)
        event_repo = EventRepository(conn)
        relationship_repo = RelationshipRepository(conn)
        plan_repo = PlanRepository(conn)
        memory = MemoryService(memory_repo, embeddings)
        world = WorldState.from_config(world_cfg)
        bus = EventBus()
        transcript = Transcript("simulation.log")
        engine = SimulationEngine(sim_cfg, gateway, memory, world, bus, transcript, event_repo)
        agent_files = AgentFiles("agents")

        agent_ids_by_name = {}
        for bio in agents_cfg["agents"]:
            agent_id = await agent_repo.upsert(
                bio["name"],
                bio,
                bio.get("start_location", "cafe"),
                state=initial_state_from_bio(bio),
            )
            bio["id"] = agent_id
            agent_ids_by_name[bio["name"]] = agent_id
            agent_files.ensure(bio)

        for bio in agents_cfg["agents"]:
            world.place_agent(bio["name"], bio.get("start_location", "cafe"))
            engine.add_agent(
                Agent(
                    bio,
                    memory,
                    gateway,
                    world,
                    bus,
                    transcript,
                    sim_cfg.get("reflection_threshold", 25),
                    relationship_repo=relationship_repo,
                    plan_repo=plan_repo,
                    agent_ids_by_name=agent_ids_by_name,
                    agent_files=agent_files,
                )
            )

        await engine.run(args.ticks)
        print("\nSimulation complete. See simulation.log and events.jsonl")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
