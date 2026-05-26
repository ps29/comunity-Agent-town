import argparse
import asyncio
import os
import random
import subprocess
from pathlib import Path

import yaml

from src.agents.agent import Agent
from src.agents.files import AgentFiles
from src.agents.personality import initial_state_from_bio
from src.cognition import embeddings
from src.cognition.memory import MemoryService
from src.engine.simulation import SimulationEngine
from src.llm.benchmark import (
    DEFAULT_ALIAS,
    DEFAULT_LLAMA_DIR,
    DEFAULT_MODEL,
    require_gpu_offload,
    start_server,
    wait_for_server,
)
from src.llm.client import LlamaCppClient
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
    SemanticMemoryRepository,
)
from src.world.events import EventBus
from src.world.state import WorldState


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", default=None)
    parser.add_argument("--events", default="events.jsonl")
    parser.add_argument("--transcript", default="simulation.log")
    parser.add_argument("--agents-root", default="agents")
    parser.add_argument("--start-llama-server", action="store_true")
    parser.add_argument("--llama-dir", type=Path, default=DEFAULT_LLAMA_DIR)
    parser.add_argument("--llama-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--llama-alias", default=DEFAULT_ALIAS)
    parser.add_argument("--llama-ctx-size", type=int, default=8192)
    parser.add_argument("--llama-server-log", type=Path, default=Path(".tmp/llama_cpp_main_server.log"))
    args = parser.parse_args()
    random.seed(args.seed)

    sim_cfg = yaml.safe_load(Path("config/simulation.yaml").read_text(encoding="utf-8"))
    agents_cfg = yaml.safe_load(Path("config/agents.yaml").read_text(encoding="utf-8"))
    world_cfg = yaml.safe_load(Path("config/world.yaml").read_text(encoding="utf-8"))

    server_process: subprocess.Popen | None = None
    if args.start_llama_server:
        server_env = os.environ.copy()
        server_env["GGML_VK_VISIBLE_DEVICES"] = "0"
        args.llama_server_log.parent.mkdir(parents=True, exist_ok=True)
        server_process = start_server(
            args.llama_dir,
            args.llama_model,
            args.llama_alias,
            args.llama_server_log,
            server_env,
            ctx_size=args.llama_ctx_size,
        )
        log_text = wait_for_server(args.llama_server_log)
        require_gpu_offload(log_text)
        os.environ["LLAMA_CPP_URL"] = "http://127.0.0.1:8080/v1"
        os.environ["PRIMARY_MODEL"] = args.llama_alias
        os.environ["CHEAP_MODEL"] = args.llama_alias
        print(f"llama.cpp server ready on GPU. Log: {args.llama_server_log}")

    db_path = args.db or os.environ.get("DATABASE_URL") or "simulation.sqlite3"
    await init_db(db_path)
    conn = await get_connection(db_path)
    try:
        event_log = EventLog(args.events)
        base_url = os.environ.get("LLAMA_CPP_URL") or sim_cfg["llama_cpp_url"]
        primary_model = os.environ.get("PRIMARY_MODEL") or sim_cfg["primary_model"]
        cheap_model = os.environ.get("CHEAP_MODEL") or sim_cfg["cheap_model"]
        primary = LlamaCppClient(base_url, primary_model)
        cheap = LlamaCppClient(base_url, cheap_model)
        gateway = LLMGateway(primary, cheap, event_log)

        memory_repo = MemoryRepository(conn)
        semantic_repo = SemanticMemoryRepository(conn)
        agent_repo = AgentRepository(conn)
        event_repo = EventRepository(conn)
        relationship_repo = RelationshipRepository(conn)
        plan_repo = PlanRepository(conn)
        memory = MemoryService(memory_repo, embeddings, semantic_repo=semantic_repo)
        world = WorldState.from_config(world_cfg)
        bus = EventBus()
        transcript = Transcript(args.transcript)
        engine = SimulationEngine(sim_cfg, gateway, memory, world, bus, transcript, event_repo)
        agent_files = AgentFiles(args.agents_root)

        agent_ids_by_name = {}
        for bio in agents_cfg["agents"]:
            agent_id = await agent_repo.upsert(
                bio["name"],
                bio,
                bio.get("start_location", "cafe"),
                state=initial_state_from_bio(bio),
                needs=bio.get("needs", {}),
            )
            bio["id"] = agent_id
            agent_ids_by_name[bio["name"]] = agent_id
            agent_files.ensure(bio)
        event_repo.set_agent_ids_by_name(agent_ids_by_name)

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
                    agent_repo=agent_repo,
                    agent_ids_by_name=agent_ids_by_name,
                    agent_files=agent_files,
                )
            )

        await engine.run(args.ticks)
        print(f"\nSimulation complete. See {args.transcript} and {args.events}")
    finally:
        await conn.close()
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_process.kill()


if __name__ == "__main__":
    asyncio.run(main())
