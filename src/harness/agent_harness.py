from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from src.agents.agent import ACTION_PROMPT_CHAR_BUDGET, Agent
from src.agents.files import AgentFiles
from src.agents.personality import initial_state_from_bio
from src.cognition.memory import MemoryService
from src.engine.simulation import SimulationEngine
from src.llm.gateway import CallKind
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


DEFAULT_DB = ".tmp/harness.sqlite3"
DEFAULT_REPORT = ".tmp/harness_report.json"
DEFAULT_TRANSCRIPT = ".tmp/harness_simulation.log"
DEFAULT_EVENTS = ".tmp/harness_events.jsonl"
DEFAULT_AGENT_FILES = ".tmp/harness_agents"


class HarnessEmbeddings:
    @staticmethod
    def embed(text: str) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float32)
        buckets = {
            "coffee": 0,
            "cafe": 0,
            "write": 1,
            "novel": 1,
            "study": 1,
            "library": 2,
            "research": 2,
            "park": 3,
            "bench": 3,
            "talk": 4,
            "said": 4,
        }
        lowered = text.lower()
        for token, index in buckets.items():
            if token in lowered:
                vector[index] += 1
        if vector.sum() == 0:
            vector[7] = 1
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    @staticmethod
    def embedding_to_blob(emb: np.ndarray) -> bytes:
        return emb.astype(np.float32).tobytes()

    @staticmethod
    def blob_to_embedding(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)


class DeterministicGateway:
    def __init__(self):
        self.calls: list[tuple[CallKind, str]] = []
        self.action_counts: Counter[str] = Counter()
        self.prompt_chars: list[int] = []
        self.fallback_count = 0

    async def call(self, kind: CallKind, system: str, user: str, agent_name: str) -> dict:
        self.calls.append((kind, agent_name))
        self.prompt_chars.append(len(system) + len(user))
        if kind == CallKind.PLAN:
            return {"schedule": self._plan(agent_name)}
        if kind == CallKind.ACT:
            return self._action(agent_name, user)
        if kind == CallKind.REFLECT:
            return {
                "insights": [f"{agent_name} is staying grounded in the current village state."],
                "knowledge": [f"{agent_name} completed a deterministic harness reflection."],
            }
        if kind == CallKind.SCORE_IMPORTANCE:
            return {"importance": 5}
        if kind == CallKind.PERCEIVE:
            return {"observations": []}
        return {}

    def metrics(self) -> dict:
        return {
            "llm_calls": len(self.calls),
            "fallback_count": self.fallback_count,
            "max_prompt_chars": max(self.prompt_chars, default=0),
        }

    def _plan(self, agent_name: str) -> dict:
        plans = {
            "Maria": {"hour_08": "Prepare coffee at the cafe.", "hour_09": "Read the notice_board at the town_square."},
            "John": {"hour_08": "Write at the typewriter in the newspaper_office.", "hour_09": "Observe the river_marker on the riverside_path."},
            "Emma": {"hour_08": "Study at the library.", "hour_09": "Search archive_boxes in the archive_room."},
        }
        return plans.get(agent_name, {"hour_08": "Observe the village.", "hour_09": "Talk with a nearby person."})

    def _action(self, agent_name: str, user: str) -> dict:
        self.action_counts[agent_name] += 1
        nearby = self._nearby_agents(user)
        objects = self._objects_here(user)
        location = self._location(user)
        count = self.action_counts[agent_name]

        if count == 1 and objects:
            return self._use_first_known_object(objects)
        if nearby:
            target = nearby[0]
            return {
                "action": "speak_to",
                "target": target,
                "message": f"Good morning, {target}. I am checking in on today's plan.",
                "interaction": "",
                "reasoning": "A nearby person is available for a grounded conversation.",
            }
        planned = self._planned_destination(agent_name, location)
        if planned and planned != location:
            return {"action": "move_to", "target": planned, "message": "", "interaction": "", "reasoning": "Move toward the plan."}
        if objects:
            return self._use_first_known_object(objects)
        return {"action": "wait", "target": "", "message": "", "interaction": "", "reasoning": "No safe deterministic action."}

    def _nearby_agents(self, user: str) -> list[str]:
        marker = "Agents here with you:"
        if marker not in user:
            return []
        text = user.split(marker, 1)[1].split("\n", 1)[0].strip()
        if not text or text == "no one":
            return []
        return [name.strip() for name in text.split(",") if name.strip()]

    def _objects_here(self, user: str) -> list[str]:
        marker = "Objects here:"
        if marker not in user:
            return []
        text = user.split(marker, 1)[1].split("\n", 1)[0].strip()
        if not text or text == "none":
            return []
        return [item.strip() for item in text.split(",") if item.strip()]

    def _location(self, user: str) -> str:
        marker = "Your current location:"
        if marker not in user:
            return ""
        return user.split(marker, 1)[1].split("\n", 1)[0].strip()

    def _planned_destination(self, agent_name: str, location: str) -> str | None:
        destinations = {
            "Maria": "town_square",
            "John": "riverside_path",
            "Emma": "archive_room",
        }
        destination = destinations.get(agent_name)
        if destination and location != destination:
            return destination
        return None

    def _use_first_known_object(self, objects: list[str]) -> dict:
        affordance_by_object = {
            "coffee_maker": "brew_coffee",
            "pastry_case": "check_pastries",
            "corner_table": "sit",
            "desk": "write",
            "bookshelf": "read",
            "notice_board": "read_notice",
            "town_map": "study_map",
            "archive_boxes": "search_records",
            "map_table": "study_map",
            "old_ledger": "review_notes",
            "river_marker": "observe",
            "willow_bench": "sit",
            "old_mill_wheel": "inspect",
            "mill_door": "inspect",
            "grain_sacks": "inspect",
            "market_crates": "browse_market",
            "flower_stall": "browse_market",
            "meeting_table": "host_meeting",
            "event_calendar": "read_notice",
            "typewriter": "write_article",
            "clippings_wall": "review_notes",
            "reporter_notebook": "review_notes",
            "local_history_shelf": "search_records",
            "bench": "sit",
            "pond": "observe",
            "study_table": "write",
            "reading_chair": "sit",
        }
        target = objects[0]
        return {
            "action": "use_object",
            "target": target,
            "message": "",
            "interaction": affordance_by_object.get(target, "observe"),
            "reasoning": "Use a nearby object with a known safe affordance.",
        }


class HarnessTranscript(Transcript):
    pass


async def run_harness(
    ticks: int,
    seed: int,
    db_path: str,
    report_path: str,
    reject_threshold: int,
    keep_artifacts: bool = False,
) -> dict:
    random.seed(seed)
    paths = [Path(db_path), Path(report_path), Path(DEFAULT_TRANSCRIPT), Path(DEFAULT_EVENTS)]
    agent_files_root = Path(DEFAULT_AGENT_FILES)
    if not keep_artifacts:
        for path in paths:
            if path.exists():
                path.unlink()
        if agent_files_root.exists():
            for child in sorted(agent_files_root.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    await init_db(db_path)
    conn = await get_connection(db_path)
    try:
        sim_cfg = yaml.safe_load(Path("config/simulation.yaml").read_text(encoding="utf-8"))
        agents_cfg = yaml.safe_load(Path("config/agents.yaml").read_text(encoding="utf-8"))
        world_cfg = yaml.safe_load(Path("config/world.yaml").read_text(encoding="utf-8"))

        gateway = DeterministicGateway()
        memory_repo = MemoryRepository(conn)
        agent_repo = AgentRepository(conn)
        event_repo = EventRepository(conn)
        relationship_repo = RelationshipRepository(conn)
        plan_repo = PlanRepository(conn)
        memory = MemoryService(memory_repo, HarnessEmbeddings)
        world = WorldState.from_config(world_cfg)
        bus = EventBus()
        transcript = HarnessTranscript(DEFAULT_TRANSCRIPT, echo=False)
        engine = SimulationEngine(sim_cfg, gateway, memory, world, bus, transcript, event_repo)
        agent_files = AgentFiles(agent_files_root)

        agent_ids_by_name = {}
        for bio in agents_cfg["agents"]:
            bio = dict(bio)
            agent_id = await agent_repo.upsert(
                bio["name"],
                bio,
                bio.get("start_location", "cafe"),
                state=initial_state_from_bio(bio),
            )
            bio["id"] = agent_id
            agent_ids_by_name[bio["name"]] = agent_id
            agent_files.ensure(bio)
        event_repo.set_agent_ids_by_name(agent_ids_by_name)

        for bio in agents_cfg["agents"]:
            bio = dict(bio)
            bio["id"] = agent_ids_by_name[bio["name"]]
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

        await engine.run(ticks)
        runtime_metrics = {
            **gateway.metrics(),
            **memory.embedding_diagnostics(),
        }
        report = await evaluate_harness(conn, ticks, seed, db_path, reject_threshold, agent_files_root, runtime_metrics)
    finally:
        await conn.close()

    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


async def evaluate_harness(
    conn,
    ticks: int,
    seed: int,
    db_path: str,
    reject_threshold: int,
    agent_files_root: Path,
    llm_metrics: dict | None = None,
) -> dict:
    llm_metrics = {
        "llm_calls": 0,
        "fallback_count": 0,
        "max_prompt_chars": 0,
        "embedding_blob_fallbacks": 0,
        "hash_embedding_fallback_used": False,
        "hash_embedding_fallback_reason": None,
        **(llm_metrics or {}),
    }
    metrics = {
        "agents": await _scalar(conn, "SELECT COUNT(*) FROM agents"),
        "memories_total": await _scalar(conn, "SELECT COUNT(*) FROM memories"),
        "memories_by_kind": await _counts(conn, "SELECT kind, COUNT(*) AS count FROM memories GROUP BY kind"),
        "plans": await _scalar(conn, "SELECT COUNT(*) FROM plans"),
        "events_total": await _scalar(conn, "SELECT COUNT(*) FROM world_events"),
        "events_by_kind": await _counts(conn, "SELECT kind, COUNT(*) AS count FROM world_events GROUP BY kind"),
        "rejected_actions": await _scalar(conn, "SELECT COUNT(*) FROM world_events WHERE kind = 'rejectedaction'"),
        "rejected_reasons": await _rejected_reasons(conn),
        "activity_by_agent": await _activity_by_agent(conn),
        "event_diversity": await _scalar(conn, "SELECT COUNT(DISTINCT kind) FROM world_events"),
        "top_repeated_phrases": await _top_repeated_phrases(conn),
        **(llm_metrics or {}),
    }
    failures = []
    warnings = []

    if metrics["agents"] <= 0:
        failures.append("No agents were loaded.")
    if metrics["plans"] <= 0:
        failures.append("No plans were created.")
    if metrics["memories_total"] <= 0:
        failures.append("No memories were created.")
    if ticks > 1 and metrics["events_total"] <= 0:
        failures.append("No world events were created after a multi-tick run.")
    if metrics["rejected_actions"] > reject_threshold:
        failures.append(
            f"Rejected actions exceeded threshold: {metrics['rejected_actions']} > {reject_threshold}."
        )
    if metrics["max_prompt_chars"] > ACTION_PROMPT_CHAR_BUDGET:
        failures.append(
            f"Prompt size exceeded budget: {metrics['max_prompt_chars']} > {ACTION_PROMPT_CHAR_BUDGET}."
        )
    if ticks >= 8 and metrics["event_diversity"] < 2:
        failures.append("World event diversity is too low for a multi-tick harness run.")

    missing_files = await _missing_agent_files(conn, agent_files_root)
    if missing_files:
        failures.append("Required runtime agent files are missing: " + ", ".join(missing_files))

    if metrics["events_total"] == 0 and ticks <= 1:
        warnings.append("No world events were created in a one-tick run.")

    return {
        "passed": not failures,
        "ticks": ticks,
        "seed": seed,
        "db_path": db_path,
        "metrics": metrics,
        "failures": failures,
        "warnings": warnings,
    }


async def _scalar(conn, sql: str) -> int:
    cur = await conn.execute(sql)
    row = await cur.fetchone()
    return int(row[0] if row else 0)


async def _counts(conn, sql: str) -> dict[str, int]:
    cur = await conn.execute(sql)
    return {str(row["kind"]): int(row["count"]) for row in await cur.fetchall()}


async def _rejected_reasons(conn) -> list[str]:
    cur = await conn.execute("SELECT payload_json FROM world_events WHERE kind = 'rejectedaction' ORDER BY id")
    reasons = []
    for row in await cur.fetchall():
        payload = json.loads(row["payload_json"])
        reason = payload.get("reason")
        if reason:
            reasons.append(reason)
    return reasons


async def _activity_by_agent(conn) -> dict[str, dict[str, int]]:
    cur = await conn.execute("SELECT kind, payload_json FROM world_events ORDER BY id")
    activity: dict[str, Counter[str]] = defaultdict(Counter)
    for row in await cur.fetchall():
        payload = json.loads(row["payload_json"])
        agent = payload.get("speaker") or payload.get("agent")
        if agent:
            activity[agent][row["kind"]] += 1
    return {agent: dict(counts) for agent, counts in sorted(activity.items())}


async def _top_repeated_phrases(conn) -> list[dict]:
    cur = await conn.execute("SELECT payload_json FROM world_events WHERE kind = 'speech' ORDER BY id")
    counts: Counter[str] = Counter()
    for row in await cur.fetchall():
        payload = json.loads(row["payload_json"])
        content = str(payload.get("content", "")).lower()
        words = [word for word in content.replace("?", "").replace(".", "").split() if len(word) > 3]
        phrase = " ".join(words[:8])
        if phrase:
            counts[phrase] += 1
    return [{"phrase": phrase, "count": count} for phrase, count in counts.most_common(5) if count > 1]


async def _missing_agent_files(conn, root: Path) -> list[str]:
    cur = await conn.execute("SELECT name FROM agents ORDER BY name")
    missing = []
    files = AgentFiles(root)
    for row in await cur.fetchall():
        folder = files.folder(row["name"])
        for filename in ("SOUL.md", "KNOWLEDGE.md", "TODAY.md"):
            path = folder / filename
            if not path.exists():
                missing.append(str(path))
    return missing


def print_report(report: dict) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    metrics = report["metrics"]
    print(f"Harness {status}")
    print(f"Ticks: {report['ticks']}  Seed: {report['seed']}  DB: {report['db_path']}")
    print(
        "Agents: {agents}  Memories: {memories_total}  Plans: {plans}  Events: {events_total}  Rejected: {rejected_actions}".format(
            **metrics
        )
    )
    print(f"Memory kinds: {metrics['memories_by_kind']}")
    print(f"Event kinds: {metrics['events_by_kind']}")
    print(
        "LLM calls: {llm_calls}  Fallbacks: {fallback_count}  Max prompt chars: {max_prompt_chars}  Embedding blob fallbacks: {embedding_blob_fallbacks}  Hash embedding fallback: {hash_embedding_fallback_used}  Event diversity: {event_diversity}".format(
            **metrics
        )
    )
    if report["failures"]:
        print("Failures:")
        for failure in report["failures"]:
            print(f"- {failure}")
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")


def parse_args() -> argparse.Namespace:
    default_seed = 42
    cfg_path = Path("config/simulation.yaml")
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        default_seed = int(cfg.get("random_seed", default_seed))

    parser = argparse.ArgumentParser(description="Run the deterministic agent simulation harness.")
    parser.add_argument("--ticks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--reject-threshold", type=int, default=0)
    parser.add_argument("--keep-artifacts", action="store_true")
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    report = await run_harness(
        ticks=args.ticks,
        seed=args.seed,
        db_path=args.db,
        report_path=args.report,
        reject_threshold=args.reject_threshold,
        keep_artifacts=args.keep_artifacts,
    )
    print_report(report)
    return 0 if report["passed"] else 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
