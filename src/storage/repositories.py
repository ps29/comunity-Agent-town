import json
from dataclasses import asdict, is_dataclass
from typing import Any


def _row(row: Any) -> dict | None:
    return dict(row) if row is not None else None


def _json_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    return json.loads(value)


class MemoryRepository:
    def __init__(self, conn):
        self.conn = conn

    async def add(
        self,
        agent_id: int,
        kind: str,
        content: str,
        importance: int,
        embedding: bytes | None,
        sim_tick: int,
        sim_time: str,
        metadata: dict | None = None,
    ) -> int:
        cur = await self.conn.execute(
            """
            INSERT INTO memories
            (agent_id, sim_tick, sim_time, kind, content, importance, embedding, metadata_json, last_accessed_tick)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (agent_id, sim_tick, sim_time, kind, content, importance, embedding, json.dumps(metadata or {}), sim_tick),
        )
        await self.conn.commit()
        return int(cur.lastrowid)

    async def get_recent(self, agent_id: int, n: int = 10) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? ORDER BY sim_tick DESC, id DESC LIMIT ?",
            (agent_id, n),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_top_by_importance(self, agent_id: int, n: int = 10) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? ORDER BY importance DESC, sim_tick DESC LIMIT ?",
            (agent_id, n),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_all_for_agent(self, agent_id: int) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? ORDER BY sim_tick DESC, id DESC",
            (agent_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_retrieval_candidates(
        self,
        agent_id: int,
        recent_n: int = 80,
        important_n: int = 80,
    ) -> list[dict]:
        recent = await self.get_recent(agent_id, recent_n)
        important = await self.get_top_by_importance(agent_id, important_n)
        by_id = {}
        for memory in recent + important:
            by_id[memory["id"]] = memory
        return sorted(by_id.values(), key=lambda row: (row["sim_tick"], row["id"]), reverse=True)

    async def get_by_kind(self, agent_id: int, kind: str, n: int = 10) -> list[dict]:
        cur = await self.conn.execute(
            """
            SELECT * FROM memories
            WHERE agent_id = ? AND kind = ?
            ORDER BY sim_tick DESC, id DESC LIMIT ?
            """,
            (agent_id, kind, n),
        )
        return [dict(r) for r in await cur.fetchall()]


class AgentRepository:
    def __init__(self, conn):
        self.conn = conn

    async def create(
        self,
        name: str,
        bio: dict,
        current_location: str | None = None,
        state: dict | None = None,
    ) -> int:
        cur = await self.conn.execute(
            "INSERT INTO agents (name, bio_json, state_json, current_location) VALUES (?, ?, ?, ?)",
            (name, json.dumps(bio), json.dumps(state or {}), current_location),
        )
        await self.conn.commit()
        return int(cur.lastrowid)

    async def upsert(
        self,
        name: str,
        bio: dict,
        current_location: str | None = None,
        state: dict | None = None,
    ) -> int:
        await self.conn.execute(
            """
            INSERT INTO agents (name, bio_json, state_json, current_location)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET bio_json = excluded.bio_json
            """,
            (name, json.dumps(bio), json.dumps(state or {}), current_location),
        )
        await self.conn.commit()
        row = await self.get_by_name(name)
        if row is None:
            raise RuntimeError(f"Agent upsert failed for {name}")
        return int(row["id"])

    async def get_by_name(self, name: str) -> dict | None:
        cur = await self.conn.execute("SELECT * FROM agents WHERE name = ?", (name,))
        return _row(await cur.fetchone())

    async def update_location(self, agent_id: int, location: str) -> None:
        await self.conn.execute("UPDATE agents SET current_location = ? WHERE id = ?", (location, agent_id))
        await self.conn.commit()

    async def get_state(self, agent_id: int) -> dict:
        cur = await self.conn.execute("SELECT state_json FROM agents WHERE id = ?", (agent_id,))
        row = await cur.fetchone()
        if row is None or not row["state_json"]:
            return {}
        return json.loads(row["state_json"])

    async def update_state(self, agent_id: int, state: dict) -> None:
        await self.conn.execute("UPDATE agents SET state_json = ? WHERE id = ?", (json.dumps(state), agent_id))
        await self.conn.commit()


class EventRepository:
    def __init__(self, conn):
        self.conn = conn

    async def append(self, event, sim_time: str = "") -> int:
        payload = asdict(event) if is_dataclass(event) else dict(event)
        kind = event.__class__.__name__.replace("Event", "").lower()
        source_name = payload.get("speaker") or payload.get("agent")
        cur = await self.conn.execute(
            """
            INSERT INTO world_events
            (sim_tick, sim_time, kind, payload_json, source_agent_id, location)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(payload.get("sim_tick", 0)),
                sim_time,
                kind,
                json.dumps(payload),
                None if source_name is None else None,
                payload.get("location") or payload.get("to_location"),
            ),
        )
        await self.conn.commit()
        return int(cur.lastrowid)

    async def get_by_tick(self, tick: int) -> list[dict]:
        cur = await self.conn.execute("SELECT * FROM world_events WHERE sim_tick = ? ORDER BY id", (tick,))
        return [dict(r) for r in await cur.fetchall()]


class RelationshipRepository:
    def __init__(self, conn):
        self.conn = conn

    async def upsert(
        self,
        source_agent_id: int,
        target_agent_id: int,
        affinity_delta: float = 0.0,
        trust_delta: float = 0.0,
        familiarity_delta: float = 0.0,
        tension_delta: float = 0.0,
        summary: str = "",
        metadata: dict | None = None,
    ) -> int:
        existing = await self.get(source_agent_id, target_agent_id)
        if existing:
            values = {
                "affinity": float(existing["affinity"]) + affinity_delta,
                "trust": float(existing["trust"]) + trust_delta,
                "familiarity": float(existing["familiarity"]) + familiarity_delta,
                "tension": float(existing["tension"]) + tension_delta,
                "summary": summary or existing["summary"],
                    "metadata": {**_json_dict(existing["metadata_json"]), **(metadata or {})},
            }
            await self.conn.execute(
                """
                UPDATE relationships
                SET affinity = ?, trust = ?, familiarity = ?, tension = ?,
                    summary = ?, metadata_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    values["affinity"],
                    values["trust"],
                    values["familiarity"],
                    values["tension"],
                    values["summary"],
                    json.dumps(values["metadata"]),
                    existing["id"],
                ),
            )
            await self.conn.commit()
            return int(existing["id"])

        cur = await self.conn.execute(
            """
            INSERT INTO relationships
            (source_agent_id, target_agent_id, affinity, trust, familiarity, tension, summary, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_agent_id,
                target_agent_id,
                affinity_delta,
                trust_delta,
                familiarity_delta,
                tension_delta,
                summary,
                json.dumps(metadata or {}),
            ),
        )
        await self.conn.commit()
        return int(cur.lastrowid)

    async def get(self, source_agent_id: int, target_agent_id: int) -> dict | None:
        cur = await self.conn.execute(
            "SELECT * FROM relationships WHERE source_agent_id = ? AND target_agent_id = ?",
            (source_agent_id, target_agent_id),
        )
        return _row(await cur.fetchone())

    async def notes_for_agent(self, source_agent_id: int, target_agent_names_by_id: dict[int, str]) -> list[str]:
        cur = await self.conn.execute(
            "SELECT * FROM relationships WHERE source_agent_id = ? ORDER BY familiarity DESC, trust DESC",
            (source_agent_id,),
        )
        rows = [dict(r) for r in await cur.fetchall()]
        notes = []
        for row in rows:
            target = target_agent_names_by_id.get(int(row["target_agent_id"]), f"agent {row['target_agent_id']}")
            summary = row["summary"] or "no strong history yet"
            notes.append(
                f"{target}: familiarity {row['familiarity']:.1f}, trust {row['trust']:.1f}, affinity {row['affinity']:.1f}; {summary}"
            )
        return notes


class PlanRepository:
    def __init__(self, conn):
        self.conn = conn

    async def add(self, agent_id: int, plan: dict, sim_tick: int, sim_time: str, status: str = "active") -> int:
        cur = await self.conn.execute(
            "INSERT INTO plans (agent_id, sim_tick, sim_time, status, plan_json) VALUES (?, ?, ?, ?, ?)",
            (agent_id, sim_tick, sim_time, status, json.dumps(plan)),
        )
        await self.conn.commit()
        return int(cur.lastrowid)


class ScheduleRepository:
    def __init__(self, conn):
        self.conn = conn

    async def add(self, agent_id: int, name: str, schedule: dict, metadata: dict | None = None) -> int:
        cur = await self.conn.execute(
            "INSERT INTO schedules (agent_id, name, schedule_json, metadata_json) VALUES (?, ?, ?, ?)",
            (agent_id, name, json.dumps(schedule), json.dumps(metadata or {})),
        )
        await self.conn.commit()
        return int(cur.lastrowid)


class QuestRepository:
    def __init__(self, conn):
        self.conn = conn

    async def add(self, agent_id: int, title: str, details: dict | None = None, status: str = "active") -> int:
        cur = await self.conn.execute(
            "INSERT INTO quests (agent_id, title, status, details_json) VALUES (?, ?, ?, ?)",
            (agent_id, title, status, json.dumps(details or {})),
        )
        await self.conn.commit()
        return int(cur.lastrowid)
