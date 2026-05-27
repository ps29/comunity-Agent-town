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
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def _json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


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

    async def get_candidates(self, agent_id: int, max_rows: int = 200) -> list[dict]:
        return await self.get_retrieval_candidates(agent_id, max_rows // 2 or 50, max_rows // 2 or 50)

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

    async def retrieve_about(self, agent_id: int, subject: str, n: int = 5) -> list[dict]:
        cur = await self.conn.execute(
            """
            SELECT * FROM memories
            WHERE agent_id = ? AND content LIKE ?
            ORDER BY importance DESC, sim_tick DESC LIMIT ?
            """,
            (agent_id, f"%{subject}%", n),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def recent_speech_to(self, agent_id: int, target_name: str, n: int = 3) -> list[str]:
        cur = await self.conn.execute(
            """
            SELECT content FROM memories
            WHERE agent_id = ? AND kind = 'dialogue' AND content LIKE ?
            ORDER BY sim_tick DESC, id DESC LIMIT ?
            """,
            (agent_id, f"%I said to {target_name}%", n),
        )
        return [row["content"] for row in await cur.fetchall()]

    async def get_today(self, agent_id: int, current_tick: int, window: int = 48) -> list[dict]:
        cutoff = max(0, current_tick - window)
        cur = await self.conn.execute(
            """
            SELECT * FROM memories
            WHERE agent_id = ? AND sim_tick > ? AND consolidated = 0
            ORDER BY sim_tick ASC, id ASC
            """,
            (agent_id, cutoff),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def mark_consolidated(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        await self.conn.execute(
            f"UPDATE memories SET consolidated = 1 WHERE id IN ({placeholders})",
            tuple(memory_ids),
        )
        await self.conn.commit()


class SemanticMemoryRepository:
    def __init__(self, conn):
        self.conn = conn

    async def add(
        self,
        agent_id: int,
        subject: str,
        fact: str,
        confidence: float,
        source_memory_ids: list[int],
        sim_tick: int,
    ) -> int:
        cur = await self.conn.execute(
            """
            INSERT INTO semantic_memory
            (agent_id, subject, fact, confidence, source_memory_ids, first_seen_tick, last_reinforced_tick)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                subject,
                fact,
                max(0.0, min(1.0, float(confidence))),
                json.dumps(source_memory_ids),
                sim_tick,
                sim_tick,
            ),
        )
        await self.conn.commit()
        return int(cur.lastrowid)

    async def reinforce(self, fact_id: int, sim_tick: int, delta: float = 0.1) -> None:
        cur = await self.conn.execute("SELECT confidence FROM semantic_memory WHERE id = ?", (fact_id,))
        row = await cur.fetchone()
        if row is None:
            return
        confidence = min(1.0, float(row["confidence"]) + delta)
        await self.conn.execute(
            "UPDATE semantic_memory SET confidence = ?, last_reinforced_tick = ? WHERE id = ?",
            (confidence, sim_tick, fact_id),
        )
        await self.conn.commit()

    async def get_for_agent(self, agent_id: int, min_confidence: float = 0.0) -> list[dict]:
        cur = await self.conn.execute(
            """
            SELECT * FROM semantic_memory
            WHERE agent_id = ? AND decayed = 0 AND confidence >= ?
            ORDER BY confidence DESC, last_reinforced_tick DESC
            """,
            (agent_id, min_confidence),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_about_subject(self, agent_id: int, subject: str, n: int = 10) -> list[dict]:
        cur = await self.conn.execute(
            """
            SELECT * FROM semantic_memory
            WHERE agent_id = ? AND subject = ? AND decayed = 0
            ORDER BY confidence DESC LIMIT ?
            """,
            (agent_id, subject, n),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def find_similar(self, agent_id: int, subject: str, fact: str) -> dict | None:
        cur = await self.conn.execute(
            """
            SELECT * FROM semantic_memory
            WHERE agent_id = ? AND subject = ? AND decayed = 0
            ORDER BY confidence DESC, last_reinforced_tick DESC
            """,
            (agent_id, subject),
        )
        fact_key = _semantic_key(fact)
        for row in await cur.fetchall():
            row_dict = dict(row)
            existing_key = _semantic_key(row_dict.get("fact", ""))
            if fact_key and (fact_key == existing_key or fact_key in existing_key or existing_key in fact_key):
                return row_dict
        return None


class AgentRepository:
    def __init__(self, conn):
        self.conn = conn

    async def create(
        self,
        name: str,
        bio: dict,
        current_location: str | None = None,
        state: dict | None = None,
        needs: dict | None = None,
    ) -> int:
        cur = await self.conn.execute(
            "INSERT INTO agents (name, bio_json, state_json, needs_json, current_location) VALUES (?, ?, ?, ?, ?)",
            (name, json.dumps(bio), json.dumps(state or {}), json.dumps(needs or {}), current_location),
        )
        await self.conn.commit()
        return int(cur.lastrowid)

    async def upsert(
        self,
        name: str,
        bio: dict,
        current_location: str | None = None,
        state: dict | None = None,
        needs: dict | None = None,
    ) -> int:
        existing = await self.get_by_name(name)
        if existing:
            await self.conn.execute(
                """
                UPDATE agents
                SET bio_json = ?,
                    state_json = ?,
                    needs_json = ?,
                    current_location = ?
                WHERE id = ?
                """,
                (
                    json.dumps(bio),
                    existing["state_json"] if state is None else json.dumps(state),
                    existing["needs_json"] if needs is None else json.dumps(needs),
                    existing["current_location"] if current_location is None else current_location,
                    existing["id"],
                ),
            )
            await self.conn.commit()
            return int(existing["id"])

        cur = await self.conn.execute(
            """
            INSERT INTO agents (name, bio_json, state_json, needs_json, current_location)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, json.dumps(bio), json.dumps(state or {}), json.dumps(needs or {}), current_location),
        )
        await self.conn.commit()
        if cur.lastrowid is None:
            raise RuntimeError(f"Agent upsert failed for {name}")
        return int(cur.lastrowid)

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

    async def get_needs(self, agent_id: int) -> dict:
        cur = await self.conn.execute("SELECT needs_json FROM agents WHERE id = ?", (agent_id,))
        row = await cur.fetchone()
        return _json_dict(row["needs_json"]) if row else {}

    async def update_needs(self, agent_id: int, needs: dict) -> None:
        await self.conn.execute("UPDATE agents SET needs_json = ? WHERE id = ?", (json.dumps(needs), agent_id))
        await self.conn.commit()


class EventRepository:
    def __init__(self, conn, agent_ids_by_name: dict[str, int] | None = None):
        self.conn = conn
        self.agent_ids_by_name = agent_ids_by_name or {}

    def set_agent_ids_by_name(self, agent_ids_by_name: dict[str, int]) -> None:
        self.agent_ids_by_name = dict(agent_ids_by_name)

    async def append(self, event, sim_time: str = "") -> int:
        payload = asdict(event) if is_dataclass(event) else dict(event)
        kind = event.__class__.__name__.replace("Event", "").lower()
        source_name = payload.get("speaker") or payload.get("agent")
        source_agent_id = self._source_agent_id(source_name)
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
                source_agent_id,
                payload.get("location") or payload.get("to_location"),
            ),
        )
        await self.conn.commit()
        return int(cur.lastrowid)

    def _source_agent_id(self, source_name: str | None) -> int | None:
        if not source_name:
            return None
        agent_id = self.agent_ids_by_name.get(source_name)
        return int(agent_id) if agent_id is not None else None

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
                "affinity": _clamp(float(existing["affinity"]) + affinity_delta),
                "trust": _clamp(float(existing["trust"]) + trust_delta),
                "familiarity": _clamp(float(existing["familiarity"]) + familiarity_delta),
                "tension": _clamp(float(existing["tension"]) + tension_delta),
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
                _clamp(affinity_delta),
                _clamp(trust_delta),
                _clamp(familiarity_delta),
                _clamp(tension_delta),
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


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _semantic_key(text: str) -> str:
    import re

    stop = {
        "the", "and", "that", "with", "from", "this", "into", "about",
        "consistent", "consistently", "frequent", "frequently", "suggests",
    }
    words = [word for word in re.findall(r"[a-z0-9']+", str(text).lower()) if word not in stop]
    return " ".join(words[:18])


class PlanRepository:
    def __init__(self, conn):
        self.conn = conn

    async def add(self, agent_id: int, plan: dict, sim_tick: int, sim_time: str, status: str = "active", sim_day: int = 0) -> int:
        cur = await self.conn.execute(
            "INSERT INTO plans (agent_id, sim_tick, sim_time, sim_day, status, plan_json) VALUES (?, ?, ?, ?, ?, ?)",
            (agent_id, sim_tick, sim_time, sim_day, status, json.dumps(plan)),
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
