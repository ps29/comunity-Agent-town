import json

import pytest

from src.agents.agent import ACTION_PROMPT_CHAR_BUDGET
from src.agents.files import AgentFiles
from src.harness.agent_harness import evaluate_harness, run_harness
from src.storage.db import get_connection, init_db


async def _seed_passing_db(conn, agent_files_root):
    await conn.execute(
        "INSERT INTO agents (id, name, bio_json, current_location) VALUES (?, ?, ?, ?)",
        (1, "Maria", json.dumps({"name": "Maria"}), "cafe"),
    )
    await conn.execute(
        """
        INSERT INTO memories
        (agent_id, sim_tick, sim_time, kind, content, importance, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (1, 1, "08:00", "observation", "Maria is at the cafe.", 4, "{}"),
    )
    await conn.execute(
        "INSERT INTO plans (agent_id, sim_tick, sim_time, status, plan_json) VALUES (?, ?, ?, ?, ?)",
        (1, 1, "08:00", "active", json.dumps({"hour_08": "Prepare the cafe."})),
    )
    await conn.execute(
        """
        INSERT INTO world_events
        (sim_tick, sim_time, kind, payload_json, location)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            1,
            "08:00",
            "speech",
            json.dumps({"speaker": "Maria", "listener": "John", "location": "cafe", "content": "Good morning.", "sim_tick": 1}),
            "cafe",
        ),
    )
    await conn.commit()
    AgentFiles(agent_files_root).ensure({"name": "Maria", "goals": []})


@pytest.mark.asyncio
async def test_harness_evaluator_passes_with_core_outputs(tmp_path):
    db_path = tmp_path / "harness.sqlite3"
    agent_files_root = tmp_path / "agents"
    await init_db(str(db_path))
    conn = await get_connection(str(db_path))
    try:
        await _seed_passing_db(conn, agent_files_root)
        report = await evaluate_harness(conn, ticks=2, seed=42, db_path=str(db_path), reject_threshold=0, agent_files_root=agent_files_root)
    finally:
        await conn.close()

    assert report["passed"]
    assert report["metrics"]["agents"] == 1
    assert report["metrics"]["events_by_kind"] == {"speech": 1}
    assert report["metrics"]["fallback_count"] == 0
    assert report["metrics"]["max_prompt_chars"] == 0
    assert report["metrics"]["embedding_blob_fallbacks"] == 0
    assert report["metrics"]["hash_embedding_fallback_used"] is False
    assert report["failures"] == []


@pytest.mark.asyncio
async def test_harness_evaluator_fails_when_events_missing_after_multi_tick_run(tmp_path):
    db_path = tmp_path / "harness.sqlite3"
    agent_files_root = tmp_path / "agents"
    await init_db(str(db_path))
    conn = await get_connection(str(db_path))
    try:
        await _seed_passing_db(conn, agent_files_root)
        await conn.execute("DELETE FROM world_events")
        await conn.commit()
        report = await evaluate_harness(conn, ticks=2, seed=42, db_path=str(db_path), reject_threshold=0, agent_files_root=agent_files_root)
    finally:
        await conn.close()

    assert not report["passed"]
    assert "No world events were created after a multi-tick run." in report["failures"]


@pytest.mark.asyncio
async def test_harness_evaluator_fails_when_rejected_actions_exceed_threshold(tmp_path):
    db_path = tmp_path / "harness.sqlite3"
    agent_files_root = tmp_path / "agents"
    await init_db(str(db_path))
    conn = await get_connection(str(db_path))
    try:
        await _seed_passing_db(conn, agent_files_root)
        await conn.execute(
            """
            INSERT INTO world_events
            (sim_tick, sim_time, kind, payload_json, location)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                2,
                "08:30",
                "rejectedaction",
                json.dumps({"agent": "Maria", "proposal_type": "move", "reason": "unknown location", "location": "cafe", "sim_tick": 2}),
                "cafe",
            ),
        )
        await conn.commit()
        report = await evaluate_harness(conn, ticks=2, seed=42, db_path=str(db_path), reject_threshold=0, agent_files_root=agent_files_root)
    finally:
        await conn.close()

    assert not report["passed"]
    assert "Rejected actions exceeded threshold: 1 > 0." in report["failures"]
    assert report["metrics"]["rejected_reasons"] == ["unknown location"]


@pytest.mark.asyncio
async def test_harness_evaluator_fails_when_prompt_budget_is_exceeded(tmp_path):
    db_path = tmp_path / "harness.sqlite3"
    agent_files_root = tmp_path / "agents"
    await init_db(str(db_path))
    conn = await get_connection(str(db_path))
    try:
        await _seed_passing_db(conn, agent_files_root)
        report = await evaluate_harness(
            conn,
            ticks=2,
            seed=42,
            db_path=str(db_path),
            reject_threshold=0,
            agent_files_root=agent_files_root,
            llm_metrics={"max_prompt_chars": ACTION_PROMPT_CHAR_BUDGET + 1},
        )
    finally:
        await conn.close()

    assert not report["passed"]
    assert f"Prompt size exceeded budget: {ACTION_PROMPT_CHAR_BUDGET + 1} > {ACTION_PROMPT_CHAR_BUDGET}." in report["failures"]


@pytest.mark.asyncio
async def test_deterministic_harness_runner_writes_report(tmp_path):
    db_path = tmp_path / "harness.sqlite3"
    report_path = tmp_path / "harness_report.json"

    report = await run_harness(
        ticks=2,
        seed=42,
        db_path=str(db_path),
        report_path=str(report_path),
        reject_threshold=0,
        keep_artifacts=False,
    )

    assert report["passed"]
    assert report_path.exists()
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["metrics"]["agents"] == 3
    assert written["metrics"]["plans"] >= 3
    assert written["metrics"]["memories_total"] > 0
    assert written["metrics"]["fallback_count"] == 0
    assert written["metrics"]["max_prompt_chars"] > 0
    assert written["metrics"]["embedding_blob_fallbacks"] == 0
    assert written["metrics"]["hash_embedding_fallback_used"] is False
    assert written["metrics"]["event_diversity"] >= 1
