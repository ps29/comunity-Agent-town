from pathlib import Path
from urllib.parse import urlparse

import aiosqlite
import numpy as np

DEFAULT_DB_PATH = "simulation.sqlite3"


async def get_connection(db_path: str = DEFAULT_DB_PATH) -> aiosqlite.Connection:
    if _is_postgres_url(db_path):
        return await PostgresConnectionAdapter.connect(db_path)
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    return conn


async def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    if _is_postgres_url(db_path):
        await init_postgres(db_path)
        return
    schema_path = Path(__file__).with_name("schema.sql")
    conn = await get_connection(db_path)
    try:
        await conn.executescript(schema_path.read_text(encoding="utf-8"))
        await _ensure_sqlite_columns(conn)
        await conn.commit()
    finally:
        await conn.close()


async def init_postgres(database_url: str) -> None:
    import asyncpg

    schema_path = Path(__file__).with_name("postgres_schema.sql")
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(schema_path.read_text(encoding="utf-8"))
    finally:
        await conn.close()


def _is_postgres_url(value: str) -> bool:
    scheme = urlparse(value).scheme
    return scheme in {"postgres", "postgresql"}


async def _ensure_sqlite_columns(conn) -> None:
    await _add_column_if_missing(conn, "agents", "state_json", "TEXT DEFAULT '{}'")
    await _add_column_if_missing(conn, "agents", "needs_json", "TEXT DEFAULT '{}'")
    await _add_column_if_missing(conn, "memories", "metadata_json", "TEXT DEFAULT '{}'")
    await _add_column_if_missing(conn, "memories", "consolidated", "INTEGER NOT NULL DEFAULT 0")
    await _add_column_if_missing(conn, "plans", "sim_day", "INTEGER NOT NULL DEFAULT 0")


async def _add_column_if_missing(conn, table: str, column: str, ddl: str) -> None:
    cur = await conn.execute(f"PRAGMA table_info({table})")
    columns = {row["name"] for row in await cur.fetchall()}
    if column not in columns:
        await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


class PostgresCursorAdapter:
    def __init__(self, rows=None, lastrowid: int | None = None):
        self._rows = rows or []
        self.lastrowid = lastrowid

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return self._rows


class PostgresConnectionAdapter:
    def __init__(self, conn):
        self.conn = conn
        self.row_factory = None

    @classmethod
    async def connect(cls, database_url: str):
        import asyncpg

        return cls(await asyncpg.connect(database_url))

    async def execute(self, sql: str, params=()):
        query = _translate_placeholders(sql)
        values = tuple(_adapt_postgres_value(value) for value in params)
        lowered = query.strip().lower()
        if lowered.startswith("insert") and " returning " not in lowered:
            query = query.rstrip().rstrip(";") + " RETURNING id"
            row = await self.conn.fetchrow(query, *values)
            return PostgresCursorAdapter(lastrowid=int(row["id"]) if row else None)
        if lowered.startswith("select"):
            rows = await self.conn.fetch(query, *values)
            return PostgresCursorAdapter([dict(row) for row in rows])
        await self.conn.execute(query, *values)
        return PostgresCursorAdapter()

    async def executescript(self, sql: str):
        await self.conn.execute(sql)

    async def commit(self):
        return None

    async def close(self):
        await self.conn.close()


def _translate_placeholders(sql: str) -> str:
    out = []
    index = 1
    for char in sql:
        if char == "?":
            out.append(f"${index}")
            index += 1
        else:
            out.append(char)
    return "".join(out)


def _adapt_postgres_value(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        emb = np.frombuffer(value, dtype=np.float32)
        return "[" + ",".join(f"{float(item):.8f}" for item in emb) + "]"
    return value
