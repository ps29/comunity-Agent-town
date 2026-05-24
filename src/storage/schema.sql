CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    bio_json TEXT NOT NULL,
    state_json TEXT DEFAULT '{}',
    current_location TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    agent_id INTEGER NOT NULL,
    sim_tick INTEGER NOT NULL,
    sim_time TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    importance INTEGER NOT NULL,
    embedding BLOB,
    metadata_json TEXT DEFAULT '{}',
    last_accessed_tick INTEGER,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS world_events (
    id INTEGER PRIMARY KEY,
    sim_tick INTEGER NOT NULL,
    sim_time TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    source_agent_id INTEGER,
    location TEXT
);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY,
    source_agent_id INTEGER NOT NULL,
    target_agent_id INTEGER NOT NULL,
    affinity REAL DEFAULT 0.0,
    trust REAL DEFAULT 0.0,
    familiarity REAL DEFAULT 0.0,
    tension REAL DEFAULT 0.0,
    summary TEXT DEFAULT '',
    metadata_json TEXT DEFAULT '{}',
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_agent_id, target_agent_id),
    FOREIGN KEY (source_agent_id) REFERENCES agents(id),
    FOREIGN KEY (target_agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY,
    agent_id INTEGER NOT NULL,
    sim_tick INTEGER NOT NULL,
    sim_time TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    plan_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS schedules (
    id INTEGER PRIMARY KEY,
    agent_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    schedule_json TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS quests (
    id INTEGER PRIMARY KEY,
    agent_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    progress INTEGER NOT NULL DEFAULT 0,
    details_json TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id, sim_tick DESC);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(agent_id, importance DESC);
CREATE INDEX IF NOT EXISTS idx_events_tick ON world_events(sim_tick);
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_agent_id);
CREATE INDEX IF NOT EXISTS idx_plans_agent ON plans(agent_id, sim_tick DESC);
