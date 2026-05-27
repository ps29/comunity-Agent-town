CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS agents (
    id BIGSERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    bio_json JSONB NOT NULL,
    state_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    needs_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    current_location TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memories (
    id BIGSERIAL PRIMARY KEY,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    sim_tick INTEGER NOT NULL,
    sim_time TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    importance INTEGER NOT NULL CHECK (importance BETWEEN 1 AND 10),
    embedding vector(384),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_accessed_tick INTEGER,
    consolidated INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS semantic_memory (
    id BIGSERIAL PRIMARY KEY,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    subject TEXT NOT NULL,
    fact TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    source_memory_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    first_seen_tick INTEGER NOT NULL,
    last_reinforced_tick INTEGER NOT NULL,
    decayed INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS relationships (
    id BIGSERIAL PRIMARY KEY,
    source_agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    target_agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    affinity REAL NOT NULL DEFAULT 0,
    trust REAL NOT NULL DEFAULT 0,
    familiarity REAL NOT NULL DEFAULT 0,
    tension REAL NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(source_agent_id, target_agent_id)
);

CREATE TABLE IF NOT EXISTS world_events (
    id BIGSERIAL PRIMARY KEY,
    sim_tick INTEGER NOT NULL,
    sim_time TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    source_agent_id BIGINT REFERENCES agents(id) ON DELETE SET NULL,
    location TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS plans (
    id BIGSERIAL PRIMARY KEY,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    sim_tick INTEGER NOT NULL,
    sim_time TEXT NOT NULL,
    sim_day INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    plan_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS schedules (
    id BIGSERIAL PRIMARY KEY,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    schedule_json JSONB NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS quests (
    id BIGSERIAL PRIMARY KEY,
    agent_id BIGINT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    progress INTEGER NOT NULL DEFAULT 0,
    details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memories_agent_tick ON memories(agent_id, sim_tick DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_agent ON semantic_memory(agent_id, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_subject ON semantic_memory(agent_id, subject);
CREATE INDEX IF NOT EXISTS idx_memories_metadata ON memories USING GIN (metadata_json);
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_agent_id);
CREATE INDEX IF NOT EXISTS idx_world_events_tick ON world_events(sim_tick);
CREATE INDEX IF NOT EXISTS idx_plans_agent_tick ON plans(agent_id, sim_tick DESC);
