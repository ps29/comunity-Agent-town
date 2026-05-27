# Community Agent Town — Generative Agents Simulation v0.2

Text-based generative agents simulation inspired by Stanford's ["Generative Agents: Interactive Simulacra of Human Behavior"](https://arxiv.org/abs/2304.03442) paper. Three AI-driven characters live out daily routines in a small cozy-mystery town, with grounded actions, live agent files, needs-driven behaviour, relationship memory, and semantic long-term memory.

The simulation runs against a local `llama.cpp` server through its OpenAI-compatible API (default: `http://localhost:8080/v1`). One tick = 30 simulated minutes; 48 ticks = one full simulated day (08:00–22:00).

---

## Table of Contents

- [Town Setting](#town-setting)
- [Agents](#agents)
- [Architecture Overview](#architecture-overview)
- [Setup](#setup)
- [llama.cpp Model](#llamacpp-model)
- [AMD GPU Tuning](#amd-gpu-tuning)
- [Run](#run)
- [Deterministic Harness](#deterministic-harness)
- [Reliability Gate](#reliability-gate)
- [Agent Files](#agent-files)
- [Grounded Actions](#grounded-actions)
- [Memory System](#memory-system)
- [Needs System](#needs-system)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)

---

## Town Setting

The world is a small cozy-mystery town with ten named locations:

| Location | Key Objects |
|---|---|
| `cafe` | `coffee_maker`, `pastry_case`, `corner_table` |
| `town_square` | `notice_board`, `fountain`, `town_map` |
| `library` | `study_table`, `reading_chair`, `local_history_shelf` |
| `archive_room` | `archive_boxes`, `map_table`, `old_ledger` |
| `riverside_path` | `river_marker`, `willow_bench` |
| `old_mill` | `old_mill_wheel`, `mill_door`, `grain_sacks` |
| `market_stalls` | `market_crates`, `flower_stall`, `weighing_scale` |
| `community_hall` | `meeting_table`, `lost_and_found_box`, `event_calendar` |
| `newspaper_office` | `typewriter`, `clippings_wall`, `reporter_notebook` |
| `park` | `bench`, `pond` |

All locations and objects are defined in `config/world.yaml`. Agents may only move to listed locations and use listed object affordances — the resolver hard-rejects anything else.

---

## Agents

Three agents, each with distinct goals and preferred locations:

**Maria** (28, Cafe Owner)
- Runs the cafe as a community hub; follows notices, market gossip, and community events.
- Preferred places: cafe, town\_square, market\_stalls, community\_hall, park.

**John** (32, Novelist / Newspaper Contributor)
- Introverted routine writer; follows civic oddities for story material. Targets one good conversation per day.
- Preferred places: newspaper\_office, riverside\_path, park, archive\_room, cafe.

**Emma** (26, Folklore Researcher)
- Curious and energetic; tracing a gentle mystery through old records, riverside markers, and the mill. Wants to know everyone.
- Preferred places: library, archive\_room, old\_mill, riverside\_path, cafe, town\_square.

Agent personalities, goals, Big Five traits, and starting needs are configured in `config/agents.yaml`.

---

## Architecture Overview

```
src/main.py            CLI entry point — loads config, creates subsystems, runs simulation
     │
     └─► SimulationEngine (engine/simulation.py)
             │
             ├── Per tick:
             │   1. daily_heartbeat()    — reset plan & files at 08:00
             │   2. perceive()           — all agents in parallel
             │   3. maybe_reflect()      — sequential, importance-gated
             │   4. maybe_plan()         — sequential, once per sim-day
             │   5. end_of_day_heartbeat()— consolidate semantic memory at 22:00
             │   6. propose_action()     — all agents in parallel
             │   7. resolve()            — sorted: Speak/Use > Move > Wait
             │
             ├── Agent (agents/agent.py)
             │     perceive → reflect → plan → propose → anti-loop guards
             │
             ├── LLMGateway (llm/gateway.py)
             │     routes ACT / PLAN / REFLECT calls → LlamaCppClient
             │
             ├── MemoryService (cognition/memory.py)
             │     scored retrieval: 0.35×recency + 0.25×importance
             │                     + 0.30×relevance + 0.10×metadata_bonus
             │
             ├── WorldState (world/state.py)
             │     agent positions, object states, location definitions
             │
             └── Storage (storage/repositories.py)
                   SQLite (default) or Postgres — async via aiosqlite / asyncpg
```

---

## Setup

```powershell
pip install -e .[dev]
python -c "import src"
```

---

## llama.cpp Model

All simulation calls use a single fast model alias: `gemma3-4b`. The expected GGUF path:

```powershell
$model = "$env:USERPROFILE\.cache\huggingface\hub\models--ggml-org--gemma-3-4b-it-GGUF\snapshots\d0976223747697cb51e056d85c532013931fe52e\gemma-3-4b-it-Q4_K_M.gguf"
```

> **Note:** The Ollama `gemma3:4b` blob uses an Ollama-specific GGUF layout and is **not** accepted by this llama.cpp build as a direct server model.

---

## AMD GPU Tuning

For Ryzen AI Max / Radeon integrated GPUs on Windows, use the Vulkan-enabled llama.cpp build:

```powershell
# Pin Vulkan device 0
[Environment]::SetEnvironmentVariable("GGML_VK_VISIBLE_DEVICES", "0", "User")

# Confirm device is visible
C:\Users\pssh\llama.cpp\llama-bench.exe --list-devices
# Expected: AMD Radeon(TM) 8060S Graphics listed under Vulkan
```

Start the server with full GPU layer offload:

```powershell
C:\Users\pssh\llama.cpp\llama-server.exe `
  -m $model `
  -ngl 99 `
  -dev Vulkan0 `
  --ctx-size 16384 `
  --host 127.0.0.1 `
  --port 8080 `
  --alias gemma3-4b
```

The startup log must confirm Vulkan backend and GPU layer offload. CPU-only fallback is not an accepted configuration.

**Recommended:** let `src.main` manage the server automatically — it pins `GGML_VK_VISIBLE_DEVICES=0`, starts llama.cpp with `-ngl 99 -dev Vulkan0`, and aborts if the log does not confirm Vulkan offload:

```powershell
python -m src.main --ticks 20 --start-llama-server --llama-ctx-size 16384
```

---

## Run

### Minimal run (server already running)

```powershell
python -m src.main --ticks 5
python -m src.main --ticks 48
```

### Let main.py start and stop llama-server

```powershell
python -m src.main --ticks 10 --start-llama-server
```

### Full sim-day with larger context window

```powershell
python -m src.main --ticks 48 --start-llama-server --llama-ctx-size 16384
```

If context overflows occur and VRAM permits, try `--llama-ctx-size 32768`.

### Custom artifact paths (useful for benchmarking)

```powershell
python -m src.main --ticks 5 --seed 42 --start-llama-server `
  --db .tmp\run.sqlite3 `
  --events .tmp\run_events.jsonl `
  --transcript .tmp\run.log `
  --agents-root .tmp\run_agents
```

### GPU benchmark / Vulkan verification

```powershell
python -m src.llm.benchmark
```

Writes diagnostics under `.tmp/`:
- `.tmp/llama_cpp_devices.log`
- `.tmp/llama_cpp_bench.json`
- `.tmp/llama_cpp_server.log`
- `.tmp/llama_cpp_prompt_latency.jsonl`
- `.tmp/llama_cpp_event_summary.json`

---

## Deterministic Harness

Use the harness to verify the simulation engine, agent loop, memory writes, plans, world events, and action resolver **without a live LLM or GPU**. It uses fake embeddings and hardcoded agent responses.

```powershell
python -m src.harness.agent_harness --ticks 8 --seed 42
```

If installed with `pip install -e .[dev]`:

```powershell
agent-harness --ticks 8 --seed 42
```

Default artifacts (under `.tmp/`):
- `.tmp/harness.sqlite3`
- `.tmp/harness_report.json`
- `.tmp/harness_simulation.log`
- `.tmp/harness_agents/`

Useful options:

```powershell
# Quick 2-tick check with custom DB path
python -m src.harness.agent_harness --ticks 2 --db .tmp\harness_cli.sqlite3

# Strict: fail if any action is rejected
python -m src.harness.agent_harness --ticks 8 --reject-threshold 1

# Keep artifacts after a successful run (default: cleaned up)
python -m src.harness.agent_harness --ticks 8 --keep-artifacts
```

The harness exits non-zero if any of these invariants fail:
- No agents loaded
- No plans or memories created
- No world events after a multi-tick run
- Missing runtime agent files
- Rejected actions above the configured threshold
- Largest prompt exceeds the action prompt budget (12 000 chars)

The JSON report includes pass/fail, per-kind call metrics, rejected action reasons, LLM fallback counts, repeated speech phrases, and embedding fallback counts.

---

## Reliability Gate

Run before and after any change to core simulation behaviour:

```powershell
python -m pytest -q
python -m src.harness.agent_harness --ticks 8 --seed 42
```

---

## Agent Files

Each agent has three runtime markdown files under `agents/<name>/`:

| File | Owner | Purpose |
|---|---|---|
| `SOUL.md` | Manual | Stable personality & guardrails. Loaded fresh each LLM call — edit during a run to correct drift on the next tick. Never auto-edited. |
| `KNOWLEDGE.md` | Simulation | Durable facts appended after reflections. Concise truths extracted from raw memories. |
| `TODAY.md` | Simulation | Active daily schedule. Reset at simulated 08:00 and rewritten when the daily plan is generated. |

---

## Grounded Actions

The LLM receives a grounded action menu built from the current world state. It can only:
- **Move** to locations listed in `config/world.yaml`
- **Speak** to agents physically present in the same location
- **Use** nearby objects with affordances listed for that object

Objects define compact `allowed_states` and exact `affordances`. An interaction like `brew_coffee` sets `coffee_maker` to `brewing`; free-form prose like `make a latte for the customer` is rejected at the resolver and logged as a `RejectedActionEvent` in the transcript and event stream.

Anti-hallucination guards in `agent.py`:
- `_normalize_object_target()` — maps informal names (`"coffee"`) to exact IDs (`coffee_maker`)
- `_normalize_interaction()` — maps informal verbs (`"typing"`) to exact affordances (`write_article`)
- `_sanitize_plan()` — removes invented NPC names from the daily schedule
- `_ground_historical_claims()` — replaces fabricated history with grounded terms

---

## Memory System

### Scored retrieval

When an agent chooses an action, it retrieves the 14 most relevant memories from a candidate pool of 80 recent + 80 high-importance memories:

```
score = 0.35 × recency
      + 0.25 × importance
      + 0.30 × relevance        (cosine similarity to query embedding)
      + 0.10 × metadata_bonus   (+0.15 if location matches, +0.15 if nearby agent matches)
```

Recency uses exponential decay: `e^(−0.05 × ticks_ago)`.

### Memory kinds

| Kind | importance | When created |
|---|---|---|
| `observation` | 1–10 (rule-based) | Every tick via `perceive()` |
| `reflection` | 7 | When importance accumulator ≥ threshold |
| `plan` | 6 | Once per sim-day |
| `dialogue` | 5 | When a nearby agent speaks (absorbed via EventBus) |

### Semantic memory

At 22:00 each sim-day, `end_of_day_heartbeat()` distils up to 5 durable facts into the `semantic_memory` table with a confidence score (default 0.65). These facts are loaded back into every action prompt alongside `KNOWLEDGE.md` and `TODAY.md`, giving agents stable long-term memory independent of raw observation recency.

### Reflection

Reflection fires when `importance_accumulator ≥ reflection_threshold` (default 25). The agent receives its 15 most recent memories and the LLM returns 0-2 insights (stored as reflection memories) and 0-2 knowledge facts (appended to `KNOWLEDGE.md`). The accumulator resets to 0 after each reflection.

---

## Needs System

Each agent tracks three needs (range 0.0–1.0) that decay every tick:

| Need | Increases when | Decreases passively |
|---|---|---|
| `hunger` | Uses `coffee_maker` or `pastry_case` | Each tick |
| `energy` | Sits at `bench`, `reading_chair`, or `corner_table` | Each tick |
| `social_satiety` | Speaks to another agent | Each tick |

Current needs appear in every action prompt as a status string (e.g. `hunger: moderate | energy: good | social: lonely`). When needs are urgent the agent may override the planned action with a needs-driven fallback (eat, rest, or seek conversation).

---

## Configuration

| File | Controls |
|---|---|
| `config/agents.yaml` | Agent name, bio, goals, personality, Big Five traits, preferred places, starting needs |
| `config/world.yaml` | Locations, objects, allowed\_states, default\_state, affordances |
| `config/simulation.yaml` | tick\_duration\_sim\_seconds, reflection\_threshold, memory pool sizes |

---

## Outputs

| File | Contents |
|---|---|
| `simulation.sqlite3` | Full database: agents, memories, plans, relationships, semantic\_memory, events, call\_logs |
| `events.jsonl` | One JSON event per line — move, speech, object state change, rejected action |
| `simulation.log` | Human-readable transcript: PERCEIVE / REFLECT / PLAN / ACT / WAIT entries per agent per tick |
| `agents/<name>/KNOWLEDGE.md` | Accumulated knowledge for each agent (grows during the run) |
| `agents/<name>/TODAY.md` | Current daily schedule for each agent |

---

## Troubleshooting

**`PermissionError` on `simulation.sqlite3`**
Another process holds the file open. Stop the previous simulation or use `--db .tmp/other.sqlite3`.

**LLM returns malformed JSON**
`src/llm/parsing.py` applies fallback parsing. If the fallback also fails, the agent receives a `WaitProposal` with a reason string. Check `simulation.log` for `FALLBACK` entries.

**Agent keeps repeating the same action**
The anti-loop guards (`_avoid_stale_action`, `_avoid_repetitive_speech`) should catch this within 2–3 ticks. If it persists, check `harness_report.json` for rejected action counts and review `KNOWLEDGE.md` for stale content.

**GPU not detected / CPU-only fallback**
Run `python -m src.llm.benchmark` and check `.tmp/llama_cpp_devices.log`. Ensure `GGML_VK_VISIBLE_DEVICES=0` is set and the Vulkan-enabled llama.cpp build is used.

**Context overflow warnings**
Increase `--llama-ctx-size` (try 32768 if VRAM allows). The action prompt is hard-capped at 12 000 chars and trims memories automatically, but the LLM server's own context window is separate.
