# Generative Agents Simulation v0.2

Text-based generative agents simulation inspired by Stanford's "Generative Agents: Interactive Simulacra of Human Behavior" paper, with grounded actions, live agent files, needs-driven behavior, relationship memory, and semantic long-term memory.

This project targets a local `llama.cpp` server through its OpenAI-compatible API. The default config expects `llama-server` at `http://localhost:8080/v1`.
By default, one tick is 30 simulated minutes, so 48 ticks covers one full simulated day.

## Town Setting

The default world is now a small cozy mystery town rather than a cafe-only
village loop. The map includes the cafe, town square, library, archive room,
riverside path, old mill, market stalls, community hall, newspaper office, and
park. Agents can ground actions in town objects such as the `notice_board`,
`archive_boxes`, `map_table`, `river_marker`, `old_mill_wheel`,
`market_crates`, `event_calendar`, `typewriter`, and `clippings_wall`.

The current cast remains three agents:

- Maria runs the cafe as a community hub and follows notices, market gossip,
  and community events.
- John is a novelist and part-time newspaper contributor who follows civic
  oddities for story material.
- Emma is a folklore researcher tracing a gentle mystery around old records,
  riverside markers, and the mill.

The wider setting is still fully grounded: agents can only move to configured
locations and use listed object affordances from `config/world.yaml`.

## Setup

```powershell
pip install -e .[dev]
python -c "import src"
```

## llama.cpp Model

The config defaults use one fast model alias for all simulation calls: `gemma3-4b`.
The benchmark uses the llama.cpp-compatible GGUF for the same Gemma 3 4B instruction model:

```powershell
$model = "$env:USERPROFILE\.cache\huggingface\hub\models--ggml-org--gemma-3-4b-it-GGUF\snapshots\d0976223747697cb51e056d85c532013931fe52e\gemma-3-4b-it-Q4_K_M.gguf"
```

The local Ollama `gemma3:4b` blob is an Ollama-specific multimodal GGUF layout and is not accepted by this llama.cpp build as a direct server model.

## AMD GPU Tuning

For Ryzen AI Max / Radeon integrated GPUs on Windows, run the Vulkan-enabled llama.cpp build and pin Vulkan device 0:

```powershell
[Environment]::SetEnvironmentVariable("GGML_VK_VISIBLE_DEVICES", "0", "User")
C:\Users\pssh\llama.cpp\llama-bench.exe --list-devices
```

The device list must include `AMD Radeon(TM) 8060S Graphics` under Vulkan. Start the server with full GPU layer offload:

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

The startup log must show the Vulkan backend and GPU layer offload. CPU-only fallback is not an accepted configuration.

## Run

```powershell
python -m src.main --ticks 5
python -m src.main --ticks 48
```

To let `main.py` start and stop `llama-server` for the run:

```powershell
python -m src.main --ticks 10 --start-llama-server
```

For longer runs, increase the llama.cpp context window. On a 16 GB VRAM GPU, start with:

```powershell
python -m src.main --ticks 48 --start-llama-server --llama-ctx-size 16384
```

If the server still reports context overflows and VRAM is available, try `--llama-ctx-size 32768`.

Use custom artifact paths for benchmark or comparison runs:

```powershell
python -m src.main --ticks 5 --seed 42 --start-llama-server --db .tmp\llama_cpp_benchmark.sqlite3 --events .tmp\llama_cpp_events.jsonl --transcript .tmp\llama_cpp_simulation.log --agents-root .tmp\llama_cpp_agents
```

## llama.cpp Benchmark

Run the bundled benchmark helper to verify Vulkan GPU availability, run `llama-bench`, start `llama-server`, perform a smoke request, and write latency artifacts under `.tmp/`:

```powershell
python -m src.llm.benchmark
```

The helper writes:

- `.tmp/llama_cpp_devices.log`
- `.tmp/llama_cpp_bench.json`
- `.tmp/llama_cpp_server.log`
- `.tmp/llama_cpp_prompt_latency.jsonl`
- `.tmp/llama_cpp_event_summary.json`

## Deterministic Agent Harness

Use the harness when you want to check whether the simulation engine, agent loop,
memory writes, plans, world events, and grounded action resolver still work
without depending on llama.cpp or a live model. It is meant for fast regression
checks before/after code changes, CI-style smoke tests, and debugging core
simulation health. Use `src.main` instead when you want to observe real model
behavior and narrative quality.

```powershell
python -m src.harness.agent_harness --ticks 8 --seed 42
```

By default, harness artifacts are written under `.tmp/`:

- `.tmp/harness.sqlite3`
- `.tmp/harness_report.json`
- `.tmp/harness_simulation.log`
- `.tmp/harness_agents/`

The harness exits non-zero if core invariants fail, such as no agents loaded,
no plans or memories created, no world events after a multi-tick run, missing
runtime agent files, or rejected actions above the configured threshold.

Useful options:

```powershell
python -m src.harness.agent_harness --ticks 2 --db .tmp\harness_cli.sqlite3 --report .tmp\harness_cli_report.json
python -m src.harness.agent_harness --ticks 8 --reject-threshold 1
python -m src.harness.agent_harness --ticks 8 --keep-artifacts
```

The JSON report contains the pass/fail result, metrics by kind, rejected action
reasons, warnings, and failures. This makes it useful for automated checks as
well as quick local inspection.

## Reliability Gate

Before changing core simulation behavior, run the fast unit suite and the
deterministic harness:

```powershell
python -m pytest -q
python -m src.harness.agent_harness --ticks 8 --seed 42
```

If installed with `pip install -e .[dev]`, the harness is also available as:

```powershell
agent-harness --ticks 8 --seed 42
```

The harness fails when required runtime files are missing, no agents/plans/
memories are produced, no world events appear after a multi-tick run, rejected
actions exceed the configured threshold, or the largest prompt exceeds the
action prompt budget. Its report also includes LLM fallback counts, maximum
prompt size, repeated speech phrases, and embedding blob fallback counts.

## Agent Files

On startup the simulation creates per-agent runtime files under `agents/<name>/`:

- `SOUL.md` contains stable personality and guardrails. It is loaded fresh before LLM calls, so you can edit it during a run to correct drift on the next tick.
- `KNOWLEDGE.md` contains curated facts from reflection. Reflections append concise durable truths here instead of relying only on noisy raw memories.
- `TODAY.md` contains the active daily plan. It is reset at the simulated 08:00 heartbeat and rewritten when the daily plan is generated.

`SOUL.md` is not edited automatically. `KNOWLEDGE.md` and `TODAY.md` are simulation-owned.

## Grounded Actions

The LLM receives a grounded action menu built from the current world state. It can only move to known locations, speak to nearby agents, and use nearby objects with listed affordances.

Objects in `config/world.yaml` now define compact `allowed_states` and exact `affordances`. The resolver maps valid affordances to compact states, so an interaction like `brew_coffee` can set `coffee_maker` to `brewing`, while prose like `make a delicious latte for the customer` is rejected and never becomes object state.

Rejected actions emit `RejectedActionEvent` entries into the transcript/event stream for debugging hallucinated people, places, or object interactions.

## Memory Retrieval

Action retrieval now scores a bounded candidate pool: recent memories plus high-importance memories. This keeps long runs from scanning every memory on every tick while preserving useful recent and important context.

## Needs And Semantic Memory

Each agent has lightweight needs for hunger, energy, and social satiety. Needs decay every tick, appear in the action prompt, and are updated when agents socialize, rest, or interact with food-related objects.

The database includes a `semantic_memory` table for durable facts distilled from end-of-day reflection. Semantic facts are loaded back into the live prompt context alongside `SOUL.md`, `KNOWLEDGE.md`, and `TODAY.md`, giving agents stable high-signal memory without relying only on raw observations.
