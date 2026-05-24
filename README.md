# Generative Agents Simulation

Text-based generative agents simulation inspired by Stanford's "Generative Agents: Interactive Simulacra of Human Behavior" paper.

This project targets a local OpenAI-compatible LLM backend. The default config is set for Ollama at `http://localhost:11434/v1`.
By default, one tick is 30 simulated minutes, so 48 ticks covers one full simulated day.

## Setup

```powershell
pip install -e .[dev]
python -c "import src"
```

## Ollama Model

The config defaults use one fast model for all simulation calls: `gemma3:4b`.

```powershell
ollama pull gemma3:4b
```

## AMD GPU Tuning

For Ryzen AI Max / Radeon integrated GPUs on Windows, configure Ollama before starting the app:

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_VULKAN", "1", "User")
[Environment]::SetEnvironmentVariable("GGML_VK_VISIBLE_DEVICES", "0", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH", "4096", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "30m", "User")
```

Quit and restart Ollama, then verify placement:

```powershell
ollama run gemma3:4b "Return exactly: ok"
ollama ps
```

The `PROCESSOR` column should show `100% GPU` for `gemma3:4b`.

## Run

```powershell
python -m src.main --ticks 5
python -m src.main --ticks 48
```

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
