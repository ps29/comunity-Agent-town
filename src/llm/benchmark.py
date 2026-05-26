import argparse
import json
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI


DEFAULT_LLAMA_DIR = Path(r"C:\Users\pssh\llama.cpp")
DEFAULT_MODEL = Path(
    r"C:\Users\pssh\.cache\huggingface\hub\models--ggml-org--gemma-3-4b-it-GGUF\snapshots\d0976223747697cb51e056d85c532013931fe52e\gemma-3-4b-it-Q4_K_M.gguf"
)
DEFAULT_ALIAS = "gemma3-4b"
DEFAULT_OUT = Path(".tmp")
REQUIRED_VULKAN_DEVICE = "AMD Radeon(TM) 8060S Graphics"
REQUIRED_VULKAN_TARGET = "Vulkan0"
REQUIRED_GEMMA3_EPSILON_KEY = "gemma3.attention.layer_norm_rms_epsilon"
TOKENIZER_ARRAY_KEYS = {
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.tokens",
}


def run_command(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, check=False)


def _read_gguf_string(data: bytes, offset: int) -> tuple[str, int]:
    size = int.from_bytes(data[offset : offset + 8], "little")
    offset += 8
    return data[offset : offset + size].decode("utf-8"), offset + size


def _skip_gguf_value(data: bytes, offset: int, value_type: int) -> int:
    scalar_sizes = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }
    if value_type in scalar_sizes:
        return offset + scalar_sizes[value_type]
    if value_type == 8:
        _, offset = _read_gguf_string(data, offset)
        return offset
    if value_type == 9:
        item_type = int.from_bytes(data[offset : offset + 4], "little")
        item_count = int.from_bytes(data[offset + 4 : offset + 12], "little")
        offset += 12
        for _ in range(item_count):
            offset = _skip_gguf_value(data, offset, item_type)
        return offset
    raise ValueError(f"Unsupported GGUF metadata type: {value_type}")


def _metadata_end_and_keys(data: bytes) -> tuple[int, set[str]]:
    if data[:4] != b"GGUF":
        raise ValueError("Model file is not GGUF")
    metadata_count = int.from_bytes(data[16:24], "little")
    offset = 24
    keys = set()
    for _ in range(metadata_count):
        key, offset = _read_gguf_string(data, offset)
        keys.add(key)
        value_type = int.from_bytes(data[offset : offset + 4], "little")
        offset = _skip_gguf_value(data, offset + 4, value_type)
    return offset, keys


def _array_item_end(data: bytes, offset: int, item_type: int) -> int:
    return _skip_gguf_value(data, offset, item_type)


def _trim_gguf_array_value(data: bytes, value_offset: int) -> bytes:
    item_type = int.from_bytes(data[value_offset : value_offset + 4], "little")
    item_count = int.from_bytes(data[value_offset + 4 : value_offset + 12], "little")
    if item_count == 0:
        return data[value_offset : value_offset + 12]
    body_offset = value_offset + 12
    offset = body_offset
    for _ in range(item_count - 1):
        offset = _array_item_end(data, offset, item_type)
    return (
        item_type.to_bytes(4, "little")
        + (item_count - 1).to_bytes(8, "little")
        + data[body_offset:offset]
    )


def _align_offset(offset: int, alignment: int = 32) -> int:
    return offset + ((alignment - (offset % alignment)) % alignment)


def _tensor_info_end(data: bytes, offset: int, tensor_count: int) -> int:
    for _ in range(tensor_count):
        _, offset = _read_gguf_string(data, offset)
        dimensions = int.from_bytes(data[offset : offset + 4], "little")
        offset += 4 + (8 * dimensions) + 4 + 8
    return offset


def _patch_gguf_metadata(data: bytes) -> bytes:
    if data[:4] != b"GGUF":
        raise ValueError("Model file is not GGUF")
    tensor_count = int.from_bytes(data[8:16], "little")
    metadata_count = int.from_bytes(data[16:24], "little")
    offset = 24
    entries = []
    keys = set()
    for _ in range(metadata_count):
        entry_start = offset
        key, offset = _read_gguf_string(data, offset)
        keys.add(key)
        value_type_offset = offset
        value_type = int.from_bytes(data[offset : offset + 4], "little")
        value_offset = offset + 4
        value_end = _skip_gguf_value(data, value_offset, value_type)
        if key in TOKENIZER_ARRAY_KEYS and value_type == 9:
            entries.append(data[entry_start:value_offset] + _trim_gguf_array_value(data, value_offset))
        else:
            entries.append(data[entry_start:value_end])
        offset = value_end

    if REQUIRED_GEMMA3_EPSILON_KEY not in keys:
        metadata_count += 1
        entries.append(_encoded_float32_metadata(REQUIRED_GEMMA3_EPSILON_KEY, 1e-6))

    original_metadata_end = offset
    original_tensor_info_end = _tensor_info_end(data, original_metadata_end, tensor_count)
    original_data_start = _align_offset(original_tensor_info_end)
    patched_prefix = data[:16] + metadata_count.to_bytes(8, "little") + b"".join(entries)
    patched_tensor_infos = data[original_metadata_end:original_tensor_info_end]
    patched_tensor_info_end = len(patched_prefix) + len(patched_tensor_infos)
    padding = b"\0" * (_align_offset(patched_tensor_info_end) - patched_tensor_info_end)
    return patched_prefix + patched_tensor_infos + padding + data[original_data_start:]


def _encoded_float32_metadata(key: str, value: float) -> bytes:
    key_bytes = key.encode("utf-8")
    return (
        len(key_bytes).to_bytes(8, "little")
        + key_bytes
        + (6).to_bytes(4, "little")
        + struct.pack("<f", value)
    )


def prepare_gguf_path(model: Path, out_dir: Path, alias: str) -> Path:
    data = model.read_bytes()
    _, keys = _metadata_end_and_keys(data)
    if model.suffix.lower() == ".gguf" and REQUIRED_GEMMA3_EPSILON_KEY in keys:
        return model

    patched_model = out_dir / f"{alias}.gguf"
    if patched_model.exists():
        patched_model.unlink()
    patched_model.write_bytes(_patch_gguf_metadata(data))
    return patched_model


def require_gpu_device(llama_dir: Path, env: dict[str, str]) -> str:
    result = run_command([str(llama_dir / "llama-bench.exe"), "--list-devices"], llama_dir, env)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"llama-bench --list-devices failed:\n{output}")
    if "Vulkan" not in output or REQUIRED_VULKAN_DEVICE not in output:
        raise RuntimeError(f"Required Vulkan GPU was not detected:\n{output}")
    return output


def run_token_benchmark(llama_dir: Path, model: Path, out_dir: Path, env: dict[str, str]) -> Path:
    out_path = out_dir / "llama_cpp_bench.json"
    result = run_command(
        [
            str(llama_dir / "llama-bench.exe"),
            "-m",
            str(model),
            "-ngl",
            "99",
            "-dev",
            "Vulkan0",
            "-p",
            "512",
            "-n",
            "128",
            "-fa",
            "1",
            "-o",
            "json",
        ],
        llama_dir,
        env,
    )
    out_path.write_text(result.stdout, encoding="utf-8")
    (out_dir / "llama_cpp_bench.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"llama-bench failed; see {out_path} and llama_cpp_bench.stderr.log")
    return out_path


def start_server(
    llama_dir: Path,
    model: Path,
    alias: str,
    log_path: Path,
    env: dict[str, str],
    ctx_size: int = 8192,
) -> subprocess.Popen:
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.Popen(
            [
                str(llama_dir / "llama-server.exe"),
                "-m",
                str(model),
                "-ngl",
                "99",
                "-dev",
                REQUIRED_VULKAN_TARGET,
                "--ctx-size",
                str(ctx_size),
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--alias",
                alias,
                "--verbose",
            ],
            cwd=llama_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )


def wait_for_server(log_path: Path, timeout_seconds: int = 120) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if "error loading model" in text.lower():
            raise RuntimeError(f"llama-server failed to load model:\n{text}")
        if "listening" in text.lower() or "server is listening" in text.lower():
            return text
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for llama-server. Log:\n{log_path.read_text(encoding='utf-8', errors='replace')}")


def require_gpu_offload(log_text: str) -> None:
    status = gpu_offload_status(log_text)
    if not status["has_vulkan"] or not status["has_required_device"] or int(status["offloaded_layers"] or 0) <= 0:
        raise RuntimeError(f"llama-server did not prove Vulkan layer offload:\n{log_text}")


def gpu_offload_status(log_text: str) -> dict[str, bool | int | None | str]:
    offload_match = re.search(r"offload(?:ing|ed)?\s+(\d+)/(\d+)\s+layers", log_text, re.IGNORECASE)
    assigned_layers = len(re.findall(r"assigned to device\s+Vulkan0\b", log_text, re.IGNORECASE))
    offloaded_layers = int(offload_match.group(1)) if offload_match else assigned_layers
    total_layers = int(offload_match.group(2)) if offload_match else None
    return {
        "has_vulkan": "loaded Vulkan backend" in log_text or "ggml_vulkan" in log_text or REQUIRED_VULKAN_TARGET in log_text,
        "has_required_device": REQUIRED_VULKAN_DEVICE in log_text or REQUIRED_VULKAN_TARGET in log_text,
        "offloaded_layers": offloaded_layers,
        "total_layers": total_layers,
        "device": REQUIRED_VULKAN_TARGET,
        "device_name": REQUIRED_VULKAN_DEVICE,
    }


def format_gpu_offload_status(log_text: str) -> str:
    status = gpu_offload_status(log_text)
    layers = status["offloaded_layers"]
    total = status["total_layers"]
    layer_text = f"{layers}/{total} layers" if total else f"{layers} layers"
    return f"{status['device']} ({status['device_name']}), offloaded {layer_text}"


def smoke_request(base_url: str, alias: str) -> str:
    client = OpenAI(base_url=base_url, api_key="llama.cpp")
    response = client.chat.completions.create(
        model=alias,
        messages=[
            {"role": "system", "content": "Return exactly one JSON object."},
            {"role": "user", "content": '{"ok": true}'},
        ],
        temperature=0,
        max_tokens=32,
    )
    return response.choices[0].message.content or ""


def run_prompt_latency(base_url: str, alias: str, out_dir: Path) -> Path:
    prompts = [
        ("perceive", "Return JSON with observations for a quiet cafe.", 80),
        ("plan", "Return JSON with a compact daily schedule for Maria.", 120),
        ("act", "Return JSON choosing a grounded action: wait, move, or speak.", 120),
        ("dialogue", "Return JSON with one short friendly message.", 80),
    ]
    client = OpenAI(base_url=base_url, api_key="llama.cpp")
    rows = []
    for kind, prompt, max_tokens in prompts:
        start = time.time()
        error = None
        content = ""
        try:
            response = client.chat.completions.create(
                model=alias,
                messages=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:
            error = repr(exc)
        rows.append(
            {
                "kind": kind,
                "latency_ms": int((time.time() - start) * 1000),
                "output_chars": len(content),
                "error": error,
            }
        )

    out_path = out_dir / "llama_cpp_prompt_latency.jsonl"
    out_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return out_path


def run_simulation(ticks: int, events_path: Path, out_dir: Path, env: dict[str, str]) -> Path:
    db_path = out_dir / "llama_cpp_benchmark.sqlite3"
    transcript_path = out_dir / "llama_cpp_simulation.log"
    agents_root = out_dir / "llama_cpp_agents"
    stdout_path = out_dir / "llama_cpp_simulation.stdout.log"
    stderr_path = out_dir / "llama_cpp_simulation.stderr.log"
    for path in [db_path, events_path, transcript_path]:
        if path.exists():
            path.unlink()
    if agents_root.exists():
        shutil.rmtree(agents_root)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.main",
            "--ticks",
            str(ticks),
            "--seed",
            "42",
            "--db",
            str(db_path),
            "--events",
            str(events_path),
            "--transcript",
            str(transcript_path),
            "--agents-root",
            str(agents_root),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Simulation benchmark failed; see {stdout_path} and {stderr_path}")
    return transcript_path


def summarize_events(events_path: Path, out_dir: Path) -> Path:
    records_by_kind: dict[str, list[dict]] = {}
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "latency_ms" in record and "kind" in record:
                records_by_kind.setdefault(record["kind"], []).append(record)

    summary = {}
    for kind, records in records_by_kind.items():
        latencies = sorted(int(record["latency_ms"]) for record in records)
        p95_index = min(len(latencies) - 1, int(len(latencies) * 0.95))
        summary[kind] = {
            "count": len(latencies),
            "min_ms": latencies[0],
            "median_ms": int(statistics.median(latencies)),
            "p95_ms": latencies[p95_index],
            "max_ms": latencies[-1],
            "fallback_count": sum(1 for record in records if record.get("parsed", {}).get("fallback")),
            "error_count": sum(1 for record in records if record.get("error")),
        }

    out_path = out_dir / "llama_cpp_event_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp GPU execution for the simulation model.")
    parser.add_argument("--llama-dir", type=Path, default=DEFAULT_LLAMA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--events", type=Path, default=DEFAULT_OUT / "llama_cpp_events.jsonl")
    parser.add_argument("--simulation-ticks", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = prepare_gguf_path(args.model, args.out_dir, args.alias)
    env = os.environ.copy()
    env["GGML_VK_VISIBLE_DEVICES"] = "0"
    env["LLAMA_CPP_URL"] = "http://127.0.0.1:8080/v1"
    env["PRIMARY_MODEL"] = args.alias
    env["CHEAP_MODEL"] = args.alias

    device_log = require_gpu_device(args.llama_dir, env)
    (args.out_dir / "llama_cpp_devices.log").write_text(device_log, encoding="utf-8")
    bench_path = run_token_benchmark(args.llama_dir, model_path, args.out_dir, env)

    server_log = args.out_dir / "llama_cpp_server.log"
    server = start_server(args.llama_dir, model_path, args.alias, server_log, env)
    try:
        log_text = wait_for_server(server_log)
        require_gpu_offload(log_text)
        smoke = smoke_request("http://127.0.0.1:8080/v1", args.alias)
        (args.out_dir / "llama_cpp_smoke.txt").write_text(smoke, encoding="utf-8")
        prompt_latency_path = run_prompt_latency("http://127.0.0.1:8080/v1", args.alias, args.out_dir)
        transcript_path = run_simulation(args.simulation_ticks, args.events, args.out_dir, env)
        event_summary_path = summarize_events(args.events, args.out_dir)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()

    print(json.dumps(
        {
            "bench": str(bench_path),
            "prompt_latency": str(prompt_latency_path),
            "simulation_transcript": str(transcript_path),
            "event_summary": str(event_summary_path),
            "server_log": str(server_log),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
