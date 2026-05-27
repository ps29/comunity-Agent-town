import json
import re
from typing import Any


def parse_json_forgiving(text: str | None, fallback: dict | None = None) -> dict:
    fallback = fallback or {}
    if not text:
        return fallback

    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else fallback
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            return parsed if isinstance(parsed, dict) else fallback
        except json.JSONDecodeError:
            pass

    candidate = _first_balanced_object(cleaned)
    if candidate:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else fallback
        except json.JSONDecodeError:
            pass

    partial = _partial_object_fields(cleaned)
    if partial:
        return partial

    return fallback


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _partial_object_fields(text: str) -> dict:
    start = text.find("{")
    if start >= 0:
        text = text[start:]
    fields = {}
    for key, value in re.findall(r'"([^"{}:,]+)"\s*:\s*"((?:\\.|[^"\\])*)"', text):
        try:
            fields[key] = json.loads(f'"{value}"')
        except json.JSONDecodeError:
            fields[key] = value
    return fields
