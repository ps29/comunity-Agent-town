import time
from enum import Enum

from src.llm.parsing import parse_json_forgiving


class CallKind(str, Enum):
    PERCEIVE = "perceive"
    SCORE_IMPORTANCE = "score_importance"
    REFLECT = "reflect"
    PLAN = "plan"
    ACT = "act"
    DIALOGUE = "dialogue"


class LLMGateway:
    def __init__(self, primary, cheap, event_log, use_json_mode: bool = True):
        self.primary = primary
        self.cheap = cheap
        self.event_log = event_log
        self.use_json_mode = use_json_mode
        self.routing = {
            CallKind.PERCEIVE: (cheap, 0.0, 180),
            CallKind.SCORE_IMPORTANCE: (cheap, 0.0, 80),
            CallKind.REFLECT: (primary, 0.7, 260),
            CallKind.PLAN: (primary, 0.3, 520),
            CallKind.ACT: (primary, 0.5, 320),
            CallKind.DIALOGUE: (primary, 0.8, 180),
        }

    async def call(self, kind: CallKind, system: str, user: str, agent_name: str) -> dict:
        client, temp, max_tok = self.routing[kind]
        start = time.time()
        raw = ""
        parsed = {}
        error = None
        parse_failed = False
        retry_used = False
        fallback_reason = None
        try:
            first_raw = await client.complete(
                system,
                user,
                temperature=temp,
                max_tokens=max_tok,
                response_format={"type": "json_object"} if self.use_json_mode else None,
            )
            raw = first_raw
            parsed = parse_json_forgiving(raw)
            if not parsed and self.use_json_mode:
                parse_failed = True
                retry_used = True
                retry_raw = await client.complete(
                    system,
                    user + "\n\nReturn the JSON object now. /no_think",
                    temperature=0.0,
                    max_tokens=max_tok,
                    response_format=None,
                )
                parsed = parse_json_forgiving(retry_raw) or parse_json_forgiving(first_raw)
                raw = retry_raw or first_raw
            if not parsed:
                parse_failed = True
                fallback_reason = "parse_failed"
                parsed = self._fallback_for_kind(kind)
        except Exception as exc:
            error = repr(exc)
            fallback_reason = "client_error"
            parsed = self._fallback_for_kind(kind)
            parsed["llm_error"] = error
        latency = time.time() - start
        await self.event_log.write(
            {
                "kind": kind.value,
                "agent": agent_name,
                "system": system,
                "user": user,
                "system_chars": len(system),
                "user_chars": len(user),
                "prompt_chars": len(system) + len(user),
                "raw_response": raw,
                "parsed": parsed,
                "fallback": bool(parsed.get("fallback")),
                "parse_failed": parse_failed,
                "retry_used": retry_used,
                "client_error": error,
                "fallback_reason": fallback_reason,
                "llm_error": parsed.get("llm_error"),
                "error": error,
                "latency_ms": int(latency * 1000),
            }
        )
        return parsed

    def _fallback_for_kind(self, kind: CallKind) -> dict:
        if kind == CallKind.PERCEIVE:
            return {"observations": [], "fallback": True}
        if kind == CallKind.SCORE_IMPORTANCE:
            return {"importance": 3, "fallback": True}
        if kind == CallKind.REFLECT:
            return {"insights": [], "fallback": True}
        if kind == CallKind.PLAN:
            return {"schedule": {}, "fallback": True}
        if kind == CallKind.ACT:
            return {"action": "wait", "target": "", "message": "", "interaction": "", "reasoning": "LLM fallback.", "fallback": True}
        if kind == CallKind.DIALOGUE:
            return {"message": "", "fallback": True}
        return {"fallback": True}
