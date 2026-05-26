import pytest

from src.llm.gateway import CallKind, LLMGateway
from src.llm.parsing import parse_json_forgiving


def test_parse_json_forgiving_variants():
    assert parse_json_forgiving('{"ok": true}') == {"ok": True}
    assert parse_json_forgiving('```json\n{"ok": true}\n```') == {"ok": True}
    assert parse_json_forgiving('Here: {"ok": {"nested": 1}} thanks') == {"ok": {"nested": 1}}
    assert parse_json_forgiving('{"action":"use_object","target":"coffee_maker","reasoning":"The') == {
        "action": "use_object",
        "target": "coffee_maker",
    }
    assert parse_json_forgiving("nope", {"fallback": 1}) == {"fallback": 1}


class FakeClient:
    async def complete(self, *args, **kwargs):
        return '{"observations":["A clean observation."]}'


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeLog:
    def __init__(self):
        self.records = []

    async def write(self, record):
        self.records.append(record)


@pytest.mark.asyncio
async def test_gateway_routes_and_logs():
    log = FakeLog()
    gateway = LLMGateway(FakeClient(), FakeClient(), log)
    result = await gateway.call(CallKind.PERCEIVE, "system", "user", "Maria")
    assert result["observations"] == ["A clean observation."]
    assert log.records[0]["kind"] == "perceive"
    assert log.records[0]["parsed"] == result
    assert log.records[0]["system_chars"] == len("system")
    assert log.records[0]["user_chars"] == len("user")
    assert log.records[0]["prompt_chars"] == len("systemuser")
    assert log.records[0]["fallback"] is False
    assert log.records[0]["llm_error"] is None
    assert log.records[0]["parse_failed"] is False
    assert log.records[0]["retry_used"] is False
    assert log.records[0]["client_error"] is None
    assert log.records[0]["fallback_reason"] is None


@pytest.mark.asyncio
async def test_gateway_logs_retry_success_after_malformed_json():
    log = FakeLog()
    client = SequenceClient(["not json", '{"observations":["Recovered."]}'])
    gateway = LLMGateway(client, client, log)

    result = await gateway.call(CallKind.PERCEIVE, "system", "user", "Maria")

    assert result == {"observations": ["Recovered."]}
    assert len(client.calls) == 2
    assert log.records[0]["parse_failed"] is True
    assert log.records[0]["retry_used"] is True
    assert log.records[0]["fallback"] is False
    assert log.records[0]["fallback_reason"] is None


@pytest.mark.asyncio
async def test_gateway_logs_parse_fallback_after_retry_failure():
    log = FakeLog()
    client = SequenceClient(["not json", "still not json"])
    gateway = LLMGateway(client, client, log)

    result = await gateway.call(CallKind.PERCEIVE, "system", "user", "Maria")

    assert result["fallback"] is True
    assert result["observations"] == []
    assert log.records[0]["parse_failed"] is True
    assert log.records[0]["retry_used"] is True
    assert log.records[0]["fallback_reason"] == "parse_failed"
    assert log.records[0]["client_error"] is None


@pytest.mark.asyncio
async def test_gateway_logs_client_error_fallback():
    log = FakeLog()
    client = SequenceClient([RuntimeError("server unavailable")])
    gateway = LLMGateway(client, client, log)

    result = await gateway.call(CallKind.PERCEIVE, "system", "user", "Maria")

    assert result["fallback"] is True
    assert "RuntimeError" in result["llm_error"]
    assert log.records[0]["parse_failed"] is False
    assert log.records[0]["retry_used"] is False
    assert "RuntimeError" in log.records[0]["client_error"]
    assert log.records[0]["fallback_reason"] == "client_error"
