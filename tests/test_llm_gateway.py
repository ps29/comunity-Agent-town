import pytest

from src.llm.gateway import CallKind, LLMGateway
from src.llm.parsing import parse_json_forgiving


def test_parse_json_forgiving_variants():
    assert parse_json_forgiving('{"ok": true}') == {"ok": True}
    assert parse_json_forgiving('```json\n{"ok": true}\n```') == {"ok": True}
    assert parse_json_forgiving('Here: {"ok": {"nested": 1}} thanks') == {"ok": {"nested": 1}}
    assert parse_json_forgiving("nope", {"fallback": 1}) == {"fallback": 1}


class FakeClient:
    async def complete(self, *args, **kwargs):
        return '{"observations":["A clean observation."]}'


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
