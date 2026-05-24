import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path


class EventLog:
    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def write(self, record: dict) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True) if self.path.parent != Path(".") else None
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
