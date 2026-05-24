from dataclasses import asdict, is_dataclass
from pathlib import Path


class Transcript:
    def __init__(self, path: str, echo: bool = True):
        self.path = Path(path)
        self.echo = echo
        self.path.write_text("", encoding="utf-8")

    def section(self, title: str) -> None:
        self._write(f"\n=== {title} ===")

    def log(self, agent: str, kind: str, content: str) -> None:
        self._write(f"[{agent}] {kind}: {content}")

    def log_event(self, event) -> None:
        data = asdict(event) if is_dataclass(event) else dict(event)
        if event.__class__.__name__ == "SpeechEvent":
            target = data.get("listener") or "everyone nearby"
            self._write(f"EVENT: {data['speaker']} says to {target}: \"{data['content']}\"")
        elif event.__class__.__name__ == "MoveEvent":
            self._write(f"EVENT: {data['agent']} moves {data['from_location']} -> {data['to_location']}")
        else:
            self._write(f"EVENT: {event.__class__.__name__} {data}")

    def _write(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.echo:
            print(line)
