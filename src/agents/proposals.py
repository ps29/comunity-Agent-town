from dataclasses import dataclass


@dataclass
class MoveProposal:
    agent: str
    target_location: str


@dataclass
class SpeakProposal:
    agent: str
    target: str | None
    message: str


@dataclass
class WaitProposal:
    agent: str
    reason: str = ""


@dataclass
class UseObjectProposal:
    agent: str
    object: str
    interaction: str


ActionProposal = MoveProposal | SpeakProposal | WaitProposal | UseObjectProposal
