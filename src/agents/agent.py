import json

from src.agents.proposals import MoveProposal, SpeakProposal, UseObjectProposal, WaitProposal
from src.agents.personality import build_character_capsule
from src.llm.gateway import CallKind
from src.prompts import act as act_prompt
from src.prompts import plan as plan_prompt
from src.prompts import reflect as reflect_prompt
from src.prompts import score_importance as score_importance_prompt
from src.world.events import MoveEvent, ObjectStateChangeEvent, RejectedActionEvent, SpeechEvent
from src.world.context import build_action_menu, format_action_menu


class Agent:
    def __init__(
        self,
        bio: dict,
        memory,
        gateway,
        world,
        bus,
        transcript,
        reflection_threshold: int = 25,
        relationship_repo=None,
        plan_repo=None,
        agent_ids_by_name: dict[str, int] | None = None,
        agent_files=None,
    ):
        self.bio = bio
        self.memory = memory
        self.gateway = gateway
        self.world = world
        self.bus = bus
        self.transcript = transcript
        self.current_plan: dict | None = None
        self.last_reflection_tick = 0
        self.importance_accumulator = 0
        self.reflection_threshold = reflection_threshold
        self.relationship_repo = relationship_repo
        self.plan_repo = plan_repo
        self.agent_ids_by_name = agent_ids_by_name or {}
        self.agent_files = agent_files
        self._last_daily_plan_key: str | None = None
        self._last_end_of_day_key: str | None = None
        self._recent_messages: list[str] = []

    async def perceive(self, sim_tick: int, sim_time: str) -> list[dict]:
        location = self.world.agent_location(self.bio["name"])
        agents_here = [a for a in self.world.agents_at(location) if a != self.bio["name"]]
        objects_here = self.world.objects_at(location)
        recent_events = self.bus.events_at_location_for_tick(location, sim_tick - 1)
        await self._absorb_relationship_events(recent_events, sim_tick, sim_time)
        await self._refresh_prompt_context(agents_here)
        observations = self._deterministic_observations(location, agents_here, objects_here, recent_events)
        new_mems = []
        for obs in observations[:3]:
            obs = self._normalize_observation(obs)
            if not obs:
                continue
            importance = await self._score_importance(obs)
            await self.memory.add(
                self.bio["id"],
                "observation",
                obs,
                importance,
                sim_tick,
                sim_time,
                metadata={"location": location, "agents": agents_here, "kind": "observation"},
            )
            self.importance_accumulator += importance
            new_mems.append({"content": obs, "importance": importance})
            self.transcript.log(self.bio["name"], "PERCEIVE", obs)
        return new_mems

    async def _score_importance(self, observation: str) -> int:
        system, user = score_importance_prompt.build(self.bio, observation)
        result = await self.gateway.call(CallKind.SCORE_IMPORTANCE, system, user, self.bio["name"])
        try:
            return max(1, min(10, int(result.get("importance", 3))))
        except (TypeError, ValueError):
            return 3

    async def maybe_reflect(self, sim_tick: int, sim_time: str, threshold: int | None = None) -> None:
        threshold = threshold or self.reflection_threshold
        if self.importance_accumulator < threshold:
            return
        recent_mems = await self.memory.repo.get_recent(self.bio["id"], n=15)
        await self._refresh_prompt_context([])
        system, user = reflect_prompt.build(self.bio, recent_mems)
        result = await self.gateway.call(CallKind.REFLECT, system, user, self.bio["name"])
        insights = result.get("insights") or []
        for insight in insights[:3]:
            if isinstance(insight, str) and insight.strip():
                await self.memory.add(
                    self.bio["id"],
                    "reflection",
                    insight,
                    7,
                    sim_tick,
                    sim_time,
                    metadata={"kind": "reflection", "character_update": True},
                )
                self.transcript.log(self.bio["name"], "REFLECT", insight)
        knowledge = result.get("knowledge") or []
        if self.agent_files:
            self.agent_files.append_knowledge(self.bio, knowledge[:3], sim_time)
        self.importance_accumulator = 0
        self.last_reflection_tick = sim_tick

    async def maybe_plan(self, sim_tick: int, sim_time: str, force: bool = False) -> None:
        if self.current_plan and not force:
            return
        recent_reflections = await self.memory.repo.get_by_kind(self.bio["id"], "reflection", n=5)
        await self._refresh_prompt_context([])
        system, user = plan_prompt.build(
            self.bio,
            sim_time,
            recent_reflections,
            {
                "locations": list(self.world.locations.keys()),
                "agents": list(self.world.agent_positions.keys()),
                "objects": list(self.world.objects.keys()),
            },
        )
        result = await self.gateway.call(CallKind.PLAN, system, user, self.bio["name"])
        self.current_plan = result.get("schedule") or self._fallback_plan()
        if self.agent_files:
            self.agent_files.write_today(self.bio, self.current_plan, sim_time)
        if self.plan_repo:
            await self.plan_repo.add(self.bio["id"], self.current_plan, sim_tick, sim_time)
        await self.memory.add(
            self.bio["id"],
            "plan",
            json.dumps(self.current_plan),
            6,
            sim_tick,
            sim_time,
            metadata={"kind": "plan", "goals": self.bio.get("goals", [])},
        )
        self.transcript.log(self.bio["name"], "PLAN", str(self.current_plan))

    async def propose_action(self, sim_tick: int, sim_time: str):
        location = self.world.agent_location(self.bio["name"])
        agents_here = [a for a in self.world.agents_at(location) if a != self.bio["name"]]
        plan_chunk = self._current_plan_chunk(sim_time)
        query = f"What should I do now? Location: {location}. Plan: {plan_chunk}. Nearby: {', '.join(agents_here)}."
        metadata_boosts = {"location": location}
        if agents_here:
            metadata_boosts["agents"] = agents_here[0]
        memories = await self.memory.retrieve(self.bio["id"], query, sim_tick, top_k=14, metadata_boosts=metadata_boosts)
        memories = self._memories_for_action_prompt(memories)
        await self._refresh_prompt_context(agents_here)
        action_menu = build_action_menu(self.world, self.bio["name"])
        system, user = act_prompt.build(
            self.bio,
            {
                "sim_time": sim_time,
                "location": location,
                "agents_present": agents_here,
                "objects_here": self.world.objects_at(location),
                "all_locations": list(self.world.locations.keys()),
                "action_menu": format_action_menu(action_menu),
            },
            plan_chunk,
            memories,
        )
        result = await self.gateway.call(CallKind.ACT, system, user, self.bio["name"])
        proposal = self._proposal_from_dict(result, agents_here, self.world.objects_at(location), location, plan_chunk, sim_time)
        return self._avoid_repetitive_speech(proposal, objects_here=self.world.objects_at(location), location=location, plan_chunk=plan_chunk)

    async def daily_heartbeat(self, sim_tick: int, sim_time: str) -> None:
        day_key = f"{sim_tick // 48}:{sim_time}"
        if sim_time == "08:00" and self._last_daily_plan_key != day_key:
            self._last_daily_plan_key = day_key
            if self.agent_files:
                self.agent_files.reset_today(self.bio, sim_time)
            self.current_plan = None
            await self.maybe_plan(sim_tick, sim_time, force=True)

    async def end_of_day_heartbeat(self, sim_tick: int, sim_time: str) -> None:
        day_key = f"{sim_tick // 48}:{sim_time}"
        if sim_time != "22:00" or self._last_end_of_day_key == day_key:
            return
        self._last_end_of_day_key = day_key
        recent_mems = await self.memory.repo.get_recent(self.bio["id"], n=20)
        await self._refresh_prompt_context([])
        system, user = reflect_prompt.build(self.bio, recent_mems)
        result = await self.gateway.call(CallKind.REFLECT, system, user, self.bio["name"])
        knowledge = result.get("knowledge") or result.get("insights") or []
        if self.agent_files:
            self.agent_files.append_knowledge(self.bio, knowledge[:3], sim_time)

    def _proposal_from_dict(
        self,
        result: dict,
        agents_here: list[str] | None = None,
        objects_here: list[str] | None = None,
        location: str | None = None,
        plan_chunk: str = "",
        sim_time: str = "",
    ):
        action = result.get("action", "wait")
        name = self.bio["name"]
        if result.get("fallback"):
            if agents_here:
                target = agents_here[0]
                message = self._sanitize_message_for_time(f"Good morning, {target}. How is your day going?", sim_time)
                return SpeakProposal(name, target, message)
            planned_location = self._location_from_text(plan_chunk)
            if planned_location and planned_location != location:
                return MoveProposal(name, planned_location)
            if objects_here:
                affordance = self._default_affordance(objects_here[0])
                if affordance:
                    return UseObjectProposal(name, objects_here[0], affordance)
            return WaitProposal(name, "LLM fallback.")
        if action == "move_to":
            target = result.get("target", "")
            object_target = self._normalize_object_target(target, objects_here or [])
            if object_target in (objects_here or []):
                affordance = self._default_affordance(object_target)
                if affordance:
                    return UseObjectProposal(name, object_target, affordance)
            if object_target in self.world.objects:
                object_location = self.world.object_info(object_target).get("location")
                if object_location and object_location != location:
                    return MoveProposal(name, object_location)
            return MoveProposal(name, target)
        if action == "speak_to":
            target = result.get("target")
            message = self._sanitize_message_for_time(result.get("message", ""), sim_time)
            if (not target or target == name or target not in (agents_here or [])) and agents_here:
                target = agents_here[0]
            elif target not in (agents_here or []):
                if target and self.world.has_agent(target):
                    target_location = self.world.agent_location(target)
                if target_location != location:
                    return MoveProposal(name, target_location)
                return WaitProposal(name, "No nearby person to speak with.")
            message = self._sanitize_message_for_target(message, target)
            return SpeakProposal(name, target, message)
        if action == "use_object":
            target = self._normalize_object_target(result.get("target", ""), objects_here or [])
            if target not in (objects_here or []) and target in self.world.objects:
                object_location = self.world.object_info(target).get("location")
                if object_location and object_location != location:
                    return MoveProposal(name, object_location)
            interaction = self._normalize_interaction(target, result.get("interaction", ""))
            return UseObjectProposal(name, target, interaction)
        if agents_here and not result:
            message = self._sanitize_message_for_time(f"Good morning, {agents_here[0]}. How is your day going?", sim_time)
            return SpeakProposal(name, agents_here[0], message)
        return WaitProposal(name, result.get("reasoning", "No clear action."))

    def _deterministic_observations(self, location: str, agents_here: list[str], objects_here: list[str], recent_events: list) -> list[str]:
        observations = [self._fallback_observation(location, agents_here, objects_here)]
        for event in recent_events:
            if isinstance(event, SpeechEvent):
                if event.speaker == self.bio["name"]:
                    continue
                target = f" to {event.listener}" if event.listener else ""
                observations.append(f"{event.speaker} said{target}: {event.content}")
            elif isinstance(event, MoveEvent):
                observations.append(f"{event.agent} moved from {event.from_location} to {event.to_location}.")
            elif isinstance(event, ObjectStateChangeEvent):
                observations.append(f"{event.object} changed from {event.old_state} to {event.new_state}.")
            elif isinstance(event, RejectedActionEvent):
                observations.append(f"{event.agent}'s {event.proposal_type} action was rejected: {event.reason}.")
        if objects_here:
            states = [f"{obj} is {self.world.get_object_state(obj) or 'unknown'}" for obj in objects_here]
            observations.append("Nearby object states: " + "; ".join(states) + ".")
        return observations[:3]

    def _current_plan_chunk(self, sim_time: str) -> str:
        if not self.current_plan:
            return "(no plan yet)"
        hour = sim_time.split(":")[0]
        return self.current_plan.get(f"hour_{hour}", "(no specific plan for this hour)")

    def _fallback_observation(self, location: str, agents_here: list[str], objects_here: list[str]) -> str:
        others = ", ".join(agents_here) if agents_here else "no one else"
        objects = ", ".join(objects_here) if objects_here else "no notable objects"
        return f"{self.bio['name']} is at the {location} with {others}; nearby objects include {objects}."

    def _normalize_observation(self, obs) -> str:
        if isinstance(obs, str):
            return obs.strip()
        if isinstance(obs, dict):
            parts = []
            for key, value in obs.items():
                if isinstance(value, bool):
                    if value:
                        parts.append(str(key).replace("_", " "))
                    continue
                text = str(value).strip()
                if text and text.lower() not in {"true", "false", "none", "null"}:
                    parts.append(text)
            return " ".join(parts)
        return ""

    def _fallback_plan(self) -> dict:
        if self.bio["name"] == "Maria":
            return {"hour_08": "Prepare the cafe.", "hour_09": "Welcome visitors at the cafe.", "hour_10": "Visit the park for a short break."}
        if self.bio["name"] == "John":
            return {"hour_08": "Write quietly in the study room.", "hour_09": "Walk to the cafe for coffee.", "hour_10": "Talk with one friend."}
        return {"hour_08": "Study at the library.", "hour_09": "Visit the cafe and meet people.", "hour_10": "Read in the park."}

    def _location_from_text(self, text: str) -> str | None:
        text = text.lower()
        for location in self.world.locations:
            if location in text or location.replace("_", " ") in text:
                return location
        return None

    def _normalize_object_target(self, target: str, objects_here: list[str]) -> str:
        if target in objects_here:
            return target
        normalized = str(target).strip().lower().replace(" ", "_")
        for obj in list(objects_here) + list(getattr(self.world, "objects", {}).keys()):
            if normalized == obj.lower() or normalized == obj.lower().replace(" ", "_"):
                return obj
        return target

    def _normalize_interaction(self, target: str, interaction: str) -> str:
        affordances = self.world.object_affordances(target)
        if interaction in affordances:
            return interaction
        text = str(interaction).lower()
        keyword_map = [
            ("brew", "brew_coffee"),
            ("espresso", "brew_coffee"),
            ("coffee", "brew_coffee"),
            ("pastry", "serve_pastry"),
            ("croissant", "serve_pastry"),
            ("write", "write"),
            ("notebook", "write"),
            ("organize", "organize_notes"),
            ("arrange", "organize_notes"),
            ("read", "read"),
            ("browse", "browse"),
            ("sit", "sit"),
            ("rest", "rest"),
            ("observe", "observe"),
        ]
        for keyword, affordance in keyword_map:
            if keyword in text and affordance in affordances:
                return affordance
        return interaction

    def _default_affordance(self, target: str) -> str | None:
        affordances = self.world.object_affordances(target)
        for preferred in ("sit", "read", "write", "observe", "browse", "organize_notes"):
            if preferred in affordances:
                return preferred
        return affordances[0] if affordances else None

    def _sanitize_message_for_time(self, message: str, sim_time: str) -> str:
        if not sim_time:
            return message
        try:
            hour = int(sim_time.split(":", 1)[0])
        except (TypeError, ValueError):
            return message
        if hour >= 12 and "good morning" in message.lower():
            replacement = "Good afternoon" if hour < 18 else "Good evening"
            return message.replace("Good morning", replacement).replace("good morning", replacement.lower())
        if hour >= 12 and message.lower().startswith("morning"):
            replacement = "Good afternoon" if hour < 18 else "Good evening"
            return replacement + message[len("morning"):]
        return message

    def _sanitize_message_for_target(self, message: str, target: str | None) -> str:
        if not target:
            return message
        for name in self.world.agent_positions:
            if name != target and message.startswith(f"{name},"):
                return f"{target}," + message[len(name) + 1:]
        return message

    def _avoid_repetitive_speech(self, proposal, objects_here: list[str], location: str, plan_chunk: str):
        if not isinstance(proposal, SpeakProposal):
            return proposal
        message = proposal.message.strip()
        if not message:
            return proposal
        if not self._is_repetitive_message(message):
            self._remember_message(message)
            return proposal
        planned_location = self._location_from_text(plan_chunk)
        if planned_location and planned_location != location:
            return MoveProposal(self.bio["name"], planned_location)
        for obj in objects_here:
            affordance = self._default_affordance(obj)
            if affordance:
                return UseObjectProposal(self.bio["name"], obj, affordance)
        return WaitProposal(self.bio["name"], "Let the conversation breathe instead of repeating the same point.")

    def _remember_message(self, message: str) -> None:
        self._recent_messages.append(message)
        self._recent_messages = self._recent_messages[-8:]

    def _is_repetitive_message(self, message: str) -> bool:
        normalized = self._message_terms(message)
        if not normalized:
            return False
        generic = "good morning" in message.lower() and "how is your day going" in message.lower()
        topic_terms = {"silas", "gatherings", "square", "river", "spirits", "folklore"}
        topic_hits = normalized & topic_terms
        if len(topic_hits) >= 2:
            recent_topic_hits = sum(
                1 for prior in self._recent_messages[-5:] if len(self._message_terms(prior) & topic_terms) >= 2
            )
            if recent_topic_hits >= 2:
                return True
        for prior in self._recent_messages[-6:]:
            prior_terms = self._message_terms(prior)
            overlap = len(normalized & prior_terms) / max(1, len(normalized | prior_terms))
            if overlap >= 0.52:
                return True
        return generic and any("how is your day going" in prior.lower() for prior in self._recent_messages)

    def _message_terms(self, message: str) -> set[str]:
        import re

        stop = {
            "the", "and", "that", "you", "your", "about", "with", "this", "have", "just",
            "really", "think", "what", "does", "like", "good", "morning", "afternoon",
            "evening", "john", "maria", "emma",
        }
        return {word for word in re.findall(r"[a-zA-Z']{4,}", message.lower()) if word not in stop}

    def _memories_for_action_prompt(self, memories: list[dict]) -> list[dict]:
        selected = []
        dialogue_count = 0
        seen_phrases: set[str] = set()
        for memory in memories:
            content = str(memory.get("content", ""))
            is_dialogue = memory.get("kind") == "dialogue" or " said: " in content
            phrase = " ".join(content.lower().split()[:12])
            if phrase in seen_phrases:
                continue
            if is_dialogue:
                dialogue_count += 1
                if dialogue_count > 4:
                    continue
            seen_phrases.add(phrase)
            selected.append(memory)
            if len(selected) >= 10:
                break
        return selected

    async def _character_capsule(self, nearby_agents: list[str]) -> str:
        relationship_notes = []
        if self.relationship_repo and self.agent_ids_by_name:
            id_to_name = {agent_id: name for name, agent_id in self.agent_ids_by_name.items()}
            notes = await self.relationship_repo.notes_for_agent(self.bio["id"], id_to_name)
            relationship_notes = [
                note for note in notes if any(note.startswith(f"{agent}:") for agent in nearby_agents)
            ]
        return build_character_capsule(self.bio, relationship_notes)

    async def _refresh_prompt_context(self, nearby_agents: list[str]) -> None:
        self.bio["character_capsule"] = await self._character_capsule(nearby_agents)
        if self.agent_files:
            files = self.agent_files.load_context(self.bio)
            self.bio["file_context"] = (
                f"SOUL.md:\n{files['soul']}\n\n"
                f"KNOWLEDGE.md:\n{files['knowledge']}\n\n"
                f"TODAY.md:\n{files['today']}"
            )
        else:
            self.bio["file_context"] = "No live agent files loaded."

    async def _absorb_relationship_events(self, recent_events: list, sim_tick: int, sim_time: str) -> None:
        if not self.relationship_repo or not self.agent_ids_by_name:
            return
        for event in recent_events:
            speaker = getattr(event, "speaker", None)
            if not speaker or speaker == self.bio["name"]:
                continue
            speaker_id = self.agent_ids_by_name.get(speaker)
            if not speaker_id:
                continue
            await self.relationship_repo.upsert(
                self.bio["id"],
                speaker_id,
                affinity_delta=0.1,
                trust_delta=0.05,
                familiarity_delta=0.2,
                summary=f"Recently heard {speaker} say: {getattr(event, 'content', '')[:120]}",
                metadata={"last_shared_tick": sim_tick, "last_location": getattr(event, "location", "")},
            )
            await self.memory.add(
                self.bio["id"],
                "dialogue",
                f"{speaker} said: {getattr(event, 'content', '')}",
                5,
                sim_tick,
                sim_time,
                metadata={"kind": "dialogue", "agents": [speaker], "location": getattr(event, "location", "")},
            )
