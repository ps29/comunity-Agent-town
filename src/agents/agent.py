import json
import re

from src.agents.proposals import MoveProposal, SpeakProposal, UseObjectProposal, WaitProposal
from src.agents.personality import build_character_capsule
from src.cognition.needs import AgentNeeds
from src.llm.gateway import CallKind
from src.prompts import act as act_prompt
from src.prompts import plan as plan_prompt
from src.prompts import reflect as reflect_prompt
from src.world.events import MoveEvent, ObjectStateChangeEvent, RejectedActionEvent, SpeechEvent
from src.world.context import build_action_menu, format_action_menu


ACTION_PROMPT_CHAR_BUDGET = 12000
MEMORY_TEXT_LIMIT = 240


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
        agent_repo=None,
        agent_ids_by_name: dict[str, int] | None = None,
        agent_files=None,
        needs: AgentNeeds | None = None,
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
        self.agent_repo = agent_repo
        self.agent_ids_by_name = agent_ids_by_name or {}
        self.agent_files = agent_files
        self.needs = needs or AgentNeeds.from_dict(bio.get("needs"))
        self._last_daily_plan_key: str | None = None
        self._last_end_of_day_key: str | None = None
        self._recent_messages: list[str] = []
        self._recent_reflections: list[str] = []

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
        text = observation.lower()
        score = 3
        if "said" in text or "moved from" in text:
            score += 2
        if "rejected" in text or "changed from" in text:
            score += 1
        if any(name.lower() in text for name in self.world.agent_positions if name != self.bio["name"]):
            score += 1
        if any(term in text for term in ("goal", "plan", "promise", "argument", "help", "research", "writing")):
            score += 1
        return max(1, min(10, score))

    async def maybe_reflect(self, sim_tick: int, sim_time: str, threshold: int | None = None) -> None:
        threshold = threshold or self.reflection_threshold
        if self.importance_accumulator < threshold:
            return
        recent_mems = await self.memory.repo.get_recent(self.bio["id"], n=15)
        await self._refresh_prompt_context([], purpose="reflect")
        recent_mems = self._compact_memories(recent_mems, limit=10, text_limit=MEMORY_TEXT_LIMIT)
        system, user = reflect_prompt.build(self.bio, recent_mems)
        result = await self.gateway.call(CallKind.REFLECT, system, user, self.bio["name"])
        insights = result.get("insights") or []
        for insight in insights[:3]:
            if isinstance(insight, str) and insight.strip() and self._is_novel_reflection(insight):
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
                self._remember_reflection(insight)
        knowledge = result.get("knowledge") or []
        if self.agent_files:
            self.agent_files.append_knowledge(self.bio, self._novel_lines(knowledge[:3]), sim_time)
        self.importance_accumulator = 0
        self.last_reflection_tick = sim_tick

    async def maybe_plan(self, sim_tick: int, sim_time: str, force: bool = False) -> None:
        if self.current_plan and not force:
            return
        recent_reflections = await self.memory.repo.get_by_kind(self.bio["id"], "reflection", n=5)
        await self._refresh_prompt_context([], purpose="plan")
        system, user = plan_prompt.build(
            self.bio,
            sim_time,
            recent_reflections,
            {
                "locations": list(self.world.locations.keys()),
                "location_descriptions": {
                    name: info.get("description", "")
                    for name, info in self.world.locations.items()
                },
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
        self.needs.decay_tick()
        await self._persist_needs()
        location = self.world.agent_location(self.bio["name"])
        agents_here = [a for a in self.world.agents_at(location) if a != self.bio["name"]]
        plan_chunk = self._current_plan_chunk(sim_time)
        query = f"What should I do now? Location: {location}. Plan: {plan_chunk}. Nearby: {', '.join(agents_here)}."
        metadata_boosts = {"location": location}
        if agents_here:
            metadata_boosts["agents"] = agents_here[0]
        memories = await self.memory.retrieve(self.bio["id"], query, sim_tick, top_k=14, metadata_boosts=metadata_boosts)
        memories = self._memories_for_action_prompt(memories)
        await self._refresh_prompt_context(agents_here, purpose="act")
        action_menu = build_action_menu(self.world, self.bio["name"])
        system, user, memories = self._build_budgeted_action_prompt(
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

    def _build_budgeted_action_prompt(self, world_context: dict, plan_chunk: str, memories: list[dict]) -> tuple[str, str, list[dict]]:
        kept = list(memories)
        system, user = act_prompt.build(
            self.bio,
            world_context,
            plan_chunk,
            kept,
            self.needs.status_string(),
        )
        while len(system) + len(user) > ACTION_PROMPT_CHAR_BUDGET and kept:
            kept.pop()
            system, user = act_prompt.build(self.bio, world_context, plan_chunk, kept, self.needs.status_string())
        if len(system) + len(user) > ACTION_PROMPT_CHAR_BUDGET:
            self.bio["file_context"] = self._truncate_text(self.bio.get("file_context", ""), 2200)
            system, user = act_prompt.build(self.bio, world_context, plan_chunk, kept[:3], self.needs.status_string())
        return system, user, kept

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
        await self._refresh_prompt_context([], purpose="reflect")
        recent_mems = self._compact_memories(recent_mems, limit=12, text_limit=MEMORY_TEXT_LIMIT)
        system, user = reflect_prompt.build(self.bio, recent_mems)
        result = await self.gateway.call(CallKind.REFLECT, system, user, self.bio["name"])
        knowledge = result.get("knowledge") or result.get("insights") or []
        if self.agent_files:
            self.agent_files.append_knowledge(self.bio, self._novel_lines(knowledge[:3]), sim_time)
        await self._consolidate_semantic_memories(recent_mems, knowledge, sim_tick)

    async def note_resolved_events(self, events: list) -> None:
        changed = False
        for event in events:
            if isinstance(event, SpeechEvent) and event.speaker == self.bio["name"]:
                self.needs.socialize()
                changed = True
            elif isinstance(event, ObjectStateChangeEvent):
                obj = event.object
                if obj in {"coffee_maker", "pastry_case"} and self.world.agent_location(self.bio["name"]) == event.location:
                    self.needs.satisfy_hunger(0.2)
                    changed = True
                if event.new_state == "occupied" and obj in {"bench", "reading_chair", "corner_table"}:
                    self.needs.rest(0.15)
                    changed = True
        if changed:
            await self._persist_needs()

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
            urgent = self._fallback_for_needs(objects_here or [], agents_here or [], location or "", sim_time)
            if urgent:
                return urgent
            planned_location = self._location_from_text(plan_chunk)
            if planned_location and planned_location != location:
                return MoveProposal(name, planned_location)
            if objects_here:
                target = self._best_fallback_object(objects_here)
                affordance = self._default_affordance(target)
                if affordance:
                    return UseObjectProposal(name, target, affordance)
            if agents_here:
                message = self._non_generic_fallback_message(agents_here[0], sim_time)
                if message:
                    return SpeakProposal(name, agents_here[0], message)
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
            if target == location:
                return WaitProposal(name, "Already at the planned location.")
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
            if not message.strip():
                message = self._fallback_message_for_target(target, sim_time)
            message = self._sanitize_message_for_target(message, target)
            return SpeakProposal(name, target, message)
        if action == "use_object":
            target = self._normalize_object_target(
                result.get("target", ""),
                objects_here or [],
                result.get("interaction", ""),
            )
            if target not in (objects_here or []) and target in self.world.objects:
                object_location = self.world.object_info(target).get("location")
                if object_location and object_location != location:
                    return MoveProposal(name, object_location)
            interaction = self._normalize_interaction(target, result.get("interaction", ""))
            if not interaction.strip():
                interaction = self._default_affordance(target) or ""
            return UseObjectProposal(name, target, interaction)
        if agents_here and not result:
            message = self._non_generic_fallback_message(agents_here[0], sim_time)
            if message:
                return SpeakProposal(name, agents_here[0], message)
        return WaitProposal(name, result.get("reasoning", "No clear action."))

    def _deterministic_observations(self, location: str, agents_here: list[str], objects_here: list[str], recent_events: list) -> list[str]:
        observations = [self._fallback_observation(location, agents_here, objects_here)]
        for event in recent_events:
            if isinstance(event, SpeechEvent):
                if event.speaker == self.bio["name"]:
                    continue
                self._remember_message(event.content)
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
        return (
            self.current_plan.get(sim_time)
            or self.current_plan.get(f"hour_{hour}")
            or self.current_plan.get(hour)
            or "(no specific plan for this hour)"
        )

    def _fallback_observation(self, location: str, agents_here: list[str], objects_here: list[str]) -> str:
        others = ", ".join(agents_here) if agents_here else "no one else"
        objects = ", ".join(objects_here) if objects_here else "no notable objects"
        return f"I am at the {location} with {others}; nearby objects include {objects}."

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
            return {
                "hour_08": "Prepare the cafe.",
                "hour_09": "Read the notice_board at the town_square.",
                "hour_10": "Browse market_crates at the market_stalls.",
                "hour_11": "Host a brief check-in at the community_hall.",
            }
        if self.bio["name"] == "John":
            return {
                "hour_08": "Write at the typewriter in the newspaper_office.",
                "hour_09": "Review notes on the clippings_wall.",
                "hour_10": "Walk the riverside_path and observe the river_marker.",
                "hour_11": "Visit the cafe for one conversation.",
            }
        return {
            "hour_08": "Study at the library.",
            "hour_09": "Search archive_boxes in the archive_room.",
            "hour_10": "Study the map_table for old place names.",
            "hour_11": "Inspect the old_mill_wheel at the old_mill.",
        }

    def _location_from_text(self, text: str) -> str | None:
        text = text.lower()
        for location in self.world.locations:
            if location in text or location.replace("_", " ") in text:
                return location
        return None

    def _normalize_object_target(self, target: str, objects_here: list[str], interaction: str = "") -> str:
        interaction_text = str(interaction).lower()
        if any(term in interaction_text for term in ("sit", "rest")):
            affordances = self.world.object_affordances(target)
            if "sit" not in affordances and "rest" not in affordances:
                for candidate in ("reading_chair", "willow_bench", "bench", "corner_table"):
                    if candidate in objects_here:
                        return candidate
        if target in objects_here:
            return target
        normalized = str(target).strip().lower().replace(" ", "_")
        alias_map = {
            "notebook": "desk",
            "notes": "desk",
            "pen": "desk",
            "pens": "desk",
            "writing_tools": "desk",
            "article": "typewriter",
            "story": "typewriter",
            "draft": "typewriter",
            "typewriter": "typewriter",
            "clippings": "clippings_wall",
            "clipping": "clippings_wall",
            "newspaper": "clippings_wall",
            "reporter_notes": "reporter_notebook",
            "reporter_notebook": "reporter_notebook",
            "book": "bookshelf",
            "books": "bookshelf",
            "shelf": "local_history_shelf",
            "local_history": "local_history_shelf",
            "local_history_shelf": "local_history_shelf",
            "archive": "archive_boxes",
            "archives": "archive_boxes",
            "records": "archive_boxes",
            "record_boxes": "archive_boxes",
            "boxes": "archive_boxes",
            "ledger": "old_ledger",
            "old_ledger": "old_ledger",
            "map": "map_table",
            "maps": "map_table",
            "map_table": "map_table",
            "notice": "notice_board",
            "notice_board": "notice_board",
            "notices": "notice_board",
            "calendar": "event_calendar",
            "event_calendar": "event_calendar",
            "lost_and_found": "lost_and_found_box",
            "lost_and_found_box": "lost_and_found_box",
            "meeting": "meeting_table",
            "meeting_table": "meeting_table",
            "mill": "old_mill_wheel",
            "mill_wheel": "old_mill_wheel",
            "old_mill_wheel": "old_mill_wheel",
            "mill_door": "mill_door",
            "door": "mill_door",
            "grain": "grain_sacks",
            "grain_sacks": "grain_sacks",
            "river_marker": "river_marker",
            "marker": "river_marker",
            "willow_bench": "willow_bench",
            "market": "market_crates",
            "market_crates": "market_crates",
            "crates": "market_crates",
            "flower_stall": "flower_stall",
            "flowers": "flower_stall",
            "coffee": "coffee_maker",
            "espresso": "coffee_maker",
            "latte": "coffee_maker",
            "pastry": "pastry_case",
            "croissant": "pastry_case",
            "chair": "reading_chair",
            "table": "corner_table",
        }
        if normalized in alias_map and alias_map[normalized] in objects_here:
            return alias_map[normalized]
        if "write" in interaction_text or "notebook" in interaction_text:
            for candidate in ("typewriter", "reporter_notebook", "desk", "study_table"):
                if candidate in objects_here:
                    return candidate
        if any(term in interaction_text for term in ("notice", "posted", "calendar")):
            for candidate in ("notice_board", "event_calendar"):
                if candidate in objects_here:
                    return candidate
        if any(term in interaction_text for term in ("archive", "record", "ledger")):
            for candidate in ("archive_boxes", "old_ledger", "local_history_shelf", "clippings_wall"):
                if candidate in objects_here:
                    return candidate
        if "map" in interaction_text:
            for candidate in ("map_table", "town_map"):
                if candidate in objects_here:
                    return candidate
        if any(term in interaction_text for term in ("mill", "wheel")):
            for candidate in ("old_mill_wheel", "mill_door", "grain_sacks"):
                if candidate in objects_here:
                    return candidate
        if any(term in interaction_text for term in ("market", "crate", "stall", "flower")):
            for candidate in ("market_crates", "flower_stall"):
                if candidate in objects_here:
                    return candidate
        if "read" in interaction_text:
            for candidate in ("local_history_shelf", "bookshelf", "reading_chair", "notice_board", "event_calendar"):
                if candidate in objects_here:
                    return candidate
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
            ("notice", "read_notice"),
            ("posted", "read_notice"),
            ("calendar", "read_notice"),
            ("post", "post_notice"),
            ("archive", "search_records"),
            ("record", "search_records"),
            ("consult", "search_records"),
            ("ledger", "review_notes"),
            ("clipping", "review_notes"),
            ("open", "review_notes"),
            ("consult", "review_notes"),
            ("examine", "review_notes"),
            ("examine", "inspect"),
            ("examine", "observe"),
            ("notes", "review_notes"),
            ("review", "review_notes"),
            ("map", "study_map"),
            ("mill", "inspect"),
            ("wheel", "inspect"),
            ("door", "inspect"),
            ("grain", "inspect"),
            ("inspect", "inspect"),
            ("market", "browse_market"),
            ("crate", "browse_market"),
            ("stall", "browse_market"),
            ("meeting", "host_meeting"),
            ("host", "host_meeting"),
            ("article", "write_article"),
            ("newspaper", "write_article"),
            ("write", "write"),
            ("notebook", "write"),
            ("organize", "organize_notes"),
            ("arrange", "organize_notes"),
            ("sort", "organize_notes"),
            ("outline", "organize_notes"),
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
        for preferred in (
            "read_notice",
            "search_records",
            "study_map",
            "inspect",
            "browse_market",
            "write_article",
            "review_notes",
            "host_meeting",
            "serve_coffee",
            "brew_coffee",
            "sit",
            "write",
            "read",
            "browse",
            "organize_notes",
            "serve_pastry",
            "check_pastries",
            "observe",
        ):
            if preferred in affordances:
                return preferred
        return affordances[0] if affordances else None

    def _sanitize_message_for_time(self, message: str, sim_time: str) -> str:
        message = self._strip_json_tail(message)
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

    def _strip_json_tail(self, message: str) -> str:
        markers = [
            '", "interaction"',
            '", "reasoning"',
            '”, “interaction”',
            '”, “reasoning”',
            ',"interaction"',
            ', “interaction”',
        ]
        for marker in markers:
            idx = message.find(marker)
            if idx >= 0:
                message = message[:idx]
        return message.strip().strip('"').strip()

    def _sanitize_message_for_target(self, message: str, target: str | None) -> str:
        if not target:
            return message
        for name in self.world.agent_positions:
            if name != target and message.startswith(f"{name},"):
                return f"{target}," + message[len(name) + 1:]
        return message

    def _fallback_message_for_target(self, target: str | None, sim_time: str) -> str:
        if not target:
            return "I am nearby and paying attention."
        message = f"Hello, {target}. How is your day going?"
        return self._sanitize_message_for_time(message, sim_time)

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
        generic = "how is your day going" in message.lower()
        topic_terms = {
            "silas", "gatherings", "square", "river", "spirits", "folklore", "henderson", "sketches", "croissant",
            "higgins", "mill", "miller", "daughter", "espresso", "artwork", "david", "history", "thesis",
        }
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
        stop = {
            "the", "and", "that", "you", "your", "about", "with", "this", "have", "just",
            "really", "think", "what", "does", "like", "good", "morning", "afternoon",
            "evening", "john", "maria", "emma",
        }
        return {word for word in re.findall(r"[a-zA-Z']{4,}", message.lower()) if word not in stop}

    def _memories_for_action_prompt(self, memories: list[dict]) -> list[dict]:
        selected = []
        dialogue_count = 0
        reflection_count = 0
        seen_phrases: set[str] = set()
        for memory in memories:
            if memory.get("kind") == "plan":
                continue
            content = str(memory.get("content", ""))
            if self._looks_like_raw_plan(content):
                continue
            is_dialogue = memory.get("kind") == "dialogue" or " said: " in content
            is_reflection = memory.get("kind") == "reflection"
            normalized = self._normalize_for_dedupe(content)
            phrase = " ".join(normalized.split()[:16])
            if phrase in seen_phrases:
                continue
            if is_dialogue:
                dialogue_count += 1
                if dialogue_count > 2:
                    continue
            if is_reflection:
                reflection_count += 1
                if reflection_count > 2:
                    continue
            seen_phrases.add(phrase)
            compact = dict(memory)
            compact["content"] = self._truncate_text(content, MEMORY_TEXT_LIMIT)
            selected.append(compact)
            if len(selected) >= 6:
                break
        return selected

    def _compact_memories(self, memories: list[dict], limit: int, text_limit: int) -> list[dict]:
        compact = []
        seen = set()
        for memory in memories:
            content = str(memory.get("content", ""))
            if self._looks_like_raw_plan(content):
                continue
            key = self._normalize_for_dedupe(content)
            if not key or key in seen:
                continue
            seen.add(key)
            row = dict(memory)
            row["content"] = self._truncate_text(content, text_limit)
            compact.append(row)
            if len(compact) >= limit:
                break
        return compact

    async def _character_capsule(self, nearby_agents: list[str]) -> str:
        relationship_notes = []
        if self.relationship_repo and self.agent_ids_by_name:
            id_to_name = {agent_id: name for name, agent_id in self.agent_ids_by_name.items()}
            notes = await self.relationship_repo.notes_for_agent(self.bio["id"], id_to_name)
            relationship_notes = [
                note for note in notes if any(note.startswith(f"{agent}:") for agent in nearby_agents)
            ]
        return build_character_capsule(self.bio, relationship_notes)

    async def _refresh_prompt_context(self, nearby_agents: list[str], purpose: str = "act") -> None:
        self.bio["character_capsule"] = await self._character_capsule(nearby_agents)
        if self.agent_files:
            files = self.agent_files.load_context(self.bio)
            knowledge = self._ground_agent_file_text(files["knowledge"], max_lines=5 if purpose == "act" else 12)
            today = self._ground_agent_file_text(files["today"], max_lines=4 if purpose == "act" else 8)
            semantic = await self.memory.get_semantic(self.bio["id"], min_confidence=0.2)
            semantic_limit = 6 if purpose == "act" else 10
            semantic_lines = "\n".join(
                self._truncate_text(f"- {row['subject']}: {row['fact']}", MEMORY_TEXT_LIMIT)
                for row in self._dedupe_semantic_rows(semantic)[:semantic_limit]
            ) or "- no durable semantic facts yet"
            if purpose == "act":
                self.bio["file_context"] = (
                    f"Knowledge:\n{knowledge}\n\n"
                    f"Today:\n{today}\n\n"
                    f"Semantic memory:\n{semantic_lines}"
                )
            else:
                soul = self._compact_soul(files["soul"])
                self.bio["file_context"] = (
                    f"SOUL.md:\n{soul}\n\n"
                    f"KNOWLEDGE.md (grounded lines only):\n{knowledge}\n\n"
                    f"TODAY.md (grounded lines only):\n{today}\n\n"
                    f"Semantic memory:\n{semantic_lines}"
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
            if not await self._has_similar_recent_dialogue(getattr(event, "content", "")):
                await self.memory.add(
                    self.bio["id"],
                    "dialogue",
                    f"{speaker} said: {getattr(event, 'content', '')}",
                    5,
                    sim_tick,
                    sim_time,
                    metadata={"kind": "dialogue", "agents": [speaker], "location": getattr(event, "location", "")},
                )

    async def _persist_needs(self) -> None:
        if self.agent_repo:
            await self.agent_repo.update_needs(self.bio["id"], self.needs.to_dict())

    async def _consolidate_semantic_memories(self, recent_mems: list[dict], knowledge: list, sim_tick: int) -> None:
        source_ids = [int(memory["id"]) for memory in recent_mems if "id" in memory]
        for fact in knowledge[:5]:
            if not isinstance(fact, str) or not fact.strip():
                continue
            if not self._is_novel_fact(fact):
                continue
            subject = self._semantic_subject(fact)
            await self.memory.add_semantic(
                self.bio["id"],
                subject,
                fact.strip(),
                0.65,
                source_ids,
                sim_tick,
            )
        await self.memory.mark_consolidated(source_ids)

    def _semantic_subject(self, fact: str) -> str:
        lowered = fact.lower()
        for name in self.world.agent_positions:
            if name.lower() in lowered:
                return name
        for location in self.world.locations:
            if location.lower() in lowered or location.replace("_", " ") in lowered:
                return location
        return self.bio["name"]

    def _ground_agent_file_text(self, text: str, max_lines: int = 12) -> str:
        blocked_terms = {
            "artisan",
            "back room",
            "bakery",
            "henderson",
            "silas",
            "sketches",
        }
        kept = []
        for line in text.splitlines():
            lowered = line.lower()
            if any(term in lowered for term in blocked_terms):
                continue
            if not line.strip() or line.strip().startswith("#"):
                continue
            if line.strip().startswith("##"):
                continue
            kept.append(self._truncate_text(line, MEMORY_TEXT_LIMIT))
            if len(kept) >= max_lines:
                break
        grounded = "\n".join(kept).strip()
        return grounded or "# No grounded knowledge lines available yet."

    def _fallback_for_needs(self, objects_here: list[str], agents_here: list[str], location: str, sim_time: str):
        if self.needs.hunger > self.needs.hunger_threshold:
            for target, affordance in (("pastry_case", "serve_pastry"), ("coffee_maker", "serve_coffee")):
                if target in objects_here and affordance in self.world.object_affordances(target):
                    return UseObjectProposal(self.bio["name"], target, affordance)
        if self.needs.energy < self.needs.tiredness_threshold:
            for target in ("bench", "reading_chair", "corner_table"):
                if target in objects_here:
                    affordance = "rest" if "rest" in self.world.object_affordances(target) else "sit"
                    return UseObjectProposal(self.bio["name"], target, affordance)
        if self.needs.social_satiety < self.needs.loneliness_threshold and agents_here:
            message = self._non_generic_fallback_message(agents_here[0], sim_time)
            if message:
                return SpeakProposal(self.bio["name"], agents_here[0], message)
        return None

    def _best_fallback_object(self, objects_here: list[str]) -> str:
        preferred = (
            "notice_board",
            "archive_boxes",
            "map_table",
            "old_mill_wheel",
            "river_marker",
            "market_crates",
            "typewriter",
            "clippings_wall",
            "meeting_table",
            "coffee_maker",
            "corner_table",
            "study_table",
            "local_history_shelf",
            "bench",
            "reading_chair",
            "pastry_case",
            "pond",
        )
        for obj in preferred:
            if obj in objects_here:
                return obj
        return objects_here[0]

    def _non_generic_fallback_message(self, target: str, sim_time: str) -> str:
        if any("how is your day going" in msg.lower() for msg in self._recent_messages[-8:]):
            return ""
        hour_text = self._time_greeting(sim_time)
        message = f"{hour_text}, {target}. I was noticing what you were focused on here."
        if self._is_repetitive_message(message):
            return ""
        return message

    def _time_greeting(self, sim_time: str) -> str:
        try:
            hour = int(sim_time.split(":", 1)[0])
        except (TypeError, ValueError):
            return "Hello"
        if hour < 12:
            return "Good morning"
        if hour < 18:
            return "Good afternoon"
        return "Good evening"

    def _truncate_text(self, text: str, limit: int) -> str:
        text = " ".join(str(text).split())
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _normalize_for_dedupe(self, text: str) -> str:
        words = re.findall(r"[a-z0-9']+", str(text).lower())
        return " ".join(words[:32])

    def _looks_like_raw_plan(self, content: str) -> bool:
        stripped = content.strip()
        return stripped.startswith("{") and ("hour_" in stripped or '"08:00"' in stripped or "'08:00'" in stripped)

    def _compact_soul(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            lines.append(self._truncate_text(clean, 180))
            if len(lines) >= 10:
                break
        return "\n".join(lines)

    def _dedupe_semantic_rows(self, rows: list[dict]) -> list[dict]:
        seen = set()
        kept = []
        for row in rows:
            key = self._normalize_for_dedupe(f"{row.get('subject', '')} {row.get('fact', '')}")
            if not key or key in seen:
                continue
            seen.add(key)
            kept.append(row)
        return kept

    def _novel_lines(self, lines: list) -> list[str]:
        novel = []
        seen = set()
        existing = self._normalize_for_dedupe(self.bio.get("file_context", ""))
        for line in lines:
            if not isinstance(line, str) or not line.strip():
                continue
            key = self._normalize_for_dedupe(line)
            if not key or key in seen or key in existing:
                continue
            seen.add(key)
            novel.append(line.strip())
        return novel

    def _is_novel_reflection(self, text: str) -> bool:
        terms = self._message_terms(text)
        if not terms:
            return False
        for prior in self._recent_reflections[-8:]:
            prior_terms = self._message_terms(prior)
            if len(terms & prior_terms) / max(1, len(terms | prior_terms)) >= 0.55:
                return False
        return True

    def _remember_reflection(self, text: str) -> None:
        self._recent_reflections.append(text)
        self._recent_reflections = self._recent_reflections[-10:]

    def _is_novel_fact(self, fact: str) -> bool:
        fact_key = self._normalize_for_dedupe(fact)
        context_key = self._normalize_for_dedupe(self.bio.get("file_context", ""))
        return bool(fact_key and fact_key not in context_key)

    async def _has_similar_recent_dialogue(self, content: str) -> bool:
        if not hasattr(self.memory.repo, "get_by_kind"):
            return False
        recent = await self.memory.repo.get_by_kind(self.bio["id"], "dialogue", n=6)
        terms = self._message_terms(content)
        if not terms:
            return False
        for row in recent:
            prior_terms = self._message_terms(str(row.get("content", "")))
            if len(terms & prior_terms) / max(1, len(terms | prior_terms)) >= 0.62:
                return True
        return False
