from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import copy

from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from agent.schemas import AgentConfig, AgentDecision, SkillName
from agent.skill.skill_registry import SkillRegistry


class ReflectiveNavigationAgent:
    """High-level skill selector.

    In mock mode this is deterministic and requires no API key. In VLM mode the
    caller injects a provider with a ``generate(system_prompt, user_prompt)`` method.
    """

    def __init__(
        self,
        cfg: Optional[AgentConfig] = None,
        skill_registry: Optional[SkillRegistry] = None,
        vlm_provider: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg or AgentConfig()
        self.skill_registry = skill_registry
        self.vlm_provider = vlm_provider
        self.vlm_calls = 0

    def select_skill(
        self,
        state_summary: Dict[str, Any],
        role_memory: Any,
        task_memory: Any,
        working_memory: Any,
        retrieved_lessons: Optional[List[Dict[str, Any]]] = None,
        active_policy_patches: Optional[List[Dict[str, Any]]] = None,
        validator_constraints: Optional[List[str]] = None,
    ) -> AgentDecision:
        if self.cfg.force_all_decisions_to_FALLBACK_APEXNAV:
            return AgentDecision.fallback("forced fallback for baseline sanity check")
        if self.cfg.vlm_provider == "mock":
            return self._select_mock(state_summary)
        return self._select_vlm(
            state_summary=state_summary,
            role_memory=self._to_dict(role_memory),
            task_memory=self._to_dict(task_memory),
            working_memory=self._to_dict(working_memory),
            retrieved_lessons=retrieved_lessons or [],
            active_policy_patches=active_policy_patches or [],
            validator_constraints=validator_constraints or [],
        )

    def _select_mock(self, state: Dict[str, Any]) -> AgentDecision:
        candidate = self._best_target_candidate(state)
        if candidate is not None:
            confidence = candidate.get("confidence") or 0.0
            is_reliable = confidence >= self.cfg.target_stop_threshold
            if self.cfg.require_multiview_before_stop:
                is_reliable = is_reliable and bool(candidate.get("multi_view_confirmed"))
            if is_reliable and candidate.get("reachable", True) and not candidate.get("rejected_false_positive"):
                return AgentDecision(
                    selected_skill=SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
                    skill_args={"target_candidate_id": candidate.get("id")},
                    expected_postcondition="reach confirmed target and stop",
                    reason="target candidate is reliable",
                    confidence=min(1.0, float(confidence)),
                )
            if (
                not self.cfg.disable_verify_target
                and (confidence >= self.cfg.target_verify_threshold or not candidate.get("multi_view_confirmed"))
            ):
                return AgentDecision(
                    selected_skill=SkillName.VERIFY_TARGET.value,
                    skill_args={"target_candidate_id": candidate.get("id")},
                    expected_postcondition="confirm or reject target candidate",
                    reason="target candidate needs verification",
                    confidence=max(0.4, min(0.8, float(confidence or 0.4))),
                )

        history = state.get("navigation_history") or {}
        if history.get("stuck_count", 0) > 0 or history.get("recent_failures"):
            best_known = self._best_known_point(state)
            if best_known is not None:
                return AgentDecision(
                    status="recover",
                    selected_skill=SkillName.RETURN_TO_BEST_KNOWN_POINT.value,
                    skill_args={"best_known_point": best_known.get("waypoint")},
                    expected_postcondition="return to the best historical navigation point",
                    reason="recent navigation failure has a valid best known point",
                    confidence=0.72,
                )
            return AgentDecision(
                status="recover",
                selected_skill=SkillName.RECOVER_FROM_STUCK.value,
                skill_args={},
                expected_postcondition="recover from stuck or repeated failure",
                reason="recent navigation failure requires recovery",
                confidence=0.7,
            )

        if self.cfg.mock_follow_apexnav_by_default:
            return AgentDecision(
                selected_skill=SkillName.FOLLOW_APEXNAV_PROPOSAL.value,
                skill_args={},
                expected_postcondition="follow ApexNav proposal until a high-level event changes",
                reason="diagnostic mock follows ApexNav proposal",
                confidence=0.6,
            )

        semantic_stats = state.get("semantic_score_stats") or {}
        if semantic_stats.get("has_clear_peak"):
            frontier = self._best_frontier(state, prefer_semantic=True)
            return AgentDecision(
                selected_skill=SkillName.SEMANTIC_EXPLORE.value,
                skill_args={"frontier_id": None if frontier is None else frontier.get("id")},
                expected_postcondition="reach semantic frontier and gain information",
                reason="semantic map has a clear peak",
                confidence=0.68,
            )

        frontier = self._best_frontier(state, prefer_semantic=False)
        return AgentDecision(
            selected_skill=SkillName.GEOMETRIC_EXPLORE.value,
            skill_args={"frontier_id": None if frontier is None else frontier.get("id")},
            expected_postcondition="reach nearest reachable frontier",
            reason="semantic signal is weak or unavailable",
            confidence=0.55,
        )

    def _select_vlm(
        self,
        state_summary: Dict[str, Any],
        role_memory: Dict[str, Any],
        task_memory: Dict[str, Any],
        working_memory: Dict[str, Any],
        retrieved_lessons: List[Dict[str, Any]],
        active_policy_patches: List[Dict[str, Any]],
        validator_constraints: List[str],
    ) -> AgentDecision:
        if self.vlm_provider is None:
            return AgentDecision.fallback("vlm provider unavailable")
        if self.vlm_calls >= self.cfg.max_vlm_calls_per_episode:
            return AgentDecision.fallback("vlm call budget exhausted")

        available_skills = self._available_skills_for_prompt()
        prompt_state_summary = self._state_for_vlm_prompt(state_summary)
        user_prompt = build_user_prompt(
            state_summary=prompt_state_summary,
            role_memory=role_memory,
            task_memory=task_memory,
            working_memory=working_memory,
            available_skills=available_skills,
            retrieved_lessons=retrieved_lessons,
            active_policy_patches=active_policy_patches,
            validator_constraints=validator_constraints,
        )
        self.vlm_calls += 1
        try:
            image_data_urls = self._extract_visual_data_urls(prompt_state_summary)
            if image_data_urls:
                raw = self.vlm_provider.generate(
                    SYSTEM_PROMPT,
                    user_prompt,
                    image_data_urls=image_data_urls,
                )
            else:
                raw = self.vlm_provider.generate(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            return AgentDecision.fallback(f"vlm provider error: {type(exc).__name__}")
        try:
            data = self._parse_json_response(raw)
            if "selected_skill" not in data:
                return AgentDecision.fallback("vlm response missing selected_skill")
            return AgentDecision(
                status=data.get("status", "continue"),
                selected_skill=data.get("selected_skill", SkillName.FALLBACK_APEXNAV.value),
                skill_args=data.get("skill_args") or {},
                expected_postcondition=data.get("expected_postcondition"),
                monitoring_plan=data.get("monitoring_plan") or {},
                recovery_hint=data.get("recovery_hint") or {},
                reason=data.get("reason", ""),
                confidence=float(data.get("confidence", 0.0) or 0.0),
                raw_response=raw,
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            decision = AgentDecision.fallback("vlm response was not valid json")
            decision.raw_response = raw
            return decision

    def _available_skills_for_prompt(self) -> Dict[str, Any]:
        if self.skill_registry is None:
            return {}
        skills = self.skill_registry.specs_as_dict()
        if self.cfg.disable_verify_target:
            skills.pop(SkillName.VERIFY_TARGET.value, None)
            for spec in skills.values():
                if not isinstance(spec, dict):
                    continue
                if spec.get("recovery_action") == SkillName.VERIFY_TARGET.value:
                    spec["recovery_action"] = SkillName.FOLLOW_APEXNAV_PROPOSAL.value
        return skills


    def _state_for_vlm_prompt(self, state_summary: Dict[str, Any]) -> Dict[str, Any]:
        state = copy.deepcopy(state_summary)
        if not self.cfg.enable_semantic_map_observation:
            state.pop("semantic_map_observation", None)
        return state

    @staticmethod
    def _parse_json_response(raw: str) -> Dict[str, Any]:
        text = (raw or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    @staticmethod
    def _best_target_candidate(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidates = [
            c
            for c in state.get("target_candidates", [])
            if not c.get("rejected_false_positive") and c.get("reachable", True)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.get("confidence") or 0.0)

    @staticmethod
    def _best_frontier(state: Dict[str, Any], prefer_semantic: bool) -> Optional[Dict[str, Any]]:
        frontiers = [
            f
            for f in state.get("frontiers", [])
            if f.get("reachable", True) and not f.get("blocked") and not f.get("low_value")
        ]
        if not frontiers:
            return None
        if prefer_semantic:
            return max(frontiers, key=lambda item: item.get("semantic_score") or 0.0)
        return min(frontiers, key=lambda item: item.get("distance") if item.get("distance") is not None else 1e9)

    @staticmethod
    def _best_known_point(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        history = state.get("navigation_history") or {}
        best = history.get("best_known_point") or state.get("best_known_point")
        if not isinstance(best, dict) or not best.get("available", False):
            return None
        waypoint = best.get("waypoint") or best.get("position")
        if not waypoint:
            return None
        return dict(best, waypoint=waypoint)

    @staticmethod
    def _to_dict(value: Any) -> Dict[str, Any]:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _extract_visual_data_urls(state: Dict[str, Any]) -> List[str]:
        urls = []
        for key in ("rgb_observation", "semantic_map_observation"):
            observation = state.get(key) or {}
            if not isinstance(observation, dict) or not observation.get("available", False):
                continue
            data_url = observation.get("data_url") or observation.get("image_url")
            if isinstance(data_url, str) and data_url.startswith("data:image/"):
                urls.append(data_url)
        return urls
