from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from agent.memory.experience_memory import ExperienceMemory
from agent.reflection.failure_taxonomy import classify_failure, suggest_recovery_skill
from agent.reflection.policy_patch import PolicyPatchTable
from agent.schemas import (
    AgentConfig,
    EpisodeReflection,
    ExperienceMemoryItem,
    FailureClass,
    PolicyPatchProposal,
    SkillName,
)


EPISODE_REFLECTION_SYSTEM_PROMPT = """You are a high-level reflection agent for object-goal navigation.
You analyze completed episodes to improve future high-level skill selection.
Use only the structured episode trace, validator outcomes, target/frontier evidence, failure signals, and optional GT feedback.
Do not output hidden chain-of-thought. Return concise public explanations and reusable navigation lessons.
Return only valid JSON."""


class ReflectionEngine:
    """Rule-based trajectory-to-memory feedback for navigation episodes."""

    def __init__(
        self,
        cfg: Optional[AgentConfig] = None,
        experience_memory: Optional[ExperienceMemory] = None,
        policy_patch_table: Optional[PolicyPatchTable] = None,
        vlm_provider: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg or AgentConfig()
        self.experience_memory = experience_memory or ExperienceMemory(
            memory_path=self.cfg.memory_path,
            max_items=self.cfg.max_reflection_memory_items,
            read_mode=self.cfg.memory_read_mode,
            write_mode=self.cfg.memory_write_mode,
        )
        self.policy_patch_table = policy_patch_table or PolicyPatchTable(self.cfg.policy_patch_path, self.cfg)
        self.vlm_provider = vlm_provider
        if self.vlm_provider is None and self.cfg.enable_vlm_episode_reflection and (self.cfg.vlm_provider or "mock") != "mock":
            try:
                from agent.vlm_provider import build_vlm_provider

                self.vlm_provider = build_vlm_provider(self.cfg)
            except Exception:
                self.vlm_provider = None

    def reflect_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        episode = self._with_logged_decisions(episode)
        split = episode.get("split") or "unknown"
        success = bool(episode.get("success", False))
        skill_sequence = self._skill_sequence(episode)
        failure_type = None if success else (episode.get("failure_type") or self._infer_failure_type(episode))
        consecutive = self._consecutive_failure_count(episode, failure_type)
        failure_class = None if success else (episode.get("failure_class") or classify_failure(failure_type, consecutive, cfg=self.cfg))
        if failure_class == FailureClass.UNKNOWN.value and failure_type:
            failure_class = classify_failure(failure_type, 3, cfg=self.cfg)

        lesson, bad_decision, better_decision, confidence = self._lesson_for_episode(
            episode, success, failure_type, failure_class, skill_sequence
        )
        state_condition = self._state_condition(episode, skill_sequence)
        proposal = self._policy_patch_for_episode(episode, failure_type, confidence)
        reflection_source = "rule_based"
        vlm_reflection = self._vlm_reflect_episode(
            episode=episode,
            success=success,
            failure_type=failure_type,
            failure_class=failure_class,
            skill_sequence=skill_sequence,
            state_condition=state_condition,
            rule_lesson=lesson,
            rule_bad_decision=bad_decision,
            rule_better_decision=better_decision,
        )
        if vlm_reflection.get("source") == "vlm":
            reflection_source = "vlm_episode_reflection"
            failure_type = vlm_reflection.get("failure_type") or failure_type
            failure_class = vlm_reflection.get("failure_class") or failure_class
            lesson = vlm_reflection.get("lesson") or lesson
            bad_decision = vlm_reflection.get("bad_decision") or bad_decision
            better_decision = (
                vlm_reflection.get("better_decision")
                or vlm_reflection.get("better_skill")
                or better_decision
            )
            confidence = self._bounded_confidence(vlm_reflection.get("confidence"), confidence)
            state_condition.update(vlm_reflection.get("state_condition_updates") or {})
            proposal = self._vlm_policy_patch_for_episode(
                episode, vlm_reflection, fallback=proposal, confidence=confidence
            )

        memory_item = ExperienceMemoryItem(
            split=split,
            scene_id=episode.get("scene_id"),
            episode_id=episode.get("episode_id"),
            target_category=episode.get("target_category"),
            scene_context=list(episode.get("scene_context") or []),
            success=success,
            failure_type=failure_type,
            failure_class=failure_class,
            state_condition=state_condition,
            bad_decision=bad_decision,
            better_decision=better_decision,
            lesson=lesson,
            suggested_policy_patch=proposal.to_dict() if proposal else None,
            confidence=confidence,
        )
        reflection = EpisodeReflection(
            episode_id=episode.get("episode_id"),
            scene_id=episode.get("scene_id"),
            split=split,
            target_category=episode.get("target_category"),
            success=success,
            failure_type=failure_type,
            failure_class=failure_class,
            summary=self._summary(success, failure_type, lesson),
            selected_skill_sequence=skill_sequence,
            validator_rejection_count=len(episode.get("validator_rejections") or []),
            metrics=self._metrics(episode),
        )
        return {
            "episode_reflection": reflection.to_dict(),
            "experience_memory_item": memory_item.to_dict(),
            "policy_patch_proposals": [proposal.to_dict()] if proposal else [],
            "reflection_source": reflection_source,
            "vlm_episode_reflection": vlm_reflection,
        }

    def finalize_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        result = self.reflect_episode(episode)
        split = episode.get("split") or "unknown"
        memory_written = False
        if self.cfg.enable_episode_reflection and self.cfg.enable_reflection_memory:
            memory_written = self.experience_memory.append_memory(result["experience_memory_item"], split=split)
        recorded_patches = []
        if self.cfg.enable_policy_patch_table:
            for proposal in result.get("policy_patch_proposals") or []:
                recorded = self.policy_patch_table.record_proposal(proposal, split=split)
                if recorded is not None:
                    recorded_patches.append(recorded.to_dict())
        result["memory_written"] = memory_written
        result["recorded_policy_patches"] = recorded_patches
        return result

    def _lesson_for_episode(
        self,
        episode: Dict[str, Any],
        success: bool,
        failure_type: Optional[str],
        failure_class: Optional[str],
        skill_sequence: List[str],
    ) -> Tuple[str, Optional[str], Optional[str], float]:
        target = episode.get("target_category") or "target"
        if success:
            sequence = " -> ".join(skill_sequence) if skill_sequence else "ApexNav fallback"
            return (
                f"For {target}, the skill sequence {sequence} completed the episode successfully.",
                None,
                sequence,
                0.65,
            )
        if failure_type == "false_positive_stop":
            return (
                f"For {target}, do not stop on a low-confidence or single-view candidate; verify the target before final navigation.",
                f"{SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value} was selected too early.",
                f"Use {SkillName.VERIFY_TARGET.value} before stopping.",
                0.82,
            )
        if failure_type == "semantic_explore_no_progress":
            return (
                f"For {target}, switch to {SkillName.GEOMETRIC_EXPLORE.value} when semantic exploration repeats without target confidence or map gain.",
                f"Repeated {SkillName.SEMANTIC_EXPLORE.value} did not improve evidence.",
                f"Try {SkillName.GEOMETRIC_EXPLORE.value} or recovery after repeated no-progress semantic frontiers.",
                0.72,
            )
        if failure_type == "repeated_bad_frontier":
            return (
                "Do not revisit a frontier or frontier cluster after repeated failures; mark it low_value or blocked before selecting alternatives.",
                "The same failed frontier was selected repeatedly.",
                f"Blacklist the frontier and use {SkillName.RECOVER_FROM_STUCK.value} or an alternate frontier.",
                0.78,
            )
        if failure_type in {"planner_stuck", "repeated_collision", "timeout", "timeout_near_goal"}:
            return (
                f"Trigger {SkillName.RECOVER_FROM_STUCK.value} after stuck, repeated collision, or timeout instead of reissuing the same waypoint.",
                "Navigation retried a degraded low-level state.",
                suggest_recovery_skill(failure_type, self.cfg.disable_recover_from_stuck),
                0.8,
            )
        if failure_type == "missing_target":
            return (
                f"For {target}, after reaching a high-semantic area, use verification or active observation because the target may be small or occluded.",
                "Exploration did not produce a reliable target candidate.",
                f"Use {SkillName.VERIFY_TARGET.value} when a weak candidate appears, otherwise continue exploration.",
                0.66,
            )
        if failure_type == "no_frontier_deadend":
            return (
                "When no frontier remains, avoid blind retries and fall back to ApexNav or a safe reachable waypoint to recover map consistency.",
                "Planner reached a no-frontier deadend.",
                SkillName.FALLBACK_APEXNAV.value,
                0.76,
            )
        if failure_class == FailureClass.DEGRADING.value:
            return (
                f"A degrading failure ({failure_type}) requires recovery or fallback before continuing normal exploration.",
                "The episode entered a degrading failure state.",
                suggest_recovery_skill(failure_type, self.cfg.disable_recover_from_stuck),
                0.7,
            )
        return (
            f"A non-degrading failure ({failure_type}) can retry once or switch to an alternate exploration skill.",
            "The selected skill did not satisfy its postcondition.",
            f"Switch between {SkillName.SEMANTIC_EXPLORE.value} and {SkillName.GEOMETRIC_EXPLORE.value} based on evidence.",
            0.6,
        )

    def _vlm_reflect_episode(
        self,
        episode: Dict[str, Any],
        success: bool,
        failure_type: Optional[str],
        failure_class: Optional[str],
        skill_sequence: List[str],
        state_condition: Dict[str, Any],
        rule_lesson: str,
        rule_bad_decision: Optional[str],
        rule_better_decision: Optional[str],
    ) -> Dict[str, Any]:
        if not self.cfg.enable_vlm_episode_reflection or self.vlm_provider is None:
            return {}
        payload = {
            "task": "Reflect on the completed ObjectNav episode and produce reusable skill-selection memory.",
            "required_json_format": {
                "failure_analysis": "short public explanation of why the episode succeeded or failed",
                "bad_decision": "which high-level skill choice or stop/retry choice was harmful, or null",
                "better_decision": "what the agent should do differently next time",
                "better_skill": "one allowed skill name if a different skill is recommended, or null",
                "lesson": "one concise reusable lesson for future episodes",
                "failure_type": failure_type or "null",
                "failure_class": failure_class or "null",
                "state_condition_updates": {
                    "key": "small structured conditions that make the lesson applicable"
                },
                "suggested_policy_patch": {
                    "trigger_condition": "structured condition for applying this lesson",
                    "recommended_action": "skill name",
                    "rationale": "short public rationale",
                    "confidence": 0.0
                },
                "confidence": 0.0,
            },
            "episode": self._compact_episode(episode, success, failure_type, failure_class, skill_sequence),
            "rule_based_reflection": {
                "lesson": rule_lesson,
                "bad_decision": rule_bad_decision,
                "better_decision": rule_better_decision,
                "state_condition": state_condition,
            },
            "allowed_skills": [item.value for item in SkillName],
            "reflection_instructions": [
                "Prefer lessons that change future skill arbitration, verification, recovery, or stop decisions.",
                "If GT feedback is present, explain why the GT path or progress signal implies a better skill.",
                "Do not recommend low-level continuous actions.",
                "Do not invent unavailable skills.",
            ],
        }
        try:
            raw = self.vlm_provider.generate(
                EPISODE_REFLECTION_SYSTEM_PROMPT,
                json.dumps(payload, indent=2, sort_keys=True, default=str),
            )
            parsed = self._parse_json(raw)
        except Exception as exc:
            return {"source": "vlm_error", "error": f"{type(exc).__name__}: {exc}"}
        if not parsed:
            return {}
        parsed["source"] = "vlm"
        parsed["raw_response"] = raw
        if parsed.get("better_skill") and parsed.get("better_skill") not in {item.value for item in SkillName}:
            parsed["better_skill"] = None
        if parsed.get("suggested_policy_patch") and not isinstance(parsed["suggested_policy_patch"], dict):
            parsed["suggested_policy_patch"] = None
        return parsed

    def _compact_episode(
        self,
        episode: Dict[str, Any],
        success: bool,
        failure_type: Optional[str],
        failure_class: Optional[str],
        skill_sequence: List[str],
    ) -> Dict[str, Any]:
        decisions = episode.get("decisions") or episode.get("decision_trace") or []
        compact_decisions = []
        max_items = max(1, int(self.cfg.vlm_episode_reflection_max_decisions))
        if decisions:
            selected = decisions[-max_items:]
            for item in selected:
                state = item.get("state_summary") or {}
                compact_decisions.append(
                    {
                        "timestep": item.get("timestep"),
                        "selected_skill": (item.get("agent_decision") or {}).get("selected_skill")
                        or item.get("selected_skill")
                        or item.get("executed_skill"),
                        "executed_skill": item.get("executed_skill"),
                        "skill_args": (item.get("agent_decision") or {}).get("skill_args")
                        or (item.get("validator_result") or {}).get("final_arguments")
                        or {},
                        "validator_result": item.get("validator_result"),
                        "retrieved_lessons": item.get("retrieved_lessons") or [],
                        "semantic_score_stats": state.get("semantic_score_stats"),
                        "target_candidates": state.get("target_candidates"),
                        "frontier_count": len(state.get("frontiers") or []),
                        "navigation_history": state.get("navigation_history"),
                        "gt_feedback": state.get("gt_feedback"),
                    }
                )
        return {
            "episode_id": episode.get("episode_id"),
            "scene_id": episode.get("scene_id"),
            "split": episode.get("split") or "unknown",
            "target_category": episode.get("target_category"),
            "success": success,
            "failure_type": failure_type,
            "failure_class": failure_class,
            "stop_reason": episode.get("stop_reason"),
            "steps": episode.get("steps"),
            "spl": episode.get("spl"),
            "softspl": episode.get("softspl"),
            "final_distance_to_goal": episode.get("final_distance_to_goal"),
            "skill_sequence": skill_sequence,
            "failure_signals": episode.get("failure_signals") or [],
            "validator_rejections": episode.get("validator_rejections") or [],
            "target_detection_trace": episode.get("target_detection_trace") or [],
            "target_candidate_trace": episode.get("target_candidate_trace") or [],
            "selected_frontier_trace": episode.get("selected_frontier_trace") or [],
            "agent_diagnostics": episode.get("agent_diagnostics") or {},
            "recent_decisions": compact_decisions,
        }

    def _with_logged_decisions(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        if episode.get("decisions") or not self.cfg.enable_episode_logger:
            return episode
        path = self._episode_log_path(episode.get("episode_id"))
        if not path or not os.path.exists(path):
            return episode
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return episode
        decisions = data.get("decisions") or []
        if not decisions:
            return episode
        enriched = dict(episode)
        enriched["decisions"] = decisions
        enriched.setdefault("episode_log_path", path)
        return enriched

    def _episode_log_path(self, episode_id: Any) -> Optional[str]:
        root = self.cfg.episode_log_root
        if not root:
            return None
        run_id = self.cfg.run_id or "default"
        episodes_dir = os.path.join(root, run_id, "episodes")
        candidates = []
        if episode_id:
            candidates.append(os.path.join(episodes_dir, f"{self._safe_name(episode_id)}.json"))
        candidates.append(os.path.join(episodes_dir, "unknown_episode.json"))
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def _safe_name(value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:160]

    def _vlm_policy_patch_for_episode(
        self,
        episode: Dict[str, Any],
        vlm_reflection: Dict[str, Any],
        fallback: Optional[PolicyPatchProposal],
        confidence: float,
    ) -> Optional[PolicyPatchProposal]:
        patch = vlm_reflection.get("suggested_policy_patch")
        if not isinstance(patch, dict):
            return fallback
        recommended = patch.get("recommended_action") or vlm_reflection.get("better_skill")
        if recommended not in {item.value for item in SkillName}:
            return fallback
        trigger = patch.get("trigger_condition") or {}
        if not isinstance(trigger, dict):
            trigger = {"failure_type": vlm_reflection.get("failure_type") or episode.get("failure_type")}
        return PolicyPatchProposal(
            target_scope=patch.get("target_scope") or episode.get("target_category"),
            trigger_condition=trigger,
            recommended_action=recommended,
            rationale=patch.get("rationale")
            or vlm_reflection.get("failure_analysis")
            or vlm_reflection.get("lesson")
            or "",
            confidence=self._bounded_confidence(patch.get("confidence"), confidence),
            support_count=int(patch.get("support_count") or 1),
            source_episode_id=patch.get("source_episode_id") or episode.get("episode_id"),
            active=bool(patch.get("active", False)),
        )

    def _policy_patch_for_episode(
        self,
        episode: Dict[str, Any],
        failure_type: Optional[str],
        confidence: float,
    ) -> Optional[PolicyPatchProposal]:
        target = episode.get("target_category")
        if failure_type == "false_positive_stop":
            return PolicyPatchProposal(
                target_scope=target,
                trigger_condition={
                    "selected_skill": SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
                    "target_confidence_lt": self.cfg.target_stop_threshold,
                    "multi_view_confirmed": False,
                },
                recommended_action=SkillName.VERIFY_TARGET.value,
                rationale="Previous false_positive_stop occurred under low-confidence or single-view target evidence.",
                confidence=max(confidence, 0.82),
                support_count=1,
                source_episode_id=episode.get("episode_id"),
            )
        if failure_type == "semantic_explore_no_progress":
            return PolicyPatchProposal(
                target_scope=target,
                trigger_condition={
                    "selected_skill": SkillName.SEMANTIC_EXPLORE.value,
                    "failure_type": "semantic_explore_no_progress",
                    "repeated_failures_gte": 2,
                },
                recommended_action=SkillName.GEOMETRIC_EXPLORE.value,
                rationale="Repeated semantic exploration without progress should switch to geometric exploration.",
                confidence=max(confidence, 0.72),
                support_count=1,
                source_episode_id=episode.get("episode_id"),
            )
        if failure_type in {"planner_stuck", "repeated_collision", "timeout", "repeated_bad_frontier"}:
            return PolicyPatchProposal(
                target_scope=target,
                trigger_condition={"failure_type": failure_type},
                recommended_action=suggest_recovery_skill(failure_type, self.cfg.disable_recover_from_stuck),
                rationale=f"Failure {failure_type} should trigger recovery instead of blind retry.",
                confidence=max(confidence, 0.78),
                support_count=1,
                source_episode_id=episode.get("episode_id"),
            )
        if failure_type == "no_frontier_deadend":
            return PolicyPatchProposal(
                target_scope=target,
                trigger_condition={"failure_type": failure_type},
                recommended_action=SkillName.FALLBACK_APEXNAV.value,
                rationale="No-frontier deadends should fallback instead of continuing invalid waypoint selection.",
                confidence=max(confidence, 0.76),
                support_count=1,
                source_episode_id=episode.get("episode_id"),
            )
        return None

    @staticmethod
    def _skill_sequence(episode: Dict[str, Any]) -> List[str]:
        if episode.get("selected_skill_sequence"):
            return list(episode.get("selected_skill_sequence") or [])
        trace = episode.get("skill_trace") or []
        skills = []
        for item in trace:
            if isinstance(item, str):
                skills.append(item)
            elif isinstance(item, dict):
                skill = item.get("skill_name") or item.get("executed_skill") or item.get("selected_skill")
                if skill:
                    skills.append(skill)
        return skills

    @staticmethod
    def _infer_failure_type(episode: Dict[str, Any]) -> str:
        signals = list(episode.get("failure_signals") or [])
        if episode.get("stop_reason") in {"false_positive_stop", "premature_stop"}:
            return episode.get("stop_reason")
        for preferred in [
            "false_positive_stop",
            "planner_stuck",
            "repeated_collision",
            "repeated_bad_frontier",
            "semantic_explore_no_progress",
            "no_frontier_deadend",
            "missing_target",
            "timeout",
        ]:
            if preferred in signals:
                return preferred
        if episode.get("stuck_count", 0) or episode.get("collision_count", 0):
            return "planner_stuck"
        if episode.get("no_frontier_count", 0):
            return "no_frontier_deadend"
        if episode.get("timeout_count", 0) or episode.get("stop_reason") == "timeout":
            return "timeout"
        return "missing_target"

    @staticmethod
    def _consecutive_failure_count(episode: Dict[str, Any], failure_type: Optional[str]) -> int:
        if not failure_type:
            return 0
        failures = episode.get("failure_signals") or episode.get("recent_failures") or []
        count = 0
        for failure in reversed(failures):
            if failure == failure_type:
                count += 1
            else:
                break
        return max(count, 1)

    @staticmethod
    def _state_condition(episode: Dict[str, Any], skill_sequence: List[str]) -> Dict[str, Any]:
        state = episode.get("state_summary") or {}
        stats = state.get("semantic_score_stats") or episode.get("semantic_score_stats") or {}
        candidates = state.get("target_candidates") or episode.get("target_candidates") or []
        best_candidate = max(candidates, key=lambda item: item.get("confidence") or 0.0) if candidates else {}
        return {
            "semantic_peak": stats.get("has_clear_peak"),
            "target_confidence": best_candidate.get("confidence") or episode.get("target_confidence"),
            "multi_view_confirmed": best_candidate.get("multi_view_confirmed") or episode.get("multi_view_confirmed", False),
            "selected_skill_sequence": skill_sequence,
        }

    @staticmethod
    def _metrics(episode: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "spl",
            "softspl",
            "steps",
            "final_distance_to_goal",
            "false_positive_stop_count",
            "timeout_count",
            "stuck_count",
            "collision_count",
            "fallback_count",
            "validator_rejection_count",
            "memory_retrieval_count",
            "memory_write_count",
        ]
        return {key: episode.get(key) for key in keys if key in episode}

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    return {}
        return {}

    @staticmethod
    def _bounded_confidence(value: Any, default: float) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = float(default)
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _summary(success: bool, failure_type: Optional[str], lesson: str) -> str:
        prefix = "success" if success else f"failure:{failure_type}"
        return f"{prefix} - {lesson}"
