from __future__ import annotations

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


class ReflectionEngine:
    """Rule-based trajectory-to-memory feedback for navigation episodes."""

    def __init__(
        self,
        cfg: Optional[AgentConfig] = None,
        experience_memory: Optional[ExperienceMemory] = None,
        policy_patch_table: Optional[PolicyPatchTable] = None,
    ) -> None:
        self.cfg = cfg or AgentConfig()
        self.experience_memory = experience_memory or ExperienceMemory(
            memory_path=self.cfg.memory_path,
            max_items=self.cfg.max_reflection_memory_items,
            read_mode=self.cfg.memory_read_mode,
            write_mode=self.cfg.memory_write_mode,
        )
        self.policy_patch_table = policy_patch_table or PolicyPatchTable(self.cfg.policy_patch_path, self.cfg)

    def reflect_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
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
    def _summary(success: bool, failure_type: Optional[str], lesson: str) -> str:
        prefix = "success" if success else f"failure:{failure_type}"
        return f"{prefix} - {lesson}"
