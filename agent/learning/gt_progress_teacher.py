from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agent.learning.trajectory_schema import TeacherAction, ToolUseLearningSample, TrajectoryEpisode, TrajectoryStep
from agent.reflection.failure_taxonomy import classify_failure
from agent.schemas import AgentConfig, SkillName


class GTProgressTeacher:
    """Distill lessons from Habitat GT distance-to-goal progress.

    This teacher does not expose GT feedback to the online VLM prompt. It reads
    the episode log after completion, compares distance-to-goal across high-level
    decisions, and writes compact lessons about which skills helped or failed to
    reduce oracle distance.
    """

    def __init__(self, cfg: Optional[AgentConfig] = None) -> None:
        self.cfg = cfg or AgentConfig()

    def build_samples(self, episode: TrajectoryEpisode) -> List[ToolUseLearningSample]:
        steps = [step for step in episode.trajectory_steps if step.selected_skill]
        if not steps:
            return []

        samples: List[ToolUseLearningSample] = []
        for index, step in enumerate(steps):
            before = self._gt_distance(step)
            after = self._next_gt_distance(steps, index, episode)
            if before is None or after is None:
                continue

            delta = before - after
            if delta >= self.cfg.gt_progress_min_delta:
                samples.append(self._helpful_sample(episode, step, before, after, delta))
            elif delta <= -self.cfg.gt_progress_min_delta:
                samples.append(self._regression_sample(episode, step, before, after, delta))
            elif step.selected_skill in {
                SkillName.SEMANTIC_EXPLORE.value,
                SkillName.GEOMETRIC_EXPLORE.value,
                SkillName.FOLLOW_APEXNAV_PROPOSAL.value,
            }:
                samples.append(self._no_progress_sample(episode, step, before, after, delta))

            if len(samples) >= self.cfg.gt_learning_max_samples_per_episode:
                break
        return samples

    def _helpful_sample(
        self,
        episode: TrajectoryEpisode,
        step: TrajectoryStep,
        before: float,
        after: float,
        delta: float,
    ) -> ToolUseLearningSample:
        teacher = TeacherAction(
            skill_name=step.selected_skill or SkillName.FALLBACK_APEXNAV.value,
            tool_name=step.tool_name or self._tool_for_skill(step.selected_skill),
            arguments=dict(step.skill_args),
            reason=f"GT distance-to-goal decreased by {delta:.2f}m after this skill.",
            confidence=0.78,
            source="gt_progress",
        )
        return self._sample(
            episode,
            step,
            teacher,
            outcome="gt_progress_skill_helped",
            failure_type=None,
            lesson=(
                f"GT progress supervision says {teacher.skill_name} was useful in this state: "
                f"distance_to_goal decreased from {before:.2f}m to {after:.2f}m. In similar states, "
                "commit to the same skill until its postcondition or a stable trigger fires."
            ),
            confidence=0.78,
            before=before,
            after=after,
            delta=delta,
        )

    def _regression_sample(
        self,
        episode: TrajectoryEpisode,
        step: TrajectoryStep,
        before: float,
        after: float,
        delta: float,
    ) -> ToolUseLearningSample:
        teacher = self._better_teacher_action(step, source="gt_progress")
        return self._sample(
            episode,
            step,
            teacher,
            outcome="gt_progress_skill_regressed",
            failure_type=self._failure_type_for_step(step, "inefficient_exploration"),
            lesson=(
                f"GT progress supervision says {step.selected_skill} moved away from the goal "
                f"({before:.2f}m to {after:.2f}m). In similar states, prefer {teacher.skill_name} "
                "instead of repeating the same skill."
            ),
            confidence=0.82,
            before=before,
            after=after,
            delta=delta,
        )

    def _no_progress_sample(
        self,
        episode: TrajectoryEpisode,
        step: TrajectoryStep,
        before: float,
        after: float,
        delta: float,
    ) -> ToolUseLearningSample:
        teacher = self._better_teacher_action(step, source="gt_progress")
        return self._sample(
            episode,
            step,
            teacher,
            outcome="gt_progress_no_progress",
            failure_type=self._failure_type_for_step(step, "low_information_gain"),
            lesson=(
                f"GT progress supervision found little useful progress after {step.selected_skill} "
                f"({before:.2f}m to {after:.2f}m). If this repeats, switch to {teacher.skill_name}."
            ),
            confidence=0.62,
            before=before,
            after=after,
            delta=delta,
        )

    def _sample(
        self,
        episode: TrajectoryEpisode,
        step: TrajectoryStep,
        teacher: TeacherAction,
        outcome: str,
        failure_type: Optional[str],
        lesson: str,
        confidence: float,
        before: float,
        after: float,
        delta: float,
    ) -> ToolUseLearningSample:
        return ToolUseLearningSample(
            source="gt_progress",
            episode_id=episode.episode_id,
            scene_id=episode.scene_id,
            split=episode.split,
            target_category=episode.target_category or step.state_summary.get("target_category"),
            timestep=step.timestep,
            state_condition=self._state_condition(step, before, after, delta),
            student_action={
                "skill": step.selected_skill,
                "tool": step.tool_name,
                "arguments": step.skill_args,
                "validator_result": step.validator_result,
            },
            teacher_action=teacher.to_dict(),
            outcome=outcome,
            failure_type=failure_type,
            failure_class=classify_failure(failure_type),
            lesson=lesson,
            policy_patch_proposal=(
                self._policy_patch(step, teacher, failure_type, episode) if failure_type else None
            ),
            confidence=confidence,
            metadata={
                "gt_distance_before": before,
                "gt_distance_after": after,
                "gt_distance_delta": delta,
                "gt_signal": "distance_to_goal",
            },
        )

    def _better_teacher_action(self, step: TrajectoryStep, source: str) -> TeacherAction:
        state = step.state_summary or {}
        candidate = self._best_target_candidate(state)
        if candidate:
            if self._candidate_stop_ready(candidate):
                return TeacherAction(
                    skill_name=SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
                    tool_name="navigate_to_confirmed_target",
                    arguments={"target_candidate_id": candidate.get("id")},
                    reason="GT progress was poor and a confirmed target candidate is available.",
                    confidence=0.82,
                    source=source,
                )
            return TeacherAction(
                skill_name=SkillName.VERIFY_TARGET.value,
                tool_name="verify_target_candidate",
                arguments={"target_candidate_id": candidate.get("id")},
                reason="GT progress was poor and target evidence should be verified before more exploration.",
                confidence=0.8,
                source=source,
            )

        history = state.get("navigation_history") or {}
        recent_failures = set(history.get("recent_failures") or [])
        if recent_failures & {"planner_stuck", "repeated_collision", "frontier_failure", "no_frontier_deadend"}:
            return TeacherAction(
                skill_name=SkillName.RECOVER_FROM_STUCK.value,
                tool_name="recover_from_stuck",
                arguments={},
                reason="GT progress was poor under an objective stuck or frontier failure signal.",
                confidence=0.78,
                source=source,
            )

        semantic_peak = bool((state.get("semantic_score_stats") or {}).get("has_clear_peak"))
        if step.selected_skill == SkillName.SEMANTIC_EXPLORE.value:
            frontier = self._best_frontier(state, prefer_semantic=False)
            if frontier is not None:
                return TeacherAction(
                    skill_name=SkillName.GEOMETRIC_EXPLORE.value,
                    tool_name="select_nearest_reachable_frontier",
                    arguments={"frontier_id": frontier.get("id")},
                    reason="Semantic exploration did not reduce GT distance; use geometric coverage.",
                    confidence=0.68,
                    source=source,
                )
        if semantic_peak:
            frontier = self._best_frontier(state, prefer_semantic=True)
            if frontier is not None:
                return TeacherAction(
                    skill_name=SkillName.SEMANTIC_EXPLORE.value,
                    tool_name="select_semantic_frontier",
                    arguments={"frontier_id": frontier.get("id")},
                    reason="Semantic map has a clear peak that is more likely to reduce GT distance.",
                    confidence=0.72,
                    source=source,
                )

        frontier = self._best_frontier(state, prefer_semantic=False)
        if frontier is not None:
            return TeacherAction(
                skill_name=SkillName.GEOMETRIC_EXPLORE.value,
                tool_name="select_nearest_reachable_frontier",
                arguments={"frontier_id": frontier.get("id")},
                reason="No reliable target exists; choose reachable geometric exploration for GT progress.",
                confidence=0.66,
                source=source,
            )
        return TeacherAction(
            skill_name=SkillName.FALLBACK_APEXNAV.value,
            tool_name="call_original_apexnav_policy",
            arguments={},
            reason="No reachable frontier or target evidence is available.",
            confidence=0.55,
            source=source,
        )

    def _failure_type_for_step(self, step: TrajectoryStep, default: str) -> str:
        if step.failure_type:
            return step.failure_type
        selected = step.selected_skill
        if selected == SkillName.SEMANTIC_EXPLORE.value:
            return "semantic_explore_no_progress"
        if selected == SkillName.GEOMETRIC_EXPLORE.value:
            return "geometric_explore_no_progress"
        return default

    def _state_condition(
        self,
        step: TrajectoryStep,
        before: float,
        after: float,
        delta: float,
    ) -> Dict[str, Any]:
        state = step.state_summary or {}
        history = state.get("navigation_history") or {}
        candidates = state.get("target_candidates") or []
        best_candidate = self._best_target_candidate(state) or {}
        return {
            "semantic_peak": (state.get("semantic_score_stats") or {}).get("has_clear_peak"),
            "target_candidate_count": len(candidates),
            "target_confidence": best_candidate.get("confidence"),
            "multi_view_confirmed": best_candidate.get("multi_view_confirmed", False),
            "frontier_count": len(state.get("frontiers") or []),
            "stuck_count": history.get("stuck_count"),
            "collision_count": history.get("collision_count"),
            "student_selected_skill": step.selected_skill,
            "gt_distance_before": before,
            "gt_distance_after": after,
            "gt_distance_delta": delta,
        }

    def _policy_patch(
        self,
        step: TrajectoryStep,
        teacher: TeacherAction,
        failure_type: Optional[str],
        episode: TrajectoryEpisode,
    ) -> Dict[str, Any]:
        return {
            "target_scope": episode.target_category or step.state_summary.get("target_category"),
            "trigger_condition": {
                "selected_skill": step.selected_skill,
                "failure_type": failure_type,
                "gt_distance_delta_lte": -self.cfg.gt_progress_min_delta
                if failure_type == "inefficient_exploration"
                else 0.0,
            },
            "recommended_action": teacher.skill_name,
            "rationale": teacher.reason,
            "confidence": teacher.confidence,
            "support_count": 1,
            "source_episode_id": episode.episode_id,
        }

    @staticmethod
    def _gt_distance(step: TrajectoryStep) -> Optional[float]:
        feedback = (step.state_summary or {}).get("gt_feedback") or {}
        value = feedback.get("distance_to_goal")
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def _next_gt_distance(
        self,
        steps: List[TrajectoryStep],
        index: int,
        episode: TrajectoryEpisode,
    ) -> Optional[float]:
        for next_step in steps[index + 1 :]:
            value = self._gt_distance(next_step)
            if value is not None:
                return value
        try:
            return None if episode.final_distance_to_goal is None else float(episode.final_distance_to_goal)
        except (TypeError, ValueError):
            return None

    def _best_target_candidate(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidates = list(state.get("target_candidates") or [])
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.get("confidence") or 0.0)

    def _candidate_stop_ready(self, candidate: Dict[str, Any]) -> bool:
        confidence = candidate.get("confidence") or 0.0
        if confidence < self.cfg.target_stop_threshold:
            return False
        if self.cfg.require_multiview_before_stop and not candidate.get("multi_view_confirmed"):
            return False
        return bool(candidate.get("reachable", True)) and not candidate.get("rejected_false_positive", False)

    @staticmethod
    def _best_frontier(state: Dict[str, Any], prefer_semantic: bool) -> Optional[Dict[str, Any]]:
        frontiers = [
            item
            for item in state.get("frontiers", [])
            if item.get("reachable", True) and not item.get("blocked") and not item.get("low_value")
        ]
        if not frontiers:
            return None
        if prefer_semantic:
            return max(frontiers, key=lambda item: item.get("semantic_score") or 0.0)
        return min(frontiers, key=lambda item: item.get("distance") if item.get("distance") is not None else 1e9)

    @staticmethod
    def _tool_for_skill(skill: Optional[str]) -> str:
        return {
            SkillName.SEMANTIC_EXPLORE.value: "select_semantic_frontier",
            SkillName.GEOMETRIC_EXPLORE.value: "select_nearest_reachable_frontier",
            SkillName.VERIFY_TARGET.value: "verify_target_candidate",
            SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value: "navigate_to_confirmed_target",
            SkillName.RECOVER_FROM_STUCK.value: "recover_from_stuck",
            SkillName.FOLLOW_APEXNAV_PROPOSAL.value: "call_original_apexnav_policy",
            SkillName.FALLBACK_APEXNAV.value: "call_original_apexnav_policy",
        }.get(skill, "call_original_apexnav_policy")
