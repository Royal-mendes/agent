from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agent.reflection.failure_taxonomy import classify_failure
from agent.schemas import SkillName
from agent.learning.trajectory_schema import (
    TeacherAction,
    ToolUseLearningSample,
    TrajectoryEpisode,
    TrajectoryStep,
)


class HindsightLabeler:
    """Convert trace disagreements into structured tool-use lessons."""

    def label(
        self,
        student_episode: TrajectoryEpisode,
        teacher_episode: Optional[TrajectoryEpisode] = None,
        gt_episode: Optional[TrajectoryEpisode] = None,
    ) -> List[ToolUseLearningSample]:
        samples: List[ToolUseLearningSample] = []
        samples.extend(self._label_student_corrections(student_episode))
        if teacher_episode is not None:
            sample = self._label_baseline_comparison(student_episode, teacher_episode)
            if sample is not None:
                samples.append(sample)
        if gt_episode is not None:
            sample = self._label_gt_progress(student_episode, gt_episode)
            if sample is not None:
                samples.append(sample)
        return samples

    def _label_student_corrections(self, episode: TrajectoryEpisode) -> List[ToolUseLearningSample]:
        samples = []
        stuck_sample_added = False
        validator_sample_keys = set()
        for step in episode.trajectory_steps:
            if not stuck_sample_added and self._should_recover_from_stuck(step):
                teacher = TeacherAction(
                    skill_name=SkillName.RECOVER_FROM_STUCK.value,
                    tool_name="recover_from_stuck",
                    arguments={"blocked_frontier_id": self._last_selected_frontier(step)},
                    reason="Navigation history indicates repeated stuck or collision signals.",
                    confidence=0.82,
                    source="self_hindsight",
                )
                samples.append(
                    self._sample(
                        episode,
                        step,
                        teacher,
                        outcome="student_repeated_retry_while_stuck",
                        failure_type="planner_stuck",
                        lesson=(
                            "When stuck_count reaches the recovery threshold, do not keep retrying the same "
                            "ordinary exploration waypoint; call RECOVER_FROM_STUCK or fall back to ApexNav recovery."
                        ),
                        confidence=0.82,
                        metadata={"stuck_count": self._stuck_count(step)},
                    )
                )
                stuck_sample_added = True

            if step.selected_skill == SkillName.VERIFY_TARGET.value and not self._target_candidates(step):
                teacher = self._exploration_teacher_action(step.state_summary, source="self_hindsight")
                samples.append(
                    self._sample(
                        episode,
                        step,
                        teacher,
                        outcome="student_invalid_tool_call",
                        failure_type="unconfirmed_target_candidate",
                        lesson=(
                            "When no target candidate exists, do not call VERIFY_TARGET; "
                            "choose a valid exploration tool from the current frontier evidence."
                        ),
                        confidence=0.78,
                    )
                )
                continue

            if step.selected_skill == SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value:
                candidate = self._best_target_candidate(step)
                if candidate and self._candidate_needs_verification(candidate):
                    teacher = TeacherAction(
                        skill_name=SkillName.VERIFY_TARGET.value,
                        tool_name="verify_target_candidate",
                        arguments={"target_candidate_id": candidate.get("id")},
                        reason="Target evidence is below stop requirements or single-view.",
                        confidence=0.84,
                        source="self_hindsight",
                    )
                    samples.append(
                        self._sample(
                            episode,
                            step,
                            teacher,
                            outcome="premature_target_navigation",
                            failure_type="false_positive_stop",
                            lesson=(
                                "Before final target navigation, verify low-confidence or single-view "
                                "target candidates instead of stopping early."
                            ),
                            confidence=0.84,
                        )
                    )
                    continue

            validator = step.validator_result or {}
            if validator and not validator.get("accepted", True):
                final_skill = validator.get("final_skill") or SkillName.FALLBACK_APEXNAV.value
                rejection_reason = validator.get("rejection_reason")
                correction_key = (step.selected_skill, final_skill, rejection_reason or step.failure_type)
                if correction_key in validator_sample_keys:
                    continue
                if final_skill == SkillName.RECOVER_FROM_STUCK.value and stuck_sample_added:
                    continue
                validator_sample_keys.add(correction_key)
                teacher = TeacherAction(
                    skill_name=final_skill,
                    tool_name=self._tool_for_skill(final_skill),
                    arguments=validator.get("final_arguments") or {},
                    reason=f"Validator corrected invalid decision: {rejection_reason}",
                    confidence=0.72,
                    source="validator_hindsight",
                )
                samples.append(
                    self._sample(
                        episode,
                        step,
                        teacher,
                        outcome="validator_corrected_student",
                        failure_type=step.failure_type or "validator_rejection",
                        lesson=(
                            "Treat validator corrections as supervised feedback: under this state, "
                            f"prefer {final_skill} over the rejected tool call."
                        ),
                        confidence=0.72,
                    )
                )
        return samples

    def _label_baseline_comparison(
        self,
        student: TrajectoryEpisode,
        baseline: TrajectoryEpisode,
    ) -> Optional[ToolUseLearningSample]:
        baseline_better, comparison_reason = self._baseline_is_better(student, baseline)
        if not baseline_better:
            return None
        step = student.trajectory_steps[0] if student.trajectory_steps else TrajectoryStep(timestep=0)
        teacher = TeacherAction(
            skill_name=SkillName.FALLBACK_APEXNAV.value,
            tool_name="call_original_apexnav_policy",
            arguments={},
            reason=f"The ApexNav baseline is a better teacher for this episode: {comparison_reason}.",
            confidence=0.7,
            source=baseline.source,
        )
        return self._sample(
            student,
            step,
            teacher,
            outcome="baseline_teacher_better",
            failure_type=student.failure_type or ("missing_target" if not student.success else "inefficient_exploration"),
            lesson=(
                "When the reflective agent is less efficient than the ApexNav baseline, prefer the baseline "
                "fallback until stronger semantic or target evidence justifies a different high-level skill."
            ),
            confidence=0.7,
            metadata={
                "baseline_steps": baseline.steps,
                "student_steps": student.steps,
                "baseline_spl": baseline.spl,
                "student_spl": student.spl,
                "baseline_success": baseline.success,
                "comparison_reason": comparison_reason,
            },
        )

    def _label_gt_progress(
        self,
        student: TrajectoryEpisode,
        gt: TrajectoryEpisode,
    ) -> Optional[ToolUseLearningSample]:
        if not gt.trajectory_steps:
            return None
        if student.success and student.steps is not None and gt.steps is not None and student.steps <= gt.steps * 1.5:
            return None
        step = student.trajectory_steps[0] if student.trajectory_steps else TrajectoryStep(timestep=0)
        gt_step = gt.trajectory_steps[min(len(gt.trajectory_steps) - 1, max(0, int(step.timestep or 0)))]
        teacher = TeacherAction(
            skill_name=gt_step.selected_skill or SkillName.GEOMETRIC_EXPLORE.value,
            tool_name=gt_step.tool_name or "select_oracle_progress_waypoint",
            arguments=dict(gt_step.skill_args),
            waypoint=gt_step.waypoint,
            reason="GT trajectory indicates a progress waypoint for this navigation state.",
            confidence=0.9,
            source=gt.source,
        )
        return self._sample(
            student,
            step,
            teacher,
            outcome="gt_oracle_progress",
            failure_type=student.failure_type or "missing_target",
            lesson=(
                "Use GT hindsight to identify progress waypoints: if exploration drifts away from the "
                "oracle route, prefer frontiers or waypoints that reduce distance to the GT path."
            ),
            confidence=0.9,
            metadata={"gt_episode_id": gt.episode_id},
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolUseLearningSample:
        failure_class = classify_failure(failure_type)
        return ToolUseLearningSample(
            source=teacher.source,
            episode_id=episode.episode_id,
            scene_id=episode.scene_id,
            split=episode.split,
            target_category=episode.target_category or step.state_summary.get("target_category"),
            timestep=step.timestep,
            state_condition=self._state_condition(step),
            student_action={
                "skill": step.selected_skill,
                "tool": step.tool_name,
                "arguments": step.skill_args,
                "validator_result": step.validator_result,
            },
            teacher_action=teacher.to_dict(),
            outcome=outcome,
            failure_type=failure_type,
            failure_class=failure_class,
            lesson=lesson,
            policy_patch_proposal=self._policy_patch(step, teacher, failure_type, episode),
            confidence=confidence,
            metadata=metadata or {},
        )

    def _exploration_teacher_action(self, state: Dict[str, Any], source: str) -> TeacherAction:
        stats = state.get("semantic_score_stats") or {}
        if stats.get("has_clear_peak"):
            frontier = self._best_frontier(state, prefer_semantic=True)
            return TeacherAction(
                skill_name=SkillName.SEMANTIC_EXPLORE.value,
                tool_name="select_semantic_frontier",
                arguments={"frontier_id": None if frontier is None else frontier.get("id")},
                reason="Semantic score has a clear peak and no target candidate is available.",
                confidence=0.78,
                source=source,
            )
        frontier = self._best_frontier(state, prefer_semantic=False)
        return TeacherAction(
            skill_name=SkillName.GEOMETRIC_EXPLORE.value,
            tool_name="select_nearest_reachable_frontier",
            arguments={"frontier_id": None if frontier is None else frontier.get("id")},
            reason="No target candidate is available, so continue geometric exploration.",
            confidence=0.68,
            source=source,
        )

    @staticmethod
    def _policy_patch(
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
            },
            "recommended_action": teacher.skill_name,
            "rationale": teacher.reason,
            "confidence": teacher.confidence,
            "support_count": 1,
            "source_episode_id": episode.episode_id,
        }

    @staticmethod
    def _state_condition(step: TrajectoryStep) -> Dict[str, Any]:
        state = step.state_summary or {}
        history = state.get("navigation_history") or {}
        candidates = state.get("target_candidates") or []
        best_candidate = max(candidates, key=lambda item: item.get("confidence") or 0.0) if candidates else {}
        return {
            "semantic_peak": (state.get("semantic_score_stats") or {}).get("has_clear_peak"),
            "target_candidate_count": len(candidates),
            "target_confidence": best_candidate.get("confidence"),
            "multi_view_confirmed": best_candidate.get("multi_view_confirmed", False),
            "frontier_count": len(state.get("frontiers") or []),
            "stuck_count": history.get("stuck_count"),
            "collision_count": history.get("collision_count"),
            "student_selected_skill": step.selected_skill,
        }

    @staticmethod
    def _should_recover_from_stuck(step: TrajectoryStep) -> bool:
        if step.selected_skill in {SkillName.RECOVER_FROM_STUCK.value, SkillName.FALLBACK_APEXNAV.value}:
            return False
        if step.failure_type in {"planner_stuck", "repeated_collision", "no_frontier_deadend", "repeated_bad_frontier"}:
            return True
        history = (step.state_summary or {}).get("navigation_history") or {}
        recent_failures = set(history.get("recent_failures") or [])
        return bool(recent_failures & {"planner_stuck", "repeated_collision", "no_frontier_deadend", "repeated_bad_frontier"})

    @staticmethod
    def _stuck_count(step: TrajectoryStep) -> int:
        history = (step.state_summary or {}).get("navigation_history") or {}
        return int(history.get("stuck_count") or 0)

    @staticmethod
    def _last_selected_frontier(step: TrajectoryStep) -> Optional[Any]:
        for frontier in (step.state_summary or {}).get("frontiers") or []:
            if frontier.get("last_selected"):
                return frontier.get("id")
        return step.skill_args.get("frontier_id")

    @staticmethod
    def _target_candidates(step: TrajectoryStep) -> List[Dict[str, Any]]:
        return list((step.state_summary or {}).get("target_candidates") or [])

    @staticmethod
    def _best_target_candidate(step: TrajectoryStep) -> Optional[Dict[str, Any]]:
        candidates = HindsightLabeler._target_candidates(step)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.get("confidence") or 0.0)

    @staticmethod
    def _candidate_needs_verification(candidate: Dict[str, Any]) -> bool:
        confidence = candidate.get("confidence") or 0.0
        return confidence < 0.75 or not candidate.get("multi_view_confirmed") or candidate.get("num_views", 0) <= 1

    @staticmethod
    def _baseline_is_better(student: TrajectoryEpisode, baseline: TrajectoryEpisode) -> Tuple[bool, str]:
        if baseline.success and not student.success:
            return True, "baseline succeeded while the reflective agent failed"
        if not baseline.success:
            return False, "baseline did not succeed"
        if baseline.spl is not None and student.spl is not None:
            if baseline.spl >= student.spl + 0.05:
                return True, f"baseline SPL {baseline.spl:.3f} exceeds reflective SPL {student.spl:.3f}"
            if student.spl > baseline.spl:
                return False, f"reflective SPL {student.spl:.3f} is not worse than baseline SPL {baseline.spl:.3f}"
        if baseline.steps is not None and student.steps is not None:
            min_step_gain = max(5, int(student.steps * 0.1))
            if baseline.steps + min_step_gain <= student.steps:
                return True, f"baseline used fewer steps ({baseline.steps} < {student.steps})"
        return False, "baseline was not measurably better"

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
            SkillName.FALLBACK_APEXNAV.value: "call_original_apexnav_policy",
        }.get(skill, "call_original_apexnav_policy")
