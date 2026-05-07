from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.schemas import AgentConfig, AgentDecision, SkillName, ValidatorResult
from agent.skill.skill_registry import SkillRegistry


class DecisionValidator:
    def __init__(self, cfg: Optional[AgentConfig] = None, skill_registry: Optional[SkillRegistry] = None) -> None:
        self.cfg = cfg or AgentConfig()
        self.skill_registry = skill_registry

    def validate(
        self,
        decision: AgentDecision,
        state_summary: Dict[str, Any],
        role_memory: Any,
        task_memory: Any,
        working_memory: Any,
        retrieved_lessons: Optional[List[Dict[str, Any]]] = None,
        active_policy_patches: Optional[List[Dict[str, Any]]] = None,
    ) -> ValidatorResult:
        retrieved_lessons = retrieved_lessons or []
        active_policy_patches = active_policy_patches or []
        original = decision.selected_skill

        if not self.cfg.enable_decision_validator:
            return ValidatorResult(
                final_skill=original,
                final_arguments=decision.skill_args,
                original_skill=original,
                accepted=True,
            )

        if self.cfg.force_all_decisions_to_FALLBACK_APEXNAV:
            return ValidatorResult(
                final_skill=SkillName.FALLBACK_APEXNAV.value,
                final_arguments={},
                original_skill=original,
                accepted=True,
                fallback_used=True,
            )

        if self.skill_registry is not None and not self.skill_registry.has_skill(original):
            return self._fallback(original, "unknown skill")

        target_preemption = self._target_preemption_result(decision, state_summary)
        if target_preemption is not None:
            return target_preemption

        if original in {
            SkillName.RECOVER_FROM_STUCK.value,
            SkillName.RETURN_TO_BEST_KNOWN_POINT.value,
        } and self._recent_recovery_active(state_summary):
            return self._reject(
                original,
                SkillName.FALLBACK_APEXNAV.value,
                {},
                "recovery cooldown; use ApexNav fallback before another recovery",
                fallback=True,
            )
        if original == SkillName.RECOVER_FROM_STUCK.value and not self._recovery_precondition_met(state_summary):
            return self._fallback_to_exploration(
                original,
                state_summary,
                "recovery requires objective stuck, collision, timeout, frontier, or planner failure evidence",
            )
        if original == SkillName.RETURN_TO_BEST_KNOWN_POINT.value and not self._return_to_best_precondition_met(state_summary):
            return self._fallback_to_exploration(
                original,
                state_summary,
                "return-to-best requires a valid best known point",
            )

        if self._current_failure_is_degrading(state_summary):
            if original not in {
                SkillName.RECOVER_FROM_STUCK.value,
                SkillName.RETURN_TO_BEST_KNOWN_POINT.value,
                SkillName.FALLBACK_APEXNAV.value,
            }:
                if self._best_known_point(state_summary) and not self.cfg.disable_recover_from_stuck:
                    final_skill = SkillName.RETURN_TO_BEST_KNOWN_POINT.value
                else:
                    final_skill = (
                        SkillName.FALLBACK_APEXNAV.value
                        if self.cfg.disable_recover_from_stuck
                        else SkillName.RECOVER_FROM_STUCK.value
                    )
                return self._reject(
                    original,
                    final_skill,
                    self._recommended_arguments(final_skill, state_summary),
                    "degrading failure requires recovery",
                )

        patch_result = self._validate_policy_patches(decision, state_summary, active_policy_patches)
        if patch_result is not None:
            return patch_result

        memory_result = self._validate_memory_efficiency_lessons(decision, state_summary, retrieved_lessons)
        if memory_result is not None:
            return memory_result

        if original == SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value:
            return self._validate_target_navigation(decision, state_summary, retrieved_lessons)
        if original == SkillName.VERIFY_TARGET.value:
            if self.cfg.disable_verify_target:
                return self._fallback_to_exploration(original, state_summary, "verify target disabled")
            return self._validate_verify_target(decision, state_summary)
        if original == SkillName.SEMANTIC_EXPLORE.value:
            return self._validate_semantic_explore(decision, state_summary)
        if original == SkillName.GEOMETRIC_EXPLORE.value:
            return self._validate_geometric_explore(decision, state_summary)
        if original == SkillName.RETURN_TO_BEST_KNOWN_POINT.value:
            return self._validate_return_to_best_known_point(decision, state_summary)
        if original == SkillName.RECOVER_FROM_STUCK.value:
            if self.cfg.disable_recover_from_stuck:
                return self._reject(original, SkillName.FALLBACK_APEXNAV.value, {}, "recovery skill disabled")
            return ValidatorResult(
                final_skill=original,
                final_arguments=decision.skill_args,
                original_skill=original,
                accepted=True,
            )
        if original == SkillName.FOLLOW_APEXNAV_PROPOSAL.value:
            return ValidatorResult(
                final_skill=original,
                final_arguments=decision.skill_args,
                original_skill=original,
                accepted=True,
            )
        return ValidatorResult(
            final_skill=original,
            final_arguments=decision.skill_args,
            original_skill=original,
            accepted=True,
        )

    def _target_preemption_result(
        self,
        decision: AgentDecision,
        state: Dict[str, Any],
    ) -> Optional[ValidatorResult]:
        candidate = self._candidate_by_id(state, decision.skill_args.get("target_candidate_id"))
        if candidate is None:
            return None
        if candidate.get("rejected_false_positive"):
            return None
        if not candidate.get("reachable", True):
            return None

        original = decision.selected_skill
        confidence = candidate.get("confidence") or 0.0
        confirmed = confidence >= self.cfg.target_stop_threshold
        if self.cfg.require_multiview_before_stop:
            confirmed = confirmed and bool(candidate.get("multi_view_confirmed"))

        if confirmed and original != SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value:
            return ValidatorResult(
                final_skill=SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
                final_arguments={"target_candidate_id": candidate.get("id")},
                original_skill=original,
                accepted=True,
                rejection_reason="confirmed target candidate preempts exploration",
                fallback_used=False,
            )
        if not confirmed and self.cfg.disable_verify_target:
            return None
        if not confirmed and original != SkillName.VERIFY_TARGET.value:
            return ValidatorResult(
                final_skill=SkillName.VERIFY_TARGET.value,
                final_arguments={"target_candidate_id": candidate.get("id")},
                original_skill=original,
                accepted=False,
                rejection_reason="uncertain target candidate preempts exploration",
                fallback_used=False,
            )
        return None

    def _validate_target_navigation(
        self,
        decision: AgentDecision,
        state: Dict[str, Any],
        lessons: List[Dict[str, Any]],
    ) -> ValidatorResult:
        original = decision.selected_skill
        candidate = self._candidate_by_id(state, decision.skill_args.get("target_candidate_id"))
        if candidate is None:
            return self._reject(original, SkillName.FALLBACK_APEXNAV.value, {}, "no target candidate")
        if candidate.get("rejected_false_positive"):
            if self.cfg.disable_verify_target:
                return self._fallback_to_exploration(original, state, "candidate rejected")
            return self._reject(original, SkillName.VERIFY_TARGET.value, {"target_candidate_id": candidate.get("id")}, "candidate rejected")
        if not candidate.get("reachable", True):
            if self.cfg.disable_verify_target:
                return self._fallback_to_exploration(original, state, "candidate unreachable")
            return self._reject(original, SkillName.VERIFY_TARGET.value, {"target_candidate_id": candidate.get("id")}, "candidate unreachable")
        confidence = candidate.get("confidence") or 0.0
        if confidence < self.cfg.target_stop_threshold:
            if self.cfg.disable_verify_target:
                return self._fallback_to_exploration(original, state, "confidence below stop threshold")
            return self._reject(
                original,
                SkillName.VERIFY_TARGET.value,
                {"target_candidate_id": candidate.get("id")},
                "confidence below stop threshold",
            )
        if self.cfg.require_multiview_before_stop and not candidate.get("multi_view_confirmed"):
            if self.cfg.disable_verify_target:
                return self._fallback_to_exploration(original, state, "multiview confirmation required")
            return self._reject(
                original,
                SkillName.VERIFY_TARGET.value,
                {"target_candidate_id": candidate.get("id")},
                "multiview confirmation required",
            )
        if self._memory_blocks_target_stop(candidate, state, lessons):
            if self.cfg.disable_verify_target:
                result = self._fallback_to_exploration(original, state, "retrieved memory blocks target stop")
                result.memory_rule_applied = True
                return result
            return ValidatorResult(
                final_skill=SkillName.VERIFY_TARGET.value,
                final_arguments={"target_candidate_id": candidate.get("id")},
                original_skill=original,
                accepted=False,
                rejection_reason="retrieved memory blocks target stop",
                fallback_used=False,
                memory_rule_applied=True,
            )
        return ValidatorResult(
            final_skill=original,
            final_arguments={"target_candidate_id": candidate.get("id")},
            original_skill=original,
            accepted=True,
        )

    def _validate_verify_target(self, decision: AgentDecision, state: Dict[str, Any]) -> ValidatorResult:
        original = decision.selected_skill
        candidate = self._candidate_by_id(state, decision.skill_args.get("target_candidate_id"))
        if candidate is not None:
            return ValidatorResult(
                final_skill=original,
                final_arguments={"target_candidate_id": candidate.get("id")},
                original_skill=original,
                accepted=True,
            )
        return self._fallback_to_exploration(original, state, "no target candidate to verify")

    def _validate_semantic_explore(self, decision: AgentDecision, state: Dict[str, Any]) -> ValidatorResult:
        original = decision.selected_skill
        requested_id = decision.skill_args.get("frontier_id")
        frontier = self._frontier_by_id(state, requested_id, prefer_semantic=True)
        if frontier is None:
            return self._fallback_to_exploration(original, state, "no reachable semantic frontier")
        if frontier.get("failure_count", 0) >= self.cfg.same_frontier_failure_threshold:
            return self._reject(
                original,
                SkillName.GEOMETRIC_EXPLORE.value,
                {},
                "frontier failed repeatedly",
            )
        if requested_id is not None and not self._same_id(frontier.get("id"), requested_id):
            return self._reject(
                original,
                original,
                {"frontier_id": frontier.get("id")},
                "invalid frontier id replaced with best reachable semantic frontier",
            )
        return ValidatorResult(
            final_skill=original,
            final_arguments={"frontier_id": frontier.get("id")},
            original_skill=original,
            accepted=True,
        )

    def _validate_geometric_explore(self, decision: AgentDecision, state: Dict[str, Any]) -> ValidatorResult:
        original = decision.selected_skill
        requested_id = decision.skill_args.get("frontier_id")
        frontier = self._frontier_by_id(state, requested_id, prefer_semantic=False)
        if frontier is None:
            final_skill = SkillName.FALLBACK_APEXNAV.value if self.cfg.disable_recover_from_stuck else SkillName.RECOVER_FROM_STUCK.value
            return self._reject(original, final_skill, {}, "no reachable frontier")
        if requested_id is not None and not self._same_id(frontier.get("id"), requested_id):
            return self._reject(
                original,
                original,
                {"frontier_id": frontier.get("id")},
                "invalid frontier id replaced with best reachable geometric frontier",
            )
        return ValidatorResult(
            final_skill=original,
            final_arguments={"frontier_id": frontier.get("id")},
            original_skill=original,
            accepted=True,
        )

    def _validate_return_to_best_known_point(self, decision: AgentDecision, state: Dict[str, Any]) -> ValidatorResult:
        original = decision.selected_skill
        if self.cfg.disable_recover_from_stuck:
            return self._reject(original, SkillName.FALLBACK_APEXNAV.value, {}, "recovery skills disabled")
        best = self._best_known_point(state)
        if best is None:
            return self._fallback_to_exploration(original, state, "best known point unavailable")
        if not self._return_to_best_precondition_met(state):
            return self._fallback_to_exploration(
                original,
                state,
                "return-to-best requires a valid best known point",
            )
        args = {
            "best_known_point": best.get("waypoint"),
            "best_known_timestep": best.get("timestep"),
            "best_known_score": best.get("score"),
        }
        return ValidatorResult(
            final_skill=original,
            final_arguments={k: v for k, v in args.items() if v is not None},
            original_skill=original,
            accepted=True,
        )

    def _validate_policy_patches(
        self,
        decision: AgentDecision,
        state: Dict[str, Any],
        active_policy_patches: List[Dict[str, Any]],
    ) -> Optional[ValidatorResult]:
        for patch in active_policy_patches:
            if patch.get("active", True) is False:
                continue
            action = patch.get("recommended_action") or patch.get("action")
            if not action or action == decision.selected_skill:
                continue
            if not self._policy_patch_matches(patch, decision, state):
                continue
            if action == SkillName.VERIFY_TARGET.value and self.cfg.disable_verify_target:
                result = self._fallback_to_exploration(
                    decision.selected_skill,
                    state,
                    "active policy patch recommended disabled VERIFY_TARGET",
                )
                result.policy_patch_applied = True
                return result
            return ValidatorResult(
                final_skill=action,
                final_arguments=self._recommended_arguments(action, state),
                original_skill=decision.selected_skill,
                accepted=False,
                rejection_reason="active policy patch applied",
                fallback_used=action == SkillName.FALLBACK_APEXNAV.value,
                policy_patch_applied=True,
            )
        return None

    def _validate_memory_efficiency_lessons(
        self,
        decision: AgentDecision,
        state: Dict[str, Any],
        lessons: List[Dict[str, Any]],
    ) -> Optional[ValidatorResult]:
        if decision.selected_skill not in {
            SkillName.GEOMETRIC_EXPLORE.value,
            SkillName.SEMANTIC_EXPLORE.value,
        }:
            return None
        if state.get("target_candidates"):
            return None
        if (state.get("semantic_score_stats") or {}).get("has_clear_peak"):
            return None

        target = state.get("target_category")
        if not target:
            return None
        matches = []
        for lesson in lessons:
            lesson_target = lesson.get("target_category")
            if lesson_target != target:
                continue
            if lesson.get("failure_type") != "inefficient_exploration":
                continue
            if not (
                self._lesson_recommends_skill(lesson, SkillName.FOLLOW_APEXNAV_PROPOSAL.value)
                or self._lesson_recommends_skill(lesson, SkillName.FALLBACK_APEXNAV.value)
            ):
                continue
            if float(lesson.get("confidence") or 0.0) < 0.65:
                continue
            matches.append(lesson)

        if len(matches) < 2:
            return None
        return ValidatorResult(
            final_skill=SkillName.FOLLOW_APEXNAV_PROPOSAL.value,
            final_arguments={},
            original_skill=decision.selected_skill,
            accepted=False,
            rejection_reason="retrieved memory indicates repeated inefficient exploration; follow ApexNav proposal",
            fallback_used=False,
            memory_rule_applied=True,
        )

    def _policy_patch_matches(self, patch: Dict[str, Any], decision: AgentDecision, state: Dict[str, Any]) -> bool:
        scope = patch.get("target_scope") or patch.get("target_category")
        target = state.get("target_category")
        if scope not in {None, "", "global", target}:
            return False
        condition = patch.get("trigger_condition") or patch.get("condition") or {}
        if isinstance(condition, str):
            return self._condition_string_matches(condition, decision, state)
        if not isinstance(condition, dict):
            return False
        selected_skill = condition.get("selected_skill")
        if selected_skill and selected_skill != decision.selected_skill:
            return False
        failure_type = condition.get("failure_type")
        recent_failures = (state.get("navigation_history") or {}).get("recent_failures") or []
        if failure_type and failure_type != state.get("failure_type") and failure_type not in recent_failures:
            if (
                self.cfg.enable_stuck_recovery_override
                and failure_type == "planner_stuck"
                and self._stuck_or_collision_over_threshold(state)
            ):
                return True
            return False
        if "semantic_peak" in condition:
            has_peak = bool((state.get("semantic_score_stats") or {}).get("has_clear_peak"))
            if has_peak != bool(condition.get("semantic_peak")):
                return False
        if "steps_left_lt" in condition:
            steps_left = (state.get("navigation_history") or {}).get("steps_left")
            if steps_left is None or not float(steps_left) < float(condition["steps_left_lt"]):
                return False
        candidate = self._candidate_by_id(state, decision.skill_args.get("target_candidate_id"))
        if "target_confidence_lt" in condition:
            confidence = (candidate or {}).get("confidence") or 0.0
            if not confidence < float(condition["target_confidence_lt"]):
                return False
        if "target_confidence_gte" in condition:
            confidence = (candidate or {}).get("confidence") or 0.0
            if not confidence >= float(condition["target_confidence_gte"]):
                return False
        if "multi_view_confirmed" in condition:
            if bool((candidate or {}).get("multi_view_confirmed")) != bool(condition["multi_view_confirmed"]):
                return False
        if "repeated_failures_gte" in condition and len(recent_failures) < int(condition["repeated_failures_gte"]):
            return False
        return True

    def _condition_string_matches(self, condition: str, decision: AgentDecision, state: Dict[str, Any]) -> bool:
        parts = [part.strip() for part in condition.split("AND") if part.strip()]
        for part in parts:
            lower = part.lower().replace(" ", "")
            if lower.startswith("semantic_peak="):
                expected = lower.split("=", 1)[1] == "true"
                actual = bool((state.get("semantic_score_stats") or {}).get("has_clear_peak"))
                if actual != expected:
                    return False
            elif lower.startswith("steps_left<"):
                value = float(lower.split("<", 1)[1])
                steps_left = (state.get("navigation_history") or {}).get("steps_left")
                if steps_left is None or not float(steps_left) < value:
                    return False
            elif lower.startswith("selected_skill="):
                expected = part.split("=", 1)[1].strip()
                if decision.selected_skill != expected:
                    return False
            else:
                return False
        return True

    def _recommended_arguments(self, action: str, state: Dict[str, Any]) -> Dict[str, Any]:
        if action in {SkillName.VERIFY_TARGET.value, SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value}:
            candidate = self._candidate_by_id(state, None)
            return {"target_candidate_id": candidate.get("id")} if candidate else {}
        if action == SkillName.SEMANTIC_EXPLORE.value:
            frontier = self._frontier_by_id(state, None, prefer_semantic=True)
            return {"frontier_id": frontier.get("id")} if frontier else {}
        if action == SkillName.GEOMETRIC_EXPLORE.value:
            frontier = self._frontier_by_id(state, None, prefer_semantic=False)
            return {"frontier_id": frontier.get("id")} if frontier else {}
        if action == SkillName.RETURN_TO_BEST_KNOWN_POINT.value:
            best = self._best_known_point(state)
            return {"best_known_point": best.get("waypoint")} if best else {}
        return {}

    @staticmethod
    def _lesson_recommends_skill(lesson: Dict[str, Any], skill: str) -> bool:
        suggested = lesson.get("suggested_policy_patch") or lesson.get("policy_patch_proposal") or {}
        action = suggested.get("recommended_action") or suggested.get("action")
        if action == skill:
            return True
        teacher = lesson.get("teacher_action") or {}
        if teacher.get("skill_name") == skill:
            return True
        better = lesson.get("better_decision")
        return isinstance(better, str) and skill in better

    def _fallback_to_exploration(self, original: str, state: Dict[str, Any], reason: str) -> ValidatorResult:
        if self._reachable_frontiers(state):
            stats = state.get("semantic_score_stats") or {}
            skill = SkillName.SEMANTIC_EXPLORE.value if stats.get("has_clear_peak") else SkillName.GEOMETRIC_EXPLORE.value
            return self._reject(original, skill, self._recommended_arguments(skill, state), reason)
        return self._fallback(original, reason)

    def _fallback(self, original: str, reason: str) -> ValidatorResult:
        return self._reject(original, SkillName.FALLBACK_APEXNAV.value, {}, reason, fallback=True)

    @staticmethod
    def _reject(
        original: str,
        final_skill: str,
        args: Dict[str, Any],
        reason: str,
        fallback: bool = False,
    ) -> ValidatorResult:
        return ValidatorResult(
            final_skill=final_skill,
            final_arguments=args,
            accepted=False,
            rejection_reason=reason,
            original_skill=original,
            fallback_used=fallback or final_skill == SkillName.FALLBACK_APEXNAV.value,
        )

    def _candidate_by_id(self, state: Dict[str, Any], candidate_id: Any) -> Optional[Dict[str, Any]]:
        candidates = state.get("target_candidates") or []
        if candidate_id is not None:
            for candidate in candidates:
                if self._same_id(candidate.get("id"), candidate_id):
                    return candidate
        valid = [
            c
            for c in candidates
            if not c.get("rejected_false_positive") and c.get("reachable", True)
        ]
        return max(valid, key=lambda item: item.get("confidence") or 0.0) if valid else None

    def _frontier_by_id(self, state: Dict[str, Any], frontier_id: Any, prefer_semantic: bool) -> Optional[Dict[str, Any]]:
        frontiers = self._reachable_frontiers(state)
        if frontier_id is not None:
            for frontier in frontiers:
                if self._same_id(frontier.get("id"), frontier_id):
                    return frontier
        if not frontiers:
            return None
        if prefer_semantic:
            return max(frontiers, key=lambda item: item.get("semantic_score") or 0.0)
        return min(frontiers, key=lambda item: item.get("distance") if item.get("distance") is not None else 1e9)

    @staticmethod
    def _same_id(left: Any, right: Any) -> bool:
        return left == right or (left is not None and right is not None and str(left) == str(right))

    @staticmethod
    def _reachable_frontiers(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            frontier
            for frontier in state.get("frontiers", [])
            if frontier.get("reachable", True)
            and not frontier.get("blocked")
            and not frontier.get("low_value")
        ]

    def _current_failure_is_degrading(self, state: Dict[str, Any]) -> bool:
        failure_class = state.get("failure_class")
        if failure_class == "degrading":
            return True
        if DecisionValidator._recent_recovery_active(state):
            return False
        if self.cfg.enable_stuck_recovery_override and DecisionValidator._stuck_or_collision_over_threshold(state):
            return True
        failures = (state.get("navigation_history") or {}).get("recent_failures") or []
        return bool(failures and failures[-1] in {"false_positive_stop", "repeated_bad_frontier", "planner_stuck"})

    @staticmethod
    def _stuck_or_collision_over_threshold(state: Dict[str, Any]) -> bool:
        history = state.get("navigation_history") or {}
        stuck_count = int(history.get("stuck_count") or 0)
        collision_count = int(history.get("collision_count") or 0)
        return stuck_count >= 3 or collision_count >= 3

    def _recovery_precondition_met(self, state: Dict[str, Any]) -> bool:
        history = state.get("navigation_history") or {}
        stuck_count = int(history.get("stuck_count") or 0)
        collision_count = int(history.get("collision_count") or 0)
        if stuck_count >= self.cfg.stuck_threshold or collision_count >= self.cfg.stuck_threshold:
            return True
        if state.get("failure_class") == "degrading":
            return True
        if state.get("failure_type") in {
            "planner_stuck",
            "repeated_collision",
            "no_frontier_deadend",
            "repeated_bad_frontier",
            "timeout",
            "waypoint_timeout",
            "frontier_failure",
            "low_level_planner_failure",
        }:
            return True
        recent_failures = set(history.get("recent_failures") or [])
        if recent_failures & {
            "planner_stuck",
            "repeated_collision",
            "no_frontier_deadend",
            "repeated_bad_frontier",
            "timeout",
            "waypoint_timeout",
            "frontier_failure",
            "low_level_planner_failure",
        }:
            return True
        return not bool(self._reachable_frontiers(state))

    @staticmethod
    def _recent_recovery_active(state: Dict[str, Any]) -> bool:
        skills = (state.get("navigation_history") or {}).get("recent_selected_skills") or []
        return bool(
            skills
            and skills[-1]
            in {
                SkillName.RECOVER_FROM_STUCK.value,
                SkillName.RETURN_TO_BEST_KNOWN_POINT.value,
            }
        )

    def _return_to_best_precondition_met(self, state: Dict[str, Any]) -> bool:
        return self._best_known_point(state) is not None

    def _gt_deviation_signal(self, state: Dict[str, Any]) -> bool:
        reflection = state.get("online_gt_deviation_reflection") or {}
        if reflection.get("triggered"):
            return True
        gt = state.get("gt_feedback") or {}
        trajectory = gt.get("gt_trajectory") if isinstance(gt.get("gt_trajectory"), dict) else {}
        distance = trajectory.get("distance_to_gt_path")
        try:
            return distance is not None and float(distance) >= float(self.cfg.gt_path_deviation_threshold)
        except (TypeError, ValueError):
            return False

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

    def _memory_blocks_target_stop(
        self,
        candidate: Dict[str, Any],
        state: Dict[str, Any],
        lessons: List[Dict[str, Any]],
    ) -> bool:
        target = state.get("target_category")
        confidence = candidate.get("confidence") or 0.0
        single_view = not candidate.get("multi_view_confirmed") or candidate.get("num_views", 0) <= 1
        for lesson in lessons:
            if target and lesson.get("target_category") not in {target, None}:
                continue
            text = (lesson.get("lesson") or "").lower()
            failure_type = lesson.get("failure_type")
            if failure_type == "false_positive_stop" and single_view:
                return True
            if "do not stop" in text and ("single-view" in text or "single view" in text) and single_view:
                return True
            if "confidence" in text and "0.75" in text and confidence < self.cfg.target_stop_threshold:
                return True
        return False
