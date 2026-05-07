from __future__ import annotations

from typing import Any, Dict, Optional

from agent.execution.postcondition_checker import PostconditionChecker
from agent.reflection.failure_taxonomy import classify_failure, suggest_recovery_skill
from agent.schemas import AgentConfig, SkillExecutionResult, SkillName, SkillStatus
from agent.skill.skill_registry import SkillRegistry


class MonitoredNavigationSkillExecutor:
    def __init__(
        self,
        cfg: Optional[AgentConfig] = None,
        skill_registry: Optional[SkillRegistry] = None,
        postcondition_checker: Optional[PostconditionChecker] = None,
    ) -> None:
        self.cfg = cfg or AgentConfig()
        self.skill_registry = skill_registry
        self.postcondition_checker = postcondition_checker or PostconditionChecker(self.cfg)

    def execute(self, skill_name: str, skill_args: Dict[str, Any], context: Any) -> SkillExecutionResult:
        start_snapshot = self._snapshot(context)
        start_timestep = start_snapshot.get("timestep")

        if self.skill_registry is None or not self.skill_registry.has_skill(skill_name):
            return SkillExecutionResult(
                skill_name=skill_name,
                status=SkillStatus.REJECTED.value,
                start_timestep=start_timestep,
                end_timestep=start_timestep,
                precondition_passed=False,
                failure_type="unknown_skill",
            )

        handler = self.skill_registry.get_handler(skill_name)
        if handler is None:
            return SkillExecutionResult(
                skill_name=skill_name,
                status=SkillStatus.REJECTED.value,
                start_timestep=start_timestep,
                end_timestep=start_timestep,
                precondition_passed=False,
                failure_type="skill_handler_missing",
            )

        try:
            result = handler(skill_args or {}, context)
        except Exception as exc:  # pragma: no cover - safety net for runtime integration
            return SkillExecutionResult(
                skill_name=skill_name,
                status=SkillStatus.FAILED.value,
                start_timestep=start_timestep,
                end_timestep=start_timestep,
                precondition_passed=True,
                failure_type="skill_exception",
                raw_metadata={"exception": repr(exc)},
            )

        if not isinstance(result, SkillExecutionResult):
            result = SkillExecutionResult(skill_name=skill_name, raw_metadata={"raw_result": result})

        end_snapshot = self._snapshot(context)
        result.start_timestep = result.start_timestep if result.start_timestep is not None else start_timestep
        result.end_timestep = result.end_timestep if result.end_timestep is not None else end_snapshot.get("timestep")

        self._apply_monitoring(skill_name, result, start_snapshot, end_snapshot)
        postcondition, reason = self.postcondition_checker.check(
            skill_name, result, start_snapshot, end_snapshot
        )
        result.postcondition_passed = postcondition
        result.raw_metadata.setdefault("postcondition_reason", reason)

        if postcondition is False and result.status == SkillStatus.SUCCESS.value:
            result.status = SkillStatus.FAILED.value
            result.failure_type = result.failure_type or self._default_failure_for_skill(skill_name)
            result.failure_class = result.failure_class or self._failure_class(result.failure_type)
            result.recovery_skill_suggested = result.recovery_skill_suggested or self._recovery_for_failure(result.failure_type)

        self._apply_side_effect_markers(skill_name, result, context)
        return result

    def _apply_monitoring(
        self,
        skill_name: str,
        result: SkillExecutionResult,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> None:
        if result.status in {SkillStatus.TIMEOUT.value, SkillStatus.FAILED.value} and result.failure_type:
            result.failure_class = result.failure_class or self._failure_class(result.failure_type)
            return

        timeout_steps = (result.raw_metadata or {}).get("timeout_steps")
        if timeout_steps is not None and result.start_timestep is not None and result.end_timestep is not None:
            if result.end_timestep - result.start_timestep >= timeout_steps:
                result.status = SkillStatus.TIMEOUT.value
                result.failure_type = "timeout"

        if after.get("stuck_count", 0) >= self.cfg.stuck_threshold:
            result.status = SkillStatus.FAILED.value
            result.failure_type = result.failure_type or "planner_stuck"

        if (
            skill_name in {SkillName.SEMANTIC_EXPLORE.value, SkillName.GEOMETRIC_EXPLORE.value}
            and self._delta(after, before, "explored_area") <= self.cfg.low_information_gain_threshold
            and result.raw_metadata.get("monitor_information_gain", False)
        ):
            result.status = SkillStatus.FAILED.value
            result.failure_type = result.failure_type or "low_information_gain"

        if (
            skill_name == SkillName.VERIFY_TARGET.value
            and self._has(before, after, "target_confidence")
            and self._delta(after, before, "target_confidence") < 0
        ):
            result.status = SkillStatus.FAILED.value
            result.failure_type = result.failure_type or "false_positive_candidate"

        if result.raw_metadata.get("frontier_failure_count", 0) >= self.cfg.same_frontier_failure_threshold:
            result.status = SkillStatus.FAILED.value
            result.failure_type = result.failure_type or "repeated_bad_frontier"

        if result.failure_type:
            result.failure_class = result.failure_class or self._failure_class(result.failure_type)
            result.recovery_skill_suggested = result.recovery_skill_suggested or self._recovery_for_failure(result.failure_type)

    def _apply_side_effect_markers(self, skill_name: str, result: SkillExecutionResult, context: Any) -> None:
        if not result.failure_type:
            return
        frontier_id = result.selected_frontier_id
        if frontier_id is not None:
            if result.failure_type in {"low_information_gain", "repeated_bad_frontier"}:
                self._call_optional(context, "mark_frontier_low_value", frontier_id)
                result.memory_updates.append({"type": "mark_frontier_low_value", "frontier_id": frontier_id})
            if result.failure_type in {"timeout", "planner_stuck", "unreachable_waypoint"}:
                self._call_optional(context, "mark_frontier_blocked", frontier_id)
                result.memory_updates.append({"type": "mark_frontier_blocked", "frontier_id": frontier_id})

        target_id = result.target_candidate_id
        if target_id is not None and result.failure_type in {"false_positive_candidate", "false_positive_stop"}:
            self._call_optional(context, "mark_target_candidate_rejected", target_id)
            result.memory_updates.append(
                {"type": "mark_target_candidate_rejected", "target_candidate_id": target_id}
            )

    @staticmethod
    def _call_optional(context: Any, method_name: str, *args: Any) -> None:
        if hasattr(context, method_name):
            getattr(context, method_name)(*args)

    @staticmethod
    def _snapshot(context: Any) -> Dict[str, Any]:
        if hasattr(context, "get_navigation_monitor_state"):
            data = context.get_navigation_monitor_state()
            return dict(data or {})
        if hasattr(context, "monitor_state"):
            return dict(getattr(context, "monitor_state") or {})
        if isinstance(context, dict):
            return dict(context.get("monitor_state") or context)
        return {}

    @staticmethod
    def _delta(after: Dict[str, Any], before: Dict[str, Any], key: str) -> float:
        try:
            return float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _has(before: Dict[str, Any], after: Dict[str, Any], key: str) -> bool:
        return key in before and key in after

    @staticmethod
    def _default_failure_for_skill(skill_name: str) -> str:
        if skill_name == SkillName.SEMANTIC_EXPLORE.value:
            return "semantic_explore_no_progress"
        if skill_name == SkillName.GEOMETRIC_EXPLORE.value:
            return "geometric_explore_no_progress"
        if skill_name == SkillName.VERIFY_TARGET.value:
            return "unconfirmed_target_candidate"
        if skill_name == SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value:
            return "false_positive_stop"
        if skill_name == SkillName.RETURN_TO_BEST_KNOWN_POINT.value:
            return "return_to_best_point_failed"
        if skill_name == SkillName.RECOVER_FROM_STUCK.value:
            return "planner_stuck"
        return "skill_failed"

    def _failure_class(self, failure_type: str) -> str:
        return classify_failure(failure_type, cfg=self.cfg)

    def _recovery_for_failure(self, failure_type: str) -> str:
        return suggest_recovery_skill(failure_type, self.cfg.disable_recover_from_stuck)
