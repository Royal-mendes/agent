from __future__ import annotations

from typing import Any, Dict, Optional

from agent.schemas import SkillExecutionResult, SkillName, SkillSpec, SkillStatus
from agent.skill.skill_registry import SkillRegistry


class ApexNavToolAdapter:
    """Thin wrapper around original ApexNav high-level tools.

    The adapter expects a context object with methods named below. Tests can use
    a fake context; a ROS/C++ bridge can expose the same surface later.
    """

    def __init__(self, context: Any) -> None:
        self.context = context

    def call_original_apexnav_policy(self, skill_args: Dict[str, Any]) -> SkillExecutionResult:
        return self._call("call_original_apexnav_policy", skill_args, SkillName.FALLBACK_APEXNAV.value)

    def follow_apexnav_proposal(self, skill_args: Dict[str, Any]) -> SkillExecutionResult:
        return self._call(
            "call_original_apexnav_policy",
            skill_args,
            SkillName.FOLLOW_APEXNAV_PROPOSAL.value,
        )

    def select_semantic_frontier(self, skill_args: Dict[str, Any]) -> SkillExecutionResult:
        return self._call("select_semantic_frontier", skill_args, SkillName.SEMANTIC_EXPLORE.value)

    def select_nearest_reachable_frontier(self, skill_args: Dict[str, Any]) -> SkillExecutionResult:
        return self._call("select_nearest_reachable_frontier", skill_args, SkillName.GEOMETRIC_EXPLORE.value)

    def navigate_to_confirmed_target(self, skill_args: Dict[str, Any]) -> SkillExecutionResult:
        return self._call(
            "navigate_to_confirmed_target",
            skill_args,
            SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
        )

    def verify_target(self, skill_args: Dict[str, Any]) -> SkillExecutionResult:
        if hasattr(self.context, "verify_target"):
            return self._call("verify_target", skill_args, SkillName.VERIFY_TARGET.value)
        result = self._call(
            "navigate_to_target_observation_viewpoint",
            skill_args,
            SkillName.VERIFY_TARGET.value,
            allow_missing=True,
        )
        result.raw_metadata.setdefault("observe_action_unavailable", True)
        return result

    def recover_from_stuck(self, skill_args: Dict[str, Any]) -> SkillExecutionResult:
        if hasattr(self.context, "recover_from_stuck"):
            return self._call("recover_from_stuck", skill_args, SkillName.RECOVER_FROM_STUCK.value)
        return self._call("call_original_apexnav_policy", skill_args, SkillName.RECOVER_FROM_STUCK.value)

    def _call(
        self,
        method_name: str,
        skill_args: Dict[str, Any],
        skill_name: str,
        allow_missing: bool = False,
    ) -> SkillExecutionResult:
        if not hasattr(self.context, method_name):
            if allow_missing:
                return SkillExecutionResult(
                    skill_name=skill_name,
                    status=SkillStatus.UNAVAILABLE.value,
                    precondition_passed=False,
                    failure_type="tool_unavailable",
                    raw_metadata={"missing_tool": method_name},
                )
            return SkillExecutionResult(
                skill_name=skill_name,
                status=SkillStatus.FAILED.value,
                precondition_passed=False,
                failure_type="tool_unavailable",
                raw_metadata={"missing_tool": method_name},
            )
        raw = getattr(self.context, method_name)(skill_args)
        return normalize_skill_result(raw, skill_name)


def normalize_skill_result(raw: Any, skill_name: str) -> SkillExecutionResult:
    if isinstance(raw, SkillExecutionResult):
        if raw.skill_name != skill_name:
            raw.skill_name = skill_name
        return raw
    if isinstance(raw, dict):
        raw = dict(raw)
        raw.setdefault("skill_name", skill_name)
        return SkillExecutionResult(**raw)
    return SkillExecutionResult(
        skill_name=skill_name,
        status=SkillStatus.SUCCESS.value,
        raw_metadata={"raw_result": raw},
    )


def build_default_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()

    registry.register(
        SkillSpec(
            name=SkillName.SEMANTIC_EXPLORE.value,
            purpose="Use ApexNav semantic planning to select a high-value frontier.",
            inputs=["frontiers", "semantic_score_stats", "target_category"],
            preconditions=["reachable_frontier_exists", "no_confirmed_target"],
            forward_action="call original ApexNav semantic frontier selection",
            expected_postconditions=[
                "waypoint reached",
                "explored area increases",
                "semantic score or target confidence improves",
            ],
            failure_signals=[
                "unreachable_waypoint",
                "no_map_expansion",
                "target_confidence_not_improved",
                "repeated_bad_frontier",
            ],
            recovery_action=SkillName.GEOMETRIC_EXPLORE.value,
            memory_update_on_failure=["semantic_frontier_failure"],
            validator_constraints=["must_have_reachable_frontier"],
        ),
        _semantic_explore_handler,
    )
    registry.register(
        SkillSpec(
            name=SkillName.GEOMETRIC_EXPLORE.value,
            purpose="Use ApexNav geometric planning to select the nearest reachable frontier.",
            inputs=["frontiers"],
            preconditions=["reachable_frontier_exists"],
            forward_action="call original ApexNav nearest frontier selection",
            expected_postconditions=["waypoint reached", "explored area increases"],
            failure_signals=["unreachable_waypoint", "no_map_expansion", "stuck"],
            recovery_action=SkillName.RECOVER_FROM_STUCK.value,
            memory_update_on_failure=["blocked_frontier"],
            validator_constraints=["must_have_reachable_frontier"],
        ),
        _geometric_explore_handler,
    )
    registry.register(
        SkillSpec(
            name=SkillName.VERIFY_TARGET.value,
            purpose="Actively verify a target candidate before stop.",
            inputs=["target_candidates"],
            preconditions=["target_candidate_exists"],
            forward_action="navigate to safe observation viewpoint or rotate/observe if available",
            expected_postconditions=[
                "candidate confirmed",
                "candidate rejected",
                "candidate confidence or num_views updated",
            ],
            failure_signals=[
                "confidence_decreases",
                "candidate_disappears",
                "single_view_remains",
                "verification_viewpoint_unreachable",
            ],
            recovery_action=SkillName.SEMANTIC_EXPLORE.value,
            memory_update_on_failure=["false_positive_candidate"],
            validator_constraints=["must_have_target_candidate"],
        ),
        _verify_target_handler,
    )
    registry.register(
        SkillSpec(
            name=SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
            purpose="Navigate to a reliable target candidate and stop.",
            inputs=["target_candidates"],
            preconditions=["confirmed_reachable_target_candidate_exists"],
            forward_action="call original ApexNav target navigation path",
            expected_postconditions=["success condition satisfied", "stop accepted"],
            failure_signals=[
                "false_positive_stop",
                "premature_stop",
                "confidence_drops",
                "target_unreachable",
                "stuck",
            ],
            recovery_action=SkillName.VERIFY_TARGET.value,
            memory_update_on_failure=["premature_stop", "false_positive_stop"],
            validator_constraints=[
                "confidence_above_stop_threshold",
                "reachable_target",
                "multiview_required_if_enabled",
            ],
        ),
        _navigate_target_handler,
    )
    registry.register(
        SkillSpec(
            name=SkillName.RECOVER_FROM_STUCK.value,
            purpose="Recover from stuck, repeated collision, or bad frontier states.",
            inputs=["navigation_history", "frontiers"],
            preconditions=["stuck_or_bad_frontier_or_recovery_requested"],
            forward_action="mark current frontier blocked and call recovery/fallback planning",
            expected_postconditions=["stuck_count lowers", "new valid waypoint selected"],
            failure_signals=["repeated_stuck", "no_reachable_frontier", "map_inconsistent"],
            recovery_action=SkillName.FALLBACK_APEXNAV.value,
            memory_update_on_failure=["planner_stuck"],
        ),
        _recover_handler,
    )
    registry.register(
        SkillSpec(
            name=SkillName.FOLLOW_APEXNAV_PROPOSAL.value,
            purpose="Agent deliberately follows the current ApexNav high-level proposal.",
            inputs=["apexnav_context", "frontiers", "target_candidates"],
            preconditions=["apexnav_proposal_available"],
            forward_action="call original ApexNav policy as the selected non-error skill",
            expected_postconditions=["original ApexNav proposal advances one high-level event"],
            failure_signals=["original_policy_failed", "stuck", "timeout"],
            recovery_action=SkillName.RECOVER_FROM_STUCK.value,
            validator_constraints=["does_not_bypass_target_preemption_or_stop_gate"],
        ),
        _follow_apexnav_handler,
    )
    registry.register(
        SkillSpec(
            name=SkillName.FALLBACK_APEXNAV.value,
            purpose="Emergency fallback to original ApexNav when the agent output is invalid or unsafe.",
            inputs=["apexnav_context"],
            preconditions=[],
            forward_action="call original ApexNav policy",
            expected_postconditions=["original ApexNav decision returned"],
            failure_signals=["original_policy_failed"],
            recovery_action=None,
        ),
        _fallback_handler,
    )
    return registry


def _adapter(context: Any) -> ApexNavToolAdapter:
    return ApexNavToolAdapter(context)


def _semantic_explore_handler(skill_args: Dict[str, Any], context: Any) -> SkillExecutionResult:
    return _adapter(context).select_semantic_frontier(skill_args)


def _geometric_explore_handler(skill_args: Dict[str, Any], context: Any) -> SkillExecutionResult:
    return _adapter(context).select_nearest_reachable_frontier(skill_args)


def _verify_target_handler(skill_args: Dict[str, Any], context: Any) -> SkillExecutionResult:
    return _adapter(context).verify_target(skill_args)


def _navigate_target_handler(skill_args: Dict[str, Any], context: Any) -> SkillExecutionResult:
    return _adapter(context).navigate_to_confirmed_target(skill_args)


def _recover_handler(skill_args: Dict[str, Any], context: Any) -> SkillExecutionResult:
    return _adapter(context).recover_from_stuck(skill_args)


def _follow_apexnav_handler(skill_args: Dict[str, Any], context: Any) -> SkillExecutionResult:
    return _adapter(context).follow_apexnav_proposal(skill_args)


def _fallback_handler(skill_args: Dict[str, Any], context: Any) -> SkillExecutionResult:
    return _adapter(context).call_original_apexnav_policy(skill_args)
