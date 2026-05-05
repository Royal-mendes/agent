from __future__ import annotations

from typing import Optional

from agent.schemas import FailureClass, SkillName

NON_DEGRADING_FAILURES = {
    "low_information_gain",
    "transient_unreachable_frontier",
    "weak_semantic_peak",
    "unconfirmed_target_candidate",
    "semantic_explore_no_progress",
    "geometric_explore_no_progress",
    "inefficient_exploration",
    "missing_target",
}

DEGRADING_FAILURES = {
    "false_positive_stop",
    "map_corruption",
    "localization_drift",
    "repeated_collision",
    "planner_stuck",
    "no_frontier_deadend",
    "target_candidate_pollution",
    "repeated_bad_frontier",
    "gt_trajectory_deviation",
    "timeout_near_goal",
    "timeout",
    "false_positive_candidate",
}

DEFAULT_ESCALATION_THRESHOLD = 3


def classify_failure(
    failure_type: Optional[str],
    consecutive_count: int = 1,
    escalation_threshold: int = DEFAULT_ESCALATION_THRESHOLD,
    cfg: object = None,
) -> str:
    if getattr(cfg, "disable_failure_taxonomy", False):
        return FailureClass.UNKNOWN.value
    if not failure_type:
        return FailureClass.UNKNOWN.value
    normalized = str(failure_type)
    if normalized in DEGRADING_FAILURES:
        return FailureClass.DEGRADING.value
    if normalized in NON_DEGRADING_FAILURES:
        if consecutive_count >= escalation_threshold:
            return FailureClass.DEGRADING.value
        return FailureClass.NON_DEGRADING.value
    if consecutive_count >= escalation_threshold:
        return FailureClass.DEGRADING.value
    return FailureClass.UNKNOWN.value


def is_degrading_failure(failure_type: Optional[str], consecutive_count: int = 1) -> bool:
    return classify_failure(failure_type, consecutive_count) == FailureClass.DEGRADING.value


def suggest_recovery_skill(failure_type: Optional[str], disable_recover_from_stuck: bool = False) -> str:
    if failure_type in {"planner_stuck", "repeated_collision", "timeout", "timeout_near_goal", "repeated_bad_frontier"}:
        return SkillName.FALLBACK_APEXNAV.value if disable_recover_from_stuck else SkillName.RECOVER_FROM_STUCK.value
    if failure_type in {"false_positive_candidate", "false_positive_stop", "premature_stop", "target_candidate_pollution"}:
        return SkillName.VERIFY_TARGET.value
    if failure_type in {"semantic_explore_no_progress", "weak_semantic_peak", "low_information_gain"}:
        return SkillName.GEOMETRIC_EXPLORE.value
    if failure_type == "no_frontier_deadend":
        return SkillName.FALLBACK_APEXNAV.value
    return SkillName.FALLBACK_APEXNAV.value


def taxonomy_snapshot() -> dict:
    return {
        "non_degrading": sorted(NON_DEGRADING_FAILURES),
        "degrading": sorted(DEGRADING_FAILURES),
        "escalation_threshold": DEFAULT_ESCALATION_THRESHOLD,
    }
