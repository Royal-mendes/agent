from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from agent.schemas import AgentConfig, SkillName


class PostconditionChecker:
    def __init__(self, cfg: Optional[AgentConfig] = None) -> None:
        self.cfg = cfg or AgentConfig()

    def check(
        self,
        skill_name: str,
        result: Any,
        before: Optional[Dict[str, Any]] = None,
        after: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[bool], str]:
        result_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        before = before or {}
        after = after or {}

        if skill_name == SkillName.SEMANTIC_EXPLORE.value:
            return self._semantic_explore(result_dict, before, after)
        if skill_name == SkillName.GEOMETRIC_EXPLORE.value:
            return self._geometric_explore(result_dict, before, after)
        if skill_name == SkillName.VERIFY_TARGET.value:
            return self._verify_target(result_dict, before, after)
        if skill_name == SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value:
            return self._navigate_target(result_dict, before, after)
        if skill_name == SkillName.RECOVER_FROM_STUCK.value:
            return self._recover(result_dict, before, after)
        if skill_name in {
            SkillName.FOLLOW_APEXNAV_PROPOSAL.value,
            SkillName.FALLBACK_APEXNAV.value,
        }:
            return True, "fallback delegated to original ApexNav policy"
        return None, "postcondition unavailable for unknown skill"

    def _semantic_explore(
        self, result: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any]
    ) -> Tuple[Optional[bool], str]:
        if result.get("waypoint_reached") or self._meta(result, "waypoint_reached"):
            return True, "waypoint reached"
        if self._delta(after, before, "explored_area") > self.cfg.low_information_gain_threshold:
            return True, "explored area increased"
        if self._delta(after, before, "semantic_score") > 0:
            return True, "semantic score improved"
        if self._delta(after, before, "target_confidence") > 0:
            return True, "target confidence improved"
        if self._has_any(before, after, ["explored_area", "semantic_score", "target_confidence"]):
            return False, "semantic explore produced no useful progress"
        return None, "semantic explore postcondition unavailable"

    def _geometric_explore(
        self, result: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any]
    ) -> Tuple[Optional[bool], str]:
        waypoint_reached = bool(result.get("waypoint_reached") or self._meta(result, "waypoint_reached"))
        info_gain = self._delta(after, before, "explored_area")
        new_frontiers = self._delta(after, before, "frontier_count")
        if waypoint_reached and (
            info_gain > self.cfg.low_information_gain_threshold or new_frontiers != 0
        ):
            return True, "waypoint reached with new information"
        if waypoint_reached and self._has_any(before, after, ["explored_area", "frontier_count"]):
            return False, "geometric explore reached waypoint without information gain"
        if self._has_any(before, after, ["explored_area", "frontier_count"]):
            return False, "geometric explore waypoint not reached"
        return None, "geometric explore postcondition unavailable"

    def _verify_target(
        self, result: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any]
    ) -> Tuple[Optional[bool], str]:
        if self._meta(result, "candidate_confirmed") or self._meta(result, "candidate_rejected"):
            return True, "candidate confirmation state changed"
        if self._delta(after, before, "target_confidence") != 0:
            return True, "target confidence updated"
        if self._delta(after, before, "target_num_views") > 0:
            return True, "target views updated"
        if self._has_any(before, after, ["target_confidence", "target_num_views"]):
            return False, "target verification produced no evidence update"
        return None, "verify target postcondition unavailable"

    def _navigate_target(
        self, result: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any]
    ) -> Tuple[Optional[bool], str]:
        if self._meta(result, "evaluator_success") or self._meta(result, "success"):
            return True, "evaluator accepted stop"
        if result.get("status") == "success" and self._meta(result, "stop_decision"):
            return True, "stop decision issued"
        if self._has_any(before, after, ["evaluator_success", "distance_to_goal"]):
            return False, "target navigation did not satisfy success"
        return None, "target navigation postcondition unavailable"

    def _recover(
        self, result: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any]
    ) -> Tuple[Optional[bool], str]:
        if self._delta(after, before, "stuck_count") < 0:
            return True, "stuck count decreased"
        if result.get("selected_waypoint") is not None or self._meta(result, "new_valid_waypoint"):
            return True, "new valid waypoint selected"
        if self._delta(after, before, "blocked_frontier_count") > 0:
            return True, "blocked frontier updated"
        if self._has_any(before, after, ["stuck_count", "blocked_frontier_count"]):
            return False, "recovery did not improve navigation state"
        return None, "recovery postcondition unavailable"

    @staticmethod
    def _meta(result: Dict[str, Any], name: str) -> Any:
        return (result.get("raw_metadata") or {}).get(name)

    @staticmethod
    def _delta(after: Dict[str, Any], before: Dict[str, Any], key: str) -> float:
        a = after.get(key)
        b = before.get(key)
        if a is None or b is None:
            return 0.0
        try:
            return float(a) - float(b)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _has_any(before: Dict[str, Any], after: Dict[str, Any], keys: list) -> bool:
        return any(key in before or key in after for key in keys)
