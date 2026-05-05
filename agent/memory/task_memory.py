from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskMemory:
    episode_id: Optional[str] = None
    scene_id: Optional[str] = None
    target_category: Optional[str] = None
    task_status: str = "running"
    frontier_status: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "visited": [],
            "blocked": [],
            "low_value": [],
            "high_value": [],
        }
    )
    target_candidate_status: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "confirmed": [],
            "unverified": [],
            "rejected_false_positive": [],
        }
    )
    subgoals: List[Dict[str, Any]] = field(default_factory=list)

    def update_from_state(self, state_summary: Dict[str, Any]) -> None:
        self.episode_id = state_summary.get("episode_id", self.episode_id)
        self.scene_id = state_summary.get("scene_id", self.scene_id)
        self.target_category = state_summary.get("target_category", self.target_category)

        for frontier in state_summary.get("frontiers") or []:
            frontier_id = frontier.get("id")
            if frontier_id is None:
                continue
            if frontier.get("visited"):
                self._append_unique("frontier_status", "visited", frontier_id)
            if frontier.get("blocked"):
                self._append_unique("frontier_status", "blocked", frontier_id)
            if frontier.get("low_value"):
                self._append_unique("frontier_status", "low_value", frontier_id)
            if frontier.get("semantic_score") is not None:
                self._append_unique("frontier_status", "high_value", frontier_id)

        for candidate in state_summary.get("target_candidates") or []:
            candidate_id = candidate.get("id")
            if candidate_id is None:
                continue
            if candidate.get("rejected_false_positive"):
                self._append_unique("target_candidate_status", "rejected_false_positive", candidate_id)
            elif candidate.get("multi_view_confirmed"):
                self._append_unique("target_candidate_status", "confirmed", candidate_id)
            else:
                self._append_unique("target_candidate_status", "unverified", candidate_id)

    def update_after_skill(self, result: Any) -> None:
        result_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        frontier_id = result_dict.get("selected_frontier_id")
        if frontier_id is not None:
            if result_dict.get("status") == "success":
                self._append_unique("frontier_status", "visited", frontier_id)
            if result_dict.get("failure_type") in {"repeated_bad_frontier", "low_information_gain"}:
                self._append_unique("frontier_status", "low_value", frontier_id)
            if result_dict.get("failure_type") in {"unreachable_waypoint", "planner_stuck", "timeout"}:
                self._append_unique("frontier_status", "blocked", frontier_id)

        target_id = result_dict.get("target_candidate_id")
        if target_id is not None:
            if result_dict.get("failure_type") in {"false_positive_candidate", "false_positive_stop"}:
                self._append_unique("target_candidate_status", "rejected_false_positive", target_id)
            elif result_dict.get("status") == "success":
                self._append_unique("target_candidate_status", "confirmed", target_id)

    def _append_unique(self, group: str, key: str, value: Any) -> None:
        bucket = getattr(self, group)[key]
        if value not in bucket:
            bucket.append(value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
