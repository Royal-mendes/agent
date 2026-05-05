from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorkingMemory:
    current_skill: Optional[str] = None
    current_waypoint: Optional[List[float]] = None
    recent_skill_trace: List[str] = field(default_factory=list)
    recent_waypoint_trace: List[Any] = field(default_factory=list)
    recent_observation_summaries: List[Any] = field(default_factory=list)
    recent_failures: List[str] = field(default_factory=list)
    validator_rejections: List[Dict[str, Any]] = field(default_factory=list)
    tool_or_planner_invocations: List[Dict[str, Any]] = field(default_factory=list)
    max_trace: int = 20

    def update_before_decision(self, state_summary: Dict[str, Any]) -> None:
        obs = state_summary.get("current_observation_summary")
        if obs:
            self.recent_observation_summaries.append(obs)
            self.recent_observation_summaries = self.recent_observation_summaries[-self.max_trace :]

    def update_after_decision(self, decision: Any) -> None:
        decision_dict = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
        self.current_skill = decision_dict.get("selected_skill")
        if self.current_skill:
            self.recent_skill_trace.append(self.current_skill)
            self.recent_skill_trace = self.recent_skill_trace[-self.max_trace :]

    def update_after_validation(self, result: Any) -> None:
        result_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        if not result_dict.get("accepted", True):
            self.validator_rejections.append(result_dict)
            self.validator_rejections = self.validator_rejections[-self.max_trace :]

    def update_after_skill(self, result: Any) -> None:
        result_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        self.current_skill = result_dict.get("skill_name", self.current_skill)
        waypoint = result_dict.get("selected_waypoint")
        if waypoint is not None:
            self.current_waypoint = list(waypoint)
            self.recent_waypoint_trace.append(waypoint)
            self.recent_waypoint_trace = self.recent_waypoint_trace[-self.max_trace :]
        failure = result_dict.get("failure_type")
        if failure:
            self.recent_failures.append(failure)
            self.recent_failures = self.recent_failures[-self.max_trace :]

    def record_tool_invocation(self, tool_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.tool_or_planner_invocations.append(
            {"tool_name": tool_name, "metadata": metadata or {}}
        )
        self.tool_or_planner_invocations = self.tool_or_planner_invocations[-self.max_trace :]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("max_trace", None)
        return data
