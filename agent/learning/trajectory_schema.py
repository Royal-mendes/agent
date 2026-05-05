from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.schemas import SkillName


def _known_kwargs(cls: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    known = {field_name for field_name in cls.__dataclass_fields__}
    return {key: value for key, value in data.items() if key in known}


@dataclass
class TeacherAction:
    """A high-level tool/skill action proposed by a teacher trace."""

    skill_name: str = SkillName.FALLBACK_APEXNAV.value
    tool_name: str = "call_original_apexnav_policy"
    arguments: Dict[str, Any] = field(default_factory=dict)
    waypoint: Optional[Tuple[float, float]] = None
    reason: str = ""
    confidence: float = 0.5
    source: str = "unknown"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherAction":
        return cls(**_known_kwargs(cls, data or {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryStep:
    """A compact high-level decision point from a navigation episode."""

    timestep: Optional[int] = None
    state_summary: Dict[str, Any] = field(default_factory=dict)
    selected_skill: Optional[str] = None
    skill_args: Dict[str, Any] = field(default_factory=dict)
    tool_name: Optional[str] = None
    action: Optional[Any] = None
    waypoint: Optional[Tuple[float, float]] = None
    outcome: Optional[str] = None
    validator_result: Dict[str, Any] = field(default_factory=dict)
    agent_decision: Dict[str, Any] = field(default_factory=dict)
    failure_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        return cls(**_known_kwargs(cls, data or {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryEpisode:
    """Unified trajectory representation for baseline, GT, and agent traces."""

    source: str = "unknown"
    episode_id: Optional[str] = None
    scene_id: Optional[str] = None
    split: str = "unknown"
    target_category: Optional[str] = None
    success: bool = False
    result: Optional[str] = None
    stop_reason: Optional[str] = None
    failure_type: Optional[str] = None
    steps: Optional[int] = None
    spl: Optional[float] = None
    softspl: Optional[float] = None
    final_distance_to_goal: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    trajectory_steps: List[TrajectoryStep] = field(default_factory=list)
    raw_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryEpisode":
        kwargs = _known_kwargs(cls, data or {})
        kwargs["trajectory_steps"] = [
            step if isinstance(step, TrajectoryStep) else TrajectoryStep.from_dict(step)
            for step in kwargs.get("trajectory_steps", [])
        ]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_reflection_episode(self) -> Dict[str, Any]:
        skill_trace = []
        validator_rejections = []
        failure_signals = []
        for step in self.trajectory_steps:
            if step.selected_skill:
                skill_trace.append({"selected_skill": step.selected_skill, "timestep": step.timestep})
            if step.validator_result and not step.validator_result.get("accepted", True):
                validator_rejections.append(step.validator_result)
            if step.failure_type:
                failure_signals.append(step.failure_type)
        episode = {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "split": self.split,
            "target_category": self.target_category,
            "success": self.success,
            "spl": self.spl,
            "softspl": self.softspl,
            "steps": self.steps if self.steps is not None else len(self.trajectory_steps),
            "final_distance_to_goal": self.final_distance_to_goal,
            "stop_reason": self.stop_reason or self.result,
            "failure_type": self.failure_type,
            "skill_trace": skill_trace,
            "validator_rejections": validator_rejections,
            "failure_signals": failure_signals,
        }
        episode.update({key: value for key, value in self.metrics.items() if key not in episode})
        return episode


@dataclass
class ToolUseLearningSample:
    """One state/action lesson distilled from teacher, GT, or self traces."""

    source: str = "unknown"
    episode_id: Optional[str] = None
    scene_id: Optional[str] = None
    split: str = "unknown"
    target_category: Optional[str] = None
    timestep: Optional[int] = None
    state_condition: Dict[str, Any] = field(default_factory=dict)
    student_action: Dict[str, Any] = field(default_factory=dict)
    teacher_action: Dict[str, Any] = field(default_factory=dict)
    outcome: str = ""
    failure_type: Optional[str] = None
    failure_class: Optional[str] = None
    lesson: str = ""
    policy_patch_proposal: Optional[Dict[str, Any]] = None
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolUseLearningSample":
        return cls(**_known_kwargs(cls, data or {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
