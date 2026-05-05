from __future__ import annotations

from typing import List, Optional

from agent.learning.hindsight_labeler import HindsightLabeler
from agent.learning.trajectory_schema import TeacherAction, ToolUseLearningSample, TrajectoryEpisode


class GTTeacher:
    """Use oracle trajectory waypoints as hindsight progress supervision."""

    def __init__(self, labeler: Optional[HindsightLabeler] = None) -> None:
        self.labeler = labeler or HindsightLabeler()

    def build_samples(
        self,
        student_episode: TrajectoryEpisode,
        gt_episode: TrajectoryEpisode,
    ) -> List[ToolUseLearningSample]:
        return self.labeler.label(student_episode, gt_episode=gt_episode)

    @staticmethod
    def action_at(gt_episode: TrajectoryEpisode, timestep: int = 0) -> Optional[TeacherAction]:
        if not gt_episode.trajectory_steps:
            return None
        step = gt_episode.trajectory_steps[min(len(gt_episode.trajectory_steps) - 1, max(0, timestep))]
        return TeacherAction(
            skill_name=step.selected_skill,
            tool_name=step.tool_name or "select_oracle_progress_waypoint",
            arguments=dict(step.skill_args),
            waypoint=step.waypoint,
            reason="GT trajectory progress waypoint.",
            confidence=0.9,
            source=gt_episode.source,
        )
