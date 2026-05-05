from __future__ import annotations

from typing import List, Optional

from agent.learning.hindsight_labeler import HindsightLabeler
from agent.learning.trajectory_schema import ToolUseLearningSample, TrajectoryEpisode


class BaselineTeacher:
    """Distill ApexNav baseline behavior into tool-use feedback."""

    def __init__(self, labeler: Optional[HindsightLabeler] = None) -> None:
        self.labeler = labeler or HindsightLabeler()

    def build_samples(
        self,
        student_episode: TrajectoryEpisode,
        baseline_episode: TrajectoryEpisode,
    ) -> List[ToolUseLearningSample]:
        return self.labeler.label(student_episode, teacher_episode=baseline_episode)

    @staticmethod
    def is_baseline_better(student_episode: TrajectoryEpisode, baseline_episode: TrajectoryEpisode) -> bool:
        better, _ = HindsightLabeler._baseline_is_better(student_episode, baseline_episode)
        return better
