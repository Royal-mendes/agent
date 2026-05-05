from __future__ import annotations

import os
import glob
from typing import Any, Dict, List, Optional

from agent.learning.baseline_teacher import BaselineTeacher
from agent.learning.gt_progress_teacher import GTProgressTeacher
from agent.learning.gt_trajectory_teacher import GTTrajectoryTeacher
from agent.learning.gt_teacher import GTTeacher
from agent.learning.hindsight_labeler import HindsightLabeler
from agent.learning.lesson_builder import LessonBuilder
from agent.learning.tool_call_dataset import ToolCallDataset
from agent.learning.trajectory_ingestor import TrajectoryIngestor
from agent.learning.trajectory_schema import ToolUseLearningSample, TrajectoryEpisode
from agent.memory.experience_memory import ExperienceMemory
from agent.reflection.policy_patch import PolicyPatchTable
from agent.schemas import AgentConfig


class TraceLearningManager:
    """Episode-end learning loop from logged decisions and optional teachers."""

    def __init__(self, cfg: Optional[AgentConfig] = None) -> None:
        self.cfg = cfg or AgentConfig()
        self.ingestor = TrajectoryIngestor()
        self.labeler = HindsightLabeler()

    def finalize_episode(self, episode_summary: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cfg.enable_learning_from_traces:
            return {"enabled": False, "reason": "learning disabled"}

        episode_log = self._select_episode_log(episode_summary)
        if not episode_log:
            return {"enabled": True, "samples": [], "reason": "episode log not found"}

        student = self.ingestor.load_episode_logger_json(episode_log, source="self_reflection")
        self._merge_episode_summary(student, episode_summary)

        samples = []
        if self.cfg.enable_self_hindsight_learning:
            samples.extend(self.labeler.label(student))

        baseline_log = self._abs_path(self.cfg.baseline_teacher_log_path)
        if self.cfg.enable_baseline_teacher_learning and baseline_log and os.path.exists(baseline_log):
            baseline = self.ingestor.load_apexnav_text_log(baseline_log, source="baseline_apexnav")
            samples.extend(BaselineTeacher(self.labeler).build_samples(student, baseline))

        gt_path = self._abs_path(self.cfg.gt_trajectory_path)
        if self.cfg.enable_gt_teacher_learning:
            if self.cfg.enable_runtime_gt_progress_learning:
                samples.extend(GTProgressTeacher(self.cfg).build_samples(student))
            if self.cfg.enable_gt_trajectory_learning:
                samples.extend(GTTrajectoryTeacher(self.cfg, self._build_reflection_vlm_provider()).build_samples(student))
            if gt_path and os.path.exists(gt_path):
                gt = self.ingestor.load_gt_trajectory(gt_path, source="gt")
                samples.extend(GTTeacher(self.labeler).build_samples(student, gt))

        dataset_path = self._abs_path(self.cfg.tool_call_dataset_path) or self.cfg.tool_call_dataset_path
        dataset_written = ToolCallDataset(dataset_path).append_many(samples)
        result: Dict[str, Any] = {
            "enabled": True,
            "episode_log": episode_log,
            "dataset_path": dataset_path,
            "dataset_written": dataset_written,
            "samples": [sample.to_dict() for sample in samples],
            "memory_written": 0,
            "policy_patches_recorded": 0,
        }

        if self._can_write_learning(student.split):
            persist_result = LessonBuilder(self.cfg).persist(
                samples,
                experience_memory=ExperienceMemory(
                    memory_path=self._abs_path(self.cfg.memory_path) or self.cfg.memory_path,
                    max_items=self.cfg.max_reflection_memory_items,
                    read_mode=self.cfg.memory_read_mode,
                    write_mode=self.cfg.learning_write_mode,
                ),
                policy_patch_table=PolicyPatchTable(
                    self._abs_path(self.cfg.policy_patch_path) or self.cfg.policy_patch_path,
                    self.cfg,
                ),
                record_policy_patches=self.cfg.learning_record_policy_patches,
            )
            result["memory_written"] = persist_result.get("memory_written", 0)
            result["policy_patches_recorded"] = (
                persist_result.get("policy_patches_recorded", 0)
                if self.cfg.learning_record_policy_patches
                else 0
            )
        else:
            result["write_skipped_reason"] = f"split {student.split} not allowed by {self.cfg.learning_write_mode}"
        return result

    def _select_episode_log(self, episode_summary: Dict[str, Any]) -> Optional[str]:
        root = self._abs_path(self.cfg.episode_log_root) or self.cfg.episode_log_root
        episode_id = str(episode_summary.get("episode_id") or "")
        candidates = []
        run_ids: List[str] = [self.cfg.run_id] if self.cfg.run_id else []
        if not run_ids:
            run_ids = ["default"]
            run_ids.extend(
                os.path.basename(os.path.dirname(os.path.dirname(path)))
                for path in glob.glob(os.path.join(root, "*", "episodes", "*.json"))
            )
        for run_id in dict.fromkeys(item for item in run_ids if item):
            episodes_dir = os.path.join(root, run_id, "episodes")
            if episode_id:
                candidates.append(os.path.join(episodes_dir, f"{self._safe_name(episode_id)}.json"))
            candidates.append(os.path.join(episodes_dir, "unknown_episode.json"))
        candidates.extend(sorted(glob.glob(os.path.join(root, "*", "episodes", "*.json")), key=os.path.getmtime, reverse=True))
        for path in candidates:
            if self._has_decisions(path):
                return path
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _merge_episode_summary(self, episode: TrajectoryEpisode, summary: Dict[str, Any]) -> None:
        episode.episode_id = str(summary.get("episode_id") or episode.episode_id or "")
        episode.scene_id = summary.get("scene_id") or episode.scene_id
        episode.split = summary.get("split") or episode.split or "unknown"
        episode.target_category = summary.get("target_category") or episode.target_category
        episode.success = bool(summary.get("success", episode.success))
        episode.result = summary.get("result") or episode.result
        episode.stop_reason = summary.get("stop_reason") or episode.stop_reason
        episode.failure_type = summary.get("failure_type") or episode.failure_type
        episode.steps = summary.get("steps") or episode.steps
        episode.spl = summary.get("spl") if summary.get("spl") is not None else episode.spl
        episode.softspl = summary.get("softspl") if summary.get("softspl") is not None else episode.softspl
        episode.final_distance_to_goal = (
            summary.get("final_distance_to_goal")
            if summary.get("final_distance_to_goal") is not None
            else episode.final_distance_to_goal
        )
        for step in episode.trajectory_steps:
            step.state_summary.setdefault("episode_id", episode.episode_id)
            step.state_summary.setdefault("scene_id", episode.scene_id)
            step.state_summary.setdefault("split", episode.split)
            if not step.state_summary.get("target_category"):
                step.state_summary["target_category"] = episode.target_category

    def _can_write_learning(self, split: str) -> bool:
        mode = self.cfg.learning_write_mode
        if mode == "disabled":
            return False
        if mode == "all":
            return True
        if mode == "train_only":
            return split == "train"
        if mode == "val_only":
            return split == "val"
        if mode == "test_only":
            return split == "test"
        if mode == "eval_only":
            return split in {"val", "test"}
        return False

    def _abs_path(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if os.path.isabs(path):
            return path
        return os.path.join(self.cfg.project_root or os.getcwd(), path)

    def _build_reflection_vlm_provider(self) -> Optional[Any]:
        if not self.cfg.enable_vlm_gt_trajectory_reflection or (self.cfg.vlm_provider or "mock") == "mock":
            return None
        try:
            from agent.vlm_provider import build_vlm_provider

            return build_vlm_provider(self.cfg)
        except Exception:
            return None

    @staticmethod
    def _has_decisions(path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            import json

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return bool(data.get("decisions"))
        except Exception:
            return False

    @staticmethod
    def _safe_name(value: str) -> str:
        import re

        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:160]
