import json
import os
import tempfile
import unittest

from agent.learning.baseline_teacher import BaselineTeacher
from agent.learning.gt_progress_teacher import GTProgressTeacher
from agent.learning.gt_trajectory_teacher import GTTrajectoryTeacher
from agent.learning.gt_teacher import GTTeacher
from agent.learning.hindsight_labeler import HindsightLabeler
from agent.learning.lesson_builder import LessonBuilder
from agent.learning.tool_call_dataset import ToolCallDataset
from agent.learning.trace_learning_manager import TraceLearningManager
from agent.learning.trajectory_ingestor import TrajectoryIngestor
from agent.learning.trajectory_schema import TrajectoryEpisode, TrajectoryStep
from agent.memory.experience_memory import ExperienceMemory
from agent.schemas import AgentConfig, SkillName


def student_episode():
    state = {
        "target_category": "toilet",
        "semantic_score_stats": {"has_clear_peak": True},
        "target_candidates": [],
        "frontiers": [{"id": 3, "reachable": True, "semantic_score": 0.91, "distance": 4.0}],
    }
    return TrajectoryEpisode(
        source="self_reflection",
        episode_id="e1",
        split="train",
        target_category="toilet",
        success=False,
        steps=20,
        trajectory_steps=[
            TrajectoryStep(
                timestep=4,
                state_summary=state,
                selected_skill=SkillName.VERIFY_TARGET.value,
                skill_args={},
                tool_name="verify_target_candidate",
                validator_result={
                    "accepted": False,
                    "final_skill": SkillName.SEMANTIC_EXPLORE.value,
                    "final_arguments": {"frontier_id": 3},
                    "rejection_reason": "no target candidate to verify",
                },
                failure_type="unconfirmed_target_candidate",
            )
        ],
    )


class LearningLayerTests(unittest.TestCase):
    def test_ingests_episode_logger_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "episode.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "episode_id": "e1",
                        "decisions": [
                            {
                                "timestep": 1,
                                "state_summary": {"target_category": "toilet"},
                                "agent_decision": {"selected_skill": "VERIFY_TARGET"},
                                "validator_result": {
                                    "accepted": False,
                                    "final_skill": "SEMANTIC_EXPLORE",
                                    "final_arguments": {"frontier_id": 1},
                                    "rejection_reason": "no target candidate to verify",
                                },
                            }
                        ],
                        "episode_end": {
                            "episode_id": "e1",
                            "split": "train",
                            "target_category": "toilet",
                            "success": False,
                            "steps": 10,
                        },
                    },
                    f,
                )
            episode = TrajectoryIngestor().load(path)
            self.assertEqual(episode.trajectory_steps[0].selected_skill, SkillName.VERIFY_TARGET.value)
            self.assertEqual(episode.trajectory_steps[0].metadata["final_skill"], SkillName.SEMANTIC_EXPLORE.value)
            self.assertEqual(episode.trajectory_steps[0].failure_type, "unconfirmed_target_candidate")

    def test_ingests_apexnav_text_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "baseline.log")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "--------------Step: 1--------------\n"
                    "Finding [toilet]; Action: 2;\n"
                    "Result: success\n"
                    "Average Success | 100.00%\n"
                    "Average SPL | 81.05%\n"
                )
            episode = TrajectoryIngestor().load(path)
            self.assertTrue(episode.success)
            self.assertEqual(episode.target_category, "toilet")
            self.assertEqual(episode.trajectory_steps[0].tool_name, "call_original_apexnav_policy")

    def test_hindsight_labels_verify_without_candidate(self):
        samples = HindsightLabeler().label(student_episode())
        self.assertGreaterEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.teacher_action["skill_name"], SkillName.SEMANTIC_EXPLORE.value)
        self.assertIn("no target candidate", sample.lesson)

    def test_hindsight_treats_raw_stuck_as_advisory(self):
        state = {
            "target_category": "chair",
            "semantic_score_stats": {"has_clear_peak": False},
            "target_candidates": [],
            "frontiers": [{"id": 7, "reachable": True, "distance": 1.2, "last_selected": True}],
            "navigation_history": {"stuck_count": 3, "collision_count": 0, "recent_failures": []},
        }
        episode = TrajectoryEpisode(
            source="self_reflection",
            episode_id="e_stuck",
            split="train",
            target_category="chair",
            success=True,
            steps=160,
            trajectory_steps=[
                TrajectoryStep(
                    timestep=42,
                    state_summary=state,
                    selected_skill=SkillName.GEOMETRIC_EXPLORE.value,
                    skill_args={"frontier_id": 7},
                    tool_name="select_nearest_reachable_frontier",
                )
            ],
        )
        samples = HindsightLabeler().label(episode)
        self.assertFalse(any(sample.teacher_action["skill_name"] == SkillName.RECOVER_FROM_STUCK.value for sample in samples))

    def test_hindsight_labels_explicit_planner_stuck_as_recovery(self):
        state = {
            "target_category": "chair",
            "semantic_score_stats": {"has_clear_peak": False},
            "target_candidates": [],
            "frontiers": [{"id": 7, "reachable": True, "distance": 1.2, "last_selected": True}],
            "navigation_history": {"stuck_count": 3, "collision_count": 0, "recent_failures": ["planner_stuck"]},
        }
        episode = TrajectoryEpisode(
            source="self_reflection",
            episode_id="e_stuck",
            split="train",
            target_category="chair",
            success=False,
            steps=160,
            trajectory_steps=[
                TrajectoryStep(
                    timestep=42,
                    state_summary=state,
                    selected_skill=SkillName.GEOMETRIC_EXPLORE.value,
                    skill_args={"frontier_id": 7},
                    tool_name="select_nearest_reachable_frontier",
                    failure_type="planner_stuck",
                )
            ],
        )
        samples = HindsightLabeler().label(episode)
        recovery_samples = [
            sample for sample in samples if sample.teacher_action["skill_name"] == SkillName.RECOVER_FROM_STUCK.value
        ]
        self.assertEqual(len(recovery_samples), 1)
        self.assertEqual(recovery_samples[0].failure_type, "planner_stuck")
        self.assertIn("stuck_count", recovery_samples[0].lesson)

    def test_baseline_teacher_creates_sample_when_baseline_better(self):
        baseline = TrajectoryEpisode(
            source="baseline_apexnav",
            episode_id="e1_baseline",
            split="train",
            target_category="toilet",
            success=True,
            steps=8,
        )
        samples = BaselineTeacher().build_samples(student_episode(), baseline)
        self.assertTrue(any(sample.outcome == "baseline_teacher_better" for sample in samples))

    def test_baseline_teacher_uses_spl_gap_as_efficiency_signal(self):
        student = student_episode()
        student.success = True
        student.steps = 100
        student.spl = 0.47
        baseline = TrajectoryEpisode(
            source="baseline_apexnav",
            episode_id="e1_baseline",
            split="train",
            target_category="toilet",
            success=True,
            steps=100,
            spl=0.62,
        )
        samples = BaselineTeacher().build_samples(student, baseline)
        baseline_samples = [sample for sample in samples if sample.outcome == "baseline_teacher_better"]
        self.assertTrue(any(sample.failure_type == "inefficient_exploration" for sample in baseline_samples))
        self.assertIn("baseline_spl", baseline_samples[0].metadata)

    def test_baseline_teacher_does_not_override_better_reflective_spl_for_tiny_step_gap(self):
        student = student_episode()
        student.success = True
        student.steps = 163
        student.spl = 0.519
        baseline = TrajectoryEpisode(
            source="baseline_apexnav",
            episode_id="e1_baseline",
            split="train",
            target_category="toilet",
            success=True,
            steps=162,
            spl=0.511,
        )
        samples = BaselineTeacher().build_samples(student, baseline)
        self.assertFalse(any(sample.outcome == "baseline_teacher_better" for sample in samples))

    def test_gt_teacher_uses_oracle_waypoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gt.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"episode_id": "e1", "target_category": "toilet", "waypoints": [[0, 0], [1, 2]]}, f)
            gt = TrajectoryIngestor().load(path)
            samples = GTTeacher().build_samples(student_episode(), gt)
            self.assertTrue(samples)
            self.assertTrue(any("oracle" in sample.teacher_action["tool_name"] for sample in samples))

    def test_gt_progress_teacher_keeps_skill_when_distance_decreases(self):
        episode = TrajectoryEpisode(
            source="self_reflection",
            episode_id="e_gt_progress",
            split="val",
            target_category="chair",
            final_distance_to_goal=3.2,
            trajectory_steps=[
                TrajectoryStep(
                    timestep=1,
                    state_summary={
                        "target_category": "chair",
                        "gt_feedback": {"available": True, "distance_to_goal": 5.0},
                        "semantic_score_stats": {"has_clear_peak": False},
                        "frontiers": [{"id": 1, "reachable": True, "distance": 2.0}],
                    },
                    selected_skill=SkillName.GEOMETRIC_EXPLORE.value,
                    skill_args={"frontier_id": 1},
                    tool_name="select_nearest_reachable_frontier",
                ),
                TrajectoryStep(
                    timestep=20,
                    state_summary={
                        "target_category": "chair",
                        "gt_feedback": {"available": True, "distance_to_goal": 4.2},
                        "semantic_score_stats": {"has_clear_peak": False},
                        "frontiers": [{"id": 2, "reachable": True, "distance": 1.0}],
                    },
                    selected_skill=SkillName.GEOMETRIC_EXPLORE.value,
                    skill_args={"frontier_id": 2},
                    tool_name="select_nearest_reachable_frontier",
                ),
            ],
        )
        samples = GTProgressTeacher(AgentConfig(gt_progress_min_delta=0.25)).build_samples(episode)
        self.assertTrue(samples)
        helpful = [sample for sample in samples if sample.outcome == "gt_progress_skill_helped"]
        self.assertTrue(helpful)
        self.assertEqual(helpful[0].teacher_action["skill_name"], SkillName.GEOMETRIC_EXPLORE.value)
        self.assertGreater(helpful[0].state_condition["gt_distance_delta"], 0)

    def test_gt_progress_teacher_switches_after_no_progress(self):
        episode = TrajectoryEpisode(
            source="self_reflection",
            episode_id="e_gt_no_progress",
            split="val",
            target_category="toilet",
            final_distance_to_goal=6.1,
            trajectory_steps=[
                TrajectoryStep(
                    timestep=1,
                    state_summary={
                        "target_category": "toilet",
                        "gt_feedback": {"available": True, "distance_to_goal": 6.0},
                        "semantic_score_stats": {"has_clear_peak": True},
                        "target_candidates": [],
                        "frontiers": [
                            {"id": 3, "reachable": True, "semantic_score": 0.9, "distance": 4.0}
                        ],
                    },
                    selected_skill=SkillName.GEOMETRIC_EXPLORE.value,
                    skill_args={"frontier_id": 3},
                    tool_name="select_nearest_reachable_frontier",
                )
            ],
        )
        samples = GTProgressTeacher(AgentConfig(gt_progress_min_delta=0.25)).build_samples(episode)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].outcome, "gt_progress_no_progress")
        self.assertEqual(samples[0].teacher_action["skill_name"], SkillName.SEMANTIC_EXPLORE.value)

    def test_gt_trajectory_teacher_reflects_when_path_deviation_grows(self):
        episode = TrajectoryEpisode(
            source="self_reflection",
            episode_id="e_gt_path",
            split="val",
            target_category="chair",
            trajectory_steps=[
                TrajectoryStep(
                    timestep=1,
                    state_summary={
                        "target_category": "chair",
                        "gt_feedback": {
                            "gt_trajectory": {
                                "gt_path_available": True,
                                "distance_to_gt_path": 0.4,
                                "nearest_gt_path_index": 2,
                                "gt_path_progress_ratio": 0.2,
                                "gt_next_waypoint": [1.0, 0.0, 1.0],
                                "agent_position": [0.0, 0.0, 0.0],
                            }
                        },
                        "semantic_score_stats": {"has_clear_peak": True},
                        "target_candidates": [],
                        "frontiers": [{"id": 3, "reachable": True, "semantic_score": 0.9, "distance": 4.0}],
                    },
                    selected_skill=SkillName.GEOMETRIC_EXPLORE.value,
                    skill_args={"frontier_id": 4},
                    tool_name="select_nearest_reachable_frontier",
                ),
                TrajectoryStep(
                    timestep=30,
                    state_summary={
                        "target_category": "chair",
                        "gt_feedback": {
                            "gt_trajectory": {
                                "gt_path_available": True,
                                "distance_to_gt_path": 2.2,
                                "nearest_gt_path_index": 2,
                                "gt_path_progress_ratio": 0.2,
                                "gt_next_waypoint": [1.0, 0.0, 1.0],
                                "agent_position": [3.0, 0.0, 3.0],
                            }
                        },
                        "semantic_score_stats": {"has_clear_peak": True},
                        "target_candidates": [],
                        "frontiers": [{"id": 3, "reachable": True, "semantic_score": 0.9, "distance": 4.0}],
                    },
                    selected_skill=SkillName.SEMANTIC_EXPLORE.value,
                    skill_args={"frontier_id": 3},
                    tool_name="select_semantic_frontier",
                ),
            ],
        )
        samples = GTTrajectoryTeacher(
            AgentConfig(gt_path_deviation_threshold=1.5, gt_path_deviation_growth_threshold=0.5)
        ).build_samples(episode)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].outcome, "gt_trajectory_deviation_reflection")
        self.assertEqual(samples[0].failure_type, "gt_trajectory_deviation")
        self.assertEqual(samples[0].teacher_action["skill_name"], SkillName.SEMANTIC_EXPLORE.value)
        self.assertGreater(samples[0].state_condition["distance_to_gt_path_after"], 1.5)

    def test_gt_trajectory_teacher_uses_vlm_reflection_when_available(self):
        class DummyProvider:
            def generate(self, system_prompt, user_prompt, image_data_url=None, image_data_urls=None):
                return json.dumps(
                    {
                        "failure_analysis": "The chosen frontier pulled away from the GT path.",
                        "gt_rationale": "The GT path follows the shortest corridor toward the goal viewpoint.",
                        "better_skill": SkillName.SEMANTIC_EXPLORE.value,
                        "lesson": "When geometric exploration increases GT-path deviation, switch to semantic exploration.",
                        "confidence": 0.81,
                    }
                )

        episode = TrajectoryEpisode(
            source="self_reflection",
            episode_id="e_gt_path_vlm",
            split="val",
            target_category="chair",
            trajectory_steps=[
                TrajectoryStep(
                    timestep=1,
                    state_summary={
                        "gt_feedback": {"gt_trajectory": {"gt_path_available": True, "distance_to_gt_path": 0.1}},
                        "semantic_score_stats": {"has_clear_peak": True},
                        "frontiers": [{"id": 3, "reachable": True, "semantic_score": 0.9, "distance": 4.0}],
                    },
                    selected_skill=SkillName.GEOMETRIC_EXPLORE.value,
                ),
                TrajectoryStep(
                    timestep=20,
                    state_summary={
                        "gt_feedback": {"gt_trajectory": {"gt_path_available": True, "distance_to_gt_path": 2.0}},
                        "semantic_score_stats": {"has_clear_peak": True},
                        "frontiers": [{"id": 3, "reachable": True, "semantic_score": 0.9, "distance": 4.0}],
                    },
                    selected_skill=SkillName.SEMANTIC_EXPLORE.value,
                ),
            ],
        )
        samples = GTTrajectoryTeacher(
            AgentConfig(
                gt_path_deviation_threshold=1.5,
                gt_path_deviation_growth_threshold=0.5,
                enable_vlm_gt_trajectory_reflection=True,
            ),
            DummyProvider(),
        ).build_samples(episode)
        self.assertEqual(samples[0].source, "gt_trajectory_vlm")
        self.assertIn("geometric exploration increases", samples[0].lesson)

    def test_lesson_builder_writes_memory_and_dataset(self):
        sample = HindsightLabeler().label(student_episode())[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "samples.jsonl")
            memory_path = os.path.join(tmpdir, "memory.jsonl")
            dataset = ToolCallDataset(dataset_path)
            dataset.append(sample)
            self.assertEqual(len(dataset.load()), 1)
            memory = ExperienceMemory(memory_path, write_mode="train_only")
            result = LessonBuilder(AgentConfig(memory_path=memory_path)).persist(
                [sample],
                experience_memory=memory,
            )
            self.assertEqual(result["memory_written"], 1)
            self.assertTrue(os.path.exists(memory_path))

    def test_trace_learning_manager_uses_unknown_episode_log_and_skips_val_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "logs")
            episodes_dir = os.path.join(root, "run1", "episodes")
            os.makedirs(episodes_dir)
            with open(os.path.join(episodes_dir, "unknown_episode.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "episode_id": "",
                        "decisions": [
                            {
                                "timestep": 1,
                                "state_summary": {
                                    "target_category": "",
                                    "semantic_score_stats": {"has_clear_peak": False},
                                    "target_candidates": [],
                                    "frontiers": [{"id": 2, "reachable": True, "distance": 1.0}],
                                },
                                "agent_decision": {
                                    "selected_skill": SkillName.VERIFY_TARGET.value,
                                    "skill_args": {"target_candidate_id": None},
                                },
                                "validator_result": {
                                    "accepted": False,
                                    "final_skill": SkillName.GEOMETRIC_EXPLORE.value,
                                    "final_arguments": {"frontier_id": 2},
                                    "rejection_reason": "no target candidate to verify",
                                },
                            }
                        ],
                    },
                    f,
                )
            dataset_path = os.path.join(tmpdir, "samples.jsonl")
            memory_path = os.path.join(tmpdir, "memory.jsonl")
            cfg = AgentConfig(
                enable_learning_from_traces=True,
                enable_self_hindsight_learning=True,
                enable_gt_teacher_learning=False,
                episode_log_root=root,
                run_id="run1",
                project_root=tmpdir,
                tool_call_dataset_path=dataset_path,
                memory_path=memory_path,
                learning_write_mode="train_only",
            )
            result = TraceLearningManager(cfg).finalize_episode(
                {
                    "episode_id": "1",
                    "split": "val",
                    "target_category": "toilet",
                    "success": True,
                    "steps": 5,
                }
            )
            self.assertEqual(result["dataset_written"], 1)
            self.assertIn("val", result["write_skipped_reason"])
            self.assertFalse(os.path.exists(memory_path))
            sample = ToolCallDataset(dataset_path).load()[0]
            self.assertEqual(sample.target_category, "toilet")
            self.assertEqual(sample.student_action["skill"], SkillName.VERIFY_TARGET.value)

    def test_trace_learning_manager_writes_val_memory_from_gt_progress_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "logs")
            episodes_dir = os.path.join(root, "run_gt", "episodes")
            os.makedirs(episodes_dir)
            with open(os.path.join(episodes_dir, "e_gt.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "episode_id": "e_gt",
                        "decisions": [
                            {
                                "timestep": 1,
                                "state_summary": {
                                    "target_category": "chair",
                                    "gt_feedback": {"available": True, "distance_to_goal": 5.0},
                                    "semantic_score_stats": {"has_clear_peak": False},
                                    "target_candidates": [],
                                    "frontiers": [{"id": 2, "reachable": True, "distance": 1.0}],
                                },
                                "agent_decision": {
                                    "selected_skill": SkillName.GEOMETRIC_EXPLORE.value,
                                    "skill_args": {"frontier_id": 2},
                                },
                                "validator_result": {"accepted": True, "final_skill": SkillName.GEOMETRIC_EXPLORE.value},
                            }
                        ],
                    },
                    f,
                )
            dataset_path = os.path.join(tmpdir, "samples.jsonl")
            memory_path = os.path.join(tmpdir, "memory.jsonl")
            cfg = AgentConfig(
                enable_learning_from_traces=True,
                enable_self_hindsight_learning=False,
                enable_baseline_teacher_learning=False,
                enable_gt_teacher_learning=True,
                enable_runtime_gt_progress_learning=True,
                episode_log_root=root,
                run_id="run_gt",
                project_root=tmpdir,
                tool_call_dataset_path=dataset_path,
                memory_path=memory_path,
                learning_write_mode="all",
            )
            result = TraceLearningManager(cfg).finalize_episode(
                {
                    "episode_id": "e_gt",
                    "split": "val",
                    "target_category": "chair",
                    "success": False,
                    "steps": 50,
                    "final_distance_to_goal": 4.2,
                }
            )
            self.assertEqual(result["dataset_written"], 1)
            self.assertEqual(result["memory_written"], 1)
            sample = ToolCallDataset(dataset_path).load()[0]
            self.assertEqual(sample.source, "gt_progress")
            self.assertEqual(sample.split, "val")
            self.assertTrue(os.path.exists(memory_path))


if __name__ == "__main__":
    unittest.main()
