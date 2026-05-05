import json
import os
import tempfile
import unittest

from agent.logging.episode_logger import EpisodeLogger
from agent.memory.experience_memory import ExperienceMemory
from agent.reflection.failure_taxonomy import classify_failure
from agent.reflection.policy_patch import PolicyPatchTable
from agent.reflection.reflection_engine import ReflectionEngine
from agent.schemas import AgentConfig, AgentDecision, FailureClass, PolicyPatchProposal, SkillName
from agent.skill.skills import build_default_skill_registry
from agent.validator import DecisionValidator


def target_state(confidence=0.68, multi_view=False):
    return {
        "target_category": "toilet",
        "target_candidates": [
            {
                "id": 3,
                "label": "toilet",
                "confidence": confidence,
                "reachable": True,
                "multi_view_confirmed": multi_view,
                "num_views": 2 if multi_view else 1,
                "rejected_false_positive": False,
            }
        ],
        "frontiers": [{"id": 1, "reachable": True, "semantic_score": 0.4, "distance": 2.0}],
        "semantic_score_stats": {"has_clear_peak": True},
        "navigation_history": {"recent_failures": [], "steps_left": 100},
    }


class ReflectionPipelineTests(unittest.TestCase):
    def test_failure_taxonomy_classifies_and_escalates(self):
        self.assertEqual(classify_failure("false_positive_stop"), FailureClass.DEGRADING.value)
        self.assertEqual(classify_failure("semantic_explore_no_progress"), FailureClass.NON_DEGRADING.value)
        self.assertEqual(
            classify_failure("semantic_explore_no_progress", consecutive_count=3),
            FailureClass.DEGRADING.value,
        )

    def test_reflection_engine_generates_false_positive_memory_and_patch(self):
        engine = ReflectionEngine(AgentConfig(enable_episode_reflection=True, enable_reflection_memory=True))
        result = engine.reflect_episode(
            {
                "episode_id": "ep_fp",
                "scene_id": "scene",
                "split": "train",
                "target_category": "toilet",
                "success": False,
                "failure_type": "false_positive_stop",
                "selected_skill_sequence": [
                    SkillName.SEMANTIC_EXPLORE.value,
                    SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
                ],
                "target_confidence": 0.62,
                "multi_view_confirmed": False,
            }
        )
        item = result["experience_memory_item"]
        self.assertEqual(item["failure_class"], FailureClass.DEGRADING.value)
        self.assertIn("verify", item["lesson"].lower())
        self.assertEqual(result["policy_patch_proposals"][0]["recommended_action"], SkillName.VERIFY_TARGET.value)

    def test_reflection_engine_writes_train_memory_and_skips_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = os.path.join(tmpdir, "memory.jsonl")
            patch_path = os.path.join(tmpdir, "patches.json")
            cfg = AgentConfig(
                enable_reflection_memory=True,
                enable_episode_reflection=True,
                memory_path=memory_path,
                memory_write_mode="train_only",
                policy_patch_path=patch_path,
            )
            engine = ReflectionEngine(cfg)
            train_result = engine.finalize_episode(
                {
                    "episode_id": "train_ep",
                    "split": "train",
                    "target_category": "toilet",
                    "success": False,
                    "failure_type": "semantic_explore_no_progress",
                }
            )
            self.assertTrue(train_result["memory_written"])
            test_result = engine.finalize_episode(
                {
                    "episode_id": "test_ep",
                    "split": "test",
                    "target_category": "toilet",
                    "success": False,
                    "failure_type": "false_positive_stop",
                }
            )
            self.assertFalse(test_result["memory_written"])
            memory = ExperienceMemory(memory_path, write_mode="all")
            self.assertEqual(len(memory.load_memory()), 1)

    def test_policy_patch_table_requires_support_for_activation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "patches.json")
            cfg = AgentConfig(
                auto_activate_policy_patches=True,
                min_policy_patch_support=3,
                policy_patch_confidence_threshold=0.7,
            )
            table = PolicyPatchTable(path, cfg)
            proposal = PolicyPatchProposal(
                target_scope="toilet",
                trigger_condition={
                    "selected_skill": SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
                    "target_confidence_lt": 0.75,
                    "multi_view_confirmed": False,
                },
                recommended_action=SkillName.VERIFY_TARGET.value,
                confidence=0.82,
            )
            self.assertFalse(table.record_proposal(proposal, split="train").active)
            table.record_proposal(proposal, split="train")
            activated = table.record_proposal(proposal, split="train")
            self.assertTrue(activated.active)
            self.assertEqual(len(table.get_active_patches("toilet")), 1)

    def test_validator_applies_active_policy_patch(self):
        registry = build_default_skill_registry()
        validator = DecisionValidator(
            AgentConfig(require_multiview_before_stop=False, target_stop_threshold=0.6),
            registry,
        )
        patch = {
            "target_scope": "toilet",
            "active": True,
            "trigger_condition": {
                "selected_skill": SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
                "target_confidence_lt": 0.75,
                "multi_view_confirmed": False,
            },
            "recommended_action": SkillName.VERIFY_TARGET.value,
        }
        result = validator.validate(
            AgentDecision(
                selected_skill=SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
                skill_args={"target_candidate_id": 3},
            ),
            target_state(confidence=0.7, multi_view=False),
            {},
            {},
            {},
            active_policy_patches=[patch],
        )
        self.assertFalse(result.accepted)
        self.assertTrue(result.policy_patch_applied)
        self.assertEqual(result.final_skill, SkillName.VERIFY_TARGET.value)

    def test_episode_logger_records_decision_and_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EpisodeLogger(tmpdir, run_id="run")
            logger.log_decision(
                episode_id="ep1",
                timestep=1,
                state_summary={"target_category": "toilet"},
                role_memory={},
                task_memory_snapshot={},
                working_memory_snapshot={},
                retrieved_lessons=[],
                active_policy_patches=[],
                agent_decision={"selected_skill": SkillName.GEOMETRIC_EXPLORE.value},
                validator_result={"final_skill": SkillName.GEOMETRIC_EXPLORE.value},
                executed_skill=SkillName.GEOMETRIC_EXPLORE.value,
                skill_result=None,
            )
            logger.log_episode_end(
                "ep1",
                {"success": False, "stop_action_source": "terminal_no_frontier"},
                {"memory_written": False},
            )
            with open(logger.episode_path("ep1"), "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data["decisions"]), 1)
            self.assertIn("reflection", data)
            diagnostics = data["episode_end"]["agent_diagnostics"]
            self.assertEqual(diagnostics["vlm_call_count"], 1)
            self.assertEqual(diagnostics["skill_distribution"][SkillName.GEOMETRIC_EXPLORE.value], 1)
            self.assertEqual(diagnostics["stop_action_source_histogram"]["terminal_no_frontier"], 1)


if __name__ == "__main__":
    unittest.main()
