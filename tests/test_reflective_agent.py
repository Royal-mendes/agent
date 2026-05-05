import os
import tempfile
import unittest

from agent.execution.monitored_navigation_skill_executor import MonitoredNavigationSkillExecutor
from agent.memory.experience_memory import ExperienceMemory
from agent.reflective_navigation_agent import ReflectiveNavigationAgent
from agent.runtime import ReflectiveNavigationRuntime
from agent.schemas import AgentConfig, AgentDecision, ExperienceMemoryItem, SkillExecutionResult, SkillName
from agent.skill.skills import build_default_skill_registry
from agent.state_summarizer import StateSummarizer
from agent.validator import DecisionValidator
from agent.vlm_provider import OpenAICompatibleVLMProvider
from agent.bridge_cli import decide as bridge_decide


class FakeContext:
    def __init__(self, snapshots=None):
        self.snapshots = list(snapshots or [{"timestep": 0}, {"timestep": 1}])
        self.low_value_frontiers = []
        self.blocked_frontiers = []
        self.rejected_targets = []

    def get_navigation_monitor_state(self):
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]

    def call_original_apexnav_policy(self, args):
        return SkillExecutionResult(
            skill_name=SkillName.FALLBACK_APEXNAV.value,
            selected_waypoint=(0.0, 0.0),
        )

    def select_semantic_frontier(self, args):
        return SkillExecutionResult(
            skill_name=SkillName.SEMANTIC_EXPLORE.value,
            selected_frontier_id=args.get("frontier_id"),
            selected_waypoint=(1.0, 2.0),
            raw_metadata={"monitor_information_gain": True},
        )

    def select_nearest_reachable_frontier(self, args):
        return SkillExecutionResult(
            skill_name=SkillName.GEOMETRIC_EXPLORE.value,
            selected_frontier_id=args.get("frontier_id"),
            selected_waypoint=(2.0, 1.0),
        )

    def navigate_to_confirmed_target(self, args):
        return SkillExecutionResult(
            skill_name=SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
            target_candidate_id=args.get("target_candidate_id"),
            raw_metadata={"stop_decision": True},
        )

    def navigate_to_target_observation_viewpoint(self, args):
        return SkillExecutionResult(
            skill_name=SkillName.VERIFY_TARGET.value,
            target_candidate_id=args.get("target_candidate_id"),
        )

    def mark_frontier_low_value(self, frontier_id):
        self.low_value_frontiers.append(frontier_id)

    def mark_frontier_blocked(self, frontier_id):
        self.blocked_frontiers.append(frontier_id)

    def mark_target_candidate_rejected(self, target_id):
        self.rejected_targets.append(target_id)


class BadJsonProvider:
    def generate(self, system_prompt, user_prompt):
        return "not json"


class FencedJsonProvider:
    def generate(self, system_prompt, user_prompt):
        return '```json\n{"selected_skill":"GEOMETRIC_EXPLORE","skill_args":{"frontier_id":1},"confidence":0.6}\n```'


class FakeOpenAIClient:
    class _Message:
        content = '{"selected_skill":"GEOMETRIC_EXPLORE","skill_args":{},"confidence":0.6}'

    class _Choice:
        message = None

    class _Response:
        choices = []

    def __init__(self):
        self.last_kwargs = None
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        choice = self._Choice()
        choice.message = self._Message()
        response = self._Response()
        response.choices = [choice]
        return response


def state_with_target(confidence, multi_view=False, reachable=True):
    return {
        "target_category": "toilet",
        "target_candidates": [
            {
                "id": 7,
                "label": "toilet",
                "confidence": confidence,
                "reachable": reachable,
                "multi_view_confirmed": multi_view,
                "num_views": 2 if multi_view else 1,
                "rejected_false_positive": False,
            }
        ],
        "frontiers": [{"id": 1, "reachable": True, "semantic_score": 0.9, "distance": 2.0}],
        "semantic_score_stats": {"has_clear_peak": True},
        "navigation_history": {"stuck_count": 0, "recent_failures": []},
    }


class ReflectiveAgentTests(unittest.TestCase):
    def test_default_config_keeps_reflective_agent_disabled(self):
        self.assertFalse(AgentConfig().enable_reflective_agent)

    def test_disabled_runtime_calls_original_apexnav_policy(self):
        runtime = ReflectiveNavigationRuntime(AgentConfig(enable_reflective_agent=False))
        result = runtime.decide_and_execute(FakeContext())
        self.assertEqual(result.skill_name, SkillName.FALLBACK_APEXNAV.value)
        self.assertEqual(result.selected_waypoint, (0.0, 0.0))

    def test_mock_agent_selects_confirmed_target_navigation(self):
        agent = ReflectiveNavigationAgent(AgentConfig(vlm_provider="mock"))
        decision = agent.select_skill(state_with_target(0.9, multi_view=True), {}, {}, {})
        self.assertEqual(decision.selected_skill, SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value)

    def test_mock_agent_verifies_low_confidence_single_view_target(self):
        agent = ReflectiveNavigationAgent(AgentConfig(vlm_provider="mock"))
        decision = agent.select_skill(state_with_target(0.68, multi_view=False), {}, {}, {})
        self.assertEqual(decision.selected_skill, SkillName.VERIFY_TARGET.value)

    def test_state_summarizer_includes_yolo_landmark_detections(self):
        state = StateSummarizer(AgentConfig(include_detected_objects_in_state=True)).summarize(
            {
                "target_category": "toilet",
                "detected_objects": {
                    "available": True,
                    "detections": [
                        {
                            "id": 4,
                            "label": "sink",
                            "confidence": 0.82,
                            "bbox": [0.1, 0.2, 0.3, 0.5],
                            "center": [0.2, 0.35],
                            "direction": "left",
                            "source": "yolov7_landmark",
                            "is_landmark": True,
                        }
                    ],
                },
            }
        )
        self.assertEqual(state["detected_objects"][0]["label"], "sink")
        self.assertEqual(state["detected_objects"][0]["direction"], "left")
        self.assertTrue(state["detected_objects"][0]["grounded_in_current_observation"])

    def test_validator_rejects_unreliable_target_navigation(self):
        registry = build_default_skill_registry()
        validator = DecisionValidator(AgentConfig(), registry)
        decision = AgentDecision(
            selected_skill=SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
            skill_args={"target_candidate_id": 7},
        )
        result = validator.validate(decision, state_with_target(0.68, multi_view=False), {}, {}, {})
        self.assertFalse(result.accepted)
        self.assertEqual(result.final_skill, SkillName.VERIFY_TARGET.value)

    def test_validator_uses_memory_to_reject_false_positive_condition(self):
        cfg = AgentConfig(require_multiview_before_stop=False)
        registry = build_default_skill_registry()
        validator = DecisionValidator(cfg, registry)
        lesson = {
            "target_category": "toilet",
            "failure_type": "false_positive_stop",
            "lesson": "For toilet, do not stop on single-view detection.",
        }
        decision = AgentDecision(
            selected_skill=SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value,
            skill_args={"target_candidate_id": 7},
        )
        result = validator.validate(
            decision, state_with_target(0.9, multi_view=False), {}, {}, {}, [lesson]
        )
        self.assertFalse(result.accepted)
        self.assertTrue(result.memory_rule_applied)
        self.assertEqual(result.final_skill, SkillName.VERIFY_TARGET.value)

    def test_validator_uses_repeated_efficiency_memory_to_fallback(self):
        validator = DecisionValidator(AgentConfig(), build_default_skill_registry())
        state = {
            "target_category": "bed",
            "target_candidates": [],
            "frontiers": [{"id": 2, "reachable": True, "semantic_score": 0.1, "distance": 1.0}],
            "semantic_score_stats": {"has_clear_peak": False},
            "navigation_history": {"recent_failures": []},
        }
        lessons = [
            {
                "target_category": "bed",
                "failure_type": "inefficient_exploration",
                "confidence": 0.7,
                "suggested_policy_patch": {"recommended_action": SkillName.FALLBACK_APEXNAV.value},
            },
            {
                "target_category": "bed",
                "failure_type": "inefficient_exploration",
                "confidence": 0.72,
                "better_decision": "skill=FALLBACK_APEXNAV",
            },
        ]
        result = validator.validate(
            AgentDecision(selected_skill=SkillName.GEOMETRIC_EXPLORE.value, skill_args={"frontier_id": 2}),
            state,
            {},
            {},
            {},
            lessons,
        )
        self.assertFalse(result.accepted)
        self.assertTrue(result.memory_rule_applied)
        self.assertEqual(result.final_skill, SkillName.FOLLOW_APEXNAV_PROPOSAL.value)

    def test_validator_does_not_apply_efficiency_memory_without_target(self):
        validator = DecisionValidator(AgentConfig(), build_default_skill_registry())
        state = {
            "target_category": "",
            "target_candidates": [],
            "frontiers": [{"id": 2, "reachable": True, "semantic_score": 0.1, "distance": 1.0}],
            "semantic_score_stats": {"has_clear_peak": False},
            "navigation_history": {"recent_failures": []},
        }
        lessons = [
            {
                "target_category": "bed",
                "failure_type": "inefficient_exploration",
                "confidence": 0.7,
                "suggested_policy_patch": {"recommended_action": SkillName.FALLBACK_APEXNAV.value},
            },
            {
                "target_category": "bed",
                "failure_type": "inefficient_exploration",
                "confidence": 0.72,
                "better_decision": "skill=FALLBACK_APEXNAV",
            },
        ]
        result = validator.validate(
            AgentDecision(selected_skill=SkillName.GEOMETRIC_EXPLORE.value, skill_args={"frontier_id": 2}),
            state,
            {},
            {},
            {},
            lessons,
        )
        self.assertTrue(result.accepted)
        self.assertFalse(result.memory_rule_applied)
        self.assertEqual(result.final_skill, SkillName.GEOMETRIC_EXPLORE.value)

    def test_no_reachable_frontier_falls_back_without_crash(self):
        registry = build_default_skill_registry()
        validator = DecisionValidator(AgentConfig(), registry)
        state = {
            "frontiers": [{"id": 1, "reachable": False, "blocked": True}],
            "target_candidates": [],
            "navigation_history": {},
        }
        result = validator.validate(
            AgentDecision(selected_skill=SkillName.GEOMETRIC_EXPLORE.value),
            state,
            {},
            {},
            {},
        )
        self.assertFalse(result.accepted)
        self.assertIn(result.final_skill, {SkillName.RECOVER_FROM_STUCK.value, SkillName.FALLBACK_APEXNAV.value})

    def test_validator_recovers_when_stuck_threshold_reached(self):
        registry = build_default_skill_registry()
        validator = DecisionValidator(AgentConfig(stuck_threshold=3, enable_stuck_recovery_override=True), registry)
        state = {
            "frontiers": [{"id": 1, "reachable": True, "distance": 1.0}],
            "target_candidates": [],
            "navigation_history": {"stuck_count": 3, "collision_count": 0, "recent_failures": []},
        }
        result = validator.validate(
            AgentDecision(selected_skill=SkillName.GEOMETRIC_EXPLORE.value, skill_args={"frontier_id": 1}),
            state,
            {},
            {},
            {},
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.final_skill, SkillName.RECOVER_FROM_STUCK.value)

    def test_validator_does_not_force_recovery_from_raw_stuck_by_default(self):
        registry = build_default_skill_registry()
        validator = DecisionValidator(AgentConfig(stuck_threshold=3), registry)
        state = {
            "frontiers": [{"id": 1, "reachable": True, "distance": 1.0}],
            "target_candidates": [],
            "navigation_history": {"stuck_count": 3, "collision_count": 0, "recent_failures": []},
        }
        result = validator.validate(
            AgentDecision(selected_skill=SkillName.GEOMETRIC_EXPLORE.value, skill_args={"frontier_id": 1}),
            state,
            {},
            {},
            {},
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.final_skill, SkillName.GEOMETRIC_EXPLORE.value)

    def test_validator_cools_down_repeated_recovery(self):
        registry = build_default_skill_registry()
        validator = DecisionValidator(AgentConfig(stuck_threshold=3), registry)
        state = {
            "frontiers": [{"id": 1, "reachable": True, "distance": 1.0}],
            "target_candidates": [],
            "navigation_history": {
                "stuck_count": 3,
                "collision_count": 0,
                "recent_failures": [],
                "recent_selected_skills": [SkillName.RECOVER_FROM_STUCK.value],
            },
        }
        result = validator.validate(
            AgentDecision(selected_skill=SkillName.RECOVER_FROM_STUCK.value),
            state,
            {},
            {},
            {},
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.final_skill, SkillName.FALLBACK_APEXNAV.value)

    def test_validator_allows_recovery_with_objective_stuck_signal(self):
        registry = build_default_skill_registry()
        validator = DecisionValidator(AgentConfig(stuck_threshold=3), registry)
        state = {
            "frontiers": [{"id": 1, "reachable": True, "distance": 1.0}],
            "target_candidates": [],
            "semantic_score_stats": {"has_clear_peak": False},
            "navigation_history": {"stuck_count": 3, "collision_count": 0, "recent_failures": []},
        }
        result = validator.validate(
            AgentDecision(selected_skill=SkillName.RECOVER_FROM_STUCK.value),
            state,
            {},
            {},
            {},
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.final_skill, SkillName.RECOVER_FROM_STUCK.value)

    def test_validator_rejects_recovery_without_objective_signal(self):
        registry = build_default_skill_registry()
        validator = DecisionValidator(AgentConfig(stuck_threshold=3), registry)
        state = {
            "frontiers": [{"id": 1, "reachable": True, "distance": 1.0}],
            "target_candidates": [],
            "semantic_score_stats": {"has_clear_peak": False},
            "navigation_history": {"stuck_count": 0, "collision_count": 0, "recent_failures": []},
        }
        result = validator.validate(
            AgentDecision(selected_skill=SkillName.RECOVER_FROM_STUCK.value),
            state,
            {},
            {},
            {},
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.final_skill, SkillName.GEOMETRIC_EXPLORE.value)

    def test_validator_allows_recovery_with_explicit_failure_marker(self):
        registry = build_default_skill_registry()
        validator = DecisionValidator(AgentConfig(stuck_threshold=3), registry)
        state = {
            "frontiers": [{"id": 1, "reachable": True, "distance": 1.0}],
            "target_candidates": [],
            "navigation_history": {"stuck_count": 3, "recent_failures": ["planner_stuck"]},
        }
        result = validator.validate(
            AgentDecision(selected_skill=SkillName.RECOVER_FROM_STUCK.value),
            state,
            {},
            {},
            {},
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.final_skill, SkillName.RECOVER_FROM_STUCK.value)

    def test_executor_detects_timeout(self):
        registry = build_default_skill_registry()
        context = FakeContext([{"timestep": 0}, {"timestep": 100}])

        def slow_handler(args, ctx):
            return SkillExecutionResult(
                skill_name=SkillName.GEOMETRIC_EXPLORE.value,
                selected_frontier_id=2,
                raw_metadata={"timeout_steps": 50},
            )

        registry.register(registry.get_spec(SkillName.GEOMETRIC_EXPLORE.value), slow_handler, replace=True)
        executor = MonitoredNavigationSkillExecutor(AgentConfig(), registry)
        result = executor.execute(SkillName.GEOMETRIC_EXPLORE.value, {"frontier_id": 2}, context)
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.failure_type, "timeout")

    def test_executor_detects_stuck(self):
        registry = build_default_skill_registry()
        context = FakeContext([{"timestep": 0, "stuck_count": 0}, {"timestep": 3, "stuck_count": 3}])
        executor = MonitoredNavigationSkillExecutor(AgentConfig(stuck_threshold=3), registry)
        result = executor.execute(SkillName.GEOMETRIC_EXPLORE.value, {"frontier_id": 1}, context)
        self.assertEqual(result.failure_type, "planner_stuck")

    def test_semantic_failure_marks_frontier_low_value(self):
        registry = build_default_skill_registry()
        context = FakeContext(
            [
                {"timestep": 0, "explored_area": 10.0, "semantic_score": 0.5},
                {"timestep": 1, "explored_area": 10.0, "semantic_score": 0.5},
            ]
        )
        executor = MonitoredNavigationSkillExecutor(AgentConfig(), registry)
        result = executor.execute(SkillName.SEMANTIC_EXPLORE.value, {"frontier_id": 1}, context)
        self.assertEqual(result.failure_type, "low_information_gain")
        self.assertIn(1, context.low_value_frontiers)

    def test_verify_failure_marks_candidate_rejected(self):
        registry = build_default_skill_registry()
        context = FakeContext(
            [
                {"timestep": 0, "target_confidence": 0.8},
                {"timestep": 1, "target_confidence": 0.3},
            ]
        )
        executor = MonitoredNavigationSkillExecutor(AgentConfig(), registry)
        result = executor.execute(SkillName.VERIFY_TARGET.value, {"target_candidate_id": 7}, context)
        self.assertEqual(result.failure_type, "false_positive_candidate")
        self.assertIn(7, context.rejected_targets)

    def test_experience_memory_writes_and_retrieves_by_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.jsonl")
            memory = ExperienceMemory(path, write_mode="all")
            memory.append_memory(
                ExperienceMemoryItem(
                    split="train",
                    target_category="toilet",
                    failure_type="false_positive_stop",
                    lesson="Do not stop on single-view toilet.",
                    confidence=0.8,
                )
            )
            retrieved = memory.retrieve({"target_category": "toilet"}, top_k=1)
            self.assertEqual(len(retrieved), 1)
            self.assertEqual(retrieved[0]["target_category"], "toilet")

    def test_test_split_does_not_write_memory_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.jsonl")
            memory = ExperienceMemory(path, write_mode="train_only")
            wrote = memory.append_memory(ExperienceMemoryItem(split="test", target_category="toilet"))
            self.assertFalse(wrote)
            self.assertFalse(os.path.exists(path))

    def test_invalid_vlm_json_falls_back_to_apexnav(self):
        agent = ReflectiveNavigationAgent(
            AgentConfig(vlm_provider="openai"),
            build_default_skill_registry(),
            vlm_provider=BadJsonProvider(),
        )
        decision = agent.select_skill(state_with_target(0.9, multi_view=True), {}, {}, {})
        self.assertEqual(decision.selected_skill, SkillName.FALLBACK_APEXNAV.value)

    def test_fenced_vlm_json_is_parsed(self):
        agent = ReflectiveNavigationAgent(
            AgentConfig(vlm_provider="local"),
            build_default_skill_registry(),
            vlm_provider=FencedJsonProvider(),
        )
        decision = agent.select_skill(state_with_target(0.4), {}, {}, {})
        self.assertEqual(decision.selected_skill, SkillName.GEOMETRIC_EXPLORE.value)
        self.assertEqual(decision.skill_args["frontier_id"], 1)

    def test_validator_replaces_invalid_frontier_id_with_reachable_choice(self):
        validator = DecisionValidator(AgentConfig(), build_default_skill_registry())
        state = state_with_target(0.4)
        state["target_candidates"] = []
        decision = AgentDecision(
            selected_skill=SkillName.SEMANTIC_EXPLORE.value,
            skill_args={"frontier_id": 12},
        )
        result = validator.validate(decision, state, {}, {}, {})
        self.assertFalse(result.accepted)
        self.assertEqual(result.final_skill, SkillName.SEMANTIC_EXPLORE.value)
        self.assertEqual(result.final_arguments["frontier_id"], 1)

    def test_validator_preempts_exploration_when_target_candidate_exists(self):
        validator = DecisionValidator(AgentConfig(), build_default_skill_registry())
        result = validator.validate(
            AgentDecision(selected_skill=SkillName.GEOMETRIC_EXPLORE.value, skill_args={"frontier_id": 1}),
            state_with_target(0.2, multi_view=False),
            {},
            {},
            {},
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.final_skill, SkillName.VERIFY_TARGET.value)

    def test_mock_agent_can_follow_apexnav_for_baseline_sanity(self):
        agent = ReflectiveNavigationAgent(AgentConfig(vlm_provider="mock", mock_follow_apexnav_by_default=True))
        state = state_with_target(0.1)
        state["target_candidates"] = []
        decision = agent.select_skill(state, {}, {}, {})
        self.assertEqual(decision.selected_skill, SkillName.FOLLOW_APEXNAV_PROPOSAL.value)

    def test_force_all_decisions_to_fallback(self):
        agent = ReflectiveNavigationAgent(
            AgentConfig(
                vlm_provider="mock",
                force_all_decisions_to_FALLBACK_APEXNAV=True,
            )
        )
        decision = agent.select_skill(state_with_target(0.9, multi_view=True), {}, {}, {})
        self.assertEqual(decision.selected_skill, SkillName.FALLBACK_APEXNAV.value)

    def test_validator_fallback_to_exploration_keeps_frontier_argument(self):
        validator = DecisionValidator(AgentConfig(), build_default_skill_registry())
        state = state_with_target(0.4)
        state["target_candidates"] = []
        result = validator.validate(
            AgentDecision(selected_skill=SkillName.VERIFY_TARGET.value, skill_args={}),
            state,
            {},
            {},
            {},
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.final_skill, SkillName.SEMANTIC_EXPLORE.value)
        self.assertEqual(result.final_arguments["frontier_id"], 1)

    def test_openai_compatible_provider_uses_chat_completions_client(self):
        fake_client = FakeOpenAIClient()
        provider = OpenAICompatibleVLMProvider(
            model="test-model",
            api_key="test-key",
            base_url="https://relay.example/v1",
            client=fake_client,
        )
        text = provider.generate("system", "user")
        self.assertIn("GEOMETRIC_EXPLORE", text)
        self.assertEqual(fake_client.last_kwargs["model"], "test-model")
        self.assertEqual(fake_client.last_kwargs["messages"][0]["role"], "system")

    def test_bridge_cli_returns_validated_skill_json(self):
        result = bridge_decide(
            {
                "config": {
                    "enable_reflective_agent": True,
                    "vlm_provider": "mock",
                    "enable_reflection_memory": False,
                },
                "state": state_with_target(0.68, multi_view=False),
            }
        )
        self.assertEqual(result["selected_skill"], SkillName.VERIFY_TARGET.value)
        self.assertIn("agent_decision", result)
        self.assertIn("validator_result", result)

    def test_bridge_cli_force_fallback_bypasses_target_preemption(self):
        result = bridge_decide(
            {
                "config": {
                    "enable_reflective_agent": True,
                    "vlm_provider": "mock",
                    "enable_reflection_memory": False,
                    "force_all_decisions_to_FALLBACK_APEXNAV": True,
                },
                "state": state_with_target(0.9, multi_view=True),
            }
        )
        self.assertEqual(result["selected_skill"], SkillName.FALLBACK_APEXNAV.value)
        self.assertTrue(result["validator_result"]["fallback_used"])

    def test_bridge_cli_disabled_falls_back(self):
        result = bridge_decide({"config": {"enable_reflective_agent": False}, "state": {}})
        self.assertEqual(result["selected_skill"], SkillName.FALLBACK_APEXNAV.value)
        self.assertTrue(result["fallback_used"])


if __name__ == "__main__":
    unittest.main()
