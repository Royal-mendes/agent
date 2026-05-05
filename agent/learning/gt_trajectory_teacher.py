from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from agent.learning.trajectory_schema import TeacherAction, ToolUseLearningSample, TrajectoryEpisode, TrajectoryStep
from agent.reflection.failure_taxonomy import classify_failure
from agent.schemas import AgentConfig, SkillName


GT_TRAJECTORY_REFLECTION_SYSTEM_PROMPT = """You are a trajectory reflection teacher for object-goal navigation.
Use GT shortest-path context only for post-episode learning, not online control.
Do not require or inspect RGB image pixels here; use structured YOLO landmarks, frontier/map state, and GT path context.
Explain why the student's selected high-level skill deviated from the GT trajectory, which observable landmark or map cue should have anchored the decision, why the GT path was preferable, and what skill should be selected next time.
Return only valid JSON. Do not reveal chain-of-thought; provide concise public reasons."""


class GTTrajectoryTeacher:
    """Generate lessons when the agent deviates from the GT shortest path."""

    def __init__(self, cfg: Optional[AgentConfig] = None, vlm_provider: Optional[Any] = None) -> None:
        self.cfg = cfg or AgentConfig()
        self.vlm_provider = vlm_provider

    def build_samples(self, episode: TrajectoryEpisode) -> List[ToolUseLearningSample]:
        steps = [step for step in episode.trajectory_steps if step.selected_skill]
        samples: List[ToolUseLearningSample] = []
        for index, step in enumerate(steps):
            after_step = self._next_step_with_gt(steps, index)
            if after_step is None:
                continue
            before = self._gt_context(step)
            after = self._gt_context(after_step)
            if not before.get("gt_path_available") or not after.get("gt_path_available"):
                continue
            before_dev = self._float(before.get("distance_to_gt_path"))
            after_dev = self._float(after.get("distance_to_gt_path"))
            if before_dev is None or after_dev is None:
                continue
            growth = after_dev - before_dev
            if after_dev < self.cfg.gt_path_deviation_threshold and growth < self.cfg.gt_path_deviation_growth_threshold:
                continue
            samples.append(self._deviation_sample(episode, step, before, after, before_dev, after_dev, growth))
            if len(samples) >= self.cfg.max_gt_trajectory_reflections_per_episode:
                break
        return samples

    def _deviation_sample(
        self,
        episode: TrajectoryEpisode,
        step: TrajectoryStep,
        before: Dict[str, Any],
        after: Dict[str, Any],
        before_dev: float,
        after_dev: float,
        growth: float,
    ) -> ToolUseLearningSample:
        reflection = self._reflect_with_vlm(episode, step, before, after, before_dev, after_dev, growth)
        teacher_skill = reflection.get("better_skill") or self._heuristic_better_skill(step)
        teacher_args = self._heuristic_args_for_skill(step, teacher_skill)
        teacher = TeacherAction(
            skill_name=teacher_skill,
            tool_name=self._tool_for_skill(teacher_skill),
            arguments=teacher_args,
            reason=reflection.get("gt_rationale")
            or "GT shortest path stayed on a lower-deviation corridor toward a valid goal viewpoint.",
            confidence=float(reflection.get("confidence") or 0.78),
            source="gt_trajectory_vlm" if reflection.get("source") == "vlm" else "gt_trajectory",
        )
        failure_type = "gt_trajectory_deviation"
        lesson = reflection.get("lesson") or (
            f"After {step.selected_skill}, distance to the GT shortest path increased from "
            f"{before_dev:.2f}m to {after_dev:.2f}m. In similar states, prefer {teacher.skill_name} "
            "and avoid committing to a frontier that pulls the agent away from the GT corridor."
        )
        return ToolUseLearningSample(
            source=teacher.source,
            episode_id=episode.episode_id,
            scene_id=episode.scene_id,
            split=episode.split,
            target_category=episode.target_category or step.state_summary.get("target_category"),
            timestep=step.timestep,
            state_condition=self._state_condition(step, before, after, before_dev, after_dev, growth),
            student_action={
                "skill": step.selected_skill,
                "tool": step.tool_name,
                "arguments": step.skill_args,
                "validator_result": step.validator_result,
            },
            teacher_action=teacher.to_dict(),
            outcome="gt_trajectory_deviation_reflection",
            failure_type=failure_type,
            failure_class=classify_failure(failure_type),
            lesson=lesson,
            policy_patch_proposal={
                "target_scope": episode.target_category or step.state_summary.get("target_category"),
                "trigger_condition": {
                    "selected_skill": step.selected_skill,
                    "failure_type": failure_type,
                    "distance_to_gt_path_gte": self.cfg.gt_path_deviation_threshold,
                },
                "recommended_action": teacher.skill_name,
                "rationale": teacher.reason,
                "confidence": teacher.confidence,
                "support_count": 1,
                "source_episode_id": episode.episode_id,
            },
            confidence=teacher.confidence,
            metadata={
                "gt_deviation_before": before_dev,
                "gt_deviation_after": after_dev,
                "gt_deviation_growth": growth,
                "gt_nearest_index_before": before.get("nearest_gt_path_index"),
                "gt_nearest_index_after": after.get("nearest_gt_path_index"),
                "gt_next_waypoint_before": before.get("gt_next_waypoint"),
                "landmark": reflection.get("landmark") or self._heuristic_landmark(step),
                "direction": reflection.get("direction") or self._heuristic_direction(before, after),
                "agent_position_before": before.get("agent_position"),
                "agent_position_after": after.get("agent_position"),
                "reflection": reflection,
            },
        )

    def _reflect_with_vlm(
        self,
        episode: TrajectoryEpisode,
        step: TrajectoryStep,
        before: Dict[str, Any],
        after: Dict[str, Any],
        before_dev: float,
        after_dev: float,
        growth: float,
    ) -> Dict[str, Any]:
        if not self.cfg.enable_vlm_gt_trajectory_reflection or self.vlm_provider is None:
            return {"source": "heuristic"}
        payload = {
            "task": "Reflect on a high-level navigation skill that deviated from the GT shortest path.",
            "required_json_format": {
                "failure_analysis": "short public explanation of what went wrong",
                "landmark": "observable YOLO landmark or map/frontier cue that should anchor the decision",
                "direction": "short navigation direction relative to that landmark or map cue",
                "gt_rationale": "why the GT trajectory direction was preferable",
                "better_skill": "one of SEMANTIC_EXPLORE, GEOMETRIC_EXPLORE, VERIFY_TARGET, NAVIGATE_TO_CONFIRMED_TARGET, RECOVER_FROM_STUCK, FOLLOW_APEXNAV_PROPOSAL, FALLBACK_APEXNAV",
                "lesson": "short reusable lesson for future VLM skill selection",
                "confidence": 0.0,
            },
            "episode": {
                "episode_id": episode.episode_id,
                "scene_id": episode.scene_id,
                "split": episode.split,
                "target_category": episode.target_category,
                "success": episode.success,
            },
            "student_decision": {
                "timestep": step.timestep,
                "selected_skill": step.selected_skill,
                "skill_args": step.skill_args,
                "validator_result": step.validator_result,
            },
            "state_summary": self._prompt_safe_state(step.state_summary),
            "gt_trajectory_context_before": before,
            "gt_trajectory_context_after": after,
            "deviation_metrics": {
                "distance_to_gt_path_before": before_dev,
                "distance_to_gt_path_after": after_dev,
                "deviation_growth": growth,
                "deviation_threshold": self.cfg.gt_path_deviation_threshold,
            },
        }
        try:
            raw = self.vlm_provider.generate(
                GT_TRAJECTORY_REFLECTION_SYSTEM_PROMPT,
                json.dumps(payload, indent=2, sort_keys=True),
            )
            parsed = self._parse_json(raw)
            if parsed:
                parsed["source"] = "vlm"
                parsed["raw_response"] = raw
                if parsed.get("better_skill") not in {item.value for item in SkillName}:
                    parsed["better_skill"] = self._heuristic_better_skill(step)
                return parsed
        except Exception as exc:
            return {"source": "heuristic", "error": f"{type(exc).__name__}: {exc}"}
        return {"source": "heuristic"}

    def _heuristic_better_skill(self, step: TrajectoryStep) -> str:
        state = step.state_summary or {}
        candidate = self._best_target_candidate(state)
        if candidate:
            if self._candidate_stop_ready(candidate):
                return SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value
            return SkillName.VERIFY_TARGET.value
        recent_failures = set((state.get("navigation_history") or {}).get("recent_failures") or [])
        if recent_failures & {"planner_stuck", "repeated_collision", "frontier_failure", "no_frontier_deadend"}:
            return SkillName.RECOVER_FROM_STUCK.value
        if (state.get("semantic_score_stats") or {}).get("has_clear_peak"):
            return SkillName.SEMANTIC_EXPLORE.value
        if state.get("frontiers"):
            return SkillName.GEOMETRIC_EXPLORE.value
        return SkillName.FALLBACK_APEXNAV.value

    @staticmethod
    def _heuristic_landmark(step: TrajectoryStep) -> str:
        state = step.state_summary or {}
        detections = [
            item for item in state.get("detected_objects", []) if item.get("label") and item.get("is_landmark", True)
        ]
        if detections:
            best = max(detections, key=lambda item: item.get("confidence") or 0.0)
            direction = best.get("direction") or "visible"
            return f"{best.get('label')} landmark in the {direction} view"
        if (state.get("semantic_score_stats") or {}).get("has_clear_peak"):
            return "high semantic score region on the semantic map"
        if state.get("frontiers"):
            return "reachable frontier cluster on the map"
        return "current map evidence"

    @staticmethod
    def _heuristic_direction(before: Dict[str, Any], after: Dict[str, Any]) -> str:
        waypoint = before.get("gt_next_waypoint")
        if waypoint is not None:
            return f"move toward GT local waypoint {waypoint}"
        if after.get("distance_to_gt_path") is not None:
            return "reduce deviation from the GT path corridor"
        return "follow the map cue closest to the GT corridor"

    def _heuristic_args_for_skill(self, step: TrajectoryStep, skill: str) -> Dict[str, Any]:
        state = step.state_summary or {}
        if skill in {SkillName.VERIFY_TARGET.value, SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value}:
            candidate = self._best_target_candidate(state)
            return {"target_candidate_id": None if candidate is None else candidate.get("id")}
        if skill == SkillName.SEMANTIC_EXPLORE.value:
            frontier = self._best_frontier(state, prefer_semantic=True)
            return {"frontier_id": None if frontier is None else frontier.get("id")}
        if skill == SkillName.GEOMETRIC_EXPLORE.value:
            frontier = self._best_frontier(state, prefer_semantic=False)
            return {"frontier_id": None if frontier is None else frontier.get("id")}
        return {}

    def _state_condition(
        self,
        step: TrajectoryStep,
        before: Dict[str, Any],
        after: Dict[str, Any],
        before_dev: float,
        after_dev: float,
        growth: float,
    ) -> Dict[str, Any]:
        state = step.state_summary or {}
        candidates = state.get("target_candidates") or []
        best_candidate = self._best_target_candidate(state) or {}
        return {
            "student_selected_skill": step.selected_skill,
            "semantic_peak": (state.get("semantic_score_stats") or {}).get("has_clear_peak"),
            "frontier_count": len(state.get("frontiers") or []),
            "target_candidate_count": len(candidates),
            "target_confidence": best_candidate.get("confidence"),
            "multi_view_confirmed": best_candidate.get("multi_view_confirmed", False),
            "distance_to_gt_path_before": before_dev,
            "distance_to_gt_path_after": after_dev,
            "gt_deviation_growth": growth,
            "gt_progress_ratio_before": before.get("gt_path_progress_ratio"),
            "gt_progress_ratio_after": after.get("gt_path_progress_ratio"),
        }

    @staticmethod
    def _gt_context(step: TrajectoryStep) -> Dict[str, Any]:
        return dict(((step.state_summary or {}).get("gt_feedback") or {}).get("gt_trajectory") or {})

    def _next_step_with_gt(self, steps: List[TrajectoryStep], index: int) -> Optional[TrajectoryStep]:
        for step in steps[index + 1 :]:
            context = self._gt_context(step)
            if context.get("gt_path_available"):
                return step
        return None

    @staticmethod
    def _prompt_safe_state(state: Dict[str, Any]) -> Dict[str, Any]:
        safe = json.loads(json.dumps(state, default=str))
        safe.pop("gt_feedback", None)
        for key in ("rgb_observation", "semantic_map_observation"):
            observation = safe.get(key)
            if isinstance(observation, dict):
                observation.pop("data_url", None)
                observation.pop("image_url", None)
        return safe

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    return {}
        return {}

    @staticmethod
    def _best_target_candidate(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidates = list(state.get("target_candidates") or [])
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.get("confidence") or 0.0)

    def _candidate_stop_ready(self, candidate: Dict[str, Any]) -> bool:
        confidence = candidate.get("confidence") or 0.0
        if confidence < self.cfg.target_stop_threshold:
            return False
        if self.cfg.require_multiview_before_stop and not candidate.get("multi_view_confirmed"):
            return False
        return bool(candidate.get("reachable", True)) and not candidate.get("rejected_false_positive", False)

    @staticmethod
    def _best_frontier(state: Dict[str, Any], prefer_semantic: bool) -> Optional[Dict[str, Any]]:
        frontiers = [
            item
            for item in state.get("frontiers", [])
            if item.get("reachable", True) and not item.get("blocked") and not item.get("low_value")
        ]
        if not frontiers:
            return None
        if prefer_semantic:
            return max(frontiers, key=lambda item: item.get("semantic_score") or 0.0)
        return min(frontiers, key=lambda item: item.get("distance") if item.get("distance") is not None else math.inf)

    @staticmethod
    def _tool_for_skill(skill: Optional[str]) -> str:
        return {
            SkillName.SEMANTIC_EXPLORE.value: "select_semantic_frontier",
            SkillName.GEOMETRIC_EXPLORE.value: "select_nearest_reachable_frontier",
            SkillName.VERIFY_TARGET.value: "verify_target_candidate",
            SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value: "navigate_to_confirmed_target",
            SkillName.RECOVER_FROM_STUCK.value: "recover_from_stuck",
            SkillName.FOLLOW_APEXNAV_PROPOSAL.value: "call_original_apexnav_policy",
            SkillName.FALLBACK_APEXNAV.value: "call_original_apexnav_policy",
        }.get(skill, "call_original_apexnav_policy")

    @staticmethod
    def _float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
