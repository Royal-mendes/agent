from __future__ import annotations

import copy
import math
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

from agent.schemas import AgentConfig


class StateSummarizer:
    """Build a compact, robust state summary for high-level skill selection."""

    def __init__(self, cfg: Optional[AgentConfig] = None) -> None:
        self.cfg = cfg or AgentConfig()

    def summarize(
        self,
        context: Optional[Any] = None,
        retrieved_lessons: Optional[List[Dict[str, Any]]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        overrides = overrides or {}

        state = {
            "episode_id": self._get(context, "episode_id"),
            "scene_id": self._get(context, "scene_id"),
            "split": self._get(context, "split", "unknown"),
            "timestep": self._get(context, "timestep", self._get(context, "step", None)),
            "target_category": self._get(context, "target_category", self._get(context, "label", None)),
            "current_observation_summary": self._get(context, "current_observation_summary", None),
            "semantic_score_stats": self._semantic_score_stats(context),
            "frontiers": self._frontiers(context),
            "target_candidates": self._target_candidates(context),
            "rgb_observation": self._rgb_observation(context),
            "semantic_map_observation": self._semantic_map_observation(context),
            "gt_feedback": self._gt_feedback(context),
            "navigation_history": self._navigation_history(context),
            "retrieved_lessons": retrieved_lessons or self._get(context, "retrieved_lessons", []),
            "bridge_diagnostics": self._get(context, "bridge_diagnostics", {}),
            "stop_validator": self._get(context, "stop_validator", {}),
        }
        if self.cfg.include_detected_objects_in_state:
            state["detected_objects"] = self._detected_objects(context)
        state.update({k: v for k, v in overrides.items() if v is not None})
        if state["current_observation_summary"] is None:
            state["current_observation_summary"] = self._default_observation_summary(state)
        return state

    def _semantic_score_stats(self, context: Any) -> Dict[str, Any]:
        explicit = self._get(context, "semantic_score_stats", None)
        if isinstance(explicit, dict):
            stats = {
                "max": explicit.get("max"),
                "mean": explicit.get("mean"),
                "std": explicit.get("std"),
                "max_to_mean": explicit.get("max_to_mean"),
                "has_clear_peak": explicit.get("has_clear_peak"),
            }
            return self._fill_peak_flag(stats)

        scores = []
        for frontier in self._frontiers(context):
            value = frontier.get("semantic_score")
            if value is not None:
                try:
                    scores.append(float(value))
                except (TypeError, ValueError):
                    pass
        if not scores:
            return {
                "max": None,
                "mean": None,
                "std": None,
                "max_to_mean": None,
                "has_clear_peak": False,
            }
        score_mean = mean(scores)
        score_std = pstdev(scores) if len(scores) > 1 else 0.0
        stats = {
            "max": max(scores),
            "mean": score_mean,
            "std": score_std,
            "max_to_mean": max(scores) / max(score_mean, 1e-6),
            "has_clear_peak": None,
        }
        return self._fill_peak_flag(stats)

    def _fill_peak_flag(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        if stats.get("has_clear_peak") is None:
            ratio = stats.get("max_to_mean")
            std = stats.get("std")
            stats["has_clear_peak"] = bool(
                ratio is not None
                and std is not None
                and ratio >= self.cfg.semantic_peak_ratio_threshold
                and std >= self.cfg.semantic_peak_std_threshold
            )
        return stats

    def _frontiers(self, context: Any) -> List[Dict[str, Any]]:
        frontiers = self._get(context, "frontiers", [])
        result = []
        for idx, frontier in enumerate(self._iter_dicts(frontiers)):
            item = {
                "id": frontier.get("id", idx),
                "semantic_score": frontier.get("semantic_score"),
                "distance": frontier.get("distance"),
                "reachable": frontier.get("reachable", True),
                "visited": bool(frontier.get("visited", False)),
                "blocked": bool(frontier.get("blocked", False)),
                "low_value": bool(frontier.get("low_value", False)),
                "room_hint": frontier.get("room_hint"),
                "last_selected": bool(frontier.get("last_selected", False)),
                "failure_count": int(frontier.get("failure_count", 0) or 0),
                "waypoint": frontier.get("waypoint"),
            }
            result.append(item)
        return result

    def _target_candidates(self, context: Any) -> List[Dict[str, Any]]:
        candidates = self._get(context, "target_candidates", self._get(context, "targets", []))
        result = []
        for idx, candidate in enumerate(self._iter_dicts(candidates)):
            item = {
                "id": candidate.get("id", idx),
                "label": candidate.get("label", self._get(context, "target_category", None)),
                "confidence": candidate.get("confidence"),
                "distance": candidate.get("distance"),
                "reachable": candidate.get("reachable", True),
                "multi_view_confirmed": bool(candidate.get("multi_view_confirmed", False)),
                "num_views": int(candidate.get("num_views", 0) or 0),
                "rejected_false_positive": bool(candidate.get("rejected_false_positive", False)),
                "waypoint": candidate.get("waypoint"),
            }
            result.append(item)
        return result

    def _detected_objects(self, context: Any) -> List[Dict[str, Any]]:
        objects = self._get(context, "detected_objects", [])
        if isinstance(objects, dict) and "detections" in objects:
            objects = objects.get("detections") or []
        result = []
        for idx, detected in enumerate(self._iter_dicts(objects)):
            item = {
                "id": detected.get("id", idx),
                "label": detected.get("label"),
                "label_id": detected.get("label_id"),
                "confidence": detected.get("confidence"),
                "bbox": detected.get("bbox"),
                "center": detected.get("center"),
                "direction": detected.get("direction"),
                "distance": detected.get("distance"),
                "reachable": detected.get("reachable", True),
                "multi_view_confirmed": bool(detected.get("multi_view_confirmed", False)),
                "num_views": int(detected.get("num_views", 0) or 0),
                "is_target_candidate": bool(detected.get("is_target_candidate", False)),
                "is_landmark": bool(detected.get("is_landmark", True)),
                "grounded_in_current_observation": bool(
                    detected.get("grounded_in_current_observation", True)
                ),
                "source": detected.get("source"),
                "waypoint": detected.get("waypoint"),
            }
            result.append(item)
        return result

    def _rgb_observation(self, context: Any) -> Dict[str, Any]:
        observation = self._get(context, "rgb_observation", {})
        if not isinstance(observation, dict):
            return {"available": False}
        result = copy.deepcopy(observation)
        result.setdefault("available", bool(result.get("data_url") or result.get("image_url")))
        return result

    def _semantic_map_observation(self, context: Any) -> Dict[str, Any]:
        observation = self._get(context, "semantic_map_observation", {})
        if not isinstance(observation, dict):
            return {"available": False}
        result = copy.deepcopy(observation)
        result.setdefault("available", bool(result.get("data_url") or result.get("image_url")))
        return result

    def _gt_feedback(self, context: Any) -> Dict[str, Any]:
        feedback = self._get(context, "gt_feedback", {})
        if not isinstance(feedback, dict):
            return {"available": False}
        result = copy.deepcopy(feedback)
        result.setdefault("available", result.get("distance_to_goal") is not None)
        return result

    def _navigation_history(self, context: Any) -> Dict[str, Any]:
        history = self._get(context, "navigation_history", {})
        history = history if isinstance(history, dict) else {}
        return {
            "visited_frontier_ids": history.get("visited_frontier_ids", []),
            "recent_selected_skills": history.get("recent_selected_skills", []),
            "recent_failures": history.get("recent_failures", []),
            "stuck_count": int(history.get("stuck_count", self._get(context, "stuck_count", 0)) or 0),
            "collision_count": int(
                history.get("collision_count", self._get(context, "collision_count", 0)) or 0
            ),
            "steps_left": history.get("steps_left", self._get(context, "steps_left", None)),
            "best_known_point": history.get(
                "best_known_point",
                self._get(context, "best_known_point", {"available": False}),
            ),
        }

    def _default_observation_summary(self, state: Dict[str, Any]) -> str:
        target = state.get("target_category") or "unknown target"
        frontier_count = len(state.get("frontiers") or [])
        candidate_count = len(state.get("target_candidates") or [])
        landmark_count = len(state.get("detected_objects") or [])
        rgb_available = bool((state.get("rgb_observation") or {}).get("available"))
        semmap_available = bool((state.get("semantic_map_observation") or {}).get("available"))
        return (
            f"target={target}; frontiers={frontier_count}; "
            f"candidates={candidate_count}; landmarks={landmark_count}; "
            f"rgb_observation={rgb_available}; "
            f"semantic_map_observation={semmap_available}"
        )

    @staticmethod
    def _iter_dicts(values: Any) -> Iterable[Dict[str, Any]]:
        if values is None:
            return []
        if isinstance(values, dict):
            values = values.values()
        result = []
        for value in values:
            if isinstance(value, dict):
                result.append(value)
            elif hasattr(value, "to_dict"):
                result.append(value.to_dict())
            else:
                result.append(
                    {
                        key: getattr(value, key)
                        for key in dir(value)
                        if not key.startswith("_") and not callable(getattr(value, key))
                    }
                )
        return result

    @staticmethod
    def _get(context: Any, name: str, default: Any = None) -> Any:
        if isinstance(context, dict):
            return context.get(name, default)
        if hasattr(context, name):
            return getattr(context, name)
        getter = f"get_{name}"
        if hasattr(context, getter):
            try:
                return getattr(context, getter)()
            except TypeError:
                return default
        return default
