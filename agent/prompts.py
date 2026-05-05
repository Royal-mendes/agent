from __future__ import annotations

import copy
import json
from typing import Any, Dict, List


SYSTEM_PROMPT = """You are a high-level skill selection agent for object-goal navigation.
You do not directly control low-level actions.
You must select exactly one skill from the provided skill library.
Your goal is to reduce false positives, inefficient exploration, repeated failures, premature stopping, and unsafe retries.
Use retrieved lessons from previous episodes when relevant.
Return only valid JSON.
Do not invent unavailable skills.
Do not select NAVIGATE_TO_CONFIRMED_TARGET unless the target candidate is sufficiently reliable.
If uncertain, select VERIFY_TARGET or FALLBACK_APEXNAV.
Do not reveal chain-of-thought. Provide only a short reason."""


REQUIRED_JSON_FORMAT = {
    "status": "continue | done | fallback | recover",
    "selected_skill": "one exact skill name from available_skills",
    "skill_args": {
        "frontier_id": "use an id from state_summary.frontiers only, or null",
        "target_candidate_id": "use an id from state_summary.target_candidates only, or null",
    },
    "expected_postcondition": "brief public postcondition for the selected skill",
    "monitoring_plan": {
        "timeout_steps": 80,
        "failure_signals": ["unreachable_waypoint", "no_map_expansion"],
    },
    "recovery_hint": {
        "on_failure": "RECOVER_FROM_STUCK",
        "memory_update": "mark_frontier_low_value",
    },
    "reason": "short public explanation",
    "confidence": 0.74,
}


def build_user_prompt(
    state_summary: Dict[str, Any],
    role_memory: Dict[str, Any],
    task_memory: Dict[str, Any],
    working_memory: Dict[str, Any],
    available_skills: Dict[str, Any],
    retrieved_lessons: List[Dict[str, Any]],
    active_policy_patches: List[Dict[str, Any]],
    validator_constraints: List[str],
) -> str:
    prompt_state_summary = _prompt_safe_state_summary(state_summary)
    prompt_available_skills = _prompt_safe_available_skills(available_skills)
    payload = {
        "target_category": prompt_state_summary.get("target_category"),
        "state_summary": prompt_state_summary,
        "role_memory": role_memory,
        "task_memory": task_memory,
        "working_memory": working_memory,
        "available_skills": prompt_available_skills,
        "retrieved_lessons": retrieved_lessons,
        "active_policy_patches": active_policy_patches,
        "validator_constraints": validator_constraints,
        "id_selection_rules": [
            "Do not copy placeholder ids.",
            "If selecting an exploration skill, frontier_id must be one of the visible frontier ids in state_summary.frontiers.",
            "If selecting a target skill, target_candidate_id must be one of the visible target candidate ids in state_summary.target_candidates.",
            "Use null when no valid id is available; the validator will choose a safe fallback.",
        ],
        "visual_input_rules": [
            "Only the current RGB observation image is attached when available.",
            "No semantic score map image or semantic map metadata is attached in the current experiment setting.",
            "YOLO detected_objects are landmarks only, not confirmed targets.",
            "Return one high-level skill, no pixel coordinates or low-level actions.",
        ],
        "required_json_format": REQUIRED_JSON_FORMAT,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _prompt_safe_state_summary(state_summary: Dict[str, Any]) -> Dict[str, Any]:
    safe = copy.deepcopy(state_summary)
    safe["frontiers"] = _limit_frontiers_for_prompt(safe.get("frontiers") or [], max_items=8)
    safe["target_candidates"] = _limit_candidates_for_prompt(
        safe.get("target_candidates") or [], max_items=5
    )
    safe["detected_objects"] = _limit_candidates_for_prompt(
        safe.get("detected_objects") or [], max_items=10
    )
    # GT feedback is privileged supervision for episode-end learning. It must
    # stay in logs but never be shown to the online decision VLM.
    safe.pop("gt_feedback", None)
    rgb = safe.get("rgb_observation")
    if isinstance(rgb, dict):
        has_image = bool(rgb.get("data_url") or rgb.get("image_url"))
        safe["rgb_observation"] = {
            "available": bool(rgb.get("available", has_image)),
            "image_attached": has_image,
            "encoding": rgb.get("encoding"),
            "width": rgb.get("width"),
            "height": rgb.get("height"),
            "source": rgb.get("source"),
            "timestep": rgb.get("timestep"),
        }
    semmap = safe.get("semantic_map_observation")
    if isinstance(semmap, dict):
        has_image = bool(semmap.get("data_url") or semmap.get("image_url"))
        safe["semantic_map_observation"] = {
            "available": bool(semmap.get("available", has_image)),
            "image_attached": has_image,
            "encoding": semmap.get("encoding"),
            "width": semmap.get("width"),
            "height": semmap.get("height"),
            "source": semmap.get("source"),
            "timestep": semmap.get("timestep"),
            "legend": semmap.get("legend"),
            "crop_size_m": semmap.get("crop_size_m"),
        }
    return safe


def _prompt_safe_available_skills(available_skills: Dict[str, Any]) -> Dict[str, Any]:
    compact = {}
    for name, spec in (available_skills or {}).items():
        if not isinstance(spec, dict):
            compact[name] = spec
            continue
        compact[name] = {
            "name": spec.get("name", name),
            "purpose": spec.get("purpose"),
            "preconditions": (spec.get("preconditions") or [])[:4],
            "expected_postconditions": (spec.get("expected_postconditions") or [])[:3],
            "failure_signals": (spec.get("failure_signals") or [])[:4],
            "recovery_action": spec.get("recovery_action"),
        }
    return compact


def _limit_frontiers_for_prompt(frontiers: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    def sort_key(item: Dict[str, Any]):
        semantic = item.get("semantic_score") or 0.0
        distance = item.get("distance") if item.get("distance") is not None else 1e9
        reachable_bonus = 0 if item.get("reachable", True) else 1
        return (reachable_bonus, -float(semantic), float(distance))

    compact = []
    for item in sorted(frontiers, key=sort_key)[:max_items]:
        compact.append({
            "id": item.get("id"),
            "semantic_score": item.get("semantic_score"),
            "distance": item.get("distance"),
            "reachable": item.get("reachable", True),
            "visited": item.get("visited", False),
            "blocked": item.get("blocked", False),
            "low_value": item.get("low_value", False),
            "failure_count": item.get("failure_count", 0),
            "last_selected": item.get("last_selected", False),
            "room_hint": item.get("room_hint"),
        })
    return compact


def _limit_candidates_for_prompt(candidates: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    def sort_key(item: Dict[str, Any]):
        confidence = item.get("confidence") or item.get("score") or 0.0
        distance = item.get("distance") if item.get("distance") is not None else 1e9
        return (-float(confidence), float(distance))

    compact = []
    for item in sorted(candidates, key=sort_key)[:max_items]:
        compact.append({
            "id": item.get("id"),
            "label": item.get("label"),
            "confidence": item.get("confidence"),
            "distance": item.get("distance"),
            "reachable": item.get("reachable", True),
            "multi_view_confirmed": item.get("multi_view_confirmed"),
            "num_views": item.get("num_views"),
            "rejected_false_positive": item.get("rejected_false_positive", False),
            "direction": item.get("direction"),
            "source": item.get("source"),
            "is_landmark": item.get("is_landmark"),
        })
    return compact
