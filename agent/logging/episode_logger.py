from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if str(k) in {"data_url", "image_url"} and isinstance(v, str) and v.startswith("data:image/"):
                result[str(k)] = f"<omitted_image_data_url:{len(v)} chars>"
            else:
                result[str(k)] = _jsonable(v)
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class EpisodeLogger:
    """JSON episode logger for reflective-agent decisions and reflections."""

    def __init__(self, root: str = "logs/reflective_agent", run_id: Optional[str] = None) -> None:
        self.root = root
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.episodes_dir = os.path.join(self.root, self.run_id, "episodes")
        os.makedirs(self.episodes_dir, exist_ok=True)

    def episode_path(self, episode_id: Optional[str]) -> str:
        safe = self._safe_name(episode_id or "unknown_episode")
        return os.path.join(self.episodes_dir, f"{safe}.json")

    def log_decision(
        self,
        episode_id: Optional[str],
        timestep: Any,
        state_summary: Dict[str, Any],
        role_memory: Any,
        task_memory_snapshot: Any,
        working_memory_snapshot: Any,
        retrieved_lessons: Any,
        active_policy_patches: Any,
        agent_decision: Any,
        validator_result: Any,
        executed_skill: Optional[str] = None,
        skill_result: Any = None,
        apexnav_fallback_used: bool = False,
    ) -> str:
        data = self._load_episode(episode_id)
        data.setdefault("episode_id", episode_id)
        data.setdefault("decisions", [])
        data["decisions"].append(
            _jsonable(
                {
                    "logged_at": _now_iso(),
                    "timestep": timestep,
                    "state_summary": state_summary,
                    "role_memory": role_memory,
                    "task_memory_snapshot": task_memory_snapshot,
                    "working_memory_snapshot": working_memory_snapshot,
                    "retrieved_lessons": retrieved_lessons,
                    "active_policy_patches": active_policy_patches,
                    "agent_decision": agent_decision,
                    "validator_result": validator_result,
                    "executed_skill": executed_skill,
                    "skill_result": skill_result,
                    "apexnav_fallback_used": apexnav_fallback_used,
                }
            )
        )
        return self._write_episode(episode_id, data)

    def log_episode_end(
        self,
        episode_id: Optional[str],
        episode_summary: Dict[str, Any],
        reflection_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        data = self._load_episode(episode_id)
        data.setdefault("episode_id", episode_id)
        episode_summary = dict(episode_summary or {})
        episode_summary.setdefault("agent_diagnostics", self._compute_agent_diagnostics(data, episode_summary))
        data["episode_end"] = _jsonable(episode_summary)
        if reflection_result is not None:
            data["reflection"] = _jsonable(reflection_result)
        data["updated_at"] = _now_iso()
        return self._write_episode(episode_id, data)

    def read_episode_log(self, episode_id: Optional[str]) -> Dict[str, Any]:
        return self._load_episode(episode_id)

    def _load_episode(self, episode_id: Optional[str]) -> Dict[str, Any]:
        path = self.episode_path(episode_id)
        if not os.path.exists(path):
            return {"episode_id": episode_id, "created_at": _now_iso(), "decisions": []}
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"episode_id": episode_id, "created_at": _now_iso(), "decisions": []}

    def _write_episode(self, episode_id: Optional[str], data: Dict[str, Any]) -> str:
        path = self.episode_path(episode_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(data), f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
        return path

    @staticmethod
    def _compute_agent_diagnostics(data: Dict[str, Any], episode_summary: Dict[str, Any]) -> Dict[str, Any]:
        decisions = data.get("decisions") or []
        trigger_histogram: Dict[str, int] = {}
        skill_distribution: Dict[str, int] = {}
        stop_action_source_histogram: Dict[str, int] = {}
        ages = []
        switch_count = 0
        previous_skill = None
        recover_count = 0
        fallback_count = 0

        for decision in decisions:
            state = decision.get("state_summary") or {}
            diagnostics = state.get("bridge_diagnostics") or {}
            reasons = diagnostics.get("trigger_reasons") or []
            if isinstance(reasons, str):
                reasons = [item for item in reasons.split(",") if item]
            for reason in reasons:
                trigger_histogram[str(reason)] = trigger_histogram.get(str(reason), 0) + 1

            age = diagnostics.get("commitment_age_steps")
            if isinstance(age, (int, float)):
                ages.append(float(age))

            skill = decision.get("executed_skill")
            if not skill:
                validator = decision.get("validator_result") or {}
                skill = validator.get("final_skill")
            if skill:
                skill_distribution[str(skill)] = skill_distribution.get(str(skill), 0) + 1
                if previous_skill is not None and previous_skill != skill:
                    switch_count += 1
                previous_skill = skill
                if skill == "RECOVER_FROM_STUCK":
                    recover_count += 1
                if skill == "FALLBACK_APEXNAV":
                    fallback_count += 1

            stop_validator = state.get("stop_validator") or diagnostics.get("stop_validator") or {}
            source = stop_validator.get("stop_action_source")
            if source:
                stop_action_source_histogram[str(source)] = (
                    stop_action_source_histogram.get(str(source), 0) + 1
                )

        episode_stop_source = episode_summary.get("stop_action_source")
        if episode_stop_source:
            stop_action_source_histogram[str(episode_stop_source)] = (
                stop_action_source_histogram.get(str(episode_stop_source), 0) + 1
            )

        return {
            "vlm_call_count": len(decisions),
            "trigger_reason_histogram": trigger_histogram,
            "average_commitment_age_steps": sum(ages) / len(ages) if ages else 0.0,
            "committed_skill_switch_count": switch_count,
            "skill_distribution": skill_distribution,
            "recover_count": recover_count,
            "fallback_count": fallback_count,
            "stop_action_source_histogram": stop_action_source_histogram,
        }

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:160]
