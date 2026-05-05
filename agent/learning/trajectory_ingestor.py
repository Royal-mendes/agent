from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from agent.schemas import SkillName
from agent.learning.trajectory_schema import TrajectoryEpisode, TrajectoryStep


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class TrajectoryIngestor:
    """Load ApexNav text logs, reflective episode JSON, and simple GT traces."""

    def load(self, path: str, source: Optional[str] = None) -> TrajectoryEpisode:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "decisions" in data or "episode_end" in data:
                return self.load_episode_logger_json(path, source=source or "self_reflection")
            return self.load_gt_trajectory(path, source=source or "gt")
        return self.load_apexnav_text_log(path, source=source or "baseline_apexnav")

    def load_many(self, paths: Iterable[str], source: Optional[str] = None) -> List[TrajectoryEpisode]:
        return [self.load(path, source=source) for path in paths]

    def load_episode_logger_json(self, path: str, source: str = "self_reflection") -> TrajectoryEpisode:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        episode_end = data.get("episode_end") or {}
        steps = []
        for item in data.get("decisions") or []:
            validator = item.get("validator_result") or {}
            agent_decision = item.get("agent_decision") or {}
            agent_skill = agent_decision.get("selected_skill")
            final_skill = validator.get("final_skill") or item.get("executed_skill")
            selected_skill = agent_skill or final_skill
            skill_args = agent_decision.get("skill_args") or validator.get("final_arguments") or {}
            steps.append(
                TrajectoryStep(
                    timestep=item.get("timestep"),
                    state_summary=item.get("state_summary") or {},
                    selected_skill=selected_skill,
                    skill_args=skill_args,
                    tool_name=self._tool_for_skill(selected_skill),
                    outcome=self._outcome_from_validator(validator),
                    validator_result=validator,
                    agent_decision=agent_decision,
                    failure_type=self._failure_from_validator(validator),
                    metadata={
                        "final_skill": final_skill,
                        "final_arguments": validator.get("final_arguments") or {},
                        "executed_skill": item.get("executed_skill"),
                        "retrieved_lessons": item.get("retrieved_lessons") or [],
                        "active_policy_patches": item.get("active_policy_patches") or [],
                        "apexnav_fallback_used": item.get("apexnav_fallback_used", False),
                    },
                )
            )

        return TrajectoryEpisode(
            source=source,
            episode_id=str(episode_end.get("episode_id") or data.get("episode_id") or os.path.basename(path)),
            scene_id=episode_end.get("scene_id"),
            split=episode_end.get("split") or "unknown",
            target_category=episode_end.get("target_category"),
            success=bool(episode_end.get("success", False)),
            result=episode_end.get("result"),
            stop_reason=episode_end.get("stop_reason"),
            failure_type=episode_end.get("failure_type"),
            steps=episode_end.get("steps") or len(steps),
            spl=episode_end.get("spl"),
            softspl=episode_end.get("softspl"),
            final_distance_to_goal=episode_end.get("final_distance_to_goal"),
            metrics={key: value for key, value in episode_end.items() if key not in {"episode_id", "scene_id", "split", "target_category"}},
            trajectory_steps=steps,
            raw_path=path,
        )

    def load_apexnav_text_log(self, path: str, source: str = "baseline_apexnav") -> TrajectoryEpisode:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = self._strip_ansi(f.read())
        target = self._last_match(r"Finding \[([^\]]+)\]", text) or self._last_match(r"Answer for ([^:]+):", text)
        result = self._last_match(r"^Result:\s*(.+)$", text, flags=re.M)
        success_pct = self._float_match(r"Average Success\s*\|\s*([0-9.]+)%", text)
        success = (result == "success") or (success_pct is not None and success_pct > 0.0) or "Reach the object successfully" in text
        steps = []
        for match in re.finditer(r"-+Step:\s*(\d+)-+\s*\nFinding \[([^\]]+)\]; Action:\s*([^;]+);", text):
            timestep = int(match.group(1))
            steps.append(
                TrajectoryStep(
                    timestep=timestep,
                    state_summary={
                        "timestep": timestep,
                        "target_category": match.group(2),
                        "navigation_history": {"recent_selected_skills": [SkillName.FALLBACK_APEXNAV.value]},
                    },
                    selected_skill=SkillName.FALLBACK_APEXNAV.value,
                    skill_args={},
                    tool_name="call_original_apexnav_policy",
                    action=match.group(3).strip(),
                    outcome="success" if success else "unknown",
                    metadata={"source_log": "habitat_evaluation"},
                )
            )
        if not steps:
            for match in re.finditer(r"\[ReflectiveAgentBridge\] selected_skill=([A-Z_]+).*?reason=([^\n]*)", text):
                steps.append(
                    TrajectoryStep(
                        timestep=len(steps),
                        selected_skill=match.group(1),
                        tool_name=self._tool_for_skill(match.group(1)),
                        outcome="success" if success else "unknown",
                        metadata={"reason": match.group(2).strip()},
                    )
                )
        spl_pct = self._float_match(r"Average SPL\s*\|\s*([0-9.]+)%", text)
        soft_pct = self._float_match(r"Average Soft SPL\s*\|\s*([0-9.]+)%", text)
        return TrajectoryEpisode(
            source=source,
            episode_id=os.path.splitext(os.path.basename(path))[0],
            split=self._infer_split(path, text),
            target_category=target,
            success=success,
            result=result or ("success" if success else "unknown"),
            stop_reason=result,
            steps=len(steps),
            spl=None if spl_pct is None else spl_pct / 100.0,
            softspl=None if soft_pct is None else soft_pct / 100.0,
            final_distance_to_goal=self._float_match(r"Average Distance to Goal\s*\|\s*([0-9.]+)", text),
            metrics={
                "success_pct": success_pct,
                "duration_log_path": path,
            },
            trajectory_steps=steps,
            raw_path=path,
        )

    def load_gt_trajectory(self, path: str, source: str = "gt") -> TrajectoryEpisode:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            data = {"waypoints": data}
        waypoints = data.get("waypoints") or data.get("trajectory") or data.get("path") or []
        steps = []
        for idx, waypoint in enumerate(waypoints):
            is_final = idx == len(waypoints) - 1
            skill = SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value if is_final else SkillName.GEOMETRIC_EXPLORE.value
            tool = "select_oracle_goal_waypoint" if is_final else "select_oracle_progress_waypoint"
            steps.append(
                TrajectoryStep(
                    timestep=idx,
                    selected_skill=skill,
                    skill_args={"gt_waypoint_index": idx},
                    tool_name=tool,
                    waypoint=tuple(waypoint[:2]) if isinstance(waypoint, list) and len(waypoint) >= 2 else None,
                    outcome="oracle",
                    metadata={"gt_waypoint": waypoint},
                )
            )
        return TrajectoryEpisode(
            source=source,
            episode_id=str(data.get("episode_id") or os.path.splitext(os.path.basename(path))[0]),
            scene_id=data.get("scene_id"),
            split=data.get("split") or "unknown",
            target_category=data.get("target_category"),
            success=True,
            result="oracle",
            stop_reason="oracle",
            steps=len(steps),
            metrics={key: value for key, value in data.items() if key not in {"waypoints", "trajectory", "path"}},
            trajectory_steps=steps,
            raw_path=path,
        )

    @staticmethod
    def _tool_for_skill(skill: Optional[str]) -> Optional[str]:
        return {
            SkillName.SEMANTIC_EXPLORE.value: "select_semantic_frontier",
            SkillName.GEOMETRIC_EXPLORE.value: "select_nearest_reachable_frontier",
            SkillName.VERIFY_TARGET.value: "verify_target_candidate",
            SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value: "navigate_to_confirmed_target",
            SkillName.RECOVER_FROM_STUCK.value: "recover_from_stuck",
            SkillName.FALLBACK_APEXNAV.value: "call_original_apexnav_policy",
        }.get(skill)

    @staticmethod
    def _outcome_from_validator(validator: Dict[str, Any]) -> str:
        if not validator:
            return "unknown"
        return "accepted" if validator.get("accepted", True) else "validator_corrected"

    @staticmethod
    def _failure_from_validator(validator: Dict[str, Any]) -> Optional[str]:
        if not validator or validator.get("accepted", True):
            return None
        reason = (validator.get("rejection_reason") or "").lower()
        if "no target candidate" in reason:
            return "unconfirmed_target_candidate"
        if "frontier" in reason:
            return "transient_unreachable_frontier"
        if "confidence" in reason or "multiview" in reason:
            return "false_positive_stop"
        return "validator_rejection"

    @staticmethod
    def _strip_ansi(text: str) -> str:
        return ANSI_RE.sub("", text)

    @staticmethod
    def _last_match(pattern: str, text: str, flags: int = 0) -> Optional[str]:
        matches = re.findall(pattern, text, flags=flags)
        return matches[-1].strip() if matches else None

    @staticmethod
    def _float_match(pattern: str, text: str) -> Optional[float]:
        matches = re.findall(pattern, text)
        if not matches:
            return None
        try:
            return float(matches[-1])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _infer_split(path: str, text: str) -> str:
        lowered = f"{path}\n{text}".lower()
        for split in ("train", "val", "test"):
            if f"/{split}/" in lowered or f" {split} " in lowered:
                return split
        return "unknown"
