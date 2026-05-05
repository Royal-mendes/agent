from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from agent.schemas import AgentConfig, SkillName


@dataclass
class RoleMemory:
    agent_role: str = "reflective_objectnav_skill_selector"
    allowed_skills: List[str] = field(
        default_factory=lambda: [skill.value for skill in SkillName]
    )
    hard_constraints: List[str] = field(
        default_factory=lambda: [
            "do_not_stop_on_single_view_low_confidence_target",
            "do_not_execute_unreachable_waypoint",
            "do_not_call_vlm_every_env_step",
            "always_validate_vlm_decision",
            "do_not_modify_test_memory_by_default",
        ]
    )
    thresholds: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: AgentConfig) -> "RoleMemory":
        role = cls()
        role.update_from_config(cfg)
        return role

    def update_from_config(self, cfg: AgentConfig) -> None:
        self.thresholds = {
            "target_verify_threshold": cfg.target_verify_threshold,
            "target_stop_threshold": cfg.target_stop_threshold,
            "semantic_peak_ratio_threshold": cfg.semantic_peak_ratio_threshold,
            "semantic_peak_std_threshold": cfg.semantic_peak_std_threshold,
            "stuck_threshold": cfg.stuck_threshold,
            "same_frontier_failure_threshold": cfg.same_frontier_failure_threshold,
        }
        if cfg.disable_recover_from_stuck and SkillName.RECOVER_FROM_STUCK.value in self.allowed_skills:
            self.allowed_skills.remove(SkillName.RECOVER_FROM_STUCK.value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
