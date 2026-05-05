from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from agent.logging.episode_logger import EpisodeLogger
from agent.memory.experience_memory import ExperienceMemory
from agent.memory.role_memory import RoleMemory
from agent.memory.task_memory import TaskMemory
from agent.memory.working_memory import WorkingMemory
from agent.reflection.policy_patch import PolicyPatchTable
from agent.reflective_navigation_agent import ReflectiveNavigationAgent
from agent.schemas import AgentConfig, AgentDecision, SkillName
from agent.skill.skills import build_default_skill_registry
from agent.state_summarizer import StateSummarizer
from agent.validator import DecisionValidator
from agent.vlm_provider import build_vlm_provider


def decide(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = AgentConfig.from_mapping(payload.get("config") or {})
    if not cfg.enable_reflective_agent:
        return _fallback("reflective agent disabled")

    state = payload.get("state_summary") or payload.get("state") or {}
    state = StateSummarizer(cfg).summarize(state)
    registry = build_default_skill_registry()
    role_memory = RoleMemory.from_config(cfg)
    task_memory = TaskMemory()
    task_memory.update_from_state(state)
    working_memory = WorkingMemory()

    lessons: List[Dict[str, Any]] = payload.get("retrieved_lessons") or []
    if cfg.enable_reflection_memory and cfg.memory_read_mode != "disabled":
        memory = ExperienceMemory(
            memory_path=cfg.memory_path,
            max_items=cfg.max_reflection_memory_items,
            read_mode=cfg.memory_read_mode,
            write_mode=cfg.memory_write_mode,
        )
        lessons = memory.retrieve(state, top_k=cfg.max_retrieved_lessons)

    active_policy_patches = payload.get("active_policy_patches") or []
    if cfg.enable_policy_patch_table:
        patch_table = PolicyPatchTable(cfg.policy_patch_path, cfg)
        active_policy_patches = active_policy_patches or patch_table.get_active_patches(state.get("target_category"))
        patch_lessons = patch_table.get_relevant_lessons(state, top_k=cfg.max_retrieved_lessons)
        if patch_lessons:
            lessons = (lessons + patch_lessons)[: cfg.max_retrieved_lessons]
    state["retrieved_lessons"] = lessons

    provider = None
    if cfg.vlm_provider != "mock":
        provider = build_vlm_provider(cfg)

    agent = ReflectiveNavigationAgent(cfg, registry, provider)
    decision = agent.select_skill(
        state_summary=state,
        role_memory=role_memory,
        task_memory=task_memory,
        working_memory=working_memory,
        retrieved_lessons=lessons,
        active_policy_patches=active_policy_patches,
        validator_constraints=role_memory.hard_constraints,
    )
    validator = DecisionValidator(cfg, registry)
    validated = validator.validate(
        decision=decision,
        state_summary=state,
        role_memory=role_memory,
        task_memory=task_memory,
        working_memory=working_memory,
        retrieved_lessons=lessons,
        active_policy_patches=active_policy_patches,
    )
    result = {
        "selected_skill": validated.final_skill,
        "skill_args": validated.final_arguments,
        "accepted": validated.accepted,
        "fallback_used": validated.fallback_used,
        "rejection_reason": validated.rejection_reason,
        "memory_rule_applied": validated.memory_rule_applied,
        "policy_patch_applied": validated.policy_patch_applied,
        "agent_decision": decision.to_dict(),
        "validator_result": validated.to_dict(),
        "retrieved_lessons": lessons,
        "active_policy_patches": active_policy_patches,
    }
    _log_bridge_decision(
        cfg,
        state,
        role_memory,
        task_memory,
        working_memory,
        lessons,
        active_policy_patches,
        decision,
        validated,
        result,
    )
    return result


def _log_bridge_decision(
    cfg: AgentConfig,
    state: Dict[str, Any],
    role_memory: RoleMemory,
    task_memory: TaskMemory,
    working_memory: WorkingMemory,
    lessons: List[Dict[str, Any]],
    active_policy_patches: List[Dict[str, Any]],
    decision: AgentDecision,
    validated: Any,
    result: Dict[str, Any],
) -> None:
    if not cfg.enable_episode_logger:
        return
    try:
        logger = EpisodeLogger(root=cfg.episode_log_root, run_id=cfg.run_id)
        logger.log_decision(
            episode_id=state.get("episode_id"),
            timestep=state.get("timestep"),
            state_summary=state,
            role_memory=role_memory.to_dict() if hasattr(role_memory, "to_dict") else role_memory,
            task_memory_snapshot=task_memory.to_dict() if hasattr(task_memory, "to_dict") else task_memory,
            working_memory_snapshot=working_memory.to_dict() if hasattr(working_memory, "to_dict") else working_memory,
            retrieved_lessons=lessons,
            active_policy_patches=active_policy_patches,
            agent_decision=decision.to_dict(),
            validator_result=validated.to_dict(),
            executed_skill=result.get("selected_skill"),
            skill_result=None,
            apexnav_fallback_used=result.get("fallback_used", False),
        )
    except Exception:
        return


def _fallback(reason: str) -> Dict[str, Any]:
    decision = AgentDecision.fallback(reason)
    return {
        "selected_skill": SkillName.FALLBACK_APEXNAV.value,
        "skill_args": {},
        "accepted": False,
        "fallback_used": True,
        "rejection_reason": reason,
        "memory_rule_applied": False,
        "policy_patch_applied": False,
        "agent_decision": decision.to_dict(),
        "validator_result": {
            "final_skill": SkillName.FALLBACK_APEXNAV.value,
            "final_arguments": {},
            "accepted": False,
            "rejection_reason": reason,
            "original_skill": SkillName.FALLBACK_APEXNAV.value,
            "fallback_used": True,
            "memory_rule_applied": False,
            "policy_patch_applied": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reflective ApexNav bridge CLI")
    parser.add_argument("--input", help="JSON input path. Defaults to stdin.")
    parser.add_argument("--output", help="JSON output path. Defaults to stdout.")
    args = parser.parse_args()

    try:
        if args.input:
            with open(args.input, "r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = json.load(sys.stdin)
        result = decide(payload)
    except Exception as exc:  # pragma: no cover - bridge must never crash C++ caller
        result = _fallback(f"bridge_error: {type(exc).__name__}: {exc}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f)
            f.write("\n")
    else:
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
