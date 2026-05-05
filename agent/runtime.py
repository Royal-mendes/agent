from __future__ import annotations

from typing import Any, Dict, Optional

from agent.execution.monitored_navigation_skill_executor import MonitoredNavigationSkillExecutor
from agent.logging.episode_logger import EpisodeLogger
from agent.memory.experience_memory import ExperienceMemory
from agent.memory.role_memory import RoleMemory
from agent.memory.task_memory import TaskMemory
from agent.memory.working_memory import WorkingMemory
from agent.reflection.policy_patch import PolicyPatchTable
from agent.reflection.reflection_engine import ReflectionEngine
from agent.reflective_navigation_agent import ReflectiveNavigationAgent
from agent.schemas import AgentConfig, SkillExecutionResult
from agent.skill.skills import ApexNavToolAdapter, build_default_skill_registry
from agent.state_summarizer import StateSummarizer
from agent.validator import DecisionValidator
from agent.vlm_provider import build_vlm_provider


class ReflectiveNavigationRuntime:
    """Optional orchestration layer for reflective ApexNav.

    Existing ApexNav code should only enter this runtime when
    ``enable_reflective_agent`` is true. When disabled, it delegates directly to
    the original ApexNav policy and does not retrieve memory, call VLM, validate,
    log decisions, or write reflection memory.
    """

    def __init__(
        self,
        cfg: Optional[AgentConfig] = None,
        experience_memory: Optional[ExperienceMemory] = None,
        vlm_provider: Optional[Any] = None,
        episode_logger: Optional[EpisodeLogger] = None,
        policy_patch_table: Optional[PolicyPatchTable] = None,
    ) -> None:
        self.cfg = cfg or AgentConfig()
        self.skill_registry = build_default_skill_registry()
        self.role_memory = RoleMemory.from_config(self.cfg)
        self.task_memory = TaskMemory()
        self.working_memory = WorkingMemory()
        self.state_summarizer = StateSummarizer(self.cfg)
        self.experience_memory = experience_memory or ExperienceMemory(
            memory_path=self.cfg.memory_path,
            max_items=self.cfg.max_reflection_memory_items,
            read_mode=self.cfg.memory_read_mode,
            write_mode=self.cfg.memory_write_mode,
        )
        self.policy_patch_table = policy_patch_table or PolicyPatchTable(self.cfg.policy_patch_path, self.cfg)
        self.episode_logger = episode_logger or EpisodeLogger(self.cfg.episode_log_root, self.cfg.run_id)
        if vlm_provider is None and self.cfg.vlm_provider != "mock":
            vlm_provider = build_vlm_provider(self.cfg)
        self.agent = ReflectiveNavigationAgent(self.cfg, self.skill_registry, vlm_provider)
        self.validator = DecisionValidator(self.cfg, self.skill_registry)
        self.executor = MonitoredNavigationSkillExecutor(self.cfg, self.skill_registry)
        self.reflection_engine = ReflectionEngine(self.cfg, self.experience_memory, self.policy_patch_table)

    def decide_and_execute(
        self,
        context: Any,
        state_overrides: Optional[Dict[str, Any]] = None,
    ) -> SkillExecutionResult:
        if not self.cfg.enable_reflective_agent:
            return ApexNavToolAdapter(context).call_original_apexnav_policy({})

        state = self.state_summarizer.summarize(context, overrides=state_overrides)
        lessons = []
        if self.cfg.enable_reflection_memory and self.cfg.memory_read_mode != "disabled":
            lessons = self.experience_memory.retrieve(
                state, top_k=self.cfg.max_retrieved_lessons
            )
        active_policy_patches = []
        if self.cfg.enable_policy_patch_table:
            active_policy_patches = self.policy_patch_table.get_active_patches(state.get("target_category"))
            patch_lessons = self.policy_patch_table.get_relevant_lessons(state, self.cfg.max_retrieved_lessons)
            lessons = (lessons + patch_lessons)[: self.cfg.max_retrieved_lessons]
        state["retrieved_lessons"] = lessons

        self.role_memory.update_from_config(self.cfg)
        self.task_memory.update_from_state(state)
        self.working_memory.update_before_decision(state)

        decision = self.agent.select_skill(
            state_summary=state,
            role_memory=self.role_memory,
            task_memory=self.task_memory,
            working_memory=self.working_memory,
            retrieved_lessons=lessons,
            active_policy_patches=active_policy_patches,
            validator_constraints=self.role_memory.hard_constraints,
        )
        self.working_memory.update_after_decision(decision)

        validated = self.validator.validate(
            decision=decision,
            state_summary=state,
            role_memory=self.role_memory,
            task_memory=self.task_memory,
            working_memory=self.working_memory,
            retrieved_lessons=lessons,
            active_policy_patches=active_policy_patches,
        )
        self.working_memory.update_after_validation(validated)

        result = self.executor.execute(
            skill_name=validated.final_skill,
            skill_args=validated.final_arguments,
            context=context,
        )
        self.working_memory.update_after_skill(result)
        self.task_memory.update_after_skill(result)
        result.raw_metadata.setdefault("agent_decision", decision.to_dict())
        result.raw_metadata.setdefault("validator_result", validated.to_dict())
        if self.cfg.enable_episode_logger:
            self.episode_logger.log_decision(
                episode_id=state.get("episode_id"),
                timestep=state.get("timestep"),
                state_summary=state,
                role_memory=self.role_memory,
                task_memory_snapshot=self.task_memory,
                working_memory_snapshot=self.working_memory,
                retrieved_lessons=lessons,
                active_policy_patches=active_policy_patches,
                agent_decision=decision,
                validator_result=validated,
                executed_skill=validated.final_skill,
                skill_result=result,
                apexnav_fallback_used=validated.fallback_used,
            )
        return result

    def finalize_episode(self, episode_summary: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cfg.enable_reflective_agent or not self.cfg.enable_episode_reflection:
            return {}
        reflection = self.reflection_engine.finalize_episode(episode_summary)
        if self.cfg.enable_episode_logger:
            self.episode_logger.log_episode_end(
                episode_id=episode_summary.get("episode_id"),
                episode_summary=episode_summary,
                reflection_result=reflection,
            )
        return reflection
