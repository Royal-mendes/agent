from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from agent.memory.experience_memory import ExperienceMemory
from agent.reflection.policy_patch import PolicyPatchTable
from agent.schemas import AgentConfig, ExperienceMemoryItem, PolicyPatchProposal
from agent.learning.trajectory_schema import ToolUseLearningSample


class LessonBuilder:
    """Convert tool-use learning samples into persistent memory artifacts."""

    def __init__(self, cfg: Optional[AgentConfig] = None) -> None:
        self.cfg = cfg or AgentConfig()

    def build_memory_item(self, sample: ToolUseLearningSample) -> ExperienceMemoryItem:
        teacher = sample.teacher_action or {}
        student = sample.student_action or {}
        positive_outcomes = {
            "baseline_teacher_better",
            "gt_oracle_progress",
            "gt_progress_skill_helped",
            "validator_corrected_student",
        }
        return ExperienceMemoryItem(
            split=sample.split,
            scene_id=sample.scene_id,
            episode_id=sample.episode_id,
            target_category=sample.target_category,
            success=sample.outcome in positive_outcomes,
            failure_type=sample.failure_type,
            failure_class=sample.failure_class,
            state_condition=dict(sample.state_condition),
            bad_decision=self._decision_text(student),
            better_decision=self._decision_text(teacher),
            lesson=sample.lesson,
            suggested_policy_patch=sample.policy_patch_proposal,
            confidence=sample.confidence,
        )

    def build_policy_patch(self, sample: ToolUseLearningSample) -> Optional[PolicyPatchProposal]:
        patch = sample.policy_patch_proposal
        if not patch:
            return None
        patch = dict(patch)
        patch.setdefault("target_scope", sample.target_category)
        patch.setdefault("confidence", sample.confidence)
        patch.setdefault("source_episode_id", sample.episode_id)
        return PolicyPatchProposal.from_dict(patch)

    def persist(
        self,
        samples: Iterable[ToolUseLearningSample],
        experience_memory: Optional[ExperienceMemory] = None,
        policy_patch_table: Optional[PolicyPatchTable] = None,
        record_policy_patches: bool = True,
    ) -> Dict[str, Any]:
        memory = experience_memory or ExperienceMemory(
            memory_path=self.cfg.memory_path,
            max_items=self.cfg.max_reflection_memory_items,
            read_mode=self.cfg.memory_read_mode,
            write_mode=self.cfg.memory_write_mode,
        )
        patch_table = policy_patch_table or PolicyPatchTable(self.cfg.policy_patch_path, self.cfg)
        memory_items: List[Dict[str, Any]] = []
        patches: List[Dict[str, Any]] = []
        memory_written = 0
        patch_written = 0
        for sample in samples:
            item = self.build_memory_item(sample)
            memory_items.append(item.to_dict())
            if memory.append_memory(item, split=sample.split):
                memory_written += 1
            patch = self.build_policy_patch(sample)
            if patch is not None and record_policy_patches:
                patches.append(patch.to_dict())
                if patch_table.record_proposal(patch, split=sample.split) is not None:
                    patch_written += 1
        return {
            "memory_items": memory_items,
            "policy_patches": patches,
            "memory_written": memory_written,
            "policy_patches_recorded": patch_written,
        }

    @staticmethod
    def _decision_text(action: Dict[str, Any]) -> Optional[str]:
        if not action:
            return None
        skill = action.get("skill") or action.get("skill_name")
        tool = action.get("tool") or action.get("tool_name")
        args = action.get("arguments") or action.get("args") or {}
        if not skill and not tool:
            return None
        return f"skill={skill}; tool={tool}; args={args}"
