from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from agent.schemas import AgentConfig, PolicyPatchProposal


class PolicyPatchTable:
    """Persistent table for proposed validator/prompt policy patches.

    Patches are conservative by default: proposals are recorded, but activation
    only happens when auto activation is enabled and support/confidence
    thresholds are met. Test split proposals are not persisted to avoid leakage.
    """

    def __init__(self, path: str = "data/policy_patches.json", cfg: Optional[AgentConfig] = None) -> None:
        self.path = path
        self.cfg = cfg or AgentConfig()
        self.patches: List[PolicyPatchProposal] = []

    def load_patches(self) -> List[PolicyPatchProposal]:
        self.patches = []
        if not os.path.exists(self.path):
            return self.patches
        with open(self.path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return self.patches

        raw_patches: List[Dict[str, Any]] = []
        if isinstance(data, dict) and "patches" in data:
            raw_patches.extend(data.get("patches") or [])
        if isinstance(data, dict) and "targets" in data:
            for target, group in (data.get("targets") or {}).items():
                for patch in (group or {}).get("patches", []):
                    patch = dict(patch)
                    patch.setdefault("target_scope", target)
                    raw_patches.append(patch)
        if isinstance(data, dict) and "target_category" in data:
            for patch in data.get("patches") or []:
                patch = dict(patch)
                patch.setdefault("target_scope", data.get("target_category"))
                raw_patches.append(patch)
        if isinstance(data, list):
            raw_patches.extend(data)
        self.patches = [PolicyPatchProposal.from_dict(item) for item in raw_patches if isinstance(item, dict)]
        return self.patches

    def save_patches(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for patch in self.patches:
            key = patch.target_scope or "global"
            grouped.setdefault(key, {"patches": []})["patches"].append(patch.to_dict())
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"targets": grouped}, f, indent=2, sort_keys=True)
            f.write("\n")

    def record_proposal(self, proposal: Any, split: str = "unknown") -> Optional[PolicyPatchProposal]:
        if split == "test":
            return None
        patch = proposal if isinstance(proposal, PolicyPatchProposal) else PolicyPatchProposal.from_dict(proposal)
        if not self.patches:
            self.load_patches()
        existing = self._find_equivalent(patch)
        if existing is None:
            patch.active = self._should_activate(patch)
            self.patches.append(patch)
            self.save_patches()
            return patch

        existing.support_count += max(1, patch.support_count)
        existing.confidence = max(existing.confidence, patch.confidence)
        existing.rationale = patch.rationale or existing.rationale
        existing.source_episode_id = patch.source_episode_id or existing.source_episode_id
        existing.active = existing.active or self._should_activate(existing)
        self.save_patches()
        return existing

    def get_active_patches(self, target_category: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.patches:
            self.load_patches()
        return [
            patch.to_dict()
            for patch in self.patches
            if patch.active and self._target_matches(patch.target_scope, target_category)
        ]

    def get_relevant_patches(self, state: Dict[str, Any], include_inactive: bool = True) -> List[Dict[str, Any]]:
        if not self.patches:
            self.load_patches()
        target = state.get("target_category")
        patches = []
        for patch in self.patches:
            if not include_inactive and not patch.active:
                continue
            if self._target_matches(patch.target_scope, target):
                patches.append(patch.to_dict())
        return patches

    def get_relevant_lessons(self, state: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        lessons = []
        for patch in self.get_relevant_patches(state, include_inactive=True):
            condition = patch.get("trigger_condition") or {}
            lessons.append(
                {
                    "target_category": patch.get("target_scope"),
                    "failure_type": self._failure_type_from_condition(condition),
                    "failure_class": "degrading"
                    if condition.get("selected_skill") == "NAVIGATE_TO_CONFIRMED_TARGET"
                    else "non_degrading",
                    "lesson": patch.get("rationale")
                    or f"When {condition}, prefer {patch.get('recommended_action')}.",
                    "suggested_policy_patch": patch,
                    "confidence": patch.get("confidence", 0.5),
                }
            )
        lessons.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
        return lessons[:top_k]

    def _find_equivalent(self, patch: PolicyPatchProposal) -> Optional[PolicyPatchProposal]:
        for existing in self.patches:
            if (
                existing.target_scope == patch.target_scope
                and existing.trigger_condition == patch.trigger_condition
                and existing.recommended_action == patch.recommended_action
            ):
                return existing
        return None

    def _should_activate(self, patch: PolicyPatchProposal) -> bool:
        return (
            bool(self.cfg.auto_activate_policy_patches)
            and patch.support_count >= self.cfg.min_policy_patch_support
            and patch.confidence >= self.cfg.policy_patch_confidence_threshold
        )

    @staticmethod
    def _target_matches(scope: Optional[str], target: Optional[str]) -> bool:
        return scope in {None, "", "global", target}

    @staticmethod
    def _failure_type_from_condition(condition: Any) -> Optional[str]:
        if isinstance(condition, dict):
            if condition.get("selected_skill") == "NAVIGATE_TO_CONFIRMED_TARGET":
                return "false_positive_stop"
            return condition.get("failure_type")
        return None
