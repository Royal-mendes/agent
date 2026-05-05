from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from agent.schemas import ExperienceMemoryItem


class ExperienceMemory:
    """JSONL-backed long-term memory with simple lexical retrieval."""

    def __init__(
        self,
        memory_path: str = "data/reflection_memory.jsonl",
        max_items: int = 10000,
        read_mode: str = "enabled",
        write_mode: str = "train_only",
    ) -> None:
        self.memory_path = memory_path
        self.max_items = max_items
        self.read_mode = read_mode
        self.write_mode = write_mode
        self.items: List[ExperienceMemoryItem] = []

    def load_memory(self) -> List[ExperienceMemoryItem]:
        self.items = []
        if self.read_mode == "disabled" or not os.path.exists(self.memory_path):
            return self.items
        with open(self.memory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.items.append(ExperienceMemoryItem.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        return self.items

    def append_memory(self, item: Any, split: Optional[str] = None) -> bool:
        memory_item = item if isinstance(item, ExperienceMemoryItem) else ExperienceMemoryItem.from_dict(item)
        split_name = split or memory_item.split or "unknown"
        if not self._can_write(split_name):
            return False
        os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(memory_item.to_dict(), sort_keys=True) + "\n")
        self.items.append(memory_item)
        self.prune_memory(self.max_items)
        return True

    def retrieve(self, query_state: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.read_mode == "disabled":
            return []
        if not self.items:
            self.load_memory()
        scored = []
        for item in self.items:
            score = self._score_item(item, query_state)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item.to_dict() for _, item in scored[:top_k]]

    def prune_memory(self, max_items: Optional[int] = None) -> None:
        max_items = max_items or self.max_items
        if max_items <= 0 or len(self.items) <= max_items:
            return
        self.items = self.items[-max_items:]
        os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")

    def increment_usage(self, memory_id: str) -> bool:
        if not self.items:
            self.load_memory()
        found = False
        for item in self.items:
            if item.memory_id == memory_id:
                item.usage_count += 1
                found = True
                break
        if found:
            self._rewrite()
        return found

    def _rewrite(self) -> None:
        os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")

    def _can_write(self, split: str) -> bool:
        if self.write_mode == "disabled":
            return False
        if self.write_mode == "all":
            return True
        if self.write_mode == "train_only":
            return split == "train"
        if self.write_mode == "val_only":
            return split == "val"
        if self.write_mode == "test_only":
            return split == "test"
        if self.write_mode == "eval_only":
            return split in {"val", "test"}
        return False

    def _score_item(self, item: ExperienceMemoryItem, state: Dict[str, Any]) -> float:
        score = 0.0
        target = state.get("target_category")
        if target and item.target_category == target:
            score += 3.0

        state_failure = state.get("failure_type") or self._last_failure(state)
        if state_failure and item.failure_type == state_failure:
            score += 2.0

        selected_skill = self._last_skill(state)
        item_skills = item.state_condition.get("selected_skill_sequence") or []
        if selected_skill and selected_skill in item_skills:
            score += 2.0

        context_terms = set(self._context_terms(state))
        if context_terms and context_terms.intersection(set(item.scene_context)):
            score += 1.0

        if item.confidence >= 0.7:
            score += 1.0
        if self._is_recent(item.created_at):
            score += 1.0
        if item.confidence < 0.35:
            score -= 1.0
        return score

    @staticmethod
    def _last_failure(state: Dict[str, Any]) -> Optional[str]:
        failures = (state.get("navigation_history") or {}).get("recent_failures") or []
        return failures[-1] if failures else None

    @staticmethod
    def _last_skill(state: Dict[str, Any]) -> Optional[str]:
        skills = (state.get("navigation_history") or {}).get("recent_selected_skills") or []
        return skills[-1] if skills else None

    @staticmethod
    def _context_terms(state: Dict[str, Any]) -> Iterable[str]:
        summary = state.get("current_observation_summary") or ""
        if isinstance(summary, str):
            for token in summary.replace(",", " ").replace(".", " ").split():
                token = token.strip().lower()
                if len(token) > 2:
                    yield token

    @staticmethod
    def _is_recent(created_at: str) -> bool:
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - created).days
            return age_days <= 30
        except (ValueError, TypeError):
            return False
