from __future__ import annotations

import json
import os
from typing import Any, Iterable, List

from agent.learning.trajectory_schema import ToolUseLearningSample


class ToolCallDataset:
    """JSONL dataset for later SFT/preference training without doing training now."""

    def __init__(self, path: str = "data/tool_call_learning_samples.jsonl") -> None:
        self.path = path

    def append(self, sample: Any) -> None:
        item = sample if isinstance(sample, ToolUseLearningSample) else ToolUseLearningSample.from_dict(sample)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")

    def append_many(self, samples: Iterable[Any]) -> int:
        count = 0
        for sample in samples:
            self.append(sample)
            count += 1
        return count

    def load(self) -> List[ToolUseLearningSample]:
        if not os.path.exists(self.path):
            return []
        items = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(ToolUseLearningSample.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        return items

    def prune(self, max_items: int) -> None:
        if max_items <= 0:
            return
        items = self.load()
        if len(items) <= max_items:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for item in items[-max_items:]:
                f.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
