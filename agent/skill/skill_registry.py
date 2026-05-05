from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

from agent.schemas import SkillExecutionResult, SkillSpec

SkillHandler = Callable[[dict, object], SkillExecutionResult]


class SkillRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, SkillSpec] = {}
        self._handlers: Dict[str, SkillHandler] = {}

    def register(
        self,
        spec: SkillSpec,
        handler: Optional[SkillHandler] = None,
        replace: bool = False,
    ) -> None:
        if not replace and spec.name in self._specs:
            raise ValueError(f"Skill already registered: {spec.name}")
        self._specs[spec.name] = spec
        if handler is not None:
            self._handlers[spec.name] = handler

    def has_skill(self, skill_name: str) -> bool:
        return skill_name in self._specs

    def get_spec(self, skill_name: str) -> SkillSpec:
        return self._specs[skill_name]

    def get_handler(self, skill_name: str) -> Optional[SkillHandler]:
        return self._handlers.get(skill_name)

    def available_skills(self) -> Iterable[str]:
        return self._specs.keys()

    def specs_as_dict(self) -> Dict[str, dict]:
        return {name: spec.to_dict() for name, spec in self._specs.items()}
