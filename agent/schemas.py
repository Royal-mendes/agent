"""Shared schemas for the reflective ApexNav wrapper.

The schemas are lightweight dataclasses so the agent layer can run in the
existing Python 3.9 environment without adding validation dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


class SkillName(str, Enum):
    SEMANTIC_EXPLORE = "SEMANTIC_EXPLORE"
    GEOMETRIC_EXPLORE = "GEOMETRIC_EXPLORE"
    VERIFY_TARGET = "VERIFY_TARGET"
    NAVIGATE_TO_CONFIRMED_TARGET = "NAVIGATE_TO_CONFIRMED_TARGET"
    RETURN_TO_BEST_KNOWN_POINT = "RETURN_TO_BEST_KNOWN_POINT"
    RECOVER_FROM_STUCK = "RECOVER_FROM_STUCK"
    FOLLOW_APEXNAV_PROPOSAL = "FOLLOW_APEXNAV_PROPOSAL"
    FALLBACK_APEXNAV = "FALLBACK_APEXNAV"


class SkillStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    RECOVERED = "recovered"
    UNAVAILABLE = "unavailable"


class FailureClass(str, Enum):
    NON_DEGRADING = "non_degrading"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise TypeError(f"Cannot serialize {type(value)!r} to dict")


@dataclass
class AgentConfig:
    enable_reflective_agent: bool = False
    enable_vlm_skill_selector: bool = True
    vlm_provider: str = "mock"
    vlm_model: Optional[str] = None
    vlm_model_env: str = "OPENAI_MODEL"
    vlm_api_key: Optional[str] = None
    vlm_base_url: Optional[str] = None
    vlm_base_url_env: str = "OPENAI_BASE_URL"
    vlm_api_key_env: str = "OPENAI_API_KEY"
    vlm_timeout_seconds: float = 30.0
    vlm_temperature: float = 0.0
    vlm_force_json_response_format: bool = False
    max_vlm_calls_per_episode: int = 100
    force_all_decisions_to_FALLBACK_APEXNAV: bool = False
    mock_follow_apexnav_by_default: bool = True
    enable_rgb_observation: bool = True
    rgb_observation_max_width: int = 320
    rgb_observation_jpeg_quality: int = 70
    enable_semantic_map_observation: bool = True
    semantic_map_max_width: int = 320
    semantic_map_crop_size_m: float = 12.0
    semantic_map_jpeg_quality: int = 75
    include_detected_objects_in_state: bool = True
    enable_yolo_landmark_detection: bool = True
    max_landmark_detections: int = 12

    min_commitment_steps_for_explore: int = 10
    max_commitment_steps_semantic: int = 80
    max_commitment_steps_geometric: int = 80
    max_commitment_steps_fallback: int = 80
    max_commitment_steps_recovery: int = 40
    structural_frontier_count_change_ratio: float = 0.3
    structural_frontier_stable_k: int = 3
    stable_target_event_k: int = 2

    enable_decision_validator: bool = True
    enable_monitored_skill_executor: bool = True

    enable_reflection_memory: bool = True
    enable_episode_reflection: bool = True
    enable_vlm_episode_reflection: bool = True
    vlm_episode_reflection_max_decisions: int = 12
    memory_path: str = "data/reflection_memory.jsonl"
    memory_read_mode: str = "enabled"
    memory_write_mode: str = "all"
    max_retrieved_lessons: int = 5
    max_reflection_memory_items: int = 10000

    agent_decision_interval: str = "waypoint"
    fallback_policy: str = "apexnav"

    target_verify_threshold: float = 0.65
    target_stop_threshold: float = 0.75
    require_multiview_before_stop: bool = True

    semantic_peak_ratio_threshold: float = 2.5
    semantic_peak_std_threshold: float = 0.15

    stuck_threshold: int = 3
    same_frontier_failure_threshold: int = 2
    low_information_gain_threshold: float = 0.01

    enable_policy_patch_table: bool = True
    policy_patch_path: str = "data/policy_patches.json"
    auto_activate_policy_patches: bool = False
    min_policy_patch_support: int = 3
    policy_patch_confidence_threshold: float = 0.7

    enable_episode_logger: bool = True
    episode_log_root: str = "logs/reflective_agent"
    run_id: Optional[str] = None
    project_root: Optional[str] = None
    python_executable: Optional[str] = None

    enable_learning_from_traces: bool = False
    enable_self_hindsight_learning: bool = False
    enable_baseline_teacher_learning: bool = False
    enable_gt_teacher_learning: bool = True
    enable_runtime_gt_progress_learning: bool = True
    enable_gt_trajectory_learning: bool = True
    enable_vlm_gt_trajectory_reflection: bool = True
    enable_online_gt_deviation_reflection: bool = True
    online_gt_deviation_reflection_debounce_steps: int = 5
    gt_progress_min_delta: float = 0.25
    gt_path_deviation_threshold: float = 1.5
    gt_path_deviation_growth_threshold: float = 0.5
    gt_path_max_points: int = 80
    gt_path_reflection_lookahead: int = 3
    max_gt_trajectory_reflections_per_episode: int = 3
    gt_learning_max_samples_per_episode: int = 8
    tool_call_dataset_path: str = "data/tool_call_learning_samples.jsonl"
    baseline_teacher_log_path: Optional[str] = None
    gt_trajectory_path: Optional[str] = None
    learning_write_mode: str = "all"
    learning_record_policy_patches: bool = True

    disable_failure_taxonomy: bool = False
    disable_recover_from_stuck: bool = False
    disable_verify_target: bool = True
    enable_stuck_recovery_override: bool = False

    @classmethod
    def from_mapping(cls, cfg: Optional[Dict[str, Any]]) -> "AgentConfig":
        if cfg is None:
            return cls()
        if not isinstance(cfg, dict):
            cfg = dict(cfg)
        known = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in cfg.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrontierCandidate:
    id: Optional[int] = None
    semantic_score: Optional[float] = None
    distance: Optional[float] = None
    reachable: Optional[bool] = None
    visited: bool = False
    blocked: bool = False
    low_value: bool = False
    room_hint: Optional[str] = None
    last_selected: bool = False
    failure_count: int = 0
    waypoint: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TargetCandidate:
    id: Optional[int] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    distance: Optional[float] = None
    reachable: Optional[bool] = None
    multi_view_confirmed: bool = False
    num_views: int = 0
    rejected_false_positive: bool = False
    waypoint: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NavigationActionPair:
    skill_name: str
    forward_action: Any = None
    expected_postcondition: Any = None
    failure_signals: List[str] = field(default_factory=list)
    recovery_action: Optional[str] = None
    memory_update_on_success: List[str] = field(default_factory=list)
    memory_update_on_failure: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillSpec:
    name: str
    purpose: str = ""
    inputs: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    forward_action: str = ""
    expected_postconditions: List[str] = field(default_factory=list)
    failure_signals: List[str] = field(default_factory=list)
    recovery_action: Optional[str] = None
    memory_update_on_success: List[str] = field(default_factory=list)
    memory_update_on_failure: List[str] = field(default_factory=list)
    validator_constraints: List[str] = field(default_factory=list)
    logging_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentDecision:
    status: str = "continue"
    selected_skill: str = SkillName.FALLBACK_APEXNAV.value
    skill_args: Dict[str, Any] = field(default_factory=dict)
    expected_postcondition: Optional[str] = None
    monitoring_plan: Dict[str, Any] = field(default_factory=dict)
    recovery_hint: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.0
    raw_response: Optional[str] = None

    @classmethod
    def fallback(cls, reason: str = "fallback") -> "AgentDecision":
        return cls(
            status="fallback",
            selected_skill=SkillName.FALLBACK_APEXNAV.value,
            reason=reason,
            confidence=0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidatorResult:
    final_skill: str
    final_arguments: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = True
    rejection_reason: Optional[str] = None
    original_skill: Optional[str] = None
    fallback_used: bool = False
    memory_rule_applied: bool = False
    policy_patch_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillExecutionResult:
    skill_name: str
    status: str = SkillStatus.SUCCESS.value
    start_timestep: Optional[int] = None
    end_timestep: Optional[int] = None
    selected_waypoint: Optional[Tuple[float, float]] = None
    selected_frontier_id: Optional[int] = None
    target_candidate_id: Optional[int] = None
    precondition_passed: bool = True
    postcondition_passed: Optional[bool] = None
    failure_type: Optional[str] = None
    failure_class: Optional[str] = None
    recovery_skill_suggested: Optional[str] = None
    memory_updates: List[Dict[str, Any]] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperienceMemoryItem:
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=now_iso)
    split: str = "unknown"
    scene_id: Optional[str] = None
    episode_id: Optional[str] = None
    target_category: Optional[str] = None
    scene_context: List[str] = field(default_factory=list)
    success: bool = False
    failure_type: Optional[str] = None
    failure_class: Optional[str] = None
    state_condition: Dict[str, Any] = field(default_factory=dict)
    bad_decision: Optional[str] = None
    better_decision: Optional[str] = None
    lesson: str = ""
    suggested_policy_patch: Optional[Dict[str, Any]] = None
    confidence: float = 0.5
    usage_count: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceMemoryItem":
        known = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyPatchProposal:
    patch_id: str = field(default_factory=lambda: str(uuid4()))
    target_scope: Optional[str] = None
    trigger_condition: Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = SkillName.FALLBACK_APEXNAV.value
    rationale: str = ""
    confidence: float = 0.5
    support_count: int = 1
    source_episode_id: Optional[str] = None
    active: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyPatchProposal":
        known = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeReflection:
    episode_id: Optional[str] = None
    scene_id: Optional[str] = None
    split: str = "unknown"
    target_category: Optional[str] = None
    success: bool = False
    failure_type: Optional[str] = None
    failure_class: Optional[str] = None
    summary: str = ""
    selected_skill_sequence: List[str] = field(default_factory=list)
    validator_rejection_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    memory_ids: List[str] = field(default_factory=list)
    policy_patch_proposals: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeReflection":
        known = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
