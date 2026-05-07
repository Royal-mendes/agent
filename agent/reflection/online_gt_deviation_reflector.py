from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from agent.memory.experience_memory import ExperienceMemory
from agent.reflection.failure_taxonomy import classify_failure
from agent.schemas import AgentConfig, ExperienceMemoryItem, SkillName


ONLINE_GT_DEVIATION_SYSTEM_PROMPT = """You are an online teacher-guided reflection agent for object-goal navigation.
GT deviation is only the trigger telling you that the previous high-level skill may have been a bad learning example.
Your job is not to write "do not deviate from GT". Your job is to infer why the teacher/GT route was more reasonable from the information that would also be available at test time: RGB image, YOLO landmarks, target category, target candidates, semantic/frontier state, navigation history, and skill outcome.
Compare the previous skill choice with the local intent of the GT route. Explain the test-time cues that support the teacher route, such as target candidate evidence, target-relevant room cues, open passages, frontier structure, semantic peak, or progress toward a more informative area.
Convert that analysis into a reusable skill-arbitration lesson that can be applied without GT at test time.
Do not infer the target location from generic non-target landmarks unless you explicitly mark the evidence as weak. Do not invent object associations. Do not recommend NAVIGATE_TO_CONFIRMED_TARGET unless the target candidate is reliable under validator constraints.
The final lesson must not mention GT, shortest path, oracle, deviation distance, or coordinates. It should say: when these visual/state cues appear, avoid this skill and prefer that skill.
Return only valid JSON. Do not reveal hidden chain-of-thought; provide concise public reasons."""


class OnlineGTDeviationReflector:
    """Detect online GT-path deviation and immediately write a reusable lesson.

    The bridge CLI is invoked as a fresh process for each high-level decision, so
    this class uses the episode JSON log plus a small sidecar state file to avoid
    reflecting on the same previous decision repeatedly.
    """

    def __init__(
        self,
        cfg: Optional[AgentConfig] = None,
        vlm_provider: Optional[Any] = None,
        memory: Optional[ExperienceMemory] = None,
    ) -> None:
        self.cfg = cfg or AgentConfig()
        self.vlm_provider = vlm_provider
        self.memory = memory or ExperienceMemory(
            memory_path=self.cfg.memory_path,
            max_items=self.cfg.max_reflection_memory_items,
            read_mode=self.cfg.memory_read_mode,
            write_mode=self.cfg.memory_write_mode,
        )

    def maybe_reflect(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cfg.enable_online_gt_deviation_reflection:
            return {"triggered": False, "reason": "disabled"}
        current_gt = self._gt_context(state)
        if not current_gt.get("gt_path_available"):
            return {"triggered": False, "reason": "gt_path_unavailable"}

        episode_id = state.get("episode_id")
        episode_log = self._episode_log_path(episode_id)
        if not episode_log or not os.path.exists(episode_log):
            return {"triggered": False, "reason": "episode_log_missing"}
        try:
            with open(episode_log, "r", encoding="utf-8") as f:
                episode_data = json.load(f)
        except Exception:
            return {"triggered": False, "reason": "episode_log_unreadable"}

        previous = self._previous_decision(episode_data)
        if previous is None:
            return {"triggered": False, "reason": "previous_decision_missing"}
        previous_timestep = previous.get("timestep")
        sidecar = self._load_sidecar(episode_id)
        if str(previous_timestep) in set(str(x) for x in sidecar.get("reflected_timesteps", [])):
            return {"triggered": False, "reason": "already_reflected", "previous_timestep": previous_timestep}
        if self._debounced(previous_timestep, sidecar):
            return {"triggered": False, "reason": "debounced", "previous_timestep": previous_timestep}

        previous_gt = self._gt_context(previous.get("state_summary") or {})
        if not previous_gt.get("gt_path_available"):
            return {"triggered": False, "reason": "previous_gt_path_unavailable"}

        before_dev = self._float(previous_gt.get("distance_to_gt_path"))
        after_dev = self._float(current_gt.get("distance_to_gt_path"))
        if before_dev is None or after_dev is None:
            return {"triggered": False, "reason": "missing_deviation_distance"}
        growth = after_dev - before_dev
        crossed = before_dev < self.cfg.gt_path_deviation_threshold <= after_dev
        worsened = after_dev >= self.cfg.gt_path_deviation_threshold and growth >= self.cfg.gt_path_deviation_growth_threshold
        if not (crossed or worsened):
            return {
                "triggered": False,
                "reason": "below_deviation_trigger",
                "distance_to_gt_path_before": before_dev,
                "distance_to_gt_path_after": after_dev,
                "gt_deviation_growth": growth,
            }

        reflection = self._reflect_with_vlm(state, previous, previous_gt, current_gt, before_dev, after_dev, growth)
        if self.cfg.enable_vlm_gt_trajectory_reflection and reflection.get("source") != "vlm":
            self._record_skipped_vlm_reflection(
                episode_id=episode_id,
                previous_timestep=previous_timestep,
                before_dev=before_dev,
                after_dev=after_dev,
                growth=growth,
                reflection=reflection,
            )
            return {
                "triggered": True,
                "source": reflection.get("source", "vlm_error"),
                "previous_timestep": previous_timestep,
                "previous_skill": self._previous_skill(previous),
                "distance_to_gt_path_before": before_dev,
                "distance_to_gt_path_after": after_dev,
                "gt_deviation_growth": growth,
                "memory_written": False,
                "memory_id": None,
                "lesson": None,
                "better_skill": None,
                "reflection": reflection,
            }
        memory_item = self._memory_item(state, previous, previous_gt, current_gt, before_dev, after_dev, growth, reflection)
        memory_written = self.memory.append_memory(memory_item, split=state.get("split") or "unknown")

        sidecar.setdefault("reflected_timesteps", []).append(previous_timestep)
        sidecar["last_reflection_timestep"] = previous_timestep
        sidecar["last_reflection"] = {
            "distance_to_gt_path_before": before_dev,
            "distance_to_gt_path_after": after_dev,
            "gt_deviation_growth": growth,
            "memory_id": memory_item.memory_id,
        }
        self._write_sidecar(episode_id, sidecar)
        return {
            "triggered": True,
            "source": reflection.get("source", "heuristic"),
            "previous_timestep": previous_timestep,
            "previous_skill": self._previous_skill(previous),
            "distance_to_gt_path_before": before_dev,
            "distance_to_gt_path_after": after_dev,
            "gt_deviation_growth": growth,
            "memory_written": memory_written,
            "memory_id": memory_item.memory_id,
            "lesson": memory_item.lesson,
            "better_skill": reflection.get("better_skill") or self._heuristic_better_skill(previous),
            "reflection": reflection,
        }

    def _reflect_with_vlm(
        self,
        state: Dict[str, Any],
        previous: Dict[str, Any],
        previous_gt: Dict[str, Any],
        current_gt: Dict[str, Any],
        before_dev: float,
        after_dev: float,
        growth: float,
    ) -> Dict[str, Any]:
        if self.vlm_provider is None:
            return {"source": "vlm_unavailable", "error": "vlm provider unavailable"}
        if not self.cfg.enable_vlm_gt_trajectory_reflection:
            return {"source": "disabled", "error": "VLM GT trajectory reflection disabled"}
        visual_snapshot = self._load_visual_snapshot(previous)
        image_data_urls = self._image_data_urls_for_reflection(visual_snapshot)
        payload = {
            "task": "Reflect on the previous high-level navigation decision because it diverged from the teacher/GT route.",
            "reflection_policy": {
                "primary_evidence": [
                    "previous selected skill and validator result",
                    "teacher/GT local route direction and progress change",
                    "frontier/target candidate state at the previous decision",
                    "RGB/YOLO visual evidence at the previous decision",
                ],
                "visual_evidence_role": (
                    "RGB/YOLO evidence must be interpreted by you, not by a fixed lookup table. "
                    "Judge how the visible objects, room layout, and target candidates relate to "
                    "the target category. Strong evidence means a reliable target-category candidate. "
                    "Medium evidence means an uncertain target candidate or a well-justified scene/room cue. "
                    "Weak evidence means generic landmarks whose relation to the target is speculative."
                ),
                "visual_reasoning_requirements": [
                    "explicitly state why each visual cue is related or unrelated to the target",
                    "separate target candidates, room/scene cues, and generic landmarks",
                    "mark uncertainty instead of forcing a relation",
                    "explain what visual/state cue makes the teacher route more reasonable",
                    "prefer skill-level lessons over object co-occurrence shortcuts",
                    "write the final lesson so it can be used at test time without GT",
                ],
                "forbidden_patterns": [
                    "do not write 'do not deviate from GT' as the lesson",
                    "do not mention GT, oracle, shortest path, deviation distance, or coordinates in the final lesson",
                    "do not claim that generic landmarks imply the target location",
                    "do not recommend stopping or target navigation without a reliable target candidate",
                    "do not write lessons like 'for target X, go toward unrelated object Y'",
                    "do not convert one GT-deviation event into a global object-association rule",
                    "do not rely on memorized category pairs unless the current observation supports the relation",
                ],
            },
            "previous_decision_to_analyze": self._compact_decision(previous),
            "visual_evidence_at_previous_decision": self._visual_evidence(previous, visual_snapshot),
            "current_state_summary": self._prompt_safe_state(state),
            "gt_trajectory_context_before_previous_decision": previous_gt,
            "gt_trajectory_context_now": current_gt,
            "deviation_metrics": {
                "distance_to_gt_path_before": before_dev,
                "distance_to_gt_path_now": after_dev,
                "gt_deviation_growth": growth,
                "deviation_threshold": self.cfg.gt_path_deviation_threshold,
                "growth_threshold": self.cfg.gt_path_deviation_growth_threshold,
            },
            "allowed_skills": [
                item.value for item in SkillName
                if not (self.cfg.disable_verify_target and item == SkillName.VERIFY_TARGET)
            ],
            "required_json_format": {
                "teacher_path_interpretation": "why the teacher/GT route appears more reasonable, citing visual/state cues; this field may mention GT",
                "decision_error": "what the previous skill optimized incorrectly or ignored",
                "test_time_condition": {
                    "target_category": "target category",
                    "visual_cues": ["target candidate, room cue, passage/opening, or generic landmark cues"],
                    "navigation_state": ["frontier/semantic/target/stuck/progress conditions"],
                    "skill_context": "previous selected skill and why it was risky"
                },
                "bad_decision": "previous harmful skill decision",
                "better_skill": "one allowed high-level skill",
                "better_decision": "what the agent should have done at that decision",
                "visual_evidence": {
                    "target_candidate_evidence": "list target-category candidates with confidence/reachability/views, or []",
                    "room_cues": "list scene/room cues you judge relevant to the target, or []",
                    "generic_landmarks": "list visible landmarks you judge not enough for target inference, or []",
                    "reliability": "strong | medium | weak | none",
                    "target_relation": "target_candidate | room_cue | generic_landmark | none",
                    "reason": "short public explanation of how visual evidence relates to the target, previous skill, GT deviation, and better skill",
                    "uncertainty": "short note on what remains uncertain"
                },
                "visual_evidence_used": "short note consistent with visual_evidence, or null",
                "skill_preference_rule": "test-time rule: when the condition appears, prefer the better skill and why",
                "avoid_rule": "test-time rule: when the condition appears, avoid the previous skill and why",
                "lesson": "final reusable test-time lesson. Do not mention GT/oracle/deviation/coordinates.",
                "confidence": 0.0,
            },
        }
        first_error = None
        if image_data_urls:
            result = self._call_vlm_reflector(
                payload=payload,
                image_data_urls=image_data_urls,
                previous=previous,
                visual_snapshot=visual_snapshot,
            )
            if result.get("source") == "vlm":
                return result
            first_error = result.get("error")

        if first_error:
            payload["visual_attachment_retry_note"] = (
                "The RGB image attachment failed at the API layer. Use the structured "
                "YOLO/landmark objects, target candidates, and GT trajectory context to "
                "produce the reflection."
            )
        result = self._call_vlm_reflector(
            payload=payload,
            image_data_urls=[],
            previous=previous,
            visual_snapshot=visual_snapshot,
        )
        if result.get("source") == "vlm":
            if first_error:
                result.setdefault("visual_input", {})["image_call_error"] = first_error
                result.setdefault("visual_input", {})["used_text_retry"] = True
            return result
        if first_error and result.get("error"):
            result["image_call_error"] = first_error
        return result

    def _call_vlm_reflector(
        self,
        payload: Dict[str, Any],
        image_data_urls: list[str],
        previous: Dict[str, Any],
        visual_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            user_prompt = json.dumps(payload, indent=2, sort_keys=True, default=str)
            if image_data_urls:
                raw = self.vlm_provider.generate(
                    ONLINE_GT_DEVIATION_SYSTEM_PROMPT,
                    user_prompt,
                    image_data_urls=image_data_urls,
                )
            else:
                raw = self.vlm_provider.generate(ONLINE_GT_DEVIATION_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            return {"source": "vlm_error", "error": f"{type(exc).__name__}: {exc}"}

        parsed = self._parse_json(raw)
        if not parsed:
            return {
                "source": "vlm_error",
                "error": "VLM response was not valid JSON",
                "raw_response_excerpt": (raw or "")[:800],
            }
        parsed["source"] = "vlm"
        parsed["raw_response"] = raw
        parsed["visual_input"] = {
            "previous_rgb_attached": bool(image_data_urls),
            "previous_detected_object_count": len(
                self._detected_objects_from_previous(previous, visual_snapshot)
            ),
        }
        return self._sanitize_reflection(parsed, previous)

    def _memory_item(
        self,
        state: Dict[str, Any],
        previous: Dict[str, Any],
        previous_gt: Dict[str, Any],
        current_gt: Dict[str, Any],
        before_dev: float,
        after_dev: float,
        growth: float,
        reflection: Dict[str, Any],
    ) -> ExperienceMemoryItem:
        previous_skill = self._previous_skill(previous)
        visual_evidence = self._visual_evidence_grade(reflection, previous)
        better_skill = self._safe_better_skill(
            reflection.get("better_skill"), previous, visual_evidence
        )
        lesson = self._actionable_lesson(
            reflection=reflection,
            state=state,
            previous_skill=previous_skill,
            better_skill=better_skill,
            visual_evidence=visual_evidence,
        )
        rationale = self._conservative_rationale(
            reflection=reflection,
            previous_skill=previous_skill,
            better_skill=better_skill,
            before_dev=before_dev,
            after_dev=after_dev,
            growth=growth,
            visual_evidence=visual_evidence,
        )
        return ExperienceMemoryItem(
            split=state.get("split") or "unknown",
            scene_id=state.get("scene_id"),
            episode_id=state.get("episode_id"),
            target_category=state.get("target_category"),
            success=False,
            failure_type="gt_trajectory_deviation",
            failure_class=classify_failure("gt_trajectory_deviation"),
            state_condition={
                "student_selected_skill": previous_skill,
                "selected_skill_sequence": [previous_skill] if previous_skill else [],
                "learning_trigger": "online_teacher_route_deviation",
                "teacher_signal_used_for_learning": True,
                "teacher_path_interpretation": self._safe_text(
                    reflection.get("teacher_path_interpretation") or reflection.get("gt_difference"),
                    520,
                ),
                "decision_error": self._safe_text(
                    reflection.get("decision_error") or reflection.get("failure_analysis"),
                    520,
                ),
                "test_time_condition": self._safe_mapping(
                    reflection.get("test_time_condition"),
                    limit=900,
                ),
                "skill_preference_rule": self._safe_text(
                    reflection.get("skill_preference_rule"),
                    420,
                ),
                "avoid_rule": self._safe_text(reflection.get("avoid_rule"), 420),
                "teacher_signal_debug": {
                    "distance_to_gt_path_before": before_dev,
                    "distance_to_gt_path_after": after_dev,
                    "gt_deviation_growth": growth,
                    "gt_progress_ratio_before": previous_gt.get("gt_path_progress_ratio"),
                    "gt_progress_ratio_after": current_gt.get("gt_path_progress_ratio"),
                },
                "visual_evidence_used": self._safe_visual_evidence_note(
                    reflection.get("visual_evidence_used"),
                    visual_evidence,
                ),
                "visual_evidence": visual_evidence,
                "visual_evidence_reliability": visual_evidence.get("reliability"),
                "visual_target_relation": visual_evidence.get("target_relation"),
                "visual_room_cues": visual_evidence.get("room_cues"),
                "visual_generic_landmarks": visual_evidence.get("generic_landmarks"),
                "target_candidate_evidence": visual_evidence.get("target_candidate_evidence"),
                "previous_rgb_attached": (reflection.get("visual_input") or {}).get(
                    "previous_rgb_attached"
                ),
                "previous_detected_object_count": (reflection.get("visual_input") or {}).get(
                    "previous_detected_object_count"
                ),
            },
            bad_decision=reflection.get("bad_decision")
            or f"skill={previous_skill}; args={self._previous_skill_args(previous)}",
            better_decision=self._safe_better_decision(reflection, better_skill),
            lesson=lesson,
            suggested_policy_patch={
                "target_scope": state.get("target_category"),
                "trigger_condition": {
                    "selected_skill": previous_skill,
                    "failure_type": "gt_trajectory_deviation",
                    "distance_to_gt_path_gte": self.cfg.gt_path_deviation_threshold,
                },
                "recommended_action": better_skill,
                "rationale": rationale,
                "confidence": self._bounded_confidence(reflection.get("confidence"), 0.78),
                "support_count": 1,
                "source_episode_id": state.get("episode_id"),
            },
            confidence=self._bounded_confidence(reflection.get("confidence"), 0.78),
        )

    def _episode_log_path(self, episode_id: Any) -> Optional[str]:
        root = self.cfg.episode_log_root
        if not root:
            return None
        run_id = self.cfg.run_id or "default"
        episodes_dir = os.path.join(root, run_id, "episodes")
        if episode_id:
            return os.path.join(episodes_dir, f"{self._safe_name(episode_id)}.json")
        return os.path.join(episodes_dir, "unknown_episode.json")

    def _sidecar_path(self, episode_id: Any) -> Optional[str]:
        root = self.cfg.episode_log_root
        if not root:
            return None
        run_id = self.cfg.run_id or "default"
        return os.path.join(
            root,
            run_id,
            "gt_deviation_reflections",
            f"{self._safe_name(episode_id or 'unknown_episode')}.json",
        )

    def _load_sidecar(self, episode_id: Any) -> Dict[str, Any]:
        path = self._sidecar_path(episode_id)
        if not path or not os.path.exists(path):
            return {"reflected_timesteps": []}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("reflected_timesteps", [])
            return data
        except Exception:
            return {"reflected_timesteps": []}

    def _write_sidecar(self, episode_id: Any, data: Dict[str, Any]) -> None:
        path = self._sidecar_path(episode_id)
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)

    def _record_skipped_vlm_reflection(
        self,
        episode_id: Any,
        previous_timestep: Any,
        before_dev: float,
        after_dev: float,
        growth: float,
        reflection: Dict[str, Any],
    ) -> None:
        sidecar = self._load_sidecar(episode_id)
        sidecar.setdefault("skipped_vlm_reflections", []).append({
            "previous_timestep": previous_timestep,
            "distance_to_gt_path_before": before_dev,
            "distance_to_gt_path_after": after_dev,
            "gt_deviation_growth": growth,
            "source": reflection.get("source"),
            "error": reflection.get("error") or reflection.get("image_call_error"),
        })
        self._write_sidecar(episode_id, sidecar)

    def _debounced(self, previous_timestep: Any, sidecar: Dict[str, Any]) -> bool:
        try:
            last = int(sidecar.get("last_reflection_timestep"))
            current = int(previous_timestep)
        except (TypeError, ValueError):
            return False
        return current - last < int(self.cfg.online_gt_deviation_reflection_debounce_steps)

    @staticmethod
    def _previous_decision(episode_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        decisions = episode_data.get("decisions") or []
        if not decisions:
            return None
        return decisions[-1]

    @staticmethod
    def _gt_context(state: Dict[str, Any]) -> Dict[str, Any]:
        feedback = state.get("gt_feedback") or {}
        return dict(feedback.get("gt_trajectory") or {})

    @staticmethod
    def _previous_skill(previous: Dict[str, Any]) -> Optional[str]:
        decision = previous.get("agent_decision") or {}
        validator = previous.get("validator_result") or {}
        return decision.get("selected_skill") or validator.get("final_skill") or previous.get("executed_skill")

    @staticmethod
    def _previous_skill_args(previous: Dict[str, Any]) -> Dict[str, Any]:
        decision = previous.get("agent_decision") or {}
        validator = previous.get("validator_result") or {}
        return decision.get("skill_args") or validator.get("final_arguments") or {}

    def _heuristic_better_skill(self, previous: Dict[str, Any]) -> str:
        state = previous.get("state_summary") or {}
        candidates = state.get("target_candidates") or []
        if candidates:
            best = max(candidates, key=lambda item: item.get("confidence") or 0.0)
            if (best.get("confidence") or 0.0) >= self.cfg.target_stop_threshold:
                if not self.cfg.require_multiview_before_stop or best.get("multi_view_confirmed"):
                    return SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value
            if self.cfg.disable_verify_target:
                return SkillName.FOLLOW_APEXNAV_PROPOSAL.value
            return SkillName.VERIFY_TARGET.value
        if (state.get("semantic_score_stats") or {}).get("has_clear_peak"):
            return SkillName.SEMANTIC_EXPLORE.value
        if state.get("frontiers"):
            return SkillName.GEOMETRIC_EXPLORE.value
        return SkillName.FOLLOW_APEXNAV_PROPOSAL.value

    def _sanitize_reflection(
        self,
        reflection: Dict[str, Any],
        previous: Dict[str, Any],
    ) -> Dict[str, Any]:
        reflection = dict(reflection or {})
        visual_evidence = self._visual_evidence_grade(reflection, previous)
        reflection["visual_evidence"] = visual_evidence
        requested_skill = reflection.get("better_skill")
        safe_skill = self._safe_better_skill(requested_skill, previous, visual_evidence)
        if requested_skill != safe_skill:
            reflection["better_skill_sanitized_from"] = requested_skill
            reflection["better_skill"] = safe_skill

        safe_visual_note = self._safe_visual_evidence_note(
            reflection.get("visual_evidence_used"), visual_evidence
        )
        if safe_visual_note != reflection.get("visual_evidence_used"):
            reflection["visual_evidence_rejected_reason"] = "rewritten_as_structured_evidence_grade"
        reflection["visual_evidence_used"] = safe_visual_note

        if self._contains_unstable_visual_association(reflection.get("lesson"), visual_evidence):
            reflection["lesson_rejected_reason"] = "lesson_relied_on_unstable_visual_association"
            reflection["lesson"] = None
        if self._contains_unstable_visual_association(reflection.get("better_decision"), visual_evidence):
            reflection["better_decision_rejected_reason"] = "better_decision_relied_on_unstable_visual_association"
            reflection["better_decision"] = safe_skill
        return reflection

    def _safe_better_skill(
        self,
        requested: Any,
        previous: Dict[str, Any],
        visual_evidence: Optional[Dict[str, Any]] = None,
    ) -> str:
        allowed = {item.value for item in SkillName}
        if self.cfg.disable_verify_target:
            allowed.discard(SkillName.VERIFY_TARGET.value)
        skill = requested if requested in allowed else self._heuristic_better_skill(previous)
        state = previous.get("state_summary") or {}
        visual_evidence = visual_evidence or self._visual_evidence_grade({}, previous)
        visual_reliability = visual_evidence.get("reliability")
        if skill == SkillName.NAVIGATE_TO_CONFIRMED_TARGET.value:
            if visual_reliability != "strong" or not self._has_reliable_target_candidate(state):
                if self._has_any_target_candidate(state) and not self.cfg.disable_verify_target:
                    return SkillName.VERIFY_TARGET.value
                return SkillName.FOLLOW_APEXNAV_PROPOSAL.value
        if skill == SkillName.VERIFY_TARGET.value and self.cfg.disable_verify_target:
            skill = SkillName.FOLLOW_APEXNAV_PROPOSAL.value
        if skill == SkillName.VERIFY_TARGET.value and not self._has_any_target_candidate(state):
            if (state.get("semantic_score_stats") or {}).get("has_clear_peak"):
                skill = SkillName.SEMANTIC_EXPLORE.value
            else:
                skill = SkillName.FOLLOW_APEXNAV_PROPOSAL.value
        previous_skill = self._previous_skill(previous)
        if skill == previous_skill:
            skill = self._alternative_skill_after_deviation(state, previous_skill)
        return str(skill)

    def _alternative_skill_after_deviation(
        self,
        state: Dict[str, Any],
        previous_skill: Optional[str],
    ) -> str:
        if self._best_known_point_available(state) and previous_skill != SkillName.RETURN_TO_BEST_KNOWN_POINT.value:
            return SkillName.RETURN_TO_BEST_KNOWN_POINT.value
        if (
            (state.get("semantic_score_stats") or {}).get("has_clear_peak")
            and previous_skill != SkillName.SEMANTIC_EXPLORE.value
        ):
            return SkillName.SEMANTIC_EXPLORE.value
        if state.get("frontiers") and previous_skill != SkillName.GEOMETRIC_EXPLORE.value:
            return SkillName.GEOMETRIC_EXPLORE.value
        if previous_skill != SkillName.FOLLOW_APEXNAV_PROPOSAL.value:
            return SkillName.FOLLOW_APEXNAV_PROPOSAL.value
        return SkillName.FALLBACK_APEXNAV.value

    def _actionable_lesson(
        self,
        reflection: Dict[str, Any],
        state: Dict[str, Any],
        previous_skill: Optional[str],
        better_skill: str,
        visual_evidence: Optional[Dict[str, Any]] = None,
    ) -> str:
        visual_evidence = visual_evidence or {}
        for key in ("lesson", "skill_preference_rule", "avoid_rule"):
            candidate = self._safe_test_time_lesson(
                reflection.get(key),
                visual_evidence=visual_evidence,
            )
            if candidate:
                return candidate
        return self._conservative_lesson(
            state=state,
            previous_skill=previous_skill,
            better_skill=better_skill,
            visual_evidence=visual_evidence,
        )

    def _conservative_lesson(
        self,
        state: Dict[str, Any],
        previous_skill: Optional[str],
        better_skill: str,
        visual_evidence: Optional[Dict[str, Any]] = None,
    ) -> str:
        target = state.get("target_category") or "the target"
        visual_evidence = visual_evidence or {}
        reliability = visual_evidence.get("reliability") or "none"
        target_relation = visual_evidence.get("target_relation") or "none"
        room_cues = visual_evidence.get("room_cues") or []
        candidate_evidence = visual_evidence.get("target_candidate_evidence") or []
        visual_clause = self._visual_lesson_clause(
            reliability=reliability,
            target_relation=target_relation,
            room_cues=room_cues,
            candidate_evidence=candidate_evidence,
        )
        state_clause = self._state_lesson_clause(state)
        return (
            f"For target={target}, when the agent has just used {previous_skill} under "
            f"{state_clause}, use the current visual/frontier evidence to reconsider the "
            f"skill instead of repeating the same subgoal. {visual_clause} Prefer "
            f"{better_skill} when its validator preconditions are satisfied."
        )

    def _conservative_rationale(
        self,
        reflection: Dict[str, Any],
        previous_skill: Optional[str],
        better_skill: str,
        before_dev: float,
        after_dev: float,
        growth: float,
        visual_evidence: Optional[Dict[str, Any]] = None,
    ) -> str:
        candidate = reflection.get("gt_difference") or reflection.get("failure_analysis")
        visual_evidence = visual_evidence or {}
        if candidate and not self._contains_unstable_visual_association(candidate, visual_evidence):
            base = str(candidate)[:420]
            return f"{base} Visual evidence reliability={visual_evidence.get('reliability', 'none')}."
        return (
            f"GT-path deviation grew from {before_dev:.2f}m to {after_dev:.2f}m "
            f"(+{growth:.2f}m) after {previous_skill}; this is evidence to reconsider "
            f"the high-level skill and prefer {better_skill} under valid preconditions. "
            f"Visual evidence reliability={visual_evidence.get('reliability', 'none')}."
        )

    def _safe_test_time_lesson(
        self,
        value: Any,
        visual_evidence: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        text = self._safe_text(value, 620)
        if not text:
            return None
        lowered = text.lower()
        forbidden = (
            "gt", "ground truth", "teacher", "oracle", "shortest path",
            "deviation", "distance-to-gt", "distance to gt", "coordinate",
        )
        if any(term in lowered for term in forbidden):
            return None
        if self._contains_unstable_visual_association(text, visual_evidence):
            return None
        return text

    @staticmethod
    def _state_lesson_clause(state: Dict[str, Any]) -> str:
        semantic = state.get("semantic_score_stats") or {}
        history = state.get("navigation_history") or {}
        target_candidates = state.get("target_candidates") or []
        frontiers = state.get("frontiers") or []
        clauses = []
        if semantic.get("has_clear_peak"):
            clauses.append("a clear semantic peak")
        elif semantic:
            clauses.append("no clear semantic peak")
        if target_candidates:
            clauses.append("visible target candidates")
        else:
            clauses.append("no confirmed target candidate")
        if frontiers:
            clauses.append("available frontiers")
        else:
            clauses.append("no available frontier")
        if history.get("stuck_count"):
            clauses.append("recent stuck signal")
        return ", ".join(clauses[:4]) or "the current navigation state"

    def _visual_evidence_grade(
        self,
        reflection: Dict[str, Any],
        previous: Dict[str, Any],
    ) -> Dict[str, Any]:
        state = previous.get("state_summary") or {}
        target = state.get("target_category")
        raw = reflection.get("visual_evidence") if isinstance(reflection, dict) else {}
        raw = raw if isinstance(raw, dict) else {}

        target_candidates = self._matching_target_candidates(state)
        reliable_candidates = [
            item for item in target_candidates if self._candidate_is_reliable(item, target)
        ]
        uncertain_candidates = [
            item for item in target_candidates if item not in reliable_candidates
        ]

        raw_room_cues = self._as_string_list(raw.get("room_cues"))
        raw_generic = self._as_string_list(raw.get("generic_landmarks"))
        room_cues = raw_room_cues[:6]
        generic_landmarks = [
            item for item in raw_generic
            if item and item not in room_cues
        ][:6]

        detected = self._compact_detected_objects(state.get("detected_objects") or [], 8)
        if not generic_landmarks:
            generic_landmarks = [
                str(item.get("label"))
                for item in detected
                if item.get("label")
                and (not target or str(item.get("label")).lower() != str(target).lower())
            ][:6]

        requested = str(raw.get("reliability") or reflection.get("visual_evidence_reliability") or "").lower()
        if requested not in {"strong", "medium", "weak", "none"}:
            requested = ""
        if reliable_candidates:
            reliability = "strong"
            target_relation = "target_candidate"
        elif uncertain_candidates:
            reliability = "medium"
            target_relation = "target_candidate"
        elif room_cues and requested in {"strong", "medium"}:
            reliability = "medium"
            target_relation = "room_cue"
        elif room_cues and requested in {"weak", "none"}:
            reliability = requested
            target_relation = "room_cue" if requested == "weak" else "none"
        elif generic_landmarks or detected or (reflection.get("visual_input") or {}).get("previous_rgb_attached"):
            reliability = requested or "weak"
            if reliability == "medium":
                target_relation = str(raw.get("target_relation") or "room_cue")
            elif reliability == "none":
                target_relation = "none"
            else:
                target_relation = "generic_landmark"
        else:
            reliability = "none"
            target_relation = "none"

        candidate_evidence = self._compact_candidate_evidence(
            reliable_candidates or uncertain_candidates
        )
        summary = self._visual_evidence_summary(
            reliability=reliability,
            target_relation=target_relation,
            target=target,
            room_cues=room_cues,
            generic_landmarks=generic_landmarks,
            candidate_evidence=candidate_evidence,
        )
        return {
            "reliability": reliability,
            "target_relation": target_relation,
            "target_candidate_evidence": candidate_evidence,
            "room_cues": room_cues,
            "generic_landmarks": generic_landmarks,
            "summary": summary,
            "vlm_reason": self._safe_text(raw.get("reason"), 300),
            "uncertainty": self._safe_text(raw.get("uncertainty"), 240),
            "grading_source": "vlm_with_safety_overrides",
        }

    @staticmethod
    def _visual_evidence_summary(
        reliability: str,
        target_relation: str,
        target: Any,
        room_cues: list[str],
        generic_landmarks: list[str],
        candidate_evidence: list[Dict[str, Any]],
    ) -> str:
        if reliability == "strong":
            return (
                f"Strong visual evidence: reliable {target} target candidate is present; "
                "target navigation may be considered only through validator gates."
            )
        if reliability == "medium" and target_relation == "target_candidate":
            return (
                f"Medium visual evidence: an uncertain {target} target candidate is present; "
                "verification is preferred before target navigation."
            )
        if reliability == "medium":
            cues = ", ".join(room_cues[:3]) if room_cues else "target-related room cues"
            return (
                f"Medium visual evidence: {cues} may relate to target={target}; "
                "use it to bias semantic exploration, not direct stop."
            )
        if reliability == "weak":
            landmarks = ", ".join(generic_landmarks[:3]) if generic_landmarks else "generic landmarks"
            return (
                f"Weak visual evidence: only {landmarks} were available; "
                "do not treat them as proof of target location."
            )
        return "No useful visual evidence was available."

    @staticmethod
    def _visual_lesson_clause(
        reliability: str,
        target_relation: str,
        room_cues: list[str],
        candidate_evidence: list[Dict[str, Any]],
    ) -> str:
        if reliability == "strong":
            return (
                "Visual evidence was strong because a reliable target candidate was present; "
                "target-oriented skills can be considered through validator gates."
            )
        if reliability == "medium" and target_relation == "target_candidate":
            return (
                "Visual evidence was medium because the target candidate was not fully reliable; "
                "prefer verification before any stop or final target navigation."
            )
        if reliability == "medium":
            cues = ", ".join(room_cues[:3]) if room_cues else "target-related room cues"
            return (
                f"Visual evidence was medium ({cues}); use it to bias semantic exploration "
                "or verification, not as proof of target location."
            )
        if reliability == "weak":
            return (
                "Visual evidence was weak and mostly generic; it can explain the scene "
                "but should not drive target navigation."
            )
        return "No reliable visual evidence supported a target-specific shortcut."

    @staticmethod
    def _safe_better_decision(reflection: Dict[str, Any], better_skill: str) -> str:
        candidate = reflection.get("better_decision")
        visual_evidence = reflection.get("visual_evidence") if isinstance(reflection, dict) else {}
        if candidate and not OnlineGTDeviationReflector._contains_unstable_visual_association(candidate, visual_evidence):
            return str(candidate)[:500]
        return better_skill

    @staticmethod
    def _safe_visual_evidence_note(
        note: Any,
        visual_evidence: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        visual_evidence = visual_evidence or {}
        reliability = visual_evidence.get("reliability", "none")
        relation = visual_evidence.get("target_relation", "none")
        if reliability == "none":
            return None
        summary = visual_evidence.get("summary")
        if isinstance(summary, str) and summary:
            return summary[:300]
        if isinstance(note, str) and note.strip():
            return note.strip()[:240]
        return f"Visual evidence reliability={reliability}, target_relation={relation}."

    @staticmethod
    def _contains_unstable_visual_association(
        text: Any,
        visual_evidence: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not isinstance(text, str):
            return False
        visual_evidence = visual_evidence or {}
        if visual_evidence.get("reliability") in {"strong", "medium"}:
            return False
        lowered = text.lower()
        association_terms = (
            "infer", "likely", "suggest", "indicate", "associated", "near",
            "toward", "towards", "go to", "navigate to", "move to", "location",
            "showing", "visible", "prioritize", "detected",
        )
        return any(term in lowered for term in association_terms) and any(
            landmark in lowered for landmark in OnlineGTDeviationReflector._generic_landmark_terms()
        )

    @staticmethod
    def _mentions_generic_landmark(text: Any) -> bool:
        if not isinstance(text, str):
            return False
        lowered = text.lower()
        return any(landmark in lowered for landmark in OnlineGTDeviationReflector._generic_landmark_terms())

    @staticmethod
    def _generic_landmark_terms() -> tuple[str, ...]:
        return (
            "landmark", "table", "chair", "door", "sofa", "couch", "cabinet",
            "counter", "shelf", "furniture", "dining", "room", "hallway",
            "wall", "refrigerator", "fridge", "sink", "oven", "microwave",
        )

    def _matching_target_candidates(self, state: Dict[str, Any]) -> list[Dict[str, Any]]:
        target = state.get("target_category")
        matches = []
        for candidate in state.get("target_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            label = candidate.get("label") or candidate.get("category")
            if target and label and str(label).lower() != str(target).lower():
                continue
            if target and not label and not candidate.get("is_target_candidate"):
                continue
            matches.append(candidate)
        return matches

    def _candidate_is_reliable(self, candidate: Dict[str, Any], target: Any = None) -> bool:
        label = candidate.get("label") or candidate.get("category")
        if target and label and str(label).lower() != str(target).lower():
            return False
        confidence = candidate.get("confidence")
        if confidence is None:
            confidence = candidate.get("score")
        try:
            confidence_value = float(confidence or 0.0)
        except (TypeError, ValueError):
            confidence_value = 0.0
        if confidence_value < float(self.cfg.target_stop_threshold):
            return False
        if candidate.get("reachable") is False:
            return False
        if candidate.get("rejected_false_positive"):
            return False
        if self.cfg.require_multiview_before_stop and not candidate.get("multi_view_confirmed"):
            return False
        return True

    @staticmethod
    def _compact_candidate_evidence(candidates: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        evidence = []
        for item in candidates[:3]:
            evidence.append({
                "id": item.get("id"),
                "label": item.get("label") or item.get("category"),
                "confidence": item.get("confidence") if item.get("confidence") is not None else item.get("score"),
                "distance": item.get("distance"),
                "reachable": item.get("reachable", True),
                "multi_view_confirmed": item.get("multi_view_confirmed"),
                "num_views": item.get("num_views"),
            })
        return evidence

    @staticmethod
    def _as_string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text and text not in result:
                result.append(text[:80])
        return result

    @staticmethod
    def _safe_text(value: Any, limit: int) -> Optional[str]:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        return text[:limit]

    @staticmethod
    def _safe_mapping(value: Any, limit: int = 900) -> Optional[Dict[str, Any]]:
        if not isinstance(value, dict):
            return None
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return None
        if len(text) > limit:
            text = text[:limit]
            try:
                return {"summary": text}
            except Exception:
                return None
        return value

    def _has_any_target_candidate(self, state: Dict[str, Any]) -> bool:
        target = state.get("target_category")
        for candidate in state.get("target_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            label = candidate.get("label") or candidate.get("category")
            if target and label and str(label).lower() != str(target).lower():
                continue
            return True
        return False

    def _has_reliable_target_candidate(self, state: Dict[str, Any]) -> bool:
        target = state.get("target_category")
        for candidate in state.get("target_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            label = candidate.get("label") or candidate.get("category")
            if target and label and str(label).lower() != str(target).lower():
                continue
            confidence = candidate.get("confidence")
            if confidence is None:
                confidence = candidate.get("score")
            try:
                confidence_value = float(confidence or 0.0)
            except (TypeError, ValueError):
                confidence_value = 0.0
            if confidence_value < float(self.cfg.target_stop_threshold):
                continue
            if candidate.get("reachable") is False:
                continue
            if candidate.get("rejected_false_positive"):
                continue
            if self.cfg.require_multiview_before_stop and not candidate.get("multi_view_confirmed"):
                continue
            return True
        return False

    @staticmethod
    def _best_known_point_available(state: Dict[str, Any]) -> bool:
        history = state.get("navigation_history") or {}
        best = history.get("best_known_point") or state.get("best_known_point")
        return isinstance(best, dict) and bool(best.get("available"))

    @staticmethod
    def _compact_decision(previous: Dict[str, Any]) -> Dict[str, Any]:
        state = previous.get("state_summary") or {}
        return {
            "timestep": previous.get("timestep"),
            "agent_decision": OnlineGTDeviationReflector._small_decision(previous.get("agent_decision")),
            "validator_result": OnlineGTDeviationReflector._small_validator(previous.get("validator_result")),
            "executed_skill": previous.get("executed_skill"),
            "semantic_score_stats": state.get("semantic_score_stats"),
            "frontiers": OnlineGTDeviationReflector._compact_frontiers(state.get("frontiers") or [], 5),
            "target_candidates": OnlineGTDeviationReflector._compact_candidates(state.get("target_candidates") or [], 3),
            "detected_objects": OnlineGTDeviationReflector._compact_detected_objects(
                state.get("detected_objects") or [], 5
            ),
            "rgb_observation": OnlineGTDeviationReflector._compact_observation(
                state.get("rgb_observation")
            ),
            "navigation_history": OnlineGTDeviationReflector._compact_navigation_history(state.get("navigation_history") or {}),
            "retrieved_lessons": OnlineGTDeviationReflector._compact_lessons(previous.get("retrieved_lessons") or [], 3),
        }

    @staticmethod
    def _prompt_safe_state(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "episode_id": state.get("episode_id"),
            "scene_id": state.get("scene_id"),
            "split": state.get("split"),
            "timestep": state.get("timestep"),
            "target_category": state.get("target_category"),
            "semantic_score_stats": state.get("semantic_score_stats"),
            "frontiers": OnlineGTDeviationReflector._compact_frontiers(state.get("frontiers") or [], 5),
            "target_candidates": OnlineGTDeviationReflector._compact_candidates(state.get("target_candidates") or [], 3),
            "detected_objects": OnlineGTDeviationReflector._compact_detected_objects(state.get("detected_objects") or [], 5),
            "navigation_history": OnlineGTDeviationReflector._compact_navigation_history(state.get("navigation_history") or {}),
            "recent_lessons": OnlineGTDeviationReflector._compact_lessons(state.get("retrieved_lessons") or [], 3),
            "rgb_observation": OnlineGTDeviationReflector._compact_observation(state.get("rgb_observation")),
            "semantic_map_observation": OnlineGTDeviationReflector._compact_observation(state.get("semantic_map_observation")),
        }

    @staticmethod
    def _small_decision(decision: Any) -> Dict[str, Any]:
        decision = decision or {}
        if not isinstance(decision, dict):
            return {}
        return {
            "status": decision.get("status"),
            "selected_skill": decision.get("selected_skill"),
            "skill_args": decision.get("skill_args") or {},
            "reason": decision.get("reason"),
            "confidence": decision.get("confidence"),
        }

    @staticmethod
    def _small_validator(validator: Any) -> Dict[str, Any]:
        validator = validator or {}
        if not isinstance(validator, dict):
            return {}
        return {
            "final_skill": validator.get("final_skill"),
            "final_arguments": validator.get("final_arguments") or {},
            "accepted": validator.get("accepted"),
            "rejection_reason": validator.get("rejection_reason"),
            "fallback_used": validator.get("fallback_used"),
        }

    @staticmethod
    def _compact_navigation_history(history: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "recent_selected_skills": (history.get("recent_selected_skills") or [])[-6:],
            "recent_failures": (history.get("recent_failures") or [])[-6:],
            "visited_frontier_ids": (history.get("visited_frontier_ids") or [])[-8:],
            "stuck_count": history.get("stuck_count"),
            "collision_count": history.get("collision_count"),
            "steps_left": history.get("steps_left"),
        }

    @staticmethod
    def _compact_frontiers(frontiers: Any, max_items: int) -> list[Dict[str, Any]]:
        def sort_key(item: Dict[str, Any]):
            semantic = item.get("semantic_score") or 0.0
            distance = item.get("distance") if item.get("distance") is not None else 1e9
            reachable_penalty = 0 if item.get("reachable", True) else 1
            return (reachable_penalty, -float(semantic), float(distance))

        compact = []
        for item in sorted([x for x in frontiers if isinstance(x, dict)], key=sort_key)[:max_items]:
            compact.append({
                "id": item.get("id"),
                "semantic_score": item.get("semantic_score"),
                "distance": item.get("distance"),
                "reachable": item.get("reachable", True),
                "blocked": item.get("blocked", False),
                "low_value": item.get("low_value", False),
                "failure_count": item.get("failure_count", 0),
                "last_selected": item.get("last_selected", False),
            })
        return compact

    @staticmethod
    def _compact_candidates(candidates: Any, max_items: int) -> list[Dict[str, Any]]:
        def sort_key(item: Dict[str, Any]):
            confidence = item.get("confidence") or item.get("score") or 0.0
            distance = item.get("distance") if item.get("distance") is not None else 1e9
            return (-float(confidence), float(distance))

        compact = []
        for item in sorted([x for x in candidates if isinstance(x, dict)], key=sort_key)[:max_items]:
            compact.append({
                "id": item.get("id"),
                "label": item.get("label"),
                "label_id": item.get("label_id"),
                "confidence": item.get("confidence") or item.get("score"),
                "distance": item.get("distance"),
                "reachable": item.get("reachable", True),
                "multi_view_confirmed": item.get("multi_view_confirmed"),
                "num_views": item.get("num_views"),
                "bbox": item.get("bbox"),
                "center": item.get("center"),
                "direction": item.get("direction"),
                "source": item.get("source"),
                "is_landmark": item.get("is_landmark"),
                "is_target_candidate": item.get("is_target_candidate"),
            })
        return compact

    @staticmethod
    def _compact_detected_objects(objects: Any, max_items: int) -> list[Dict[str, Any]]:
        if isinstance(objects, dict) and "detections" in objects:
            objects = objects.get("detections") or []

        def sort_key(item: Dict[str, Any]):
            confidence = item.get("confidence") or item.get("score") or 0.0
            distance = item.get("distance") if item.get("distance") is not None else 1e9
            return (-float(confidence), float(distance))

        compact = []
        for item in sorted([x for x in objects if isinstance(x, dict)], key=sort_key)[:max_items]:
            compact.append({
                "id": item.get("id"),
                "label": item.get("label"),
                "label_id": item.get("label_id"),
                "confidence": item.get("confidence") or item.get("score"),
                "bbox": item.get("bbox"),
                "center": item.get("center"),
                "direction": item.get("direction"),
                "distance": item.get("distance"),
                "reachable": item.get("reachable", True),
                "is_target_candidate": item.get("is_target_candidate"),
                "is_landmark": item.get("is_landmark"),
                "grounded_in_current_observation": item.get("grounded_in_current_observation"),
                "source": item.get("source"),
            })
        return compact

    @staticmethod
    def _load_visual_snapshot(previous: Dict[str, Any]) -> Dict[str, Any]:
        ref = previous.get("visual_snapshot") or {}
        path = ref.get("path") if isinstance(ref, dict) else None
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _image_data_urls_for_reflection(snapshot: Dict[str, Any]) -> list[str]:
        urls = []
        rgb = snapshot.get("rgb_observation") if isinstance(snapshot, dict) else {}
        if isinstance(rgb, dict):
            data_url = rgb.get("data_url") or rgb.get("image_url")
            if isinstance(data_url, str) and data_url.startswith("data:image/"):
                urls.append(data_url)
        return urls

    @staticmethod
    def _visual_evidence(previous: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
        state = previous.get("state_summary") or {}
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        rgb = snapshot.get("rgb_observation") or state.get("rgb_observation") or {}
        semmap = snapshot.get("semantic_map_observation") or state.get("semantic_map_observation") or {}
        return {
            "previous_rgb_observation": OnlineGTDeviationReflector._compact_observation(rgb),
            "previous_rgb_image_attached": bool(
                OnlineGTDeviationReflector._image_data_urls_for_reflection(snapshot)
            ),
            "previous_semantic_map_observation": OnlineGTDeviationReflector._compact_observation(semmap),
            "previous_detected_objects": OnlineGTDeviationReflector._detected_objects_from_previous(
                previous, snapshot
            ),
            "previous_target_candidates": OnlineGTDeviationReflector._compact_candidates(
                snapshot.get("target_candidates") or state.get("target_candidates") or [], 3
            ),
        }

    @staticmethod
    def _detected_objects_from_previous(
        previous: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        state = previous.get("state_summary") or {}
        objects = []
        if isinstance(snapshot, dict):
            objects = snapshot.get("detected_objects") or []
        if not objects:
            objects = state.get("detected_objects") or []
        return OnlineGTDeviationReflector._compact_detected_objects(objects, 8)

    @staticmethod
    def _compact_lessons(lessons: Any, max_items: int) -> list[Dict[str, Any]]:
        compact = []
        for item in [x for x in lessons if isinstance(x, dict)][:max_items]:
            compact.append({
                "target_category": item.get("target_category"),
                "failure_type": item.get("failure_type"),
                "lesson": item.get("lesson"),
                "confidence": item.get("confidence"),
            })
        return compact

    @staticmethod
    def _compact_observation(observation: Any) -> Dict[str, Any]:
        if not isinstance(observation, dict):
            return {}
        return {
            "available": observation.get("available"),
            "image_attached": bool(observation.get("data_url") or observation.get("image_url")),
            "width": observation.get("width"),
            "height": observation.get("height"),
            "timestep": observation.get("timestep"),
        }

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    return {}
        return {}

    @staticmethod
    def _bounded_confidence(value: Any, default: float) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = float(default)
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_name(value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:160]
