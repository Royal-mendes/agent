# VERIFY_TARGET

- purpose: Verify a suspected target before allowing stop.
- inputs: target candidate id, confidence, num views, reachability.
- preconditions: at least one target candidate exists.
- forward_action: move to a safe observation viewpoint near the candidate or use rotate/look-around if available; trigger existing detection and fusion pipeline.
- expected_postconditions: candidate becomes confirmed or rejected, or confidence/num_views changes.
- failure_signals: confidence decreases, candidate disappears, remains single-view, verification viewpoint unreachable.
- recovery_action: reject false positive candidate; resume SEMANTIC_EXPLORE or GEOMETRIC_EXPLORE.
- memory_updates: false_positive_candidate evidence or verified_target evidence.
- validator_constraints: candidate required; use when stop threshold or multiview requirements are not met.
- logging_fields: target_candidate_id, confidence_before, confidence_after, num_views, observe_action_unavailable.
