# NAVIGATE_TO_CONFIRMED_TARGET

- purpose: Navigate to a reliable target candidate and stop.
- inputs: target candidate id, confidence, reachability, multiview flag.
- preconditions: candidate confidence above stop threshold, reachable, and multiview confirmed when required.
- forward_action: call the original ApexNav target navigation path and trigger stop/finish.
- expected_postconditions: success condition satisfied or stop accepted by evaluator.
- failure_signals: false_positive_stop, premature_stop, confidence drops, target unreachable, collision, stuck.
- recovery_action: cancel stop when possible; continue exploration, RECOVER_FROM_STUCK, or FALLBACK_APEXNAV.
- memory_updates: premature_stop or false_positive_stop on failure; confirmed_target_success on success.
- validator_constraints: never stop on low-confidence single-view target.
- logging_fields: target_candidate_id, confidence, multi_view_confirmed, stop_reason, evaluator_success.
