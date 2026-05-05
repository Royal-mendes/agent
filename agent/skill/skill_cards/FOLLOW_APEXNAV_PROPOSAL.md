# FOLLOW_APEXNAV_PROPOSAL

- purpose: Deliberately follow the current ApexNav high-level proposal as a normal agent choice.
- inputs: ApexNav planner context, frontier state, target candidate state.
- preconditions: ApexNav proposal is available and no validator preemption applies.
- forward_action: Call the original ApexNav high-level policy.
- expected_postconditions: ApexNav advances to the next high-level event boundary.
- failure_signals: original_policy_failed, stuck, timeout, no_frontier.
- recovery_action: RECOVER_FROM_STUCK when objective failure evidence exists.
- memory_update_on_failure: record inefficient_or_failed_follow_apexnav if needed.
- validator_constraints: cannot bypass target preemption, recovery gate, or stop gate.
- logging_fields: trigger_reasons, commitment_age_steps, selected_skill, validator_result.
