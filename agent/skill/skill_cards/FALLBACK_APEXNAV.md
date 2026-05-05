# FALLBACK_APEXNAV

- purpose: Execute the original ApexNav high-level policy unchanged.
- inputs: original ApexNav planner context.
- preconditions: none.
- forward_action: call original ApexNav policy.
- expected_postconditions: original ApexNav decision returned in the existing waypoint/action format.
- failure_signals: original policy failure, no frontier, stuck.
- recovery_action: none.
- memory_updates: optional fallback usage count.
- validator_constraints: always valid fallback.
- logging_fields: fallback_reason, original_policy_metadata.
