# SEMANTIC_EXPLORE

- purpose: Use ApexNav semantic planning to select a high-value frontier.
- inputs: semantic score stats, reachable frontier candidates, target category.
- preconditions: at least one reachable frontier; no reliable confirmed target.
- forward_action: call the original ApexNav semantic frontier selector and navigate to the selected waypoint.
- expected_postconditions: waypoint reached, explored area grows, or target confidence/semantic score improves.
- failure_signals: unreachable waypoint, no map expansion, target confidence not improved, repeated same frontier failure.
- recovery_action: mark frontier low_value or blocked; switch to GEOMETRIC_EXPLORE or RECOVER_FROM_STUCK.
- memory_updates: semantic_frontier_success or semantic_frontier_failure with score, distance, target category.
- validator_constraints: reachable frontier required; reject repeated failed frontier beyond threshold.
- logging_fields: selected_frontier_id, semantic_score, distance, waypoint, postcondition status.
