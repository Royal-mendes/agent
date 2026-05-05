# RECOVER_FROM_STUCK

- purpose: Recover from repeated collision, stuck state, bad frontier, or map inconsistency.
- inputs: current waypoint/frontier, stuck count, recent failures, available frontiers.
- preconditions: stuck_count or repeated failure exceeds threshold, or validator requests recovery.
- forward_action: stop current execution, mark current frontier or waypoint temporarily blocked, and call recovery/fallback planning.
- expected_postconditions: stuck_count lowers, a new valid waypoint is selected, or blocked frontier is replaced.
- failure_signals: repeated stuck, no reachable frontier, map/localization inconsistent.
- recovery_action: FALLBACK_APEXNAV or terminate with structured failure.
- memory_updates: planner_stuck or repeated_bad_frontier.
- validator_constraints: can be disabled by ablation flag.
- logging_fields: blocked_frontiers, recovery_waypoint, stuck_count_before, stuck_count_after.
