# GEOMETRIC_EXPLORE

- purpose: Use ApexNav geometric planning to select the nearest reachable frontier.
- inputs: reachable frontier candidates and navigation history.
- preconditions: at least one reachable frontier.
- forward_action: call the original ApexNav nearest-frontier selector and navigate to the selected waypoint.
- expected_postconditions: waypoint reached and explored area grows or new frontier/observation appears.
- failure_signals: unreachable waypoint, no new explored area, repeated collision, stuck.
- recovery_action: mark frontier blocked, choose next reachable frontier, or call RECOVER_FROM_STUCK.
- memory_updates: geometric_frontier_success or blocked_frontier.
- validator_constraints: reachable frontier required.
- logging_fields: selected_frontier_id, distance, waypoint, collision/stuck counters.
