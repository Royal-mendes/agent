# RETURN_TO_BEST_KNOWN_POINT

## Purpose
Return to the best historical navigation point recorded in the current episode when the agent judges that continuing semantic or geometric exploration is unlikely to improve navigation.

The exposed best point is not simply the point with the strongest historical signal. It is selected from a short history by return utility:

`return_utility = candidate_runtime_evidence - current_runtime_evidence - distance_penalty * distance_to_current`

## Inputs
- `navigation_history.best_known_point`
- `target_candidates`
- `semantic_score_stats`
- `frontiers`

## Preconditions
- A valid best known point is available.
- Target preemption still applies. If a target candidate exists, `NAVIGATE_TO_CONFIRMED_TARGET` has priority when the stop gate confirms the target.

## Forward Action
Call ApexNav's existing path search to navigate from the current position back to the recorded best known point.

## Expected Postconditions
- The robot reaches the best known point, or
- the distance to the best known point decreases, or
- a valid waypoint toward the best known point is selected.

## Failure Signals
- `best_known_point_unavailable`
- `best_known_point_unreachable`
- `planner_stuck`
- `timeout`

## Recovery Action
Use `RECOVER_FROM_STUCK` if objective stuck evidence remains, otherwise use `FALLBACK_APEXNAV`.

## Memory Updates
- On failure: record `return_to_best_point_failed`.

## Validator Constraints
- Do not execute without a valid best known point.
- Target preemption still applies before this skill.

## Logging Fields
- best known point waypoint
- best known timestep
- best known return utility score
- raw runtime evidence score
- distance from current position
- target confidence / target views at that point
- local semantic score
- best frontier score
- selection signal
