"""The runtime's terminal verdict of one episode: the single owner of its result and success."""

from __future__ import annotations

from typing import Final

# arena_runtime_msgs/msg/EpisodeRecord outcome_state values; QUEUED and RUNNING are never resolved.
_VERDICTS: Final[dict[int, tuple[str, bool]]] = {
    2: ("GOAL_REACHED", True),
    3: ("FAILED", False),
    4: ("CANCELLED", False),
    5: ("FATAL", False),
}
UNRESOLVED: Final[str] = "UNRESOLVED"


def outcome_verdict(outcome_state: int | None, outcome_info: str | None) -> tuple[str, bool]:
    """(result, success) the runtime recorded; a state it never resolved is UNRESOLVED and no success."""
    if outcome_info == "collision":
        return "COLLISION", False
    if outcome_state is None:
        return UNRESOLVED, False
    return _VERDICTS.get(outcome_state, (UNRESOLVED, False))
