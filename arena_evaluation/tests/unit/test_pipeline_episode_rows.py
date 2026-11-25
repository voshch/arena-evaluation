from __future__ import annotations


import polars as pl


_OLD_CONDITIONS = '[{"op": "always", "p": "old episode"}]'
_OLD_PHASES = '{"jackal": {"phases": [], "conditions": [], "map_poses": [[1.0, 2.0, 0.0]], "goal_inputs": [], "instructions": []}}'


def _bag_records() -> pl.DataFrame:
    """The previous episode's latched result, this episode's run and result, then the next episode's queued record."""
    return pl.DataFrame(
        {
            "t_ns": [50, 100, 900, 950],
            "episode_id": [22, 23, 23, 24],
            "outcome_state": [3, 1, 2, 0],
            "outcome_info": ["collision", "", "", ""],
            "integrity": [["sim_death"], [], [], ["queued"]],
            "conditions_yaml": [_OLD_CONDITIONS, "[]", "[]", "[]"],
            "phases_yaml": [_OLD_PHASES, "", "", ""],
        }
    )
