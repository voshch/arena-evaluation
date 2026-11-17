"""The one complete ``RobotParams`` builder for unit-test fixtures.

``RobotParams`` deliberately declares no defaults: production builds it only from the
recorded descriptor and parameter rows. Test fixtures still need a complete robot, so
this module owns the single literal fixture value set (the historical defaults the
registry used to assume) and every test constructs its robot through
``make_robot_params(**overrides)`` instead of guessing the missing fields.
"""

from __future__ import annotations

import typing

from arena_evaluation.storage.schemas import RobotParams

#: The complete fixture robot: one value for every field of ``RobotParams``.
_FIXTURE_ROBOT: typing.Final[dict[str, typing.Any]] = {
    "model": "unknown",
    "robot_radius": 0.25,
    "laser_min_range": 0.0,
    "laser_max_range": 30.0,
    "base_mass": 0.0,
    "component_masses": {},
}


def make_robot_params(**overrides: typing.Any) -> RobotParams:
    """A complete ``RobotParams`` for tests; each override replaces one field."""
    unknown = set(overrides) - set(_FIXTURE_ROBOT)
    if unknown:
        raise TypeError(
            f"make_robot_params got unknown field(s) {sorted(unknown)}; "
            f"fields are {sorted(_FIXTURE_ROBOT)}"
        )
    return RobotParams(**{**_FIXTURE_ROBOT, **overrides})
