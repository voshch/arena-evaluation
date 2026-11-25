from __future__ import annotations

import re
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.common.alignment import asof_index

_ROBOT_DIR_PREFIX: typing.Final[re.Pattern[str]] = re.compile(r"^env_\d+_")


def _bare_robot_name(robot_dir: str) -> str:
    """`env_0_jackal_0` -> `jackal_0`, the name phases and atoms use."""
    return _ROBOT_DIR_PREFIX.sub("", robot_dir, count=1)


def resample_fleet(fleet: dict[str, pl.DataFrame], t_ns: np.ndarray, ego: str | None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Other robots' judge poses held onto the ego axis, keyed by the bare name the judge atoms use."""
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, frame in fleet.items():
        if name == ego or frame is None or len(frame) == 0:
            continue
        frame = frame.sort("t_ns")
        times = frame["t_ns"].to_numpy()
        idx = asof_index(times, t_ns)
        if np.any(idx < 0):
            continue
        out[_bare_robot_name(name)] = (
            frame["pos_x"].to_numpy()[idx],
            frame["pos_y"].to_numpy()[idx],
            frame["yaw"].to_numpy()[idx],
        )
    return out
