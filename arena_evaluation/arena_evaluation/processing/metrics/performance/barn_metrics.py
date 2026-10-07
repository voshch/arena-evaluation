from __future__ import annotations

import math
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

MAX_SPEED_MPS = 2.0
GOAL_RADIUS_M = 1.0
START_RADIUS_M = 0.1
TIMEOUT_S = 100.0
POLL_S = 0.1
CLIP_LOW = 2.0
CLIP_HIGH = 8.0

_TIMEOUT_POLLS = round(TIMEOUT_S / POLL_S)


class Run(typing.NamedTuple):
    success: bool
    collided: bool
    timeout: bool
    time: float | None


def _point(raw: dict, key: str) -> list[float]:
    value = raw[key]
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        raise ValueError(f"barn: {key} must be [x, y] in meters, got {value!r}")
    return [float(value[0]), float(value[1])]


def parse_reference(raw: object) -> dict[str, typing.Any]:
    """Validated suite `metrics.barn` block: start, goal and the reference path length per map."""
    if not isinstance(raw, dict) or set(raw) != {"start", "goal", "path_length"}:
        raise ValueError(f"barn: needs exactly start, goal and path_length, got {sorted(raw) if isinstance(raw, dict) else type(raw).__name__}")
    lengths = raw["path_length"]
    if not isinstance(lengths, dict) or not lengths:
        raise ValueError(f"barn: path_length must map each stage map to meters, got {lengths!r}")
    path_length: dict[str, float] = {}
    for world, length in lengths.items():
        if not isinstance(length, (int, float)) or isinstance(length, bool) or not math.isfinite(length) or length <= 0:
            raise ValueError(f"barn: path_length of {world!r} must be a positive number of meters, got {length!r}")
        path_length[str(world)] = float(length)
    return {"start": _point(raw, "start"), "goal": _point(raw, "goal"), "path_length": path_length}


def score(success: bool, actual_time: float, path_length: float) -> float:
    """Navigation metric of run.py: 1_success * optimal_time / clip(actual_time, 2 * optimal_time, 8 * optimal_time)."""
    optimal_time = path_length / MAX_SPEED_MPS
    return float(int(success) * optimal_time / np.clip(actual_time, CLIP_LOW * optimal_time, CLIP_HIGH * optimal_time))


def replay(t: np.ndarray, x: np.ndarray, y: np.ndarray, start: typing.Sequence[float], goal: typing.Sequence[float], collision_t: float | None) -> Run:
    """Outcome of a recorded run under the polling loop of run.py, the last pose held for the poll after the recording ends."""
    moved = np.nonzero(np.hypot(x - start[0], y - start[1]) >= START_RADIUS_M)[0]
    if len(moved) == 0:
        return Run(False, False, False, None)
    t0 = float(t[moved[0]])
    end = float(t[-1])
    polls = np.arange(min(int((end - t0) // POLL_S) + 2, _TIMEOUT_POLLS + 1))
    at = t0 + POLL_S * polls
    reached = np.hypot(np.interp(at, t, x) - goal[0], np.interp(at, t, y) - goal[1]) <= GOAL_RADIUS_M
    collided = at >= min(collision_t, end) if collision_t is not None else np.zeros(len(polls), dtype=bool)
    timeout = polls >= _TIMEOUT_POLLS
    done = np.nonzero(reached | collided | timeout)[0]
    if len(done) == 0:
        return Run(False, False, False, end - t0)
    poll = int(done[0])
    if collided[poll]:
        return Run(False, True, False, POLL_S * poll)
    if timeout[poll]:
        return Run(False, False, True, POLL_S * poll)
    return Run(True, False, False, POLL_S * poll)


def _first_collision_s(episode: AlignedEpisodeBundle) -> float | None:
    frame = (episode.topics or {}).get("collision_events")
    if frame is None or "collision_event" not in frame.columns:
        return None
    hits = frame.filter(pl.col("collision_event") > 0)
    return float(hits.select(pl.col("time_ns").min()).item()) / 1e9 if len(hits) else None


class BARNScoreCalculator(BaseMetricCalculator):
    """BARN challenge score of an episode whose map has a reference path in the suite's `metrics.barn`."""

    NAME = "barn_score"
    CATEGORY = "performance"
    REQUIRED_TOPICS = ["collision_events", ("tf_gt", "odom")]

    UNITS = {
        "barn_score": "",
        "barn_optimal_time": "s",
        "barn_time": "s",
        "barn_success": "",
        "barn_collided": "",
        "barn_timeout": "",
    }

    PRIMARY_OUTPUTS = ["barn_score"]
    OUTPUT_DIRECTIONS = {"barn_score": "higher", "barn_success": "higher"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "barn_score",
            "barn_optimal_time",
            "barn_time",
            "barn_success",
            "barn_collided",
            "barn_timeout",
        ]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        nulls = dict.fromkeys(self.output_keys())
        reference = self.metrics_config.get("barn")
        if reference is None or episode.map not in reference["path_length"]:
            return nulls
        x, y, _, time_ns = self.resolve_native_pose(episode)
        if len(time_ns) == 0:
            return nulls

        path_length = reference["path_length"][episode.map]
        run = replay(time_ns / 1e9, x, y, reference["start"], reference["goal"], _first_collision_s(episode))
        return {
            "barn_score": 0.0 if run.time is None else score(run.success, run.time, path_length),
            "barn_optimal_time": path_length / MAX_SPEED_MPS,
            "barn_time": run.time,
            "barn_success": run.success,
            "barn_collided": run.collided,
            "barn_timeout": run.timeout,
        }
