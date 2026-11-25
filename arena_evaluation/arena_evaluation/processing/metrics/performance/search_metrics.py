from __future__ import annotations

import json
import math
import typing

import numpy as np
import polars as pl
from arena_simulation_setup.shared.conditions import SeesAtom
from arena_simulation_setup.shared.judge import Sample, WallGrid, atom_holds
from arena_simulation_setup.shared.task import GoToPhase, TaskPhase

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, Direction
from arena_evaluation.processing.metrics.common.events import reconstruct_events
from arena_evaluation.processing.metrics.common.rosters import ped_aliases
from arena_evaluation.processing.metrics.ecological.condition_metrics import load_wall_grid
from arena_evaluation.processing.metrics.performance.task_metrics import (
    _robot_events,
    _split_indices,
)
from arena_evaluation.processing.path.static_environment import grid_for_map
from arena_evaluation.processing.path.theta_star import StaticGrid, plan_route
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

SEARCH_STATE_TOPIC = "robot_hearing_search_state"
COUNTER_KEYS = ("stops", "listened_s", "goals_sent", "goals_failed", "calls")
FIND_RADII_M = {"1m": 1.0, "1_5m": 1.5, "3m": 3.0}
STOPPED_SPEED_MPS = 0.05
DEFAULT_BUDGET_S = 180.0

Track = tuple[np.ndarray, np.ndarray, np.ndarray]

SEARCH_FOUND = define_metric(
    "search_found",
    pl.Boolean,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Whether the search phase closed as met before the episode budget.",
    null_reason=True,
)
SEARCH_TIME_TO_FIND_S = define_metric(
    "search_time_to_find_s",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="Seconds from the first robot pose to the find, the budget when the person was not found.",
    null_reason=True,
)
SEARCH_FOUND_1M = define_metric(
    "search_found_1m",
    pl.Boolean,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Whether the robot sees the person within 1 m at some moment before the budget, replayed from the recorded paths.",
    null_reason=True,
)
SEARCH_FOUND_1_5M = define_metric(
    "search_found_1_5m",
    pl.Boolean,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Whether the robot sees the person within 1.5 m at some moment before the budget, replayed from the recorded paths.",
    null_reason=True,
)
SEARCH_FOUND_3M = define_metric(
    "search_found_3m",
    pl.Boolean,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Whether the robot sees the person within 3 m at some moment before the budget, replayed from the recorded paths.",
    null_reason=True,
)
SEARCH_TIME_TO_FIND_1M_S = define_metric(
    "search_time_to_find_1m_s",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="Seconds to the first sight of the person within 1 m, the budget when never.",
    null_reason=True,
)
SEARCH_TIME_TO_FIND_1_5M_S = define_metric(
    "search_time_to_find_1_5m_s",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="Seconds to the first sight of the person within 1.5 m, the budget when never.",
    null_reason=True,
)
SEARCH_TIME_TO_FIND_3M_S = define_metric(
    "search_time_to_find_3m_s",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="Seconds to the first sight of the person within 3 m, the budget when never.",
    null_reason=True,
)
SEARCH_TSR = define_metric(
    "search_tsr",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="1",
    description="Time-weighted success: search_found times one minus the find time over the budget, 0 to 1.",
    null_reason=True,
)
SEARCH_SPL = define_metric(
    "search_spl",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="1",
    description="Success weighted by the Theta* length to the person's position at the find, less the find radius, over the driven path to the find, 0 to 1.",
    null_reason=True,
)
SEARCH_STOPS = define_metric(
    "search_stops",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Stops the search driver made, from its last recorded state.",
    null_reason=True,
)
SEARCH_LISTENED_S = define_metric(
    "search_listened_s",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="Seconds the search driver spent listening, from its last recorded state.",
    null_reason=True,
)
SEARCH_GOALS_SENT = define_metric(
    "search_goals_sent",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Goals the search driver sent, from its last recorded state.",
    null_reason=True,
)
SEARCH_GOALS_FAILED = define_metric(
    "search_goals_failed",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Goals the search driver sent that failed, from its last recorded state.",
    null_reason=True,
)
SEARCH_CALLS = define_metric(
    "search_calls",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Calls the search driver made, from its last recorded state.",
    null_reason=True,
)
SEARCH_STOPPED_S = define_metric(
    "search_stopped_s",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="Seconds under 0.05 m/s before the find, from any cause.",
    null_reason=True,
)
SEARCH_DRIVING_S = define_metric(
    "search_driving_s",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="Seconds to the find that the robot was not stopped.",
    null_reason=True,
)


def search_phase(phases: dict | None) -> tuple[int, str, float | None] | None:
    """(index, target ped, find radius) of the judge-only goto on a named target, None without one."""
    if not phases:
        return None
    for index, raw in enumerate(phases.get("phases") or []):
        phase = TaskPhase.parse(raw)
        if isinstance(phase, GoToPhase) and phase.target is not None and not phase.dispatch:
            return index, phase.target, phase.tolerance_radius
    return None


def target_track(peds: pl.DataFrame | None, target: str) -> Track | None:
    """(t_ns, x, y) of the target at each pedestrian frame, NaN where the frame omits it."""
    if peds is None or len(peds) == 0 or not {"t_ns", "ped_name", "x", "y"} <= set(peds.columns):
        return None
    names = [name for name in peds["ped_name"].drop_nulls().unique() if target in ped_aliases(name)]
    seen = peds.filter(pl.col("ped_name").is_in(names)).group_by("t_ns").agg(pl.col("x").first(), pl.col("y").first())
    frame = peds.select("t_ns").unique().sort("t_ns").join(seen, on="t_ns", how="left")
    xs = frame["x"].fill_null(math.nan).to_numpy()
    ys = frame["y"].fill_null(math.nan).to_numpy()
    if not np.isfinite(xs).any():
        return None
    return frame["t_ns"].to_numpy(), xs, ys


def held_position(track: Track, t_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The track's latest position at or before each instant, NaN before its first sample."""
    t, x, y = track
    idx = np.searchsorted(t, t_ns, side="right") - 1
    safe = np.clip(idx, 0, None)
    return np.where(idx >= 0, x[safe], np.nan), np.where(idx >= 0, y[safe], np.nan)


def met_time_ns(episode: AlignedEpisodeBundle, index: int, start_ns: int) -> tuple[bool, int | None]:
    """(whether `met` is recorded at all, first time the phase reads as met)."""
    series = _robot_events(reconstruct_events(episode.semantic_snapshot), episode.robot_name or "")
    met = series.get("met")
    if met is None:
        return False, None
    for t, value in zip(met[0].tolist(), met[1], strict=True):
        if t >= start_ns and (episode.window_ns[1] is None or t <= episode.window_ns[1]) and index in _split_indices(value):
            return True, int(t)
    return True, None


def driver_counters(episode: AlignedEpisodeBundle) -> dict[str, float | int | None]:
    """The last search state message of the episode, every counter None without one."""
    out: dict[str, float | int | None] = dict.fromkeys(f"search_{key}" for key in COUNTER_KEYS)
    frame = (episode.topics or {}).get(SEARCH_STATE_TOPIC)
    if frame is None or len(frame) == 0:
        return out
    start, end = episode.window_ns
    if start is not None:
        frame = frame.filter(pl.col("t_ns") >= start)
    if end is not None:
        frame = frame.filter(pl.col("t_ns") <= end)
    if frame.is_empty():
        return out
    try:
        state = json.loads(frame.sort("t_ns", maintain_order=True)["state"][-1])
    except (TypeError, ValueError):
        return out
    if not isinstance(state, dict):
        return out
    for key in COUNTER_KEYS:
        if state.get(key) is not None:
            out[f"search_{key}"] = float(state[key]) if key == "listened_s" else int(state[key])
    return out


def stopped_seconds(pos_x: np.ndarray, pos_y: np.ndarray, t_ns: np.ndarray, start_ns: int, end_ns: int) -> float:
    """Seconds in [start_ns, end_ns] under 0.05 m/s, finite-differenced from the pose."""
    if len(t_ns) < 2:
        return 0.0
    dt = np.diff(t_ns) / 1e9
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, np.hypot(np.diff(pos_x), np.diff(pos_y)) / dt, np.nan)
    inside = np.clip(np.minimum(t_ns[1:], end_ns) - np.maximum(t_ns[:-1], start_ns), 0, None) / 1e9
    return float(np.sum(inside[speed < STOPPED_SPEED_MPS]))


class SearchMetricsCalculator(BaseMetricCalculator):
    """Find time, find at 1, 1.5 and 3 m, TSR, SPL, search driver counters and stopped time of an episode with a judge-only goto on a pedestrian."""

    NAME = "search_metrics"
    CATEGORY = "performance"
    REQUIRED_TOPICS = [("robot_task_pose", "robot_tracks", "robot_odom")]

    OUTPUTS = (
        SEARCH_FOUND,
        SEARCH_TIME_TO_FIND_S,
        SEARCH_FOUND_1M,
        SEARCH_FOUND_1_5M,
        SEARCH_FOUND_3M,
        SEARCH_TIME_TO_FIND_1M_S,
        SEARCH_TIME_TO_FIND_1_5M_S,
        SEARCH_TIME_TO_FIND_3M_S,
        SEARCH_TSR,
        SEARCH_SPL,
        SEARCH_STOPS,
        SEARCH_LISTENED_S,
        SEARCH_GOALS_SENT,
        SEARCH_GOALS_FAILED,
        SEARCH_CALLS,
        SEARCH_STOPPED_S,
        SEARCH_DRIVING_S,
    )
    DIRECTIONS = {
        "search_found": Direction.HIGHER,
        "search_time_to_find_s": Direction.LOWER,
        "search_found_1m": Direction.HIGHER,
        "search_found_1_5m": Direction.HIGHER,
        "search_found_3m": Direction.HIGHER,
        "search_time_to_find_1m_s": Direction.LOWER,
        "search_time_to_find_1_5m_s": Direction.LOWER,
        "search_time_to_find_3m_s": Direction.LOWER,
        "search_tsr": Direction.HIGHER,
        "search_spl": Direction.HIGHER,
        "search_goals_failed": Direction.LOWER,
        "search_stopped_s": Direction.LOWER,
    }
    PRIMARY_OUTPUTS = ["search_time_to_find_s", "search_found", "search_tsr"]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        if search_phase(episode.phases) is None:
            return dict.fromkeys(self.output_keys())
        try:
            grid = grid_for_map(episode.map, episode.run_dir, self.robot_params.robot_radius)
        except FileNotFoundError:
            grid = None
        walls = load_wall_grid(self.world) if self.world else None
        return self.calculate_on_grid(episode, prior_results, grid, walls)

    def calculate_on_grid(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any], grid: StaticGrid | None, walls: WallGrid | None) -> dict[str, typing.Any]:
        """Metrics against a static grid and a wall grid, None where an input is missing."""
        del prior_results
        results: dict[str, typing.Any] = dict.fromkeys(self.output_keys())
        phase = search_phase(episode.phases)
        if phase is None:
            return results
        index, target, radius = phase
        budget = DEFAULT_BUDGET_S if episode.timeout_s is None else episode.timeout_s
        results.update(driver_counters(episode))

        pos_x, pos_y, yaw, t_ns = self._pose(episode)
        start, end = episode.window_ns
        lo = t_ns.searchsorted(start) if start is not None else 0
        hi = t_ns.searchsorted(end, side="right") if end is not None else len(t_ns)
        pos_x, pos_y, yaw, t_ns = (values[lo:hi] for values in (pos_x, pos_y, yaw, t_ns))
        if len(t_ns) == 0:
            return results
        t0 = int(t_ns[0])
        budget_ns = t0 + round(budget * 1e9)
        track = target_track(episode.peds, target)

        recorded, met_ns = met_time_ns(episode, index, t0)
        if recorded:
            found = met_ns is not None and met_ns < budget_ns
            find_s = (met_ns - t0) / 1e9 if found and met_ns is not None else budget
            end_ns = met_ns if found and met_ns is not None else budget_ns
            results["search_found"] = found
            results["search_time_to_find_s"] = find_s
            results["search_tsr"] = float(found) * (1.0 - find_s / budget)
            results["search_spl"] = self._spl(grid, track, pos_x, pos_y, t_ns, met_ns, radius) if found and met_ns is not None else 0.0
            results["search_stopped_s"] = stopped_seconds(pos_x, pos_y, t_ns, t0, end_ns)
            results["search_driving_s"] = find_s - results["search_stopped_s"]

        if track is not None and walls is not None:
            px, py = held_position(track, t_ns)
            rel = (t_ns - t0) / 1e9
            for suffix, find_radius in FIND_RADII_M.items():
                first = self._first_sight(target, find_radius, walls, pos_x, pos_y, yaw, px, py, rel, budget)
                results[f"search_found_{suffix}"] = first is not None
                results[f"search_time_to_find_{suffix}_s"] = budget if first is None else first
        return results

    def _pose(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The judge's own poses when recorded, else the ground-truth native pose."""
        poses = self.native_topics(episode).get("robot_task_pose")
        if poses is not None and len(poses) > 0:
            poses = poses.sort("t_ns")
            return poses["pos_x"].to_numpy(), poses["pos_y"].to_numpy(), poses["yaw"].to_numpy(), poses["t_ns"].to_numpy().astype(np.int64)
        return self.resolve_native_pose(episode)

    @staticmethod
    def _spl(grid: StaticGrid | None, track: Track | None, pos_x: np.ndarray, pos_y: np.ndarray, t_ns: np.ndarray, find_ns: int, radius: float | None) -> float | None:
        """L0 / max(L, L0) with L the driven path to the find and L0 the Theta* length to the person's position there, less the find radius."""
        if grid is None or track is None:
            return None
        at = int(np.searchsorted(t_ns, find_ns, side="right"))
        driven = float(np.sum(np.hypot(np.diff(pos_x[:at]), np.diff(pos_y[:at]))))
        if 0 < at < len(t_ns):
            fraction = (find_ns - t_ns[at - 1]) / (t_ns[at] - t_ns[at - 1])
            driven += fraction * math.hypot(pos_x[at] - pos_x[at - 1], pos_y[at] - pos_y[at - 1])
        px, py = held_position(track, np.asarray([find_ns], dtype=np.int64))
        if not (math.isfinite(px[0]) and math.isfinite(py[0])):
            return None
        reference = plan_route(grid, (float(pos_x[0]), float(pos_y[0])), (float(px[0]), float(py[0]))).length_m
        if reference is None or not math.isfinite(reference):
            return None
        l0 = max(reference - (radius or 0.0), 0.0)
        denom = max(driven, l0)
        return min(l0 / denom, 1.0) if denom > 0 else 1.0

    @staticmethod
    def _first_sight(target: str, radius: float, walls: WallGrid, pos_x: np.ndarray, pos_y: np.ndarray, yaw: np.ndarray, px: np.ndarray, py: np.ndarray, rel: np.ndarray, budget: float) -> float | None:
        """First time before the budget the shared judge holds `robot sees <target> within <radius>`."""
        atom = SeesAtom(subject="robot", other=target, range=radius)
        with np.errstate(invalid="ignore"):
            near = np.flatnonzero((rel < budget) & (np.hypot(pos_x - px, pos_y - py) <= radius))
        for i in near.tolist():
            sample = Sample(t=float(rel[i]), robots={"robot": (float(pos_x[i]), float(pos_y[i]), float(yaw[i]))}, peds={target: (float(px[i]), float(py[i]))}, walls=walls)
            if atom_holds(atom, sample, {}) is True:
                return float(rel[i])
        return None
