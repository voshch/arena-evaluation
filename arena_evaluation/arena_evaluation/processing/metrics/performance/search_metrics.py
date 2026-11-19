from __future__ import annotations

import json
import logging
import math
import typing

import numpy as np
import polars as pl
from arena_simulation_setup.shared.conditions import SeesAtom
from arena_simulation_setup.shared.judge import Sample, WallGrid, atom_holds

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.compliance_metrics import (
    _reconstruct_events,
)
from arena_evaluation.processing.metrics.ecological.condition_metrics import (
    _ped_aliases,
    bare_robot_name,
    load_wall_grid,
)
from arena_evaluation.processing.metrics.performance.collision_metrics import (
    CollisionMetricsCalculator,
)
from arena_evaluation.processing.metrics.performance.task_metrics import (
    _robot_events,
    _split_indices,
)
from arena_evaluation.processing.path.theta_star import (
    GeometricThetaStar,
    load_map_solver,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

logger = logging.getLogger(__name__)

SEARCH_STATE_TOPIC = "hearing_search_state"
COUNTER_KEYS = ("stops", "listened_s", "goals_sent", "goals_failed", "calls")
FIND_RADII_M = {"1m": 1.0, "1_5m": 1.5, "3m": 3.0}
STOPPED_SPEED_MPS = 0.05

Track = tuple[np.ndarray, np.ndarray, np.ndarray]


def search_phase(phases: dict | None) -> tuple[int, str, float | None] | None:
    """(index, target ped, find radius) of the judge-only goto on a named target, None without one."""
    if not phases:
        return None
    for index, phase in enumerate(phases.get("phases") or []):
        goto = phase.get("goto")
        if isinstance(goto, str) and phase.get("dispatch") is False:
            radius = phase.get("tolerance_radius")
            return index, goto, None if radius is None else float(radius)
    return None


def target_track(peds: pl.DataFrame | None, target: str) -> Track | None:
    """(time_ns, x, y) of the target at each peds message, NaN when absent."""
    if peds is None or len(peds) == 0 or "peds_names" not in peds.columns or "peds_positions" not in peds.columns:
        return None
    xs: list[float] = []
    ys: list[float] = []
    frame = peds.sort("time_ns")
    for names, positions in frame.select("peds_names", "peds_positions").iter_rows():
        for j, name in enumerate(names or []):
            if target in _ped_aliases(name) and 3 * j + 1 < len(positions or []):
                xs.append(float(positions[3 * j]))
                ys.append(float(positions[3 * j + 1]))
                break
        else:
            xs.append(math.nan)
            ys.append(math.nan)
    if not np.isfinite(xs).any():
        return None
    return frame["time_ns"].to_numpy(), np.asarray(xs), np.asarray(ys)


def held_position(track: Track, time_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The track's latest position at or before each instant, NaN before its first sample."""
    t, x, y = track
    idx = np.searchsorted(t, time_ns, side="right") - 1
    safe = np.clip(idx, 0, None)
    return np.where(idx >= 0, x[safe], np.nan), np.where(idx >= 0, y[safe], np.nan)


def met_time_ns(episode: AlignedEpisodeBundle, index: int, start_ns: int) -> tuple[bool, int | None]:
    """(whether `met` is recorded at all, first time the phase reads as met)."""
    series = _robot_events(_reconstruct_events(episode.semantic_snapshot), bare_robot_name(episode.robot_name or ""))
    met = series.get("met")
    if met is None:
        return False, None
    for t, value in zip(met[0].tolist(), met[1], strict=True):
        if t >= start_ns and (episode.window_ns[1] is None or t <= episode.window_ns[1]) and index in _split_indices(value):
            return True, int(t)
    return True, None


def driver_counters(episode: AlignedEpisodeBundle) -> dict[str, float | int | None]:
    """The last hearing_search_state message of the episode, every counter None without the topic."""
    out: dict[str, float | int | None] = dict.fromkeys(COUNTER_KEYS)
    frame = (episode.topics or {}).get(SEARCH_STATE_TOPIC)
    if frame is None or len(frame) == 0 or "state" not in frame.columns:
        return out
    start, end = episode.window_ns
    if start is not None:
        frame = frame.filter(pl.col("time_ns") >= start)
    if end is not None:
        frame = frame.filter(pl.col("time_ns") <= end)
    if frame.is_empty():
        return out
    try:
        state = json.loads(frame.sort("time_ns", maintain_order=True)["state"][-1])
    except (TypeError, ValueError):
        return out
    if not isinstance(state, dict):
        return out
    for key in COUNTER_KEYS:
        if state.get(key) is not None:
            out[key] = float(state[key]) if key == "listened_s" else int(state[key])
    return out


def stopped_seconds(pos_x: np.ndarray, pos_y: np.ndarray, time_ns: np.ndarray, start_ns: int, end_ns: int) -> float:
    """Seconds in [start_ns, end_ns] under 0.05 m/s, finite-differenced from the pose."""
    if len(time_ns) < 2:
        return 0.0
    dt = np.diff(time_ns) / 1e9
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, np.hypot(np.diff(pos_x), np.diff(pos_y)) / dt, np.nan)
    inside = np.clip(np.minimum(time_ns[1:], end_ns) - np.maximum(time_ns[:-1], start_ns), 0, None) / 1e9
    return float(np.sum(inside[speed < STOPPED_SPEED_MPS]))


class SearchMetricsCalculator(BaseMetricCalculator):
    """Find time, find at 1, 1.5 and 3 m, TSR, SPL, search driver counters and stopped time of an episode with a judge-only goto on a pedestrian."""

    NAME = "search_metrics"
    CATEGORY = "performance"
    DEPENDS_ON: typing.ClassVar[list[str]] = ["time_metrics", "path_metrics", "collision_metrics"]
    REQUIRED_TOPICS: typing.ClassVar[list[str | tuple[str, ...]]] = [("task_pose", "tf_gt", "odom")]

    UNITS: typing.ClassVar[dict[str, str]] = {
        "found": "",
        "time_to_find_s": "s",
        "found_1m": "",
        "found_1_5m": "",
        "found_3m": "",
        "time_to_find_1m_s": "s",
        "time_to_find_1_5m_s": "s",
        "time_to_find_3m_s": "s",
        "tsr": "",
        "spl_search": "",
        "stops": "",
        "listened_s": "s",
        "goals_sent": "",
        "goals_failed": "",
        "calls": "",
        "stopped_s": "s",
        "driving_s": "s",
    }

    PRIMARY_OUTPUTS: typing.ClassVar[list[str]] = ["time_to_find_s", "found", "tsr"]
    OUTPUT_DIRECTIONS: typing.ClassVar[dict[str, str]] = {
        "found": "higher",
        "time_to_find_s": "lower",
        "found_1m": "higher",
        "found_1_5m": "higher",
        "found_3m": "higher",
        "time_to_find_1m_s": "lower",
        "time_to_find_1_5m_s": "lower",
        "time_to_find_3m_s": "lower",
        "tsr": "higher",
        "spl_search": "higher",
        "goals_failed": "lower",
        "stopped_s": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return list(cls.UNITS)

    def budget_s(self) -> float:
        """The suite's metrics.timeout_s, the value collision_metrics times out at."""
        return float(self.metrics_config.get("timeout_s", CollisionMetricsCalculator.TIMEOUT_THRESHOLD_S))

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        if search_phase(episode.phases) is None:
            return dict.fromkeys(self.output_keys())
        solver = load_map_solver(episode.map or "", robot_radius=self.robot_params.robot_radius, run_dir=episode.run_dir)
        walls = load_wall_grid(self.world) if self.world else None
        return self.calculate_on_grid(episode, prior_results, solver, walls)

    def calculate_on_grid(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any], solver: GeometricThetaStar | None, walls: WallGrid | None) -> dict[str, typing.Any]:
        """Metrics against a map solver and a wall grid, None where an input is missing."""
        del prior_results
        results: dict[str, typing.Any] = dict.fromkeys(self.output_keys())
        phase = search_phase(episode.phases)
        if phase is None:
            return results
        index, target, radius = phase
        budget = self.budget_s()
        results.update(driver_counters(episode))

        pos_x, pos_y, yaw, time_ns = self._pose(episode)
        start, end = episode.window_ns
        lo = time_ns.searchsorted(start) if start is not None else 0
        hi = time_ns.searchsorted(end, side="right") if end is not None else len(time_ns)
        pos_x, pos_y, yaw, time_ns = (values[lo:hi] for values in (pos_x, pos_y, yaw, time_ns))
        if len(time_ns) == 0:
            return results
        t0 = int(time_ns[0])
        budget_ns = t0 + round(budget * 1e9)
        track = target_track(episode.peds, target)

        recorded, met_ns = met_time_ns(episode, index, t0)
        if recorded:
            found = met_ns is not None and met_ns < budget_ns
            find_s = (met_ns - t0) / 1e9 if found and met_ns is not None else budget
            end_ns = met_ns if found and met_ns is not None else budget_ns
            results["found"] = found
            results["time_to_find_s"] = find_s
            results["tsr"] = float(found) * (1.0 - find_s / budget)
            results["spl_search"] = self._spl(episode, solver, track, pos_x, pos_y, time_ns, met_ns, radius) if found and met_ns is not None else 0.0
            results["stopped_s"] = stopped_seconds(pos_x, pos_y, time_ns, t0, end_ns)
            results["driving_s"] = find_s - results["stopped_s"]

        if track is not None and walls is not None:
            px, py = held_position(track, time_ns)
            rel = (time_ns - t0) / 1e9
            for suffix, find_radius in FIND_RADII_M.items():
                first = self._first_sight(target, find_radius, walls, pos_x, pos_y, yaw, px, py, rel, budget)
                results[f"found_{suffix}"] = first is not None
                results[f"time_to_find_{suffix}_s"] = budget if first is None else first
        return results

    def _pose(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The judge's own poses when recorded, else the ground-truth-first native pose."""
        poses = episode.fleet.get(bare_robot_name(episode.robot_name or ""))
        if poses is None:
            poses = (episode.topics or {}).get("task_pose")
        if poses is not None and len(poses) > 0:
            poses = poses.sort("time_ns")
            return poses["pos_x"].to_numpy(), poses["pos_y"].to_numpy(), poses["yaw"].to_numpy(), poses["time_ns"].to_numpy().astype(np.int64)
        return self.resolve_native_pose(episode)

    @staticmethod
    def _spl(episode: AlignedEpisodeBundle, solver: GeometricThetaStar | None, track: Track | None, pos_x: np.ndarray, pos_y: np.ndarray, time_ns: np.ndarray, find_ns: int, radius: float | None) -> float | None:
        """L0 / max(L, L0) with L the driven path to the find and L0 the Theta* length to the person's position there, less the find radius."""
        if solver is None or track is None:
            return None
        at = int(np.searchsorted(time_ns, find_ns, side="right"))
        driven = float(np.sum(np.hypot(np.diff(pos_x[:at]), np.diff(pos_y[:at]))))
        if 0 < at < len(time_ns):
            fraction = (find_ns - time_ns[at - 1]) / (time_ns[at] - time_ns[at - 1])
            driven += fraction * math.hypot(pos_x[at] - pos_x[at - 1], pos_y[at] - pos_y[at - 1])
        px, py = held_position(track, np.asarray([find_ns], dtype=np.int64))
        if not (math.isfinite(px[0]) and math.isfinite(py[0])):
            return None
        try:
            reference = float(solver.solve((float(pos_x[0]), float(pos_y[0])), (float(px[0]), float(py[0])), map_id=episode.map or "")[1])
        except (ValueError, RuntimeError, OSError) as e:
            logger.debug(f"search_metrics: no Theta* length: {e!r}")
            return None
        if not math.isfinite(reference):
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
