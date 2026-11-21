"""Per-encounter costs against planned routes."""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import polars as pl

from arena_evaluation.processing.alignment import interpolate_values
from arena_evaluation.processing.metrics.base import (
    NULL_REASON_SUFFIX,
    BaseMetricCalculator,
    Direction,
)
from arena_evaluation.processing.metrics.ecological.energy import JOULES_PER_WH
from arena_evaluation.processing.metrics.social.interaction_common import (
    NS_PER_S,
    PEDESTRIAN_ARENA_PEDS_TOPIC,
    ROBOT_TRACKS_TOPIC,
    PedTrack,
    Trajectory2D,
    detector_params_from_config,
    extract_episode_ped_tracks,
    pedestrian_encounter_spans,
    resolve_ped_track,
    resolve_robot_trajectory,
)
from arena_evaluation.processing.path.static_environment import grid_for_map
from arena_evaluation.processing.path.theta_star import (
    RouteResult,
    StaticGrid,
    plan_route,
)
from arena_evaluation.storage.registry import MetricScope, NullReason, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

INTIMATE_ZONE_M: Final[float] = 0.45
_DEGENERATE_SEGMENT_M2: Final[float] = 1e-12

DETOUR_TIME = define_metric(
    "detour_time_s",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="s",
    description="Pedestrian detour time in the encounter window: window duration minus the time its planned route takes at its free speed.",
    null_reason=True,
)
PED_PATH_EXCESS = define_metric(
    "ped_path_excess_m",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="m",
    description="Pedestrian path length in the encounter window minus its planned route length.",
    null_reason=True,
)
ROBOT_PATH_EXCESS = define_metric(
    "robot_path_excess_m",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="m",
    description="Robot path length in the encounter window minus its planned route length.",
    null_reason=True,
)
ROBOT_PATH_DEVIATION = define_metric(
    "robot_path_deviation_vs_planned_m",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="m",
    description="Largest distance from the robot's path in the encounter window to its planned route.",
    null_reason=True,
)
MAR = define_metric(
    "mar_vs_planned_route",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="1",
    description="Mutual accommodation ratio: robot path excess over the sum of the robot and pedestrian path excesses, each floored at zero. Higher is better.",
    null_reason=True,
)
PFI = define_metric(
    "pfi_vs_planned_route",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="m/s",
    description="Pedestrian frustration index: mean pedestrian deceleration over the window times its dwell in the intimate zone.",
    null_reason=True,
)
KCSC_WH = define_metric(
    "kcsc_wh_vs_planned_route",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="Wh",
    description="Kinematic cost of social compliance: robot energy in the encounter window minus the energy of its planned route, floored at zero.",
    null_reason=True,
)
ECSI_J = define_metric(
    "ecsi_j",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="J",
    description="Energy cost of social interaction: robot energy in the encounter window minus the energy of its planned route in Joules.",
    null_reason=True,
)
ENERGY_PER_SOCIAL_INTERACTION_J = define_metric(
    "energy_per_social_interaction_j",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="J",
    description="Total electrical energy in Joules consumed by the robot during this encounter window.",
    null_reason=True,
)
PSYCHOACOUSTIC_YIELDING_COST = define_metric(
    "psychoacoustic_yielding_cost",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="m*s",
    description="Psychoacoustic yielding cost: integral of pedestrian deviation from planned route weighted by log10 barrier-aware acoustic level.",
    null_reason=True,
)


def mutual_accommodation_ratio(robot_excess_m: float, pedestrian_excess_m: float) -> float | None:
    """Robot excess over the summed excesses, each floored at zero; None when neither agent deviated."""
    robot = max(robot_excess_m, 0.0)
    pedestrian = max(pedestrian_excess_m, 0.0)
    total = robot + pedestrian
    return robot / total if total > 0.0 else None


def orthogonal_distances_m(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Point-to-polyline Euclidean distance per point: perpendicular to a segment, clipped to ends."""
    starts = polyline[:-1]
    deltas = polyline[1:] - starts
    length_sq = np.einsum("ij,ij->i", deltas, deltas)
    usable = length_sq > _DEGENERATE_SEGMENT_M2
    if not np.any(usable):
        return np.linalg.norm(points - polyline[0], axis=1)
    starts, deltas, length_sq = starts[usable], deltas[usable], length_sq[usable]
    offsets = points[:, None, :] - starts[None, :, :]
    fraction = np.clip(np.einsum("nmk,mk->nm", offsets, deltas) / length_sq, 0.0, 1.0)
    projected = starts[None, :, :] + fraction[..., None] * deltas[None, :, :]
    return np.linalg.norm(points[:, None, :] - projected, axis=2).min(axis=1)


def max_orthogonal_deviation_m(points: np.ndarray, polyline: np.ndarray) -> float:
    """Largest distance of any point to a polyline: perpendicular to a segment, clipped to its ends."""
    return float(np.max(orthogonal_distances_m(points, polyline)))


def _polyline_length(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def _outside(t_ns: np.ndarray, spans: Sequence[tuple[int, int]]) -> np.ndarray:
    inside = np.zeros(len(t_ns), dtype=bool)
    for start_ns, end_ns in spans:
        inside |= (t_ns >= start_ns) & (t_ns <= end_ns)
    return ~inside


def _pair_speeds(t_ns: np.ndarray, x: np.ndarray, y: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Speeds (m/s) of consecutive sample pairs whose two samples are both free."""
    dt_s = np.diff(t_ns).astype(np.float64) / NS_PER_S
    distance_m = np.hypot(np.diff(x), np.diff(y))
    both_free = free[:-1] & free[1:] & (dt_s > 0.0)
    return distance_m[both_free] / dt_s[both_free]


def _pair_energy_wh(t_ns: np.ndarray, power_w: np.ndarray, pairs: np.ndarray) -> float:
    """Energy (Wh) of the selected consecutive pairs, each pair's left power times its duration."""
    dt_s = np.diff(t_ns).astype(np.float64) / NS_PER_S
    return float(np.sum(power_w[:-1][pairs] * dt_s[pairs]) / JOULES_PER_WH)


def _power_samples(
    episode: AlignedEpisodeBundle,
) -> tuple[np.ndarray, np.ndarray] | None:
    table = (episode.topics or {}).get("robot_power")
    if table is None or table.height == 0:
        return None
    ordered = table.sort("t_ns")
    return ordered["t_ns"].to_numpy().astype(np.float64), ordered["total_power_w"].to_numpy().astype(np.float64)


def _speed_between(trajectory: Trajectory2D, first: int, second: int) -> float:
    dt_s = float(trajectory.t_ns[second] - trajectory.t_ns[first]) / NS_PER_S
    distance_m = float(
        np.hypot(
            trajectory.x[second] - trajectory.x[first],
            trajectory.y[second] - trajectory.y[first],
        )
    )
    return distance_m / dt_s


def _deceleration_mps2(pedestrian_window: Trajectory2D) -> float | None:
    """Speed lost between the window's entry and exit, per second of window."""
    if len(pedestrian_window.t_ns) < 3:
        return None
    duration_s = float(pedestrian_window.t_ns[-1] - pedestrian_window.t_ns[0]) / NS_PER_S
    if duration_s <= 0.0:
        return None
    entry_speed = _speed_between(pedestrian_window, 0, 1)
    exit_speed = _speed_between(pedestrian_window, -2, -1)
    return max(0.0, entry_speed - exit_speed) / duration_s


def _intimate_dwell_s(robot_window: Trajectory2D, track: PedTrack, robot_radius_m: float) -> float | None:
    if track.radius_m is None or not np.isfinite(track.radius_m):
        return None
    on_robot_times = track.trajectory.resample_onto(robot_window.t_ns)
    clearance_m = np.hypot(on_robot_times.x - robot_window.x, on_robot_times.y - robot_window.y) - (robot_radius_m + float(track.radius_m))
    dt_s = np.diff(robot_window.t_ns).astype(np.float64) / NS_PER_S
    return float(np.sum(dt_s[clearance_m[:-1] < INTIMATE_ZONE_M]))


def _planned(grid: StaticGrid | None, start_xy: np.ndarray, goal_xy: np.ndarray) -> RouteResult:
    if grid is None:
        return RouteResult(points=None, length_m=None, null_reason="no_map")
    return plan_route(
        grid,
        (float(start_xy[0]), float(start_xy[1])),
        (float(goal_xy[0]), float(goal_xy[1])),
    )


def _refusal(route: RouteResult) -> str:
    """Closed-vocabulary reason for a missing planned-route reference.

    A missing recorded parameter stays distinct from a route the planner could not find; neither makes
    the episode unrankable, so an unregistered ``invalid_episode:<why>`` is never written.
    """
    if route.null_reason == NullReason.NOT_RECORDED.value:
        return NullReason.NOT_RECORDED.value
    return NullReason.REFERENCE_MISSING.value


def _encounter_outputs(
    *,
    track: PedTrack,
    window_ns: tuple[int, int],
    robot: Trajectory2D,
    spans: Sequence[tuple[str, int, int]],
    map_name: str | None,
    run_dir: pathlib.Path | None,
    robot_radius_m: float,
    power: tuple[np.ndarray, np.ndarray] | None,
    acoustic_exposure: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    t_start_ns, t_end_ns = window_ns
    robot_window = robot.slice_ns(t_start_ns, t_end_ns)
    pedestrian_window = track.trajectory.slice_ns(t_start_ns, t_end_ns)
    outputs: dict[str, Any] = {metric.name: None for metric in EncounterCostsCalculator.OUTPUTS}
    reasons: dict[str, str] = {}
    if len(robot_window.t_ns) < 2 or len(pedestrian_window.t_ns) < 2:
        reasons = {name: NullReason.NO_SAMPLES.value for name in outputs}
        return _with_reasons(outputs, reasons)

    robot_route = _planned(
        grid_for_map(map_name, run_dir, robot_radius_m),
        robot_window.xy[0],
        robot_window.xy[-1],
    )
    if track.radius_m is None or not np.isfinite(track.radius_m):
        ped_route = RouteResult(points=None, length_m=None, null_reason=NullReason.NOT_RECORDED.value)
    else:
        ped_route = _planned(
            grid_for_map(map_name, run_dir, float(track.radius_m)),
            pedestrian_window.xy[0],
            pedestrian_window.xy[-1],
        )

    robot_excess_m: float | None = None
    if robot_route.points is not None and robot_route.length_m is not None:
        robot_excess_m = _polyline_length(robot_window.x, robot_window.y) - robot_route.length_m
        outputs["robot_path_excess_m"] = robot_excess_m
        outputs["robot_path_deviation_vs_planned_m"] = max_orthogonal_deviation_m(robot_window.xy, robot_route.points)
    else:
        reasons["robot_path_excess_m"] = _refusal(robot_route)
        reasons["robot_path_deviation_vs_planned_m"] = _refusal(robot_route)

    ped_excess_m: float | None = None
    if ped_route.points is not None and ped_route.length_m is not None:
        ped_excess_m = _polyline_length(pedestrian_window.x, pedestrian_window.y) - ped_route.length_m
        outputs["ped_path_excess_m"] = ped_excess_m
    else:
        reasons["ped_path_excess_m"] = _refusal(ped_route)

    own_spans = [(start_ns, end_ns) for ped_id, start_ns, end_ns in spans if str(ped_id) == str(track.ped_id)]
    free_speeds = _pair_speeds(track.t_ns, track.x, track.y, _outside(track.t_ns, own_spans))
    free_speed_mps = float(np.median(free_speeds)) if free_speeds.size else 0.0
    duration_s = (t_end_ns - t_start_ns) / NS_PER_S
    if ped_route.length_m is None:
        reasons["detour_time_s"] = _refusal(ped_route)
    elif free_speed_mps <= 0.0:
        reasons["detour_time_s"] = NullReason.NO_SAMPLES.value
    else:
        outputs["detour_time_s"] = duration_s - ped_route.length_m / free_speed_mps

    _populate_interaction_costs(
        outputs=outputs,
        reasons=reasons,
        robot=robot,
        robot_window=robot_window,
        pedestrian_window=pedestrian_window,
        track=track,
        spans=spans,
        window_ns=(t_start_ns, t_end_ns),
        robot_radius_m=robot_radius_m,
        robot_route=robot_route,
        ped_route=ped_route,
        robot_excess_m=robot_excess_m,
        ped_excess_m=ped_excess_m,
        power=power,
        acoustic_exposure=acoustic_exposure,
    )
    return _with_reasons(outputs, reasons)


def _psychoacoustic_yielding_cost(
    pedestrian_window: Trajectory2D,
    ped_route: RouteResult,
    acoustic_exposure: tuple[np.ndarray, np.ndarray] | None,
    reasons: dict[str, str],
) -> float | None:
    """Integral of pedestrian path deviation weighted by log10 barrier-aware acoustic level."""
    if ped_route.points is None:
        reasons["psychoacoustic_yielding_cost"] = _refusal(ped_route)
        return None
    if acoustic_exposure is None:
        reasons["psychoacoustic_yielding_cost"] = f"{NullReason.TOPIC_MISSING.value}:acoustic_exposure"
        return None
    if len(pedestrian_window.t_ns) < 2:
        reasons["psychoacoustic_yielding_cost"] = NullReason.NO_SAMPLES.value
        return None
    ac_t_ns, ac_dba = acoustic_exposure
    if ac_t_ns.size == 0:
        reasons["psychoacoustic_yielding_cost"] = NullReason.NO_SAMPLES.value
        return None
    levels = interpolate_values(ac_t_ns, ac_dba, pedestrian_window.t_ns.astype(np.float64))
    deviations = orthogonal_distances_m(pedestrian_window.xy, ped_route.points)
    dt_s = np.diff(pedestrian_window.t_ns).astype(np.float64) / NS_PER_S
    weights = np.log10(np.maximum(levels[:-1], 1.0))
    return float(np.sum(deviations[:-1] * weights * dt_s))


def _populate_interaction_costs(
    *,
    outputs: dict[str, Any],
    reasons: dict[str, str],
    robot: Trajectory2D,
    robot_window: Trajectory2D,
    pedestrian_window: Trajectory2D,
    track: PedTrack,
    spans: Sequence[tuple[str, int, int]],
    window_ns: tuple[int, int],
    robot_radius_m: float,
    robot_route: RouteResult,
    ped_route: RouteResult,
    robot_excess_m: float | None,
    ped_excess_m: float | None,
    power: tuple[np.ndarray, np.ndarray] | None,
    acoustic_exposure: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    if robot_excess_m is not None and ped_excess_m is not None:
        mar = mutual_accommodation_ratio(robot_excess_m, ped_excess_m)
        outputs["mar_vs_planned_route"] = mar
        if mar is None:
            reasons["mar_vs_planned_route"] = NullReason.ZERO_DENOMINATOR.value
    else:
        reasons["mar_vs_planned_route"] = reasons.get("ped_path_excess_m") or reasons.get("robot_path_excess_m") or NullReason.NO_SAMPLES.value

    dwell_s = _intimate_dwell_s(robot_window, track, robot_radius_m)
    deceleration = _deceleration_mps2(pedestrian_window)
    if dwell_s is None or deceleration is None:
        reasons["pfi_vs_planned_route"] = NullReason.NO_SAMPLES.value
    else:
        outputs["pfi_vs_planned_route"] = deceleration * dwell_s

    if power is None:
        missing_power = f"{NullReason.TOPIC_MISSING.value}:robot_power"
        reasons["kcsc_wh_vs_planned_route"] = missing_power
        reasons["ecsi_j"] = missing_power
        reasons["energy_per_social_interaction_j"] = missing_power
    else:
        window_energy = _window_energy_wh(robot, power, window_ns)
        if window_energy is None:
            reasons["energy_per_social_interaction_j"] = NullReason.NO_SAMPLES.value
        else:
            outputs["energy_per_social_interaction_j"] = window_energy * JOULES_PER_WH

        if robot_route.length_m is None:
            refusal = _refusal(robot_route)
            reasons["kcsc_wh_vs_planned_route"] = refusal
            reasons["ecsi_j"] = refusal
        else:
            delta_wh = _robot_excess_energy_wh(
                robot,
                robot_window,
                spans,
                power,
                window_ns,
                robot_route.length_m,
                reasons,
            )
            if delta_wh is not None:
                outputs["kcsc_wh_vs_planned_route"] = max(delta_wh, 0.0)
                outputs["ecsi_j"] = delta_wh * JOULES_PER_WH
            else:
                reasons["ecsi_j"] = reasons.get("kcsc_wh_vs_planned_route", NullReason.NO_SAMPLES.value)

    outputs["psychoacoustic_yielding_cost"] = _psychoacoustic_yielding_cost(pedestrian_window, ped_route, acoustic_exposure, reasons)


def _window_energy_wh(
    robot: Trajectory2D,
    power: tuple[np.ndarray, np.ndarray],
    window_ns: tuple[int, int],
) -> float | None:
    """Total electrical energy (Wh) in the encounter window."""
    power_t_ns, power_values = power
    window_pairs = (robot.t_ns[:-1] >= window_ns[0]) & (robot.t_ns[1:] <= window_ns[1])
    if not np.any(window_pairs):
        return None
    used_t_ns = np.concatenate([robot.t_ns[:-1][window_pairs], robot.t_ns[1:][window_pairs]])
    if used_t_ns.min() < power_t_ns[0] or used_t_ns.max() > power_t_ns[-1]:
        return None
    robot_power_w = interpolate_values(power_t_ns, power_values, robot.t_ns.astype(np.float64))
    return _pair_energy_wh(robot.t_ns, robot_power_w, window_pairs)


def _robot_excess_energy_wh(
    robot: Trajectory2D,
    robot_window: Trajectory2D,
    spans: Sequence[tuple[str, int, int]],
    power: tuple[np.ndarray, np.ndarray],
    window_ns: tuple[int, int],
    planned_length_m: float,
    reasons: dict[str, str],
) -> float | None:
    del robot_window
    power_t_ns, power_values = power
    free = _outside(robot.t_ns, [(start_ns, end_ns) for _, start_ns, end_ns in spans])
    free_pairs = free[:-1] & free[1:]
    window_pairs = (robot.t_ns[:-1] >= window_ns[0]) & (robot.t_ns[1:] <= window_ns[1])
    used_pairs = free_pairs | window_pairs
    used_t_ns = np.concatenate([robot.t_ns[:-1][used_pairs], robot.t_ns[1:][used_pairs]])
    if used_t_ns.size and (used_t_ns.min() < power_t_ns[0] or used_t_ns.max() > power_t_ns[-1]):
        reasons["kcsc_wh_vs_planned_route"] = NullReason.NO_SAMPLES.value
        return None
    robot_power_w = interpolate_values(power_t_ns, power_values, robot.t_ns.astype(np.float64))
    free_distance_m = float(np.sum(np.hypot(np.diff(robot.x), np.diff(robot.y))[free_pairs]))
    if free_distance_m <= 0.0:
        reasons["kcsc_wh_vs_planned_route"] = NullReason.NO_SAMPLES.value
        return None
    energy_per_m_wh = _pair_energy_wh(robot.t_ns, robot_power_w, free_pairs) / free_distance_m
    window_energy_wh = _pair_energy_wh(robot.t_ns, robot_power_w, window_pairs)
    return window_energy_wh - energy_per_m_wh * planned_length_m


def _with_reasons(outputs: dict[str, Any], reasons: dict[str, str]) -> dict[str, Any]:
    for name, reason in reasons.items():
        if outputs.get(name) is None:
            outputs[f"{name}{NULL_REASON_SUFFIX}"] = reason
    return outputs


class EncounterCostsCalculator(BaseMetricCalculator):
    """Per-encounter costs of the robot and the pedestrian, each against its planned route on the static grid."""

    NAME = "encounter_costs"
    CATEGORY = "social"
    REQUIRED_TOPICS = [ROBOT_TRACKS_TOPIC, PEDESTRIAN_ARENA_PEDS_TOPIC]
    PARAMS = (
        "encounter_edge_distance_m",
        "encounter_merge_gap_s",
    )
    OUTPUTS = (
        DETOUR_TIME,
        PED_PATH_EXCESS,
        ROBOT_PATH_EXCESS,
        ROBOT_PATH_DEVIATION,
        MAR,
        PFI,
        KCSC_WH,
        ECSI_J,
        ENERGY_PER_SOCIAL_INTERACTION_J,
        PSYCHOACOUSTIC_YIELDING_COST,
    )
    DIRECTIONS = {
        "detour_time_s": Direction.LOWER,
        "ped_path_excess_m": Direction.LOWER,
        "robot_path_excess_m": Direction.LOWER,
        "robot_path_deviation_vs_planned_m": Direction.LOWER,
        "mar_vs_planned_route": Direction.HIGHER,
        "pfi_vs_planned_route": Direction.LOWER,
        "kcsc_wh_vs_planned_route": Direction.LOWER,
        "ecsi_j": Direction.LOWER,
        "energy_per_social_interaction_j": Direction.LOWER,
        "psychoacoustic_yielding_cost": Direction.LOWER,
    }
    PRIMARY_OUTPUTS = [
        "mar_vs_planned_route",
        "kcsc_wh_vs_planned_route",
        "detour_time_s",
    ]

    @classmethod
    def _null_output(cls, null_reason: str | None = None) -> dict[str, Any]:
        """Every published output key mapped to None, optionally with one shared null reason."""
        output: dict[str, Any] = {metric.name: None for metric in cls.OUTPUTS}
        if null_reason is not None:
            output.update({f"{name}{NULL_REASON_SUFFIX}": null_reason for name in output})
        return output

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, Any]) -> dict[str, Any]:
        """Encounter scope only: the episode as a whole carries no encounter costs."""
        del episode, prior_results
        return self._null_output()

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, Any],
        scope_slice: Any = None,
    ) -> dict[str, Any]:
        """Costs of one encounter window of the robot and its pedestrian."""
        del prior_results
        if scope_slice is None or scope_slice.scope != MetricScope.ENCOUNTER:
            return self._null_output()
        if str(scope_slice.identity.get("actor_2_kind", "pedestrian")).lower() != "pedestrian":
            return self._null_output(NullReason.NOT_APPLICABLE.value)
        robot, _ = resolve_robot_trajectory(self, episode)
        track = resolve_ped_track(extract_episode_ped_tracks(episode), scope_slice.identity.get("actor_2"))
        if track is None or len(robot.t_ns) < 2:
            return self._null_output()
        spans = pedestrian_encounter_spans(
            episode,
            float(self.robot_params.robot_radius),
            detector_params_from_config(self.metrics_config),
        )
        ac_frame = self.series_frame(episode, "acoustic_exposure")
        acoustic_exposure: tuple[np.ndarray, np.ndarray] | None = None
        if ac_frame is not None and ac_frame.height > 0:
            ped_ac = (
                ac_frame.filter(pl.col("ped_id").cast(pl.String) == str(track.ped_id))
                .select(
                    pl.col("t_ns").cast(pl.Int64),
                    pl.col("exposure_dba").cast(pl.Float64),
                )
                .drop_nulls()
                .sort("t_ns")
            )
            if ped_ac.height > 0:
                acoustic_exposure = (
                    ped_ac.get_column("t_ns").to_numpy().astype(np.int64),
                    ped_ac.get_column("exposure_dba").to_numpy().astype(np.float64),
                )
        return _encounter_outputs(
            track=track,
            window_ns=(int(scope_slice.t_start_ns), int(scope_slice.t_end_ns)),
            robot=robot,
            spans=spans,
            map_name=episode.map,
            run_dir=episode.run_dir,
            robot_radius_m=float(self.robot_params.robot_radius),
            power=_power_samples(episode),
            acoustic_exposure=acoustic_exposure,
        )
