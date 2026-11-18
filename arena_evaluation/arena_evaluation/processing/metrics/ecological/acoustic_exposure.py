"""Barrier-aware pedestrian acoustic exposure and encounter sound dose."""

from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.acoustics.door_map import (
    _entity_matches_door,
    barrier_door_tl,
    build_pixel_tl,
    load_barrier_map,
)
from arena_evaluation.processing.acoustics.door_state import DoorStateTimeline
from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations
from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.common.alignment import (
    TRACK_ALIGN_TOLERANCE_NS,
    asof_index,
    asof_values,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import Metric, MetricScope, define_metric
from arena_evaluation.storage.registry.series import SeriesAcousticExposure
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

logger = logging.getLogger(__name__)

POSITION_EVALUATION_THRESHOLD_M: Final[float] = 0.5
WALL_TRANSMISSION_LOSS_DB: Final[float] = 47.0
MIC_DISTANCE_M: Final[float] = 1.0
NANOSECONDS_PER_SECOND: Final[float] = 1e9
ACOUSTIC_REFERENCE_PRESSURE_PA: Final[float] = 20e-6
ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2: Final[float] = (
    ACOUSTIC_REFERENCE_PRESSURE_PA * ACOUSTIC_REFERENCE_PRESSURE_PA
)

PED_MAX_EXPOSURE_DBA = define_metric(
    "ped_max_exposure_dba",
    pl.Float64,
    scopes=(MetricScope.PEDESTRIAN,),
    unit="dBA",
    description="Maximum barrier-aware A-weighted exposure level experienced by this pedestrian.",
    null_reason=True,
)
PED_LEQ_EXPOSURE_DBA = define_metric(
    "ped_leq_exposure_dba",
    pl.Float64,
    scopes=(MetricScope.PEDESTRIAN,),
    unit="dBA",
    description="Energy-equivalent A-weighted exposure level experienced by this pedestrian.",
    null_reason=True,
)
PED_MAX_STARTLE_FACTOR = define_metric(
    "ped_max_startle_factor",
    pl.Float64,
    scopes=(MetricScope.PEDESTRIAN,),
    unit="dBA/s",
    description="Maximum positive rise rate of barrier-aware exposure experienced by this pedestrian.",
    null_reason=True,
)
WORST_CASE_ACOUSTIC_T_NS = define_metric(
    "worst_case_acoustic_t_ns",
    pl.Int64,
    scopes=(MetricScope.PEDESTRIAN,),
    unit="ns",
    description="Timestamp of the loudest exposure frame for this pedestrian in series_acoustic_exposure.",
    null_reason=True,
)
SOUND_EXPOSURE_DOSE_PA2S = define_metric(
    "sound_exposure_dose_pa2s",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER,),
    unit="Pa²·s",
    description="Barrier-aware acoustic exposure dose integrated over the encounter window.",
    null_reason=True,
)

_METRICS: Final[tuple[Metric, ...]] = (
    PED_MAX_EXPOSURE_DBA,
    PED_LEQ_EXPOSURE_DBA,
    PED_MAX_STARTLE_FACTOR,
    WORST_CASE_ACOUSTIC_T_NS,
    SOUND_EXPOSURE_DOSE_PA2S,
)
_PEDESTRIAN_KEYS: Final[tuple[str, ...]] = (
    "ped_max_exposure_dba",
    "ped_leq_exposure_dba",
    "ped_max_startle_factor",
    "worst_case_acoustic_t_ns",
)

SERIES_SCHEMA: Final[pl.Schema] = SeriesAcousticExposure.polars_schema()


@dataclasses.dataclass(frozen=True, slots=True)
class AcousticExposureResult:
    """Scalar exposure summaries plus the per-frame series the long table is built from."""

    scalars: dict[str, object]
    frame_times_ns: np.ndarray
    frame_robot_x_m: np.ndarray
    frame_robot_y_m: np.ndarray
    frame_source_dba: np.ndarray
    frame_ped_ids: list[list[str]]
    frame_ped_x_m: list[list[float]]
    frame_ped_y_m: list[list[float]]
    frame_exposure_dba: list[list[float]]
    frame_attenuation_db: list[list[float]]
    frame_open_doors: list[str]
    worst_frame_index: int


class AcousticExposureCalculator(BaseMetricCalculator):
    """Pedestrian and encounter exposure to robotic ego-noise from the barrier-map acoustic solver."""

    NAME = "acoustic_exposure"
    SERIES_KEY = ("t_ns", "ped_id")
    SERIES_REQUIRED_TOPICS = ("robot_tracks", "pedestrian_tracks", "robot_acoustics")
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["proxemics"]
    REQUIRED_TOPICS = ["robot_tracks", "pedestrian_tracks", "robot_acoustics"]
    PARAMS = ()

    OUTPUTS = _METRICS
    DIRECTIONS = {
        metric.name: Direction.LOWER
        for metric in _METRICS
        if metric.name != "worst_case_acoustic_t_ns"
    }
    PRIMARY_OUTPUTS = [
        "ped_max_exposure_dba",
        "ped_leq_exposure_dba",
        "sound_exposure_dose_pa2s",
    ]

    world: str | None = None

    def __init__(self, robot_params: RobotParams) -> None:
        """Memoise the per-episode frame solve for pedestrian rollups, encounter doses and series_table."""
        super().__init__(robot_params)
        self._episode_frame_cache: dict[
            tuple[int, str, str, str], AcousticExposureResult
        ] = {}

    def _source_level(
        self, episode: AlignedEpisodeBundle, t_ns: np.ndarray
    ) -> np.ndarray | None:
        """Emitted A-weighted level at the sample times from robot_acoustics, or None when absent."""
        acoustics = self.native_topics(episode).get("robot_acoustics")
        if acoustics is None or len(acoustics) == 0:
            return None
        reference = (
            acoustics.select(
                pl.col("t_ns").cast(pl.Int64),
                pl.col("total_level_af_dba").cast(pl.Float64),
            )
            .drop_nulls()
            .sort("t_ns")
        )
        if len(reference) == 0:
            return None
        aligned = asof_values(
            reference["t_ns"].to_numpy(),
            reference["total_level_af_dba"].to_numpy(),
            t_ns.astype(np.int64),
            tolerance_ns=TRACK_ALIGN_TOLERANCE_NS,
        )
        if np.all(np.isnan(aligned)):
            return None
        return aligned

    def _time_base(
        self, episode: AlignedEpisodeBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Sample times, ground-truth robot x/y and validity on the scope's clock, None without ground truth."""
        df = episode.data
        if df is not None and len(df) > 0:
            t_ns = df["t_ns"].to_numpy().astype(np.int64)
            pos_x_native, pos_y_native, _yaw, native_times_ns = (
                self.resolve_native_pose(episode)
            )
            if len(native_times_ns) == 0:
                return None
            rx_m = asof_values(
                native_times_ns.astype(np.int64),
                pos_x_native.astype(np.float64),
                t_ns,
                tolerance_ns=TRACK_ALIGN_TOLERANCE_NS,
            )
            ry_m = asof_values(
                native_times_ns.astype(np.int64),
                pos_y_native.astype(np.float64),
                t_ns,
                tolerance_ns=TRACK_ALIGN_TOLERANCE_NS,
            )
            return t_ns, rx_m, ry_m, ~np.isnan(rx_m) & ~np.isnan(ry_m)

        pos_x, pos_y, _yaw, t_ns = self.resolve_native_pose(episode)
        if len(t_ns) == 0:
            return None
        return (
            t_ns,
            pos_x.astype(np.float64),
            pos_y.astype(np.float64),
            np.ones(len(t_ns), dtype=bool),
        )

    @staticmethod
    def _null_result() -> AcousticExposureResult:
        empty = np.array([], dtype=np.float64)
        return AcousticExposureResult(
            scalars=dict.fromkeys(_PEDESTRIAN_KEYS, None),
            frame_times_ns=np.array([], dtype=np.int64),
            frame_robot_x_m=empty,
            frame_robot_y_m=empty.copy(),
            frame_source_dba=empty.copy(),
            frame_ped_ids=[],
            frame_ped_x_m=[],
            frame_ped_y_m=[],
            frame_exposure_dba=[],
            frame_attenuation_db=[],
            frame_open_doors=[],
            worst_frame_index=0,
        )

    def _summarize(
        self,
        frame_times_ns: np.ndarray,
        frame_robot_x_m: np.ndarray,
        frame_robot_y_m: np.ndarray,
        frame_source_dba: np.ndarray,
        frame_ped_ids: list[list[str]],
        frame_ped_x_m: list[list[float]],
        frame_ped_y_m: list[list[float]],
        frame_exposure_dba: list[list[float]],
        frame_attenuation_db: list[list[float]],
        frame_open_doors: list[str],
    ) -> AcousticExposureResult:
        """Reduce one frame series to the scalar summaries and worst-frame index."""
        if not any(frame_exposure_dba):
            logger.info("No acoustic exposure values produced")
            return self._null_result()

        all_exposure = np.concatenate(
            [
                np.asarray(frame, dtype=np.float64)
                for frame in frame_exposure_dba
                if frame
            ]
        )
        max_exposure_dba = float(np.max(all_exposure))
        leq_exposure_dba = float(
            10.0 * np.log10(np.mean(10.0 ** (all_exposure / 10.0)))
        )

        time_s = frame_times_ns / NANOSECONDS_PER_SECOND
        startle_rates: list[float] = []
        for index in range(1, len(frame_exposure_dba)):
            previous = frame_exposure_dba[index - 1]
            current = frame_exposure_dba[index]
            dt = time_s[index] - time_s[index - 1]
            if dt > 0 and len(previous) == len(current) and len(current) > 0:
                startle_rates.extend(
                    ((np.asarray(current) - np.asarray(previous)) / dt).tolist()
                )
        max_startle = float(np.max(startle_rates)) if startle_rates else None

        worst_index = 0
        worst_value = -np.inf
        for index, frame_exposure in enumerate(frame_exposure_dba):
            if frame_exposure:
                frame_max = max(frame_exposure)
                if frame_max > worst_value:
                    worst_value = frame_max
                    worst_index = index

        return AcousticExposureResult(
            scalars={
                "ped_max_exposure_dba": max_exposure_dba,
                "ped_leq_exposure_dba": leq_exposure_dba,
                "ped_max_startle_factor": max_startle,
                "worst_case_acoustic_t_ns": int(frame_times_ns[worst_index]),
            },
            frame_times_ns=frame_times_ns,
            frame_robot_x_m=frame_robot_x_m,
            frame_robot_y_m=frame_robot_y_m,
            frame_source_dba=frame_source_dba,
            frame_ped_ids=frame_ped_ids,
            frame_ped_x_m=frame_ped_x_m,
            frame_ped_y_m=frame_ped_y_m,
            frame_exposure_dba=frame_exposure_dba,
            frame_attenuation_db=frame_attenuation_db,
            frame_open_doors=frame_open_doors,
            worst_frame_index=worst_index,
        )

    def _compute(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> AcousticExposureResult:
        """Scalar summaries and per-frame pedestrian exposure series for the episode."""
        map_name = prior_results.get("map") or episode.map or self.world
        if not map_name:
            raise MetricInputError(
                "AcousticExposureCalculator: missing world map name (checked prior_results['map'], episode.map, and calculator.world)"
            )

        time_base = self._time_base(episode)
        if time_base is None:
            return self._null_result()

        peds_df = self.native_ped_frame(episode)
        if peds_df is None or len(peds_df) == 0:
            return self._null_result()

        run_dir = episode.run_dir
        if (
            run_dir is None
            and episode.folder_manager
            and episode.run is not None
            and episode.run.benchmark_id
        ):
            run_dir = episode.folder_manager.data_root / episode.run.benchmark_id

        barrier = load_barrier_map(map_name, run_dir=run_dir)
        grid, resolution = barrier.occupancy, barrier.resolution
        origin_x, origin_y = barrier.origin

        doors = barrier_door_tl(barrier)
        tl_cache: dict[tuple[str, ...], np.ndarray] = {}
        state_timeline = DoorStateTimeline.from_semantic_frame(
            episode.semantic_snapshot
        )
        if doors and state_timeline is None:
            logger.info(
                "AcousticExposureCalculator: %d doors but no door-state timeline for episode %s; returning NULL",
                len(doors),
                episode.episode_id,
            )
            return self._null_result()

        t_ns, rx_m, ry_m, robot_valid = time_base
        source_dba = self._source_level(episode, t_ns)
        if source_dba is None:
            return self._null_result()

        peds_at_time = self._group_pedestrians_by_time(t_ns, peds_df)

        frame_ped_ids: list[list[str]] = []
        frame_ped_x_m: list[list[float]] = []
        frame_ped_y_m: list[list[float]] = []
        ts_exposure: list[list[float]] = []
        ts_attenuation: list[list[float]] = []
        frame_open_doors: list[str] = []

        last_eval_rx: float | None = None
        last_eval_ry: float | None = None
        last_eval_peds: tuple[np.ndarray, np.ndarray] | None = None
        last_eval_tl_key: tuple[str, ...] | None = None
        last_attenuations: np.ndarray | None = None
        total_frames = len(t_ns)

        for index in range(total_frames):
            frame = peds_at_time.get(int(t_ns[index]))
            raw_open = (
                state_timeline.open_doors_at(int(t_ns[index]))
                if state_timeline is not None
                else frozenset()
            )
            open_doors_here = tuple(
                sorted(
                    name
                    for name in doors
                    if any(_entity_matches_door(name, entity) for entity in raw_open)
                )
            )
            frame_open_doors.append(",".join(open_doors_here))
            current_source = source_dba[index]
            if (
                not robot_valid[index]
                or np.isnan(current_source)
                or frame is None
                or len(frame) == 0
            ):
                frame_ped_ids.append([])
                frame_ped_x_m.append([])
                frame_ped_y_m.append([])
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            px_m = frame["x"].to_numpy().astype(np.float32)
            py_m = frame["y"].to_numpy().astype(np.float32)
            if len(px_m) == 0:
                frame_ped_ids.append([])
                frame_ped_x_m.append([])
                frame_ped_y_m.append([])
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            should_eval = (
                index == 0
                or index == total_frames - 1
                or last_eval_rx is None
                or last_eval_ry is None
                or last_eval_peds is None
                or last_attenuations is None
                or open_doors_here != last_eval_tl_key
            )
            if (
                not should_eval
                and last_eval_rx is not None
                and last_eval_ry is not None
                and last_eval_peds is not None
            ):
                robot_moved = (
                    np.hypot(rx_m[index] - last_eval_rx, ry_m[index] - last_eval_ry)
                    > POSITION_EVALUATION_THRESHOLD_M
                )
                pedestrians_changed = len(px_m) != len(last_eval_peds[0]) or np.any(
                    np.hypot(px_m - last_eval_peds[0], py_m - last_eval_peds[1])
                    > POSITION_EVALUATION_THRESHOLD_M
                )
                should_eval = bool(robot_moved or pedestrians_changed)

            if should_eval or last_attenuations is None:
                last_eval_rx = float(rx_m[index])
                last_eval_ry = float(ry_m[index])
                last_eval_peds = (px_m, py_m)
                last_eval_tl_key = open_doors_here

                rx_px = (rx_m[index] - origin_x) / resolution
                ry_px = (ry_m[index] - origin_y) / resolution
                px_px = (px_m - origin_x) / resolution
                py_px = (py_m - origin_y) / resolution

                pixel_tl = tl_cache.get(open_doors_here)
                if pixel_tl is None:
                    pixel_tl = build_pixel_tl(
                        grid, doors, open_doors=set(open_doors_here)
                    )
                    tl_cache[open_doors_here] = pixel_tl

                attenuations = compute_attenuations(
                    occupancy_grid=grid,
                    resolution=resolution,
                    start_x_px=rx_px,
                    start_y_px=ry_px,
                    target_xs_px=px_px,
                    target_ys_px=py_px,
                    wall_tl=WALL_TRANSMISSION_LOSS_DB,
                    mic_distance=MIC_DISTANCE_M,
                    pixel_tl=pixel_tl,
                )
                last_attenuations = attenuations
            else:
                attenuations = last_attenuations

            valid = ~np.isinf(attenuations)
            if not np.any(valid):
                frame_ped_ids.append([])
                frame_ped_x_m.append([])
                frame_ped_y_m.append([])
                ts_exposure.append([])
                ts_attenuation.append([])
                continue

            valid_attenuation = attenuations[valid]
            valid_exposure = current_source - valid_attenuation
            frame_ped_ids.append(
                [str(name) for name in frame["ped_id"].to_numpy()[valid]]
            )
            frame_ped_x_m.append([float(value) for value in px_m[valid]])
            frame_ped_y_m.append([float(value) for value in py_m[valid]])
            ts_attenuation.append(valid_attenuation.tolist())
            ts_exposure.append(valid_exposure.tolist())

        return self._summarize(
            t_ns,
            rx_m,
            ry_m,
            source_dba,
            frame_ped_ids,
            frame_ped_x_m,
            frame_ped_y_m,
            ts_exposure,
            ts_attenuation,
            frame_open_doors,
        )

    @staticmethod
    def _group_pedestrians_by_time(
        t_ns: np.ndarray, peds_df: pl.DataFrame
    ) -> dict[int, pl.DataFrame]:
        """Map each clock sample to the latest pedestrian frame within TRACK_ALIGN_TOLERANCE_NS."""
        if len(peds_df) == 0 or len(t_ns) == 0:
            return {}
        clean = peds_df.drop_nulls(["t_ns", "x", "y"])
        if len(clean) == 0:
            return {}
        grouped = clean.partition_by("t_ns", as_dict=True)
        ped_times = np.sort(clean["t_ns"].unique().to_numpy().astype(np.int64))
        query_times = np.unique(t_ns.astype(np.int64))
        matched = asof_index(
            ped_times, query_times, tolerance_ns=TRACK_ALIGN_TOLERANCE_NS
        )
        return {
            int(query_t): grouped[(int(ped_times[idx]),)]
            for query_t, idx in zip(query_times, matched, strict=True)
            if idx >= 0 and (int(ped_times[idx]),) in grouped
        }

    def _episode_result(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> AcousticExposureResult:
        """The episode's exposure frame series, solved once and reused across scopes and series_table."""
        map_token = str(prior_results.get("map") or episode.map or self.world or "")
        key = (
            int(episode.episode_id),
            str(episode.robot_name or ""),
            map_token,
            str(episode.run_dir or ""),
        )
        cached = self._episode_frame_cache.get(key)
        if cached is None:
            cached = self._compute(episode, prior_results)
            self._episode_frame_cache[key] = cached
        return cached

    def _restrict_to_pedestrian(
        self, result: AcousticExposureResult, ped_id: str
    ) -> AcousticExposureResult:
        """The same solved frame series reduced to the rows of one pedestrian."""
        frame_ped_ids: list[list[str]] = []
        frame_ped_x_m: list[list[float]] = []
        frame_ped_y_m: list[list[float]] = []
        frame_exposure_dba: list[list[float]] = []
        frame_attenuation_db: list[list[float]] = []
        for index, ped_ids in enumerate(result.frame_ped_ids):
            keep = [position for position, name in enumerate(ped_ids) if name == ped_id]
            frame_ped_ids.append([ped_ids[position] for position in keep])
            frame_ped_x_m.append(
                [result.frame_ped_x_m[index][position] for position in keep]
            )
            frame_ped_y_m.append(
                [result.frame_ped_y_m[index][position] for position in keep]
            )
            frame_exposure_dba.append(
                [result.frame_exposure_dba[index][position] for position in keep]
            )
            frame_attenuation_db.append(
                [result.frame_attenuation_db[index][position] for position in keep]
            )
        return self._summarize(
            result.frame_times_ns,
            result.frame_robot_x_m,
            result.frame_robot_y_m,
            result.frame_source_dba,
            frame_ped_ids,
            frame_ped_x_m,
            frame_ped_y_m,
            frame_exposure_dba,
            frame_attenuation_db,
            result.frame_open_doors,
        )

    @staticmethod
    def _encounter_dose_pa2s(
        result: AcousticExposureResult, ped_id: str, t_start_ns: int, t_end_ns: int
    ) -> float | None:
        """Integrate barrier-aware pressure-squared exposure over [t_start_ns, t_end_ns] for one pedestrian."""
        times_ns: list[int] = []
        exposures_dba: list[float] = []
        for index, t_val in enumerate(result.frame_times_ns):
            t_int = int(t_val)
            if t_int < t_start_ns or t_int > t_end_ns:
                continue
            for pos, name in enumerate(result.frame_ped_ids[index]):
                if name == ped_id:
                    times_ns.append(t_int)
                    exposures_dba.append(result.frame_exposure_dba[index][pos])
                    break
        if len(times_ns) < 2:
            return None
        dt_s = (
            np.diff(np.asarray(times_ns, dtype=np.int64)).astype(np.float64)
            / NANOSECONDS_PER_SECOND
        )
        p_sq = ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2 * np.power(
            10.0, np.asarray(exposures_dba, dtype=np.float64) / 10.0
        )
        return float(np.sum(0.5 * (p_sq[:-1] + p_sq[1:]) * dt_s))

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        scalars = self._episode_result(episode, prior_results).scalars
        return {key: scalars.get(key) for key in _PEDESTRIAN_KEYS}

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.PEDESTRIAN:
            full_result = self._episode_result(scope_slice.full_episode, prior_results)
            ped_id = str(scope_slice.identity.get("ped_id") or scope_slice.slice_id)
            restricted = self._restrict_to_pedestrian(full_result, ped_id)
            return {key: restricted.scalars.get(key) for key in _PEDESTRIAN_KEYS}
        if scope_slice.scope == MetricScope.ENCOUNTER:
            actor_2_kind = scope_slice.identity.get("actor_2_kind")
            if actor_2_kind is not None and actor_2_kind != "pedestrian":
                return {
                    "sound_exposure_dose_pa2s": None,
                    "sound_exposure_dose_pa2s_null_reason": "not_applicable",
                }
            full_result = self._episode_result(scope_slice.full_episode, prior_results)
            ped_id = str(scope_slice.identity.get("actor_2") or scope_slice.slice_id)
            dose = self._encounter_dose_pa2s(
                full_result, ped_id, scope_slice.t_start_ns, scope_slice.t_end_ns
            )
            return {"sound_exposure_dose_pa2s": dose}
        raise ValueError(
            f"acoustic_exposure does not serve the {scope_slice.scope.value} scope"
        )

    def series_table(
        self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = ""
    ) -> pl.DataFrame:
        """1NF flat pedestrian acoustic exposure series, one row per (sample, pedestrian)."""
        result = self._episode_result(episode, {})
        if not any(result.frame_exposure_dba):
            return pl.DataFrame(schema=SERIES_SCHEMA)

        t_ns: list[int] = []
        ped_ids: list[str] = []
        exposure: list[float] = []
        attenuation: list[float] = []
        robot_x: list[float] = []
        robot_y: list[float] = []
        source: list[float] = []
        ped_x: list[float] = []
        ped_y: list[float] = []
        is_worst: list[bool] = []
        open_doors: list[str] = []
        for index, (
            ped_id_row,
            ped_x_row,
            ped_y_row,
            exposure_row,
            attenuation_row,
        ) in enumerate(
            zip(
                result.frame_ped_ids,
                result.frame_ped_x_m,
                result.frame_ped_y_m,
                result.frame_exposure_dba,
                result.frame_attenuation_db,
                strict=True,
            )
        ):
            for ped_id, x_m, y_m, exposure_dba, attenuation_db in zip(
                ped_id_row,
                ped_x_row,
                ped_y_row,
                exposure_row,
                attenuation_row,
                strict=True,
            ):
                t_ns.append(int(result.frame_times_ns[index]))
                ped_ids.append(ped_id)
                exposure.append(float(exposure_dba))
                attenuation.append(float(attenuation_db))
                robot_x.append(float(result.frame_robot_x_m[index]))
                robot_y.append(float(result.frame_robot_y_m[index]))
                source.append(float(result.frame_source_dba[index]))
                ped_x.append(x_m)
                ped_y.append(y_m)
                is_worst.append(index == result.worst_frame_index)
                open_doors.append(result.frame_open_doors[index])

        frame = pl.DataFrame(
            {
                "run_id": [run_id] * len(t_ns),
                "episode_id": [int(episode.episode_id)] * len(t_ns),
                "robot_id": [robot_id or (episode.robot_name or "robot")] * len(t_ns),
                "t_ns": t_ns,
                "ped_id": ped_ids,
                "exposure_dba": exposure,
                "attenuation_db": attenuation,
                "robot_x_m": robot_x,
                "robot_y_m": robot_y,
                "source_dba": source,
                "ped_x_m": ped_x,
                "ped_y_m": ped_y,
                "is_worst_frame": is_worst,
                "open_doors": open_doors,
            },
            schema=SERIES_SCHEMA,
        )
        return frame.sort(["t_ns", "ped_id", "exposure_dba"]).unique(
            subset=["run_id", "episode_id", "robot_id", "t_ns", "ped_id"],
            keep="last",
            maintain_order=True,
        )
