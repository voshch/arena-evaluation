"""Barrier-aware pedestrian acoustic exposure and encounter sound dose."""

from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.acoustics.constants import (
    ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2,
    MIC_DISTANCE_M,
    NANOSECONDS_PER_SECOND,
    POSITION_EVALUATION_THRESHOLD_M,
    ROBOT_ACOUSTICS_TOPIC,
    WALL_TRANSMISSION_LOSS_DB,
)
from arena_evaluation.processing.acoustics.door_map import (
    _entity_matches_door,
    barrier_door_tl,
    build_pixel_tl,
    load_barrier_map,
)
from arena_evaluation.processing.acoustics.door_state import DoorStateTimeline
from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations
from arena_evaluation.processing.acoustics.statistics import (
    energy_equivalent_level_dba,
    frame_durations_s,
    max_positive_rise_dba_per_s,
    rise_rates_dba_per_s,
)
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
    description="Maximum positive rise rate of barrier-aware exposure experienced by this pedestrian, 0 dBA/s when it never rose and null below two reachable samples.",
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
POI_PSYCHOACOUSTIC_EXPOSURE = define_metric(
    "poi_psychoacoustic_exposure",
    pl.Float64,
    scopes=(MetricScope.PEDESTRIAN,),
    unit="s",
    description="Integrated linear A-weighted acoustic exposure at this pedestrian's position over the episode.",
    null_reason=True,
)

_METRICS: Final[tuple[Metric, ...]] = (
    PED_MAX_EXPOSURE_DBA,
    PED_LEQ_EXPOSURE_DBA,
    PED_MAX_STARTLE_FACTOR,
    WORST_CASE_ACOUSTIC_T_NS,
    SOUND_EXPOSURE_DOSE_PA2S,
    POI_PSYCHOACOUSTIC_EXPOSURE,
)
_PEDESTRIAN_KEYS: Final[tuple[str, ...]] = (
    "ped_max_exposure_dba",
    "ped_leq_exposure_dba",
    "ped_max_startle_factor",
    "worst_case_acoustic_t_ns",
    "poi_psychoacoustic_exposure",
)

SERIES_SCHEMA: Final[pl.Schema] = SeriesAcousticExposure.polars_schema()


@dataclasses.dataclass(frozen=True, slots=True)
class AcousticExposureResult:
    """Per-pedestrian scalar summaries plus the per-frame series the long table is built from."""

    pedestrian_scalars: dict[str, dict[str, object]]
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
    SERIES_REQUIRED_TOPICS = (
        "robot_tracks",
        "pedestrian_arena_peds",
        ROBOT_ACOUSTICS_TOPIC,
    )
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_tracks", "pedestrian_arena_peds", ROBOT_ACOUSTICS_TOPIC]
    PARAMS = ()

    OUTPUTS = _METRICS
    DIRECTIONS = {metric.name: Direction.LOWER for metric in _METRICS if metric.name != "worst_case_acoustic_t_ns"}
    PRIMARY_OUTPUTS = [
        "ped_max_exposure_dba",
        "ped_leq_exposure_dba",
        "sound_exposure_dose_pa2s",
    ]

    world: str | None = None

    def __init__(self, robot_params: RobotParams) -> None:
        """Memoise the per-episode frame solve for pedestrian rollups, encounter doses and series_table."""
        super().__init__(robot_params)
        self._episode_frame_cache: dict[tuple[int, str, str, str], AcousticExposureResult] = {}

    def _source_level(self, episode: AlignedEpisodeBundle, t_ns: np.ndarray) -> np.ndarray | None:
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

    def _sample_grid(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Sample times, ground-truth robot x/y and validity on the scope's clock, None without ground truth."""
        pos_x, pos_y, _yaw, native_t_ns = self.resolve_native_pose(episode)
        if len(native_t_ns) == 0:
            return None
        df = episode.data
        if df is None or len(df) == 0:
            return (
                native_t_ns,
                pos_x.astype(np.float64),
                pos_y.astype(np.float64),
                np.ones(len(native_t_ns), dtype=bool),
            )
        t_ns = df["t_ns"].to_numpy().astype(np.int64)
        rx_m = asof_values(
            native_t_ns.astype(np.int64),
            pos_x.astype(np.float64),
            t_ns,
            tolerance_ns=TRACK_ALIGN_TOLERANCE_NS,
        )
        ry_m = asof_values(
            native_t_ns.astype(np.int64),
            pos_y.astype(np.float64),
            t_ns,
            tolerance_ns=TRACK_ALIGN_TOLERANCE_NS,
        )
        return t_ns, rx_m, ry_m, ~np.isnan(rx_m) & ~np.isnan(ry_m)

    @staticmethod
    def _null_result() -> AcousticExposureResult:
        """The empty result: no frame carried an exposure value."""
        empty = np.array([], dtype=np.float64)
        return AcousticExposureResult(
            pedestrian_scalars={},
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
        """Reduce one frame series to per-pedestrian summaries and the episode's worst-frame index."""
        if not any(frame_exposure_dba):
            logger.info("No acoustic exposure values produced")
            return self._null_result()

        samples: dict[str, list[tuple[int, float]]] = {}
        worst_index, worst_value = 0, -np.inf
        for index, ped_ids in enumerate(frame_ped_ids):
            t_ns = int(frame_times_ns[index])
            if frame_exposure_dba[index]:
                frame_max = max(frame_exposure_dba[index])
                if frame_max > worst_value:
                    worst_value, worst_index = frame_max, index
            for ped_id, exposure_dba in zip(ped_ids, frame_exposure_dba[index], strict=True):
                samples.setdefault(ped_id, []).append((t_ns, float(exposure_dba)))

        return AcousticExposureResult(
            pedestrian_scalars={ped_id: self.pedestrian_summary(rows) for ped_id, rows in samples.items()},
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

    @staticmethod
    def pedestrian_summary(samples: list[tuple[int, float]]) -> dict[str, object]:
        """One pedestrian's maximum, energy-equivalent level, maximum positive rise rate and worst instant."""
        ordered = sorted(samples)
        times_ns = np.array([t_ns for t_ns, _ in ordered], dtype=np.int64)
        levels_dba = np.array([level for _, level in ordered], dtype=np.float64)
        worst = int(np.argmax(levels_dba))
        rates = rise_rates_dba_per_s(levels_dba, times_ns)
        durations_s = frame_durations_s(times_ns)
        return {
            "ped_max_exposure_dba": float(levels_dba[worst]),
            "ped_leq_exposure_dba": energy_equivalent_level_dba(levels_dba, durations_s),
            "ped_max_startle_factor": (max_positive_rise_dba_per_s(rates) if len(rates) else None),
            "worst_case_acoustic_t_ns": int(times_ns[worst]),
            "poi_psychoacoustic_exposure": float(np.sum(np.power(10.0, levels_dba / 10.0) * durations_s)),
        }

    def _compute(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> AcousticExposureResult:
        """Scalar summaries and per-frame pedestrian exposure series for the episode."""
        map_name = prior_results.get("map") or episode.map or self.world
        if not map_name:
            raise MetricInputError("AcousticExposureCalculator: missing world map name (checked prior_results['map'], episode.map, and calculator.world)")

        time_base = self._sample_grid(episode)
        if time_base is None:
            return self._null_result()

        peds_df = self.native_ped_frame(episode)
        if peds_df is None or len(peds_df) == 0:
            return self._null_result()

        run_dir = episode.run_dir
        if run_dir is None and episode.folder_manager and episode.run is not None and episode.run.benchmark_id:
            run_dir = episode.folder_manager.data_root / episode.run.benchmark_id

        barrier = load_barrier_map(map_name, run_dir=run_dir)
        grid, resolution = barrier.occupancy, barrier.resolution
        origin_x, origin_y = barrier.origin

        doors = barrier_door_tl(barrier)
        tl_cache: dict[tuple[str, ...], np.ndarray] = {}
        state_timeline = DoorStateTimeline.from_semantic_frame(episode.semantic_snapshot)
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
            raw_open = state_timeline.open_doors_at(int(t_ns[index])) if state_timeline is not None else frozenset()
            open_doors_here = tuple(sorted(name for name in doors if any(_entity_matches_door(name, entity) for entity in raw_open)))
            frame_open_doors.append(",".join(open_doors_here))
            current_source = source_dba[index]

            ped_ids_row: list[str] = []
            ped_x_row: list[float] = []
            ped_y_row: list[float] = []
            exposure_row: list[float] = []
            attenuation_row: list[float] = []

            if robot_valid[index] and not np.isnan(current_source) and frame is not None and len(frame):
                px_m = frame["x"].to_numpy().astype(np.float32)
                py_m = frame["y"].to_numpy().astype(np.float32)
                if len(px_m):
                    should_eval = index == 0 or index == total_frames - 1 or last_eval_rx is None or last_eval_ry is None or last_eval_peds is None or last_attenuations is None or open_doors_here != last_eval_tl_key
                    if not should_eval and last_eval_peds is not None:
                        robot_moved = np.hypot(rx_m[index] - last_eval_rx, ry_m[index] - last_eval_ry) > POSITION_EVALUATION_THRESHOLD_M
                        pedestrians_changed = len(px_m) != len(last_eval_peds[0]) or np.any(np.hypot(px_m - last_eval_peds[0], py_m - last_eval_peds[1]) > POSITION_EVALUATION_THRESHOLD_M)
                        should_eval = bool(robot_moved or pedestrians_changed)

                    if should_eval or last_attenuations is None:
                        last_eval_rx = float(rx_m[index])
                        last_eval_ry = float(ry_m[index])
                        last_eval_peds = (px_m, py_m)
                        last_eval_tl_key = open_doors_here

                        pixel_tl = tl_cache.get(open_doors_here)
                        if pixel_tl is None:
                            pixel_tl = build_pixel_tl(grid, doors, open_doors=set(open_doors_here))
                            tl_cache[open_doors_here] = pixel_tl

                        last_attenuations = compute_attenuations(
                            occupancy_grid=grid,
                            resolution=resolution,
                            start_x_px=(rx_m[index] - origin_x) / resolution,
                            start_y_px=(ry_m[index] - origin_y) / resolution,
                            target_xs_px=(px_m - origin_x) / resolution,
                            target_ys_px=(py_m - origin_y) / resolution,
                            wall_tl=WALL_TRANSMISSION_LOSS_DB,
                            mic_distance=MIC_DISTANCE_M,
                            pixel_tl=pixel_tl,
                        )

                    attenuations = last_attenuations
                    assert attenuations is not None
                    valid = ~np.isinf(attenuations)
                    if np.any(valid):
                        ped_ids_row = [str(name) for name in frame["ped_id"].to_numpy()[valid]]
                        ped_x_row = [float(value) for value in px_m[valid]]
                        ped_y_row = [float(value) for value in py_m[valid]]
                        attenuation_row = attenuations[valid].tolist()
                        exposure_row = (current_source - attenuations[valid]).tolist()

            frame_ped_ids.append(ped_ids_row)
            frame_ped_x_m.append(ped_x_row)
            frame_ped_y_m.append(ped_y_row)
            ts_exposure.append(exposure_row)
            ts_attenuation.append(attenuation_row)

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
    def _group_pedestrians_by_time(t_ns: np.ndarray, peds_df: pl.DataFrame) -> dict[int, pl.DataFrame]:
        """Map each clock sample to the latest pedestrian frame within TRACK_ALIGN_TOLERANCE_NS."""
        if len(peds_df) == 0 or len(t_ns) == 0:
            return {}
        clean = peds_df.drop_nulls(["t_ns", "x", "y"])
        if len(clean) == 0:
            return {}
        grouped = clean.partition_by("t_ns", as_dict=True)
        ped_times = np.sort(clean["t_ns"].unique().to_numpy().astype(np.int64))
        query_times = np.unique(t_ns.astype(np.int64))
        matched = asof_index(ped_times, query_times, tolerance_ns=TRACK_ALIGN_TOLERANCE_NS)
        return {int(query_t): grouped[(int(ped_times[idx]),)] for query_t, idx in zip(query_times, matched, strict=True) if idx >= 0 and (int(ped_times[idx]),) in grouped}

    def _episode_result(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> AcousticExposureResult:
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

    @staticmethod
    def _encounter_dose_pa2s(result: AcousticExposureResult, ped_id: str, t_start_ns: int, t_end_ns: int) -> float | None:
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
        dt_s = np.diff(np.asarray(times_ns, dtype=np.int64)).astype(np.float64) / NANOSECONDS_PER_SECOND
        p_sq = ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2 * np.power(10.0, np.asarray(exposures_dba, dtype=np.float64) / 10.0)
        return float(np.sum(0.5 * (p_sq[:-1] + p_sq[1:]) * dt_s))

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Refuse a robot-scope summary: exposure belongs to a pedestrian, never to the pool of them."""
        del episode, prior_results
        raise MetricInputError("AcousticExposureCalculator publishes the PEDESTRIAN and ENCOUNTER scopes only; a robot-scope value would pool distinct pedestrians into one exposure")

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.PEDESTRIAN:
            result = self._episode_result(scope_slice.full_episode, prior_results)
            ped_id = str(scope_slice.identity.get("ped_id") or scope_slice.slice_id)
            scalars = result.pedestrian_scalars.get(ped_id)
            if scalars is None:
                return dict.fromkeys(_PEDESTRIAN_KEYS)
            return {key: scalars.get(key) for key in _PEDESTRIAN_KEYS}
        if scope_slice.scope == MetricScope.ENCOUNTER:
            actor_2_kind = scope_slice.identity.get("actor_2_kind")
            if actor_2_kind is not None and actor_2_kind != "pedestrian":
                return {
                    "sound_exposure_dose_pa2s": None,
                    "sound_exposure_dose_pa2s_null_reason": "not_applicable",
                }
            full_result = self._episode_result(scope_slice.full_episode, prior_results)
            ped_id = str(scope_slice.identity.get("actor_2") or scope_slice.slice_id)
            dose = self._encounter_dose_pa2s(full_result, ped_id, scope_slice.t_start_ns, scope_slice.t_end_ns)
            return {"sound_exposure_dose_pa2s": dose}
        raise ValueError(f"acoustic_exposure does not serve the {scope_slice.scope.value} scope")

    def missing_input_provenance(self, episode: AlignedEpisodeBundle) -> dict[str, str]:
        """The door state decides which doors transmit; a recording without it leaves exposure unknown."""
        if DoorStateTimeline.from_semantic_frame(episode.semantic_snapshot) is not None:
            return {}
        return {key: f"{key}:topic_missing:world_semantics" for key in _PEDESTRIAN_KEYS}

    def series_table(self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = "") -> pl.DataFrame:
        """1NF flat pedestrian acoustic exposure series, one row per (sample, pedestrian).

        `is_episode_worst_frame` marks the episode-global loudest frame for every pedestrian of that
        frame (see the series declaration); the per-pedestrian worst instant is the published
        `worst_case_acoustic_t_ns` metric, so a per-pedestrian filter must use the metric, not
        this column.
        """
        result = self._episode_result(episode, {})
        if not any(result.frame_exposure_dba):
            return pl.DataFrame(schema=SERIES_SCHEMA)

        frame = (
            pl.DataFrame(
                {
                    "frame_index": np.arange(len(result.frame_times_ns)),
                    "t_ns": result.frame_times_ns,
                    "robot_x_m": result.frame_robot_x_m,
                    "robot_y_m": result.frame_robot_y_m,
                    "source_dba": result.frame_source_dba,
                    "ped_id": result.frame_ped_ids,
                    "ped_x_m": result.frame_ped_x_m,
                    "ped_y_m": result.frame_ped_y_m,
                    "exposure_dba": result.frame_exposure_dba,
                    "attenuation_db": result.frame_attenuation_db,
                    "open_doors": result.frame_open_doors,
                }
            )
            .explode(
                "ped_id",
                "ped_x_m",
                "ped_y_m",
                "exposure_dba",
                "attenuation_db",
            )
            .filter(pl.col("ped_id").is_not_null())
            .with_columns(
                run_id=pl.lit(run_id),
                episode_id=pl.lit(int(episode.episode_id), dtype=pl.Int64),
                robot_id=pl.lit(robot_id or (episode.robot_name or "robot")),
                is_episode_worst_frame=pl.col("frame_index") == result.worst_frame_index,
            )
            .select(SERIES_SCHEMA.names())
        )
        return frame.sort(["t_ns", "ped_id", "exposure_dba"]).unique(
            subset=["run_id", "episode_id", "robot_id", "t_ns", "ped_id"],
            keep="last",
            maintain_order=True,
        )
