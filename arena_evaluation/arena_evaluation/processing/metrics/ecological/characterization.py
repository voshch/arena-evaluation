"""Open-loop characterization metrics (energy/acoustic profiles per working point).

Per-episode calculator: attaches the recorded ``characterization_phase`` markers
to every sample, resolves each label against the ``characterization_schedule``
table the sweep published (rebuilt from the robot envelope only when a recording
predates that topic), computes per-sample power / mechanical power / acoustic
level / energy intensity, and keeps only scalar summaries in the metric row. The
per-sample values are published by :meth:`CharacterizationCalculator.series_table`
as a flat 1NF long table (native nanosecond timestamps, explicit nulls), never as
list columns in Parquet.
"""

from __future__ import annotations

import functools
import logging
import math
import typing
from dataclasses import dataclass

import polars as pl

from arena_evaluation.storage.registry.series import SeriesCharacterization
from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.common.alignment import TRACK_ALIGN_TOLERANCE_NS, asof_join
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

logger = logging.getLogger(__name__)

CHAR_PHASE_COVERAGE = define_metric("char_phase_coverage", pl.Float64, scopes=(MetricScope.ROBOT,), unit="1", description="ecological metric of the characterization calculator", null_reason=True)
CHAR_MEAN_EFFICIENCY = define_metric("char_mean_efficiency", pl.Float64, scopes=(MetricScope.ROBOT,), unit="1", description="Mean mechanical-to-electrical power ratio over the characterization samples.", null_reason=True)
CHAR_LEQ_DBA = define_metric("char_leq_dba", pl.Float64, scopes=(MetricScope.ROBOT,), unit="dBA", description="Energy-equivalent A-weighted level over the characterization samples, dBA.", null_reason=True)
CHAR_MEAN_ENERGY_INTENSITY_J_M = define_metric("char_mean_energy_intensity_j_m", pl.Float64, scopes=(MetricScope.ROBOT,), unit="J/m", description="Mean energy per travelled metre over the characterization samples.", null_reason=True)
CHAR_SAMPLE_COUNT = define_metric("char_sample_count", pl.Int64, scopes=(MetricScope.ROBOT,), description="Number of characterization samples the robot published.", null_reason=True)

# Last-resort constants, mirroring config/acoustic_profile.yaml.
_ACOUSTIC_DEFAULTS: typing.Final[dict[str, float]] = {
    "L_base_0": 42.0,
    "beta_0": 45.0,
    "beta_1": 18.0,
    "beta_2": 5.0,
    "omega_ref": 5.0,
    "tau_ref": 10.0,
    "omega_active": 0.2,
}

# Head of a dwell where the robot is still accelerating, so not yet steady state.
_TRANSIENT_S: typing.Final[float] = 1.5  # s, linear, lateral and arc dwells
_TRANSIENT_ANGULAR_S: typing.Final[float] = 1.0  # s, in-place pivots

_NANOSECONDS_PER_SECOND: typing.Final[float] = 1e9
_MIN_SPEED_MPS: typing.Final[float] = 0.05
_MIN_TURN_RATE_RADPS: typing.Final[float] = 0.05
_MIN_ACCEL_DT_S: typing.Final[float] = 1e-4
_MIN_EFFICIENCY_POWER_W: typing.Final[float] = 0.1


@dataclass(frozen=True, slots=True, kw_only=True)
class AcousticProfile:
    """Drivetrain acoustic model coefficients, loaded once at calculator construction."""

    l_base_0_db: float
    beta_0_db: float
    beta_1_db: float
    beta_2_db: float
    omega_ref_radps: float
    tau_ref_nm: float
    omega_active_radps: float


BUILTIN_ACOUSTIC_PROFILE: typing.Final[AcousticProfile] = AcousticProfile(
    l_base_0_db=_ACOUSTIC_DEFAULTS["L_base_0"],
    beta_0_db=_ACOUSTIC_DEFAULTS["beta_0"],
    beta_1_db=_ACOUSTIC_DEFAULTS["beta_1"],
    beta_2_db=_ACOUSTIC_DEFAULTS["beta_2"],
    omega_ref_radps=_ACOUSTIC_DEFAULTS["omega_ref"],
    tau_ref_nm=_ACOUSTIC_DEFAULTS["tau_ref"],
    omega_active_radps=_ACOUSTIC_DEFAULTS["omega_active"],
)


@functools.cache
def load_acoustic_profile(model: str) -> AcousticProfile:
    """Per-robot acoustic profile from the arena_robots share directory, the shared profile, else the built-in."""
    from pathlib import Path

    if not isinstance(model, str) or not model:
        logger.warning(f"robot model {model!r} is not a name, using built-in acoustic constants")
        return BUILTIN_ACOUSTIC_PROFILE

    import yaml
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

    try:
        share = Path(get_package_share_directory("arena_robots"))
    except PackageNotFoundError:
        logger.warning("arena_robots not installed, using built-in acoustic constants")
        return BUILTIN_ACOUSTIC_PROFILE

    per_robot = share / "robots" / model / "telemetry" / "acoustics.yaml"
    for candidate in (per_robot, share / "config" / "acoustic_profile.yaml"):
        if not candidate.is_file():
            continue
        try:
            config = yaml.safe_load(candidate.read_text())
        except (OSError, yaml.YAMLError) as error:
            logger.warning(f"unreadable acoustic profile {candidate}: {error!r}")
            continue
        if not isinstance(config, dict):
            logger.warning(f"acoustic profile {candidate} is not a mapping, using built-in acoustic constants")
            return BUILTIN_ACOUSTIC_PROFILE
        missing = sorted(key for key in _ACOUSTIC_DEFAULTS if key not in config)
        if missing:
            logger.warning(f"acoustic profile {candidate} lacks {missing}, using built-in values for those")
        return AcousticProfile(
            l_base_0_db=float(config.get("L_base_0", _ACOUSTIC_DEFAULTS["L_base_0"])),
            beta_0_db=float(config.get("beta_0", _ACOUSTIC_DEFAULTS["beta_0"])),
            beta_1_db=float(config.get("beta_1", _ACOUSTIC_DEFAULTS["beta_1"])),
            beta_2_db=float(config.get("beta_2", _ACOUSTIC_DEFAULTS["beta_2"])),
            omega_ref_radps=float(config.get("omega_ref", _ACOUSTIC_DEFAULTS["omega_ref"])),
            tau_ref_nm=float(config.get("tau_ref", _ACOUSTIC_DEFAULTS["tau_ref"])),
            omega_active_radps=float(config.get("omega_active", _ACOUSTIC_DEFAULTS["omega_active"])),
        )

    logger.warning(f"no acoustic profile for {model!r} at {per_robot}, using built-in acoustic constants")
    return BUILTIN_ACOUSTIC_PROFILE


def _leq_power(level_dba: pl.Series | pl.Expr) -> pl.Series | pl.Expr:
    """Linear acoustic power proxy 10^(L/10), L_Aeq = 10*log10(mean(x))."""
    return 10.0 ** (level_dba / 10.0)


SERIES_SCHEMA: typing.Final[pl.Schema] = SeriesCharacterization.polars_schema()


class CharacterizationCalculator(BaseMetricCalculator):
    NAME = "characterization"
    SERIES_KEY = ("t_ns",)
    SERIES_REQUIRED_TOPICS = ("robot_tracks",)
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_tracks"]
    PARAMS = ()

    OUTPUTS = (
        CHAR_PHASE_COVERAGE,
        CHAR_MEAN_EFFICIENCY,
        CHAR_LEQ_DBA,
        CHAR_MEAN_ENERGY_INTENSITY_J_M,
        CHAR_SAMPLE_COUNT,
    )

    # One row per scheduled phase, the same columns the mcap reader decodes.
    _SCHEDULE_SCHEMA = {
        "phase_label": pl.Utf8,
        "phase_kind": pl.Utf8,
        "vx_target": pl.Float64,
        "vy_target": pl.Float64,
        "wz_target": pl.Float64,
        "duration_s": pl.Float64,
        "ramp_s": pl.Float64,
        "turn_radius_m": pl.Float64,
    }

    def __init__(self, robot_params: RobotParams) -> None:
        super().__init__(robot_params)
        self._acoustic_profile = load_acoustic_profile(self.robot_params.model)

    def _rebuilt_schedule(self) -> pl.DataFrame:
        """The default schedule for this robot's envelope, for recordings that carry no schedule table."""
        from arena_robots.characterization import build_schedule, resolve_envelope

        envelope = resolve_envelope(self.robot_params.model)
        schedule = build_schedule(
            vx_max=float(envelope["vx_max"]),
            vy_max=float(envelope["vy_max"]),
            wz_max=float(envelope["wz_max"]),
            radius=float(envelope["radius"]),
            is_holonomic=bool(envelope["is_holonomic"]),
        )
        rows = [
            {
                "phase_label": phase.name,
                "phase_kind": phase.kind.value,
                "vx_target": phase.vx_target,
                "vy_target": phase.vy_target,
                "wz_target": phase.wz_target,
                "duration_s": phase.duration_s,
                "ramp_s": phase.ramp_s,
                "turn_radius_m": phase.radius_m,
            }
            for phase in schedule
        ]
        return pl.DataFrame(rows, schema=self._SCHEDULE_SCHEMA)

    def _phase_map(self, episode: AlignedEpisodeBundle) -> pl.DataFrame:
        """Labels and targets of the schedule that ran, with the signed acceleration each ramp commands."""
        recorded = self.native_topics(episode).get("robot_characterization_schedule")
        if recorded is not None and len(recorded) > 0:
            table = recorded.select(list(self._SCHEDULE_SCHEMA)).unique(subset=["phase_label"], keep="last", maintain_order=True)
        else:
            logger.warning("characterization: no recorded schedule, rebuilding the default one from the robot envelope")
            table = self._rebuilt_schedule()

        accel = pl.when(pl.col("ramp_s") > 0.0).then(pl.col("vx_target") / pl.col("ramp_s")).otherwise(0.0)
        accel = pl.when(pl.col("phase_kind") == "ramp_down").then(-accel).otherwise(accel)
        return table.with_columns(accel.alias("accel_target")).drop(["duration_s", "ramp_s"])

    def _attach_phases(self, episode: AlignedEpisodeBundle) -> tuple[pl.DataFrame, float | None]:
        """Aligned frame with phase labels/targets attached, and the fraction of samples the schedule recognised.

        None when no phase markers were recorded: the coverage of a schedule that does not exist is
        unknown, not zero, so the robot fact publishes null with the missing-topic reason.
        """
        out = episode.data
        marked = "label" in out.columns
        if not marked:
            # No phase markers at all: not a characterization episode, every sample classifies from cmd_vel.
            out = out.with_columns(pl.lit("unknown").alias("phase_label"))
            phase_map = pl.DataFrame(schema=self._SCHEDULE_SCHEMA).drop(["duration_s", "ramp_s"]).with_columns(pl.lit(0.0).alias("accel_target"))
        else:
            out = out.with_columns(pl.col("label").fill_null("unknown").alias("phase_label"))
            phase_map = self._phase_map(episode)
        out = out.join(phase_map, on="phase_label", how="left")

        unmatched = int(out["phase_kind"].null_count())
        phase_coverage = (1.0 - unmatched / len(out) if len(out) else 1.0) if marked else None
        # Unmarked samples ("unknown") are expected before the first marker, foreign labels are not.
        foreign = sorted(set(out.filter(pl.col("phase_kind").is_null())["phase_label"].to_list()) - {"unknown"})
        if foreign:
            logger.warning(f"characterization: {unmatched}/{len(out)} samples carry a label the schedule table does not contain, so they fall back to cmd_vel classification. First unmatched labels: {foreign[:5]}")

        # Fallback classification for unmarked samples (markers never recorded).
        def optional_command(column: str) -> pl.Expr:
            return pl.col(column).cast(pl.Float64) if column in out.columns else pl.lit(None, dtype=pl.Float64)

        cmd_vx = optional_command("linear_x")
        cmd_vy = optional_command("linear_y")
        cmd_wz = optional_command("angular_z")
        idle = (cmd_vx == 0.0) & (cmd_vy == 0.0) & (cmd_wz == 0.0)
        lateral = (cmd_vy != 0.0) & (cmd_vx == 0.0) & (cmd_wz == 0.0)
        arc = (cmd_vx != 0.0) & (cmd_wz != 0.0)
        fallback_kind = pl.when(idle).then(pl.lit("idle")).when(lateral).then(pl.lit("lateral")).when(arc).then(pl.lit("arc")).when(cmd_wz != 0.0).then(pl.lit("angular")).otherwise(pl.lit("linear"))
        out = out.with_columns(
            pl.col("phase_kind").fill_null(fallback_kind).alias("phase_kind"),
            pl.col("vx_target").fill_null(cmd_vx).alias("vx_target"),
            pl.col("vy_target").fill_null(cmd_vy).alias("vy_target"),
            pl.col("wz_target").fill_null(cmd_wz).alias("wz_target"),
            pl.col("turn_radius_m").fill_null(pl.when((cmd_vx.abs() >= 0.05) & (cmd_wz.abs() >= 0.05)).then(cmd_vx.abs() / cmd_wz.abs()).otherwise(0.0)).alias("turn_radius_m"),
        )

        # Split the transient windows off the steady-state part of every dwell:
        # the accelerating head, plus the trailing decel/cutoff on long blocks.
        if "t_ns" in out.columns and len(out) > 0:
            out = out.with_columns(((pl.col("phase_label") != pl.col("phase_label").shift(1)) | (pl.col("vx_target") != pl.col("vx_target").shift(1)) | (pl.col("vy_target") != pl.col("vy_target").shift(1)) | (pl.col("wz_target") != pl.col("wz_target").shift(1))).fill_null(True).cum_sum().alias("_phase_block_id"))
            out = out.with_columns(
                ((pl.col("t_ns") - pl.col("t_ns").min().over("_phase_block_id")).cast(pl.Float64) / _NANOSECONDS_PER_SECOND).alias("_phase_elapsed_s"),
                ((pl.col("t_ns").max().over("_phase_block_id") - pl.col("t_ns").min().over("_phase_block_id")).cast(pl.Float64) / _NANOSECONDS_PER_SECOND).alias("_phase_total_duration_s"),
            )
            is_settle = pl.col("phase_label").str.contains("settle")
            is_transient_dwell = pl.col("phase_label").str.starts_with("linear_vx_") | pl.col("phase_label").str.starts_with("lateral_vy_") | pl.col("phase_label").str.starts_with("arc_vx_")
            is_angular_dwell = pl.col("phase_label").str.starts_with("angular_wz_")
            is_long_block = pl.col("_phase_total_duration_s") >= 4.0
            tail_s = pl.min_horizontal(pl.lit(0.8), pl.col("_phase_total_duration_s") * 0.15)
            in_tail = is_long_block & (pl.col("_phase_elapsed_s") > (pl.col("_phase_total_duration_s") - tail_s))
            out = out.with_columns(
                pl.when(is_settle)
                .then(pl.lit("transient"))
                .when(is_transient_dwell & ((pl.col("_phase_elapsed_s") < _TRANSIENT_S) | in_tail))
                .then(pl.lit("transient"))
                .when(is_angular_dwell & ((pl.col("_phase_elapsed_s") < _TRANSIENT_ANGULAR_S) | in_tail))
                .then(pl.lit("transient"))
                .otherwise(pl.col("phase_kind"))
                .alias("phase_kind")
            )

        return out, phase_coverage

    def _ground_truth_xy(self, episode: AlignedEpisodeBundle, sample_times_ns: pl.Series) -> tuple[pl.Series, pl.Series]:
        """Map-frame ground-truth x/y at ``sample_times_ns`` from the native /tf robot_tracks table, nulls without ground truth.

        Wheel odometry is never a substitute: without a ground-truth track both series are entirely null.
        """
        empty = pl.Series("_gt_x", [None] * len(sample_times_ns), dtype=pl.Float64)
        tracks = self.native_topics(episode).get("robot_tracks")
        if tracks is None or tracks.is_empty():
            return empty, empty.clone()
        required = {"t_ns", "x", "y"}
        if not required <= set(tracks.columns):
            raise ValueError(f"robot_tracks lacks {sorted(required - set(tracks.columns))}; the telemetry contract is (t_ns, x, y, yaw)")

        reference = (
            tracks.select(
                pl.col("t_ns").cast(pl.Int64),
                pl.col("x").cast(pl.Float64).alias("_gt_x"),
                pl.col("y").cast(pl.Float64).alias("_gt_y"),
            )
            .drop_nulls()
            .sort("t_ns")
        )
        query = pl.DataFrame({"t_ns": sample_times_ns.cast(pl.Int64)})
        joined = asof_join(query, reference, tolerance_ns=TRACK_ALIGN_TOLERANCE_NS)
        return joined["_gt_x"], joined["_gt_y"]

    def _ground_truth_yaw(self, episode: AlignedEpisodeBundle, sample_times_ns: pl.Series) -> pl.Series:
        """Ground-truth yaw at ``sample_times_ns`` from the native /tf robot_tracks table, nulls without it."""
        empty = pl.Series("_gt_yaw", [None] * len(sample_times_ns), dtype=pl.Float64)
        tracks = self.native_topics(episode).get("robot_tracks")
        if tracks is None or tracks.is_empty() or "yaw" not in tracks.columns:
            return empty
        reference = tracks.select(pl.col("t_ns").cast(pl.Int64), pl.col("yaw").cast(pl.Float64).alias("_gt_yaw")).drop_nulls().sort("t_ns")
        query = pl.DataFrame({"t_ns": sample_times_ns.cast(pl.Int64)})
        joined = asof_join(query, reference, tolerance_ns=TRACK_ALIGN_TOLERANCE_NS)
        return joined["_gt_yaw"]

    def _enrich(self, episode: AlignedEpisodeBundle, frame: pl.DataFrame) -> pl.DataFrame:
        """Add per-sample step duration/distance from ground truth, power, acoustic level, acceleration and efficiency."""
        sample_times_ns = frame["t_ns"]
        ground_truth_x, ground_truth_y = self._ground_truth_xy(episode, sample_times_ns)
        # The first sample has no previous step: its duration is NULL, never 0.
        dt = (sample_times_ns.diff().cast(pl.Float64) / _NANOSECONDS_PER_SECOND).clip(lower_bound=0.0)
        ds = ((ground_truth_x.diff() ** 2) + (ground_truth_y.diff() ** 2)).sqrt().clip(lower_bound=0.0)

        out = frame.with_columns(
            dt.alias("_dt"),
            ds.alias("_ds"),
        )

        if "total_mechanical_power_w" in out.columns:
            power_mech = pl.col("total_mechanical_power_w").cast(pl.Float64)
        elif "effort" in out.columns and "velocity" in out.columns:
            power_mech = (pl.col("effort") * pl.col("velocity")).list.eval(pl.element().abs()).list.sum()
        else:
            power_mech = pl.lit(None, dtype=pl.Float64)

        out = out.with_columns(
            pl.col("total_power_w").cast(pl.Float64).alias("_p_total") if "total_power_w" in out.columns else pl.lit(None, dtype=pl.Float64).alias("_p_total"),
            power_mech.alias("_p_mech"),
            pl.col("total_level_af_dba").cast(pl.Float64).alias("_dba") if "total_level_af_dba" in out.columns else pl.lit(None, dtype=pl.Float64).alias("_dba"),
        )

        if out["_dba"].null_count() == len(out) and "velocity" in out.columns and "effort" in out.columns:
            # Fallback: steady-state drivetrain model from joint states
            # (omega_eq = RMS wheel speed, t_eq = mean |effort|).
            profile = self._acoustic_profile
            joints = (
                out.select(["t_ns", "velocity", "effort"])
                .explode(["velocity", "effort"])
                .group_by("t_ns")
                .agg(
                    pl.col("velocity").pow(2).mean().sqrt().alias("_omega_eq"),
                    pl.col("effort").abs().mean().alias("_t_eq"),
                )
            )
            out = out.join(joints, on="t_ns", how="left").with_columns(
                pl.col("_omega_eq").clip(lower_bound=profile.omega_active_radps),
            )
            p_drive = (10.0 ** (profile.beta_0_db / 10.0)) * (pl.col("_omega_eq") / profile.omega_ref_radps) ** (profile.beta_1_db / 10.0) * (1.0 + pl.col("_t_eq") / profile.tau_ref_nm) ** (profile.beta_2_db / 10.0)
            out = out.with_columns((10.0 * ((10.0 ** (profile.l_base_0_db / 10.0)) + p_drive).log10()).alias("_dba"))

        return out

    def _series_frame(self, episode: AlignedEpisodeBundle) -> tuple[pl.DataFrame, float | None]:
        """Per-sample characterization frame with every derived series column, and the phase coverage."""
        frame, phase_coverage = self._attach_phases(episode)
        out = self._enrich(episode, frame)

        elapsed_s = (out["t_ns"].cast(pl.Float64) - out["t_ns"].cast(pl.Float64).min()) / _NANOSECONDS_PER_SECOND
        speed = out["vel_linear"].cast(pl.Float64).abs() if "vel_linear" in out.columns else pl.when(out["_dt"] > 1e-6).then(out["_ds"] / out["_dt"]).otherwise(None)
        turn_rate = out["vel_angular"].cast(pl.Float64).abs() if "vel_angular" in out.columns else pl.lit(None, dtype=pl.Float64)

        ground_truth_x, ground_truth_y = self._ground_truth_xy(episode, out["t_ns"])
        # The first sample has no previous step: its duration is NULL, never 0.
        step_dt = (out["t_ns"].diff().cast(pl.Float64) / _NANOSECONDS_PER_SECOND).clip(lower_bound=0.0)
        ground_truth_yaw = out["yaw"].cast(pl.Float64) if "yaw" in out.columns else self._ground_truth_yaw(episode, out["t_ns"])
        lateral_speed = pl.when(step_dt > 1e-9).then((-ground_truth_x.diff() * ground_truth_yaw.sin() + ground_truth_y.diff() * ground_truth_yaw.cos()) / step_dt).otherwise(None)
        energy_intensity = pl.when(speed >= _MIN_SPEED_MPS).then(out["_p_total"] / speed).otherwise(None)
        energy_per_rad = pl.when(turn_rate >= _MIN_TURN_RATE_RADPS).then(out["_p_total"] / turn_rate).otherwise(None)
        accel_achieved = pl.when(out["_dt"] > _MIN_ACCEL_DT_S).then(speed.diff() / out["_dt"]).otherwise(None)
        efficiency = pl.when(out["_p_total"] > _MIN_EFFICIENCY_POWER_W).then((out["_p_mech"] / out["_p_total"]).clip(0.0, 1.0)).otherwise(None)

        return (
            out.with_columns(
                elapsed_s.alias("_time_s"),
                speed.alias("_speed"),
                turn_rate.alias("_turn_rate"),
                out["turn_radius_m"].alias("_turn_radius"),
                lateral_speed.alias("_vel_lateral"),
                energy_intensity.alias("_energy_intensity"),
                energy_per_rad.alias("_energy_per_rad"),
                accel_achieved.alias("_accel_achieved"),
                efficiency.alias("_efficiency"),
            ),
            phase_coverage,
        )

    def missing_input_provenance(self, episode: AlignedEpisodeBundle) -> dict[str, str]:
        """Phase coverage is unknown, not zero, when no characterization markers were recorded."""
        if episode.data is None or "label" not in episode.data.columns:
            return {"char_phase_coverage": "char_phase_coverage:topic_missing:robot_characterization_phase"}
        return {}

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        if episode.data is None or len(episode.data) == 0:
            return {metric.name: None for metric in self.OUTPUTS}

        series, phase_coverage = self._series_frame(episode)
        mean_efficiency = series["_efficiency"].mean()
        mean_intensity = series["_energy_intensity"].mean()
        dba = series["_dba"].drop_nulls()
        leq_dba = 10.0 * math.log10(float((10.0 ** (dba / 10.0)).mean())) if len(dba) > 0 else None

        return {
            "char_phase_coverage": phase_coverage,
            "char_mean_efficiency": None if mean_efficiency is None else float(mean_efficiency),
            "char_leq_dba": leq_dba,
            "char_mean_energy_intensity_j_m": None if mean_intensity is None else float(mean_intensity),
            "char_sample_count": len(series),
        }

    def series_table(
        self,
        episode: AlignedEpisodeBundle,
        run_id: str = "",
        robot_id: str = "",
    ) -> pl.DataFrame:
        """1NF flat series table of characterization timeseries, one row per sample; empty with the same schema without data."""
        if episode.data is None or len(episode.data) == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)

        series, _coverage = self._series_frame(episode)
        return series.select(
            pl.lit(run_id).alias("run_id"),
            pl.lit(int(episode.episode_id), dtype=pl.Int64).alias("episode_id"),
            pl.lit(robot_id or (episode.robot_name or "robot")).alias("robot_id"),
            pl.col("t_ns").cast(pl.Int64).alias("t_ns"),
            pl.col("_time_s").alias("time_s"),
            pl.col("_p_total").alias("power_total_w"),
            pl.col("_p_mech").alias("power_mech_w"),
            pl.col("_dba").alias("dba"),
            pl.col("vel_linear").cast(pl.Float64).alias("vx_achieved_mps") if "vel_linear" in series.columns else pl.lit(None, dtype=pl.Float64).alias("vx_achieved_mps"),
            pl.col("_vel_lateral").alias("vy_achieved_mps"),
            pl.col("vel_angular").cast(pl.Float64).alias("wz_achieved_radps") if "vel_angular" in series.columns else pl.lit(None, dtype=pl.Float64).alias("wz_achieved_radps"),
            pl.col("_speed").alias("speed_achieved_mps"),
            pl.col("_accel_achieved").alias("accel_achieved_mps2"),
            pl.col("accel_target").alias("accel_target_mps2"),
            pl.col("_efficiency").alias("efficiency"),
            pl.col("phase_kind").fill_null("unknown").alias("phase_kind"),
            pl.col("vx_target").alias("vx_target_mps"),
            pl.col("vy_target").alias("vy_target_mps"),
            pl.col("wz_target").alias("wz_target_radps"),
            pl.col("vx_target").cast(pl.Float64).abs().alias("speed_target_mps"),
            pl.col("_turn_radius").alias("turn_radius_m"),
            pl.col("_energy_intensity").alias("energy_intensity_j_m"),
            pl.col("_energy_per_rad").alias("energy_per_rad_j_rad"),
            _leq_power(pl.col("_dba")).alias("leq_power_w"),
        )
