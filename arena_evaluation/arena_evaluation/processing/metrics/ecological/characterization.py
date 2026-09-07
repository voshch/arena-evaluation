"""Open-loop characterization metrics (energy/acoustic profiles per working point).

Per-episode calculator: attaches the recorded ``characterization_phase`` markers
to every sample, resolves each label against the ``characterization_schedule``
table the sweep published (rebuilt from the robot envelope only when a recording
predates that topic), computes per-sample power / mechanical power / acoustic
level / energy intensity, and exposes them as ``timeseries_char_*`` list columns
in the metrics row, the same wide per-episode shape as the energy calculator's
``timeseries_power_*`` columns. The report layer derives long-format frames and
per-working-point aggregates from these columns (see the ``line`` plot type and
the ``characterization`` report manifest).
"""

from __future__ import annotations

import logging
import typing

import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


logger = logging.getLogger(__name__)

# Last-resort constants, mirroring config/acoustic_profile.yaml.
_ACOUSTIC_DEFAULTS = {
    "L_base_0": 42.0,
    "beta_0": 45.0,
    "beta_1": 18.0,
    "beta_2": 5.0,
    "omega_ref": 5.0,
    "tau_ref": 10.0,
    "omega_active": 0.2,
}


# Head of a dwell where the robot is still accelerating, so not yet steady state.
_TRANSIENT_S = 1.5  # s, linear, lateral and arc dwells
_TRANSIENT_ANGULAR_S = 1.0  # s, in-place pivots


def _leq_power(dba: pl.Series) -> pl.Series:
    """Linear acoustic power proxy 10^(L/10), L_Aeq = 10*log10(mean(x))."""
    return 10.0 ** (dba / 10.0)


class CharacterizationCalculator(BaseMetricCalculator):
    NAME = "characterization"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["odom"]

    UNITS = {
        "timeseries_char_time_s": "s",
        "timeseries_char_power_total_w": "W",
        "timeseries_char_power_mech_w": "W",
        "timeseries_char_dba": "dBA",
        "timeseries_char_vx_achieved": "m/s",
        "timeseries_char_vy_achieved": "m/s",
        "timeseries_char_wz_achieved": "rad/s",
        "timeseries_char_speed_achieved": "m/s",
        "timeseries_char_accel_achieved": "m/s^2",
        "timeseries_char_accel_target": "m/s^2",
        "timeseries_char_efficiency": "",
        "timeseries_char_vx_target": "m/s",
        "timeseries_char_vy_target": "m/s",
        "timeseries_char_wz_target": "rad/s",
        "timeseries_char_speed_target": "m/s",
        "timeseries_char_turn_radius_m": "m",
        "timeseries_char_energy_intensity": "J/m",
        "timeseries_char_energy_per_rad": "J/rad",
        "char_phase_coverage": "",
    }

    _TIMESERIES_KEYS = [
        "timeseries_char_time_s",
        "timeseries_char_power_total_w",
        "timeseries_char_power_mech_w",
        "timeseries_char_dba",
        "timeseries_char_vx_achieved",
        "timeseries_char_vy_achieved",
        "timeseries_char_wz_achieved",
        "timeseries_char_speed_achieved",
        "timeseries_char_accel_achieved",
        "timeseries_char_accel_target",
        "timeseries_char_efficiency",
        "timeseries_char_phase_kind",
        "timeseries_char_vx_target",
        "timeseries_char_vy_target",
        "timeseries_char_wz_target",
        "timeseries_char_speed_target",
        "timeseries_char_turn_radius_m",
        "timeseries_char_energy_intensity",
        "timeseries_char_energy_per_rad",
        "timeseries_char_leq_power",
    ]

    # Fraction of samples whose recorded label the schedule table recognised.
    _SUMMARY_KEYS = ["char_phase_coverage"]

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

    _phase_coverage: float = 1.0

    @classmethod
    def output_keys(cls) -> list[str]:
        return [*cls._TIMESERIES_KEYS, *cls._SUMMARY_KEYS]

    def _rebuilt_schedule(self) -> pl.DataFrame:
        """The default schedule for this robot's envelope, for recordings that carry no schedule table."""
        try:
            from task_generator.tasks.robots.characterization.schedule import build_schedule, resolve_envelope
        except ImportError:
            logger.warning("task_generator is not importable, characterization phases fall back to cmd_vel classification")
            return pl.DataFrame(schema=self._SCHEDULE_SCHEMA)

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
                "phase_label": p.name,
                "phase_kind": p.kind.value,
                "vx_target": p.vx_target,
                "vy_target": p.vy_target,
                "wz_target": p.wz_target,
                "duration_s": p.duration_s,
                "ramp_s": p.ramp_s,
                "turn_radius_m": p.radius_m,
            }
            for p in schedule
        ]
        return pl.DataFrame(rows, schema=self._SCHEDULE_SCHEMA)

    def _phase_map(self, episode: AlignedEpisodeBundle) -> pl.DataFrame:
        """Labels and targets of the schedule that ran, with the signed acceleration each ramp commands."""
        recorded = self.native_topics(episode).get("characterization_schedule")
        if recorded is not None and len(recorded) > 0:
            table = recorded.select(list(self._SCHEDULE_SCHEMA)).unique(subset=["phase_label"], keep="last", maintain_order=True)
        else:
            logger.warning("characterization: no recorded schedule, rebuilding the default one from the robot envelope")
            table = self._rebuilt_schedule()

        accel = pl.when(pl.col("ramp_s") > 0.0).then(pl.col("vx_target") / pl.col("ramp_s")).otherwise(0.0)
        accel = pl.when(pl.col("phase_kind") == "ramp_down").then(-accel).otherwise(accel)
        return table.with_columns(accel.alias("accel_target")).drop(["duration_s", "ramp_s"])

    def _attach_phases(self, episode: AlignedEpisodeBundle) -> pl.DataFrame:
        out = episode.data
        if "label" not in out.columns:
            # No phase markers at all: not a characterization episode, every sample classifies from cmd_vel.
            out = out.with_columns(pl.lit("unknown").alias("phase_label"))
            phase_map = pl.DataFrame(schema=self._SCHEDULE_SCHEMA).drop(["duration_s", "ramp_s"]).with_columns(pl.lit(0.0).alias("accel_target"))
        else:
            out = out.with_columns(pl.col("label").fill_null("unknown").alias("phase_label"))
            phase_map = self._phase_map(episode)
        out = out.join(phase_map, on="phase_label", how="left")

        unmatched = int(out["phase_kind"].null_count())
        self._phase_coverage = 1.0 - unmatched / len(out) if len(out) else 1.0
        # Unmarked samples ("unknown") are expected before the first marker, foreign labels are not.
        foreign = sorted(set(out.filter(pl.col("phase_kind").is_null())["phase_label"].to_list()) - {"unknown"})
        if foreign:
            logger.warning(f"characterization: {unmatched}/{len(out)} samples carry a label the schedule table does not contain, so they fall back to cmd_vel classification. First unmatched labels: {foreign[:5]}")

        # Fallback classification for unmarked samples (markers never recorded).
        cmd_vx = pl.col("linear_x").cast(pl.Float64).fill_null(0.0) if "linear_x" in out.columns else pl.lit(0.0)
        cmd_vy = pl.col("linear_y").cast(pl.Float64).fill_null(0.0) if "linear_y" in out.columns else pl.lit(0.0)
        cmd_wz = pl.col("angular_z").cast(pl.Float64).fill_null(0.0) if "angular_z" in out.columns else pl.lit(0.0)
        fb_kind = pl.when((cmd_vx == 0.0) & (cmd_vy == 0.0) & (cmd_wz == 0.0)).then(pl.lit("idle")).when((cmd_vy != 0.0) & (cmd_vx == 0.0) & (cmd_wz == 0.0)).then(pl.lit("lateral")).when((cmd_vx != 0.0) & (cmd_wz != 0.0)).then(pl.lit("arc")).when(cmd_wz != 0.0).then(pl.lit("angular")).otherwise(pl.lit("linear"))
        out = out.with_columns(
            pl.col("phase_kind").fill_null(fb_kind).alias("phase_kind"),
            pl.col("vx_target").fill_null(cmd_vx).alias("vx_target"),
            pl.col("vy_target").fill_null(cmd_vy).alias("vy_target"),
            pl.col("wz_target").fill_null(cmd_wz).alias("wz_target"),
            pl.col("accel_target").fill_null(0.0).alias("accel_target"),
            pl.col("turn_radius_m").fill_null(pl.when((cmd_vx.abs() >= 0.05) & (cmd_wz.abs() >= 0.05)).then(cmd_vx.abs() / cmd_wz.abs()).otherwise(0.0)).alias("turn_radius_m"),
        )

        # Split the transient windows off the steady-state part of every dwell:
        # the accelerating head, plus the trailing decel/cutoff on long blocks.
        if "time_ns" in out.columns and len(out) > 0:
            out = out.with_columns(((pl.col("phase_label") != pl.col("phase_label").shift(1)) | (pl.col("vx_target") != pl.col("vx_target").shift(1)) | (pl.col("vy_target") != pl.col("vy_target").shift(1)) | (pl.col("wz_target") != pl.col("wz_target").shift(1))).fill_null(True).cum_sum().alias("_phase_block_id"))
            out = out.with_columns(
                ((pl.col("time_ns") - pl.col("time_ns").min().over("_phase_block_id")).cast(pl.Float64) / 1e9).alias("_phase_elapsed_s"),
                ((pl.col("time_ns").max().over("_phase_block_id") - pl.col("time_ns").min().over("_phase_block_id")).cast(pl.Float64) / 1e9).alias("_phase_total_duration_s"),
            )
            is_settle = pl.col("phase_label").str.contains("settle")
            is_linear_dwell = pl.col("phase_label").str.starts_with("linear_vx_")
            is_lateral_dwell = pl.col("phase_label").str.starts_with("lateral_vy_")
            is_arc_dwell = pl.col("phase_label").str.starts_with("arc_vx_")
            is_trans_dwell = is_linear_dwell | is_lateral_dwell | is_arc_dwell
            is_angular_dwell = pl.col("phase_label").str.starts_with("angular_wz_")
            is_long_block = pl.col("_phase_total_duration_s") >= 4.0
            t_out = pl.min_horizontal(pl.lit(0.8), pl.col("_phase_total_duration_s") * 0.15)
            in_tail = is_long_block & (pl.col("_phase_elapsed_s") > (pl.col("_phase_total_duration_s") - t_out))
            out = out.with_columns(
                pl.when(is_settle)
                .then(pl.lit("transient"))
                .when(is_trans_dwell & ((pl.col("_phase_elapsed_s") < _TRANSIENT_S) | in_tail))
                .then(pl.lit("transient"))
                .when(is_angular_dwell & ((pl.col("_phase_elapsed_s") < _TRANSIENT_ANGULAR_S) | in_tail))
                .then(pl.lit("transient"))
                .otherwise(pl.col("phase_kind"))
                .alias("phase_kind")
            )

        return out

    def _acoustic_model(self) -> dict:
        """Profile backing the fallback model: per-robot, then shared, then built-in."""
        from pathlib import Path

        import yaml
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

        try:
            share = Path(get_package_share_directory("arena_robots"))
        except PackageNotFoundError:
            logger.warning("arena_robots not installed, using built-in acoustic constants")
            return dict(_ACOUSTIC_DEFAULTS)

        per_robot = share / "robots" / self.robot_params.model / "telemetry" / "acoustics.yaml"
        for cand in (per_robot, share / "config" / "acoustic_profile.yaml"):
            if not cand.is_file():
                continue
            try:
                cfg = yaml.safe_load(cand.read_text())
            except (OSError, yaml.YAMLError) as e:
                logger.warning(f"unreadable acoustic profile {cand}: {e!r}")
                continue
            missing = sorted(k for k in _ACOUSTIC_DEFAULTS if k not in cfg)
            if missing:
                logger.warning(f"acoustic profile {cand} lacks {missing}, using built-in values for those")
            return {k: float(cfg.get(k, _ACOUSTIC_DEFAULTS[k])) for k in _ACOUSTIC_DEFAULTS}

        logger.warning(f"no acoustic profile for {self.robot_params.model!r} at {per_robot}, using built-in acoustic constants")
        return dict(_ACOUSTIC_DEFAULTS)

    def _enrich(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df.with_columns(
            (pl.col("time_ns").diff().fill_null(0).cast(pl.Float64) / 1e9).clip(lower_bound=0.0).alias("_dt"),
            ((pl.col("pos_x").diff().fill_null(0.0).cast(pl.Float64) ** 2) + (pl.col("pos_y").diff().fill_null(0.0).cast(pl.Float64) ** 2)).sqrt().clip(lower_bound=0.0).alias("_ds"),
        )
        if "total_mechanical_power_w" in out.columns:
            out = out.with_columns(pl.col("total_mechanical_power_w").cast(pl.Float64).fill_null(0.0).alias("_p_mech"))
        elif "effort" in out.columns and "velocity" in out.columns:
            out = out.with_columns((pl.col("effort") * pl.col("velocity")).list.eval(pl.element().abs()).list.sum().fill_null(0.0).alias("_p_mech"))
        else:
            out = out.with_columns(pl.lit(0.0).alias("_p_mech"))

        out = out.with_columns(
            pl.col("total_power_w").cast(pl.Float64).fill_null(0.0).alias("_p_total") if "total_power_w" in out.columns else pl.lit(0.0).alias("_p_total"),
            pl.col("total_level_af_dba").cast(pl.Float64).alias("_dba") if "total_level_af_dba" in out.columns else pl.lit(None, dtype=pl.Float64).alias("_dba"),
        )
        if out["_dba"].null_count() == len(out) and "velocity" in out.columns:
            # Fallback: steady-state drivetrain model from joint states
            # (omega_eq = RMS wheel speed, t_eq = mean |effort|).
            m = self._acoustic_model()
            joints = (
                out.select(["time_ns", "velocity", "effort"])
                .explode(["velocity", "effort"])
                .group_by("time_ns")
                .agg(
                    pl.col("velocity").pow(2).mean().sqrt().alias("_omega_eq"),
                    pl.col("effort").abs().mean().alias("_t_eq"),
                )
            )
            out = out.join(joints, on="time_ns", how="left").with_columns(
                pl.col("_omega_eq").fill_null(0.0).clip(lower_bound=m["omega_active"]),
                pl.col("_t_eq").fill_null(0.0),
            )
            p_drive = (10.0 ** (m["beta_0"] / 10.0)) * (out["_omega_eq"] / m["omega_ref"]) ** (m["beta_1"] / 10.0) * (1.0 + out["_t_eq"] / m["tau_ref"]) ** (m["beta_2"] / 10.0)
            out = out.with_columns((10.0 * ((10.0 ** (m["L_base_0"] / 10.0)) + p_drive).log10()).alias("_dba"))
        else:
            out = out.with_columns(pl.col("_dba").fill_null(0.0))

        return out

    def calculate(self, episode: AlignedEpisodeBundle, dependencies: dict[str, typing.Any]) -> dict[str, typing.Any]:
        df = episode.data
        if df is None or len(df) == 0:
            return {k: None for k in self.output_keys()}

        out = self._attach_phases(episode)
        out = self._enrich(out)

        t_s = (out["time_ns"].cast(pl.Float64) - out["time_ns"].cast(pl.Float64).min()) / 1e9
        ds = out["_ds"].fill_null(0.0)
        dt = out["_dt"].fill_null(0.0)
        if "vel_linear" in out.columns:
            speed = out["vel_linear"].cast(pl.Float64).abs().fill_null(0.0)
        else:
            speed = pl.when(dt > 1e-6).then(ds / dt).otherwise(0.0)

        wz_achieved = out["vel_angular"].cast(pl.Float64).abs().fill_null(0.0) if "vel_angular" in out.columns else pl.lit(0.0)

        out = out.with_columns(
            pl.when(speed >= 0.05).then(out["_p_total"] / speed).otherwise(None).alias("_e_per_m"),
            pl.when(wz_achieved >= 0.05).then(out["_p_total"] / wz_achieved).otherwise(None).alias("_e_per_rad"),
            out["turn_radius_m"].fill_null(0.0).alias("_turn_radius"),
        )

        import numpy as np

        # Compute achieved linear acceleration (dv/dt)
        if "vel_linear" in out.columns:
            vx_arr = out["vel_linear"].cast(pl.Float64).fill_null(0.0).to_numpy()
        else:
            vx_arr = speed.to_numpy()
        dt_arr = dt.to_numpy()

        accel_achieved = np.zeros_like(vx_arr)
        for i in range(1, len(vx_arr)):
            if dt_arr[i] > 1e-4:
                accel_achieved[i] = (vx_arr[i] - vx_arr[i - 1]) / dt_arr[i]

        p_tot_arr = out["_p_total"].to_numpy()
        p_mech_arr = out["_p_mech"].to_numpy()
        eff_arr = np.zeros_like(p_tot_arr)
        for i in range(len(p_tot_arr)):
            if p_tot_arr[i] > 0.1:
                eff_arr[i] = max(0.0, min(1.0, float(p_mech_arr[i] / p_tot_arr[i])))

        rows = {
            "timeseries_char_time_s": t_s.to_list(),
            "timeseries_char_power_total_w": out["_p_total"].to_list(),
            "timeseries_char_power_mech_w": out["_p_mech"].to_list(),
            "timeseries_char_dba": out["_dba"].to_list(),
            "timeseries_char_vx_achieved": out["vel_linear"].cast(pl.Float64).fill_null(0.0).to_list()
            if "vel_linear" in out.columns else [0.0] * len(out),
            "timeseries_char_vy_achieved": out["vel_lateral"].cast(pl.Float64).fill_null(0.0).to_list()
            if "vel_lateral" in out.columns else [0.0] * len(out),
            "timeseries_char_wz_achieved": out["vel_angular"].cast(pl.Float64).fill_null(0.0).to_list()
            if "vel_angular" in out.columns else [0.0] * len(out),
            "timeseries_char_speed_achieved": out["vel_linear"].cast(pl.Float64).abs().fill_null(0.0).to_list()
            if "vel_linear" in out.columns else [0.0] * len(out),
            "timeseries_char_accel_achieved": accel_achieved.tolist(),
            "timeseries_char_accel_target": out["accel_target"].fill_null(0.0).to_list(),
            "timeseries_char_efficiency": eff_arr.tolist(),
            "timeseries_char_phase_kind": out["phase_kind"].fill_null("unknown").to_list(),
            "timeseries_char_vx_target": out["vx_target"].fill_null(0.0).to_list(),
            "timeseries_char_vy_target": out["vy_target"].fill_null(0.0).to_list(),
            "timeseries_char_wz_target": out["wz_target"].fill_null(0.0).to_list(),
            "timeseries_char_speed_target": out["vx_target"].cast(pl.Float64).abs().fill_null(0.0).to_list(),
            "timeseries_char_turn_radius_m": out["_turn_radius"].to_list(),
            "timeseries_char_energy_intensity": out["_e_per_m"].to_list(),
            "timeseries_char_energy_per_rad": out["_e_per_rad"].to_list(),
            "timeseries_char_leq_power": _leq_power(out["_dba"].fill_null(0.0)).to_list(),
            "char_phase_coverage": self._phase_coverage,
        }
        return rows
