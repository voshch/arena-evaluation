"""Unit tests for the CharacterizationCalculator (ecological metric)."""

import polars as pl
import pytest
from polars.testing import assert_series_equal

from arena_evaluation.processing.metrics.ecological.characterization import (
    CharacterizationCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000


def _calc() -> CharacterizationCalculator:
    return CharacterizationCalculator(make_robot_params(model="jackal"))


def _episode(frame: pl.DataFrame, **kwargs) -> AlignedEpisodeBundle:
    """Bundle whose native /tf ground truth follows the aligned x/y columns."""
    tracks = pl.DataFrame(
        {
            "t_ns": frame["t_ns"].cast(pl.Int64),
            "x": frame["x"].cast(pl.Float64),
            "y": frame["y"].cast(pl.Float64),
            "yaw": frame["yaw"].cast(pl.Float64) if "yaw" in frame.columns else [0.0] * len(frame),
        }
    )
    topics: dict[str, pl.DataFrame] = {"robot_tracks": tracks}
    if "label" in frame.columns:
        topics["robot_characterization_phase"] = frame.select("t_ns", "label")
    if {"linear_x", "linear_y", "angular_z"} <= set(frame.columns):
        topics["robot_cmd_vel"] = frame.select("t_ns", "linear_x", "linear_y", "angular_z")
    if "total_power_w" in frame.columns:
        if "total_mech_power_w" not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("total_mech_power_w"))
        topics["robot_power"] = frame.select("t_ns", "total_power_w", "total_mech_power_w")
    if "total_level_af_dba" in frame.columns:
        topics["robot_acoustics"] = frame.select("t_ns", "total_level_af_dba")
    if "speed_mps" in frame.columns:
        if "omega_z" not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("omega_z"))
        topics["robot_odom"] = frame.select("t_ns", "speed_mps", "omega_z")
    if {"effort", "velocity"} <= set(frame.columns):
        topics["robot_joint_states"] = frame.select("t_ns", "effort", "velocity")
        frame = frame.drop("effort", "velocity")
    if "topics" in kwargs:
        topics.update(kwargs.pop("topics"))
    return AlignedEpisodeBundle(
        episode_id=0,
        data=frame,
        start_pos=[],
        goal_pos=[],
        robot_name="env_0_jackal",
        topics=topics,
        **kwargs,
    )


def _series(calc: CharacterizationCalculator, episode: AlignedEpisodeBundle) -> pl.DataFrame:
    return calc.series_table(episode, run_id="run", robot_id="robot")


def _sample_episode() -> AlignedEpisodeBundle:
    """A small aligned frame: odom + power + joints + phase labels."""
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "x": [0.0, 0.25, 0.75, 1.75],
            "y": [0.0, 0.0, 0.0, 0.0],
            "speed_mps": [0.0, 0.25, 0.5, 1.0],
            "total_power_w": [50.0, 55.0, 60.0, 65.0],
            "total_level_af_dba": [42.0, 46.0, 51.0, 56.0],
            "label": [
                "ramp_apex_vx_0.25_h_1.00",
                "ramp_apex_vx_0.25_h_1.00",
                "ramp_apex_vx_0.50_h_1.00",
                "ramp_apex_vx_0.50_h_1.00",
            ],
        }
    )
    joints = pl.DataFrame(
        {
            "t_ns": [0, 0, SEC, SEC, 2 * SEC, 2 * SEC, 3 * SEC, 3 * SEC],
            "effort": [0.1, 0.1, 0.5, -0.5, 1.0, 1.0, 1.0, -1.0],
            "velocity": [0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 8.0, 8.0],
        }
    )
    return _episode(frame, topics={"robot_joint_states": joints})


def test_speed_falls_back_to_path_rate_without_speed_mps():
    frame = _sample_episode().data.drop("speed_mps")
    table = _series(_calc(), _episode(frame))
    expected = pl.Series("energy_intensity_j_m", [None, 220.0, 120.0, 65.0], dtype=pl.Float64)
    assert_series_equal(table["energy_intensity_j_m"], expected, check_names=False, rel_tol=1e-9)


def test_arc_and_lateral_characterization_phases():
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "x": [0.0, 0.5, 1.0, 1.5],
            "y": [0.0, 0.1, 0.3, 0.6],
            "speed_mps": [0.5, 0.5, 1.0, 1.0],
            "omega_z": [0.5, 0.5, 1.0, 1.0],
            "total_power_w": [60.0, 60.0, 75.0, 75.0],
            "total_level_af_dba": [45.0, 45.0, 50.0, 50.0],
            "label": ["arc_vx_0.50_r_0.67_left"] * 2 + ["arc_vx_1.00_r_0.67_left"] * 2,
        }
    )
    calc = _calc()
    episode = _episode(frame)
    assert calc.calculate(episode, {})["char_phase_coverage"] == 1.0
    table = _series(calc, episode)
    # turn radius is the schedule's commanded radius, not the achieved vx/wz ratio
    assert abs(table["turn_radius_m"][0] - 0.6675) < 1e-4
    # Energy per rad: 60 W / 0.5 rad/s = 120 J/rad
    assert abs(table["energy_per_rad_j_rad"][0] - 120.0) < 1e-4


def test_calculate_missing_power_and_acoustics_never_models_the_level():
    sample = _sample_episode()
    frame = sample.data.drop(["total_power_w", "total_level_af_dba"])
    ep = _episode(frame, topics={"robot_joint_states": sample.topics["robot_joint_states"]})
    calc = _calc()
    out = calc.calculate(ep, {})

    # Missing power is an explicit null, never a silent zero; the level is never modelled from built-in constants.
    assert out["char_mean_efficiency"] is None
    assert out["char_leq_dba"] is None
    assert _series(calc, ep)["dba"].is_null().all()
    assert calc.missing_input_provenance(ep)["char_leq_dba"] == "char_leq_dba:topic_missing:robot_acoustics"


def test_unmatched_labels_lower_phase_coverage():
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "x": [0.0, 0.5],
            "y": [0.0, 0.0],
            "speed_mps": [0.5, 0.5],
            "total_power_w": [60.0, 60.0],
            "linear_x": [0.5, 0.5],
            "angular_z": [0.0, 0.0],
            "label": ["linear_vx_0.50", "linear_vx_99.00"],
        }
    )
    calc = _calc()
    episode = _episode(frame)
    assert calc.calculate(episode, {})["char_phase_coverage"] == 0.5
    assert _series(calc, episode)["phase_kind"][1] in ("linear", "transient")


def test_recorded_schedule_beats_the_rebuild():
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "x": [0.0, 0.5],
            "y": [0.0, 0.0],
            "speed_mps": [0.33, -0.75],
            "total_power_w": [60.0, 60.0],
            "label": ["arc_vx_0.33_r_0.90_left", "ramp_down_vx_0.75_h_2.00"],
        }
    )
    schedule = pl.DataFrame(
        {
            "phase_label": ["arc_vx_0.33_r_0.90_left", "ramp_down_vx_0.75_h_2.00"],
            "phase_kind": ["arc", "ramp_down"],
            "vx_target": [0.33, 0.75],
            "vy_target": [0.0, 0.0],
            "wz_target": [0.37, 0.0],
            "duration_s": [10.0, 2.0],
            "ramp_s": [0.0, 2.0],
            "turn_radius_m": [0.90, 0.0],
        }
    )
    calc = _calc()
    episode = _episode(frame)
    episode = AlignedEpisodeBundle(
        episode_id=episode.episode_id,
        data=episode.data,
        start_pos=episode.start_pos,
        goal_pos=episode.goal_pos,
        robot_name=episode.robot_name,
        topics={**episode.topics, "robot_characterization_schedule": schedule},
    )
    assert calc.calculate(episode, {})["char_phase_coverage"] == 1.0
    table = _series(calc, episode)
    assert_series_equal(
        table["vx_target_mps"],
        pl.Series("vx_target_mps", [0.33, 0.75], dtype=pl.Float64),
        check_names=False,
    )
    assert table["turn_radius_m"][0] == 0.90
    # ramp_down decelerates, so its accel target is the negative of the ramp rate
    assert table["accel_target_mps2"][1] == pytest.approx(-0.375)


def test_without_ground_truth_the_series_has_null_step_distances():
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "total_power_w": [50.0, 50.0, 50.0],
            "label": ["linear_vx_0.50"] * 3,
        }
    )
    episode = AlignedEpisodeBundle(
        episode_id=0,
        data=frame,
        start_pos=[],
        goal_pos=[],
        robot_name="env_0_jackal",
        topics={},
    )
    calc = _calc()
    # No odometry fallback: without /tf ground truth the path rate is unavailable, so intensity is null.
    assert calc.calculate(episode, {})["char_mean_energy_intensity_j_m"] is None
    assert _series(calc, episode)["energy_intensity_j_m"].null_count() == 3


def test_joints_and_power_work_with_production_aligned_schema():
    """Production aligned bundles lack effort/velocity and rename mech power to total_mech_power_w."""
    t_ns = [0, SEC, 2 * SEC]
    aligned = pl.DataFrame(
        {
            "t_ns": t_ns,
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "speed_mps": [1.0, 1.0, 1.0],
            "omega_z": [0.0, 0.0, 0.0],
            "total_power_w": [50.0, 50.0, 50.0],
            "total_mech_power_w": [10.0, 10.0, 10.0],
        }
    )
    tracks = pl.DataFrame(
        {
            "t_ns": t_ns,
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "speed_mps": [1.0, 1.0, 1.0],
            "omega_z": [0.0, 0.0, 0.0],
        }
    )
    power = pl.DataFrame(
        {
            "t_ns": t_ns,
            "total_power_w": [50.0, 50.0, 50.0],
            "total_mech_power_w": [10.0, 10.0, 10.0],
        }
    )
    joints = pl.DataFrame(
        {
            "t_ns": [t for t in t_ns for _ in range(2)],
            "effort": [1.0] * (2 * len(t_ns)),
            "velocity": [2.0] * (2 * len(t_ns)),
        }
    )
    power_only = AlignedEpisodeBundle(
        episode_id=0,
        data=aligned,
        start_pos=[],
        goal_pos=[],
        robot_name="env_0_jackal",
        topics={"robot_tracks": tracks, "robot_power": power},
    )
    joints_only = AlignedEpisodeBundle(
        episode_id=0,
        data=aligned.drop("total_power_w", "total_mech_power_w"),
        start_pos=[],
        goal_pos=[],
        robot_name="env_0_jackal",
        topics={"robot_tracks": tracks, "robot_joint_states": joints},
    )
    calc = _calc()
    power_series = _series(calc, power_only)
    assert power_series["power_mech_w"].to_list() == [10.0, 10.0, 10.0]
    power_out = calc.calculate(power_only, {})
    assert abs(power_out["char_mean_efficiency"] - 0.2) < 1e-6
    joints_series = _series(calc, joints_only)
    assert joints_series["power_mech_w"].to_list() == [4.0, 4.0, 4.0]
    joints_out = calc.calculate(joints_only, {})
    assert joints_out["char_sample_count"] == 3
    # Joints carry mechanical power, not an acoustic level: no built-in profile stands in for the recording.
    assert joints_out["char_leq_dba"] is None
