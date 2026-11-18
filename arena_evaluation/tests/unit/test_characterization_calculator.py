"""Unit tests for the CharacterizationCalculator (ecological metric).

Pure Python + Polars, no ROS, no Gazebo required.
"""

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from arena_evaluation.processing.metrics.ecological.characterization import (
    SERIES_SCHEMA,
    CharacterizationCalculator,
    _leq_power,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000


def _calc() -> CharacterizationCalculator:
    return CharacterizationCalculator(RobotParams.load("jackal"))


def _episode(frame: pl.DataFrame, **kwargs) -> AlignedEpisodeBundle:
    """Bundle whose native /tf ground truth follows the aligned pos_x/pos_y columns."""
    tracks = pl.DataFrame(
        {
            "t_ns": frame["t_ns"].cast(pl.Int64),
            "x": frame["pos_x"].cast(pl.Float64),
            "y": frame["pos_y"].cast(pl.Float64),
            "yaw": [0.0] * len(frame),
        }
    )
    return AlignedEpisodeBundle(episode_id=0, data=frame, start_pos=[], goal_pos=[], robot_name="env_0_jackal", topics={"robot_tracks": tracks}, **kwargs)


def _series(calc: CharacterizationCalculator, episode: AlignedEpisodeBundle) -> pl.DataFrame:
    return calc.series_table(episode, run_id="run", robot_id="robot")


def test_output_keys_are_scalars_only():
    calc = _calc()
    keys = calc.output_keys()
    assert keys == [
        "char_phase_coverage",
        "char_mean_efficiency",
        "char_leq_dba",
        "char_mean_energy_intensity_j_m",
        "char_sample_count",
    ]
    units = {metric.name: metric.unit for metric in calc.OUTPUTS}
    assert units["char_leq_dba"] == "dBA"
    assert units["char_mean_energy_intensity_j_m"] == "J/m"
    assert not any("timeseries" in key for key in keys)


def test_leq_power():
    values = pl.Series([50.0, 60.0])
    mean_power = float(_leq_power(values).mean())
    assert abs(10.0 * math.log10(mean_power) - 10.0 * math.log10((10**5 + 10**6) / 2)) < 1e-6


def _sample_episode() -> AlignedEpisodeBundle:
    """A small aligned frame: odom + power + joints + phase labels."""
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "pos_x": [0.0, 0.25, 0.75, 1.75],
            "pos_y": [0.0, 0.0, 0.0, 0.0],
            "vel_linear": [0.0, 0.25, 0.5, 1.0],
            "total_power_w": [50.0, 55.0, 60.0, 65.0],
            "total_level_af_dba": [42.0, 46.0, 51.0, 56.0],
            "velocity": [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [8.0, 8.0]],
            "effort": [[0.1, 0.1], [0.5, -0.5], [1.0, 1.0], [1.0, -1.0]],
            "label": [
                "ramp_apex_vx_0.25_h_1.00",
                "ramp_apex_vx_0.25_h_1.00",
                "ramp_apex_vx_0.50_h_1.00",
                "ramp_apex_vx_0.50_h_1.00",
            ],
        }
    )
    return _episode(frame)


def test_calculate_reports_scalar_summaries():
    calc = _calc()
    episode = _sample_episode()
    out = calc.calculate(episode, {})
    assert out["char_phase_coverage"] == 1.0
    assert out["char_sample_count"] == 4
    # mean of 0/50, 2/55, 8/60, 16/65
    assert out["char_mean_efficiency"] == pytest.approx(sum([0.0, 2 / 55, 8 / 60, 16 / 65]) / 4)
    assert out["char_mean_energy_intensity_j_m"] == pytest.approx((220.0 + 120.0 + 65.0) / 3)
    assert out["char_leq_dba"] == pytest.approx(10.0 * math.log10(sum(10 ** (value / 10.0) for value in [42.0, 46.0, 51.0, 56.0]) / 4))


def test_series_table_is_flat_1nf_with_ground_truth_columns():
    calc = _calc()
    table = _series(calc, _sample_episode())
    assert table.schema == pl.Schema(SERIES_SCHEMA)
    expected = pl.DataFrame(
        {
            "run_id": ["run"] * 4,
            "episode_id": [0] * 4,
            "robot_id": ["robot"] * 4,
            "t_ns": [0, SEC, 2 * SEC, 3 * SEC],
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "power_total_w": [50.0, 55.0, 60.0, 65.0],
            "power_mech_w": [0.0, 2.0, 8.0, 16.0],
            "dba": [42.0, 46.0, 51.0, 56.0],
            "vx_achieved_mps": [0.0, 0.25, 0.5, 1.0],
            "vy_achieved_mps": [None, 0.0, 0.0, 0.0],
            "wz_achieved_radps": [None] * 4,
            "speed_achieved_mps": [0.0, 0.25, 0.5, 1.0],
            "accel_achieved_mps2": [None, 0.25, 0.25, 0.5],
            "accel_target_mps2": [0.0, 0.0, 0.0, 0.0],
            "efficiency": [0.0, 2 / 55, 8 / 60, 16 / 65],
            "phase_kind": ["ramp_apex"] * 4,
            "vx_target_mps": [0.25, 0.25, 0.5, 0.5],
            "vy_target_mps": [0.0] * 4,
            "wz_target_radps": [0.0] * 4,
            "speed_target_mps": [0.25, 0.25, 0.5, 0.5],
            "turn_radius_m": [0.0] * 4,
            "energy_intensity_j_m": [None, 220.0, 120.0, 65.0],
            "energy_per_rad_j_rad": [None] * 4,
            "leq_power_w": [10 ** (level / 10.0) for level in [42.0, 46.0, 51.0, 56.0]],
        },
        schema=SERIES_SCHEMA,
    )
    assert_frame_equal(table, expected, check_dtypes=True, rel_tol=1e-9)


def test_speed_falls_back_to_path_rate_without_vel_linear():
    frame = _sample_episode().data.drop("vel_linear")
    table = _series(_calc(), _episode(frame))
    expected = pl.Series("energy_intensity_j_m", [None, 220.0, 120.0, 65.0], dtype=pl.Float64)
    assert_series_equal(table["energy_intensity_j_m"], expected, check_names=False, rel_tol=1e-9)


def test_steady_state_transient_filtering():
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "pos_x": [0.0, 0.25, 0.50, 0.75],
            "pos_y": [0.0, 0.0, 0.0, 0.0],
            "vel_linear": [0.0, 0.25, 0.25, 0.25],
            "total_power_w": [500.0, 200.0, 42.2, 42.2],
            "label": ["linear_vx_0.25"] * 4,
        }
    )
    table = _series(_calc(), _episode(frame))
    expected = pl.Series("phase_kind", ["transient", "transient", "linear", "linear"], dtype=pl.String)
    assert_series_equal(table["phase_kind"], expected, check_names=False)


def test_arc_and_lateral_characterization_phases():
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "pos_x": [0.0, 0.5, 1.0, 1.5],
            "pos_y": [0.0, 0.1, 0.3, 0.6],
            "vel_linear": [0.5, 0.5, 1.0, 1.0],
            "vel_angular": [0.5, 0.5, 1.0, 1.0],
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


def test_calculate_missing_power_and_acoustics():
    frame = _sample_episode().data.drop(["total_power_w", "total_level_af_dba"])
    out = _calc().calculate(_episode(frame), {})
    # Missing power is an explicit null, never a silent zero, and acoustics falls back to the joint model.
    assert out["char_mean_efficiency"] is None
    assert out["char_leq_dba"] is not None


def test_calculate_empty_returns_none_keys():
    episode = AlignedEpisodeBundle(episode_id=0, data=pl.DataFrame(), start_pos=[], goal_pos=[], robot_name="env_0_jackal")
    out = _calc().calculate(episode, {})
    assert all(value is None for value in out.values())
    assert _series(_calc(), episode).schema == pl.Schema(SERIES_SCHEMA)


def test_energy_intensity_speed_threshold_gating():
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "pos_x": [0.0, 0.01, 0.50],
            "pos_y": [0.0, 0.0, 0.0],
            "vel_linear": [0.0, 0.01, 0.49],
            "total_power_w": [50.0, 50.0, 50.0],
            "total_level_af_dba": [42.0, 42.0, 42.0],
            "label": ["ramp_up_vx_0.50_h_1.00"] * 3,
        }
    )
    table = _series(_calc(), _episode(frame))
    assert table["energy_intensity_j_m"][0] is None
    assert table["energy_intensity_j_m"][1] is None
    assert table["energy_intensity_j_m"][2] == pytest.approx(50.0 / 0.49)


def test_accel_target_is_signed_by_direction_of_change():
    frame = pl.DataFrame(
        {
            "t_ns": [0, 500_000_000, 1_000_000_000, 1_500_000_000],
            "pos_x": [0.0, 0.5, 1.0, 1.5],
            "pos_y": [0.0, 0.0, 0.0, 0.0],
            "vel_linear": [1.0, 0.5, -1.0, -0.5],
            "total_power_w": [60.0, 60.0, 60.0, 60.0],
            "label": [
                "ramp_up_vx_1.00_h_1.00",
                "ramp_down_vx_1.00_h_1.00",
                "ramp_up_vx_-1.00_h_1.00",
                "ramp_down_vx_-1.00_h_1.00",
            ],
        }
    )
    calc = _calc()
    episode = _episode(frame)
    assert calc.calculate(episode, {})["char_phase_coverage"] == 1.0
    expected = pl.Series("accel_target_mps2", [1.0, -1.0, -1.0, 1.0], dtype=pl.Float64)
    assert_series_equal(_series(calc, episode)["accel_target_mps2"], expected, check_names=False)


def test_unmatched_labels_lower_phase_coverage():
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "pos_x": [0.0, 0.5],
            "pos_y": [0.0, 0.0],
            "vel_linear": [0.5, 0.5],
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
            "pos_x": [0.0, 0.5],
            "pos_y": [0.0, 0.0],
            "vel_linear": [0.33, -0.75],
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
    assert_series_equal(table["vx_target_mps"], pl.Series("vx_target_mps", [0.33, 0.75], dtype=pl.Float64), check_names=False)
    assert table["turn_radius_m"][0] == 0.90
    # ramp_down decelerates, so its accel target is the negative of the ramp rate
    assert table["accel_target_mps2"][1] == pytest.approx(-0.375)


def test_duplicate_recorded_schedule_rows_do_not_duplicate_samples():
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "pos_x": [0.0, 0.5],
            "pos_y": [0.0, 0.0],
            "vel_linear": [0.5, 0.5],
            "total_power_w": [60.0, 60.0],
            "label": ["linear_vx_0.50", "linear_vx_0.50"],
        }
    )
    schedule = pl.DataFrame(
        {
            "phase_label": ["linear_vx_0.50", "linear_vx_0.50"],
            "phase_kind": ["linear", "linear"],
            "vx_target": [0.5, 0.55],
            "vy_target": [0.0, 0.0],
            "wz_target": [0.0, 0.0],
            "duration_s": [5.0, 5.0],
            "ramp_s": [0.0, 0.0],
            "turn_radius_m": [0.0, 0.0],
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
    table = _series(calc, episode)
    assert len(table) == 2
    assert_series_equal(table["vx_target_mps"], pl.Series("vx_target_mps", [0.55, 0.55], dtype=pl.Float64), check_names=False)


def test_steady_state_bidirectional_windowing():
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000, 5_000_000_000],
            "pos_x": [0.0, 0.25, 0.50, 0.75, 1.0, 1.25],
            "pos_y": [0.0] * 6,
            "vel_linear": [0.0, 0.25, 0.25, 0.25, 0.25, 0.0],
            "total_power_w": [500.0, 200.0, 42.2, 42.2, 42.2, 150.0],
            "label": ["linear_vx_0.25"] * 6,
        }
    )
    table = _series(_calc(), _episode(frame))
    expected = pl.Series("phase_kind", ["transient", "transient", "linear", "linear", "linear", "transient"], dtype=pl.String)
    assert_series_equal(table["phase_kind"], expected, check_names=False)


def test_without_ground_truth_the_series_has_null_step_distances():
    frame = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "pos_x": [0.0, 1.0, 2.0],
            "pos_y": [0.0, 0.0, 0.0],
            "total_power_w": [50.0, 50.0, 50.0],
            "label": ["linear_vx_0.50"] * 3,
        }
    )
    episode = AlignedEpisodeBundle(episode_id=0, data=frame, start_pos=[], goal_pos=[], robot_name="env_0_jackal", topics={})
    calc = _calc()
    # No odometry fallback: without /tf ground truth the path rate is unavailable, so intensity is null.
    assert calc.calculate(episode, {})["char_mean_energy_intensity_j_m"] is None
    assert _series(calc, episode)["energy_intensity_j_m"].null_count() == 3


def test_phase_coverage_is_null_without_characterization_markers():
    """No marker stream means no coverage, not a fabricated 0.0."""
    frame = _sample_episode().data.drop("label")
    episode = _episode(frame)
    calc = _calc()
    assert calc.calculate(episode, {})["char_phase_coverage"] is None
    assert calc.missing_input_provenance(episode)["char_phase_coverage"] == "char_phase_coverage:topic_missing:robot_characterization_phase"


def test_vy_achieved_is_derived_from_the_ground_truth_motion():
    """A straight, non-holonomic run has zero body-frame lateral velocity: produced, not null."""
    series = _series(_calc(), _sample_episode())
    lateral = series.get_column("vy_achieved_mps")
    assert lateral.null_count() == 1  # the first sample has no previous step
    assert lateral.drop_nulls().abs().max() < 1e-9
