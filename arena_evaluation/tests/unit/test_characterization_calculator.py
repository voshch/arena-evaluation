"""Unit tests for the CharacterizationCalculator (ecological metric).

Pure Python + Polars, no ROS, no Gazebo required.
"""

import polars as pl

from arena_evaluation.processing.metrics.ecological.characterization import (
    CharacterizationCalculator,
    _leq_power,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


def _calc() -> CharacterizationCalculator:
    return CharacterizationCalculator(RobotParams.load("jackal"))


def test_output_keys_and_units():
    calc = _calc()
    keys = calc.output_keys()
    assert "timeseries_char_power_total_w" in keys
    assert "timeseries_char_phase_kind" in keys
    assert "timeseries_char_vx_achieved" in keys
    assert "timeseries_char_vy_achieved" in keys
    assert "timeseries_char_wz_achieved" in keys
    assert "timeseries_char_turn_radius_m" in keys
    assert "timeseries_char_energy_intensity" in keys
    assert "timeseries_char_energy_per_rad" in keys
    assert calc.UNITS["timeseries_char_power_total_w"] == "W"
    assert calc.UNITS["timeseries_char_dba"] == "dBA"
    assert calc.UNITS["timeseries_char_wz_achieved"] == "rad/s"
    assert calc.UNITS["timeseries_char_energy_per_rad"] == "J/rad"
    assert "char_phase_coverage" in keys
    assert len(keys) == len(set(keys))


def test_leq_power():
    import math

    s = pl.Series([50.0, 60.0])
    mean_power = float(_leq_power(s).mean())
    assert abs(10.0 * math.log10(mean_power) - 10.0 * math.log10((10**5 + 10**6) / 2)) < 1e-6


def _sample_episode() -> AlignedEpisodeBundle:
    """A small aligned frame: odom + power + joints + phase labels."""
    df = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
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
    return AlignedEpisodeBundle(episode_id=0, data=df, start_pos=[], goal_pos=[], robot_name="env_0_jackal")


def test_calculate_produces_timeseries():
    calc = _calc()
    out = calc.calculate(_sample_episode(), {})
    assert out["timeseries_char_time_s"] == [0.0, 1.0, 2.0, 3.0]
    assert out["timeseries_char_power_total_w"] == [50.0, 55.0, 60.0, 65.0]
    assert out["timeseries_char_phase_kind"] == ["ramp_apex", "ramp_apex", "ramp_apex", "ramp_apex"]
    assert out["char_phase_coverage"] == 1.0
    assert out["timeseries_char_vx_target"] == [0.25, 0.25, 0.5, 0.5]
    # P_mech = Sigma|tau*omega|: 2.0*0.5 + 2.0*0.5 = 2.0 at t=1s
    assert out["timeseries_char_power_mech_w"][1] == 2.0
    # Energy intensity = p*dt/ds: 55 W * 1s / 0.25 m = 220 J/m at t=1s
    assert out["timeseries_char_energy_intensity"][1] == 220.0
    assert len(out["timeseries_char_dba"]) == 4


def test_steady_state_transient_filtering():
    # 5s dwell: first 1.5s tagged transient, remaining tagged linear
    df = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "pos_x": [0.0, 0.25, 0.50, 0.75],
            "pos_y": [0.0, 0.0, 0.0, 0.0],
            "vel_linear": [0.0, 0.25, 0.25, 0.25],
            "total_power_w": [500.0, 200.0, 42.2, 42.2],
            "label": ["linear_vx_0.25", "linear_vx_0.25", "linear_vx_0.25", "linear_vx_0.25"],
        }
    )
    ep = AlignedEpisodeBundle(episode_id=0, data=df, start_pos=[], goal_pos=[], robot_name="env_0_jackal")
    out = _calc().calculate(ep, {})
    # t=0s, 1s (< 1.5s): transient; t=2s, 3s (>= 1.5s): steady-state linear
    assert out["timeseries_char_phase_kind"] == ["transient", "transient", "linear", "linear"]


def test_arc_and_lateral_characterization_phases():
    df = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
            "pos_x": [0.0, 0.5, 1.0, 1.5],
            "pos_y": [0.0, 0.1, 0.3, 0.6],
            "vel_linear": [0.5, 0.5, 1.0, 1.0],
            "vel_angular": [0.5, 0.5, 1.0, 1.0],
            "total_power_w": [60.0, 60.0, 75.0, 75.0],
            "total_level_af_dba": [45.0, 45.0, 50.0, 50.0],
            "label": ["arc_vx_0.50_r_0.67_left", "arc_vx_0.50_r_0.67_left", "arc_vx_1.00_r_0.67_left", "arc_vx_1.00_r_0.67_left"],
        }
    )
    ep = AlignedEpisodeBundle(episode_id=0, data=df, start_pos=[], goal_pos=[], robot_name="env_0_jackal")
    out = _calc().calculate(ep, {})
    # Jackal's arc radii scale off its 0.267 m footprint, so these labels exist.
    assert out["char_phase_coverage"] == 1.0
    # turn radius is the schedule's commanded radius, not the achieved vx/wz ratio
    assert abs(out["timeseries_char_turn_radius_m"][0] - 0.6675) < 1e-4
    # Energy per rad: 60W / 0.5 rad/s = 120 J/rad
    assert abs(out["timeseries_char_energy_per_rad"][0] - 120.0) < 1e-4


def test_calculate_missing_power_and_acoustics():
    df = _sample_episode().data.drop(["total_power_w", "total_level_af_dba"])
    ep = AlignedEpisodeBundle(episode_id=0, data=df, start_pos=[], goal_pos=[], robot_name="env_0_jackal")
    out = _calc().calculate(ep, {})
    # Power degrades to 0; acoustics falls back to the joint model (non-null).
    assert out["timeseries_char_power_total_w"] == [0.0, 0.0, 0.0, 0.0]
    assert all(v is not None for v in out["timeseries_char_dba"])


def test_calculate_empty_returns_none_keys():
    ep = AlignedEpisodeBundle(episode_id=0, data=pl.DataFrame(), start_pos=[], goal_pos=[], robot_name="env_0_jackal")
    out = _calc().calculate(ep, {})
    assert all(v is None for v in out.values())


def test_energy_intensity_speed_threshold_gating():
    # When moving below 0.05 m/s, energy intensity should be None to prevent division singularities
    df = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000, 2_000_000_000],
            "pos_x": [0.0, 0.01, 0.50],  # dt=1s -> v=0.01 m/s at t=1s, v=0.49 m/s at t=2s
            "pos_y": [0.0, 0.0, 0.0],
            "vel_linear": [0.0, 0.01, 0.49],
            "total_power_w": [50.0, 50.0, 50.0],
            "total_level_af_dba": [42.0, 42.0, 42.0],
            "label": ["ramp_up_vx_0.50_h_1.00", "ramp_up_vx_0.50_h_1.00", "ramp_up_vx_0.50_h_1.00"],
        }
    )
    ep = AlignedEpisodeBundle(episode_id=0, data=df, start_pos=[], goal_pos=[], robot_name="env_0_jackal")
    out = _calc().calculate(ep, {})
    # t=0: ds=0 -> None; t=1: speed=0.01 m/s < 0.05 m/s -> None (gated!); t=2: speed=0.49 m/s >= 0.05 -> valid float
    assert out["timeseries_char_energy_intensity"][0] is None
    assert out["timeseries_char_energy_intensity"][1] is None
    assert out["timeseries_char_energy_intensity"][2] is not None


def test_accel_target_is_signed_by_direction_of_change():
    # ramp_down decelerates from its target to rest, so its acceleration is the
    # negative of the ramp_up that reached that target.
    df = pl.DataFrame(
        {
            "time_ns": [0, 500_000_000, 1_000_000_000, 1_500_000_000],
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
    ep = AlignedEpisodeBundle(episode_id=0, data=df, start_pos=[], goal_pos=[], robot_name="env_0_jackal")
    out = _calc().calculate(ep, {})
    assert out["char_phase_coverage"] == 1.0
    assert out["timeseries_char_accel_target"] == [1.0, -1.0, -1.0, 1.0]


def test_unmatched_labels_lower_phase_coverage():
    # A label the rebuilt schedule lacks means the offline envelope disagrees
    # with the one the sweep ran, and the fallback classifies the sample instead.
    df = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000],
            "pos_x": [0.0, 0.5],
            "pos_y": [0.0, 0.0],
            "vel_linear": [0.5, 0.5],
            "total_power_w": [60.0, 60.0],
            "linear_x": [0.5, 0.5],
            "angular_z": [0.0, 0.0],
            "label": ["linear_vx_0.50", "linear_vx_99.00"],
        }
    )
    ep = AlignedEpisodeBundle(episode_id=0, data=df, start_pos=[], goal_pos=[], robot_name="env_0_jackal")
    out = _calc().calculate(ep, {})
    assert out["char_phase_coverage"] == 0.5
    assert out["timeseries_char_phase_kind"][1] in ("linear", "transient")


def test_recorded_schedule_beats_the_rebuild():
    # arc_vx_0.33_r_0.90_left is not a working point the default envelope
    # rebuild would ever produce, so a match here can only come from the
    # recorded schedule table taking precedence over the rebuild.
    df = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000],
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
    ep = AlignedEpisodeBundle(
        episode_id=0,
        data=df,
        start_pos=[],
        goal_pos=[],
        robot_name="env_0_jackal",
        topics={"characterization_schedule": schedule},
    )
    out = _calc().calculate(ep, {})
    assert out["char_phase_coverage"] == 1.0
    assert out["timeseries_char_vx_target"] == [0.33, 0.75]
    assert out["timeseries_char_turn_radius_m"][0] == 0.90
    # ramp_down decelerates, so its accel target is the negative of the ramp rate
    assert out["timeseries_char_accel_target"][1] == -0.375


def test_duplicate_recorded_schedule_rows_do_not_duplicate_samples():
    # A schedule can be republished mid-run, the last row for a label wins and
    # the join must stay one-to-one (no row count blowup).
    df = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000],
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
    ep = AlignedEpisodeBundle(
        episode_id=0,
        data=df,
        start_pos=[],
        goal_pos=[],
        robot_name="env_0_jackal",
        topics={"characterization_schedule": schedule},
    )
    out = _calc().calculate(ep, {})
    assert len(out["timeseries_char_time_s"]) == 2
    assert out["timeseries_char_vx_target"] == [0.55, 0.55]


def test_steady_state_bidirectional_windowing():
    # 5s dwell block with 6 points (t=0 to 5s):
    # t=0, 1s (< 1.5s): transient
    # t=2s, 3s, 4s (steady state): linear
    # t=5s (> 5.0 - 0.75s = 4.25s): transient (trailing decel/settle trim)
    df = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000, 5_000_000_000],
            "pos_x": [0.0, 0.25, 0.50, 0.75, 1.0, 1.25],
            "pos_y": [0.0] * 6,
            "vel_linear": [0.0, 0.25, 0.25, 0.25, 0.25, 0.0],
            "total_power_w": [500.0, 200.0, 42.2, 42.2, 42.2, 150.0],
            "label": ["linear_vx_0.25"] * 6,
        }
    )
    ep = AlignedEpisodeBundle(episode_id=0, data=df, start_pos=[], goal_pos=[], robot_name="env_0_jackal")
    out = _calc().calculate(ep, {})
    assert out["timeseries_char_phase_kind"] == ["transient", "transient", "linear", "linear", "linear", "transient"]
