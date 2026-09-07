"""Unit tests for the open-loop characterization maneuver schedule.

Pure Python, no ROS, no Gazebo required. The schedule lives in
task_generator (owned by the TM_Characterization robot task mode); the offline
calculator imports the same module so labels never drift.
"""

import dataclasses
import json
import pathlib

import pytest

pytest.importorskip("controller_manager_msgs")

from task_generator.tasks.robots.characterization.schedule import (
    ANGULAR_DWELL_S,
    ARC_RADIUS_FACTORS,
    DURATION_DEFAULTS,
    IDLE_DURATION_S,
    LINEAR_DWELL_S,
    MODES,
    SWEEP_DEFAULTS,
    VX_MAX,
    PhaseKind,
    build_schedule,
    resolve_envelope,
    schedule_duration,
)


def _caps(tmp_path: pathlib.Path, name: str, body: str) -> pathlib.Path:
    """Write a caps/mobile.yaml for a fictional robot and return its caps root."""
    caps = tmp_path / name / "caps"
    caps.mkdir(parents=True)
    (caps / "mobile.yaml").write_text(body)
    return tmp_path


def test_schedule_structure():
    phases = build_schedule()
    assert phases[0].kind == PhaseKind.IDLE
    assert phases[-1].kind == PhaseKind.IDLE
    # Baseline standstill blocks at start and end.
    idle_blocks = [p for p in phases if p.kind == PhaseKind.IDLE and p.duration_s == IDLE_DURATION_S]
    assert len(idle_blocks) == 2


def test_linear_sweep_coverage_up_to_2mps():
    phases = build_schedule()
    linear = [p for p in phases if p.name.startswith("linear_vx_")]
    targets = sorted({p.vx_target for p in linear if p.vx_target > 0.0})
    assert targets == [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    assert max(targets) == VX_MAX
    assert all(p.duration_s == LINEAR_DWELL_S for p in linear)
    # Out-and-back: every forward step has a matching backward return at the
    # same speed so the robot never runs out of map.
    forward = {p.vx_target for p in linear if p.vx_target > 0.0}
    backward = {p.vx_target for p in linear if p.vx_target < 0.0}
    assert forward == {-v for v in backward}


def test_ramp_tests():
    phases = build_schedule()
    ramps_up = [p for p in phases if p.kind == PhaseKind.RAMP_UP and p.vx_target > 0.0]
    ramps_down = [p for p in phases if p.kind == PhaseKind.RAMP_DOWN and p.vx_target > 0.0]
    ramps_up_neg = [p for p in phases if p.kind == PhaseKind.RAMP_UP and p.vx_target < 0.0]
    ramps_down_neg = [p for p in phases if p.kind == PhaseKind.RAMP_DOWN and p.vx_target < 0.0]

    # Covers the linear envelope steps
    targets = sorted(p.vx_target for p in ramps_up)
    assert targets == [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    assert sorted(p.vx_target for p in ramps_down) == targets
    assert sorted(p.vx_target for p in ramps_up_neg) == [-v for v in reversed(targets)]
    assert sorted(p.vx_target for p in ramps_down_neg) == [-v for v in reversed(targets)]

    # Settle at apex for each positive and negative step
    apexes = [p for p in phases if p.name.startswith("ramp_apex")]
    assert len(apexes) == len(targets) * 2
    assert all(p.duration_s == 1.0 for p in apexes)


def test_angular_sweep():
    phases = build_schedule()
    angular = [p for p in phases if p.kind == PhaseKind.ANGULAR]
    assert [p.wz_target for p in angular] == [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert all(p.duration_s == ANGULAR_DWELL_S for p in angular)
    assert all(p.vx_target == 0.0 for p in angular)


def test_names_unique():
    phases = build_schedule()
    names = [p.name for p in phases]
    assert len(names) == len(set(names))


def test_schedule_duration():
    phases = build_schedule()
    assert schedule_duration(phases) == pytest.approx(sum(p.duration_s for p in phases))


def test_settle_and_apex_carry_their_own_kinds():
    # Settling is not standby draw and a ramp apex is not steady-state cruise,
    # so neither may report as idle or linear.
    phases = build_schedule()
    by_name = {p.name: p for p in phases}
    assert {p.kind for p in phases if "_settle_" in p.name} == {PhaseKind.SETTLE}
    assert by_name["ramp_apex_vx_1.00_h_1.00"].kind == PhaseKind.RAMP_APEX
    assert by_name["brake_approach_vx_2.00"].kind == PhaseKind.BRAKE_APPROACH
    assert all(p.kind == PhaseKind.IDLE for p in phases if p.name.startswith("idle_"))


def test_sweeps_end_exactly_on_the_envelope():
    # 0.4 m/s and 1.9 rad/s are not multiples of the step, so a naive range
    # either overshoots the rated limit or stops short of it.
    phases = build_schedule(vx_max=0.4, wz_max=1.9)
    linear = sorted({p.vx_target for p in phases if p.kind == PhaseKind.LINEAR and p.vx_target > 0.0})
    assert linear == [0.25, 0.4]
    angular = [p.wz_target for p in phases if p.kind == PhaseKind.ANGULAR]
    assert min(angular) == -1.9
    assert max(angular) == 1.9
    assert all(abs(wz) <= 1.9 for wz in angular)


def test_angular_sweep_is_symmetric_about_zero():
    # The lower bound comes from the robot's own rated rate, not a module constant.
    for wz_max in (0.3, 1.0, 2.2, 4.0):
        angular = [p.wz_target for p in build_schedule(wz_max=wz_max) if p.kind == PhaseKind.ANGULAR]
        assert min(angular) == -wz_max
        assert max(angular) == wz_max


def test_arc_radii_scale_with_the_footprint():
    small = [p for p in build_schedule(radius=0.2) if p.kind == PhaseKind.ARC]
    large = [p for p in build_schedule(radius=0.8) if p.kind == PhaseKind.ARC]
    assert {p.radius_m for p in small} <= {round(f * 0.2, 6) for f in ARC_RADIUS_FACTORS}
    assert {p.radius_m for p in large} <= {round(f * 0.8, 6) for f in ARC_RADIUS_FACTORS}
    assert min(p.radius_m for p in small) < min(p.radius_m for p in large)
    # Every arc is a closed orbit, so left and right come in pairs.
    assert sum(p.wz_target > 0 for p in small) == sum(p.wz_target < 0 for p in small)


def test_arcs_never_exceed_the_rated_angular_rate():
    phases = build_schedule(vx_max=0.4, wz_max=1.9, radius=0.1)
    arcs = [p for p in phases if p.kind == PhaseKind.ARC]
    assert arcs
    assert all(abs(p.wz_target) <= 1.9 for p in arcs)


def test_labels_never_encode_a_duration():
    # The offline calculator rebuilds the schedule without knowing the run's
    # duration config, so no configurable duration may reach a label.
    baseline = [p.name for p in build_schedule()]
    retimed = [p.name for p in build_schedule(**{name: default + 3.0 for name, default in DURATION_DEFAULTS.items()})]
    assert baseline == retimed


def test_modes_select_blocks():
    only_linear = build_schedule(modes=["linear"])
    assert {p.kind for p in only_linear} == {PhaseKind.LINEAR, PhaseKind.SETTLE}
    assert not build_schedule(modes=[])
    assert {p.kind for p in build_schedule(modes=["angular"])} == {PhaseKind.ANGULAR}
    # No declared mode name may be one build_schedule silently ignores.
    for mode in MODES:
        assert build_schedule(modes=[mode], is_holonomic=True, vy_max=1.0), mode


def test_resolve_envelope_from_caps(tmp_path: pathlib.Path):
    caps_dir = _caps(
        tmp_path,
        "some_robot",
        "radius: 0.42\nactions:\n  continuous:\n    linear:\n      min: -1.5\n      max: 3.0\n    angular:\n      min: -4.0\n      max: 4.0\n",
    )
    assert resolve_envelope("some_robot", caps_dir=caps_dir) == {
        "vx_max": 3.0,
        "vy_max": 0.0,
        "wz_max": 4.0,
        "radius": 0.42,
        "is_holonomic": False,
    }


def test_resolve_envelope_holonomic_without_a_lateral_limit(tmp_path: pathlib.Path):
    # No robot in the fleet declares actions.continuous.lateral, so a holonomic
    # robot sweeps sideways up to its forward limit.
    caps_dir = _caps(
        tmp_path,
        "omni_robot",
        "is_holonomic: true\nradius: 0.3\nactions:\n  continuous:\n    linear:\n      max: 2.0\n",
    )
    env = resolve_envelope("omni_robot", caps_dir=caps_dir)
    assert env["is_holonomic"] is True
    assert env["vy_max"] == 2.0


def test_resolve_envelope_fallback(tmp_path: pathlib.Path):
    # Unknown robot with no caps file -> generic defaults.
    assert resolve_envelope("ghost_robot", caps_dir=tmp_path) == {
        "vx_max": VX_MAX,
        "vy_max": 0.0,
        "wz_max": 2.5,
        "radius": 0.5,
        "is_holonomic": False,
    }


def test_schedule_uses_resolved_envelope(tmp_path: pathlib.Path):
    caps_dir = _caps(
        tmp_path,
        "big_robot",
        "radius: 0.6\nactions:\n  continuous:\n    linear:\n      max: 3.0\n    angular:\n      max: 4.0\n",
    )
    env = resolve_envelope("big_robot", caps_dir=caps_dir)
    phases = build_schedule(
        vx_max=float(env["vx_max"]),
        wz_max=float(env["wz_max"]),
        radius=float(env["radius"]),
    )
    linear = [p for p in phases if p.kind == PhaseKind.LINEAR]
    assert max(p.vx_target for p in linear) == 3.0
    angular = [p for p in phases if p.kind == PhaseKind.ANGULAR]
    assert max(p.wz_target for p in angular) == 4.0
    arcs = [p for p in phases if p.kind == PhaseKind.ARC]
    assert max(p.radius_m for p in arcs) == pytest.approx(max(ARC_RADIUS_FACTORS) * 0.6)


def test_ramp_horizons_produce_distinct_families():
    phases = build_schedule(ramp_horizons_s=[0.5, 2.0])
    ups = [p for p in phases if p.kind == PhaseKind.RAMP_UP]
    assert {p.ramp_s for p in ups} == {0.5, 2.0}
    names_by_horizon = {h: {p.name for p in ups if p.ramp_s == h} for h in (0.5, 2.0)}
    for horizon, names in names_by_horizon.items():
        assert names
        assert all(name.endswith(f"_h_{horizon:.2f}") for name in names)
    assert names_by_horizon[0.5].isdisjoint(names_by_horizon[2.0])


def test_custom_arc_factors_change_targets_and_respect_the_rated_wz_limit():
    phases = build_schedule(
        vx_max=2.0,
        wz_max=2.5,
        radius=0.5,
        arc_speed_factors=[0.5],
        arc_radius_factors=[2.0],
    )
    arcs = [p for p in phases if p.kind == PhaseKind.ARC]
    assert arcs
    assert {round(p.vx_target, 6) for p in arcs} == {1.0}
    assert {round(p.radius_m, 6) for p in arcs} == {1.0}
    assert all(abs(p.wz_target) <= 2.5 for p in arcs)


def test_sweep_defaults_keys_are_accepted_by_build_schedule():
    assert set(SWEEP_DEFAULTS) == {"arc_speed_factors", "arc_radius_factors", "ramp_horizons_s"}
    assert build_schedule(**SWEEP_DEFAULTS) == build_schedule()


def test_phase_json_roundtrip_serialises_kind_as_a_plain_string():
    # This is the publish format: the sweep node dumps dataclasses.asdict(phase).
    phase = build_schedule(modes=["linear"])[0]
    decoded = json.loads(json.dumps(dataclasses.asdict(phase)))
    assert decoded["kind"] == phase.kind.value
    assert isinstance(decoded["kind"], str)
    assert decoded["name"] == phase.name
