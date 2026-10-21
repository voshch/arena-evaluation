"""Edge-case criticality and formation metrics, measured from recorded trajectories and scoped by the scenario's provenance.yaml."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
from arena_simulation_setup.tree.World import WorldView

from arena_evaluation.processing.metrics.edge_case.criticality import (
    Criticality,
    Sample,
    formation_hold,
    post_encroachment_time,
    score,
    time_to_collision,
)
from arena_evaluation.processing.metrics.edge_case.edge_case_metrics import (
    EdgeCaseCriticalityCalculator,
    EdgeCaseFormationCalculator,
    agent_name,
)
from arena_evaluation.processing.metrics.edge_case.provenance import _world_views
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams, RunMetadata

SEC = 1_000_000_000
ROBOT_RADIUS = 0.3

ONE_LEVEL = """format: arena-world/2
levels:
- id: '0'
  zones:
  - {name: hall, polygon: [[0.0, -10.0], [20.0, -10.0], [20.0, 10.0], [0.0, 10.0]]}
"""

TWO_LEVELS = """format: arena-world/2
levels:
- id: '0'
  zones:
  - {name: hall, polygon: [[0.0, -10.0], [20.0, -10.0], [20.0, 10.0], [0.0, 10.0]]}
- id: '1'
  zones:
  - {name: loft, polygon: [[0.0, -10.0], [20.0, -10.0], [20.0, 10.0], [0.0, 10.0]]}
"""


def test_head_on_closing_gives_the_expected_time() -> None:
    assert time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (10.0, 0.0), (-1.0, 0.0), 0.3) == pytest.approx((10.0 - 0.6) / 2.0)


def test_parallel_motion_never_collides() -> None:
    assert time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (0.0, 5.0), (1.0, 0.0), 0.3) is None


def test_receding_pair_never_collides() -> None:
    assert time_to_collision((0.0, 0.0), (-1.0, 0.0), 0.3, (10.0, 0.0), (1.0, 0.0), 0.3) is None


def test_already_touching_is_zero_not_none() -> None:
    assert time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (0.4, 0.0), (0.0, 0.0), 0.3) == 0.0


def test_no_relative_motion_is_not_a_collision() -> None:
    assert time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (5.0, 0.0), (1.0, 0.0), 0.3) is None


def test_clearance_is_surface_to_surface() -> None:
    out = score([Sample(t=0.0, robot=(0.0, 0.0), robot_radius=0.3, peds={"p": ((0.6, 0.0), (0.0, 0.0), 0.3)})])
    assert out.min_clearance_m == pytest.approx(0.0)
    assert out.collided is True


def test_clearance_tracks_the_closest_moment_and_names_it() -> None:
    s = [Sample(t=float(t), robot=(0.0, 0.0), peds={"a": ((x, 0.0), (0.0, 0.0), 0.3)}) for t, x in enumerate((5.0, 2.0, 4.0))]
    out = score(s)
    assert out.min_clearance_m == pytest.approx(2.0 - 0.6)
    assert out.min_clearance_with == "a"
    assert out.min_clearance_at == pytest.approx(1.0)


def test_freeze_ignores_the_settling_window() -> None:
    s = [Sample(t=float(i), robot=(0.0, 0.0), robot_vel=(0.0, 0.0)) for i in range(5)]
    assert score(s, settle_s=2.0).freeze_duration_s == pytest.approx(2.0)


def test_a_moving_robot_never_freezes() -> None:
    s = [Sample(t=float(i), robot=(float(i), 0.0), robot_vel=(1.0, 0.0)) for i in range(6)]
    assert score(s).freeze_duration_s == 0.0


def test_intrusion_time_counts_only_while_inside_personal_space() -> None:
    s = [Sample(t=float(t), robot=(0.0, 0.0), robot_vel=(1.0, 0.0), peds={"a": ((x, 0.0), (0.0, 0.0), 0.3)}) for t, x in enumerate((5.0, 0.8, 5.0))]
    assert score(s).intrusion_time_s == pytest.approx(1.0)


def test_only_restricts_scoring_to_the_named_agent() -> None:
    s = [Sample(t=0.0, robot=(0.0, 0.0), peds={"bystander": ((1.0, 0.0), (0.0, 0.0), 0.3), "edge_0": ((4.0, 0.0), (0.0, 0.0), 0.3)})]
    assert score(s).min_clearance_with == "bystander"
    assert score(s, only=["edge_0"]).min_clearance_with == "edge_0"
    assert score(s, only=["edge_0"]).min_clearance_m == pytest.approx(4.0 - 0.6)


def test_a_robot_driving_into_a_standing_pedestrian_is_at_fault() -> None:
    s = [Sample(t=float(t), robot=(float(t), 0.0), robot_vel=(1.0, 0.0), peds={"a": ((3.5, 0.0), (0.0, 0.0), 0.3)}) for t in range(4)]
    out = score(s)
    assert (out.collided, out.collision_fault, out.collision_with, out.collision_at) == (True, "robot", "a", 3.0)


def _contact(robot_vel: tuple[float, float], ped_vel: tuple[float, float]) -> Criticality:
    """Two samples, contact on the second (centers 0.5 m apart, radii 0.3 and 0.25)."""
    return score(
        [
            Sample(t=0.0, robot=(0.0, 0.0), robot_vel=robot_vel, robot_radius=0.3, peds={"p": ((3.0, 0.0), ped_vel, 0.25)}),
            Sample(t=1.0, robot=(0.0, 0.0), robot_vel=robot_vel, robot_radius=0.3, peds={"p": ((0.5, 0.0), ped_vel, 0.25)}),
        ],
        settle_s=0.0,
    )


def test_robot_driving_into_a_standing_pedestrian_is_robot_fault() -> None:
    r = _contact(robot_vel=(0.8, 0.0), ped_vel=(0.0, 0.0))
    assert r.collided and r.collision_fault == "robot"
    assert r.collision_with == "p"
    assert r.collision_robot_closing > 0.5 and abs(r.collision_ped_closing) < 0.05


def test_pedestrian_walking_into_a_standing_robot_is_pedestrian_fault() -> None:
    r = _contact(robot_vel=(0.0, 0.0), ped_vel=(-0.9, 0.0))
    assert r.collided and r.collision_fault == "pedestrian"
    assert r.collision_ped_closing > 0.5


def test_both_closing_is_mutual_and_neither_is_static() -> None:
    assert _contact((0.6, 0.0), (-0.6, 0.0)).collision_fault == "mutual"
    assert _contact((0.0, 0.0), (0.0, 0.0)).collision_fault == "static"


def test_receding_pedestrian_contact_is_not_their_fault() -> None:
    assert _contact(robot_vel=(0.7, 0.0), ped_vel=(0.7, 0.0)).collision_fault == "robot"


def test_no_contact_leaves_the_fault_empty() -> None:
    r = score([Sample(t=0.0, robot=(0.0, 0.0), peds={"p": ((5.0, 0.0), (0.0, 0.0), 0.25)})])
    assert not r.collided and r.collision_fault == "" and r.collision_at is None


def test_pet_is_the_time_gap_through_the_encounter_point() -> None:
    s = [
        Sample(t=0.0, robot=(0.0, 0.0), peds={"a": ((5.0, 4.0), (0.0, 0.0), 0.3)}),
        Sample(t=2.0, robot=(5.0, 0.0), peds={"a": ((5.0, 2.0), (0.0, 0.0), 0.3)}),
        Sample(t=5.0, robot=(9.0, 0.0), peds={"a": ((5.0, 0.2), (0.0, 0.0), 0.3)}),
    ]
    assert post_encroachment_time(s, (5.0, 0.0), "a") == pytest.approx(3.0)
    out = score(s, encounters=[("a", (5.0, 0.0))])
    assert (out.pet_s, out.pet_with) == (pytest.approx(3.0), "a")


def test_pet_is_none_when_the_pedestrian_never_goes_there() -> None:
    s = [
        Sample(t=0.0, robot=(0.0, 0.0), peds={"a": ((50.0, 50.0), (0.0, 0.0), 0.3)}),
        Sample(t=2.0, robot=(5.0, 0.0), peds={"a": ((51.0, 50.0), (0.0, 0.0), 0.3)}),
    ]
    assert post_encroachment_time(s, (5.0, 0.0), "a") is None
    assert post_encroachment_time(s, (5.0, 0.0), "nobody") is None


def test_pet_is_none_when_the_robot_never_goes_there() -> None:
    s = [Sample(t=float(t), robot=(0.0, 0.0), peds={"a": ((5.0, 0.0), (0.0, 0.0), 0.3)}) for t in range(3)]
    assert post_encroachment_time(s, (5.0, 0.0), "a") is None


def test_pet_takes_the_tightest_encounter_and_none_without_any() -> None:
    s = [Sample(t=float(t), robot=(float(t), 0.0), peds={"a": ((3.0, 0.5), (0.0, 0.0), 0.3), "b": ((6.0, 1.0 if t >= 5 else 9.0), (0.0, 0.0), 0.3)}) for t in range(8)]
    out = score(s, encounters=[("a", (3.0, 0.0)), ("b", (6.0, 0.0))])
    assert (out.pet_s, out.pet_with) == (pytest.approx(1.0), "b")
    assert score(s).pet_s is None and score(s).pet_with == ""


def test_an_episode_with_no_pedestrians_is_not_a_safe_episode() -> None:
    out = score([Sample(t=float(i), robot=(float(i), 0.0), robot_vel=(1.0, 0.0)) for i in range(4)])
    assert out.min_clearance_m is None
    assert out.min_ttc_s is None
    assert out.observed is False


def test_no_samples_at_all_is_empty_not_zero() -> None:
    assert score([]) == Criticality()
    assert score([]).observed is False


def test_measured_ttc_follows_the_robot_not_a_design() -> None:
    closing = [Sample(t=float(i), robot=(float(i), 0.0), robot_vel=(1.0, 0.0), peds={"edge_0": ((10.0, 0.0), (0.0, 0.0), 0.3)}) for i in range(4)]
    assert score(closing, only=["edge_0"]).min_ttc_s is not None
    frozen = [Sample(t=float(i), robot=(0.0, 0.0), robot_vel=(0.0, 0.0), peds={"edge_0": ((10.0, 0.0), (0.0, 0.0), 0.3)}) for i in range(4)]
    assert score(frozen, only=["edge_0"]).min_ttc_s is None


def _trio(t: float, a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> Sample:
    return Sample(t=t, robot=(0.0, 0.0), peds={"d_1": (a, (0.0, 0.0), 0.3), "d_2": (b, (0.0, 0.0), 0.3), "d_3": (c, (0.0, 0.0), 0.3)})


def test_formation_hold_reports_when_the_group_broke() -> None:
    held = [_trio(t, (0.0, 0.0), (1.0, 0.0), (0.5, 0.8)) for t in (0.0, 1.0, 2.0)]
    broke = [*held, _trio(3.0, (0.0, 0.0), (1.0, 0.0), (4.0, 4.0)), _trio(4.0, (0.0, 0.0), (1.0, 0.0), (5.0, 5.0))]
    out = formation_hold(broke, ["d_1", "d_2", "d_3"])
    assert out["seen"] and out["formed_at"] == 0.0 and out["held_s"] == 2.0 and out["broke_at"] == 3.0
    out = formation_hold(held, ["d_1", "d_2", "d_3"])
    assert out["seen"] and out["held_s"] == 2.0 and out["broke_at"] is None
    assert formation_hold(held, ["nobody"])["seen"] is False
    assert formation_hold([], ["d_1"])["seen"] is False


def test_formation_hold_around_a_center_counts_the_arrivals() -> None:
    walking = [_trio(0.0, (9.0, 9.0), (1.0, 0.0), (0.5, 0.8)), _trio(1.0, (0.2, 0.1), (1.0, 0.0), (0.5, 0.8)), _trio(2.0, (0.1, 0.0), (1.0, 0.0), (0.5, 0.8)), _trio(3.0, (6.0, 6.0), (1.0, 0.0), (0.5, 0.8))]
    out = formation_hold(walking, ["d_1", "d_2", "d_3"], radius=1.5, center=(0.5, 0.3))
    assert out["seen"] and out["formed_at"] == 1.0 and out["held_s"] == 1.0 and out["broke_at"] == 3.0 and out["arrived"] == 3
    never = formation_hold(walking[:1], ["d_1", "d_2", "d_3"], radius=1.5, center=(0.5, 0.3))
    assert never["seen"] and never["formed_at"] is None and never["held_s"] is None and never["arrived"] == 2


@pytest.mark.parametrize(("recorded", "bare"), [("/env_0/d_1", "d_1"), ("env_3/edge_0", "edge_0"), ("env_0/d_1/0", "d_1"), ("walker", "walker")])
def test_recorded_names_map_to_scenario_names(recorded: str, bare: str) -> None:
    assert agent_name(recorded) == bare


def test_both_calculators_are_registered() -> None:
    assert {"edge_case_criticality", "edge_case_formation"} <= set(MetricRegistry(RobotParams()).calculators)


def _world(root: pathlib.Path, *, world_yaml: str = ONE_LEVEL, provenance: str | None = None, robot_level: str | None = None) -> str:
    """A world under `root` with one scenario `case`, registered under a name unique to the test."""
    (root / "world.yaml").write_text(world_yaml)
    scenario = root / "scenarios" / "case"
    scenario.mkdir(parents=True)
    robots = f"robots:\n- {{start: [1.0, 0.0], level: '{robot_level}'}}\n" if robot_level else "robots: []\n"
    (scenario / "scenario.yaml").write_text("format: arena-scenario/2\n" + robots)
    if provenance is not None:
        (scenario / "provenance.yaml").write_text(provenance)
    name = f"edge_case_{root.name}"
    _world_views[name] = WorldView(root)
    return name


def _run(world: str, *, tm_obstacles: str = "scenario", tm_robots: str = "random", file: str = "case") -> RunMetadata:
    params = {"file": file}
    return RunMetadata(
        benchmark_id="edge",
        planner="nav2",
        map=world,
        stage="case",
        recording_started_at="2026-10-21T00:00:00Z",
        python_version="3.12",
        ros_distro="jazzy",
        tm_obstacles=tm_obstacles,
        tm_robots=tm_robots,
        obstacles_params=params if tm_obstacles == "scenario" else {},
        robots_params=params if tm_robots == "scenario" else {},
    )


def _episode(world: str, run: RunMetadata, peds: dict[str, list[tuple[float, float]]], *, seconds: int = 10) -> AlignedEpisodeBundle:
    """The robot drives along y = 0 at 1 m/s, each pedestrian follows its per-second track, all at 10 Hz."""
    times = [i * SEC // 10 for i in range(seconds * 10 + 1)]
    xs = [t / SEC for t in times]
    tf_gt = pl.DataFrame({"time_ns": times, "pos_x_gt": xs, "pos_y_gt": [0.0] * len(times), "yaw_gt": [0.0] * len(times)})
    names = sorted(peds)
    positions = []
    for t in times:
        second = min(t // SEC, seconds)
        positions.append([v for name in names for v in (*peds[name][second], 0.0)])
    frame = pl.DataFrame(
        {
            "time_ns": times,
            "peds_names": [[f"env_0/{name}" for name in names]] * len(times),
            "peds_positions": positions,
            "peds_twists": [[0.0] * (3 * len(names))] * len(times),
            "num_pedestrians": [len(names)] * len(times),
        }
    )
    data = pl.DataFrame({"time_ns": times, "pos_x": xs, "pos_y": [0.0] * len(times), "yaw": [0.0] * len(times)})
    return AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[0.0, 0.0, 0.0], goal_pos=[10.0, 0.0], map=world, run=run, topics={"tf_gt": tf_gt, "peds": frame})


def _standing(x: float, y: float, seconds: int = 10) -> list[tuple[float, float]]:
    return [(x, y)] * (seconds + 1)


CASE = """id: hospital_1__u_007
prompt_id: u_007
steps: ['C:intercept', 'D:formation_break']
owned: {C: [edge_0], D: [d_1, d_2], injected: []}
encounters: [{agent: edge_0, point: [6.0, 0.0]}]
formation: {center: [5.0, 3.0], radius: 1.5, variant: cluster, at: {at: 4.0}}
"""


@pytest.fixture
def criticality() -> EdgeCaseCriticalityCalculator:
    return EdgeCaseCriticalityCalculator(RobotParams(robot_radius=ROBOT_RADIUS))


@pytest.fixture
def formation() -> EdgeCaseFormationCalculator:
    return EdgeCaseFormationCalculator(RobotParams(robot_radius=ROBOT_RADIUS))


def test_a_scenario_without_provenance_yields_nothing(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator, formation: EdgeCaseFormationCalculator) -> None:
    world = _world(tmp_path)
    episode = _episode(world, _run(world), {"edge_0": _standing(5.0, 0.0)})
    assert criticality.calculate(episode, {}) == dict.fromkeys(criticality.output_keys())
    assert formation.calculate(episode, {}) == dict.fromkeys(formation.output_keys())


def test_an_episode_that_ran_no_scenario_yields_nothing(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(world, _run(world, tm_obstacles="random"), {"edge_0": _standing(5.0, 0.0)})
    assert criticality.calculate(episode, {}) == dict.fromkeys(criticality.output_keys())


def test_an_unknown_scenario_yields_nothing(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(world, _run(world, file="elsewhere"), {"edge_0": _standing(5.0, 0.0)})
    assert criticality.calculate(episode, {}) == dict.fromkeys(criticality.output_keys())


def test_the_case_is_scored_against_its_own_agents(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(world, _run(world), {"walker": _standing(3.0, 0.5), "edge_0": _standing(6.0, 2.0), "d_1": _standing(5.0, 3.0), "d_2": _standing(5.5, 3.0)})
    out = criticality.calculate(episode, {})
    assert out["edge_case_id"] == "hospital_1__u_007"
    assert out["edge_case_steps"] == ["C:intercept", "D:formation_break"]
    assert (out["edge_case_scope"], out["edge_case_agents"], out["edge_case_missing"]) == ("own", ["d_1", "d_2", "edge_0"], [])
    assert out["edge_case_min_clearance_with"] == "edge_0"
    assert out["edge_case_min_clearance_m"] == pytest.approx(2.0 - 2 * ROBOT_RADIUS)
    assert out["edge_case_min_clearance_at"] == pytest.approx(6.0)
    assert out["edge_case_collided"] is False and out["edge_case_collision_fault"] is None
    assert out["edge_case_pet_s"] is None and out["edge_case_pet_with"] is None
    assert out["edge_case_freeze_s"] == 0.0


def test_a_robot_running_into_its_intercept_collides_with_it(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    edge = [(6.0, 3.0 - min(t, 3) * 1.0) for t in range(11)]
    out = criticality.calculate(_episode(world, _run(world), {"edge_0": edge}), {})
    assert out["edge_case_missing"] == ["d_1", "d_2"]
    assert out["edge_case_collided"] is True and out["edge_case_collision_with"] == "edge_0" and out["edge_case_collision_fault"] == "robot"
    assert out["edge_case_pet_s"] == pytest.approx(3.0) and out["edge_case_pet_with"] == "edge_0"
    assert out["edge_case_min_ttc_s"] is not None


def test_a_case_whose_agents_never_appeared_falls_back_to_everyone(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    out = criticality.calculate(_episode(world, _run(world), {"walker": _standing(3.0, 0.5)}), {})
    assert (out["edge_case_scope"], out["edge_case_agents"], out["edge_case_missing"]) == ("all", [], ["d_1", "d_2", "edge_0"])
    assert out["edge_case_min_clearance_with"] == "walker"


def test_a_robot_that_stops_short_of_the_case_freezes(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(world, _run(world), {"edge_0": _standing(6.0, 1.5)})
    tf_gt = episode.topics["tf_gt"]
    episode.topics["tf_gt"] = tf_gt.with_columns(pl.col("pos_x_gt").clip(upper_bound=4.0))
    out = criticality.calculate(episode, {})
    assert out["edge_case_freeze_s"] == pytest.approx(6.0, abs=0.15)


def test_the_scenario_is_found_through_the_robots_axis_too(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    out = criticality.calculate(_episode(world, _run(world, tm_obstacles="random", tm_robots="scenario"), {"edge_0": _standing(6.0, 1.5)}), {})
    assert out["edge_case_id"] == "hospital_1__u_007"


def test_the_formation_forms_holds_and_breaks(tmp_path: pathlib.Path, formation: EdgeCaseFormationCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    d_1 = [(5.0, 9.0)] * 2 + [(5.0, 3.2)] * 5 + [(9.0, 9.0)] * 4
    d_2 = [(5.5, 3.0)] * 11
    out = formation.calculate(_episode(world, _run(world), {"d_1": d_1, "d_2": d_2}), {})
    assert out["edge_case_formation_variant"] == "cluster"
    assert out["edge_case_formation_members"] == ["d_1", "d_2"]
    assert out["edge_case_formation_seen"] is True
    assert out["edge_case_formation_formed_at"] == pytest.approx(2.0)
    assert out["edge_case_formation_held_s"] == pytest.approx(4.9)
    assert out["edge_case_formation_broke_at"] == pytest.approx(7.0)
    assert out["edge_case_formation_arrived"] == 2


def test_a_case_without_a_formation_has_no_formation_panel(tmp_path: pathlib.Path, formation: EdgeCaseFormationCalculator) -> None:
    world = _world(tmp_path, provenance="id: c\nowned: {C: [edge_0]}\n")
    out = formation.calculate(_episode(world, _run(world), {"edge_0": _standing(6.0, 1.5)}), {})
    assert out == dict.fromkeys(formation.output_keys())


def test_the_formation_center_moves_with_the_robot_level_into_the_map_frame(tmp_path: pathlib.Path, formation: EdgeCaseFormationCalculator) -> None:
    world = _world(tmp_path, world_yaml=TWO_LEVELS, provenance=CASE, robot_level="1")
    ox, oy = WorldView(tmp_path).level_origins()["1"]
    assert (ox, oy) != (0.0, 0.0)
    out = formation.calculate(_episode(world, _run(world), {"d_1": _standing(5.0 + ox, 3.0 + oy), "d_2": _standing(5.5 + ox, 3.0 + oy)}), {})
    assert out["edge_case_formation_arrived"] == 2
    assert out["edge_case_formation_formed_at"] == pytest.approx(0.0)


def test_the_encounter_point_moves_with_the_robot_level_into_the_map_frame(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, world_yaml=TWO_LEVELS, provenance=CASE, robot_level="1")
    ox, oy = WorldView(tmp_path).level_origins()["1"]
    episode = _episode(world, _run(world), {"edge_0": [(6.0 + ox, 4.0 + oy)] * 3 + [(6.0 + ox, oy)] * 8})
    shifted = {"tf_gt": episode.topics["tf_gt"].with_columns(pl.col("pos_x_gt") + ox, pl.col("pos_y_gt") + oy), "peds": episode.topics["peds"]}
    out = criticality.calculate(AlignedEpisodeBundle(episode_id=1, data=episode.data, start_pos=[], goal_pos=[], map=world, run=episode.run, topics=shifted), {})
    assert out["edge_case_pet_s"] == pytest.approx(3.0) and out["edge_case_pet_with"] == "edge_0"
