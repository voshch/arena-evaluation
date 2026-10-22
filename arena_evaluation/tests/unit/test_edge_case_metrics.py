"""Edge-case criticality and formation metrics, measured from recorded trajectories and scoped by the scenario's provenance.yaml."""

from __future__ import annotations

import os
import pathlib
from collections import defaultdict

import polars as pl
import pytest
from arena_simulation_setup.tree.World import WorldView

from arena_evaluation.processing.metrics.edge_case.criticality import (
    Criticality,
    FormationHold,
    Sample,
    formation_hold,
    post_encroachment_time,
    score,
    time_to_collision,
)
from arena_evaluation.processing.metrics.edge_case.provenance import world_view
from arena_evaluation.processing.metrics.edge_case.edge_case_metrics import (
    EdgeCaseCriticalityCalculator,
    EdgeCaseFormationCalculator,
    agent_name,
)
from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, EpisodeDescriptor, RobotParams, RunMetadata, TopicBundle

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


def test_overlapping_discs_without_relative_motion_are_already_touching() -> None:
    assert time_to_collision((0.0, 0.0), (0.0, 0.0), 0.3, (0.4, 0.0), (0.0, 0.0), 0.3) == 0.0
    assert time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (0.5, 0.0), (1.0, 0.0), 0.3) == 0.0


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
    out = score(closing, only=["edge_0"])
    assert (out.min_ttc_s, out.min_ttc_with, out.min_ttc_at) == (pytest.approx(6.4), "edge_0", pytest.approx(3.0))
    frozen = [Sample(t=float(i), robot=(0.0, 0.0), robot_vel=(0.0, 0.0), peds={"edge_0": ((10.0, 0.0), (0.0, 0.0), 0.3)}) for i in range(4)]
    assert score(frozen, only=["edge_0"]).min_ttc_s is None


def _trio(t: float, a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> Sample:
    return Sample(t=t, robot=(0.0, 0.0), peds={"d_1": (a, (0.0, 0.0), 0.3), "d_2": (b, (0.0, 0.0), 0.3), "d_3": (c, (0.0, 0.0), 0.3)})


def test_formation_hold_reports_when_the_group_broke() -> None:
    held = [_trio(t, (0.0, 0.0), (1.0, 0.0), (0.5, 0.8)) for t in (0.0, 1.0, 2.0)]
    broke = [*held, _trio(3.0, (0.0, 0.0), (1.0, 0.0), (4.0, 4.0)), _trio(4.0, (0.0, 0.0), (1.0, 0.0), (5.0, 5.0))]
    out = formation_hold(broke, ["d_1", "d_2", "d_3"])
    assert isinstance(out, FormationHold)
    assert (out.seen, out.formed_at, out.held_s, out.broke_at) == (True, 0.0, 2.0, 3.0)
    out = formation_hold(held, ["d_1", "d_2", "d_3"])
    assert out.seen and out.held_s == 2.0 and out.broke_at is None
    assert formation_hold(held, ["nobody"]).seen is False
    assert formation_hold([], ["d_1"]).seen is False


def test_formation_hold_around_a_center_counts_the_arrivals() -> None:
    walking = [_trio(0.0, (9.0, 9.0), (1.0, 0.0), (0.5, 0.8)), _trio(1.0, (0.2, 0.1), (1.0, 0.0), (0.5, 0.8)), _trio(2.0, (0.1, 0.0), (1.0, 0.0), (0.5, 0.8)), _trio(3.0, (6.0, 6.0), (1.0, 0.0), (0.5, 0.8))]
    out = formation_hold(walking, ["d_1", "d_2", "d_3"], radius=1.5, center=(0.5, 0.3))
    assert (out.seen, out.formed_at, out.held_s, out.broke_at, out.arrived) == (True, 1.0, 1.0, 3.0, 3)
    never = formation_hold(walking[:1], ["d_1", "d_2", "d_3"], radius=1.5, center=(0.5, 0.3))
    assert never.seen and never.formed_at is None and never.held_s is None and never.arrived == 2


def _pair(t: float, a: tuple[float, float] | None, b: tuple[float, float] | None) -> Sample:
    peds = {name: (p, (0.0, 0.0), 0.3) for name, p in (("d_1", a), ("d_2", b)) if p is not None}
    return Sample(t=t, robot=(0.0, 0.0), peds=peds)


def test_a_member_that_never_spawned_does_not_block_the_formation() -> None:
    out = formation_hold([_pair(float(t), (0.0, 0.0), (1.0, 0.0)) for t in range(3)], ["d_1", "d_2", "d_3"])
    assert out.members == ("d_1", "d_2", "d_3") and out.absent == ("d_3",)
    assert (out.formed_at, out.held_s, out.broke_at) == (0.0, 2.0, None)


def test_the_formation_without_a_center_forms_around_the_centroid_of_each_sample() -> None:
    samples = [_pair(0.0, (0.0, 0.0), (10.0, 0.0)), _pair(1.0, (20.0, 20.0), (20.5, 20.0)), _pair(2.0, (20.0, 20.0), (20.5, 20.0))]
    out = formation_hold(samples, ["d_1", "d_2"])
    assert (out.formed_at, out.held_s, out.broke_at) == (1.0, 1.0, None)


def test_a_member_leaving_the_recording_breaks_the_formation() -> None:
    samples = [_pair(0.0, (0.0, 0.0), (1.0, 0.0)), _pair(1.0, (0.0, 0.0), (1.0, 0.0)), _pair(2.0, (0.0, 0.0), None), _pair(3.0, (0.0, 0.0), (1.0, 0.0))]
    out = formation_hold(samples, ["d_1", "d_2"], center=(0.5, 0.0))
    assert (out.formed_at, out.held_s, out.broke_at, out.absent) == (0.0, 1.0, 2.0, ())


@pytest.mark.parametrize(("recorded", "bare"), [("/env_0/d_1", "d_1"), ("env_3/edge_0", "edge_0"), ("env_0/d_1/0", "d_1"), ("walker", "walker")])
def test_recorded_names_map_to_scenario_names(recorded: str, bare: str) -> None:
    assert agent_name(recorded) == bare


def test_both_calculators_are_registered() -> None:
    assert {"edge_case_criticality", "edge_case_formation"} <= set(MetricRegistry(RobotParams()).calculators)


def _world(root: pathlib.Path, *, world_yaml: str = ONE_LEVEL, provenance: str | None = None, robot_level: str | None = None, humans: str = "", name: str | None = None) -> str:
    """A run snapshot under the run dir `root` of a world with one scenario `case`, named uniquely to the test."""
    name = name or f"edge_case_{root.name}"
    world = root / "worlds" / name
    scenario = world / "scenarios" / "case"
    scenario.mkdir(parents=True)
    (world / "world.yaml").write_text(world_yaml)
    robots = f"robots:\n- {{start: [1.0, 0.0], level: '{robot_level}'}}\n" if robot_level else "robots: []\n"
    (scenario / "scenario.yaml").write_text("format: arena-scenario/2\n" + robots + humans)
    if provenance is not None:
        (scenario / "provenance.yaml").write_text(provenance)
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


def _episode(run_dir: pathlib.Path, world: str, run: RunMetadata, peds: dict[str, list[tuple[float, float]]], *, seconds: int = 10, twists: dict[str, tuple[float, float]] | None = None) -> AlignedEpisodeBundle:
    """The robot drives along y = 0 at 1 m/s, each pedestrian on its per-second track at its `twists` velocity, at 10 Hz."""
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
            "peds_twists": [[v for name in names for v in (*(twists or {}).get(name, (0.0, 0.0)), 0.0)]] * len(times),
            "num_pedestrians": [len(names)] * len(times),
        }
    )
    data = pl.DataFrame({"time_ns": times, "pos_x": xs, "pos_y": [0.0] * len(times), "yaw": [0.0] * len(times)})
    return AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[0.0, 0.0, 0.0], goal_pos=[10.0, 0.0], map=world, run=run, topics={"tf_gt": tf_gt, "peds": frame}, run_dir=run_dir)


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
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(5.0, 0.0)})
    assert criticality.calculate(episode, {}) == dict.fromkeys(criticality.output_keys())
    assert formation.calculate(episode, {}) == dict.fromkeys(formation.output_keys())


def test_an_episode_that_ran_no_scenario_yields_nothing(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world, tm_obstacles="random"), {"edge_0": _standing(5.0, 0.0)})
    assert criticality.calculate(episode, {}) == dict.fromkeys(criticality.output_keys())


def test_an_unknown_scenario_yields_nothing(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world, file="elsewhere"), {"edge_0": _standing(5.0, 0.0)})
    assert criticality.calculate(episode, {}) == dict.fromkeys(criticality.output_keys())


def test_the_case_is_scored_against_its_own_agents(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world), {"walker": _standing(3.0, 0.5), "edge_0": _standing(6.0, 2.0), "d_1": _standing(5.0, 3.0), "d_2": _standing(5.5, 3.0)})
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
    out = criticality.calculate(_episode(tmp_path, world, _run(world), {"edge_0": edge}), {})
    assert out["edge_case_missing"] == ["d_1", "d_2"]
    assert out["edge_case_collided"] is True and out["edge_case_collision_with"] == "edge_0" and out["edge_case_collision_fault"] == "robot"
    assert out["edge_case_pet_s"] == pytest.approx(3.0) and out["edge_case_pet_with"] == "edge_0"
    assert out["edge_case_min_ttc_s"] == pytest.approx(0.0) and out["edge_case_min_ttc_with"] == "edge_0"
    assert out["edge_case_played"] is None and out["edge_case_unplayed"] is None and out["edge_case_triggered_at"] is None


def test_a_case_whose_agents_never_appeared_falls_back_to_everyone(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    out = criticality.calculate(_episode(tmp_path, world, _run(world), {"walker": _standing(3.0, 0.5)}), {})
    assert (out["edge_case_scope"], out["edge_case_agents"], out["edge_case_missing"]) == ("all", [], ["d_1", "d_2", "edge_0"])
    assert out["edge_case_min_clearance_with"] == "walker"


def test_a_robot_that_stops_short_of_the_case_freezes(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)})
    tf_gt = episode.topics["tf_gt"]
    episode.topics["tf_gt"] = tf_gt.with_columns(pl.col("pos_x_gt").clip(upper_bound=4.0))
    out = criticality.calculate(episode, {})
    assert out["edge_case_freeze_s"] == pytest.approx(6.0, abs=0.15)


def test_each_agent_counts_with_the_radius_its_scenario_entry_pins(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    humans = "humans:\n- {name: edge_0, model: adult, agent: {agent_type: adult, radius: 0.5}}\n- {name: d_1, model: adult, agent: adult}\n"
    world = _world(tmp_path, provenance=CASE, humans=humans)
    out = criticality.calculate(_episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 2.0), "d_1": _standing(3.0, 1.5)}), {})
    assert out["edge_case_min_clearance_with"] == "d_1"
    assert out["edge_case_min_clearance_m"] == pytest.approx(1.5 - ROBOT_RADIUS - 0.3)
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 2.0)})
    assert criticality.calculate(episode, {})["edge_case_min_clearance_m"] == pytest.approx(2.0 - ROBOT_RADIUS - 0.5)


def test_a_peds_stream_that_stops_early_is_not_held_past_its_last_message(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 0.0)})
    episode.topics["peds"] = episode.topics["peds"].filter(pl.col("time_ns") <= 2 * SEC)
    out = criticality.calculate(episode, {})
    assert out["edge_case_collided"] is False
    assert out["edge_case_min_clearance_at"] == pytest.approx(2.1)
    assert out["edge_case_min_clearance_m"] == pytest.approx(6.0 - 2.1 - 2 * ROBOT_RADIUS)


HELD_CASE = CASE + "robot: {speed_mps: 1.0, hold_s: 3.0, traversals: 1, tolerance_m: null, radius_m: 0.35}\n"


@pytest.mark.parametrize(("provenance", "freeze_s"), [(CASE, 1.0), (HELD_CASE, 0.0)])
def test_the_designed_start_hold_is_not_a_freeze(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator, provenance: str, freeze_s: float) -> None:
    world = _world(tmp_path, provenance=provenance)
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)})
    episode.topics["tf_gt"] = episode.topics["tf_gt"].with_columns((pl.col("pos_x_gt") - 3.0).clip(lower_bound=0.0))
    assert criticality.calculate(episode, {})["edge_case_freeze_s"] == pytest.approx(freeze_s, abs=0.15)


def test_the_scenario_is_found_through_the_robots_axis_too(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    out = criticality.calculate(_episode(tmp_path, world, _run(world, tm_obstacles="random", tm_robots="scenario"), {"edge_0": _standing(6.0, 1.5)}), {})
    assert out["edge_case_id"] == "hospital_1__u_007"


def test_the_formation_forms_holds_and_breaks(tmp_path: pathlib.Path, formation: EdgeCaseFormationCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    d_1 = [(5.0, 9.0)] * 2 + [(5.0, 3.2)] * 5 + [(9.0, 9.0)] * 4
    d_2 = [(5.5, 3.0)] * 11
    out = formation.calculate(_episode(tmp_path, world, _run(world), {"d_1": d_1, "d_2": d_2}), {})
    assert out["edge_case_formation_variant"] == "cluster"
    assert out["edge_case_formation_members"] == ["d_1", "d_2"]
    assert out["edge_case_formation_absent"] == []
    assert out["edge_case_formation_seen"] is True
    assert out["edge_case_formation_formed_at"] == pytest.approx(2.0)
    assert out["edge_case_formation_held_s"] == pytest.approx(4.9)
    assert out["edge_case_formation_broke_at"] == pytest.approx(7.0)
    assert out["edge_case_formation_arrived"] == 2


def test_the_formation_is_scored_on_the_members_that_were_recorded(tmp_path: pathlib.Path, formation: EdgeCaseFormationCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    out = formation.calculate(_episode(tmp_path, world, _run(world), {"d_1": [(5.0, 9.0)] * 3 + [(5.0, 3.2)] * 8}), {})
    assert out["edge_case_formation_members"] == ["d_1", "d_2"]
    assert out["edge_case_formation_absent"] == ["d_2"]
    assert out["edge_case_formation_formed_at"] == pytest.approx(3.0)
    assert out["edge_case_formation_broke_at"] is None


def test_a_case_without_a_formation_has_no_formation_panel(tmp_path: pathlib.Path, formation: EdgeCaseFormationCalculator) -> None:
    world = _world(tmp_path, provenance="id: c\nowned: {C: [edge_0]}\n")
    out = formation.calculate(_episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)}), {})
    assert out == dict.fromkeys(formation.output_keys())


def test_the_formation_center_moves_with_the_robot_level_into_the_map_frame(tmp_path: pathlib.Path, formation: EdgeCaseFormationCalculator) -> None:
    world = _world(tmp_path, world_yaml=TWO_LEVELS, provenance=CASE, robot_level="1")
    ox, oy = WorldView(tmp_path / "worlds" / world).level_origins()["1"]
    assert (ox, oy) != (0.0, 0.0)
    out = formation.calculate(_episode(tmp_path, world, _run(world), {"d_1": _standing(5.0 + ox, 3.0 + oy), "d_2": _standing(5.5 + ox, 3.0 + oy)}), {})
    assert out["edge_case_formation_arrived"] == 2
    assert out["edge_case_formation_formed_at"] == pytest.approx(0.0)


def test_the_encounter_point_moves_with_the_robot_level_into_the_map_frame(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, world_yaml=TWO_LEVELS, provenance=CASE, robot_level="1")
    ox, oy = WorldView(tmp_path / "worlds" / world).level_origins()["1"]
    episode = _episode(tmp_path, world, _run(world), {"edge_0": [(6.0 + ox, 4.0 + oy)] * 3 + [(6.0 + ox, oy)] * 8})
    shifted = {"tf_gt": episode.topics["tf_gt"].with_columns(pl.col("pos_x_gt") + ox, pl.col("pos_y_gt") + oy), "peds": episode.topics["peds"]}
    out = criticality.calculate(AlignedEpisodeBundle(episode_id=1, data=episode.data, start_pos=[], goal_pos=[], map=world, run=episode.run, topics=shifted, run_dir=tmp_path), {})
    assert out["edge_case_pet_s"] == pytest.approx(3.0) and out["edge_case_pet_with"] == "edge_0"


@pytest.mark.parametrize("suffix", ["[0]", "[0,1]"])
def test_a_level_filtered_world_reads_the_snapshot_of_its_bare_name(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator, suffix: str) -> None:
    world = _world(tmp_path, provenance=CASE)
    out = criticality.calculate(_episode(tmp_path, world + suffix, _run(world + suffix), {"edge_0": _standing(6.0, 1.5)}), {})
    assert out["edge_case_id"] == "hospital_1__u_007"


def test_a_composed_world_reads_its_snapshot(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE, name=f"edge_case_{tmp_path.name},edge_case_{tmp_path.name}")
    out = criticality.calculate(_episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)}), {})
    assert out["edge_case_id"] == "hospital_1__u_007"


def test_a_regenerated_provenance_is_read_again(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    assert criticality.calculate(_episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)}), {})["edge_case_id"] == "hospital_1__u_007"
    path = tmp_path / "worlds" / world / "scenarios" / "case" / "provenance.yaml"
    path.write_text(CASE.replace("hospital_1__u_007", "hospital_1__u_008"))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert criticality.calculate(_episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)}), {})["edge_case_id"] == "hospital_1__u_008"


def test_each_run_reads_its_own_snapshot(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    world = _world(first, provenance=CASE, name="edge_case_shared")
    _world(second, provenance=CASE.replace("hospital_1__u_007", "hospital_1__u_008"), name="edge_case_shared")
    assert criticality.calculate(_episode(first, world, _run(world), {"edge_0": _standing(6.0, 1.5)}), {})["edge_case_id"] == "hospital_1__u_007"
    assert criticality.calculate(_episode(second, world, _run(world), {"edge_0": _standing(6.0, 1.5)}), {})["edge_case_id"] == "hospital_1__u_008"


def _timeline(fired: dict[str, float]) -> pl.DataFrame:
    """A semantic snapshot whose timeline entity carries the `fired` stimulus fields, flattened by the MCAP reader."""
    from task_generator_msgs.msg import SemanticEntityState

    entity = SemanticEntityState(entity="env_0/timeline", kind="timeline", discrete_names=list(fired), discrete_values=[str(at) for at in fired.values()])
    rows: defaultdict[str, list] = defaultdict(list)
    MCAPReader._append_semantic_entity(rows, 0, 0, "hall", entity)
    return pl.DataFrame(dict(rows), schema_overrides={"value_str": pl.String, "value_num": pl.Float64, "value_bool": pl.Boolean, "value_list": pl.List(pl.String)})


def _intercept(tmp_path: pathlib.Path, fired: dict[str, float] | None) -> AlignedEpisodeBundle:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world), {"edge_0": [(6.0, 3.0 - min(t, 3) * 1.0) for t in range(11)]})
    episode.semantic_snapshot = None if fired is None else _timeline(fired)
    return episode


def test_an_encounter_whose_stimulus_never_fired_is_not_scored(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    out = criticality.calculate(_intercept(tmp_path, {}), {})
    assert (out["edge_case_played"], out["edge_case_unplayed"], out["edge_case_triggered_at"]) == (False, ["edge_0"], None)
    assert (out["edge_case_scope"], out["edge_case_agents"]) == ("unplayed", [])
    assert out["edge_case_pet_s"] is None and out["edge_case_collided"] is None and out["edge_case_min_clearance_m"] is None


def test_an_encounter_whose_stimulus_fired_is_played_and_scored(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    out = criticality.calculate(_intercept(tmp_path, {"stimulus.edge_0.go": 1.5}), {})
    assert (out["edge_case_played"], out["edge_case_unplayed"], out["edge_case_triggered_at"]) == (True, [], pytest.approx(1.5))
    assert out["edge_case_scope"] == "own" and out["edge_case_collided"] is True
    assert out["edge_case_pet_s"] == pytest.approx(3.0)


def test_a_stimulus_stamped_before_the_episode_is_not_this_episodes(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    out = criticality.calculate(_intercept(tmp_path, {"stimulus.edge_0.go": -4.0, "stimulus.walker.go": 2.0}), {})
    assert (out["edge_case_played"], out["edge_case_unplayed"]) == (False, ["edge_0"])


def test_the_formation_forms_from_its_recorded_form_stimulus(tmp_path: pathlib.Path, formation: EdgeCaseFormationCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world), {"d_1": _standing(5.0, 3.2), "d_2": _standing(5.5, 3.0)})
    out = formation.calculate(episode, {})
    assert out["edge_case_formation_formed_at"] == pytest.approx(0.0)
    assert out["edge_case_formation_form_fired_at"] is None and out["edge_case_formation_break_fired_at"] is None
    episode.semantic_snapshot = _timeline({"stimulus.d_1.form": 3.0, "stimulus.d_2.form": 3.2, "stimulus.d_2.break": 6.0})
    out = formation.calculate(episode, {})
    assert out["edge_case_formation_formed_at"] == pytest.approx(3.0)
    assert out["edge_case_formation_form_fired_at"] == pytest.approx(3.0)
    assert out["edge_case_formation_break_fired_at"] == pytest.approx(6.0)


def test_the_ttc_follows_a_moving_pedestrians_twist(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    ahead = [(7.0 + 0.5 * k, 0.0) for k in range(11)]
    out = criticality.calculate(_episode(tmp_path, world, _run(world), {"edge_0": ahead}, twists={"edge_0": (0.5, 0.0)}), {})
    assert out["edge_case_collided"] is False
    assert (out["edge_case_min_ttc_s"], out["edge_case_min_ttc_with"], out["edge_case_min_ttc_at"]) == (pytest.approx(2.0), "edge_0", pytest.approx(9.9))


def test_a_pedestrian_walking_into_the_passing_robot_is_at_fault(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    crossing = [(5.0, 3.0 - 0.5 * k) for k in range(11)]
    out = criticality.calculate(_episode(tmp_path, world, _run(world), {"edge_0": crossing}, twists={"edge_0": (0.0, -0.5)}), {})
    assert (out["edge_case_collided"], out["edge_case_collision_fault"], out["edge_case_collision_at"]) == (True, "pedestrian", pytest.approx(5.0))


def test_a_head_on_contact_is_mutual_and_without_twists_the_robots(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    oncoming = [(8.0 - 0.5 * k, 0.0) for k in range(11)]
    episode = _episode(tmp_path, world, _run(world), {"edge_0": oncoming}, twists={"edge_0": (-0.5, 0.0)})
    out = criticality.calculate(episode, {})
    assert (out["edge_case_collision_fault"], out["edge_case_collision_at"]) == ("mutual", pytest.approx(5.0))
    episode.topics["peds"] = episode.topics["peds"].drop("peds_twists")
    assert criticality.calculate(episode, {})["edge_case_collision_fault"] == "robot"


def test_a_single_robot_pose_is_not_scored(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)})
    episode.topics["tf_gt"] = episode.topics["tf_gt"].head(1)
    out = criticality.calculate(episode, {})
    assert out["edge_case_id"] == "hospital_1__u_007" and out["edge_case_scope"] == "all"
    assert out["edge_case_min_clearance_m"] is None and out["edge_case_freeze_s"] is None


def test_a_world_that_resolves_nowhere_yields_nothing(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = f"edge_case_unknown_{tmp_path.name}"
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)})
    assert world_view(world, tmp_path) is None
    assert criticality.calculate(episode, {}) == dict.fromkeys(criticality.output_keys())


def test_an_episode_without_a_map_falls_back_to_the_calculator_world(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)})
    episode.map = None
    assert criticality.calculate(episode, {}) == dict.fromkeys(criticality.output_keys())
    criticality.world = world
    assert criticality.calculate(episode, {})["edge_case_id"] == "hospital_1__u_007"


def _registry_run(episode: AlignedEpisodeBundle, *, topics: set[str], pedsim: bool = True) -> dict:
    return MetricRegistry(RobotParams(robot_radius=ROBOT_RADIUS), world=episode.map).run(episode, available_topics=topics, pedsim_available=pedsim)


def test_the_registry_gates_both_calculators_on_their_topics_and_pedsim(tmp_path: pathlib.Path) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world), {"edge_0": [(6.0, 3.0 - min(t, 3) * 1.0) for t in range(11)], "d_1": _standing(5.0, 3.0), "d_2": _standing(5.5, 3.0)})
    out = _registry_run(episode, topics={"tf_gt", "peds"})
    assert out["edge_case_id"] == "hospital_1__u_007" and out["edge_case_collided"] is True
    assert out["edge_case_formation_formed_at"] == pytest.approx(0.0)
    keys = [*EdgeCaseCriticalityCalculator.output_keys(), *EdgeCaseFormationCalculator.output_keys()]
    for gated in (_registry_run(episode, topics={"tf_gt"}), _registry_run(episode, topics={"peds"}), _registry_run(episode, topics={"tf_gt", "peds"}, pedsim=False)):
        assert all(gated[key] is None for key in keys)


def test_a_malformed_encounter_leaves_the_case_unscored(tmp_path: pathlib.Path) -> None:
    world = _world(tmp_path, provenance=CASE.replace("[{agent: edge_0, point: [6.0, 0.0]}]", "[{agent: edge_0}]"))
    out = _registry_run(_episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)}), topics={"tf_gt", "peds"})
    assert all(out[key] is None for key in [*EdgeCaseCriticalityCalculator.output_keys(), *EdgeCaseFormationCalculator.output_keys()])


def test_the_pipeline_scores_a_recorded_case_against_its_run_snapshot(tmp_path: pathlib.Path) -> None:
    run_dir = tmp_path / "run"
    world = _world(run_dir, provenance=CASE)
    episode_dir = run_dir / "episodes" / "episode_000"
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode_000.mcap").touch()
    MetadataWriter.write(_run(world).model_copy(update={"pedsim_available": True}), episode_dir / "episode_000.yaml")
    recorded = _episode(run_dir, world, _run(world), {"edge_0": [(6.0, 3.0 - min(t, 3) * 1.0) for t in range(11)]})
    times = recorded.topics["tf_gt"]["time_ns"]
    odom = pl.DataFrame({"time_ns": times, "stamp_ns": times, "pos_x": recorded.topics["tf_gt"]["pos_x_gt"], "pos_y": [0.0] * len(times), "yaw": [0.0] * len(times), "vel_linear": [1.0] * len(times), "vel_angular": [0.0] * len(times)})
    record = pl.DataFrame({"time_ns": [0, 10 * SEC], "episode_id": [1, 1], "outcome_state": [1, 2], "outcome_info": ["", "finished"]})
    TopicParquetStore.write({"env_0_jackal_0": TopicBundle(odom=odom, tf_gt=recorded.topics["tf_gt"], peds=recorded.topics["peds"], episode_record=record)}, episode_dir / "topics")
    descriptor = EpisodeDescriptor(episode_dir=str(episode_dir), benchmark_id="run", episode_id=0, planner="nav2", stage="case", map=world)
    rows = ProcessingPipeline(FolderManager(data_root=tmp_path), workers=1).process_episode(descriptor)
    assert [row["edge_case_id"] for row in rows] == ["hospital_1__u_007"]
    assert rows[0]["edge_case_collided"] is True and rows[0]["edge_case_collision_with"] == "edge_0"
    assert rows[0]["edge_case_pet_s"] == pytest.approx(3.0)
