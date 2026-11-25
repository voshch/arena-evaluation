"""Edge-case criticality and formation metrics, measured from recorded trajectories and scoped by the scenario's provenance.yaml."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
from arena_runtime.env.task_layer import task_layer
from arena_simulation_setup.tree.World import WorldView

from arena_evaluation.processing.metrics.edge_case.criticality import (
    Criticality,
    FormationHold,
    Sample,
    formation_hold,
    score,
)
from arena_evaluation.processing.metrics.edge_case.provenance import (
    world_view,
)
from arena_evaluation.processing.metrics.edge_case.edge_case_metrics import (
    EdgeCaseCriticalityCalculator,
    EdgeCaseFormationCalculator,
)
from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.pipeline import ProcessingPipeline
from flat_peds import flat_peds
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import (
    AlignedEpisodeBundle,
    EpisodeDescriptor,
    RunMetadata,
)
from robot_params_factory import make_robot_params

SEC = 1_000_000_000
ROBOT_RADIUS = 0.3
PED_RADIUS_M = 0.3

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


def _contact(robot_vel: tuple[float, float], ped_vel: tuple[float, float]) -> Criticality:
    """Two samples, contact on the second (centers 0.5 m apart, radii 0.3 and 0.25)."""
    return score(
        [
            Sample(
                t=0.0,
                robot=(0.0, 0.0),
                robot_vel=robot_vel,
                robot_radius=0.3,
                peds={"p": ((3.0, 0.0), ped_vel, 0.25)},
            ),
            Sample(
                t=1.0,
                robot=(0.0, 0.0),
                robot_vel=robot_vel,
                robot_radius=0.3,
                peds={"p": ((0.5, 0.0), ped_vel, 0.25)},
            ),
        ],
        settle_s=0.0,
    )


def _assert_closing(value: float, expected: str) -> None:
    if expected == "positive":
        assert value > 0.5
    elif expected == "near-zero":
        assert abs(value) < 0.05
    elif expected == "exact-zero":
        assert value == 0.0
    else:
        assert value < 0.0


def _trio(t: float, a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> Sample:
    return Sample(
        t=t,
        robot=(0.0, 0.0),
        peds={
            "d_1": (a, (0.0, 0.0), 0.3),
            "d_2": (b, (0.0, 0.0), 0.3),
            "d_3": (c, (0.0, 0.0), 0.3),
        },
    )


def test_formation_hold_reports_when_the_group_broke() -> None:
    held = [_trio(t, (0.0, 0.0), (1.0, 0.0), (0.5, 0.8)) for t in (0.0, 1.0, 2.0)]
    broke = [
        *held,
        _trio(3.0, (0.0, 0.0), (1.0, 0.0), (4.0, 4.0)),
        _trio(4.0, (0.0, 0.0), (1.0, 0.0), (5.0, 5.0)),
    ]
    out = formation_hold(broke, ["d_1", "d_2", "d_3"])
    assert isinstance(out, FormationHold)
    assert (out.seen, out.formed_at, out.held_s, out.broke_at) == (True, 0.0, 2.0, 3.0)
    out = formation_hold(held, ["d_1", "d_2", "d_3"])
    assert out.seen and out.held_s == 2.0 and out.broke_at is None
    assert formation_hold(held, ["nobody"]).seen is False
    assert formation_hold([], ["d_1"]).seen is False


def _pair(t: float, a: tuple[float, float] | None, b: tuple[float, float] | None) -> Sample:
    peds = {name: (p, (0.0, 0.0), 0.3) for name, p in (("d_1", a), ("d_2", b)) if p is not None}
    return Sample(t=t, robot=(0.0, 0.0), peds=peds)


def test_a_member_leaving_the_recording_breaks_the_formation() -> None:
    samples = [
        _pair(0.0, (0.0, 0.0), (1.0, 0.0)),
        _pair(1.0, (0.0, 0.0), (1.0, 0.0)),
        _pair(2.0, (0.0, 0.0), None),
        _pair(3.0, (0.0, 0.0), (1.0, 0.0)),
    ]
    out = formation_hold(samples, ["d_1", "d_2"], center=(0.5, 0.0))
    assert (out.formed_at, out.held_s, out.broke_at, out.absent) == (0.0, 1.0, 2.0, ())


def _world(
    root: pathlib.Path,
    *,
    world_yaml: str = ONE_LEVEL,
    provenance: str | None = None,
    robot_level: str | None = None,
    humans: str = "",
    name: str | None = None,
) -> str:
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


def _run(
    world: str,
    *,
    obstacles: str = "scenario",
    robots: str = "random",
    file: str = "case",
) -> RunMetadata:
    namespace = task_layer().mode_namespaces()["obstacles"]["scenario"]
    return RunMetadata(
        benchmark_id="edge",
        planner="nav2",
        map=world,
        stage="case",
        recording_started_at="2026-10-21T00:00:00Z",
        python_version="3.12",
        ros_distro="jazzy",
        modes={"robots": [robots], "obstacles": [obstacles]},
        params={f"{namespace}.file": file} if "scenario" in (robots, obstacles) else {},
    )


def _episode(
    run_dir: pathlib.Path,
    world: str,
    run: RunMetadata,
    peds: dict[str, list[tuple[float, float]]],
    *,
    seconds: int = 10,
    twists: dict[str, tuple[float, float]] | None = None,
) -> AlignedEpisodeBundle:
    """The robot drives along y = 0 at 1 m/s, each pedestrian on its per-second track at its `twists` velocity, at 10 Hz."""
    times = [i * SEC // 10 for i in range(seconds * 10 + 1)]
    xs = [t / SEC for t in times]
    zeros = [0.0] * len(times)
    data = pl.DataFrame({"t_ns": times, "x": xs, "y": zeros, "yaw": zeros})
    robot_tracks = pl.DataFrame(
        {
            "t_ns": times,
            "x": xs,
            "y": zeros,
            "yaw": zeros,
            "pose_source": ["tf_gt"] * len(times),
        }
    )
    names = sorted(peds)
    frames = []
    for t in times:
        second = min(t // SEC, seconds)
        frames.append(
            (
                t,
                [
                    {
                        "ped_id": f"Pedestrian_{i}",
                        "ped_name": f"env_0/{name}",
                        "x": peds[name][second][0],
                        "y": peds[name][second][1],
                        "vx": (twists or {}).get(name, (0.0, 0.0))[0],
                        "vy": (twists or {}).get(name, (0.0, 0.0))[1],
                    }
                    for i, name in enumerate(names)
                ],
            )
        )
    pedestrians = flat_peds(frames).with_columns(radius_m=pl.lit(PED_RADIUS_M, dtype=pl.Float64))
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[10.0, 0.0],
        map=world,
        run=run,
        topics={"robot_tracks": robot_tracks, "pedestrian_arena_peds": pedestrians},
        run_dir=run_dir,
        peds=pedestrians,
    )


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
    return EdgeCaseCriticalityCalculator(make_robot_params(robot_radius=ROBOT_RADIUS))


@pytest.fixture
def formation() -> EdgeCaseFormationCalculator:
    return EdgeCaseFormationCalculator(make_robot_params(robot_radius=ROBOT_RADIUS))


def test_a_scenario_without_provenance_yields_nothing(
    tmp_path: pathlib.Path,
    criticality: EdgeCaseCriticalityCalculator,
    formation: EdgeCaseFormationCalculator,
) -> None:
    world = _world(tmp_path)
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(5.0, 0.0)})
    assert criticality.calculate(episode, {}) == dict.fromkeys(criticality.output_keys())
    assert formation.calculate(episode, {}) == dict.fromkeys(formation.output_keys())


def test_an_episode_that_ran_no_scenario_yields_nothing(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(
        tmp_path,
        world,
        _run(world, obstacles="random"),
        {"edge_0": _standing(5.0, 0.0)},
    )
    assert criticality.calculate(episode, {}) == dict.fromkeys(criticality.output_keys())


def test_an_unknown_scenario_yields_nothing(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world, file="elsewhere"), {"edge_0": _standing(5.0, 0.0)})
    assert criticality.calculate(episode, {}) == dict.fromkeys(criticality.output_keys())


def test_each_agent_counts_with_the_radius_its_scenario_entry_pins(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    humans = "humans:\n- {name: edge_0, model: adult, agent: {agent_type: adult, radius: 0.5}}\n- {name: d_1, model: adult, agent: adult}\n"
    world = _world(tmp_path, provenance=CASE, humans=humans)
    out = criticality.calculate(
        _episode(
            tmp_path,
            world,
            _run(world),
            {"edge_0": _standing(6.0, 2.0), "d_1": _standing(3.0, 1.5)},
        ),
        {},
    )
    assert out["edge_case_min_clearance_with"] == "d_1"
    assert out["edge_case_min_clearance_m"] == pytest.approx(1.5 - ROBOT_RADIUS - 0.3)
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 2.0)})
    assert criticality.calculate(episode, {})["edge_case_min_clearance_m"] == pytest.approx(2.0 - ROBOT_RADIUS - 0.5)


def test_an_unpinned_pedestrian_without_a_recorded_radius_is_refused(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)})
    episode.topics["pedestrian_arena_peds"] = episode.topics["pedestrian_arena_peds"].with_columns(radius_m=pl.lit(None, dtype=pl.Float64))
    with pytest.raises(MetricInputError):
        criticality.calculate(episode, {})


HELD_CASE = CASE + "robot: {speed_mps: 1.0, hold_s: 3.0, traversals: 1, tolerance_m: null, radius_m: 0.35}\n"


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


def _level_shifted_episode(tmp_path: pathlib.Path, stored_origins: tuple[float, float] | None) -> AlignedEpisodeBundle:
    """Episode shifted by the robot level origin, derived from the geometry or read from the stored file."""
    world = _world(tmp_path, world_yaml=TWO_LEVELS, provenance=CASE, robot_level="1")
    if stored_origins is None:
        origin = WorldView(tmp_path / "worlds" / world).level_origins()["1"]
        assert origin != (0.0, 0.0)
        map_name = world
    else:
        origin = stored_origins
        (tmp_path / "worlds" / world / "level_origins.yaml").write_text(f"'0': [0.0, 0.0]\n'1': [{origin[0]}, {origin[1]}]\n")
        map_name = world + "[1]"
    ox, oy = origin
    episode = _episode(
        tmp_path,
        map_name,
        _run(map_name),
        {"edge_0": [(6.0 + ox, 4.0 + oy)] * 3 + [(6.0 + ox, oy)] * 8},
    )
    return AlignedEpisodeBundle(
        episode_id=1,
        data=episode.data.with_columns(pl.col("x") + ox, pl.col("y") + oy),
        start_pos=[],
        goal_pos=[],
        map=map_name,
        run=episode.run,
        topics={
            "robot_tracks": episode.topics["robot_tracks"].with_columns(pl.col("x") + ox, pl.col("y") + oy),
            "pedestrian_arena_peds": episode.topics["pedestrian_arena_peds"],
        },
        run_dir=tmp_path,
    )


@pytest.mark.parametrize("stored_origins", [None, (40.0, -3.0)], ids=["derived", "stored"])
def test_the_encounter_point_moves_with_the_robot_level_into_the_map_frame(
    tmp_path: pathlib.Path,
    criticality: EdgeCaseCriticalityCalculator,
    stored_origins: tuple[float, float] | None,
) -> None:
    out = criticality.calculate(_level_shifted_episode(tmp_path, stored_origins), {})
    assert out["edge_case_pet_s"] == pytest.approx(3.0) and out["edge_case_pet_with"] == "edge_0"


def _timeline(fired: dict[str, float]) -> pl.DataFrame:
    """A semantic snapshot whose timeline entity carries the `fired` stimulus fields, matching world_semantics schema."""
    base = {
        "t_ns": 0,
        "env_id": "env_0",
        "world": "hall",
        "entity_name": "env_0/timeline",
        "kind": "timeline",
    }
    rows = [
        {
            **base,
            "field_name": "members",
            "field_kind": "members",
            "value_str": "",
            "value_num": None,
            "value_bool": None,
        }
    ]
    for name, at in fired.items():
        rows.append(
            {
                **base,
                "field_name": name,
                "field_kind": "discrete",
                "value_str": str(at),
                "value_num": None,
                "value_bool": None,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "t_ns": pl.Int64,
            "env_id": pl.String,
            "world": pl.String,
            "entity_name": pl.String,
            "kind": pl.String,
            "field_name": pl.String,
            "field_kind": pl.String,
            "value_str": pl.String,
            "value_num": pl.Float64,
            "value_bool": pl.Boolean,
        },
    )


def _intercept(tmp_path: pathlib.Path, fired: dict[str, float] | None) -> AlignedEpisodeBundle:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(
        tmp_path,
        world,
        _run(world),
        {"edge_0": [(6.0, 3.0 - min(t, 3) * 1.0) for t in range(11)]},
    )
    episode.semantic_snapshot = None if fired is None else _timeline(fired)
    return episode


def test_an_encounter_whose_stimulus_fired_is_played_and_scored(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    out = criticality.calculate(_intercept(tmp_path, {"stimulus.edge_0.go": 1.5}), {})
    assert (
        out["edge_case_played"],
        out["edge_case_unplayed"],
        out["edge_case_triggered_at"],
    ) == (True, [], pytest.approx(1.5))
    assert out["edge_case_scope"] == "own" and out["edge_case_collided"] is True
    assert out["edge_case_pet_s"] == pytest.approx(3.0)


def test_a_stimulus_stamped_before_the_episode_is_not_this_episodes(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    out = criticality.calculate(
        _intercept(tmp_path, {"stimulus.edge_0.go": -4.0, "stimulus.walker.go": 2.0}),
        {},
    )
    assert (out["edge_case_played"], out["edge_case_unplayed"]) == (False, ["edge_0"])


def test_the_formation_forms_from_its_recorded_form_stimulus(tmp_path: pathlib.Path, formation: EdgeCaseFormationCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(
        tmp_path,
        world,
        _run(world),
        {"d_1": _standing(5.0, 3.2), "d_2": _standing(5.5, 3.0)},
    )
    out = formation.calculate(episode, {})
    assert out["edge_case_formation_formed_at"] == pytest.approx(0.0)
    assert out["edge_case_formation_form_fired_at"] is None and out["edge_case_formation_break_fired_at"] is None
    episode.semantic_snapshot = _timeline({"stimulus.d_1.form": 3.0, "stimulus.d_2.form": 3.2, "stimulus.d_2.break": 6.0})
    out = formation.calculate(episode, {})
    assert out["edge_case_formation_formed_at"] == pytest.approx(3.0)
    assert out["edge_case_formation_form_fired_at"] == pytest.approx(3.0)
    assert out["edge_case_formation_break_fired_at"] == pytest.approx(6.0)


def test_a_single_robot_pose_is_not_scored(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)})
    episode.topics["robot_tracks"] = episode.topics["robot_tracks"].head(1)
    episode.data = episode.data.head(1)
    out = criticality.calculate(episode, {})
    assert out["edge_case_id"] == "hospital_1__u_007" and out["edge_case_scope"] == "all"
    assert out["edge_case_min_clearance_m"] is None and out["edge_case_freeze_s"] is None


def test_a_world_that_resolves_nowhere_yields_nothing(tmp_path: pathlib.Path, criticality: EdgeCaseCriticalityCalculator) -> None:
    world = f"edge_case_unknown_{tmp_path.name}"
    episode = _episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)})
    assert world_view(world, tmp_path) is None
    assert criticality.calculate(episode, {}) == dict.fromkeys(criticality.output_keys())


def _registry_run(episode: AlignedEpisodeBundle, *, topics: set[str], pedsim: bool = True) -> dict:
    registry = MetricRegistry(make_robot_params(robot_radius=ROBOT_RADIUS), world=episode.map)
    results = dict(registry.run(episode, available_topics=topics, pedsim_available=pedsim))
    for scope in (MetricScope.ROBOT, MetricScope.CONTACT, MetricScope.ENCOUNTER):
        scope_slice = ScopeSlice(
            scope=scope,
            slice_id=scope.name.lower(),
            t_start_ns=0,
            t_end_ns=10 * SEC,
            identity={},
            window_bundle=episode,
            full_episode=episode,
        )
        results |= registry.run_scope(
            scope,
            episode,
            available_topics=topics,
            pedsim_available=pedsim,
            scope_slice=scope_slice,
        ).results
    return results


def test_the_registry_gates_both_calculators_on_their_topics_and_pedsim(
    tmp_path: pathlib.Path,
) -> None:
    world = _world(tmp_path, provenance=CASE)
    episode = _episode(
        tmp_path,
        world,
        _run(world),
        {
            "edge_0": [(6.0, 3.0 - min(t, 3) * 1.0) for t in range(11)],
            "d_1": _standing(5.0, 3.0),
            "d_2": _standing(5.5, 3.0),
        },
    )
    out = _registry_run(episode, topics={"robot_tracks", "pedestrian_arena_peds"})
    assert out["edge_case_id"] == "hospital_1__u_007" and out["edge_case_collided"] is True
    assert out["edge_case_formation_formed_at"] == pytest.approx(0.0)
    keys = [
        *EdgeCaseCriticalityCalculator.output_keys(),
        *EdgeCaseFormationCalculator.output_keys(),
    ]
    for gated in (
        _registry_run(episode, topics={"robot_tracks"}),
        _registry_run(episode, topics={"pedestrian_arena_peds"}),
        _registry_run(episode, topics={"robot_tracks", "pedestrian_arena_peds"}, pedsim=False),
    ):
        assert all(gated[key] is None for key in keys)


def test_a_malformed_encounter_fails_loudly(tmp_path: pathlib.Path) -> None:
    world = _world(
        tmp_path,
        provenance=CASE.replace("[{agent: edge_0, point: [6.0, 0.0]}]", "[{agent: edge_0}]"),
    )
    with pytest.raises(MetricInputError, match="point"):
        _registry_run(
            _episode(tmp_path, world, _run(world), {"edge_0": _standing(6.0, 1.5)}),
            topics={"robot_tracks", "pedestrian_arena_peds"},
        )


def _flat_robot(times: pl.Series, xs: list[float], ys: list[float], robot_id: str, speed_mps: float) -> dict[str, pl.DataFrame]:
    """One robot's flat telemetry tables on the case clock: its track, odometry, descriptor and params."""
    n = len(times)
    name = robot_id.removeprefix("env_0_")
    tracks = pl.DataFrame(
        {
            "t_ns": times,
            "x": xs,
            "y": ys,
            "yaw": [0.0] * n,
            "pose_source": ["tf_gt"] * n,
        }
    ).with_columns(
        env_id=pl.lit("env_0"),
        robot_id=pl.lit(robot_id),
        z=0.0,
        vx=speed_mps,
        vy=0.0,
        vz=0.0,
        omega_z=0.0,
        speed_mps=speed_mps,
    )
    odom = pl.DataFrame(
        {
            "t_ns": times,
            "env_id": ["env_0"] * n,
            "robot_id": [robot_id] * n,
            "odom_x": xs,
            "odom_y": ys,
            "odom_yaw": [0.0] * n,
            "odom_vx": [speed_mps] * n,
            "odom_wz": [0.0] * n,
            "drift_error_m": [0.0] * n,
        }
    )
    descriptor = pl.DataFrame(
        {
            "t_ns": [0],
            "env_id": ["env_0"],
            "robot_id": [robot_id],
            "name": [name],
            "model": ["jackal"],
            "ns": [f"/env_0/{name}"],
            "frame": ["env_0/odom"],
            "base_frame": [f"env_0/{name}/base_link"],
        }
    )
    params = pl.DataFrame(
        {
            "t_ns": [0, 0],
            "env_id": ["env_0", "env_0"],
            "robot_id": [robot_id, robot_id],
            "param_key": ["robot_radius", "robot_mass_kg"],
            "param_value": [str(ROBOT_RADIUS), "18.4"],
        }
    )
    task_pose = pl.DataFrame(
        {
            "t_ns": times,
            "env_id": ["env_0"] * n,
            "robot_id": [robot_id] * n,
            "stamp_ns": times,
            "pos_x": xs,
            "pos_y": ys,
            "yaw": [0.0] * n,
        }
    )
    return {
        "robot_tracks": tracks,
        "robot_odom": odom,
        "robot_descriptor": descriptor,
        "robot_params": params,
        "robot_task_pose": task_pose,
    }


def _edge_contact(robot_id: str) -> pl.DataFrame:
    """The collision publisher's contact of the robot with edge_0 at 6 s."""
    return pl.DataFrame(
        {
            "t_ns": [6 * SEC],
            "env_id": ["env_0"],
            "robot_id": [robot_id],
            "obstacle_id": ["edge_0"],
            "kind": ["pedestrian"],
            "obstacle_x": [6.0],
            "obstacle_y": [0.0],
            "obstacle_z": [0.0],
            "distance": [0.0],
        }
    )


def _record_case_episode(
    run_dir: pathlib.Path,
    world: str,
    *,
    peds: dict[str, list[tuple[float, float]]],
    robots: dict[str, tuple[list[float], list[float], float]],
    contacts: pl.DataFrame,
) -> EpisodeDescriptor:
    """One recorded episode's flat tables as the MCAP extraction writes them: every robot's rows share one table."""
    episode_dir = run_dir / "episodes" / "episode_000"
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode_000.mcap").touch()
    MetadataWriter.write(
        _run(world).model_copy(update={"pedsim_available": True}),
        episode_dir / "episode_000.yaml",
    )
    recorded = _episode(run_dir, world, _run(world), peds)
    times = recorded.topics["robot_tracks"]["t_ns"]
    parts: dict[str, list[pl.DataFrame]] = {}
    for robot_id, (xs, ys, speed_mps) in robots.items():
        for table, frame in _flat_robot(times, xs, ys, robot_id, speed_mps).items():
            parts.setdefault(table, []).append(frame)
    topics_dir = episode_dir / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)
    for table, frames in parts.items():
        pl.concat(frames, how="diagonal").write_parquet(topics_dir / f"{table}.parquet")
    recorded.topics["pedestrian_arena_peds"].write_parquet(topics_dir / "pedestrian_arena_peds.parquet")
    pl.DataFrame(
        {
            "t_ns": [0, 10 * SEC],
            "env_id": ["env_0", "env_0"],
            "episode_id": [1, 1],
            "outcome_state": [1, 2],
            "outcome_info": ["", "finished"],
            "goal_uuid": ["", ""],
            "phases_yaml": ["", ""],
            "params_yaml": ["", ""],
            "conditions_yaml": ["", ""],
            "world": [world, world],
            "seed": [0, 0],
            "human_params": ["", ""],
        }
    ).write_parquet(topics_dir / "episode_lifecycle.parquet")
    contacts.write_parquet(topics_dir / "robot_collision_contacts.parquet")
    TopicParquetStore.mark_complete(topics_dir)
    return EpisodeDescriptor(
        episode_dir=str(episode_dir),
        benchmark_id="run",
        episode_id=0,
        planner="nav2",
        stage="case",
        map=world,
    )


def test_the_pipeline_scores_a_recorded_case_against_its_run_snapshot(
    tmp_path: pathlib.Path,
) -> None:
    run_dir = tmp_path / "run"
    world = _world(run_dir, provenance=CASE)
    descriptor = _record_case_episode(
        run_dir,
        world,
        peds={"edge_0": [(6.0, 3.0 - min(t, 3) * 1.0) for t in range(11)]},
        robots={"env_0_jackal_0": ([i / 10 for i in range(101)], [0.0] * 101, 1.0)},
        contacts=_edge_contact("env_0_jackal_0"),
    )
    evaluation = ProcessingPipeline(FolderManager(data_root=tmp_path), workers=1).evaluate_episode(descriptor)
    assert [row["edge_case_id"] for row in evaluation.rows] == ["hospital_1__u_007"]
    assert evaluation.facts.robots.get_column("edge_case_collided").to_list() == [True]
    assert evaluation.facts.encounters.get_column("edge_case_pet_s").to_list() == [pytest.approx(3.0)]
    assert evaluation.facts.contacts.get_column("edge_case_collision_with").to_list() == ["edge_0"]
    assert evaluation.facts.contacts.get_column("edge_case_collision_at").to_list() == [pytest.approx(5.4, abs=0.05)]


def test_every_published_series_is_read_back_with_its_declared_class_and_a_unique_key(
    tmp_path: pathlib.Path,
) -> None:
    from arena_evaluation.storage.registry.series import SERIES_TABLES

    run_dir = tmp_path / "run"
    world = _world(run_dir, provenance=CASE)
    descriptor = _record_case_episode(
        run_dir,
        world,
        peds={
            "edge_0": [(6.0, 3.0 - min(t, 3) * 1.0) for t in range(11)],
            "passerby": _standing(2.0, 0.0),
        },
        robots={
            "env_0_jackal_0": ([i / 10 for i in range(101)], [0.0] * 101, 1.0),
            "env_0_jackal_1": ([2.0] * 101, [1.0] * 101, 0.0),
        },
        contacts=_edge_contact("env_0_jackal_0"),
    )
    evaluation = ProcessingPipeline(FolderManager(data_root=tmp_path), workers=1).evaluate_episode(descriptor)
    stored = sorted(evaluation.metrics_dir.glob("series_*.parquet"))

    assert {path.stem.removeprefix("series_") for path in stored} >= {
        "motion_metrics",
        "path_metrics",
        "gaze",
    }
    for path in stored:
        table = SERIES_TABLES[path.stem.removeprefix("series_")]
        frame = pl.read_parquet(path)
        assert frame.schema == table.polars_schema()
        assert not frame.select(list(table.key)).is_duplicated().any()
