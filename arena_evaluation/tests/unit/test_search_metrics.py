import json
import pathlib

import numpy as np
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("arena_simulation_setup.shared.judge")
pytest.importorskip("mcap_ros2.writer")

import yaml
from arena_evaluation.benchmark.config import Suite
from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.processing.metrics.performance.collision_metrics import (
    CollisionMetricsCalculator,
)
from arena_evaluation.processing.metrics.performance.search_metrics import (
    SearchMetricsCalculator,
    search_phase,
)
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.processing.path.theta_star import GeometricThetaStar
from arena_evaluation.processing.pipeline import _stage_budget
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams, RunMetadata
from arena_simulation_setup.shared.judge import WallGrid
from mcap_ros2.writer import Writer
from PIL import Image

S = 1_000_000_000
RES = 0.1
ROWS, COLS = 60, 220
RADIUS = 0.3
START = (1.0, 3.0)
PERSON = (11.0, 3.0)
FIND_RADIUS = 3.0
WAIT_S = 4.0
ROBOT_DIR = "env_0_jackal_0"
SEARCH_GOTO = {"goto": "person_0", "tolerance_radius": FIND_RADIUS, "dispatch": False}
PHASES = {"phases": [SEARCH_GOTO], "conditions": [], "map_poses": [None]}


def _blocked(partition_x: float | None = None) -> np.ndarray:
    """Occupancy of a 22 m x 6 m corridor, row 0 at y = 0, with an optional closed door across it."""
    blocked = np.zeros((ROWS, COLS), dtype=bool)
    blocked[:3, :] = True
    blocked[-3:, :] = True
    if partition_x is not None:
        col = round(partition_x / RES)
        blocked[:, col - 1 : col + 1] = True
    return blocked


def _solver(blocked: np.ndarray) -> GeometricThetaStar:
    return GeometricThetaStar(blocked, resolution=RES, origin=(0.0, 0.0), robot_radius=RADIUS)


def _walls(blocked: np.ndarray) -> WallGrid:
    return WallGrid(cells=np.flipud(blocked), origin=(0.0, 0.0), resolution=RES)


def _poses(end_s: float, stop_x: float, step_s: float = 0.5) -> pl.DataFrame:
    """The robot waits 4 s at the start, then drives east at 1 m/s and holds at `stop_x`."""
    t = np.arange(0.0, end_s + 1e-9, step_s)
    x = np.minimum(START[0] + np.maximum(t - WAIT_S, 0.0), stop_x)
    return pl.DataFrame({"time_ns": (t * S).astype(np.int64), "pos_x": x, "pos_y": np.full_like(t, START[1]), "yaw": np.zeros_like(t)})


def _snapshot(met_s: float | None) -> pl.DataFrame:
    entity = "env_0/jackal_0"
    rows = [(0, "met", ""), (0, "phase", "0")]
    if met_s is not None:
        rows.append((int(met_s * S), "met", "0"))
    return pl.DataFrame(
        {
            "time_ns": [r[0] for r in rows],
            "entity": [entity] * len(rows),
            "kind": ["robot"] * len(rows),
            "field": [r[1] for r in rows],
            "field_kind": ["discrete"] * len(rows),
            "value_str": [r[2] for r in rows],
            "value_num": [None] * len(rows),
            "value_bool": [None] * len(rows),
        }
    )


def _peds(end_s: float) -> pl.DataFrame:
    t = np.arange(0.0, end_s + 1e-9, 1.0)
    return pl.DataFrame(
        {
            "time_ns": (t * S).astype(np.int64),
            "peds_names": [["env_0/person_1", "env_0/person_0"]] * len(t),
            "peds_positions": [[2.0, 2.0, 0.0, PERSON[0], PERSON[1], 0.0]] * len(t),
        }
    )


def _search_state(*rows: tuple[float, dict]) -> pl.DataFrame:
    return pl.DataFrame({"time_ns": [int(t * S) for t, _ in rows], "state": [json.dumps(state) for _, state in rows]})


def _episode(poses: pl.DataFrame, met_s: float | None = None, phases: dict | None = PHASES, search_state: pl.DataFrame | None = None) -> AlignedEpisodeBundle:
    topics = {"task_pose": poses}
    if search_state is not None:
        topics["hearing_search_state"] = search_state
    return AlignedEpisodeBundle(
        episode_id=1,
        data=poses.select("time_ns"),
        start_pos=[START[0], START[1], 0.0],
        goal_pos=[],
        robot_name=ROBOT_DIR,
        semantic_snapshot=_snapshot(met_s),
        phases=phases,
        peds=_peds(float(poses["time_ns"][-1]) / S),
        map="search_corridor",
        topics=topics,
    )


def _calc() -> SearchMetricsCalculator:
    return SearchMetricsCalculator(RobotParams(RADIUS, 0.0, 10.0))


def _score(episode: AlignedEpisodeBundle, partition_x: float | None = None, timeout_s: float | None = None) -> dict:
    blocked = _blocked(partition_x)
    if timeout_s is not None:
        episode.timeout_s = timeout_s
    return _calc().calculate_on_grid(episode, {}, _solver(blocked), _walls(blocked))


def test_find_at_a_known_time_scores_time_tsr_and_spl() -> None:
    out = _score(_episode(_poses(20.0, stop_x=10.0), met_s=12.0))
    assert out["found"] is True
    assert out["time_to_find_s"] == pytest.approx(12.0)
    assert out["tsr"] == pytest.approx(1.0 - 12.0 / 180.0)
    # driven 8 m (x 1 -> 9) against a 10 m Theta* path less the 3 m find radius
    assert out["spl_search"] == pytest.approx(7.0 / 8.0, abs=0.02)


def test_waiting_before_the_find_is_stopped_time_and_the_rest_driving() -> None:
    out = _score(_episode(_poses(20.0, stop_x=10.0), met_s=12.0))
    assert out["stopped_s"] == pytest.approx(WAIT_S)
    assert out["driving_s"] == pytest.approx(12.0 - WAIT_S)


def test_tsr_and_budget_follow_the_stage_timeout() -> None:
    out = _score(_episode(_poses(20.0, stop_x=10.0), met_s=12.0), timeout_s=300.0)
    assert out["found"] is True
    assert out["tsr"] == pytest.approx(1.0 - 12.0 / 300.0)


def test_find_radii_are_replayed_from_the_recorded_paths() -> None:
    out = _score(_episode(_poses(20.0, stop_x=10.0), met_s=12.0))
    # the person is 14 - t metres away, the robot stops 1 m short of them at t = 13
    assert (out["found_3m"], out["time_to_find_3m_s"]) == (True, pytest.approx(11.0))
    assert (out["found_1_5m"], out["time_to_find_1_5m_s"]) == (True, pytest.approx(12.5))
    assert (out["found_1m"], out["time_to_find_1m_s"]) == (True, pytest.approx(13.0))


def test_a_track_that_stays_out_of_range_finds_nothing() -> None:
    out = _score(_episode(_poses(30.0, stop_x=6.0)))
    assert out["found"] is False
    assert out["time_to_find_s"] == 180.0
    assert out["tsr"] == 0.0
    assert out["spl_search"] == 0.0
    assert [out[f"found_{r}"] for r in ("1m", "1_5m", "3m")] == [False, False, False]
    assert [out[f"time_to_find_{r}_s"] for r in ("1m", "1_5m", "3m")] == [180.0, 180.0, 180.0]
    assert out["stopped_s"] == pytest.approx(25.0)
    assert out["driving_s"] == pytest.approx(out["time_to_find_s"] - out["stopped_s"])


def test_a_find_after_the_budget_is_not_a_find() -> None:
    out = _score(_episode(_poses(20.0, stop_x=10.0), met_s=12.0), timeout_s=10.0)
    assert out["found"] is False
    assert out["time_to_find_s"] == 10.0
    assert out["tsr"] == 0.0
    assert out["found_3m"] is False
    assert out["time_to_find_3m_s"] == 10.0


def test_a_closed_door_at_one_metre_hides_the_person_from_every_radius() -> None:
    episode = _episode(_poses(20.0, stop_x=10.0))
    open_corridor = _score(episode)
    assert open_corridor["found_1m"] is True
    behind_door = _score(episode, partition_x=10.5)
    assert [behind_door[f"found_{r}"] for r in ("1m", "1_5m", "3m")] == [False, False, False]
    assert [behind_door[f"time_to_find_{r}_s"] for r in ("1m", "1_5m", "3m")] == [180.0, 180.0, 180.0]


def test_find_clock_starts_at_first_robot_pose() -> None:
    episode = _episode(_poses(20.0, stop_x=10.0), met_s=12.0)
    episode.data = episode.data.with_columns(pl.col("time_ns") + 80_000_000)
    out = _score(episode)
    assert out["time_to_find_s"] == pytest.approx(12.0)
    assert out["time_to_find_1m_s"] == pytest.approx(13.0)


def test_met_outside_episode_is_ignored_but_final_driver_state_is_kept() -> None:
    state = _search_state((19.0, {"stops": 1}), (20.5, {"stops": 4}), (22.0, {"stops": 0}))
    episode = _episode(_poses(20.0, stop_x=10.0), met_s=22.0, search_state=state)
    episode.window_ns = (S, 21 * S)
    episode.semantic_snapshot = pl.concat([_snapshot(0.5), episode.semantic_snapshot])
    out = _score(episode)
    assert out["found"] is False
    assert out["stops"] == 4


def test_replay_drops_target_when_latest_peds_message_omits_it() -> None:
    episode = _episode(_poses(20.0, stop_x=10.0))
    episode.peds = pl.DataFrame({
        "time_ns": [0, 5 * S],
        "peds_names": [["person_0"], []],
        "peds_positions": [[*PERSON, 0.0], []],
    })
    out = _score(episode)
    assert out["found_3m"] is False
    assert out["time_to_find_3m_s"] == 180.0


def test_spl_uses_run_map_and_partial_segment_at_find(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    episode = _episode(_poses(20.0, stop_x=10.0), met_s=12.25)
    episode.run_dir = tmp_path
    episode.map = "usar_local_corridor"
    world = tmp_path / "worlds" / episode.map
    world.mkdir(parents=True)
    Image.fromarray(np.where(np.flipud(_blocked()), 0, 255).astype(np.uint8)).save(world / "map.png")
    (world / "map.yaml").write_text(yaml.safe_dump({"image": "map.png", "resolution": RES, "origin": [0, 0, 0]}))
    assert _calc().calculate(episode, {})["spl_search"] == pytest.approx(7.0 / 8.25)


def test_driver_counters_come_from_the_last_search_state_message() -> None:
    early = {"phase": "listen", "stops": 1, "listened_s": 2.0, "goals_sent": 2, "goals_failed": 0, "calls": 1}
    last = {"phase": "drive", "goal": [3.0, 4.0], "nav": "idle", "stops": 4, "listened_s": 8.0, "goals_sent": 6, "goals_failed": 1, "calls": 2}
    state = _search_state((9.0, last), (3.0, early))
    out = _score(_episode(_poses(20.0, stop_x=10.0), met_s=12.0, search_state=state))
    assert (out["stops"], out["goals_sent"], out["goals_failed"], out["calls"]) == (4, 6, 1, 2)
    assert out["listened_s"] == 8.0


def test_without_the_search_state_topic_only_the_counters_are_none() -> None:
    out = _score(_episode(_poses(20.0, stop_x=10.0), met_s=12.0))
    assert all(out[k] is None for k in ("stops", "listened_s", "goals_sent", "goals_failed", "calls"))
    assert out["found"] is True


def test_without_a_recorded_met_list_the_find_keys_are_none_but_the_replay_runs() -> None:
    episode = _episode(_poses(20.0, stop_x=10.0))
    episode.semantic_snapshot = None
    out = _score(episode)
    assert all(out[k] is None for k in ("found", "time_to_find_s", "tsr", "spl_search", "stopped_s", "driving_s"))
    assert out["found_1m"] is True


def test_without_a_wall_grid_the_replay_keys_are_none() -> None:
    out = _calc().calculate_on_grid(_episode(_poses(20.0, stop_x=10.0), met_s=12.0), {}, _solver(_blocked()), None)
    assert out["found"] is True
    assert all(out[k] is None for k in ("found_1m", "found_1_5m", "found_3m", "time_to_find_1m_s"))


def test_without_a_map_only_spl_search_is_none() -> None:
    out = _calc().calculate_on_grid(_episode(_poses(20.0, stop_x=10.0), met_s=12.0), {}, None, _walls(_blocked()))
    assert out["spl_search"] is None
    assert out["found"] is True


@pytest.mark.parametrize(
    "phases",
    [
        None,
        {"phases": [{"goto": [5.0, 3.0, 0.0], "tolerance_radius": 0.5}], "conditions": [], "map_poses": [[5.0, 3.0, 0.0]]},
        {"phases": [{"goto": "person_0", "tolerance_radius": 1.0}], "conditions": [], "map_poses": [None]},
        {"phases": [{"gesture": "wave"}], "conditions": [], "map_poses": [None]},
    ],
)
def test_an_episode_without_a_search_phase_scores_every_key_none(phases: dict | None) -> None:
    episode = _episode(_poses(20.0, stop_x=10.0), met_s=12.0, phases=phases)
    out = _calc().calculate(episode, {})
    assert set(out) == set(SearchMetricsCalculator.output_keys())
    assert all(v is None for v in out.values())


def test_search_phase_is_the_judge_only_goto_on_a_named_target() -> None:
    phases = {"phases": [{"gesture": "wave"}, SEARCH_GOTO]}
    assert search_phase(phases) == (1, "person_0", FIND_RADIUS)
    assert search_phase({"phases": [{"goto": "person_0", "dispatch": False}]}) == (0, "person_0", None)


def test_search_metrics_runs_after_its_dependencies_and_names_its_primary_outputs() -> None:
    registry = MetricRegistry(RobotParams(RADIUS))
    stage_of = {name: i for i, stage in enumerate(registry.execution_order()) for name in stage}
    for dep in SearchMetricsCalculator.DEPENDS_ON:
        assert stage_of[dep] < stage_of[SearchMetricsCalculator.NAME]
    assert SearchMetricsCalculator.PRIMARY_OUTPUTS == ["time_to_find_s", "found", "tsr"]
    assert set(SearchMetricsCalculator.PRIMARY_OUTPUTS) <= set(SearchMetricsCalculator.output_keys())


def _metadata(stage: str, reference_type: str | None = None) -> RunMetadata:
    return RunMetadata(benchmark_id="usar", planner="nav2", map="usar_rooms", stage=stage, recording_started_at="", python_version="", ros_distro="", reference_type=reference_type)


def _manifest_budget(tmp_path: pathlib.Path, stages: list[dict], metadata: RunMetadata) -> float | None:
    run = tmp_path / "20261116-000000-usar-inline"
    episode = run / "episodes" / "nav2" / "episode_000"
    episode.mkdir(parents=True)
    (run / "manifest.yaml").write_text(yaml.safe_dump({"run_id": run.name, "steps": [{"key": stage["name"], "stage": stage} for stage in stages]}))
    return _stage_budget(episode, tmp_path, metadata)


def test_a_stage_with_timeout_300_scores_a_250s_episode_as_found_and_a_run_without_a_manifest_as_timeout(tmp_path: pathlib.Path) -> None:
    # Reach the 3 m find radius on the last pose.
    poses = _poses(250.0, stop_x=8.0, step_s=5.0).with_columns(
        (START[0] + 7.0 * pl.col("time_ns") / (250 * S)).alias("pos_x")
    )
    episode = _episode(poses, met_s=250.0)
    collision = CollisionMetricsCalculator(RobotParams(RADIUS, 0.0, 10.0))
    blocked = _blocked()

    episode.timeout_s = _manifest_budget(tmp_path, [{"name": "rooms", "timeout": 300.0}, {"name": "hall", "timeout": 60.0}], _metadata("rooms"))
    assert episode.timeout_s == 300.0
    scored = _calc().calculate_on_grid(episode, {}, _solver(blocked), _walls(blocked))
    assert scored["found"] is True
    assert scored["time_to_find_s"] == pytest.approx(250.0)
    assert scored["tsr"] == pytest.approx(1.0 - 250.0 / 300.0)
    verdict = collision.calculate(episode, {"time_to_goal": 250.0})
    assert (verdict["result"], verdict["success"]) == ("GOAL_REACHED", True)

    episode.timeout_s = _stage_budget(tmp_path / "bare" / "episode_000", tmp_path, _metadata("rooms"))
    assert episode.timeout_s is None
    scored = _calc().calculate_on_grid(episode, {}, _solver(blocked), _walls(blocked))
    assert scored["found"] is False
    assert scored["time_to_find_s"] == 180.0
    verdict = collision.calculate(episode, {"time_to_goal": 250.0})
    assert (verdict["result"], verdict["success"]) == ("TIMEOUT", False)


def test_an_unhindered_peds_reference_takes_the_stage_timeout_peds(tmp_path: pathlib.Path) -> None:
    stages = [{"name": "rooms", "timeout": 300.0, "timeout_peds": 120.0}]
    assert _manifest_budget(tmp_path / "ref", stages, _metadata("rooms", reference_type="unhindered_peds")) == 120.0
    assert _manifest_budget(tmp_path / "run", stages, _metadata("rooms")) == 300.0


@pytest.mark.parametrize("stages", [[{"name": "hall", "timeout": 60.0}], [{"name": "rooms", "timeout": "soon"}], [{"name": "rooms"}]])
def test_a_stage_the_manifest_cannot_budget_has_no_budget(tmp_path: pathlib.Path, stages: list[dict]) -> None:
    assert _manifest_budget(tmp_path, stages, _metadata("rooms")) is None


def test_a_suite_rejects_a_metrics_timeout() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        Suite._parse_metrics({"timeout_s": 300})


def test_the_reader_files_each_robots_search_state_messages(tmp_path: pathlib.Path) -> None:
    string = ("std_msgs/msg/String", "string data")
    totals = {"stops": 4, "listened_s": 8.0, "goals_sent": 6, "goals_failed": 1, "calls": 2}
    records = [
        ("/env_0/jackal_0/hearing/search_state", {"data": json.dumps({"stops": 1})}, 2 * S),
        ("/env_0/jackal_0/hearing/search_state", {"data": json.dumps(totals)}, 5 * S),
        ("/env_0/burger_0/hearing/search_state", {"data": json.dumps({"stops": 9})}, 3 * S),
    ]
    bag = tmp_path / "run.mcap"
    with bag.open("wb") as fh:
        writer = Writer(fh)
        schema = writer.register_msgdef(*string)
        for topic, msg, log_time in records:
            writer.write_message(topic, schema, msg, log_time=log_time)
        writer.finish()

    bundles = MCAPReader(bag).read(tmp_path / "topics")

    jackal = bundles["env_0_jackal_0"].hearing_search_state.collect()
    assert jackal["time_ns"].to_list() == [2 * S, 5 * S]
    assert json.loads(jackal["state"][-1]) == totals
    assert json.loads(bundles["env_0_burger_0"].hearing_search_state.collect()["state"][0]) == {"stops": 9}
