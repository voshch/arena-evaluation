import logging
import math
import os
import pathlib
import uuid
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest
import yaml

from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations
from arena_evaluation.processing.metrics.base import MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.ecological.acoustic_exposure import (
    AcousticExposureCalculator,
)
from arena_evaluation.processing.metrics.ecological.acoustics import (
    ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2,
)
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.schemas import (
    AlignedEpisodeBundle,
    EpisodeDescriptor,
    RobotParams,
)


def test_acoustics_free_space():
    grid = np.zeros((10, 10), dtype=np.uint8)
    att = compute_attenuations(
        grid,
        1.0,
        0.0,
        0.0,
        np.array([3.0], dtype=np.float32),
        np.array([4.0], dtype=np.float32),
        wall_tl=47.0,
        mic_distance=1.0,
    )
    assert np.isclose(att[0], 20.0 * np.log10(5.0), atol=0.1)


def test_acoustics_one_wall():
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[:, 2] = 255
    att = compute_attenuations(
        grid,
        1.0,
        0.0,
        0.0,
        np.array([3.0], dtype=np.float32),
        np.array([4.0], dtype=np.float32),
        wall_tl=47.0,
        mic_distance=1.0,
    )
    assert np.isclose(att[0], 20.0 * np.log10(5.0) + 47.0, atol=0.5)


def test_acoustics_pruning_dominance():
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[0:7, 2] = 255
    att = compute_attenuations(
        grid,
        1.0,
        0.0,
        3.0,
        np.array([4.0], dtype=np.float32),
        np.array([3.0], dtype=np.float32),
        wall_tl=47.0,
        mic_distance=1.0,
    )
    assert att[0] < 40.0


def _field(grid, resolution, sx, sy, mic_distance, wall_tl=47.0):
    h, w = grid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    tx = np.ascontiguousarray(xx.ravel().astype(np.float32))
    ty = np.ascontiguousarray(yy.ravel().astype(np.float32))
    return compute_attenuations(
        grid, resolution, sx, sy, tx, ty, wall_tl=wall_tl, mic_distance=mic_distance
    ).reshape(h, w)


def test_receiver_inside_mic_distance_costs_the_floor():
    att = compute_attenuations(
        np.zeros((40, 40), dtype=np.uint8),
        0.1,
        20.0,
        20.0,
        np.array([25.0, 23.0, 20.0], dtype=np.float32),
        np.array([20.0] * 3, dtype=np.float32),
        wall_tl=47.0,
        mic_distance=2.0,
    )
    assert np.allclose(att, 20.0 * np.log10(2.0), atol=1e-4)


def test_wall_inside_mic_distance_adds_only_its_tl():
    grid = np.zeros((40, 40), dtype=np.uint8)
    grid[:, 20] = 1
    att = compute_attenuations(
        grid,
        0.1,
        17.0,
        20.0,
        np.array([23.0], dtype=np.float32),
        np.array([20.0], dtype=np.float32),
        wall_tl=47.0,
        mic_distance=1.0,
    )
    assert np.isclose(att[0], 20.0 * np.log10(1.0) + 47.0, atol=1e-4)


def test_tied_costs_inside_floor_keep_shortest_paths_beyond_it():
    grid = np.zeros((40, 40), dtype=np.uint8)
    grid[18:23, 21] = 1
    floored = _field(grid, 0.1, 20.0, 20.0, mic_distance=1.0, wall_tl=1000.0)
    reference = _field(grid, 0.1, 20.0, 20.0, mic_distance=1e-3, wall_tl=1000.0)
    beyond = reference > 20.0 * np.log10(1.05)
    beyond &= reference < 100.0
    assert beyond.sum() > 1000
    assert np.abs(floored - reference)[beyond].max() < 0.25


SECOND_NS = 1_000_000_000
NULLS = dict.fromkeys(
    AcousticExposureCalculator.output_keys_for_scope(MetricScope.PEDESTRIAN)
)


def _att(distance_m: float) -> float:
    return 20.0 * math.log10(max(distance_m, 1.0))


def _leq(values: list[float]) -> float:
    return 10.0 * math.log10(
        sum(10.0 ** (value / 10.0) for value in values) / len(values)
    )


@pytest.fixture
def calc() -> AcousticExposureCalculator:
    return AcousticExposureCalculator(RobotParams())


@pytest.fixture
def world_run_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "run"


@pytest.fixture
def derived_map(world_run_dir: pathlib.Path) -> Callable[..., str]:
    """Write a real world.yaml under world_run_dir / worlds / <name>; the metric loads it by name."""

    def make(*, walled: bool = False, with_door: bool = False) -> str:
        name = f"acx-{uuid.uuid4().hex}"
        world_dir = world_run_dir / "worlds" / name
        world_dir.mkdir(parents=True, exist_ok=True)
        level: dict[str, object] = {
            "id": "0",
            "zones": [
                {
                    "name": "room",
                    "polygon": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
                }
            ],
            "doors": [],
        }
        if walled:
            level["walls"] = [
                {"path": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]]},
                {"path": [[2.5, 0.0], [2.5, 5.0]]},
            ]
        if with_door:
            level["doors"] = [
                {
                    "name": "front_door",
                    "span": [[2.5, 2.0], [2.5, 3.0]],
                    "width": 0.1,
                    "kind": "sliding",
                }
            ]
        (world_dir / "world.yaml").write_text(
            yaml.safe_dump({"format": "arena-world/2", "levels": [level]})
        )
        return name

    return make


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def module_log():
    logger = logging.getLogger(
        "arena_evaluation.processing.metrics.ecological.acoustic_exposure"
    )
    handler = _Records()
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler.messages
    logger.removeHandler(handler)
    logger.setLevel(level)


def _door_run(
    tmp_path: pathlib.Path, map_name: str
) -> tuple[FolderManager, EpisodeDescriptor]:
    world_dir = tmp_path / "bench" / "worlds" / map_name
    world_dir.mkdir(parents=True, exist_ok=True)
    door = {
        "name": "front_door",
        "span": [[2.5, 2.0], [2.5, 3.0]],
        "width": 0.1,
        "kind": "sliding",
    }
    level = {
        "id": "0",
        "zones": [
            {
                "name": "room",
                "polygon": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
            }
        ],
        "walls": [
            {"path": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]]},
            {"path": [[2.5, 0.0], [2.5, 5.0]]},
        ],
        "doors": [door],
    }
    (world_dir / "world.yaml").write_text(
        yaml.safe_dump({"format": "arena-world/2", "levels": [level]})
    )
    run = EpisodeDescriptor(
        episode_dir=str(tmp_path / "bench" / "ep"),
        benchmark_id="bench",
        episode_id=7,
        planner="p",
        stage="s",
        map=map_name,
    )
    return FolderManager(tmp_path), run


def _semantic(rows: list[tuple[int, str, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": [row[0] for row in rows],
            "kind": ["door"] * len(rows),
            "entity_name": [row[1] for row in rows],
            "field_name": ["progress"] * len(rows),
            "value_bool": [None] * len(rows),
            "value_str": [None] * len(rows),
            "value_num": [1.0 if row[2] else 0.0 for row in rows],
        },
        schema={
            "t_ns": pl.Int64,
            "kind": pl.Utf8,
            "entity_name": pl.Utf8,
            "field_name": pl.Utf8,
            "value_bool": pl.Boolean,
            "value_str": pl.Utf8,
            "value_num": pl.Float64,
        },
    )


def _frames(
    robot: list[tuple[float | None, float | None]],
    peds: list[list[tuple[float, float]]],
    *,
    source: list[float | None] | None = None,
    times: list[int] | None = None,
    omit_acoustics: bool = False,
) -> pl.DataFrame:
    columns = {
        "t_ns": pl.Series(
            times
            if times is not None
            else [index * SECOND_NS for index in range(len(robot))],
            dtype=pl.Int64,
        ),
        "pos_x_gt": pl.Series([position[0] for position in robot], dtype=pl.Float64),
        "pos_y_gt": pl.Series([position[1] for position in robot], dtype=pl.Float64),
        "peds_positions": pl.Series(
            [[list(point) for point in frame] for frame in peds],
            dtype=pl.List(pl.List(pl.Float64)),
        ),
    }
    if not omit_acoustics:
        levels = source if source is not None else [42.0] * len(robot)
        columns["total_level_af_dba"] = pl.Series(levels, dtype=pl.Float64)
    return pl.DataFrame(columns)


def _episode(
    data: pl.DataFrame | None, map_name: str | None, **kwargs
) -> AlignedEpisodeBundle:
    topics = dict(kwargs.pop("topics", {}) or {})
    if (
        data is not None
        and "t_ns" in data.columns
        and "pedestrian_tracks" not in topics
        and "peds_positions" in data.columns
    ):
        peds_rows = []
        seen = set()
        for stamp, frame in zip(
            data["t_ns"].to_list(), data["peds_positions"].to_list()
        ):
            if frame:
                for index, point in enumerate(frame):
                    key = (int(stamp), f"ped_{index}")
                    if (
                        key not in seen
                        and len(point) >= 2
                        and not math.isnan(point[0])
                        and not math.isnan(point[1])
                    ):
                        seen.add(key)
                        peds_rows.append(
                            {
                                "t_ns": int(stamp),
                                "ped_id": f"ped_{index}",
                                "x": float(point[0]),
                                "y": float(point[1]),
                            }
                        )
        schema = {"t_ns": pl.Int64, "ped_id": pl.Utf8, "x": pl.Float64, "y": pl.Float64}
        topics["pedestrian_tracks"] = (
            pl.DataFrame(peds_rows, schema=schema)
            if peds_rows
            else pl.DataFrame(schema=schema)
        )
    if (
        data is not None
        and "total_level_af_dba" in data.columns
        and "robot_acoustics" not in topics
    ):
        topics["robot_acoustics"] = pl.DataFrame(
            {
                "t_ns": data["t_ns"].cast(pl.Int64),
                "total_level_af_dba": data["total_level_af_dba"].cast(pl.Float64),
            }
        )
    if data is not None and "t_ns" in data.columns and "robot_tracks" not in topics:
        topics["robot_tracks"] = (
            pl.DataFrame(
                {
                    "t_ns": data["t_ns"].cast(pl.Int64),
                    "x": data["pos_x_gt"].cast(pl.Float64),
                    "y": data["pos_y_gt"].cast(pl.Float64),
                    "yaw": [0.0] * len(data),
                }
            )
            .drop_nulls(["x", "y"])
            .sort("t_ns")
        )
    return AlignedEpisodeBundle(
        episode_id=7,
        data=data,
        start_pos=[0.0, 0.0],
        goal_pos=[0.0, 0.0],
        map=map_name,
        topics=topics,
        **kwargs,
    )


def _series_by_time(
    calc: AcousticExposureCalculator, episode: AlignedEpisodeBundle
) -> dict[int, list[tuple[str, float, float]]]:
    table = calc.series_table(episode)
    frames: dict[int, list[tuple[str, float, float]]] = {}
    for t_ns, ped_id, exposure, attenuation in zip(
        table["t_ns"],
        table["ped_id"],
        table["exposure_dba"],
        table["attenuation_db"],
        strict=True,
    ):
        frames.setdefault(int(t_ns), []).append(
            (str(ped_id), float(exposure), float(attenuation))
        )
    return frames


def test_metric_raises_when_no_barrier_map_exists(calc, world_run_dir):
    with pytest.raises(MetricInputError, match="unknown"):
        calc.calculate(
            _episode(
                _frames([(1.0, 1.0)], [[(4.0, 1.0)]]), "unknown", run_dir=world_run_dir
            ),
            {},
        )


def test_reference_run_is_evaluated_like_any_other_episode(derived_map, world_run_dir):
    """Reference twins carry the same acoustic telemetry, so they no longer skip the exposure solve."""
    map_name = derived_map()
    run = EpisodeDescriptor(
        episode_dir="ep",
        benchmark_id="bench",
        episode_id=7,
        planner="p",
        stage="s",
        map=map_name,
        is_reference=True,
        reference_type="unhindered_peds",
    )
    episode = _episode(
        _frames([(1.0, 1.0)], [[(4.0, 1.0)]]), map_name, run=run, run_dir=world_run_dir
    )
    reference_calc = AcousticExposureCalculator(RobotParams())
    result = reference_calc.calculate(episode, {})
    assert result["ped_max_exposure_dba"] == pytest.approx(42.0 - _att(3.0), abs=1e-3)
    assert reference_calc.series_table(episode).height > 0


@pytest.mark.parametrize("data", [None, pl.DataFrame()], ids=["none", "empty"])
def test_missing_episode_data_yields_nulls(data, calc, derived_map, world_run_dir):
    assert (
        calc.calculate(_episode(data, derived_map(), run_dir=world_run_dir), {})
        == NULLS
    )


def test_missing_map_name_raises_metric_input_error(calc):
    with pytest.raises(MetricInputError, match="missing world map name"):
        calc.calculate(_episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), None), {})


def test_world_fallback_supplies_the_map(calc, derived_map, world_run_dir):
    calc.world = derived_map()
    result = calc.calculate(
        _episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), None, run_dir=world_run_dir), {}
    )
    assert result["ped_max_exposure_dba"] == pytest.approx(42.0 - _att(3.0), abs=1e-3)


def test_prior_map_overrides_episode_map(calc, derived_map, world_run_dir):
    result = calc.calculate(
        _episode(
            _frames([(1.0, 1.0)], [[(4.0, 1.0)]]), "unknown", run_dir=world_run_dir
        ),
        {"map": derived_map()},
    )
    assert result["ped_max_exposure_dba"] == pytest.approx(42.0 - _att(3.0), abs=1e-3)


def test_series_collapses_duplicate_clock_samples(calc, derived_map, world_run_dir):
    """The published natural key (t_ns, ped_id) is unique even when the recorded clock repeats a stamp."""
    repeated = _frames(
        [(1.0, 1.0), (1.0, 1.0)], [[(4.0, 1.0)], [(4.0, 1.0)]], times=[0, 0]
    )
    table = AcousticExposureCalculator(RobotParams()).series_table(
        _episode(repeated, derived_map(), run_dir=world_run_dir)
    )
    assert table.height == 1
    assert table.select("t_ns", "ped_id").unique().height == table.height


def test_scalars_follow_the_exposure_series(calc, derived_map, world_run_dir):
    frames = _frames(
        [(1.0, 1.0)] * 4,
        [
            [(4.0, 1.0)],
            [(4.0, 1.0)],
            [(4.0, 1.0), (1.0, 4.0)],
            [(4.0, 1.0), (1.0, 4.0)],
        ],
        source=[42.0, 60.0, 60.0, 60.0],
    )
    episode = _episode(frames, derived_map(), run_dir=world_run_dir)
    result = calc.calculate(episode, {})
    table = calc.series_table(episode, run_id="run", robot_id="robot")
    assert table.schema == pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "t_ns": pl.Int64,
            "ped_id": pl.String,
            "exposure_dba": pl.Float64,
            "attenuation_db": pl.Float64,
            "robot_x_m": pl.Float64,
            "robot_y_m": pl.Float64,
            "source_dba": pl.Float64,
            "ped_x_m": pl.Float64,
            "ped_y_m": pl.Float64,
            "is_worst_frame": pl.Boolean,
            "open_doors": pl.String,
        }
    )
    values = table["exposure_dba"].drop_nulls()
    assert result["ped_max_exposure_dba"] == pytest.approx(float(values.max()))
    assert result["ped_leq_exposure_dba"] == pytest.approx(_leq(values.to_list()))
    # One row per (sample, reachable pedestrian).
    assert len(table) == 6
    assert result["ped_max_startle_factor"] == pytest.approx(18.0, abs=1e-3)
    assert result["worst_case_acoustic_t_ns"] == SECOND_NS
    worst_rows = table.filter(pl.col("is_worst_frame"))
    assert len(worst_rows) == 1
    assert int(worst_rows["t_ns"][0]) == SECOND_NS
    assert float(worst_rows["robot_x_m"][0]) == 1.0
    assert float(worst_rows["ped_x_m"][0]) == 4.0
    assert worst_rows["open_doors"][0] == ""


def test_frames_without_ground_truth_produce_no_exposure(
    calc, derived_map, world_run_dir
):
    frames = _frames(
        [(None, None), (0.5, 0.5), (0.5, 0.5)], [[], [(3.5, 4.5)], [(3.5, 4.5)]]
    )
    episode = _episode(frames, derived_map(), run_dir=world_run_dir)
    result = calc.calculate(episode, {})
    expected = 42.0 - _att(5.0)
    assert result["ped_max_exposure_dba"] == pytest.approx(expected, abs=1e-3)
    assert result["ped_leq_exposure_dba"] == pytest.approx(expected, abs=1e-3)
    assert result["ped_max_startle_factor"] == 0.0
    assert result["worst_case_acoustic_t_ns"] == SECOND_NS
    table = calc.series_table(episode)
    assert table["t_ns"].unique().to_list() == [SECOND_NS, 2 * SECOND_NS]


def test_repeated_timestamp_contributes_no_startle(calc, derived_map, world_run_dir):
    frames = _frames(
        [(1.0, 2.0), (1.0, 1.0)], [[(4.0, 1.0)], [(4.0, 1.0)]], times=[0, 0]
    )
    result = calc.calculate(_episode(frames, derived_map(), run_dir=world_run_dir), {})
    assert result["ped_max_startle_factor"] is None
    assert result["ped_max_exposure_dba"] == pytest.approx(42.0 - _att(3.0), abs=1e-3)


def test_unreachable_pedestrians_are_dropped(calc, derived_map, world_run_dir):
    frames = _frames([(1.0, 1.0), (1.0, 1.0)], [[(4.0, 1.0), (9.0, 9.0)], [(9.0, 9.0)]])
    episode = _episode(frames, derived_map(), run_dir=world_run_dir)
    result = calc.calculate(episode, {})
    assert result["ped_max_exposure_dba"] == pytest.approx(42.0 - _att(3.0), abs=1e-3)
    assert calc.series_table(episode)["ped_id"].unique().to_list() == ["ped_0"]


def test_robot_off_map_yields_nulls(calc, derived_map, world_run_dir):
    result = calc.calculate(
        _episode(
            _frames([(9.0, 9.0), (9.0, 9.0)], [[(4.0, 1.0)], []]),
            derived_map(),
            run_dir=world_run_dir,
        ),
        {},
    )
    assert result == NULLS


def test_static_scene_caches_episode_frames_across_fifty_samples(
    calc, derived_map, world_run_dir
):
    episode = _episode(
        _frames([(1.0, 1.0)] * 50, [[(4.0, 1.0)]] * 50),
        derived_map(),
        run_dir=world_run_dir,
    )
    result = calc.calculate(episode, {})
    assert result["ped_max_startle_factor"] == 0.0
    assert len(calc.series_table(episode)) == 50
    assert len(calc._episode_frame_cache) == 1


def test_door_opening_switches_attenuation_and_marks_worst_frame_open(tmp_path, calc):
    name = f"acx-{uuid.uuid4().hex}"
    folder_manager, run = _door_run(tmp_path, name)
    frames = _frames([(1.0, 2.5)] * 3, [[(4.0, 2.5)]] * 3, source=[60.0] * 3)
    semantic = _semantic([(SECOND_NS, "env_0/front_door/0", True)])
    episode = _episode(
        frames, name, run=run, folder_manager=folder_manager, semantic_snapshot=semantic
    )
    result = calc.calculate(episode, {})
    closed, opened = _att(3.0) + 25.0, _att(3.0)
    attenuation_by_time = _series_by_time(calc, episode)
    assert [values[0][2] for values in attenuation_by_time.values()] == pytest.approx(
        [closed, opened, opened], abs=0.5
    )
    assert result["ped_max_exposure_dba"] == pytest.approx(60.0 - opened, abs=0.5)
    assert result["ped_leq_exposure_dba"] == pytest.approx(
        _leq([60.0 - closed, 60.0 - opened, 60.0 - opened]), abs=0.5
    )
    assert result["ped_max_startle_factor"] == pytest.approx(25.0, abs=0.5)
    assert result["worst_case_acoustic_t_ns"] == SECOND_NS
    worst_rows = calc.series_table(episode).filter(pl.col("is_worst_frame"))
    assert worst_rows["open_doors"][0] == "world/front_door"


def test_the_pipeline_run_dir_reads_the_doors_of_the_run_snapshot(tmp_path, calc):
    name = f"acx-{uuid.uuid4().hex}"
    _door_run(tmp_path, name)
    semantic = _semantic([(0, "env_0/front_door/0", False)])
    episode = _episode(
        _frames([(1.0, 2.5)], [[(4.0, 2.5)]], source=[60.0]),
        name,
        run_dir=tmp_path / "bench",
        semantic_snapshot=semantic,
    )
    calc.calculate(episode, {})
    table = calc.series_table(episode)
    assert table["open_doors"][0] == ""
    assert table["attenuation_db"][0] == pytest.approx(_att(3.0) + 25.0, abs=0.5)


def test_doors_without_semantic_timeline_return_null(tmp_path, calc):
    name = f"acx-{uuid.uuid4().hex}"
    folder_manager, run = _door_run(tmp_path, name)
    frames = _frames([(1.0, 2.5)], [[(4.0, 2.5)]], source=[60.0])
    episode = _episode(frames, name, run=run, folder_manager=folder_manager)
    assert calc.calculate(episode, {}) == NULLS
    assert calc.series_table(episode).is_empty()


def test_semantic_timeline_for_another_door_keeps_door_closed(tmp_path, calc):
    name = f"acx-{uuid.uuid4().hex}"
    folder_manager, run = _door_run(tmp_path, name)
    semantic = _semantic([(0, "env_0/back_door/0", True)])
    frames = _frames([(1.0, 2.5)], [[(4.0, 2.5)]], source=[60.0])
    episode = _episode(
        frames, name, run=run, folder_manager=folder_manager, semantic_snapshot=semantic
    )
    calc.calculate(episode, {})
    table = calc.series_table(episode)
    assert table["open_doors"][0] == ""
    assert table["attenuation_db"][0] == pytest.approx(_att(3.0) + 25.0, abs=0.5)


def test_acoustic_exposure_registers_the_pedestrian_and_encounter_scopes():
    assert AcousticExposureCalculator.effective_scopes() == frozenset(
        {MetricScope.PEDESTRIAN, MetricScope.ENCOUNTER}
    )
    assert AcousticExposureCalculator.output_keys_for_scope(MetricScope.PEDESTRIAN) == (
        "ped_max_exposure_dba",
        "ped_leq_exposure_dba",
        "ped_max_startle_factor",
        "worst_case_acoustic_t_ns",
    )
    assert AcousticExposureCalculator.output_keys_for_scope(MetricScope.ENCOUNTER) == (
        "sound_exposure_dose_pa2s",
    )


def test_pedestrian_scope_reduces_to_that_pedestrian(calc, derived_map, world_run_dir):
    map_name = derived_map()
    frames = _frames(
        [(1.0, 1.0), (1.0, 1.0)],
        [[(4.0, 1.0), (4.0, 4.0)], [(4.0, 1.0), (4.0, 4.0)]],
        source=[60.0, 60.0],
    )
    episode = _episode(frames, map_name, run_dir=world_run_dir)
    full = calc.calculate(episode, {})

    def pedestrian_result(ped_id: str) -> dict[str, object]:
        ped_topics = dict(episode.topics)
        ped_topics["pedestrian_tracks"] = episode.topics["pedestrian_tracks"].filter(
            pl.col("ped_id") == ped_id
        )
        bundle = AlignedEpisodeBundle(
            episode_id=episode.episode_id,
            data=pl.DataFrame(),
            start_pos=episode.start_pos,
            goal_pos=episode.goal_pos,
            map=episode.map,
            run_dir=world_run_dir,
            topics=ped_topics,
        )
        scope_slice = ScopeSlice(
            scope=MetricScope.PEDESTRIAN,
            slice_id=ped_id,
            t_start_ns=0,
            t_end_ns=SECOND_NS,
            identity={"ped_id": ped_id},
            window_bundle=bundle,
            full_episode=episode,
        )
        return calc.calculate_for_scope(bundle, {}, scope_slice)

    near = pedestrian_result("ped_0")
    far = pedestrian_result("ped_1")
    assert near["ped_max_exposure_dba"] == pytest.approx(60.0 - _att(3.0), abs=1e-3)
    assert far["ped_max_exposure_dba"] == pytest.approx(
        60.0 - 20.0 * math.log10(math.sqrt(18.0)), abs=1e-2
    )
    assert near["ped_max_exposure_dba"] == pytest.approx(
        full["ped_max_exposure_dba"], abs=1e-3
    )
    # The pedestrian scope publishes the worst frame's t_ns discriminator, never the nested struct.
    assert set(near) == set(
        AcousticExposureCalculator.output_keys_for_scope(MetricScope.PEDESTRIAN)
    )
    series = calc.series_table(episode)
    near_rows = series.filter(pl.col("ped_id") == "ped_0")
    near_max = near_rows["exposure_dba"].max()
    expected_t_ns = int(
        near_rows.filter(pl.col("exposure_dba") == near_max).sort("t_ns")["t_ns"][0]
    )
    assert near["worst_case_acoustic_t_ns"] == expected_t_ns


def _pedestrian_scope_slice(episode: AlignedEpisodeBundle, ped_id: str) -> ScopeSlice:
    """A PEDESTRIAN slice whose window bundle carries only that pedestrian's track."""
    topics = dict(episode.topics or {})
    topics["pedestrian_tracks"] = topics["pedestrian_tracks"].filter(
        pl.col("ped_id") == ped_id
    )
    bundle = AlignedEpisodeBundle(
        episode_id=episode.episode_id,
        data=pl.DataFrame(),
        start_pos=episode.start_pos,
        goal_pos=episode.goal_pos,
        map=episode.map,
        topics=topics,
    )
    return ScopeSlice(
        scope=MetricScope.PEDESTRIAN,
        slice_id=ped_id,
        t_start_ns=0,
        t_end_ns=SECOND_NS,
        identity={"ped_id": ped_id},
        window_bundle=bundle,
        full_episode=episode,
    )


def test_pedestrian_scope_publishes_the_worst_frame_discriminator(
    calc, derived_map, world_run_dir
):
    frames = _frames([(1.0, 1.0)] * 3, [[(4.0, 1.0)]] * 3, source=[40.0, 60.0, 60.0])
    episode = _episode(frames, derived_map(), run_dir=world_run_dir)
    pedestrian_slice = _pedestrian_scope_slice(episode, "ped_0")
    result = calc.calculate_for_scope(
        pedestrian_slice.window_bundle, {}, pedestrian_slice
    )

    assert set(result) == set(
        AcousticExposureCalculator.output_keys_for_scope(MetricScope.PEDESTRIAN)
    )
    assert "worst_case_acoustic_frame" not in result
    assert result["worst_case_acoustic_t_ns"] == SECOND_NS
    series = calc.series_table(episode)
    worst_rows = series.filter((pl.col("ped_id") == "ped_0") & pl.col("is_worst_frame"))
    assert worst_rows["t_ns"].to_list() == [SECOND_NS]


def test_one_acoustic_solve_serves_the_episode_and_every_pedestrian(
    calc, derived_map, world_run_dir
):
    """The expensive Dijkstra pass runs once per episode; the pedestrian rollups reduce the memoised series."""
    frames = _frames(
        [(1.0, 1.0)] * 3, [[(4.0, 1.0), (2.0, 4.0)]] * 3, source=[40.0, 60.0, 60.0]
    )
    episode = _episode(frames, derived_map(), run_dir=world_run_dir)
    calc.calculate(episode, {})
    for ped_id in ("ped_0", "ped_1"):
        pedestrian_slice = _pedestrian_scope_slice(episode, ped_id)
        calc.calculate_for_scope(pedestrian_slice.window_bundle, {}, pedestrian_slice)
    assert len(calc._episode_frame_cache) == 1


def test_missing_robot_acoustics_topic_returns_null_without_fallback(
    calc, derived_map, world_run_dir
):
    frames = _frames(
        [(1.0, 1.0), (1.0, 1.0)], [[(4.0, 1.0)], [(4.0, 1.0)]], omit_acoustics=True
    )
    episode = _episode(frames, derived_map(), run_dir=world_run_dir)
    assert calc.calculate(episode, {}) == NULLS

    registry = MetricRegistry(RobotParams(), calculators=[AcousticExposureCalculator])
    ped_slice = _pedestrian_scope_slice(episode, "ped_0")
    outcome = registry.run_scope(MetricScope.PEDESTRIAN, episode, scope_slice=ped_slice)
    assert outcome.results["ped_max_exposure_dba"] is None
    assert (
        outcome.provenance["ped_max_exposure_dba"]
        == "ped_max_exposure_dba:topic_missing:robot_acoustics"
    )


def test_missing_solver_library_raises_metric_input_error(
    calc, derived_map, world_run_dir
):
    from arena_evaluation.processing.acoustics import impedance_grid

    episode = _episode(
        _frames([(1.0, 1.0)], [[(4.0, 1.0)]]), derived_map(), run_dir=world_run_dir
    )
    saved = os.environ.get("ARENA_ACOUSTIC_SOLVER_SO")
    try:
        os.environ["ARENA_ACOUSTIC_SOLVER_SO"] = str(
            world_run_dir / "nonexistent_solver.so"
        )
        impedance_grid._solver_library.cache_clear()
        with pytest.raises(MetricInputError, match="arena build"):
            calc.calculate(episode, {})
    finally:
        if saved is None:
            os.environ.pop("ARENA_ACOUSTIC_SOLVER_SO", None)
        else:
            os.environ["ARENA_ACOUSTIC_SOLVER_SO"] = saved
        impedance_grid._solver_library.cache_clear()


def test_encounter_scope_golden_wall_dose_and_non_pedestrian_not_applicable(
    calc, derived_map, world_run_dir
):
    frames = _frames([(1.0, 2.5)] * 3, [[(4.0, 2.5)]] * 3, source=[70.0, 70.0, 70.0])
    walled_ep = _episode(frames, derived_map(walled=True), run_dir=world_run_dir)
    free_ep = _episode(frames, derived_map(walled=False), run_dir=world_run_dir)

    walled_slice = ScopeSlice(
        scope=MetricScope.ENCOUNTER,
        slice_id="enc_0",
        t_start_ns=0,
        t_end_ns=2 * SECOND_NS,
        identity={"actor_1": "robot", "actor_2": "ped_0", "actor_2_kind": "pedestrian"},
        window_bundle=walled_ep,
        full_episode=walled_ep,
    )
    free_slice = ScopeSlice(
        scope=MetricScope.ENCOUNTER,
        slice_id="enc_1",
        t_start_ns=0,
        t_end_ns=2 * SECOND_NS,
        identity={"actor_1": "robot", "actor_2": "ped_0", "actor_2_kind": "pedestrian"},
        window_bundle=free_ep,
        full_episode=free_ep,
    )
    walled_dose = calc.calculate_for_scope(walled_ep, {}, walled_slice)[
        "sound_exposure_dose_pa2s"
    ]
    free_dose = calc.calculate_for_scope(free_ep, {}, free_slice)[
        "sound_exposure_dose_pa2s"
    ]

    expected_exposure = 70.0 - (_att(3.0) + 47.0)
    expected_dose = (
        ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2 * (10.0 ** (expected_exposure / 10.0)) * 2.0
    )
    assert walled_dose == pytest.approx(expected_dose, rel=0.15)
    assert walled_dose < free_dose * 1e-3

    non_ped_slice = ScopeSlice(
        scope=MetricScope.ENCOUNTER,
        slice_id="enc_2",
        t_start_ns=0,
        t_end_ns=2 * SECOND_NS,
        identity={"actor_1": "robot", "actor_2": "robot_1", "actor_2_kind": "robot"},
        window_bundle=walled_ep,
        full_episode=walled_ep,
    )
    assert calc.calculate_for_scope(walled_ep, {}, non_ped_slice) == {
        "sound_exposure_dose_pa2s": None,
        "sound_exposure_dose_pa2s_null_reason": "not_applicable",
    }
