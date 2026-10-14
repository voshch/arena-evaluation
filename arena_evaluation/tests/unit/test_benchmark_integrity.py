from __future__ import annotations

import csv
import json
import pathlib

import pytest
from task_generator_msgs.msg import EpisodeRecord

from arena_evaluation.benchmark.config import Contest, Suite
from arena_evaluation.benchmark.runner import compromised_table, episode_line
from arena_evaluation.benchmark.state import Manifest, ProgressLog, RunDir, StateFile, fold_run_dir
from arena_evaluation.benchmark.step import Step, StepResult
from task_generator.constants import Constants

MOVED = "env_0/jackal moved by hand"
SPAWNED = "robot jackal spawned by hand"
OLD_HEADER = ProgressLog._HEADER.removesuffix(",integrity")


def _record(*integrity: str, outcome_state: int = EpisodeRecord.SUCCESS, info: str = "") -> EpisodeRecord:
    msg = EpisodeRecord()
    msg.world = "map_empty"
    msg.robots = ["jackal"]
    msg.outcome_state = outcome_state
    msg.outcome_info = info
    msg.integrity = list(integrity)
    return msg


def _append(log: ProgressLog, ts: str, episode_id: int, rec: EpisodeRecord, step_key: str = "dwb/preflight") -> None:
    contestant, stage = step_key.split("/")
    log.append(ts_iso=ts, run_id="r", step_key=step_key, contestant=contestant, stage=stage, env_id=0, episode_id=episode_id, episode_record=rec, started_at=0.0, ended_at=1.0)


def _rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def _result(key: str, run: int, compromised: int) -> StepResult:
    return StepResult(key, "ok", 0, 0.0, 1.0, None, None, episodes_run=run, episodes_compromised=compromised)


def _step(contestant: str, stage: str) -> Step:
    return Step(
        contestant=Contest.Contestant(name=contestant, args={"mobile.local_planner": contestant}),
        stage=Suite.Stage(
            name=stage,
            episodes=2,
            robot="jackal",
            map="map_empty",
            tm_robots=Constants.TaskMode.TM_Robots.RANDOM,
            tm_obstacles=Constants.TaskMode.TM_Obstacles.RANDOM,
            config={},
            seed=0,
            timeout=60.0,
        ),
    )


def test_progress_csv_ends_with_the_integrity_column_as_a_json_list(tmp_path: pathlib.Path) -> None:
    log = ProgressLog(tmp_path / "progress.csv")
    _append(log, "2026-10-14T10:00:00+00:00", 0, _record())
    _append(log, "2026-10-14T10:00:01+00:00", 1, _record(SPAWNED, 'robot "x" moved, then retasked'))
    log.close()

    header = (tmp_path / "progress.csv").read_text().splitlines()[0].split(",")
    assert header[-1] == "integrity"
    rows = _rows(tmp_path / "progress.csv")
    assert [json.loads(r["integrity"]) for r in rows] == [[], [SPAWNED, 'robot "x" moved, then retasked']]


def test_progress_csv_with_an_older_header_is_upgraded_on_open(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "progress.csv"
    old_row = ["2026-10-13T10:00:00+00:00", "r", "dwb/preflight", "dwb", "preflight", "0", "0", *[""] * (len(OLD_HEADER.split(",")) - 7)]
    with path.open("w", newline="") as fh:
        fh.write(OLD_HEADER + "\n")
        csv.writer(fh).writerow(old_row)
        fh.write("2026-10-13T10:00:01+00:00,r,dwb/pre")

    log = ProgressLog(path)
    log.write_comment("resumed at 2026-10-14T10:00:00+00:00")
    _append(log, "2026-10-14T10:00:01+00:00", 1, _record(MOVED))
    log.dedupe_in_place()
    log.close()

    assert path.read_text().splitlines()[0] == ProgressLog._HEADER
    rows = _rows(path)
    assert [(r["episode_id"], r["integrity"]) for r in rows] == [("0", ""), ("1", json.dumps([MOVED]))]


def test_progress_csv_matching_the_current_header_is_left_alone(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "progress.csv"
    log = ProgressLog(path)
    _append(log, "2026-10-14T10:00:00+00:00", 0, _record(MOVED))
    log.write_comment("resumed at 2026-10-14T10:00:01+00:00")
    log.close()
    before = path.read_text()

    ProgressLog(path).close()

    assert path.read_text() == before


def test_fold_carries_integrity_of_lane_files_over_an_older_progress_csv(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "shared-run"
    run.mkdir()
    old_row = ["2026-10-13T10:00:00+00:00", "shared-run", "dwb/preflight", "dwb", "preflight", "0", "0", *[""] * (len(OLD_HEADER.split(",")) - 7)]
    with (run / "progress.csv").open("w", newline="") as fh:
        fh.write(OLD_HEADER + "\n")
        csv.writer(fh).writerow(old_row)
    lane = ProgressLog(run / "progress.p1.csv")
    _append(lane, "2026-10-14T10:00:00+00:00", 1, _record(SPAWNED))
    lane.close()

    fold_run_dir(run)

    assert (run / "progress.csv").read_text().splitlines()[0] == ProgressLog._HEADER
    rows = _rows(run / "progress.csv")
    assert [(r["episode_id"], r["integrity"]) for r in rows] == [("0", ""), ("1", json.dumps([SPAWNED]))]


def test_state_file_keeps_the_compromised_episode_count(tmp_path: pathlib.Path) -> None:
    run_dir = RunDir.create(tmp_path, "r", _manifest())
    run_dir.state.write({"dwb/preflight": _result("dwb/preflight", 2, 1)})
    assert StateFile.open(run_dir.path).steps["dwb/preflight"].episodes_compromised == 1

    legacy = json.loads((run_dir.path / ".benchmark_state.json").read_text())
    del legacy["steps"]["dwb/preflight"]["episodes_compromised"]
    (run_dir.path / ".benchmark_state.json").write_text(json.dumps(legacy))
    assert StateFile.open(run_dir.path).steps["dwb/preflight"].episodes_compromised == 0


def test_episode_line_appends_integrity_reasons_only_when_compromised() -> None:
    head = "[1/2] dwb/preflight env=0"
    intact = episode_line(head, _record(info="finished after 13.7s"), 13.7, 20.04)
    compromised = episode_line(head, _record(SPAWNED, MOVED, outcome_state=EpisodeRecord.FAILED, info="timeout"), 60.0, 70.0)

    assert intact == "[1/2] dwb/preflight env=0 SUCCESS info='finished after 13.7s' sim=13.7s wall=20.0s"
    assert compromised == f"[1/2] dwb/preflight env=0 FAILED info='timeout' sim=60.0s wall=70.0s integrity={[SPAWNED, MOVED]!r}"


def test_episode_line_keeps_goal_progress_before_integrity() -> None:
    rec = _record(MOVED)
    rec.goal_dist_start = 4.0
    rec.goal_dist_min = 1.0
    rec.path_length = 3.5
    assert episode_line("[2/2] dwb/preflight env=1", rec, 5.0, 6.0).endswith(f" progress=75% path=3.5m integrity={[MOVED]!r}")


def test_compromised_table_counts_per_contestant_across_stages() -> None:
    steps = [_step("dwb", "preflight"), _step("dwb", "crowd"), _step("teb", "preflight")]
    step_map = {s.key: s for s in steps}
    results = {
        "dwb/preflight": _result("dwb/preflight", 2, 1),
        "dwb/crowd": _result("dwb/crowd", 2, 2),
        "teb/preflight": _result("teb/preflight", 2, 0),
    }

    table = compromised_table(results, step_map)

    assert table is not None
    assert table.splitlines() == ["contestant  compromised  episodes", "dwb         3            4"]


def test_compromised_table_is_absent_when_every_episode_is_intact() -> None:
    results = {"dwb/preflight": _result("dwb/preflight", 2, 0)}
    assert compromised_table(results, {}) is None


def _manifest() -> Manifest:
    return Manifest(
        run_id="r",
        created_at="2026-10-14T10:00:00+00:00",
        arena_git_sha=None,
        arena_git_dirty=False,
        cli_args=[],
        env_n=1,
        headless=True,
        config_hash="0" * 40,
        simulator="gazebo",
        scale_episodes=1.0,
        suite_name="s",
        contest_name="c",
        suite={},
        contest=[],
        steps=[],
    )


@pytest.mark.parametrize("integrity", [[], [MOVED]])
def test_progress_csv_row_width_matches_the_header(tmp_path: pathlib.Path, integrity: list[str]) -> None:
    log = ProgressLog(tmp_path / "progress.csv")
    _append(log, "2026-10-14T10:00:00+00:00", 0, _record(*integrity))
    log.close()
    with (tmp_path / "progress.csv").open(newline="") as fh:
        header, row = list(csv.reader(fh))
    assert len(row) == len(header)
