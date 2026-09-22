import pathlib

import yaml

from arena_evaluation.processing.pipeline import _suite_metrics


def _episode_dir(tmp_path: pathlib.Path, suite: dict | None) -> pathlib.Path:
    run = tmp_path / "20260911-000000-basic-inline"
    episode = run / "episodes" / "nav2" / "episode_000"
    episode.mkdir(parents=True)
    if suite is not None:
        (run / "manifest.yaml").write_text(yaml.safe_dump({"run_id": run.name, "suite": suite}))
    return episode


def test_suite_metrics_from_manifest(tmp_path: pathlib.Path) -> None:
    episode = _episode_dir(tmp_path, {"metrics": {"max_collisions": 1}})
    assert _suite_metrics(episode, tmp_path) == {"max_collisions": 1}


def test_suite_metrics_absent_block(tmp_path: pathlib.Path) -> None:
    episode = _episode_dir(tmp_path, {"stages": []})
    assert _suite_metrics(episode, tmp_path) == {}


def test_suite_metrics_no_manifest(tmp_path: pathlib.Path) -> None:
    episode = _episode_dir(tmp_path, None)
    assert _suite_metrics(episode, tmp_path) == {}


def test_suite_metrics_junk_value_ignored(tmp_path: pathlib.Path) -> None:
    episode = _episode_dir(tmp_path, {"metrics": {"max_collisions": "many"}})
    assert _suite_metrics(episode, tmp_path) == {}
