"""The fact schema and the contract are complete in a fresh process, without importing the calculators first."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import pathlib
    import sys

    import polars as pl

    from arena_evaluation.storage.contracts import check_benchmark
    from arena_evaluation.storage.layout import write_benchmark_stamp, write_episode_stamp
    from arena_evaluation.storage.registry.facts import FactEpisodes, FactRobots

    root = pathlib.Path(sys.argv[1])
    metrics = root / "episodes" / "episode_000" / "metrics"
    metrics.mkdir(parents=True)
    pl.DataFrame(schema=FactEpisodes.polars_schema()).write_parquet(
        metrics / "fact_episodes.parquet"
    )
    write_episode_stamp(metrics)
    write_benchmark_stamp(root)
    report = check_benchmark(root)
    print(
        len(FactEpisodes.polars_schema().names()),
        len(FactRobots.polars_schema().names()),
        report.ok,
    )
    """
)


def test_a_fresh_process_sees_the_full_fact_schema_and_a_clean_contract(
    tmp_path: pathlib.Path,
) -> None:
    """Importing the contract alone must load the metric definitions, not depend on a metrics import first."""
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )

    episode_columns, robot_columns, ok = done.stdout.split()
    assert int(episode_columns) > 7
    assert int(robot_columns) > 6
    assert ok == "True"
