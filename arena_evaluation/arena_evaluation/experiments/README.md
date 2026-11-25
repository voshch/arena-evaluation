# Paired paper studies over the canonical Star Schema

These modules are the reproduction code for the paper's study sections. They are not part of the
benchmark runner and they are not registered as report computations: a study fixes its scenarios, its
contrasts and its family of hypotheses, so it runs as its own command over a processed benchmark.

| Module | Study | Contrast | Entry point |
|---|---|---|---|
| `hhi.py` | A, two-person contact interactions next to the robot's path | `contact` vs `loco` (motion-matched control) | `python -m arena_evaluation.experiments.hhi {generate,analyze}` |
| `social.py` | A'/C, a planner that can read the interaction | `on` vs `off` (gesture + interaction ids + social cost layer) | `python -m arena_evaluation.experiments.social {generate,analyze}` |
| `group.py` | C, the robot crossing a standing conversation | route and clearance measurement about the formation | `python -m arena_evaluation.experiments.group measure` |
| `beckon.py` | B, a person beckons the robot over | walk-to vs ignore | `python -m arena_evaluation.experiments.beckon {generate,analyze}` |
| `usar.py` | USAR search, a driver that stops to listen | `usar_sweep` contestants vs the `usar_baseline` driver that never stops | `python -m arena_evaluation.experiments.usar analyze` |
| `blend_quality.py` | locomotion blend quality | replanned vs nominal trajectory | `python -m arena_evaluation.experiments.blend_quality` |

`python -m <module> --help` is the console entry of every study; there is no separate `arena` verb for
them because generating a study mutates the world tree and runs its own suites.

## How a study reads its data

Every study opens a processed benchmark directory through `source.py`:

- `episode_summary(benchmark_dir)`: the canonical `v_episode_summary` view, one row per robot episode.
- `contacts(benchmark_dir)`: `fact_contacts`, one row per contact window.
- `robot_tracks(benchmark_dir)`: `robot_tracks` telemetry, one lazy frame of pose samples.

Nothing reads `combined_metrics.parquet` or a legacy processed file: run `arena evaluation process
--benchmark-dir <run>` first, and the study sees exactly the numbers the catalogue publishes. An
episode the benchmark flagged `invalid_reason` keeps its row here; a study that wants valid episodes
only filters `v_episode_summary.is_valid_for_ranking`, and a metric that was never recorded stays
`NULL` with its null reason instead of becoming a zero.

## Paired inference

`stats.py` holds the shared machinery: seeds are fixed per scenario and shared by both arms, so a
difference is taken per `(scenario, seed)`; resampling is a cluster bootstrap over whole scenarios; the
interval is widened by `analytics.stats.contain_estimate` so it always contains its own estimate; and
the two-sided p-value is the shifted-null cluster-bootstrap test implemented once in
`analytics.stats.cluster_bootstrap_p_value`. Holm correction is applied within each study family.

## Generation vs analysis

`generate` writes the scenarios under `arena_simulation_setup/worlds/<map>/scenarios` and the paired
suite under `configs/suites/<name>/suite.yaml`; `analyze` reads one processed run per arm and writes
its tables as CSV next to the run (`<run>/hhi/`, or `--out`). Generation mutates the world tree, so it
is deliberately a separate, explicit command.

## Tests

`tests/unit/test_experiment_*.py` run each study against a synthetic benchmark written as real Parquet
(`tests/unit/star_benchmark.py`), so the analysis path, view construction, joins, paired inference,
is exercised without the simulator or the MCP stack.
