# benchmark

Runs multi-planner benchmark campaigns: spawns simulation environments, drives episodes, records results, and manages run state for pause/resume.

## Files

| File | Purpose |
|---|---|
| `runner.py` | `BenchmarkRunner`: spawns environments, executes episodes via `RunEpisode`, writes `progress.csv` and `.benchmark_state.json` |
| `config.py` | `Suite` and `Contest` parsers: YAML schema, sweep expansion, inline contest resolution |
| `state.py` | `Manifest`, `RunDir`, `StateFile`: run manifest, config hash, resume discovery, git SHA capture |
| `step.py` | `Step`, `StepResult`, `StepErrorKind`: grid model for contestant x stage combinations |
| `debug.py` | Process introspection: `running_processes()`, `tail_console()`, `console_log_path()` |
| `profiler.py` | `PipelineProfiler`: CPU, GPU, RAM, duration metrics |
## CLI (`arena evaluation`)

```bash
arena evaluation list                          # List all benchmark runs
arena evaluation status [<run_id>]             # Show run progress (--watch for live)
arena evaluation tail [<run_id>]               # tail -F progress.csv and every progress.<lane>.csv
arena evaluation ps                            # List running arena OS processes
arena evaluation kill [<pid>...] [-9]          # Terminate running arena/sim processes (-9 for SIGKILL)
arena evaluation console [<run_id>]            # Tail runner.log (--follow for streaming)
arena evaluation benchmark                     # Launch a benchmark campaign
```

## Execution Flow

The runner takes a suite (ordered stages) and a contest (planner lineup). It creates a step grid as the Cartesian product of contestants and stages, groups consecutive steps by `(contestant, robot, simulator)`, and spawns one environment per group.

For each group:
1. Calls `/arena/spawn_env` with launch arguments derived from the first step.
2. Waits for environment registration on `/arena/state/envs`.
3. For each step: calls `QueueEpisode` to set stage config, then executes `RunEpisode` action.
4. Despawns the environment and proceeds to the next group.

Run output structure in `$ARENA_DATA_DIR/benchmarks/<run_id>/`:
```
<run_id>/
|-- manifest.yaml              # Config snapshot
|-- progress.csv               # Append-only, one row per episode
|-- runner.log                 # Python logging and launch output
|-- .benchmark_state.json      # Per-step status
|-- episodes/                  # One MCAP per episode
|   |-- episode_000/
|   |   |-- episode_000.mcap
|   |   `-- episode_000.yaml
|   `-- ...
|-- combined_metrics.parquet   # Processed metrics
`-- report_manifest.yaml       # Used report manifest
```

## Liveness

Every wait the runner does on the runtime or an env is raced against three signals:

- the env disappearing from `/arena/state/envs` (`_EnvDied`),
- `/arena/state/sim` reporting `alive: false` (`_SimDied`, latched `arena_runtime_msgs/SimState`),
- the `arena_runtime.launch.py` subprocess exiting (`_SimDied`, reason `arena_runtime exited rc=N`).

Log-growth watching stays only as the fallback around spawn, for "nothing is dead but nothing moves".

`SimState` carries a wall-time `header.stamp`. The runner remembers when it started the current
`arena_runtime` and ignores any dead message stamped before that, so the latched death of the
previous runtime cannot kill the new one. `_restart_arena` clears the latch as well.

A step interrupted by a sim death is recorded `failed` with `error_kind=sim_dead`, which is
part of the systemic set. Exactly one worker restarts the runtime per death (an asyncio lock
plus a generation counter); the others wait for the new generation and then spawn a fresh env.

A cell that ends `failed` with a systemic kind (`sim_dead`, `env_setup`, `robot_setup`) is
retried up to `--retries` times (default 1, `0` never retries, `-1` retries forever). Every
retry runs on a fresh env: the current env is despawned, the cell goes back to the head of its
block queue, and a sim death recovers the runtime first. A retried attempt is not recorded, so
only a cell's final outcome reaches `progress.csv`, the state file and the progress display.
A cell that exhausts its retries is recorded failed and the queue moves on with a fresh env.
`--max-sim-deaths` (default 3) is the breaker that always stands: that many distinct sim deaths
abort the run (exit 1) whatever `--retries` allows. A cell that exhausts its retries while no
episode has run anywhere aborts the run as a setup failure.

A despawn whose service call or registry wait times out marks the sim dead with reason
`despawn timed out`: that is the inference rule for a hung sim that never raised a flag.
`_despawn_env` is a no-op once the sim is dead.

A spawn that stalls past `--spawn-budget` despawns not just the env it registered a response
for, but every env that showed up on `/arena/state/envs` during the wait and never went
`ready`: `arena_node` reserves an env before it activates, so a spawn that stalls before
`SpawnEnv` returns can still leave a live, unregistered env behind.

Episode budgets are sim time, enforced by the task generator through its `timeout` parameter
(a timed-out episode arrives as a normal FAILED `EpisodeRecord` with `outcome_info='timeout'`).
The runner keeps only a stall guard: while an episode result is pending it samples the sim
clock once a second and gives up after 60 wall seconds of a frozen clock outside a reset
(`<env_ns>/state/resetting`), cancelling the goal exactly like the old wall-clock timeout and
recording `episode_timeout` / `sim stalled 60s`.

The live display shows both clocks: the env table's `Sim / Wall` column counts the current
episode in sim seconds against wall seconds, so the ratio is the live rtf, and each completed
cell reports its wall time and the sim time its episodes consumed.

A watchdog beats the runner's own event loop. After 120 s of no beat it writes one line to
`runner.log`, SIGTERMs the arena launch process group and exits 4.

`--strict` exits 3 when any cell ended with an error kind or ran fewer episodes than
requested, whatever the episode outcomes were. It is the wedge check behind
`arena planners test --preflight`.

`--spawn-budget` (default 600s) caps how long a single env spawn is allowed to run before
the runner gives up, on top of the inactivity heuristic.

`--efficacy FRACTION` adds an outcome check: a non-SUCCESS episode is weak when the closed
fraction of its starting goal distance, `(goal_dist_start - goal_dist_min) / goal_dist_start`
from the `EpisodeRecord`, is below `FRACTION`. It catches a planner that drives badly without
ever erroring, which `--strict` cannot see. When set, the run ends with a
`benchmark: preflight verdict` table (contestant, stage, verdict, episodes run/total, weak
count, worst progress) and exits 3 if any cell is `wedged` (an error kind or an incomplete
episode count) or `weak` (at least one weak episode).

Exit codes: `0` ok, `1` systemic abort, `2` config error or crash, `3` lockstep, strict or
efficacy verdict, `4` runner hung (deadman), `130` interrupted.

## Programmatic Usage

```python
from arena_evaluation.benchmark.runner import BenchmarkRunner
from arena_evaluation.benchmark.config import Suite, Contest

suite = Suite.parse("basic", yaml.safe_load(suite_yaml))
contest = Contest.parse("basic", yaml.safe_load(contest_yaml))
runner = BenchmarkRunner(suite, contest, headless=True, env_n=4)
runner.run()
```

## Run State

`StateFile` manages `.benchmark_state.json` atomically. Each step has status: `ok | partial | failed | skipped | in_progress`. Interrupted runs resume with `--resume <run_id>`.

Run ID format: `{YYYYMMDD-HHMMSS}-{suite}-{contest}`, `{YYYYMMDD-HHMMSS}-{lane}-{suite}-{contest}` inside a lane, `{YYYYMMDD-HHMMSS}-{suite}-x{N}` for `--lanes N`.

## Shared Runs

`--lanes N` starts one runner per lane on the same run dir. Lanes claim env blocks so each runs once, and fold their files into `progress.csv` and `.benchmark_state.json`:

```
<run_id>/
|-- progress.<lane>.csv        # This lane's episodes
|-- .benchmark_state.<lane>.json
|-- runner.<lane>.log
|-- runner.<lane>.pid          # Signalled by --lanes on Ctrl-C
|-- claims/<token>/            # One file per (env, world) block, taken atomically
`-- episodes/.next_episode_id  # Episode id counter, locked across lanes
```

