# benchmark

Runs multi-planner benchmark campaigns: spawns simulation environments, drives episodes, records results, and manages run state for pause/resume.

## Files

| File | Purpose |
|---|---|
| `runner.py` | `BenchmarkRunner`: spawns environments, executes episodes via `RunEpisode`, writes `progress.csv` and `.benchmark_state.json` |
| `config.py` | `Suite` and `Contest` parsers: YAML schema, the suite's `task_layer`, stage `modes` per axis, sweep expansion, inline contest resolution |
| `state.py` | `Manifest`, `RunDir`, `StateFile`: run manifest, config hash, resume discovery, git SHA capture |
| `step.py` | `Step`, `StepResult`, `StepErrorKind`: grid model for contestant x stage combinations |
| `debug.py` | Process introspection: `running_processes()`, `tail_console()`, `console_log_path()` |
| `dashboard.py`, `dashboard.html` | Live web dashboard (`arena evaluation serve`, `--viz`): `run_status`, `run_detail`, the HTTP server and its page |
| `dashboard_runtime.py` | The dashboard's runtime page (`serve --runtime`): MCP tool functions on an event loop thread |
| `profiler.py` | `PipelineProfiler`: CPU, GPU, RAM, duration metrics |
## CLI (`arena evaluation`)

```bash
arena evaluation list                          # List all benchmark runs
arena evaluation status [<run_id>]             # Show run progress (--watch for live)
arena evaluation tail [<run_id>]               # tail -F progress.csv and every progress.<lane>.csv
arena evaluation ps                            # List running arena OS processes
arena evaluation kill <pid>...|--kind K|--all [-9]  # Stop arena processes, SIGINT first (bare kill lists them, -9 for SIGKILL)
arena evaluation console [<run_id>]            # Tail runner.log (--follow for streaming)
arena evaluation serve [--port N] [--host H]   # Live web dashboard of every run, http://127.0.0.1:8765 by default
arena evaluation serve --runtime                # Plus the runtime page: envs, episodes, spawn, pause, step, lockstep
_meta/tools/arena-web [--prefix PATH]           # On the host: one hub over the dev container and every fork
arena evaluation benchmark                     # Launch a benchmark campaign
```

## Execution Flow

The runner takes a suite (ordered stages) and a contest (planner lineup). It creates a step grid as the Cartesian product of contestants and stages, groups consecutive steps by `(contestant, robot, simulator)`, and spawns one environment per group.

For each group:
1. Calls `/arena/spawn_env` with launch arguments derived from the first step.
2. Waits for environment registration on `/arena/state/envs`.
3. For each step: calls `QueueEpisode` with the stage's `modes` and full-name `params` to set stage config, then executes `RunEpisode` action.
4. Despawns the environment and proceeds to the next group.

Run output structure in `$ARENA_DATA_DIR/benchmarks/<run_id>/`:
```
<run_id>/
|-- manifest.yaml              # Config snapshot
|-- progress.csv               # Append-only, one row per episode
|-- superseded_progress.csv    # Rows of a step's earlier attempts, kept out of progress.csv
|-- runner.log                 # Python logging and launch output
|-- roslog/                    # ROS_LOG_DIR of this run's runtime, arena_env_<id>_<ts>.log per spawned env
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
previous runtime cannot kill the new one. `_restart_arena` clears the latch as well. The
same stamp gates startup: `_start_arena` waits for an `alive: true` report stamped after the
launch before it touches the spawn service, since a killed `arena_runtime_node` keeps its services in
the DDS graph until its lease expires and a request sent there is never answered.

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
`ready`: `arena_runtime_node` reserves an env before it activates, so a spawn that stalls before
`SpawnEnv` returns can still leave a live, unregistered env behind. The stall then marks the
sim dead with reason `spawn_env stalled for <key>`, so the block recovers the runtime and spawns
again under `--max-sim-deaths` instead of failing its queued cells. `<key>` is the block's
representative cell, which the spawn launches, not necessarily the cell that runs next.

Episode budgets are sim time, enforced by the task generator through its `task.episode.timeout` parameter
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
the runner gives up, on top of the inactivity heuristic: a spawn fails after 120s without a
substantive line in this runner's own runtime output or in the log of the env it spawned
(a log that appears in its `roslog` directory during the wait), never another lane's.

`--efficacy FRACTION` adds an outcome check: a non-SUCCESS episode is weak when the closed
fraction of its starting goal distance, `(goal_dist_start - goal_dist_min) / goal_dist_start`
from the `EpisodeRecord`, is below `FRACTION`. It catches a planner that drives badly without
ever erroring, which `--strict` cannot see. When set, the run ends with a
`benchmark: preflight verdict` table (contestant, stage, verdict, episodes run/total, weak
count, worst progress) and exits 3 if any cell is `wedged` (an error kind or an incomplete
episode count) or `weak` (at least one weak episode).

An episode whose `EpisodeRecord.integrity` lists reasons (a robot moved or spawned by hand,
task parameters changed mid-episode, a failed elevator teleport) is compromised. Its log line
ends in `integrity=[...]`, its `progress.csv` row carries the reasons as a JSON list in the last
column, `integrity` (`[]` when intact), and the run ends with a `benchmark: compromised episodes`
table (contestant, compromised count, episodes run) when at least one episode was compromised.
A `progress.csv` written under an older header is rewritten under the current columns when a
runner opens it. The `modes` column holds the selection as a JSON mapping of axis to mode names and
`params_json` the effective params of the selected modes as a JSON list of `{name, value}` under
their full names. A file with the older `tm_robots`, `tm_obstacles`, `tm_modules`,
`robots_params_json` and `obstacles_params_json` columns reads into the two, a leaf param of the
robots or obstacles mode `m` becoming `task.<m>.<leaf>`.

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

## Dashboard

`arena evaluation serve` serves a live web page of every run under the data root, read from the
run dirs alone, so it shows shared runs, runs on other ROS domains and dead runs alike. The run
page has the step grid (contestant by stage, colored by status), each lane's busy envs with their
episode and wall and sim time, failed steps, episodes per minute over the last 10 minutes (the run's
age while it is younger, at least 1 minute) with an ETA, the newest episodes and the runner log tail. Clicking a grid cell
limits the newest episodes to that step. It updates through a server-sent event stream
that watches the run dir's file mtimes once a second.

`benchmark --viz` starts it as a child of the runner before the first step, prints
`viz: http://127.0.0.1:8765/runs/<run_id>` to the console and `runner.log`, and stops it when the
runner exits. `--viz-port N` picks another port. A dashboard of the same data root already on the
port is reused, any other program there makes the runner exit 2 before it creates the run dir.
With `--lanes N --viz` the `--lanes` driver strips both flags from the lane commands and runs one
dashboard in the dev container for the whole run. The MCP tools are `evaluation_benchmark(viz=true)`,
which returns `viz_url`, and `evaluation_dashboard`.

Every runner writes `live.json` (`live.<lane>.json` in a lane): its slot table, sim time and
step counts, replaced atomically on every slot change and every 2 s. A lane whose file is older
than 10 s while not `finished` shows as stale, a sim time that stops for 30 s as not advancing.

`serve --runtime` adds the runtime page (`/runtime`) for the runtime on the server's ROS domain,
built on the `runtime_*`, `episode_*` and `view_*` MCP tool functions it runs on the `arena_mcp`
ROS node: sim health, pause holds, lockstep, each env's robot, world and episode, spawn
(robot, world, planner, task mode per axis, count, more launch args, each a free text field with
suggestions), reset, reset into another world or with a seed, despawn,
despawn all, hold, release, step, lockstep start, pause, resume and stop, and a camera still per
env when a viewport captures (`dashboard_runtime.py`). Control requests are POSTs that must carry
`X-Arena-Dashboard: 1`, which a page of another origin cannot send. While a benchmark runner
publishes `/arena/benchmark/state` the page only watches, every control answers 409. The page
keeps its Activity list in the browser's local storage across reloads. With `--runtime` the runs
page also starts a benchmark (suite, contest, run id, launch args): `benchmark` detached on the
server's ROS domain, refused while a runtime or another runner is up there, its console in
`<data_root>/.web/<run_id>.log`.
`--port 0` binds a free port and prints it, `--exit-on-eof` stops the server when stdin closes.

`_meta/tools/arena-web` is the hub over all of them, run on the host: it finds this machine's arena
containers (the dev container and every fork, by the `ARENA_FORK` in their environment), starts
`serve --runtime --port 0 --exit-on-eof` inside a container on first use through `docker exec -i`,
and proxies its pages under `/t/<container>/`. Every machine runs its own hub over its own
containers; the hub reaches nothing over the network, so the ports a viewer names are the ones on
that machine. The hub's root is the Targets
page: every container with its runtime state, start runtime (with an installed sim and launch
args), stop runtime, connect, new fork and delete fork (typed name),
all run in the containers with the arena CLI and
listed under Activity with their output as it arrives. A row opens its target's runtime page,
connecting first. Closing the hub closes each exec's stdin, which ends the dashboards. It
opens its page with `xdg-open` once it listens when a display is set, `--no-open` leaves that out.
`--only REGEX` limits the forks shown, dev containers always show. `--prefix PATH` serves the hub
under that path for a reverse proxy in front of it: the page then roots every request it makes at
that path.

| Endpoint | Returns |
|---|---|
| `/`, `/runs/<run_id>` | The page |
| `/api/identity` | `app`, `data_root`, `pid`, how `--viz` recognizes a dashboard to reuse |
| `/api/runs` | Light status of the 100 newest runs |
| `/api/runs/<run_id>` | `run_status` plus `grid`, `live`, `episodes`, `recent` |
| `/api/runs/<run_id>/live` | The live lanes alone |
| POST `/api/runs` | Starts a benchmark (`--runtime`): `suite`, `contest`, optional `run_id` and `args` |
| `/api/runs/<run_id>/progress?last=N&step=K` | The last N folded progress rows, of step K alone when given |
| `/api/runs/<run_id>/log?tail=N` | The last N lines of every runner log |
| `/api/runs/<run_id>/events` | Server-sent `live` and `run` events on file changes |
| `/runtime` | The runtime page (`--runtime`) |
| `/api/runtime` | Runtime state, sim time, envs with their episode, benchmark owner, viewport, each env's `view` (`web_port`, `grpc_port`, `url`, or null), `age_s` and `now`. Served from a cache a loop on the bridge refreshes every second while a client asked within 30 s, every action refreshes at once. A failed refresh replaces the cache, the endpoint then answers that error until a refresh succeeds |
| `/api/runtime/options` | `robots`, `worlds`, `planners`, `launch_args`, `sims`, `suites`, `contests` for the forms, `task_modes` as `{axis: {multi, modes: {mode: {namespace, params}}}}` with each param's `name`, `leaf`, `type`, `default`, `label`, `description`, `choices`, `catalog`, `range`, `read_only` and, when declared, `pair` and `widget`, read from the mode schemas, `docs` (the first docstring line of the tool behind each action) and `catalogs` (the kinds below) |
| `/api/runtime/catalog?kind=K&world=W` | `{kind, world, items}` from an env's resolvers when one is up, else offline for `worlds`, `scenarios` (needs `world`) and `robots`, other kinds answer 409. Cached 60 s |
| `/api/runtime/envs/<env_id>/task` | The env's active `modes` per axis and the current `params` (full name to value) of those modes |
| `/api/runtime/envs/<env_id>/camera.png` | One frame of the sim viewport following the env's robot |
| POST `/api/runtime/spawn` | `count`, `launch_args`, optional `modes` (`{axis: [names]}`, validated by the task layer) and `params` (full name to value, each declared by a selected mode) merged over the launch args |
| POST `/api/runtime/<action>` | `despawn_all`, `hold`, `release`, `step`, `lockstep_start`, `lockstep_pause`, `lockstep_resume`, `lockstep_stop` |
| POST `/api/runtime/envs/<env_id>/<action>` | `view` and `view_stop` (the env's live rerun view, present when `rerun_utils` provides the `viz_rerun_*` tools), `reset` (optional `world`, `seed`, and `modes` and `params`, queued with `episode_queue` first), `despawn`, `cleanup` |
| POST `/api/runs/<run_id>/stop` | Stops the run's runners with `evaluation_stop`, 409 with its hint when none runs |

| arena-web endpoint | Returns |
|---|---|
| `/` | The Targets page |
| `/t/<container>/...` | The target's dashboard (`/t/dev/`, `/t/myfork/runtime`) |
| `/hub/targets` | Containers with dashboard state and runtime snapshot, docker errors, busy targets |
| `/hub/activity` | The last 50 hub actions with their output |
| POST `/hub/targets/<container>/<action>` | `connect`, `runtime_up` (`sim`, `args`), `runtime_down`, `fork_down` (`confirm` = fork name) |
| POST `/hub/fork_new` | `name` |

## Run State

`StateFile` manages `.benchmark_state.json` atomically. Each step has status: `ok | partial | failed | skipped | in_progress`. The status tells how completely a step recorded its episodes, not how its planner did: `ok` recorded every episode whatever the outcomes, `partial` lost some (no `EpisodeRecord`, a sim stall), `failed` recorded none or hit an error. Interrupted runs resume with `--resume <run_id>`, which reruns `partial`, `skipped` and `in_progress` steps, and `failed` ones that recorded no episode without an error, or whose error is systemic or a sim stall, or all with `--retry-failed`. A rerun replaces the step's episodes: when progress is deduped or folded, rows written before the step's latest `started_at` move to `superseded_progress.csv`.

Run ID format: `{YYYYMMDD-HHMMSS}-{suite}-{contest}`, `{YYYYMMDD-HHMMSS}-{lane}-{suite}-{contest}` inside a lane, `{YYYYMMDD-HHMMSS}-{suite}-x{N}` for `--lanes N`.

## Shared Runs

`--lanes N` starts one runner per lane on the same run dir. Lanes claim env blocks so each runs once, and fold their files into `progress.csv` and `.benchmark_state.json`:

```
<run_id>/
|-- progress.<lane>.csv        # This lane's episodes
|-- .benchmark_state.<lane>.json
|-- runner.<lane>.log
|-- roslog.<lane>/
|-- runner.<lane>.pid          # Signalled by --lanes on Ctrl-C
|-- live.<lane>.json           # This lane's slot table for the dashboard
|-- claims/<token>/            # One file per (env, world) block, taken atomically
`-- episodes/.next_episode_id  # Episode id counter, locked across lanes
```

