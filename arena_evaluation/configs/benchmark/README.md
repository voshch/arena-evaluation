# Benchmark Configurations

The `arena evaluation benchmark` runner reads benchmark configurations from `arena_evaluation/configs/benchmark/` at startup.

Invocation:
```bash
arena evaluation benchmark --suite <name> --contest <name> [--scale-episodes N]
```

## Directory Layout

```
configs/benchmark/
|-- suites/           - Stage sequences (maps, episodes, task modes)
|   |-- basic.yaml
|   |-- meta_suite.yaml
|   |-- all_maps_random.yaml
|   |-- arena_corridor.yaml
|   |-- arena_hospital_small.yaml
|   |-- map_empty.yaml
|   `-- characterization.yaml
`-- contests/         - Planner lineups
    |-- basic.yaml
    |-- allplanners.yaml
    |-- inter.yaml
    |-- planners.yaml
    `-- characterization.yaml
```

## Suite Files

A suite defines an ordered list of stages executed sequentially across all contestants.

```yaml
launch:                       # optional. launch args for the whole run, CLI passthrough wins. stage-owned keys (world, robot, run_seed, task.*, record.*) are rejected
  lockstep: true
metrics:                      # optional. metric-layer overrides, snapshotted into manifest.yaml and read back at process time
  max_collisions: 1           # collisions per episode that flip success to COLLISION (default 3, 1 = any collision fails)
stages:
  - name: scenario
    map: arena_hospital_small
    robot: jackal
    tm_robots: scenario
    tm_obstacles: random
    episodes: 1
    config:
      scenario:
        file: 4.json
      random:
        dynamic:  {min: 3, max: 5, models: [arenian]}  # -> task.random.dynamic.n=[3,5], task.random.dynamic.models
        static:   {min: 5, max: 10, models: [shelf]}
```

A `{min, max}` pair anywhere in `config` is a runner-side convenience: it collapses to a
single `<leaf>.n` param holding `[min, max]` (the actual ROS param the mode declares, e.g.
`random`'s `static.n`/`dynamic.n`/`interactive.n` - see
[task_generator/tasks/obstacles/README.md](../../../../task_generator/task_generator/tasks/obstacles/README.md)).
Writing `static: {n: [5, 10]}` directly also works. For `scenario`, a `file` value with an
extension has the extension stripped before it is sent (`4.json` and `4` are equivalent).

### Stage fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Stage label |
| `map` | string | World name |
| `robot` | string | Robot model |
| `tm_robots` | string | `Constants.TaskMode.TM_Robots` enum key (case-insensitive) |
| `tm_obstacles` | string | `Constants.TaskMode.TM_Obstacles` enum key (case-insensitive) |
| `episodes` | int | Episode count (scaled by the `--scale-episodes` CLI flag, default 1.0) |
| `config` | dict | Per-mode params; top-level keys must match `tm_robots`/`tm_obstacles` (e.g. `scenario`, `random`). Inner leaves map to `task.<mode>.<leaf>` via QueueEpisode (see [task_generator/tasks/obstacles/README.md](../../../../task_generator/task_generator/tasks/obstacles/README.md)) |
| `seed` | int | Auto-derived from a SHA-1 hash of the stage fields (excluding `config`); can be set explicitly |
| `timeout` | string | Per-episode budget in **sim** seconds, e.g. `300s`/`5m`. The runner pushes it to the env's task generator as its `timeout` parameter before every step, so the episode ends FAILED with `outcome_info='timeout'`. Defaults to 60s if absent |
| `timeout_peds` | string | Same budget for the `unhindered_peds` reference step of this stage (see `references` below). Defaults to `timeout` if absent |
| `optim` | dict | Extra `optim.<key>:=<value>` launch args for this stage's env |

### Directory bundles

A suite may also be a directory: `suites/<name>/suite.yaml`, with the same schema as a
flat file. A flat `suites/<name>.yaml` takes precedence when both exist. If the bundle
contains a `worlds/` subdirectory, the runner exports it via `ARENA_WORLD_PATH` to the
launched runtime, so the bundled world directories resolve ahead of the canonical worlds
tree in every sim process (an `ARENA_WORLD_PATH` already set in the outer environment
keeps priority). See `suites/acoustics/` for an example that ships its own worlds.

### Inline configs

`--suite` and `--contest` also accept a YAML literal instead of a name. A value whose first
non-space character is `[` or `{` is parsed with `yaml.safe_load` and used directly, with the
same schema as the corresponding file. The run is recorded under the stem `inline`, so it is
not resumable by name. Quote the literal so the shell passes it as one argument:

```bash
# simplest: name only, robot comes up with the sim's default mobile adapter and planners
arena evaluation benchmark --suite basic --contest '[{name: default}]'
arena evaluation benchmark --suite basic \
  --contest '[{name: dwb, mobile: {driver: nav2, local_planner: dwb, inter_planner: navigate_w_replanning_time}}]'
# sweep form works too
arena evaluation benchmark --suite basic --contest '{mobile: {driver: nav2, local_planner: [teb, dwa]}}'
```

The robot model is a stage field, not a contestant field. `robot` (like `world`, `sim` and
the `task.*` keys) is set by the stage and any contestant key that collides with it is dropped
with a warning. To run a single planner on a single robot without touching the config tree,
inline both:

```bash
arena evaluation benchmark \
  --suite '{stages: [{name: jackal, map: arena_hospital_small, robot: jackal, tm_robots: random, tm_obstacles: random, episodes: 5}]}' \
  --contest '[{name: default}]'
```

### Bucket-hosted configs

Suites, contests and manifests are `Identifier`s, resolved through the same chain as every
other arena asset: the package share dir, then the source tree, then one `NetResolver` per
`BENCHMARK_BUCKETS` provider (default `arena-benchmarks-prod-public`), then a write-only
fallback. A local copy always shadows a bucket copy, so hosting a config never overrides one
you are editing.

In a bucket a config is always a directory bundle, keyed by kind:
`suites/<name>/suite.yaml`, `contests/<name>/contest.yaml`, `manifests/<name>/manifest.yaml`.
A flat local `<name>.yaml` is wrapped into that shape on publish, so nothing has to be
restructured by hand. Fetched bundles cache under `$ARENA_ASSETS_DIR/<bucket>/` with a one
year TTL, since a published config is expected to be pinned rather than revised in place.

Move them with `arena asset find|ls|pull|push suite|contest|manifest <name>`. `find` prints
every resolver's verdict without downloading. Reading is anonymous. Pushing requires
`GCS_ACCESS_TOKEN` in the environment, and for a suite first checks that every world
its stages name would resolve for someone else, counting worlds bundled under the suite.
Task modes are not checked, since they are code rather than data shipped with the suite.
Contests and manifests carry no such closure and publish as-is.

Suite-level `references: true` enables automatically generated reference steps
(`unobstructed_robot` / `unhindered_peds`) per stage. The default is `false`. Enable it for
suites whose report manifest consumes reference-episode data (e.g. pedestrian disturbance).
Characterization suites leave it off because the sweep itself is the reference.

### Characterization suite

```yaml
references: false
stages:
  - name: characterization
    map: map_empty
    robot: jackal
    episodes: 3            # repetitions -> cross-episode confidence bands
    tm_robots: characterization   # TM_Robots mode that drives the open-loop sweep
    tm_obstacles: random
    config:
      random:
        dynamic: {min: 0, max: 0}
        static: {min: 0, max: 0}
        interactive: {min: 0, max: 0}
    timeout: 300s
```

The `characterization` robot task mode (in `task_generator`) drives `cmd_vel` directly through the
robot's rated envelope - idle blocks, 0.25->vx_max linear steps with 5 s out-and-back dwells,
transient ramps, pivot rates - tagging every maneuver with `characterization_phase` markers. It also
latches the full schedule as a `characterization_schedule` table, which the report's
`CharacterizationCalculator` joins against so phase labels match exactly, with `char_phase_coverage`
reporting the fraction that did. Run it like any benchmark and analyse with the characterization
report manifest:

```bash
arena evaluation benchmark --suite characterization --contest characterization
arena evaluation run --benchmark-dir <run_id> --report-manifest characterization
```

### Lockstep soak

`arena planners test <planner...|--all>` runs one short crowded stage per
contestant with `lockstep: true` (`lockstep.paused: false`, headless). The
runner always watches `/arena/state/lockstep` and records, per episode and per
step, the stall count, the longest stall, the mean measured rtf and the hard
channels registered (`lockstep_*` columns in `progress.csv`, `lockstep` in
`.benchmark_state.json`). When any step ran under lockstep the run ends with a
report table, and `--lockstep-verdict` turns a `fail` row into exit code 3. A
row fails when a stall reaches 5 s or no `nav/`/`planner/` beat ever
registered for the contestant, i.e. the planner never engaged the gate.

Suite and contest are both built inline by the `arena` CLI
(`arena_cli/features/planners.py`), so nothing here defines them: bridge planners become
`{mobile: {driver: drl, planner: <name>}}` contestants, anything else a nav2
`local_planner`, and the benchmark runs with `--lockstep-verdict`. Extra
`key:=value` tokens pass through (`sim:=isaac`, `lockstep.rtf:=5`).

## Contest files

A contest defines the set of planner configurations (contestants) to evaluate.
The runner iterates over all contestants at each suite stage.

Contestant args use **cap-scoped dicts** - each capability (`mobile`, `arm`,
`planner`) is a dict with a `driver` key identifying the driver and any extra
kwargs forwarded as launch args under the `robot.` prefix. `planner` also
accepts a bare scalar value as shorthand for `robot.planner:=<value>`:

```yaml
mobile:
  driver: nav2
  local_planner: teb
  inter_planner: polite
```

This produces launch args: `robot.mobile:=nav2 robot.mobile.local_planner:=teb robot.mobile.inter_planner:=polite`

There are two forms: **list** and **sweep**.

### List form

Top-level YAML is a sequence. Each entry must have `name`; all other keys
become `args` forwarded to `Robot.parse` via the `SpawnRobot` service.

```yaml
- name: teb-polite
  mobile:
    driver: nav2
    local_planner: teb
    inter_planner: polite
- name: dwa-rl
  mobile:
    driver: rosnav_rl
    agent: my_agent
```

### Sweep Form

Cartesian product generated across list-valued parameters:

```yaml
name: basic
mobile:
  driver: nav2
  inter_planner: bypass
  local_planner: [teb, dwa, rosnav]
```

Produces: `basic-teb`, `basic-dwa`, `basic-rosnav`.

### Inline Contest (CLI)

```bash
arena evaluation benchmark --suite basic --contest '[{name: teb, mobile: {driver: nav2, local_planner: teb}}]'
arena evaluation benchmark --suite basic --contest '{mobile: {driver: nav2, local_planner: [teb, dwa]}}'
```

The old flat dot-notation format is still accepted in both list and sweep forms:

```yaml
# Old flat list form - still works
- name: teb
  mobile.local_planner: teb
  mobile.inter_planner: bypass

# Old flat sweep form - still works
mobile.local_planner: [teb, dwa]
mobile.inter_planner: bypass
```

### Contestant args

Contestant `args` keys are forwarded verbatim as launch args to the env on
spawn (so nav2, the controller, the agent, etc. come up correctly from the
start). See
[BRINGUP.md -> Cap-scoped overrides](../../../../arena_bringup/BRINGUP.md#cap-scoped-overrides)
for the recommended key shapes.

The runner drops keys that collide with stage-owned launch args (`sim`,
`robot`, `world`, `task.robots`, `task.obstacles`, `run_seed`, `task.auto_reset`,
`task.modules`, `record.dir`, `record.auto`) and logs a warning, since those are
controlled by the suite stage. Anything else is passed through to the launch
layer, which binds it if declared or raises an error if not.

A useful passthrough example: `task.fail_on_collision: true` makes the env abort
an episode as FAILED (`outcome_info='collision'`) the moment the robot
footprint contacts a wall, static obstacle, or pedestrian, instead of the
default run-to-goal-or-timeout. See
[BRINGUP.md -> Common options](../../../../arena_bringup/BRINGUP.md#common-options).

## How the runner consumes these files

The runner is the `benchmark` console script: `arena evaluation benchmark`,
no launch file. In the Arena meta-repo, install the feature first with
`arena feature evaluation install`.

Steps are grouped by `(contestant.name, stage.robot, simulator)`. Consecutive
steps with the same key share one env; suite order is preserved within and
across groups. Robot is fixed within a group. If a contestant's stages mix
robots, the runner splits into multiple groups per contestant. Authoring
suggestion: keep one robot per contestant for fastest runs.

`env.n` caps how many groups (parallel contestants) run at once, with the rest
queued. Run time scales as `bringup_time x num_groups + episode_time x
total_episodes` spread across the `env.n` workers, not `bringup_time x
num_steps`, since steps within a group reuse one env.

For each group the runner:

1. Calls `/arena/spawn_env` once with the first step's launch args: `sim`,
   `robot`, `world`, `task.robots`, `task.obstacles`, `run_seed`,
   `task.auto_reset:=false`, `task.modules:=` (empty), any `optim.<key>:=<value>`
   from the stage's `optim` dict, and any contestant args of
   shape `mobile`, `arm`, `mobile.<key>`, or `arm.<key>` (emitted as
   `robot.mobile...` / `robot.arm...`). `record.dir` and `record.auto:=false`
   are added when recording is enabled.
   Per-mode params (`task.scenario.file`, `task.random.*`, ...) are not passed
   as launch args; the runner sets them via QueueEpisode in step 3.
2. Waits for the env to publish on `/arena/state/envs` and resolves the env
   namespace from `EnvRecord.fqn`. Sets up a `RunEpisode` action client and a
   `QueueEpisode` service client at `<env_ns>/config/queue_episode`.
3. For each step in the group:
   - Calls `<env_ns>/config/queue_episode` with the stage's `world`,
     `tm_robots`, `tm_obstacles`, and the per-mode `config` blocks routed to
     `obstacles_params` / `robots_params` as leaf-keyed `Parameter[]` (e.g.
     `file`, `dynamic.min`). MERGE semantics: empty fields leave the prior
     queued value untouched. Called for every step including the first; the
     env owns no stage-specific config until the runner pushes it.
   - Sets the env's `timeout` parameter to the stage's `timeout` (or
     `timeout_peds` for the `unhindered_peds` reference step) via
     `<env_ns>/set_parameters`, so the episode budget is spent in sim time.
     The runner keeps no wall-clock episode ceiling, only a 60 s sim-stall
     guard (see [benchmark/README.md](../../arena_evaluation/benchmark/README.md#liveness)).
   - Drives `step.episodes` goals via the `RunEpisode` action at
     `<env_ns>/lifecycle/run_episode`, with `goal.world = stage.map` and
     `goal.seed = stage.seed` overriding the queued world/seed if needed.
     Per-episode `EpisodeRecord` rows are pulled from `<env_ns>/state/episode`
     and appended to `progress.csv`.
   - Per-step failures (timeout, missing EpisodeRecord, mid-run robot stuck)
     advance to the next step; only systemic setup failures
     (`error_kind in {env_setup, robot_setup}`) **before any episode in the
     run has completed** trigger a run-level abort.
4. Despawns the env via `/arena/despawn_env` at the end of the group, then
   advances to the next group.
5. Writes `progress.csv` and `.benchmark_state.json` to
   `$ARENA_DATA_DIR/benchmarks/<run_id>/` so an interrupted run can be
   resumed with `--resume <run_id>`.

`progress.csv` schema is unchanged (one row per episode; `env_id` is shared
within a group). `.benchmark_state.json` schema is unchanged.

Run-id default format: `{YYYYMMDD-HHMMSS}-{suite}-{contest}` (lex sort = chronological).
For inline contests the contest segment is `inline`. Override with `--run-id`.

Output dir: `$ARENA_DATA_DIR/benchmarks/<run_id>/` (default `$ARENA_WS_DIR/data/benchmarks/<run_id>/`).
Override with `--data-root`. Inside Docker: `/opt/arena_ws/data/benchmarks/<run_id>/`.

```
<run_id>/
|-- manifest.yaml              # Config snapshot
|-- progress.csv               # Episode results
|-- runner.log                 # Execution logs
|-- .benchmark_state.json      # Step state (atomic)
|-- episodes/                  # Per-episode MCAP files
|-- combined_metrics.parquet   # Computed metrics
`-- report_manifest.yaml       # Used report manifest
```

## Inspection & Management Commands
```bash
arena evaluation list                          # List runs
arena evaluation status [run_id] [--watch]     # Show progress
arena evaluation tail [run_id]                 # Tail progress.csv
arena evaluation ps                            # List running arena processes
arena evaluation kill [pid...] [-9]            # Terminate running processes (-9 for SIGKILL)
arena evaluation console [run_id] [--follow]   # Tail runner.log
```
