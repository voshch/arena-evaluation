# generation (Layer 0: Scenario Generation)

Writes the scenarios the `edge_case` task modes run: base populations per world, prompt-driven edge cases on top of them, and the benchmark suite that sends them through the [runner](../benchmark/README.md). The runtime side is the `edge_case` obstacle mode ([README](../../../../task_generator/task_generator/tasks/obstacles/edge_case/README.md)) and robot mode ([README](../../../../task_generator/task_generator/tasks/robots/edge_case/README.md)).

```
world + zone roles --populate--> bases --promptgen--> <base>__<prompt_id> --suite--> suite.yaml --benchmark--> episodes --run--> report
```

---

## Files

| Path | Purpose |
|---|---|
| `cli.py` | `evaluation generate <verb>` |
| `suite.py` | scenario directories -> suite: one `edge_case` stage per (world, scenario) |
| [`scenario/`](scenario/README.md) | zone roles, the furnished occupancy grid, route screening, the four base scenarios |
| [`promptgen/`](promptgen/README.md) | prompt + world + base -> scenario with an `edge_case.yaml` effects block |
| [`formats.py`](formats.py) | worlds loaded through arena_simulation_setup's `WorldView` into the generator's working shape (`levels`, `first_level`), scenarios read (`read_scenario`) and written as format 2 + `edge_case.yaml` (`write_scenario`) |
| [`propose/`](propose/README.md) | the one model call path, and model classification of zone names no rule covers |

---

## CLI

`ros2 run arena_evaluation evaluation generate <verb>`, or `python3 -m arena_evaluation.generation.cli <verb>`. A world is found the way the runtime finds it, `WorldIdentifier(<world>)` (`ARENA_WORLD_PATH` first, then the installed worlds), and generated scenarios are written beside it; `--worlds DIR` reads and writes `DIR/<world>` instead. Set `ARENA_WORLD_PATH` to the source tree's `arena_simulation_setup/worlds` to generate into it and have the runtime run it without a rebuild. Writing verbs are dry runs without `--write`.

| Verb | Writes | Purpose |
|---|---|---|
| `zones <world>` | - | each zone's role (work, break, transit, private, unknown) |
| `classify [world...]` | `scenario/zone_roles.yaml` | ask a model for the roles no substring rule covers |
| `populate <world>` | `scenarios/{normal,break,night,empty}/` | the four base scenarios |
| `promptgen <world>` | `scenarios/<base>__<prompt_id>/` | realise every prompt on every base (`--ids` picks prompts, `--oracle` needs no model) |
| `suite <world...>` | `configs/benchmark/suites/<name>.yaml` | a suite over the generated scenarios (`--match` globs pick others) |

```bash
ros2 run arena_evaluation evaluation generate populate hospital_1 --write --prune
ros2 run arena_evaluation evaluation generate promptgen hospital_1 --write -v
ros2 run arena_evaluation evaluation generate suite hospital_1 office_1 --robot jackal --episodes 3 --name edge_case_core
arena evaluation benchmark --suite edge_case_core --contest edge_case_differential
arena evaluation run --benchmark-dir <run_id>
```

---

## Running a dataset

A suite stage is one scenario on one world with both task modes `edge_case`: `config: {scenario: {file: <name>}}`, which the runner sets as `task.scenario.file` because no queued mode owns the `scenario` block ([stage config](../../configs/benchmark/README.md#stage-fields)). Every stage carries the same `seed`, so all contestants meet the same episodes and a stage name (`<world>.<scenario>`) is the key for paired statistics.

- **One drive family per run.** The suite fixes the robot. A contest on it compares planners on one chassis, and needs the family's nav2 control: `contests/edge_case_differential.yaml` (jackal) and `contests/edge_case_omnidirectional.yaml` (mpo700) carry it.
- **Screening and dataset seeds are disjoint** (`--seed`). Ranking cases on observed separation and reporting that separation on the same episodes is selection on the outcome.
- **A generated suite names local scenarios.** Generated scenario directories are gitignored on the author's machine, so the suite file is too local to commit.
- **Criticality is not in the report yet.** The obstacle mode writes `cases.jsonl` and `scores.jsonl` under `task.edge_case.record_dir` (default `$ARENA_DATA_DIR/edge_case`); rows join to the benchmark's episodes on world, scenario and seed.

---

## Invariants

- **Format 2 on disk, format 1 shapes in memory.** Worlds load through arena_simulation_setup's `WorldView` (either format) and assets resolve through its identifiers, never by path; every scenario is written as `arena-scenario/2` by its `upgrade_scenario`, refusing to write one it would lose a key of, with the block in `edge_case.yaml` and the header on both files.
- **People are written in rooms, never coordinates.** `pose: laboratory`, `waypoints: [...]`, resolved at run time by `PointResolver`. Two exceptions: the robot's start and goal are the exact points the route was planned on, and an agent a crowd change reaches walks a route baked into its type (below).
- **Normal until the onset.** Every case is an effect with a `when`; a crowd-wide change starts at its onset, never baked into the population, except a parameter HumanSim has no `tune:` for (below). An agent belongs to at most one step.
- **Every agent a change reaches is a behaviour tree that changes itself on a stimulus.** HumanSim cannot change a running agent from outside (`UpdateAgents` is gone) and a route rewrite on a tree can crash it, so:
  - a plain walker the change reaches gets its own walking tree, `types/<type>__walk_<agent>` (`TypeFiles.walking`, `step_a.walk_route`): its `walk` sequence is one `go_to` per waypoint, drawn once in each room (`WorldContext.zone_sample`) where the runtime would draw it per episode, `reverse` and `random` order baked, looped. Offline it walks like the plain walker (1.08 vs 1.13 m/s on an open loop, a short halt at each point);
  - evacuation and stampede switch every agent (`TypeFiles.switching`, need `alarm`) to an `evacuate` / `rush` sequence walking to the exit or work-zone centroid and holding the run as its HumanSim `tune:`; no `rally`;
  - blackout and B `intrinsic` give every agent a type-level `tunes:` entry (`TypeFiles.tuned`, need `blackout` / `intrinsic`) that HumanSim holds while the need is above 50, over whatever sequence runs and without restarting its step;
  - B `erratic_route` (and `erratic_behaviour` on a plain walker) switches each agent (need `erratic`) to its own looped `erratic` sequence, six `go_to` points `radius` around its start, each given up after `period` (`step_bc.erratic_trees`), the tree form of the runtime `shuffle`;
  - an unbroken D formation (`standing`, `blocking`) switches its members (need `form`) to their spots, facing the centre or the robot, where they stand with repulsion low; a formation that breaks keeps plain walkers and its `rally` / `scatter`;
- **Formations stand in the robot's way when the prompt puts them there.** A talking circle stands `CIRCLE_SPACING_M` = 0.9 m between neighbours (two people 0.9 m apart, too tight for the robot to pass between; the old 1.5 m pairs let it drive through). One that does not fit the passage at the requested route fraction huddles to `HUDDLE_RADIUS_M` = 0.35 m, and a queue stands single file, on the route, before the generator stands it beside the route in a room. Over the dataset: standing / meet 0% and queues 2% more than 1.5 m off the route (were 22 / 13 / 24%).
  - tunes are the `TUNABLE` params (`profiles.tune_block`: tiers as plain numbers, a speed scale as `desired_velocity: {mul: s}`); a parameter outside it (radius, need decay, step duration) is pinned into the agent's type from the start (`step_a.bake`), as is B `erratic_behaviour`'s need decay on trees.
- **Every agent block carries its type's radius.** HumanSim's adapter overrides an undeclared `radius` with 0.35 m, so `formats.write_scenario` writes the type's mean (`agent_radius`) into every block, scenario-local types resolved beside the scenario.
- **The model runs at temperature 0**, every answer cached by content hash under `$ARENA_DATA_DIR/promptgen/llm_cache`; regeneration is byte-identical and free. `source: human` rows in `zone_roles.yaml` are never overwritten, and `unknown` is a permitted answer.
- **Generated files carry `GENERATED_MARKER`.** `populate --write` refuses to replace a scenario without it unless `--force`; `--prune` removes only marked bases this run did not write.
- **Same world and seed reproduce byte-identically.** World variance (tempo, crowd size, flow) is drawn from an RNG keyed on world name, profile and seed.
- **The effect grammar is mirrored, not imported.** `promptgen/spec.py` writes what `task_generator`'s `edge_case/effects.py` parses; a key the mode does not know aborts the episode there.
