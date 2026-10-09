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
| [`formats.py`](formats.py) | format 2 files in, the generator's working shape out (`read_scenario`, `levels`, `level_map`); every scenario written as format 2 + `edge_case.yaml` (`write_scenario`) |
| [`propose/`](propose/README.md) | the one model call path, and model classification of zone names no rule covers |

---

## CLI

`ros2 run arena_evaluation evaluation generate <verb>`, or `python3 -m arena_evaluation.generation.cli <verb>`. Every verb takes `--worlds DIR` (default `$ARENA_DIR/arena_simulation_setup/worlds`, the source tree: generated scenarios are written there). Writing verbs are dry runs without `--write`.

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
colcon build --packages-select arena_simulation_setup --symlink-install   # new scenario dirs are invisible until then
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

- **Format 2 on disk, format 1 shapes in memory.** Worlds and scenarios are read in either format through `formats` (format 1 worlds through arena_simulation_setup's legacy upgrader shape); every scenario is written as `arena-scenario/2` by arena_simulation_setup's `upgrade_scenario`, refusing to write one it would lose a key of, with the block in `edge_case.yaml` and the header on both files.
- **People are written in rooms, never coordinates.** `pose: laboratory`, `waypoints: [...]`, resolved at run time by `PointResolver`. The robot is the exception: its start and goal are the exact points the route was planned on.
- **Normal until the onset.** Every case is an effect with a `when`; a crowd-wide change is a `retune` or `rally` at its onset, never baked into the population. An agent belongs to at most one step.
- **`retune` has no backend under `human:=arena`** since HumanSim dropped `UpdateAgents` (2026-10-09): the obstacle mode records every refusal as a failure, so a case that relies on it measures nothing.
- **The model runs at temperature 0**, every answer cached by content hash under `$ARENA_DATA_DIR/promptgen/llm_cache`; regeneration is byte-identical and free. `source: human` rows in `zone_roles.yaml` are never overwritten, and `unknown` is a permitted answer.
- **Generated files carry `GENERATED_MARKER`.** `populate --write` refuses to replace a scenario without it unless `--force`; `--prune` removes only marked bases this run did not write.
- **Same world and seed reproduce byte-identically.** World variance (tempo, crowd size, flow) is drawn from an RNG keyed on world name, profile and seed.
- **The effect grammar is mirrored, not imported.** `promptgen/spec.py` writes what `task_generator`'s `edge_case/effects.py` parses; a key the mode does not know aborts the episode there.
