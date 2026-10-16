# promptgen

A prompt, a world and a base scenario in; `scenarios/<base>__<prompt_id>/` out. Part of [generation](../README.md); the effects it writes are run by the `edge_case` obstacle mode ([README](../../../../../task_generator/task_generator/tasks/obstacles/edge_case/README.md)).

```
prompt + world + base
  -> Step A  crowd-wide, by lexicon (+ a cheap model for who)   lexicon.yaml, step_a.py, step_a_llm.py   blackout, evacuation, stampede, busy
  -> Step D  formations, by model                               step_d.py, system_d.md                   stubborn, interaction_unbroken, formation_break
  -> Step BC tune some pedestrians; one against the robot      step_bc.py, system_bc.md
  -> scenario.yaml (format 2 population), edge_case.yaml (`{effects: [...]}`), types/*.yaml (derived profiles), record.json
```

| File | Purpose |
|---|---|
| `pipeline.py` | `generate`, `write_result`; pins the robot route to planned points |
| `world.py` | `WorldContext`: zones with roles, the furnished grid, the planned route, placement rules |
| `spec.py` | typed decisions per step, the record, the dedup key, the effect vocabulary |
| `profiles.py` | profile library and tiers; a model emits tiers, never numbers. `tune_block` turns `TUNABLE` tiers into a HumanSim `tune:` (tiers as plain numbers, a speed scale as `{mul: s}`) |
| `agents.py` | derived agent-type files beside the scenario (`derived`: parameters pinned; `walking`: a walker's route baked into a behaviour tree; `switching`: a sequence a need preempts into, holding a tune; `tuned`: a type-level `tunes:` entry held while a need is up) |
| `prompts.py`, `prompts/` | `prompts/universal.yaml`: the 13 frozen prompts the dataset is built from; `load_sets` reads it (plus `<world>*.yaml` and `--prompts` files when given); the place and duration phrases the system prompts list |
| `lexicon.py`, `lexicon.yaml` | Step A phrases, onsets and parameters |
| `llm.py` | cached model calls, and `Oracle` for `--oracle` runs |
| `build_kb.py` | renders `system_{a,bc,d}.md` from `system_*.tmpl.md`, code constants and `kb_ranges.yaml` |

`system_*.md` are build output: edit the `.tmpl.md` and run `python3 -m arena_evaluation.generation.promptgen.build_kb`; a test fails when the rendering is stale.

Placement keeps an injected pedestrian 1.5 m from the robot's start, 1.0 m from its goal and 1.5 m from the route unless the case owns the route; Step C is placed at least 2 m from a Step D placement (the model is asked again with the reason; `--oracle` moves C along the route).
