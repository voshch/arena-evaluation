# Migration guide: recordings and report declarations

> **Audited commit:** `arena_evaluation` `enhancement/evaluation-overhaul` @ **`f355ece2`** (2026-11-17).
> Everything below was checked against that tree; the one end-to-end recording attempt that was
> executed is quoted with its command and its result (§1.7). Claims that could not be reproduced are
> marked **unverified**.

The overhaul changed two file formats that users already have on disk. Both have a migration path,
and both migration paths are **loud**: they never guess a value, never invent a default, and never
leave an old artifact silently in place.

| Old artifact | New artifact | Migration |
|---|---|---|
| A benchmark recorded before the native layout (`episodes/episode_NNN/*.mcap` + old metadata, old `topics/`) | The native layout (current metadata, `topics/`, `metrics/`, benchmark stamps) | `evaluation migrate` (§1) |
| Pre-overhaul report manifests (`configs/benchmark/manifests/*.yaml`, `plot_declarations/*.yaml`) and their plot grammar | `configs/reports/<name>/report.yaml`, `configs/plots/<name>/plot.yaml`, `configs/queries/<name>/query.yaml` | Converter + MCP loop (§2) |
| Old processed outputs (`metrics.parquet`, `combined_metrics.parquet`, stale/incomplete `metrics/fact_*.parquet`) | Recomputed facts and series | Quarantined by `migrate`; recomputed by `process --force-process` (§1.6) |

---

## 1. Recording migration

### 1.1 Supported generations

`processing/legacy_schemas.py` classifies a recording from its MCAP channel and schema records
(`detect()`), before any calculator sees a message. Four names are possible:

| Generation | `EpisodeRecord` schema | `AgentMeta` | Fleet |
|---|---|---|---|
| `native` | `arena_runtime_msgs/msg/EpisodeRecord` | carries the four `robot_*` settings | `RobotFleet` |
| `legacy_2026_09` | `task_generator_msgs/msg/EpisodeRecord` (`tm_*` fields) | native or absent | `task_generator_msgs/msg/RobotState[]` |
| `legacy_2026_11` | `arena_runtime_msgs/msg/EpisodeRecord` | **no** `robot_*` fields | `RobotFleet` or absent |
| `unknown` | any other present episode schema | — | any other present fleet schema |

A recording with no `/state/episode` topic is classified from its remaining schemas
(`_absent_episode_generation`): a crowd-less `AgentMeta` is `legacy_2026_11`; a
`task_generator_msgs` fleet is `legacy_2026_09`; otherwise `native`. A **present** but unrecognised
episode or fleet schema is `unknown` and is refused with its name — the migration does not try to
interpret a schema it does not know.

### 1.2 Command and flags

`evaluation migrate` takes **one positional directory**: a benchmark root (detected by its
`episodes/` layout) or one episode folder. It is not a `--benchmark-dir`/`--run-dir` verb.

```bash
# plan only: classify every episode, write nothing
evaluation migrate <benchmark-or-episode-dir> --dry-run

# migrate in place and write the report at the default <run>/migration_report.yaml
evaluation migrate <benchmark-or-episode-dir>

# migrate and write the report elsewhere
evaluation migrate <benchmark-or-episode-dir> --report /tmp/migration_report.yaml

# re-extract and rewrite even when metadata and topics are already current
evaluation migrate <benchmark-or-episode-dir> --force
```

| Flag | Effect |
|---|---|
| `--dry-run` | Detect and report only. No metadata, no tables, no `legacy/` move; validates the descriptor it would need to synthesise. |
| `--report PATH` | Write the YAML report to `PATH` instead of `<run>/migration_report.yaml` (no report is written on `--dry-run` unless `--report` is given). |
| `--force` | Re-extract the recording and rewrite the metadata even when `topics/` is already current. Idempotence is the default: a migrated episode with a current cache is left alone. |

Until the metarepo `arena evaluation migrate` verb lands, call it through the package:
`ros2 run arena_evaluation evaluation migrate <dir>` (the metarepo `_meta/arena_cli` dispatch
currently exposes `extract`, `run`, `process`, `report`, `plot` and `acoustic`, not `migrate`).

### 1.3 What is mapped

| Old data | New data | Rule |
|---|---|---|
| `tm_robots`, `tm_obstacles`, `tm_modules` | `modes.robots/obstacles/modules` | One axis per `tm_*` field; an absent field leaves that axis out. This is `legacy_modes()`. |
| Nested `robots_params`, `obstacles_params` | `params["task.<mode>.<leaf>"]` | Flattened leaves of the first mode on each axis. This is `legacy_params()` / `flatten_params()`. |
| `EpisodeRecord` fields | Native `EpisodeFields` | `episode_id, world, seed, outcome_*, goal_*, path_length, start_t_ns, integrity, robots` are mapped from the generation's record. |
| `/state/robots` fleet | `robot_descriptor.parquet`, `robot_params.parquet` | Native and `legacy_2026_09` RobotState lists are read by the extractor; parameters are whatever the message carried (see §1.5). |
| MCAP topics | `topics/*.parquet` | The recording is re-extracted through `MCAPReader` into the native table names; the old `topics/` is replaced atomically. |
| Episode YAML | Rewritten native `RunMetadata` | The migrated metadata is validated by `RunMetadata.model_validate`; an incomplete mapping is a `MigrationError`. |

### 1.4 NULL and reason mapping

Nothing absent is invented. A field the generation did not carry is NULL and named:

| Missing construct | Result |
|---|---|
| `task_layer` on a `legacy_2026_09` EpisodeRecord | NULL; `not_recorded` (named in `EpisodeFields.not_recorded`). |
| `robot_awareness`, `robot_fov_gain`, `robot_repulsion_gain`, `robot_reaction_gain` (AgentMeta without the `robot_*` fields) | All four NULL with `not_recorded`; the crowd condition therefore has no source. |
| No `/state/robots` fleet | `robot_descriptor`/`robot_params` synthesised (§1.5); the report's `fallbacks` says so. |
| `conditions`, `phases`, `human_params` missing from an EpisodeRecord | **Not tolerated**: `_legacy_episode` reads them directly. A `legacy_2026_09` record that predates them raises `AttributeError`; this is the open H-012 defect against the September 620-set recording (owner FIX1/FIX11). |
| No environment token in topics or TF frames, or several | `MigrationError`, no writing. |
| Metadata with neither `modes` nor `tm_*` | `MigrationError` pointing at the migration. |

### 1.5 Descriptor synthesis rule

A recording without a fleet cannot name its robots' morphology. The migration synthesises the two
descriptor tables **once**, from data that already exists:

1. Robot names come from the recording's robot topics (`_robot_names`).
2. Models come from `RunMetadata.robot_model` (`_descriptor_assignment`): one model for every robot,
   or an equal-length name/model pairing. An empty model list, or a count mismatch, is a
   `MigrationError`.
3. For each model, `_installed_parameters` reads the installed `arena_robots` definition:
   `caps/mobile.yaml` → `radius`, `model_params.yaml` → `mass.base_kg`, plus a finite
   `velocity_limits.angular` as `max_angular_vel` when declared. A missing package, definition,
   radius or mass is a hinted `MigrationError` — **no default radius or mass is ever substituted**.
4. The rows are written to `topics/robot_descriptor.parquet` and `topics/robot_params.parquet`;
   `_synthesize_descriptor` returns the robot/model pairs, and the report lists the fallback.

A recording **with** a fleet gets no synthesis. That is correct in general, but a `legacy_2026_11`
fleet whose parameters are only sensor entries (no `robot_radius`/`robot_mass_kg`) leaves the
recording unprocessable at the metric stage — observed in §1.7.

### 1.6 `migration_report.yaml`

Every non-dry run writes one YAML report (`MigrationReport.as_dict`):

```yaml
benchmark_dir: /path/to/run
dry_run: false
retired:                       # old artifacts moved under legacy/, never deleted
  - /path/to/run/legacy/combined_metrics.parquet
  - /path/to/run/legacy/episodes/episode_000/metrics/fact_episodes.parquet
episodes:
  - episode: episode_000
    generation: legacy_2026_11
    migrated: true
    mapped: []                 # native fields recovered from tm_*
    fallbacks:                 # non-fatal repairs
      - re-extracted the recording into topics/
      - synthesised robot_descriptor/robot_params from RunMetadata.robot_model and ...
    unsupported:               # constructs the generation did not carry
      - AgentMeta has no robot_* fields; the crowd settings are NULL
    error: null
```

`mapped` is `["modes", "params"]` when the metadata carried `tm_*` fields; `fallbacks` names a
re-extraction and a descriptor synthesis; `unsupported` lists the generation's missing constructs;
`error` is the failure message for that episode only.

An episode that fails is recorded and the remaining episodes still migrate. The command exits
non-zero with one line per failed episode (`MigrationReport.failure_message`), and a failed
episode's metadata and tables are left as they were.

### 1.7 Old processed outputs: quarantined, never read

Before migrating, `storage/layout.retire_legacy_outputs` moves the pre-overhaul outputs under
`<benchmark>/legacy/`, never deleting anything:

- benchmark-level legacy files (`manifest.yaml`, `report_manifest.yaml`, `combined_metrics.parquet`, …);
- every `metrics/fact_*.parquet` of an episode that is **not** complete (`metrics/.complete` absent).

The native pipeline reads only the current layout. `process` refuses a run whose tables carry an old
layout stamp with `StaleBenchmarkError`; `--force-process` recomputes it in place:

```bash
evaluation process --benchmark-dir <run> --force-process
```

`progress.csv` is runner state and is **not** migrated; copy it aside or re-record the run.

### 1.8 Limits — what cannot be recovered

- **One MCAP per episode folder.** Continuous recordings split by an `EpisodeRecord` are out of scope
  (D-05); the migration refuses a layout it cannot map.
- **No back-fill of missing measurements.** A topic the recording never wrote has no table; fields the
  generation did not carry stay NULL with a reason.
- **The old processed values are not imported.** They are quarantined and recomputed; numbers can
  change because the metric definitions changed (see §2.6).
- **`legacy_2026_09` EpisodeRecord compatibility is incomplete.** H-012: a September record without
  `human_params` (and probably `conditions`/`phases`) still raises; the merge report marks the real
  September run red.
- **Legacy fleet parameters may be insufficient.** A `legacy_2026_11` fleet that carries only sensor
  parameters produces a descriptor without `robot_radius`/`robot_mass_kg`, and `process` raises
  `MetricInputError`. Observed below.

### 1.9 Executed smoke attempt (H-010 evidence, 2026-11-17)

Command sequence on a **copy** of `$ARENA_DATA_DIR/benchmarks/test-eval3-agent-mcp` (a real 5-episode
Gazebo benchmark recorded 2026-11-08; the original was not modified):

```bash
evaluation migrate --dry-run  /tmp/fix10-smoke     # Would migrate 0 episode folder(s)
evaluation migrate            /tmp/fix10-smoke --report /tmp/fix10-smoke/migration_report.yaml
# Migrated 5 episode folder(s): generation legacy_2026_11, 26 old outputs moved to legacy/
evaluation process --benchmark-dir /tmp/fix10-smoke --force-process --workers 2
evaluation report  --benchmark-dir /tmp/fix10-smoke --report-manifest standard
```

Observed:

- **Migration succeeded** for all five episodes (`legacy_2026_11`, re-extraction fallback, AgentMeta
  crowd fields NULL, old outputs retired).
- **Processing failed** at the benchmark contract with **7 violations**, all duplicate row keys:
  `episode_lifecycle.env_id,t_ns` (2–4 duplicates per episode) and
  `tf.t_ns,parent_frame,child_frame` (2 duplicates). The pre-overhaul pipeline accepted these
  recordings; the native contract (`row_key_duplicate`) rejects the re-extracted duplicates.
- A single episode (`process --run-dir .../episode_000`) failed earlier in evaluation with
  `MetricInputError: robot 'env_0_jackal' has no recorded parameter(s) ['robot_radius', 'robot_mass_kg']`
  — the legacy fleet carried only `lidar@front`/`imu@imu`, and no descriptor was synthesised because
  a fleet was present (§1.5).

So the migration path is exercised and works, but **a real migrated benchmark still does not reach
`process` → `report` → MCP** on the merged tree. This is the honest state of H-010/H-012; owners and
the exact failures are in `FIX10 - report.md`.

---

## 2. Declaration migration

### 2.1 Where declarations live now

| Old | New |
|---|---|
| `configs/benchmark/manifests/<name>.yaml` (old schema) | `configs/reports/<name>/report.yaml` |
| `plot_declarations/<name>.yaml` (old plot schema) | `configs/plots/<name>/plot.yaml` (reusable, `uses: <name>`) |
| Inline `data:` in a manifest | `configs/queries/<name>/query.yaml` (stored query), a canonical view, or a computation |
| `configs/benchmark/{suites,contests}` | `configs/suites/<name>/suite.yaml`, `configs/contests/<name>/contest.yaml` |

The resolver accepts a named bundle, a path to `report.yaml`, or inline `{...}` YAML
(`--report-manifest`). A run-local bundle lives under `<run>/analysis/reports/<name>/report.yaml`.

### 2.2 Old grammar → new grammar

The converter's field mapping (from `presentation/legacy_manifest.py`, reproduced from
`03 Implementation/12 Legacy Report Migration - Map, Required Queries and Converter.md`):

| Old | New |
|---|---|
| `data_source: metrics` | `v_episode_summary` (or `v_series_<stem>` for a series report) |
| `data_key` (a metric) | `data.source`/`params.metrics` + `channels.y` |
| `data_key: timeseries_*` | `series_profile` over `series_*` (scope prefix + unit suffix resolved by unique prefix) |
| `secondary_data_key` / `options.y` | second axis (`scatter`) or second metric |
| `differentiate` | `channels.color` / `channels.x` (old `robot` → `robot_id`) |
| `group_by` / `options.group_by` | `channels.facet_col` or `params.group_cols` |
| `filter` | `data.where` (source plots) or `params.filter` (computations) |
| `auto_differentiate` | dropped — channels are explicit |
| `table` + `options.columns` | `group_summary` + labels/formats |
| `heatmap` + `options.metrics` | `correlation_matrix` |
| `heatmap` + `options.x/y` | `group_summary` pivot (`color: mean`) |
| `radar` + `options.metrics` | `radar_profile` |
| `histogram` | `distribution_bins` |
| `trajectory` (`path`/`pedestrian_path`) | `stored_query: robot_tracks_non_ref` / `pedestrian_arena_peds_non_ref` |
| `acoustic_field*` | recorded drop; `acoustic_field` ships via A-028 |
| notes-only `table` | recorded drop; text preserved under `carried_notes` |

**Old data keys → the new binding.** A declaration now names one of three things instead of a
`data_source`/`data_key` pair:

| You want | Bind to |
|---|---|
| One row per episode/robot, the headline metrics | `v_episode_summary` (canonical view), optionally `data.where` |
| A per-sample series (`series_*`) | `series_profile` over `series_<stem>` (or `timeseries` against the native time axis) |
| A computed table (summary, correlation, ranking, …) | a registered computation (`group_summary`, `distribution_bins`, `radar_profile`, `composite_ranking`, `correlation_matrix`, `paired_diff`, `cluster_bootstrap_ci`, `series_profile`) with `params` |
| A reusable SQL question | `stored_query: <name>` from `configs/queries/<name>/query.yaml` |

The first four are the whole grammar: a plot resolves a view, a stored query, or a computation, and
then binds its declared channels.

### 2.3 The checked-in metric map

`presentation/legacy_metric_map.py` ships the default map (`LEGACY_METRIC_MAP`); `--metric-map` accepts
the same `renamed:`/`removed:` shape. The loader also prefers `configs/legacy/legacy_metric_map.yaml`
if that config kind is ever admitted (today the root admits only the five registered bundle kinds).

`renamed` (old → new):

| old | new |
|---|---|
| `spl` | `spl_geodesic` |
| `path_length` | `path_length_m` |
| `total_energy_wh`, `energy_wh` | `energy_total_wh` |
| `acceleration_max` | `max_acceleration_mps2` |
| `jerk_max` | `jerk_max_mps3` |
| `velocity_mean` | `mean_velocity_mps` |
| `total_contacts` | `contact_count` |
| `time_to_goal_s` | `time_to_goal` |

`removed` (intentional drops, recorded per plot):

| old | decision |
|---|---|
| `social_force_mean` / `_max` / `_ped_mean` / `_ped_max` | D-01/PR2 dropped the stored social-force scalars; social disturbance is `ci_mean`/`ci_max` plus pedestrian deflection |
| `total_sound_energy_joules` | no equivalent registry metric; acoustic exposure is `sound_exposure_dose_pa2s` / the `acoustic_*` scalars — recorded dropped with a TODO |

Other renames the overhaul made (from the fix reports; each is verified in the merged registry):

| old | new |
|---|---|
| `mar` / `pfi` / `kcsc_wh` / `robot_path_deviation_m` | `mar_vs_planned_route` / `pfi_vs_planned_route` / `kcsc_wh_vs_planned_route` / `robot_path_deviation_vs_planned_m` |
| `edge_case_collision_fault` | dropped (closer labels are judgements); the two closing speeds remain |
| `max_speed_mps` | `velocity_max` |
| `peak_power_w` | `power_peak_w` |
| `spl` | `spl_geodesic` |

The alias map that the pipeline itself applies (aligned calculator frame → registry column) is in
`FIX1 - report.md` §“Authoritative old→new rename map”; it is not repeated here because it is
pipeline plumbing, not a declaration rename.

**B-019 status.** The audit asked for a 65-row old→new table generated from the two vocabularies.
That exhaustive table is **not reproducible from the merged tree**: the audit's working map and the
old vocabulary snapshot were never checked in (the former `tests/parity/metric_map.yaml` is gone), and
only the converter's 9 `renamed` entries, 5 `removed` entries and the fix-round renames above are
authored. This guide publishes those; the remaining old metric names are **unverified** rather than
guessed.

### 2.4 Converter CLI

```bash
# plan only, no files written; prints converted/dropped/todo per manifest
python scripts/migrate_legacy_manifests.py configs/benchmark/manifests/social.yaml --dry-run

# write <name>/report.yaml + <name>/migration_report.yaml for every old manifest
python scripts/migrate_legacy_manifests.py configs/benchmark/manifests/*.yaml --output-dir /tmp/migrated

# classify the four legacy plot_declarations files
python scripts/migrate_legacy_manifests.py \
  --legacy-plots plot_declarations/all_metrics.yaml \
  --legacy-plots plot_declarations/eval_iros_2023.yaml \
  --legacy-plots plot_declarations/my_plots.yaml \
  --legacy-plots plot_declarations/sample_schema.yaml

# diff an old manifest against its hand-migrated replacement by plot intent
python scripts/reconcile_legacy_manifests.py old/social.yaml configs/reports/social/report.yaml \
  --out /tmp/reconciliation_social.yaml
```

Every conversion accounts for **every** old plot as `converted`, `dropped` (with a reason) or `todo`
(with the binding still to decide). Nothing is guessed: an unresolved metric or series column stays
`TODO:`.

Measured on the seven legacy manifests (FIX7, `177c3b41`, verified on the merged tree):

| Manifest | Old | Converted | Dropped | TODO |
|---|---:|---:|---:|---:|
| characterization | 27 | 25 | 2 | 0 |
| ecological | 28 | 24 | 3 | 1 |
| everything | 90 | 84 | 4 | 2 |
| failure_modes | 65 | 53 | 12 | 0 |
| safety | 29 | 29 | 0 | 0 |
| social | 31 | 28 | 3 | 0 |
| standard | 68 | 63 | 3 | 2 |
| **total** | **338** | **306** | **27** | **5** |

The five TODOs are two all-metric correlation matrices (the metric list must be chosen) and three
multi-metric power timeseries (one `series_profile` per metric). The seven hand-migrated report
bundles ship **313** declared plots (FIX7 ported 11 clear regressions back from the old declarations;
`297 → 313`).

### 2.5 The four legacy `plot_declarations/` files

Nothing is deleted. The disposition is:

| File | Disposition |
|---|---|
| `all_metrics.yaml` | best-effort conversion / archive — a code-driven schema |
| `eval_iros_2023.yaml` | best-effort conversion / archive — a study declaration whose plots the new grammar can express (bar/line/trajectory/heatmap) |
| `my_plots.yaml` | best-effort conversion / archive |
| `sample_schema.yaml` | archive — unparseable documentation, not a declaration |

### 2.6 MCP-assisted completion loop (FIX8 tools, FIX11 end-to-end)

The converter is exposed through the `evaluation` domain; the tools call the converter, never
re-implement it:

| Step | Tool | What it does |
|---|---|---|
| 1. Migrate | `evaluation_migrate_manifest(source, target=, run_id=, dry_run=, overwrite=)` | Convert one legacy manifest; writes `<run>/analysis/reports/<target>/report.yaml` + `migration_report.yaml` only when `dry_run=False` and a `run_id` is given. |
| 2. Mappings | `evaluation_manifest_mappings(manifest)` | The old→new metric/series mappings the converter applied, plus the unresolved entries. |
| 3. Deep lint | `evaluation_lint_declarations(run_id=, deep=True)` | Executes every plot's view/query/computation and binds its channels against the run; `valid=false` when any plot cannot bind. |
| 4. Render/review | `evaluation_report_draft`, `evaluation_report_plot`, `evaluation_report_view`, `evaluation_report_review` | Render a plot, see the image and its checks, approve or reject it. |
| 5. Assemble | `evaluation_report_assemble` | Build `report.html` from **approved** plots and verified findings only; `complete` reports whether every declared plot was approved. |

The Todo entries in `migration_report.yaml` are the exact work an agent must complete; the loop never
silently drops one. On the synthetic INT1 run the converter produced
`{total: 31, converted: 28, dropped: 3, todo: 0}` for the old social manifest, deep lint bound 24 of
28 plots (the rest need a run that carries the social columns), and the assembler wrote
`report.html` with `complete: false` — the path works, approval needs a real run (H-010, §1.9).

---

## 3. Where to go next

- Recording details and the generation table: `arena_evaluation/ingestion/README.md` §Migration.
- Declarations, plot types and the quality gate: `arena_evaluation/presentation/README.md`.
- Report/query/plot bundle layout: `configs/README.md`.
- Storage layout and the quarantine directory: `arena_evaluation/storage/README.md`.
- The converter's own map and required queries:
  `03 Implementation/12 Legacy Report Migration - Map, Required Queries and Converter.md` and
  `FIX7 - converter usage and stats for FIX8-FIX10.md`.
