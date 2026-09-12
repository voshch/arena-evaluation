from __future__ import annotations

import json
import logging
import pathlib
import subprocess
import traceback
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

import polars as pl
import yaml

try:
    from mcp.types import CallToolRequestParams, CallToolResult, TextContent, Tool
except ImportError:

    class Tool:  # type: ignore
        def __init__(self, name: str, description: str, inputSchema: dict):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema

    class TextContent:  # type: ignore
        def __init__(self, type: str, text: str):
            self.type = type
            self.text = text

    class CallToolResult:  # type: ignore
        def __init__(self, content: list, isError: bool = False):
            self.content = content
            self.isError = isError

    class CallToolRequestParams:  # type: ignore
        pass


from .common import run_status, validate_path_component

if TYPE_CHECKING:
    from .eval_bridge import EvalBridge

logger = logging.getLogger(__name__)


def build_tools_list(bridge: EvalBridge) -> list[Tool]:
    """Return the complete list of registered tools."""

    _manifests = bridge.discover_available_manifests()

    return [
        # Discovery
        Tool(
            name="list_available_maps",
            description="List runnable world/map names (static worlds catalog + generated worlds + "
            "recorded maps). Also returns suite_referenced_maps: names used in existing "
            "suite configs that are NOT verified runnable worlds. Relationship: use "
            "'maps' for new suites; suite_referenced_maps entries absent from 'maps' "
            "may fail at spawn unless generated at build time.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_available_robots",
            description="List all robot models that can be used in suite stages.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_available_planners",
            description="Contest authoring reference: local planner names (nav2 controllers), global planner names, inter planner names, mobile driver names, and cap keys - everything valid inside create_contest.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_available_inter_planners",
            description="List all known inter-planner names found across benchmarks.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_available_task_modes",
            description="List TM_Robots and TM_Obstacles enum values for suite stage configuration.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_available_manifests",
            description="List bundled report manifest stems (standard, social, safety, ecological, characterization).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_available_metrics",
            description="List all registered metric names with their categories and units.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_running_processes",
            description="List arena-related OS processes currently running: benchmark runners, "
            "the arena CLI wrapper, Gazebo simulation, arena/recorder nodes, and "
            "world generators. Each entry has pid, kind, elapsed_s (seconds since "
            "start), and the command line. Use this to check whether a benchmark "
            "is actually running, whether simulation processes are still up, or to "
            "spot stuck/orphaned processes before stopping them.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="kill_processes",
            description="Terminate running arena OS processes (benchmark runner, Gazebo simulation, "
            "arena/recorder nodes, world generators). Sends SIGTERM first with automatic "
            "escalation to SIGKILL (-9) if the process does not terminate within the timeout. "
            "Optionally target specific PIDs, filter by process kind, or pass force=True to "
            "send SIGKILL immediately. Use this to clean up wedged, stuck, or orphaned processes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Specific process IDs to terminate. Default: all running arena processes.",
                    },
                    "force": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, sends SIGKILL immediately without waiting for graceful SIGTERM.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["benchmark_runner", "simulation", "arena_node", "world_generator", "arena_cli"],
                        "description": "Filter by process kind (e.g. 'benchmark_runner' or 'simulation').",
                    },
                },
            },
        ),
        Tool(
            name="get_benchmark_console",
            description="Tail the console log of a benchmark run. Reads the benchmark's own "
            "runner.log inside its run directory (benchmarks/<run_id>/runner.log, "
            "written by the benchmark runner for every run). Omitting run_id uses "
            "the most recently started benchmark runner. Returns the last N lines, "
            "the log file path, the runner PID, and whether the runner is still "
            "alive. Use while a benchmark runs to watch live progress, or after "
            "completion to inspect the output.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run id to tail (e.g. '20260811-031709-hospital_complex-basic'). Default: most recently started running benchmark.",
                    },
                    "lines": {
                        "type": "integer",
                        "default": 200,
                        "description": "Number of tail lines to return.",
                    },
                },
            },
        ),
        Tool(
            name="list_benchmark_runs",
            description="List existing benchmark runs with status metadata. Optional filters: exact suite / contest name, run status, or a case-insensitive substring of the run_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum number of runs to return, most recent first.",
                    },
                    "suite": {
                        "type": "string",
                        "description": "Exact suite name filter (e.g. 'characterization').",
                    },
                    "contest": {
                        "type": "string",
                        "description": "Exact contest name filter (e.g. 'dwb_vs_teb').",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["completed", "in_progress", "unknown"],
                        "description": "Run status filter.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Case-insensitive substring of the run_id (e.g. '20260809' or 'corridor').",
                    },
                },
            },
        ),
        Tool(
            name="list_available_suites",
            description="List bundled benchmark suite config stems (e.g. basic, characterization, all_maps_random). Read one with get_config_template(kind='suite').",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_available_contests",
            description="List bundled contest config stems (e.g. basic, allplanners, drl_vs_nav2). Read one with get_config_template(kind='contest').",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_config_template",
            description="Read the full content of a bundled (or custom) suite / contest / manifest YAML so the agent can see the exact schema conventions. For manifests, custom files written into a benchmark dir are found too.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["suite", "contest", "manifest"],
                        "description": "Which config type to read.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Config stem (e.g. 'basic', 'allplanners', 'standard', 'characterization').",
                    },
                },
                "required": ["kind", "name"],
            },
        ),
        Tool(
            name="inspect_map",
            description="Inspect a map: world bounds, walkable zone polygons (rooms), and the available scenarios with REAL robot start/goal coordinates and pedestrian configs - so spawn points and routes can be chosen that don't intersect walls.",
            inputSchema={
                "type": "object",
                "properties": {
                    "map_name": {
                        "type": "string",
                        "description": "World name (e.g. 'hospital_1', 'office_1', 'map_empty').",
                    },
                },
                "required": ["map_name"],
            },
        ),
        Tool(
            name="describe_task_mode",
            description="Documentation for tm_robots / tm_obstacles task modes: purpose, config "
            "keys (e.g. scenario 'file', random dynamic/static/interactive), complete "
            "YAML examples with concrete values, pedestrian/obstacle model names, and "
            "construction guidance (seed-based reproducible paths, and: do NOT infer "
            "structure from previous benchmark configs). With no mode, returns the "
            "full catalog.",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Optional mode name (e.g. 'scenario', 'guided', 'random'). Omit for the full catalog.",
                    },
                },
            },
        ),
        Tool(
            name="describe_metric",
            description="Details for one metric: category, output keys, units, dependencies, required topics, pedsim requirement, and whether lower is better (for ranking).",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "Metric name (e.g. 'path_metrics', 'energy', 'proxemics').",
                    },
                },
                "required": ["metric_name"],
            },
        ),
        # Configure
        Tool(
            name="create_suite",
            description=(
                "Create a benchmark suite YAML. Pass the complete suite content as a YAML string - "
                "the agent has full control. Suite schema (parsed by Suite.parse()): top-level "
                "'stages' list (required), optional 'references' (bool, default true), optional "
                "'launch' (dict of launch args). Each stage: name, map, robot, episodes, tm_robots, "
                "tm_obstacles, config (dict keyed by mode name), optional timeout. "
                "CONSTRUCTION RULES - construct stage YAML directly from "
                "describe_task_mode (which contains complete schemas with concrete "
                "values); do NOT infer structure from previously existing benchmark "
                "configs, they may use outdated patterns.\n"
                "SAME PATH ACROSS STAGES (compare planners/ped configs on an "
                "identical route): use tm_robots: random and set the SAME explicit "
                "'seed: <int>' on each stage (same map, robot, episode count). "
                "Episode seeds are seed + episode_index. Auto-seeds exclude config "
                "but differ by stage name - always set seed explicitly.\n"
                "PEDESTRIANS: config.random.dynamic.{min, max, models} - bundled "
                "model: ['arenian'] (unknown models fall back to it). Static "
                "obstacles: config.random.static.{min, max, models} (empty models = "
                "all).\n"
                "tm_robots: scenario -> scripted route from config.scenario.file "
                "(inspect_map lists a map's scenarios); tm_robots: guided -> driven "
                "LIVE via set_goal(), not scriptable.\n"
                "Validated via Suite.parse() before writing; unverified map names "
                "produce map_warnings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Suite file stem."},
                    "yaml_content": {"type": "string", "description": "Complete suite YAML."},
                    "location": {
                        "type": "string",
                        "enum": ["install", "source"],
                        "default": "install",
                        "description": "install (default): arena_evaluation install share - immediately visible to the runner. source: the arena_evaluation source tree (versioned in the repo) - needs 'arena build arena_evaluation' to be runnable.",
                    },
                },
                "required": ["name", "yaml_content"],
            },
        ),
        Tool(
            name="create_contest",
            description=(
                "Create a contest YAML defining which planners to compare. List form: YAML sequence "
                "where each entry has 'name' + args as flat dotted keys (mobile.local_planner: teb) "
                "or nested cap dicts (mobile: {driver: nav2, local_planner: teb}). Sweep form: YAML "
                "mapping where top-level keys name capabilities; list-valued keys become "
                "cartesian-product sweep axes; scalar keys are shared constants.\n"
                "ARG NAMESPACES: cap keys are mobile / arm / lift. Valid mobile drivers: nav2, "
                "rosnav_rl, none, external. Inside a cap: local_planner, inter_planner, "
                "global_planner, plus any key from the robot's caps YAML (e.g. velocity limits). "
                "Common passthroughs: task.fail_on_collision: true, complexity: 1|2|3. "
                "See list_available_planners for valid names; get_config_template(kind='contest') "
                "for examples. Validated via Contest.parse() before writing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Contest file stem."},
                    "yaml_content": {"type": "string", "description": "Complete contest YAML."},
                    "location": {
                        "type": "string",
                        "enum": ["install", "source"],
                        "default": "install",
                        "description": "install (default): arena_evaluation install share - immediately visible to the runner. source: the arena_evaluation source tree (versioned in the repo) - needs 'arena build arena_evaluation' to be runnable.",
                    },
                },
                "required": ["name", "yaml_content"],
            },
        ),
        Tool(
            name="create_manifest",
            description=(
                "Create a report manifest YAML. Pass complete manifest content as a YAML string - "
                "the agent has FULL control over every PlotSpec field. Read the bundled examples "
                "first with get_config_template(kind='manifest', name=<standard|social|safety|"
                "ecological|characterization>) to see the exact schema.\n"
                "AGENT CONTENT (all optional):\n"
                "- table plot options.notes: notes.yaml -> renders as a STANDALONE callout "
                "section below the table (never mixed into metric rows).\n"
                "- table plot options.rows: [{label, value}, ...] -> an agent-authored "
                "two-column table, exactly as given (no predefined layout).\n"
                "- ANY plot: options.note: '<inline markdown string>' -> a note box under "
                "that plot; or options.notes_key: '<label>' -> pulls the matching value "
                "from the benchmark's notes.yaml.\n"
                "- summary_group_by: [local_planner] -> controls the Performance Summary "
                "grouping (defaults to planner when omitted)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Manifest file stem."},
                    "yaml_content": {"type": "string", "description": "Complete manifest YAML."},
                    "template": {
                        "type": "string",
                        "enum": _manifests,
                        "description": "Use a bundled manifest as base.",
                    },
                    "benchmark_id": {
                        "type": "string",
                        "description": "Benchmark run ID (for default output location).",
                    },
                },
                "required": ["name", "benchmark_id"],
            },
        ),
        Tool(
            name="validate_suite",
            description="Validate a suite YAML string against the Suite schema.",
            inputSchema={
                "type": "object",
                "properties": {
                    "yaml_content": {"type": "string"},
                },
                "required": ["yaml_content"],
            },
        ),
        Tool(
            name="validate_contest",
            description="Validate a contest YAML string against the Contest schema.",
            inputSchema={
                "type": "object",
                "properties": {
                    "yaml_content": {"type": "string"},
                },
                "required": ["yaml_content"],
            },
        ),
        Tool(
            name="validate_manifest",
            description="Validate a manifest YAML string against the VizManifest schema.",
            inputSchema={
                "type": "object",
                "properties": {
                    "yaml_content": {"type": "string"},
                },
                "required": ["yaml_content"],
            },
        ),
        Tool(
            name="create_scenario",
            description=("Create or update a scenario YAML for a map. Pass map_name (e.g. 'hospital_1'), scenario_name (e.g. 's01_head_on'), and the complete scenario yaml_content. Validates robot goals and dynamic obstacles before writing. Writes directly to the arena_simulation_setup scenarios directory."),
            inputSchema={
                "type": "object",
                "properties": {
                    "map_name": {"type": "string", "description": "World name (e.g. 'hospital_1')."},
                    "scenario_name": {"type": "string", "description": "Scenario directory/stem name."},
                    "yaml_content": {"type": "string", "description": "Complete scenario YAML content."},
                    "location": {
                        "type": "string",
                        "enum": ["both", "install", "source"],
                        "default": "both",
                        "description": "Where to write: both (default, immediately runnable & tracked in repo), install (share dir only), or source (repo only).",
                    },
                },
                "required": ["map_name", "scenario_name", "yaml_content"],
            },
        ),
        Tool(
            name="validate_scenario",
            description=("Validate a scenario YAML string against the Scenario schema. Optionally pass map_name to check coordinates against world bounds."),
            inputSchema={
                "type": "object",
                "properties": {
                    "yaml_content": {"type": "string", "description": "Complete scenario YAML content."},
                    "map_name": {"type": "string", "description": "Optional world name to validate bounds against."},
                },
                "required": ["yaml_content"],
            },
        ),
        # Execute
        Tool(
            name="run_benchmark",
            description=(
                "Start a benchmark simulation in the BACKGROUND and return immediately with "
                "the run_id. Spawns 'arena evaluation benchmark --suite <S> --contest <C>' "
                "with the recommended launch configuration by default: sim:=gazebo, "
                "headless:=true, env.n:=2, optim.obstacles:=bbox. All are overridable, and "
                "arbitrary extra passthrough args (e.g. task.fail_on_collision:=true) can be "
                "passed via extra_passthrough. Use read_benchmark_status(run_id) to poll."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "suite": {"type": "string", "description": "Suite name stem."},
                    "contest": {"type": "string", "description": "Contest name stem."},
                    "scale_episodes": {
                        "type": "number",
                        "default": 1.0,
                        "description": "Scale factor for episode counts.",
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Custom run ID. Auto-generated if omitted.",
                    },
                    "sim": {
                        "type": "string",
                        "enum": ["gazebo", "isaac", "dummy"],
                        "default": "gazebo",
                        "description": "Simulator backend (sim:=...). Defaults to gazebo.",
                    },
                    "headless": {
                        "type": "boolean",
                        "default": True,
                        "description": "Launch the simulation headless (no GUI). Defaults to true.",
                    },
                    "env_n": {
                        "type": "integer",
                        "default": 2,
                        "description": "Number of parallel envs (env.n:=...). Defaults to 2.",
                    },
                    "optim_obstacles": {
                        "type": "string",
                        "enum": ["full", "bbox", "none"],
                        "default": "bbox",
                        "description": "Obstacle optimization level (optim.obstacles:=...). Defaults to bbox.",
                    },
                    "extra_passthrough": {
                        "type": "object",
                        "description": "Additional launch args as {key: value} - each becomes key:=value, e.g. {\"task.fail_on_collision\": true} -> task.fail_on_collision:=true.",
                    },
                },
                "required": ["suite", "contest"],
            },
        ),
        Tool(
            name="run_processing",
            description="Process benchmark data: extract topics from MCAPs, compute all metrics -> combined_metrics.parquet. Does NOT generate a report.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                    "force_extract": {"type": "boolean", "default": False},
                    "workers": {"type": "integer", "default": -1},
                },
                "required": ["benchmark_id"],
            },
        ),
        Tool(
            name="run_report",
            description="Render report.html + plots/*.png from existing metrics using a report manifest.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                    "report_manifest": {
                        "type": "string",
                        "description": "Manifest name or path (e.g. 'standard' or 'my_custom.yaml').",
                    },
                },
                "required": ["benchmark_id"],
            },
        ),
        # Read
        Tool(
            name="read_benchmark_status",
            description="Read manifest.yaml + .benchmark_state.json + progress.csv for a benchmark.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                },
                "required": ["benchmark_id"],
            },
        ),
        Tool(
            name="read_combined_metrics",
            description="Load combined_metrics.parquet and return schema, shape, and column-level statistics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                },
                "required": ["benchmark_id"],
            },
        ),
        Tool(
            name="read_episode_data",
            description="Read a specific episode's YAML metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                    "episode_id": {
                        "type": "integer",
                        "description": "Episode number (0-based).",
                    },
                },
                "required": ["benchmark_id", "episode_id"],
            },
        ),
        Tool(
            name="read_progress_csv",
            description="Read progress.csv for a benchmark run as structured rows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                },
                "required": ["benchmark_id"],
            },
        ),
        Tool(
            name="read_notes",
            description="Read notes.yaml from a benchmark directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                },
                "required": ["benchmark_id"],
            },
        ),
        Tool(
            name="query_metrics_data",
            description="Execute a declarative aggregation query against combined_metrics.parquet.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column names to group by.",
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metric columns to aggregate (mean).",
                    },
                    "filter": {
                        "type": "object",
                        "description": "Equality filter: {column: value} or {column: [v1, v2]}.",
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Metric column to sort results by, descending.",
                    },
                },
                "required": ["benchmark_id"],
            },
        ),
        # Analyze
        Tool(
            name="compare_planners",
            description="Compare planners on metrics and return ranked comparison with normalized scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                    "planners": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of planner names to compare.",
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metric columns to compare.",
                    },
                    "group_by_stage": {
                        "type": "boolean",
                        "default": False,
                        "description": "Break down comparison by stage.",
                    },
                    "normalize": {
                        "type": "boolean",
                        "default": True,
                        "description": "Return normalized ranks (0-1).",
                    },
                },
                "required": ["benchmark_id"],
            },
        ),
        Tool(
            name="compute_correlation",
            description="Compute pairwise Pearson correlation matrix for specified metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metric columns to correlate.",
                    },
                },
                "required": ["benchmark_id", "metrics"],
            },
        ),
        Tool(
            name="find_top_n",
            description="Find top N planners by a weighted composite score across metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                    "n": {"type": "integer", "default": 3},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metric columns for composite score.",
                    },
                    "weights": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Weights for each metric. Omit for equal weights.",
                    },
                },
                "required": ["benchmark_id"],
            },
        ),
        Tool(
            name="aggregate_by_dimension",
            description="Group metrics by a dimension (planner, stage, map, robot) and aggregate.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                    "dimension": {"type": "string"},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metric columns to aggregate (mean).",
                    },
                },
                "required": ["benchmark_id", "dimension"],
            },
        ),
        Tool(
            name="summarize_benchmark",
            description="Generate a comprehensive text summary of a benchmark's results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                },
                "required": ["benchmark_id"],
            },
        ),
        # Write Notes
        Tool(
            name="write_notes",
            description="Write or update notes.yaml in a benchmark directory. Notes feed into table-type plots.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                    "notes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "value": {"type": "string"},
                            },
                            "required": ["label", "value"],
                        },
                        "description": "Structured notes as [{label, value}, ...].",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "append", "merge"],
                        "default": "replace",
                    },
                },
                "required": ["benchmark_id"],
            },
        ),
        Tool(
            name="append_insight",
            description="Append a single key:value insight to notes.yaml.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark_id": {"type": "string", "description": "The benchmark run ID."},
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["benchmark_id", "label", "value"],
            },
        ),
    ]


async def dispatch_tool_call(params: CallToolRequestParams, bridge: EvalBridge) -> CallToolResult:
    """Route a tool call request to its handler and return the result."""
    name = params.name
    args = params.arguments or {}

    try:
        result = _dispatch(name, args, bridge)
    except ValueError as exc:
        result = {"error": str(exc)}
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        result = {"error": str(exc), "traceback": traceback.format_exc()}

    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, default=str))],
        isError=isinstance(result, dict) and "error" in result,
    )


def _dispatch(name: str, args: dict[str, Any], bridge: EvalBridge) -> dict[str, Any]:
    """Route a tool call to its handler function."""

    # Discovery
    if name == "list_available_maps":
        return {
            "maps": bridge.discover_available_maps(),
            "suite_referenced_maps": bridge.discover_suite_map_names(),
            "note": "maps = verified runnable worlds (sim worlds catalog + generated "
            "worlds + recorded). suite_referenced_maps = names used in "
            "existing suite configs; entries NOT in maps (e.g. "
            "arena_corridor) exist only in suite YAMLs and may fail at "
            "spawn unless generated at build time. Use maps for new "
            "suites; treat suite_referenced_maps as unverified.",
        }

    if name == "list_available_robots":
        return {"robots": bridge.discover_available_robots()}

    if name == "list_available_planners":
        catalog = bridge.planner_catalog()
        return {
            "local_planners": catalog["local_planners"],
            "global_planners": catalog["global_planners"],
            "inter_planners": catalog["inter_planners"],
            "drivers": catalog["drivers"],
            "cap_keys": catalog["cap_keys"],
            "planners_from_contests": bridge.discover_available_planners(),
        }

    if name == "list_available_inter_planners":
        return {"inter_planners": bridge.discover_available_inter_planners()}

    if name == "list_available_suites":
        return {"suites": bridge.list_suite_stems()}

    if name == "list_available_contests":
        return {"contests": bridge.list_contest_stems()}

    if name == "get_config_template":
        kind = args.get("kind", "")
        name = args.get("name", "")
        if kind not in ("suite", "contest", "manifest"):
            return {"error": "kind must be one of: suite, contest, manifest"}
        doc = bridge.read_config_template(kind, name)
        if doc is None:
            return {"error": f"no {kind} config named '{name}' found"}
        return doc

    if name == "inspect_map":
        return bridge.inspect_map(args.get("map_name", ""))

    if name == "describe_task_mode":
        return bridge.describe_task_mode(args.get("mode"))

    if name == "describe_metric":
        return _describe_metric(args, bridge)

    if name == "list_available_task_modes":
        return bridge.discover_task_modes()

    if name == "list_available_manifests":
        return {"manifests": bridge.discover_available_manifests()}

    if name == "list_available_metrics":
        return {"metrics": bridge.discover_available_metrics()}

    if name == "list_running_processes":
        return {"processes": bridge.running_processes()}

    if name == "kill_processes":
        return {
            "results": bridge.kill_processes(
                pids=args.get("pids"),
                force=bool(args.get("force", False)),
                kind=args.get("kind"),
            )
        }

    if name == "get_benchmark_console":
        return bridge.tail_console(
            run_id=args.get("run_id"),
            lines=int(args.get("lines", 200)),
        )

    if name == "list_benchmark_runs":
        limit = int(args.get("limit", 50))
        runs = bridge.list_benchmarks(
            suite=args.get("suite"),
            contest=args.get("contest"),
            status=args.get("status"),
            query=args.get("query"),
        )
        return {"runs": runs[:limit], "n_matches": len(runs)}

    # Configure
    if name == "create_suite":
        if args["name"] in bridge.list_suite_stems():
            return {"suite_valid": False, "error": f"suite '{args['name']}' already exists as a bundled config - pick a new name to avoid clobbering it"}
        location = args.get("location", "install")
        result = _create_config(
            args,
            bridge,
            "suite",
            _validate_suite,
            lambda n: bridge.suite_path(n, location=location),
        )
        if result.get("suite_valid"):
            result["map_warnings"] = _warn_unknown_maps(args["yaml_content"], bridge)
            result["model_warnings"] = _warn_unknown_models(args["yaml_content"], bridge)
        return result

    if name == "create_contest":
        if args["name"] in bridge.list_contest_stems():
            return {"contest_valid": False, "error": f"contest '{args['name']}' already exists as a bundled config - pick a new name to avoid clobbering it"}
        location = args.get("location", "install")
        return _create_config(
            args,
            bridge,
            "contest",
            _validate_contest,
            lambda n: bridge.contest_path(n, location=location),
        )

    if name == "create_manifest":
        return _create_manifest(args, bridge)

    if name == "validate_suite":
        result = _validate_suite(args.get("yaml_content", ""))
        if result.get("valid"):
            result["model_warnings"] = _warn_unknown_models(args.get("yaml_content", ""), bridge)
        return result

    if name == "validate_contest":
        return _validate_contest(args.get("yaml_content", ""))

    if name == "validate_manifest":
        return _validate_manifest(args.get("yaml_content", ""))

    if name == "create_scenario":
        return _create_scenario(args, bridge)

    if name == "validate_scenario":
        return _validate_scenario(
            args.get("yaml_content", ""),
            map_name=args.get("map_name"),
            bridge=bridge,
        )

    # Execute
    if name == "run_benchmark":
        return _run_benchmark(args, bridge)

    if name == "run_processing":
        return _run_processing(args, bridge)

    if name == "run_report":
        return _run_report(args, bridge)

    # Read
    if name == "read_benchmark_status":
        return _read_benchmark_status(args, bridge)

    if name == "read_combined_metrics":
        return _read_combined_metrics(args, bridge)

    if name == "read_episode_data":
        return _read_episode_data(args, bridge)

    if name == "read_progress_csv":
        return _read_progress_csv(args, bridge)

    if name == "read_notes":
        return _read_notes(args, bridge)

    if name == "query_metrics_data":
        return _query_metrics_data(args, bridge)

    # Analyze
    if name == "compare_planners":
        return _compare_planners(args, bridge)

    if name == "compute_correlation":
        return _compute_correlation(args, bridge)

    if name == "find_top_n":
        return _find_top_n(args, bridge)

    if name == "aggregate_by_dimension":
        return _aggregate_by_dimension(args, bridge)

    if name == "summarize_benchmark":
        return _summarize_benchmark(args, bridge)

    # Write Notes
    if name == "write_notes":
        return _write_notes(args, bridge)

    if name == "append_insight":
        return _append_insight(args, bridge)

    return {"error": f"unknown tool: {name}"}


def _validate_suite(yaml_content: str) -> dict:
    if not yaml_content.strip():
        return {"valid": False, "error": "Empty YAML content"}
    try:
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            return {"valid": False, "error": f"Expected a mapping, got {type(data).__name__}"}
        from arena_evaluation.benchmark.config import Suite

        Suite.parse("_validate", data)
        n_stages = len(data.get("stages", []))
        return {"valid": True, "n_stages": n_stages}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def _validate_contest(yaml_content: str) -> dict:
    if not yaml_content.strip():
        return {"valid": False, "error": "Empty YAML content"}
    try:
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, (list, dict)):
            return {"valid": False, "error": f"Expected list or mapping, got {type(data).__name__}"}
        from arena_evaluation.benchmark.config import Contest

        contest = Contest.parse("_validate", data)
        return {"valid": True, "n_contestants": len(contest.contestants), "contestants": [c.name for c in contest.contestants]}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def _validate_manifest(yaml_content: str) -> dict:
    if not yaml_content.strip():
        return {"valid": False, "error": "Empty YAML content"}
    try:
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            return {"valid": False, "error": f"Expected a mapping, got {type(data).__name__}"}
        from arena_evaluation.presentation.viz_manifest import VizManifest

        vm = VizManifest.model_validate(data)
        return {"valid": True, "n_plots": len(vm.plots), "n_groups": len(vm.groups)}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def _validate_scenario(
    yaml_content: str,
    map_name: str | None = None,
    bridge: EvalBridge | None = None,
) -> dict:
    if not yaml_content.strip():
        return {"valid": False, "error": "Empty YAML content"}
    try:
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            return {"valid": False, "error": f"Expected a mapping, got {type(data).__name__}"}

        robots = data.get("robots", [])
        if not isinstance(robots, list):
            return {"valid": False, "error": "'robots' must be a list"}
        for idx, r in enumerate(robots):
            if not isinstance(r, dict):
                return {"valid": False, "error": f"robots[{idx}] must be a dict, got {type(r).__name__}"}
            if "start" not in r and "start_pos" not in r:
                return {"valid": False, "error": f"robots[{idx}] missing required 'start' (or 'start_pos') field"}
            if "phases" not in r and "goal" not in r:
                return {"valid": False, "error": f"robots[{idx}] missing 'phases' or 'goal'"}

        dynamic = data.get("dynamic", [])
        if not isinstance(dynamic, list):
            return {"valid": False, "error": "'dynamic' must be a list"}
        for idx, d in enumerate(dynamic):
            if not isinstance(d, dict):
                return {"valid": False, "error": f"dynamic[{idx}] must be a dict, got {type(d).__name__}"}
            if "pose" not in d and "position" not in d:
                return {"valid": False, "error": f"dynamic[{idx}] missing 'pose' or 'position'"}

        warnings: list[str] = []
        if map_name and bridge is not None:
            try:
                info = bridge.inspect_map(map_name)
                mb = info.get("map_bounds")
                if mb and "x" in mb and "y" in mb:
                    min_x, max_x = mb["x"]
                    min_y, max_y = mb["y"]

                    def _check_pt(pt: object, label: str) -> None:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            x, y = float(pt[0]), float(pt[1])
                            if not (min_x - 1.0 <= x <= max_x + 1.0) or not (min_y - 1.0 <= y <= max_y + 1.0):
                                warnings.append(f"{label} ({x}, {y}) is outside map bounds X:[{min_x}, {max_x}], Y:[{min_y}, {max_y}]")

                    for idx, r in enumerate(robots):
                        _check_pt(r.get("start"), f"robots[{idx}].start")
                        for pidx, p in enumerate(r.get("phases") or []):
                            if isinstance(p, dict) and "goto" in p:
                                _check_pt(p["goto"], f"robots[{idx}].phases[{pidx}].goto")

                    for idx, d in enumerate(dynamic):
                        _check_pt(d.get("pose"), f"dynamic[{idx}].pose")
                        for widx, wp in enumerate(d.get("waypoints") or []):
                            _check_pt(wp, f"dynamic[{idx}].waypoints[{widx}]")
            except Exception as exc:
                logger.warning("bounds check failed for map %s: %s", map_name, exc)

        return {
            "valid": True,
            "n_robots": len(robots),
            "n_dynamic_peds": len(dynamic),
            "n_static": len(data.get("static", [])),
            "n_regions": len(data.get("regions", {})),
            "conditions": len(data.get("conditions", [])),
            "timeline": len(data.get("timeline", [])),
            "warnings": warnings,
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def _create_scenario(args: dict, bridge: EvalBridge) -> dict:
    map_name = args.get("map_name", "")
    scenario_name = args.get("scenario_name", "")
    yaml_content = args.get("yaml_content", "")
    location = args.get("location", "both")

    validation = _validate_scenario(yaml_content, map_name=map_name, bridge=bridge)
    if not validation.get("valid"):
        return {"scenario_valid": False, "validation": validation}

    try:
        targets = bridge.scenario_write_targets(map_name, scenario_name, location=location)
    except Exception as exc:
        return {"scenario_valid": False, "error": f"Failed resolving target path: {exc}"}

    if not targets:
        return {
            "scenario_valid": False,
            "error": f"Could not resolve scenario write path for map '{map_name}' and scenario '{scenario_name}'",
        }

    written: list[str] = []
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml_content)
            written.append(str(target))
        except OSError as exc:
            logger.error("failed writing scenario to %s: %s", target, exc)

    if not written:
        return {
            "scenario_valid": False,
            "error": f"Failed writing scenario to all resolved targets: {targets}",
        }

    return {
        "scenario_valid": True,
        "scenario_name": scenario_name,
        "map_name": map_name,
        "written_paths": written,
        **validation,
    }


def _warn_unknown_models(yaml_content: str, bridge: EvalBridge) -> list[str]:
    """Non-blocking warnings for model names outside the bundled catalogs."""
    warnings: list[str] = []
    try:
        data = yaml.safe_load(yaml_content)
        stages = (data or {}).get("stages", []) or []
    except (yaml.YAMLError, AttributeError) as exc:
        logger.warning("cannot scan suite YAML for model warnings: %s", exc)
        return warnings
    peds = set(bridge.pedestrian_models().get("bundled", []))
    objs = set(bridge.static_object_models().get("bundled", []))
    for stage in stages:
        stage_name = (stage or {}).get("name", "?")
        cfg = (stage or {}).get("config") or {}
        rnd = cfg.get("random") or {}
        checks = (
            ("dynamic", peds, "pedestrian - the simulator silently falls back to 'arenian'"),
            ("static", objs, "static object - unknown models may fail to spawn"),
            ("interactive", objs, "interactive object - unknown models may fail to spawn"),
        )
        for key, catalog, what in checks:
            models = (rnd.get(key) or {}).get("models") or []
            for m in models:
                if m not in catalog:
                    warnings.append(f"stage '{stage_name}': {key} model '{m}' is not in the bundled catalog {sorted(catalog)} - {what}. Fetch extra models via `arena_models net fetch` or fix the typo.")
    return warnings


def _warn_unknown_maps(yaml_content: str, bridge: EvalBridge) -> list[str]:
    """Non-blocking warning when a stage references an unverified map name."""
    known = set(bridge.discover_available_maps())
    warnings: list[str] = []
    try:
        data = yaml.safe_load(yaml_content)
        for stage in (data or {}).get("stages", []) or []:
            m = (stage or {}).get("map")
            if m and m not in known:
                warnings.append(f"stage '{stage.get('name', '?')}': map '{m}' is not in the known worlds catalog - may fail at spawn unless generated at build time")
    except (yaml.YAMLError, AttributeError) as exc:
        logger.warning("cannot scan suite YAML for map warnings: %s", exc)
    return warnings


def _create_config(
    args: dict,
    bridge: EvalBridge,
    kind: str,
    validator: Callable[[str], dict],
    path_resolver: Callable[[str], pathlib.Path],
) -> dict:
    name = args["name"]
    yaml_content = args["yaml_content"]
    validation = validator(yaml_content)
    if not validation.get("valid"):
        return {f"{kind}_valid": False, "validation": validation}

    target = path_resolver(name if not name.endswith(".yaml") else name[:-5])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml_content)
    return {f"{kind}_path": str(target), f"{kind}_valid": True, **validation}


def _create_manifest(args: dict, bridge: EvalBridge) -> dict:
    name = args["name"]
    benchmark_id = args.get("benchmark_id", "")
    yaml_content = args.get("yaml_content", "")
    template = args.get("template", "")

    if template and not yaml_content:
        try:
            from arena_evaluation.presentation.manifest_registry import resolve_manifest

            vm = resolve_manifest(template, benchmark_dir=None)
            vm.name = name
            yaml_content = yaml.safe_dump(vm.model_dump(exclude_none=True), sort_keys=False)
        except Exception as exc:
            return {"valid": False, "error": f"Failed to load template '{template}': {exc}"}

    if not yaml_content.strip():
        return {"valid": False, "error": "Provide yaml_content or template"}

    validation = _validate_manifest(yaml_content)
    if not validation.get("valid"):
        return {"manifest_valid": False, "validation": validation}

    validate_path_component(name)
    target = bridge.benchmark_dir(benchmark_id) / f"{name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml_content)
    return {"manifest_path": str(target), "manifest_valid": True, **validation}


def _benchmark_cmd_args(args: dict) -> list[str]:
    """Build the arena evaluation benchmark passthrough args."""
    cmd_args = ["benchmark", "--suite", args["suite"], "--contest", args["contest"]]
    scale = args.get("scale_episodes", 1.0)
    if scale != 1.0:
        cmd_args.extend(["--scale-episodes", str(scale)])
    if args.get("run_id"):
        cmd_args.extend(["--run-id", args["run_id"]])

    cmd_args.append(f"sim:={args.get('sim', 'gazebo')}")
    cmd_args.append(f"headless:={str(bool(args.get('headless', True))).lower()}")
    cmd_args.append(f"env.n:={int(args.get('env_n', 2))}")
    cmd_args.append(f"optim.obstacles:={args.get('optim_obstacles', 'bbox')}")

    for key, value in (args.get("extra_passthrough") or {}).items():
        cmd_args.append(f"{key}:={value}")

    return cmd_args


def _run_benchmark(args: dict, bridge: EvalBridge) -> dict:
    suite = validate_path_component(args["suite"])
    contest = validate_path_component(args["contest"])
    run_id = args.get("run_id", "")
    if run_id:
        validate_path_component(run_id)

    cmd_args = _benchmark_cmd_args(args)

    if not run_id:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{ts}-{suite}-{contest}"

    proc = bridge.run_cli_background(*cmd_args)
    bridge._bg_processes[run_id] = proc

    launch_config = {a.split(":=", 1)[0]: a.split(":=", 1)[1] for a in cmd_args if ":=" in a}
    return {
        "run_id": run_id,
        "status": "running",
        "suite": suite,
        "contest": contest,
        "launch_config": launch_config,
        "output_dir": str(bridge.benchmark_dir(run_id)),
        "pid": proc.pid,
    }


def _run_processing(args: dict, bridge: EvalBridge) -> dict:
    """Run arena evaluation process (via the sourced CLI wrapper)."""
    benchmark_id = validate_path_component(args["benchmark_id"])
    cmd_args = ["process", "--benchmark-dir", benchmark_id]
    if args.get("force_extract"):
        cmd_args.append("--force-extract")
    workers = args.get("workers", -1)
    if workers != -1:
        cmd_args.extend(["--workers", str(workers)])

    try:
        result = bridge.run_cli(*cmd_args, timeout=1800)
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "Processing timed out after 30 minutes"}


def _run_report(args: dict, bridge: EvalBridge) -> dict:
    benchmark_id = validate_path_component(args["benchmark_id"])
    cmd_args = ["report", "--benchmark-dir", benchmark_id]
    manifest = args.get("report_manifest", "")
    if manifest:
        stem = validate_path_component(manifest[:-5] if manifest.endswith(".yaml") else manifest)
        cand = bridge.benchmark_dir(benchmark_id) / f"{stem}.yaml"
        cmd_args.extend(["--report-manifest", str(cand) if cand.is_file() else stem])

    try:
        result = bridge.run_cli(*cmd_args, timeout=600)
        report_path = str(bridge.benchmark_dir(benchmark_id) / "report.html")
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "report_path": report_path,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "Report generation timed out"}


def _read_benchmark_status(args: dict, bridge: EvalBridge) -> dict:
    bid = args["benchmark_id"]
    manifest = bridge.read_benchmark_manifest(bid)
    state = bridge.read_benchmark_state(bid)
    progress = bridge.read_progress_csv(bid)

    steps = state.get("steps", {}) if state else {}
    statuses = [s.get("status") for s in steps.values()]
    n_episodes = len(progress) if progress else 0

    return {
        "run_id": (manifest or {}).get("run_id", bid),
        "suite": (manifest or {}).get("suite_name", ""),
        "contest": (manifest or {}).get("contest_name", ""),
        "status": run_status(statuses),
        "steps_total": len(steps),
        "steps_ok": statuses.count("ok"),
        "steps_failed": statuses.count("failed"),
        "steps_partial": statuses.count("partial"),
        "steps_in_progress": statuses.count("in_progress"),
        "n_episodes": n_episodes,
        "has_combined_metrics": (bridge.benchmark_dir(bid) / "combined_metrics.parquet").exists(),
        "has_report": (bridge.benchmark_dir(bid) / "report.html").exists(),
    }


def _read_combined_metrics(args: dict, bridge: EvalBridge) -> dict:
    bid = args["benchmark_id"]
    df = bridge.load_combined_metrics(bid)
    if df is None:
        return {"error": f"No combined_metrics.parquet found for {bid}"}

    stats = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].null_count()
        if dtype.startswith("Float") or dtype.startswith("Int"):
            s = df[col].drop_nulls()
            if len(s) > 0:
                stats[col] = {
                    "dtype": dtype,
                    "null_count": null_count,
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                }
        elif dtype == "List":
            stats[col] = {"dtype": "List", "null_count": null_count}
        else:
            unique = df[col].drop_nulls().unique().to_list()
            stats[col] = {
                "dtype": dtype,
                "null_count": null_count,
                "unique_values": unique[:20],
                "n_unique": len(unique),
            }

    return {
        "benchmark_id": bid,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": df.columns,
        "column_stats": stats,
    }


def _read_episode_data(args: dict, bridge: EvalBridge) -> dict:
    bid = args["benchmark_id"]
    ep_id = int(args.get("episode_id", 0))
    ep = bridge.fm.episode_dir(bid, ep_id)
    yaml_path = ep / f"episode_{ep_id:03d}.yaml"
    data = bridge.read_yaml(yaml_path)
    if data is None:
        yaml_path = ep / "metadata.yaml"
        data = bridge.read_yaml(yaml_path)
    if data is None:
        return {"error": f"No YAML found for episode {ep_id} in {bid}"}
    return {"episode_id": ep_id, "benchmark_id": bid, "metadata": data}


def _read_progress_csv(args: dict, bridge: EvalBridge) -> dict:
    bid = args["benchmark_id"]
    rows = bridge.read_progress_csv(bid)
    if rows is None:
        return {"error": f"No progress.csv found for {bid}"}
    return {"benchmark_id": bid, "n_episodes": len(rows), "rows": rows}


def _read_notes(args: dict, bridge: EvalBridge) -> dict:
    bid = args["benchmark_id"]
    notes = bridge.read_notes(bid)
    return {"benchmark_id": bid, "notes": notes}


def _query_metrics_data(args: dict, bridge: EvalBridge) -> dict:
    bid = args["benchmark_id"]
    df = bridge.load_combined_metrics(bid)
    if df is None:
        return {"error": f"No combined_metrics.parquet found for {bid}"}

    group_by = args.get("group_by") or []
    metrics = args.get("metrics") or []
    filter_obj = args.get("filter") or {}
    sort_by = args.get("sort_by", "")

    if filter_obj:
        for col, val in filter_obj.items():
            if col in df.columns:
                if isinstance(val, list):
                    df = df.filter(pl.col(col).is_in(val))
                else:
                    df = df.filter(pl.col(col) == val)

    if not group_by or not metrics:
        return {"error": "group_by and metrics are required"}

    group_cols = [g for g in group_by if g in df.columns]
    metric_cols = [m for m in metrics if m in df.columns]
    if not group_cols or not metric_cols:
        return {"error": "No valid group_by or metric columns found", "available_columns": df.columns}

    needed = list(dict.fromkeys([*group_cols, *metric_cols, *([sort_by] if sort_by and sort_by in df.columns else [])]))
    list_cols = [c for c in needed if c in df.columns and df.schema[c] == pl.List]
    if list_cols:
        df = df.select(needed).explode(list_cols)
    else:
        df = df.select(needed)

    agg = [pl.col(m).mean().alias(m) for m in metric_cols]
    result = df.group_by(group_cols).agg(agg).sort(group_cols)

    if sort_by and sort_by in metric_cols:
        result = result.sort(sort_by, descending=True)

    return {
        "benchmark_id": bid,
        "n_rows": len(result),
        "columns": result.columns,
        "data": result.to_dicts(),
    }


def _metric_declarations(bridge: EvalBridge) -> dict[str, dict]:
    """Metric-declared metadata: primary_outputs + output_directions.

    The metrics declare these themselves (BaseMetricCalculator
    PRIMARY_OUTPUTS / OUTPUT_DIRECTIONS) - analysis tooling must not
    hardcode metric knowledge.
    """
    out: dict[str, dict] = {}
    for m in bridge.discover_available_metrics():
        out[m["name"]] = {
            "primary_outputs": m.get("primary_outputs", []),
            "output_directions": m.get("output_directions", {}),
        }
    return out


def _default_compare_metrics(bridge: EvalBridge) -> list[str]:
    """Default comparison metrics = the metrics' declared PRIMARY_OUTPUTS."""
    seen: list[str] = []
    for decl in _metric_declarations(bridge).values():
        for k in decl["primary_outputs"]:
            if k not in seen:
                seen.append(k)
    return seen


def _is_lower_better(metric: str, declarations: dict[str, dict]) -> bool:
    """Direction for a metric output key, declared by its calculator.

    Unlisted keys default to lower (most cost metrics are lower-better).
    """
    for decl in declarations.values():
        direction = decl["output_directions"].get(metric)
        if direction is not None:
            return direction != "higher"
    return True


def _describe_metric(args: dict, bridge: EvalBridge) -> dict:
    name = args.get("metric_name", "")
    metrics = bridge.discover_available_metrics()
    m = next((x for x in metrics if x["name"] == name), None)
    if m is None:
        return {
            "error": f"unknown metric '{name}'",
            "available": [x["name"] for x in metrics],
        }
    from arena_evaluation.processing.metrics.registry import MetricRegistry

    MetricRegistry.discover_calculators_cls()
    units = MetricRegistry.get_all_units()
    declarations = _metric_declarations(bridge)
    return {
        "name": m["name"],
        "category": m["category"],
        "requires_pedsim": m["requires_pedsim"],
        "depends_on": m["depends_on"],
        "required_topics": m["required_topics"],
        "primary_outputs": m.get("primary_outputs", []),
        "outputs": {
            k: {
                "unit": units.get(k),
                "lower_is_better": _is_lower_better(k, declarations),
            }
            for k in m["outputs"]
        },
        "lower_is_better": all(_is_lower_better(k, declarations) for k in m["outputs"]),
    }


def _compare_planners_frame(
    df: pl.DataFrame,
    metrics: list[str],
    declarations: dict[str, dict],
    planners: list[str] | None = None,
    group_by_stage: bool = False,
    normalize: bool = True,
) -> dict:
    planner_col = "local_planner" if "local_planner" in df.columns else "planner"

    if planners:
        df = df.filter(pl.col(planner_col).is_in(planners))

    metric_cols = [m for m in metrics if m in df.columns]
    if not metric_cols:
        return {"error": "No valid metric columns found", "available": df.columns}

    group_cols = [planner_col]
    if group_by_stage and "stage" in df.columns:
        group_cols.append("stage")

    needed = list(dict.fromkeys([*group_cols, *metric_cols]))
    list_cols = [c for c in needed if c in df.columns and df.schema[c] == pl.List]
    if list_cols:
        df = df.select(needed).explode(list_cols)
    else:
        df = df.select(needed)

    agg = [pl.col(m).mean().alias(m) for m in metric_cols]
    agg += [pl.col(m).std().alias(f"{m}_std") for m in metric_cols]
    result = df.group_by(group_cols).agg(agg).sort(group_cols)
    rankings = result.to_dicts()

    if normalize and rankings:
        for m in metric_cols:
            vals = [r[m] for r in rankings if r.get(m) is not None]
            if not vals:
                continue
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx != mn else 1.0
            for r in rankings:
                v = r.get(m)
                if v is not None:
                    norm = (v - mn) / rng
                    if _is_lower_better(m, declarations):
                        norm = 1.0 - norm
                    r[f"{m}_norm"] = round(norm, 3)
        norm_keys = [f"{m}_norm" for m in metric_cols]
        for r in rankings:
            scores = [r.get(k) for k in norm_keys if r.get(k) is not None]
            r["composite_score"] = round(sum(scores) / len(scores), 3) if scores else None
        rankings.sort(key=lambda r: r.get("composite_score", 0), reverse=True)
        for i, r in enumerate(rankings):
            r["rank"] = i + 1

    return {
        "planner_column": planner_col,
        "n_planners": len(rankings),
        "metrics_compared": metric_cols,
        "rankings": rankings,
    }


def _compare_planners(args: dict, bridge: EvalBridge) -> dict:
    bid = args["benchmark_id"]
    df = bridge.load_combined_metrics(bid)
    if df is None:
        return {"error": f"No combined_metrics.parquet found for {bid}"}

    result = _compare_planners_frame(
        df,
        metrics=args.get("metrics") or _default_compare_metrics(bridge),
        declarations=_metric_declarations(bridge),
        planners=args.get("planners") or [],
        group_by_stage=args.get("group_by_stage", False),
        normalize=args.get("normalize", True),
    )
    return {"benchmark_id": bid, **result}


def _correlation_frame(df: pl.DataFrame, metrics: list[str]) -> dict:
    metrics = [m for m in metrics if m in df.columns]
    if len(metrics) < 2:
        return {"error": "Need at least 2 valid metric columns"}

    corr_df = df.select(metrics).to_pandas().corr()
    matrix = {}
    for i, mi in enumerate(metrics):
        matrix[mi] = {mj: round(float(corr_df.iloc[i, j]), 3) for j, mj in enumerate(metrics)}
    return {"metrics": metrics, "correlation_matrix": matrix}


def _compute_correlation(args: dict, bridge: EvalBridge) -> dict:
    bid = args["benchmark_id"]
    df = bridge.load_combined_metrics(bid)
    if df is None:
        return {"error": f"No combined_metrics.parquet found for {bid}"}

    return {"benchmark_id": bid, **_correlation_frame(df, args.get("metrics", []))}


def _find_top_n_frame(
    df: pl.DataFrame,
    metrics: list[str],
    declarations: dict[str, dict],
    n: int = 3,
    weights: list[float] | None = None,
) -> dict:
    metrics = [m for m in metrics if m in df.columns]
    weights = weights or [1.0] * len(metrics)

    planner_col = "local_planner" if "local_planner" in df.columns else "planner"

    needed = list(dict.fromkeys([planner_col, *metrics]))
    list_cols = [c for c in needed if c in df.columns and df.schema[c] == pl.List]
    if list_cols:
        df = df.select(needed).explode(list_cols)
    else:
        df = df.select(needed)

    agg = [pl.col(m).mean().alias(m) for m in metrics]
    result = df.group_by(planner_col).agg(agg)

    scores = {}
    for m, w in zip(metrics, weights, strict=False):
        vals = result[m].to_list()
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx != mn else 1.0
        for i, planner in enumerate(result[planner_col].to_list()):
            if rng > 0:
                norm = (vals[i] - mn) / rng
                if _is_lower_better(m, declarations):
                    norm = 1.0 - norm
            else:
                norm = 0.5
            scores[planner] = scores.get(planner, 0.0) + norm * w

    total_weight = sum(weights) if weights else 1.0
    ranked = sorted(
        [{"planner": k, "composite_score": round(v / total_weight, 3)} for k, v in scores.items()],
        key=lambda x: x["composite_score"],
        reverse=True,
    )
    return {"top_n": ranked[:n]}


def _find_top_n(args: dict, bridge: EvalBridge) -> dict:
    bid = args["benchmark_id"]
    df = bridge.load_combined_metrics(bid)
    if df is None:
        return {"error": f"No combined_metrics.parquet found for {bid}"}

    result = _find_top_n_frame(
        df,
        metrics=args.get("metrics") or _default_compare_metrics(bridge),
        declarations=_metric_declarations(bridge),
        n=int(args.get("n", 3)),
        weights=args.get("weights") or None,
    )
    return {"benchmark_id": bid, **result}


def _aggregate_by_dimension(args: dict, bridge: EvalBridge) -> dict:
    bid = args["benchmark_id"]
    df = bridge.load_combined_metrics(bid)
    if df is None:
        return {"error": f"No combined_metrics.parquet found for {bid}"}

    dim = args["dimension"]
    if dim not in df.columns:
        return {"error": f"Dimension '{dim}' not in data", "available": df.columns}

    metric_cols = args.get("metrics") or []
    if metric_cols:
        metric_cols = [m for m in metric_cols if m in df.columns]
    else:
        metric_cols = [c for c in df.columns if str(df[c].dtype).startswith(("Float", "Int"))]

    needed = list(dict.fromkeys([dim, *metric_cols]))
    list_cols = [c for c in needed if c in df.columns and df.schema[c] == pl.List]
    if list_cols:
        df = df.select(needed).explode(list_cols)
    else:
        df = df.select(needed)

    agg = [pl.col(m).mean().alias(m) for m in metric_cols]
    result = df.group_by(dim).agg(agg).sort(dim)

    return {
        "benchmark_id": bid,
        "dimension": dim,
        "n_groups": len(result),
        "metrics": metric_cols,
        "data": result.to_dicts(),
    }


def _summarize_benchmark(args: dict, bridge: EvalBridge) -> dict:
    bid = args["benchmark_id"]
    df = bridge.load_combined_metrics(bid)
    if df is None:
        return {"error": f"No combined_metrics.parquet found for {bid}"}

    status = _read_benchmark_status(args, bridge)

    planner_col = "local_planner" if "local_planner" in df.columns else "planner"
    planners = df[planner_col].drop_nulls().unique().to_list() if planner_col in df.columns else []
    stages = df["stage"].drop_nulls().unique().to_list() if "stage" in df.columns else []
    maps = df["map"].drop_nulls().unique().to_list() if "map" in df.columns else []

    planner_stats = []
    if planner_col in df.columns:
        for p in planners:
            pdf = df.filter(pl.col(planner_col) == p)
            n = len(pdf)
            succ = pdf["success"].drop_nulls().mean() if "success" in pdf.columns else None
            planner_stats.append(
                {
                    "planner": p,
                    "n_episodes": n,
                    "success_rate": round(float(succ), 3) if succ is not None else None,
                }
            )

    lines = [
        f"Benchmark: {bid}",
        f"Planners: {', '.join(planners)} ({len(planners)} total)",
        f"Stages: {', '.join(stages)} ({len(stages)} total)",
        f"Maps: {', '.join(maps)} ({len(maps)} total)",
        f"Episodes: {status.get('n_episodes', 'N/A')}",
        "",
        "Per-Planner Summary:",
    ]
    for ps in sorted(planner_stats, key=lambda x: x.get("success_rate", 0) or 0, reverse=True):
        sr = f"{ps['success_rate'] * 100:.1f}%" if ps["success_rate"] is not None else "N/A"
        lines.append(f"  {ps['planner']}: {ps['n_episodes']} episodes, success rate {sr}")

    return {"benchmark_id": bid, "summary": "\n".join(lines), "planner_stats": planner_stats, "status": status}


def _load_notes_file(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("failed to read notes %s: %s", path, exc)
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [{"label": str(k), "value": str(v)} for k, v in data.items()]
    return []


def _save_notes_file(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rows, sort_keys=False, allow_unicode=True))


def _write_notes_file(path: pathlib.Path, notes: list[dict], mode: str) -> dict:
    if mode == "replace":
        rows = list(notes)
    elif mode == "append":
        rows = _load_notes_file(path) + list(notes)
    elif mode == "merge":
        existing_map = {r.get("label", ""): r for r in _load_notes_file(path)}
        for n in notes:
            existing_map[n.get("label", "")] = n
        rows = list(existing_map.values())
    else:
        return {"error": f"Unknown mode: {mode}"}

    _save_notes_file(path, rows)
    return {"n_notes": len(rows), "mode": mode}


def _write_notes(args: dict, bridge: EvalBridge) -> dict:
    bid = args["benchmark_id"]
    path = bridge.benchmark_dir(bid) / "notes.yaml"
    result = _write_notes_file(path, args.get("notes") or [], args.get("mode", "replace"))
    if "error" in result:
        return result
    return {"benchmark_id": bid, **result}


def _append_insight(args: dict, bridge: EvalBridge) -> dict:
    return _write_notes(
        {
            "benchmark_id": args["benchmark_id"],
            "notes": [{"label": args["label"], "value": args["value"]}],
            "mode": "append",
        },
        bridge,
    )
