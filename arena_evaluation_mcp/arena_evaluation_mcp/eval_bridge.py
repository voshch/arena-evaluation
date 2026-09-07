from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import signal
import subprocess
import time

import polars as pl
import yaml

from arena_evaluation.processing.parquet_store import ParquetStore
from arena_evaluation.storage.data_root import benchmarks_root
from arena_evaluation.storage.folder_manager import FolderManager

from .common import run_status, validate_path_component

logger = logging.getLogger(__name__)


class EvalBridge:
    """Adapter wrapping arena_evaluation APIs for MCP server tools and resources."""

    def __init__(self) -> None:
        self._data_root = self._resolve_data_root()
        self._fm = FolderManager(data_root=self._data_root)
        self._bg_processes: dict[str, subprocess.Popen] = {}

    @staticmethod
    def _resolve_data_root() -> pathlib.Path:
        env = os.environ.get("ARENA_DATA_DIR")
        if env:
            return pathlib.Path(env) / "benchmarks"
        return benchmarks_root()

    @property
    def data_root(self) -> pathlib.Path:
        return self._data_root

    @property
    def fm(self) -> FolderManager:
        return self._fm

    def list_benchmarks(
        self,
        suite: str | None = None,
        contest: str | None = None,
        status: str | None = None,
        query: str | None = None,
    ) -> list[dict]:
        """List benchmark runs under data_root, optionally filtered."""
        root = self._data_root
        if not root.is_dir():
            return []

        runs: list[dict] = []
        for run_path in sorted(root.iterdir(), reverse=True):
            if not run_path.is_dir():
                continue
            manifest_path = run_path / "manifest.yaml"
            if not manifest_path.exists():
                continue
            try:
                data = yaml.safe_load(manifest_path.read_text())
                state_path = run_path / ".benchmark_state.json"
                state: dict = {}
                if state_path.exists():
                    state = json.loads(state_path.read_text())
            except (OSError, yaml.YAMLError, json.JSONDecodeError) as exc:
                logger.warning("skipping benchmark %s: %s", run_path, exc)
                continue
            if not isinstance(data, dict) or not isinstance(state, dict):
                logger.warning("skipping benchmark %s: malformed manifest or state", run_path)
                continue

            run_id = str(data.get("run_id", run_path.name))
            suite_name = data.get("suite_name", "")
            contest_name = data.get("contest_name", "")
            if suite and suite != suite_name:
                continue
            if contest and contest != contest_name:
                continue
            if query and query.lower() not in run_id.lower():
                continue

            steps = state.get("steps", {})
            statuses = [s.get("status") for s in steps.values()]
            run_status_value = run_status(statuses)
            if status and status != run_status_value:
                continue

            runs.append(
                {
                    "run_id": run_id,
                    "suite": suite_name,
                    "contest": contest_name,
                    "created_at": data.get("created_at", ""),
                    "simulator": data.get("simulator", ""),
                    "status": run_status_value,
                    "steps_total": len(steps),
                    "steps_ok": statuses.count("ok"),
                    "steps_failed": statuses.count("failed"),
                    "steps_partial": statuses.count("partial"),
                    "steps_in_progress": statuses.count("in_progress"),
                    "running_pid": None,
                    "has_combined_metrics": (run_path / "combined_metrics.parquet").exists(),
                    "has_report": (run_path / "report.html").exists(),
                }
            )

        try:
            from arena_evaluation.benchmark.debug import running_pids_by_run_id

            pids = running_pids_by_run_id()
        except (ImportError, OSError, subprocess.SubprocessError) as exc:
            logger.warning("could not query running benchmark pids: %s", exc)
        else:
            for r in runs:
                r["running_pid"] = pids.get(r["run_id"])

        return runs

    @staticmethod
    def _clean_robot_name(name: str) -> str:
        """Strip env-scoped prefixes like 'env_0_jackal' -> 'jackal'."""
        return re.sub(r"^env_\d+_", "", str(name))

    def _config_files(self, kind: str) -> list[pathlib.Path]:
        """All YAML files for suites or contests (share dir + source tree)."""
        from arena_evaluation.presentation.manifest_registry import (
            share_dir,
            source_tree_dir,
        )

        seen: set[pathlib.Path] = set()
        paths: list[pathlib.Path] = []
        for base in (share_dir(), source_tree_dir()):
            if base is None:
                continue
            d = base / "configs" / "benchmark" / kind
            if d.is_dir():
                for p in sorted(d.glob("*.yaml")):
                    if p.resolve() not in seen:
                        seen.add(p.resolve())
                        paths.append(p)
        return paths

    def list_suite_stems(self) -> list[str]:
        """Bundled suite config stems (deduplicated)."""
        return sorted({p.stem for p in self._config_files("suites")})

    def list_contest_stems(self) -> list[str]:
        """Bundled contest config stems (deduplicated)."""
        return sorted({p.stem for p in self._config_files("contests")})

    def read_config_template(self, kind: str, name: str) -> dict | None:
        """Read a bundled (or custom) suite / contest / manifest YAML."""
        stem = validate_path_component(name[:-5] if name.endswith(".yaml") else name)
        if kind == "manifest":
            from arena_evaluation.presentation.manifest_registry import (
                find_manifest_file,
            )

            p = find_manifest_file(stem)
            if p is None:
                for bid in self._all_benchmark_ids():
                    cand = self._data_root / bid / f"{stem}.yaml"
                    if cand.is_file():
                        p = cand
                        break
            if p is None:
                return None
            return {"kind": kind, "name": stem, "path": str(p), "content": p.read_text()}

        for f in self._config_files(f"{kind}s" if not kind.endswith("s") else kind):
            if f.stem == stem:
                return {"kind": kind, "name": stem, "path": str(f), "content": f.read_text()}
        return None

    def _collect_key_values(self, node: object, key_suffix: str, values: set[str]) -> None:
        """Recursively collect string values of keys ending in key_suffix."""
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.endswith(key_suffix):
                    if isinstance(v, (list, tuple)):
                        values.update(str(x) for x in v)
                    else:
                        values.add(str(v))
                self._collect_key_values(v, key_suffix, values)
        elif isinstance(node, list):
            for item in node:
                self._collect_key_values(item, key_suffix, values)

    def _config_values(self, kind: str, key_suffix: str) -> set[str]:
        """Values of keys ending in key_suffix across all config YAMLs."""
        values: set[str] = set()
        for p in self._config_files(kind):
            try:
                data = yaml.safe_load(p.read_text())
            except (OSError, yaml.YAMLError) as exc:
                logger.warning("failed to parse config %s: %s", p, exc)
                continue
            self._collect_key_values(data, key_suffix, values)
        return {v for v in values if v and v != "null"}

    def _share_subdirs(self, pkg: str, sub: str) -> set[str]:
        """Subdirectory names of <pkg_share>/<sub>, e.g. arena_robots/robots."""
        try:
            from ament_index_python.packages import get_package_share_directory

            d = pathlib.Path(get_package_share_directory(pkg)) / sub
        except (ImportError, LookupError) as exc:
            logger.warning("share dir of %s unavailable: %s", pkg, exc)
            return set()
        if d.is_dir():
            return {p.name for p in d.iterdir() if p.is_dir()}
        return set()

    def discover_available_planners(self) -> list[str]:
        """Planner names from the repo catalogs only: contest configs +
        nav2 controller names (no recorded-benchmark data)."""
        planners = self._config_values("contests", "local_planner")
        planners.update(self.planner_catalog()["local_planners"])
        planners.discard("unhindered_peds")
        planners.discard("characterization")
        return sorted(planners)

    def discover_available_inter_planners(self) -> list[str]:
        """Inter-planner names from contest configs + the nav2 interplanner catalog."""
        inter = self._config_values("contests", "inter_planner")
        inter.update(self.planner_catalog()["inter_planners"])
        return sorted(v for v in inter if v) or ["bypass"]

    def discover_available_maps(self) -> list[str]:
        """Runnable world names: static worlds catalog + generated worlds."""
        maps_set = self._share_subdirs("arena_simulation_setup", "worlds")

        try:
            from ament_index_python.packages import get_package_share_directory

            gen = pathlib.Path(get_package_share_directory("arena_simulation_setup")) / "worlds" / ".generated"
        except (ImportError, LookupError) as exc:
            logger.warning("generated worlds unavailable: %s", exc)
        else:
            if gen.is_dir():
                maps_set.update(p.name for p in gen.iterdir() if p.is_dir() and (p / "world.yaml").is_file())

        return sorted(m for m in maps_set if m and not m.startswith("."))

    def discover_suite_map_names(self) -> list[str]:
        """Map names referenced by existing suite configs."""
        return sorted(self._config_values("suites", "map"))

    def discover_available_robots(self) -> list[str]:
        """Robot models from the arena_robots catalog + suite configs."""
        robots = self._share_subdirs("arena_robots", "robots")
        robots.update(self._config_values("suites", "robot"))
        return sorted(r for r in robots if r and r != "README.md")

    def discover_task_modes(self) -> dict:
        """Return available TM_Robots and TM_Obstacles enum values."""
        try:
            from task_generator.constants import Constants

            return {
                "tm_robots": [m.value for m in Constants.TaskMode.TM_Robots],
                "tm_obstacles": [m.value for m in Constants.TaskMode.TM_Obstacles],
            }
        except ImportError:
            return {
                "tm_robots": [
                    "random",
                    "explore",
                    "guided",
                    "stationary",
                    "scenario",
                    "characterization",
                    "demo",
                ],
                "tm_obstacles": ["parametrized", "random", "scenario", "environment", "prompt"],
            }

    def pedestrian_models(self) -> dict:
        """Available pedestrian (dynamic) model names.

        Humans are asset-bucket bundles resolved by name (`arena asset ls human`).
        'arenian' is the default.
        """
        local = self._share_subdirs("arena_simulation_setup", "assets/Common/Human")
        return {
            "bundled": sorted(local),
            "fallback": "arenian",
            "note": "Unknown models silently fall back to 'arenian', which is fetched from the asset bucket.",
        }

    def static_object_models(self) -> dict:
        """Available static obstacle model names (assets/Common/Object)."""
        local = self._share_subdirs("arena_simulation_setup", "assets/Common/Object")
        return {
            "bundled": sorted(local),
            "note": "Empty models list in config.random.static.models uses ALL available object identifiers.",
        }

    TASK_MODE_REFERENCE = {
        "guidance": {
            "same_path_across_stages": (
                "TWO WAYS TO GET AN IDENTICAL PATH ACROSS STAGES - pick by how much "
                "route control you need:\n"
                "1. tm_robots: random + explicit identical 'seed: <int>' on every "
                "stage (same map, same robot, same episode count). Each episode "
                "increments the seed (seeds are seed + episode_index), so within a "
                "stage every episode differs, but episode i of every stage is "
                "IDENTICAL - including the same randomly-sampled start/goal. Works "
                "on any map with zero authoring. Choose this for statistically "
                "reproducible comparisons (same sampling conditions, different "
                "pedestrian configs). Note: when 'seed' is omitted it is "
                "auto-derived from stage fields EXCLUDING config, but different "
                "stage names change it - always set the seed explicitly.\n"
                "2. tm_robots: scenario - start/goal/phases are read from the "
                "world's scenario file; the route is hand-authored and "
                "seed-independent. Choose this only when the trajectory itself "
                "must be EXACTLY authored (e.g. a specific corridor route), and "
                "the scenario file already exists (inspect_map lists a map's "
                "scenarios). For 'same path, different peds' comparisons the "
                "random+seed approach is simpler and needs no file authoring."
            ),
            "construction": ("Construct suite YAML directly from the schemas and examples below with concrete values. Do NOT infer structure from previously existing benchmark configs - they may use outdated patterns."),
        },
        "tm_robots": {
            "random": {
                "purpose": "Robot start/goal sampled from free space, seeded by the stage seed. Same seed + same map + same robot = same path.",
                "config_keys": {},
                "example": (
                    "stages:\n"
                    "  - name: corridor_dense\n"
                    "    map: hospital_1\n"
                    "    robot: jackal\n"
                    "    tm_robots: random\n"
                    "    tm_obstacles: random\n"
                    "    episodes: 5\n"
                    "    seed: 42\n"
                    "    timeout: 120s\n"
                    "    config:\n"
                    "      random:\n"
                    "        dynamic: {min: 8, max: 12, models: [arenian]}\n"
                    "        static: {min: 3, max: 5, models: []}\n"
                    "        interactive: {min: 0, max: 0, models: []}"
                ),
            },
            "scenario": {
                "purpose": "Robot follows a SCRIPTED fixed route (start pose + goto/gesture phases) from the world's scenario file - use this for an authored route independent of seeding.",
                "config_keys": {"file": "scenario name, resolved to worlds/<map>/scenarios/<file>/scenario.yaml (e.g. 'default', 'flow'). Shared with tm_obstacles: scenario."},
                "file_format": ("robots: [{start: [x, y, yaw], phases: [{goto: [x, y, yaw]}]}]\nstatic: [{name, model, pose: [x, y, yaw]}]\ndynamic: [{name, model, pose, waypoints: [[x, y, yaw], ...]}]\nregions: {<name>: {type: source|sink, polygon: [{x, y}, ...]}}"),
                "example": ("stages:\n  - name: scripted_route\n    map: hospital_1\n    robot: jackal\n    tm_robots: scenario\n    tm_obstacles: random\n    episodes: 3\n    config:\n      scenario:\n        file: default\n      random:\n        dynamic: {min: 5, max: 10, models: [arenian]}"),
            },
            "guided": {
                "purpose": "Robot driven to waypoints supplied LIVE via the set_goal() service by an external controller. The suite config CANNOT pre-script waypoints - not usable for autonomous benchmark runs.",
                "config_keys": {},
                "example": None,
            },
            "explore": {
                "purpose": "Robot autonomously explores the map with generated goals.",
                "config_keys": {},
                "example": None,
            },
            "stationary": {
                "purpose": "Robot stays in place (used for reference steps).",
                "config_keys": {},
                "example": None,
            },
            "characterization": {
                "purpose": "Open-loop cmd_vel sweep through the robot's rated envelope (idle, ramps, pivots). Drives cmd_vel directly; no planner required. Use references: false in the suite and the 'characterization' report manifest.",
                "config_keys": {},
                "example": (
                    "stages:\n"
                    "  - name: characterization\n"
                    "    map: map_empty\n"
                    "    robot: jackal\n"
                    "    tm_robots: characterization\n"
                    "    tm_obstacles: random\n"
                    "    episodes: 3\n"
                    "    config:\n"
                    "      random:\n"
                    "        dynamic: {min: 0, max: 0}\n"
                    "        static: {min: 0, max: 0}\n"
                    "        interactive: {min: 0, max: 0}\n"
                    "references: false"
                ),
            },
            "demo": {"purpose": "Demo/scripted behavior.", "config_keys": {}, "example": None},
        },
        "tm_obstacles": {
            "random": {
                "purpose": "Randomly placed obstacles/pedestrians per episode.",
                "config_keys": {
                    "dynamic": "PEDESTRIANS: {min, max, models: [...]} - count range sampled per episode; models from the Human catalog (bundled: ['arenian']).",
                    "static": "Inanimate obstacles: {min, max, models: [...]} - empty models = ALL available object models.",
                    "interactive": "Inanimate obstacles, same pool as static, default {min: 0, max: 0} (disabled).",
                },
                "example": ("config:\n  random:\n    dynamic: {min: 8, max: 12, models: [arenian]}\n    static: {min: 3, max: 5, models: []}\n    interactive: {min: 0, max: 0, models: []}"),
            },
            "scenario": {
                "purpose": "Loads static/dynamic entities from the world's scenario file (same 'file' key as tm_robots: scenario).",
                "config_keys": {"file": "scenario name (e.g. 'default', 'flow')."},
                "example": ("config:\n  scenario:\n    file: default"),
            },
            "parametrized": {
                "purpose": "Obstacle counts/types from an XML config.",
                "config_keys": {"file": "XML name resolved to arena_bringup/configs/parametrized/<file>.xml"},
                "example": ("config:\n  parametrized:\n    file: default"),
            },
            "environment": {"purpose": "Uses environment-provided obstacles.", "config_keys": {}, "example": None},
            "prompt": {"purpose": "Obstacles defined by prompt.", "config_keys": {}, "example": None},
        },
    }

    def describe_task_mode(self, mode: str | None = None) -> dict:
        """Reference for tm_robots / tm_obstacles modes."""
        modes = self.discover_task_modes()
        catalog = dict(self.TASK_MODE_REFERENCE)
        guidance = catalog.pop("guidance")
        if mode:
            for group in ("tm_robots", "tm_obstacles"):
                if mode in catalog[group]:
                    return {
                        "guidance": guidance,
                        mode: catalog[group][mode],
                        "group": group,
                        "valid_modes": modes,
                    }
            return {"error": f"unknown mode '{mode}'", "guidance": guidance, "valid_modes": modes, "pedestrian_models": self.pedestrian_models(), "static_object_models": self.static_object_models()}
        return {
            "guidance": guidance,
            "catalog": catalog,
            "valid_modes": modes,
            "pedestrian_models": self.pedestrian_models(),
            "static_object_models": self.static_object_models(),
        }

    def planner_catalog(self) -> dict:
        """Drivers + local/global/inter planner names from the sim configs."""
        drivers = ["nav2", "rosnav_rl", "none", "external"]
        caps = ["mobile", "arm", "lift"]
        try:
            from ament_index_python.packages import get_package_share_directory

            base = pathlib.Path(get_package_share_directory("arena_robots")) / "config" / "nav2"
            local = sorted(p.name for p in (base / "controllers").glob("*") if p.is_dir()) if (base / "controllers").is_dir() else []
            global_ = sorted(p.name for p in (base / "planners").glob("*") if p.is_dir()) if (base / "planners").is_dir() else []
            inter = sorted(p.name for p in (base / "interplanners").glob("*") if p.is_dir()) if (base / "interplanners").is_dir() else []
        except (ImportError, LookupError) as exc:
            logger.warning("nav2 planner catalog unavailable: %s", exc)
            local, global_, inter = [], [], []
        return {
            "cap_keys": caps,
            "drivers": drivers,
            "local_planners": local,
            "global_planners": global_,
            "inter_planners": inter,
        }

    def inspect_map(self, map_name: str) -> dict:
        """Metadata for a map: bounds, zones, scenarios with coordinates."""
        info: dict = {"map": map_name}
        from arena_evaluation.processing.map_registry import MapRegistry

        try:
            world_dir = MapRegistry._find_ros_map_dir(map_name)
        except (ImportError, LookupError, OSError) as exc:
            logger.warning("map lookup failed for %s: %s", map_name, exc)
            world_dir = None
        if world_dir is None:
            return {"map": map_name, "error": "map not found in arena_simulation_setup/worlds/"}
        info["world_dir"] = str(world_dir)

        try:
            # single shared map cache (MapRegistry default: /opt/arena_ws/data/maps_cache)
            meta = MapRegistry.get_map(map_name)
            if meta:
                info["map_metadata"] = {
                    "png_path": meta.get("png_path"),
                    "resolution_m_per_px": meta.get("resolution"),
                    "origin": meta.get("origin"),
                    "width_px": meta.get("width"),
                    "height_px": meta.get("height"),
                }
        except Exception as exc:
            info["map_metadata"] = {"error": str(exc)}

        zones: list[dict] = []
        for cand in (world_dir / "0" / "world.yaml", world_dir / "world.yaml"):
            if not cand.is_file():
                continue
            try:
                data = yaml.safe_load(cand.read_text())
            except (OSError, yaml.YAMLError) as exc:
                logger.warning("failed to parse %s: %s", cand, exc)
                continue
            for zone in (data or {}).get("zones", []) or []:
                corners = zone.get("corners") or []
                xs = [c["x"] for c in corners if isinstance(c, dict) and "x" in c]
                ys = [c["y"] for c in corners if isinstance(c, dict) and "y" in c]
                if xs and ys:
                    zones.append(
                        {
                            "name": zone.get("name", "?"),
                            "bounds": {"x": [round(min(xs), 2), round(max(xs), 2)], "y": [round(min(ys), 2), round(max(ys), 2)]},
                        }
                    )
        info["zones"] = zones
        if zones:
            all_x = [b["bounds"]["x"] for b in zones]
            all_y = [b["bounds"]["y"] for b in zones]
            info["map_bounds"] = {
                "x": [min(b[0] for b in all_x), max(b[1] for b in all_x)],
                "y": [min(b[0] for b in all_y), max(b[1] for b in all_y)],
            }

        scenarios: list[dict] = []
        sc_dir = world_dir / "scenarios"
        if sc_dir.is_dir():
            for sd in sorted(sc_dir.iterdir()):
                if not sd.is_dir():
                    continue
                entry: dict = {"name": sd.name}
                for cand in (sd / "scenario.yaml", sd / "scenario.json"):
                    if not cand.is_file():
                        continue
                    try:
                        data = yaml.safe_load(cand.read_text())
                    except (OSError, yaml.YAMLError) as exc:
                        logger.warning("failed to parse %s: %s", cand, exc)
                        continue
                    robots = (data or {}).get("robots") or []
                    if robots:
                        r0 = robots[0]
                        entry["robot"] = {
                            "start": r0.get("start") or r0.get("start_pos"),
                            "goal": r0.get("goal"),
                            "n_phases": len(r0.get("phases") or []) if r0.get("phases") else None,
                        }
                    entry["n_dynamic_peds"] = len((data or {}).get("dynamic") or [])
                    entry["n_static"] = len((data or {}).get("static") or [])
                    entry["regions"] = sorted((data or {}).get("regions") or {})
                    entry["has_robot_route"] = bool(robots)
                scenarios.append(entry)
        info["scenarios"] = scenarios

        return info

    def discover_available_manifests(self) -> list[str]:
        """Bundled manifests + custom manifests written into benchmark dirs."""
        from arena_evaluation.presentation.manifest_registry import (
            available_manifests,
        )

        bundled: set[str] = set(available_manifests())

        _IGNORED = {
            "manifest",
            "notes",
            "report_manifest",
            "viz_manifest",
            "combined_metrics",
            "metrics",
            "simulation_profile",
        }
        for bid in self._all_benchmark_ids():
            bdir = self._data_root / bid
            if not bdir.is_dir():
                continue
            try:
                bundled.update(p.stem for p in bdir.glob("*.yaml") if p.stem not in _IGNORED)
            except OSError as exc:
                logger.warning("failed to scan %s for manifests: %s", bdir, exc)
                continue

        return sorted(bundled)

    def discover_available_metrics(self) -> list[dict]:
        """List all registered metric names, categories, and units."""
        from arena_evaluation.processing.metrics.registry import (
            MetricRegistry,
        )

        MetricRegistry.discover_calculators_cls()
        registry = MetricRegistry(None)
        return registry.list_metrics()

    def load_combined_metrics(self, benchmark_id: str) -> pl.DataFrame | None:
        """Load combined_metrics.parquet for a benchmark."""
        path = self._fm.combined_metrics_path(benchmark_id)
        if not path.exists():
            return None
        df, _ = ParquetStore.read(path)
        return df

    def read_yaml(self, path: pathlib.Path) -> dict | None:
        """Read a YAML file safely."""
        if not path.exists():
            return None
        try:
            return yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("failed to read %s: %s", path, exc)
            return None

    def read_benchmark_manifest(self, benchmark_id: str) -> dict | None:
        """Read the manifest.yaml for a benchmark run."""
        path = self.benchmark_dir(benchmark_id) / "manifest.yaml"
        return self.read_yaml(path)

    def read_benchmark_state(self, benchmark_id: str) -> dict | None:
        """Read .benchmark_state.json for a benchmark run."""
        path = self.benchmark_dir(benchmark_id) / ".benchmark_state.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to read %s: %s", path, exc)
            return None

    def read_progress_csv(self, benchmark_id: str) -> list[dict] | None:
        """Read progress.csv as a list of dictionaries."""
        import csv
        import io

        path = self.benchmark_dir(benchmark_id) / "progress.csv"
        if not path.exists():
            return None
        try:
            text = path.read_text()
            lines = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
            reader = csv.DictReader(io.StringIO("\n".join(lines)))
            return list(reader)
        except (OSError, csv.Error) as exc:
            logger.warning("failed to read %s: %s", path, exc)
            return None

    def read_notes(self, benchmark_id: str) -> list[dict] | None:
        """Read notes.yaml from a benchmark directory."""
        path = self.benchmark_dir(benchmark_id) / "notes.yaml"
        if not path.exists():
            return None
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("failed to read %s: %s", path, exc)
            return None
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [{"label": str(k), "value": str(v)} for k, v in data.items()]
        return [{"label": "", "value": str(data)}]

    _CLI_WRAPPER = "cd /opt/arena_ws && source /opt/arena_ws/source > /dev/null 2>&1 && arena evaluation"

    @classmethod
    def _cli_command(cls, *args: str) -> list[str]:
        """Build the wrapped bash command for an arena evaluation invocation."""
        import shlex

        return [
            "bash",
            "-c",
            f"{cls._CLI_WRAPPER} {shlex.join(args)}",
        ]

    def run_cli(self, *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
        """Execute an arena evaluation CLI command synchronously."""
        return subprocess.run(
            self._cli_command(*args),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def run_cli_background(self, *args: str) -> subprocess.Popen:
        """Spawn an arena evaluation CLI command in the background.

        Used for run_benchmark which can take minutes to hours.
        Returns the Popen process handle.

        Console output is NOT captured here: the benchmark runner itself
        writes ``benchmarks/<run_id>/runner.log`` (Python logging + launch
        output), which is what the console/debug tools tail.
        """
        return subprocess.Popen(
            self._cli_command(*args),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Debugging: processes and console logs

    def running_processes(self) -> list[dict]:
        """All arena-related processes on the system, categorized."""
        from arena_evaluation.benchmark.debug import running_processes

        return running_processes()

    def kill_processes(
        self,
        pids: list[int] | None = None,
        force: bool = False,
        kind: str | None = None,
    ) -> list[dict]:
        """Terminate running arena processes with optional SIGKILL escalation."""
        from arena_evaluation.benchmark.debug import kill_processes

        return kill_processes(pids=pids, force=force, kind=kind)

    def tail_console(self, run_id: str | None = None, lines: int = 200) -> dict:
        """Tail the console log of a benchmark run.

        run_id None -> the most recently started benchmark runner.
        """
        from arena_evaluation.benchmark.debug import (
            latest_running_run_id,
            tail_console,
        )

        if run_id is None:
            run_id = latest_running_run_id()
            if run_id is None:
                return {
                    "run_id": None,
                    "exists": False,
                    "lines": [],
                    "note": "no running benchmark found and no run_id given",
                }
        return tail_console(run_id, lines=lines)

    # Stopping benchmarks

    @staticmethod
    def _proc_cmdline(pid: int) -> str:
        try:
            raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return ""
        return raw.replace(b"\0", b" ").decode(errors="replace").strip()

    @staticmethod
    def _pgid_members(pgid: int) -> list[int]:
        """PIDs whose process group is *pgid*, read from /proc."""
        members: list[int] = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                stat = pathlib.Path(f"/proc/{entry}/stat").read_text()
            except OSError:
                continue
            fields = stat.rsplit(")", 1)[-1].split()
            if len(fields) > 2 and fields[2] == str(pgid):
                members.append(int(entry))
        return members

    def stop_benchmark(self, run_id: str, force: bool = False) -> dict:
        """Stop the runner process group recorded for *run_id*."""
        if not run_id:
            return {"error": "run_id is required"}
        proc = self._bg_processes.get(run_id)
        if proc is None:
            return {"error": f"run '{run_id}' is not tracked by this server (it was started before the server restarted)"}
        if proc.poll() is not None:
            return {"run_id": run_id, "stopped": "already exited", "returncode": proc.returncode}

        pgid = proc.pid  # runner was spawned with start_new_session=True
        cmdline = self._proc_cmdline(pgid)
        if "evaluation benchmark" not in cmdline:
            return {"error": f"pid {pgid} recorded for run '{run_id}' is not a benchmark runner: {cmdline!r}"}

        sigs = [signal.SIGKILL] if force else [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
        for sig in sigs:
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                break
            time.sleep(2 if sig == signal.SIGKILL else 4)
            proc.poll()
            if not self._pgid_members(pgid):
                break
        proc.poll()

        leftover = [{"pid": p, "cmdline": self._proc_cmdline(p)[:100]} for p in self._pgid_members(pgid)]
        return {"run_id": run_id, "stopped": leftover or "stopped"}

    def benchmark_dir(self, benchmark_id: str) -> pathlib.Path:
        """Return the resolved benchmark directory. Rejects path-traversing ids."""
        validate_path_component(benchmark_id)
        return self._data_root / benchmark_id

    def suite_path(self, name: str, location: str = "install") -> pathlib.Path:
        """Resolve the WRITE TARGET for a suite config."""
        return self._config_write_target("suites", name, location)

    def contest_path(self, name: str, location: str = "install") -> pathlib.Path:
        """Resolve the WRITE TARGET for a contest config."""
        return self._config_write_target("contests", name, location)

    @staticmethod
    def _arena_evaluation_source_root() -> pathlib.Path:
        """The arena_evaluation source package root (configs live under it)."""
        here = pathlib.Path(__file__).resolve()
        cand = here.parents[3] / "arena_evaluation" / "arena_evaluation"
        if (cand / "configs" / "benchmark").is_dir():
            return cand
        fallback = pathlib.Path("/opt/arena_ws/src/Arena/arena_evaluation/arena_evaluation")
        return fallback

    def _config_write_target(self, kind: str, name: str, location: str = "install") -> pathlib.Path:
        """Absolute path where a new suite/contest YAML should be written."""
        from arena_evaluation.presentation.manifest_registry import (
            share_dir,
            source_tree_dir,
        )

        subdir = f"configs/benchmark/{kind}"
        stem = validate_path_component(name[:-5] if name.endswith(".yaml") else name)

        if location == "source":
            return self._arena_evaluation_source_root() / subdir / f"{stem}.yaml"

        # location == "install" (default): the runner only reads pkg_share.
        for base in (share_dir(), source_tree_dir()):
            if base is None:
                continue
            cand = base / subdir / f"{stem}.yaml"
            if cand.is_file():
                return cand
        share = share_dir()
        if share is not None:
            return share / subdir / f"{stem}.yaml"
        return self._arena_evaluation_source_root() / subdir / f"{stem}.yaml"

    @staticmethod
    def _arena_simulation_setup_source_root() -> pathlib.Path | None:
        """Source root of arena_simulation_setup."""
        here = pathlib.Path(__file__).resolve()
        for parent in here.parents:
            cand = parent / "arena_simulation_setup"
            if (cand / "worlds").is_dir():
                return cand
            cand2 = parent / "src" / "Arena" / "arena_simulation_setup"
            if (cand2 / "worlds").is_dir():
                return cand2
        for cand_path in (
            pathlib.Path("/opt/arena_ws/src/Arena/arena_simulation_setup"),
            pathlib.Path("/home/nelson/arena_ws/src/Arena/arena_simulation_setup"),
            pathlib.Path("u:/src/Arena/arena_simulation_setup"),
            pathlib.Path("/mnt/u/src/Arena/arena_simulation_setup"),
        ):
            if (cand_path / "worlds").is_dir():
                return cand_path
        return None

    def scenario_write_targets(self, map_name: str, scenario_name: str, location: str = "both") -> list[pathlib.Path]:
        """Resolve write targets for a scenario file."""
        map_name = validate_path_component(map_name)
        scenario_name = validate_path_component(scenario_name)
        targets: list[pathlib.Path] = []

        # 1. Source target
        src_root = self._arena_simulation_setup_source_root()
        src_target = src_root / "worlds" / map_name / "scenarios" / scenario_name / "scenario.yaml" if src_root is not None else None

        # 2. Install share target
        install_target = None
        try:
            from ament_index_python.packages import get_package_share_directory

            share = pathlib.Path(get_package_share_directory("arena_simulation_setup"))
            install_target = share / "worlds" / map_name / "scenarios" / scenario_name / "scenario.yaml"
        except (ImportError, LookupError):
            pass

        if location == "source":
            if src_target:
                targets.append(src_target)
        elif location == "install":
            if install_target:
                targets.append(install_target)
            elif src_target:
                targets.append(src_target)
        else:  # "both" (default)
            if src_target:
                targets.append(src_target)
            if install_target and (not src_target or install_target.resolve() != src_target.resolve()):
                targets.append(install_target)

        return targets

    def _all_benchmark_ids(self) -> list[str]:
        """Return all benchmark directory names."""
        root = self._data_root
        if not root.is_dir():
            return []
        return sorted(
            [p.name for p in root.iterdir() if p.is_dir()],
            reverse=True,
        )
