"""The generator's provenance.yaml beside the scenario an episode ran, read from the run's world snapshot first."""

from __future__ import annotations

import logging
import typing
from collections.abc import Iterable, Mapping
from pathlib import Path

import attrs
import yaml
from arena_simulation_setup.generative.scenario.formats import PROVENANCE
from arena_simulation_setup.shared.pathing import Point

from .criticality import FORMATION_RADIUS_M

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.World import WorldView

    from arena_evaluation.storage.schemas import RunMetadata

logger = logging.getLogger(__name__)

OWN_STEPS = ("C", "D", "injected")


@attrs.frozen
class Formation:
    center: Point | None
    radius: float
    variant: str


@attrs.frozen
class Encounter:
    agent: str
    point: Point


@attrs.frozen
class Provenance:
    id: str
    steps: tuple[str, ...]
    owned: dict[str, tuple[str, ...]]
    formation: Formation | None
    encounters: tuple[Encounter, ...] = ()
    hold_s: float = 0.0
    radii: dict[str, float] = attrs.Factory(dict)

    @property
    def agents(self) -> list[str]:
        """The case's own agents, the ones its C and D steps own and the ones it injected."""
        return sorted({name for step in OWN_STEPS for name in self.owned.get(step, ())})


def parse_provenance(raw: Mapping[str, typing.Any], origin: Point = (0.0, 0.0)) -> Provenance:
    """A provenance mapping with the formation center and encounter points moved from their level frame by `origin` into the map frame."""
    owned = {str(step): tuple(str(name) for name in names or ()) for step, names in (raw.get("owned") or {}).items()}
    spec = raw.get("formation")
    formation = None
    if spec:
        center = spec.get("center")
        formation = Formation(
            center=None if center is None else (float(center[0]) + origin[0], float(center[1]) + origin[1]),
            radius=float(spec.get("radius", FORMATION_RADIUS_M)),
            variant=str(spec.get("variant", "")),
        )
    encounters = tuple(Encounter(agent=str(e["agent"]), point=(float(e["point"][0]) + origin[0], float(e["point"][1]) + origin[1])) for e in raw.get("encounters") or ())
    hold_s = float((raw.get("robot") or {}).get("hold_s") or 0.0)
    return Provenance(id=str(raw.get("id", "")), steps=tuple(str(s) for s in raw.get("steps") or ()), owned=owned, formation=formation, encounters=encounters, hold_s=hold_s)


def human_radii(scenario: Mapping[str, typing.Any]) -> dict[str, float]:
    """The radius each scenario humans entry pins in its agent block, by entry name."""
    radii: dict[str, float] = {}
    for human in scenario.get("humans") or ():
        agent = human.get("agent")
        if isinstance(agent, Mapping) and agent.get("radius") is not None and human.get("name") is not None:
            radii[str(human["name"])] = float(agent["radius"])
    return radii


def scenario_name(run: RunMetadata | None) -> str | None:
    """The scenario file the episode's `scenario` task mode ran, None when neither axis ran one."""
    if run is None:
        return None
    for mode, params in ((run.tm_obstacles, run.obstacles_params), (run.tm_robots, run.robots_params)):
        if mode == "scenario" and params and params.get("file"):
            return str(params["file"])
    return None


_provenance_cache: dict[tuple[str, str, str], tuple[tuple[tuple[Path, int], ...], Provenance | None]] = {}


def _stamp(paths: Iterable[Path]) -> tuple[tuple[Path, int], ...]:
    return tuple((path, path.stat().st_mtime_ns if path.exists() else -1) for path in paths)


def world_view(world: str, run_dir: Path | None = None) -> WorldView | None:
    """The world an episode ran in: its snapshot under `run_dir`/worlds first, else the resolved world, None without either."""
    from arena_simulation_setup.tree.World import WorldIdentifier, WorldView

    name, _ = WorldIdentifier.split_filter(world)
    snapshot = None if run_dir is None else Path(run_dir) / "worlds" / name
    if snapshot is not None and (snapshot / "world.yaml").is_file():
        return WorldView(snapshot)
    try:
        return WorldIdentifier(name).resolve_sync()
    except FileNotFoundError as e:
        logger.warning("edge_case: world '%s' not available locally: %s", name, e)
        return None


def _frame_view(world: str, view: WorldView) -> WorldView:
    """The world whose level origins the runtime used: the full world under a level filter, which a filtered snapshot lacks."""
    from arena_simulation_setup.tree.World import WorldIdentifier

    name, level_filter = WorldIdentifier.split_filter(world)
    if level_filter is None:
        return view
    try:
        return WorldIdentifier(name).resolve_sync()
    except FileNotFoundError:
        return view


def _level_origin(view: WorldView, scenario: Mapping[str, typing.Any]) -> Point:
    origins = view.level_origins()
    if not origins:
        return (0.0, 0.0)
    robots = scenario.get("robots") or []
    level = str(robots[0].get("level")) if robots and robots[0].get("level") is not None else next(iter(origins))
    return origins.get(level, (0.0, 0.0))


def _read_provenance(world: str, scenario: str, run_dir: Path | None) -> tuple[list[Path], Provenance | None]:
    view = world_view(world, run_dir)
    if view is None:
        return [], None
    try:
        scenario_view = view.scenario(scenario).resolve_sync()
    except FileNotFoundError as e:
        logger.warning("edge_case: scenario '%s' of world '%s' not available locally: %s", scenario, world, e)
        return [], None
    path = Path(scenario_view.path) / PROVENANCE
    if not path.is_file():
        return [path], None
    frame = _frame_view(world, view)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenario_doc = scenario_view.read()
    provenance = attrs.evolve(parse_provenance(raw, _level_origin(frame, scenario_doc)), radii=human_radii(scenario_doc))
    return [path, Path(scenario_view.scenario_path), frame.world_path], provenance


def load_provenance(world: str, scenario: str, run_dir: Path | None = None) -> Provenance | None:
    """The provenance of `scenario` in `world`, the run's snapshot first, None without the scenario or its provenance.yaml."""
    key = (str(run_dir), world, scenario)
    cached = _provenance_cache.get(key)
    if cached is not None and _stamp(path for path, _ in cached[0]) == cached[0]:
        return cached[1]
    files, result = _read_provenance(world, scenario, run_dir)
    _provenance_cache[key] = (_stamp(files), result)
    return result
