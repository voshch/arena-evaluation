"""What `populate` writes (format 2), `ScenarioView.load` reads: no agent dropped, every zone reference resolved inside its zone."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from arena_simulation_setup.tree.World.Scenario import Scenario

pytestmark = pytest.mark.ros

WorldIdentifier = pytest.importorskip("arena_simulation_setup.tree.World", reason="needs a sourced environment").WorldIdentifier

WORLD = "hospital_1"
# the tracked `normal` scenario's robot; furniture is left out, this checks the written format, not placement
ROBOT = {"start": [10.0, 0.5, 1.5708], "phases": [{"goto": [15.0, 32.2, 0.0]}]}


@pytest.fixture
def loaded(worlds_root: Path, generated_scenario: str) -> Scenario:
    import numpy as np

    view = WorldIdentifier(WORLD).resolve_sync()
    world = view.load()
    origins = view.level_origins() or {level_id: (0.0, 0.0) for level_id in world.levels}
    converter = world.compact_world(origins).zone_converter(np.random.default_rng(0))
    return view.scenario(generated_scenario).resolve_sync().load(converter=converter)


@pytest.fixture
def generated_scenario(worlds_root: Path) -> Iterator[str]:
    from arena_evaluation.generation.formats import first_level, write_scenario
    from arena_evaluation.generation.scenario import (
        MIN_ROUTE_CLEARANCE,
        build,
        ref_safe_zones,
        robot_zones,
        route_points,
        screen_zones,
    )
    from arena_evaluation.generation.scenario.populate import bt_type_source

    root = worlds_root / WORLD
    zones, robot = first_level(root)["zones"], dict(ROBOT)
    route = route_points(robot, zones)
    verdicts = screen_zones(
        zones,
        world_root=root,
        route=route,
        min_route_clearance=MIN_ROUTE_CLEARANCE,
        robot_zones=robot_zones(robot),
    )
    doc = build(
        "normal",
        zones=zones,
        robot=robot,
        seed=0,
        homes=ref_safe_zones(verdicts),
    )

    name = "pytest_room_centric"
    out = root / "scenarios" / name
    write_scenario(out, doc)
    if bt_type_source().is_file():
        shutil.copy(bt_type_source(), out / bt_type_source().name)
    try:
        yield name
    finally:
        shutil.rmtree(out, ignore_errors=True)


@pytest.fixture
def worlds_root() -> Path:
    # not the package's own `__file__`: that lives under `build/`, worlds resolve under `install/.../share`, and the loader never sees a scenario written to the wrong one
    root = Path(WorldIdentifier(WORLD).resolve_sync().path).parent
    if not root.is_dir():
        pytest.skip("worlds not on this install")
    return root


def test_every_generated_agent_survives_the_load(loaded, generated_scenario, worlds_root) -> None:
    from arena_evaluation.generation.formats import read_scenario

    written = read_scenario(worlds_root / WORLD / "scenarios" / generated_scenario / "scenario.yaml")
    # A two-element waypoint drops the agent silently, so a count mismatch is the symptom to
    # catch here rather than an exception.
    assert len(loaded.humans) == len(written["dynamic"])
    assert {a.name for a in loaded.humans} == {a["name"] for a in written["dynamic"]}


def test_a_zone_reference_resolves_to_a_point_inside_that_zone(loaded, generated_scenario, worlds_root) -> None:
    from arena_evaluation.generation.formats import read_scenario

    written = read_scenario(worlds_root / WORLD / "scenarios" / generated_scenario / "scenario.yaml")
    referenced = {a["name"]: a["pose"] for a in written["dynamic"] if isinstance(a["pose"], str)}
    if not referenced:
        pytest.skip("no zone screened clean in this world")

    world = WorldIdentifier(WORLD).resolve_sync().load()
    polygons = {z.name: [(c.x, c.y) for c in z.polygon] for level in world.levels.values() for z in level.zones}
    for agent in loaded.humans:
        zone = referenced.get(agent.name)
        if zone is None:
            continue
        assert _inside((agent.pose.position.x, agent.pose.position.y), polygons[zone]), f"{agent.name} resolved outside {zone}"


def test_every_waypoint_reference_resolves_to_its_own_zone(loaded, generated_scenario, worlds_root) -> None:
    from arena_evaluation.generation.formats import read_scenario

    written = read_scenario(worlds_root / WORLD / "scenarios" / generated_scenario / "scenario.yaml")
    by_name = {a["name"]: a for a in written["dynamic"]}
    world = WorldIdentifier(WORLD).resolve_sync().load()
    polygons = {z.name: [(c.x, c.y) for c in z.polygon] for level in world.levels.values() for z in level.zones}
    for agent in loaded.humans:
        names = by_name[agent.name]["waypoints"]
        assert len(agent.waypoints) == len(names)
        for waypoint, zone in zip(agent.waypoints, names, strict=True):
            assert isinstance(zone, str), f"{agent.name} carries a coordinate waypoint"
            assert _inside((waypoint.x, waypoint.y), polygons[zone]), f"{agent.name}: {zone} resolved outside"


def _inside(pt, poly):
    x, y = pt
    hit = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            hit = not hit
    return hit
