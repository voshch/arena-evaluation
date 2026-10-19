from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml
from arena_evaluation.generation.formats import read_scenario
from arena_evaluation.generation.promptgen import prompts as P
from arena_evaluation.generation.promptgen.lexicon import match
from arena_evaluation.generation.promptgen.pipeline import generate, write_result
from arena_evaluation.generation.promptgen.profiles import ProfileError, derive, derived_name, profile, tier_value
from arena_evaluation.generation.promptgen.spec import SchemaError, parse_b, parse_c, parse_d, parse_when
from arena_evaluation.generation.promptgen.world import WorldContext
from arena_evaluation.generation.propose import client

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("PIL")


def _zone(name, x0, y0, x1, y1):
    return {"name": name, "corners": [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}], "walls": [], "doors": [], "entities": {"static": []}}


# two rooms joined by a 2 m corridor: west room x 0..8, corridor x 8..24 at y 3..5, east room x 24..32, rooms y 0..8
ZONES = [_zone("reception", 0, 0, 8, 8), _zone("central_hallway", 8, 3, 24, 5), _zone("open_work_area", 24, 0, 32, 8)]
ROBOT = {"start": [2.0, 4.0, 0.0], "phases": [{"goto": [30.0, 4.0, 0.0]}, {"goto": [2.0, 4.0, 3.14]}]}


def _world(tmp_path):
    from PIL import Image

    root = tmp_path / "worlds" / "toy"
    (root / "map").mkdir(parents=True)
    zones = [{"name": z["name"], "polygon": [[c["x"], c["y"]] for c in z["corners"]], "floor": False} for z in ZONES]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones}]}))
    res = 0.1
    w, h = int(32 / res), int(8 / res)
    px = np.full((h, w), 0, dtype=np.uint8)  # occupied everywhere...

    def free(x0, y0, x1, y1):
        px[h - int(y1 / res) : h - int(y0 / res), int(x0 / res) : int(x1 / res)] = 255

    free(0.3, 0.3, 7.7, 7.7)
    free(7.7, 3.0, 24.3, 5.0)
    free(24.3, 0.3, 31.7, 7.7)
    Image.fromarray(px, mode="L").save(root / "map" / "0.png")
    (root / "map" / "0.yaml").write_text(yaml.safe_dump({"image": "0.png", "resolution": res, "origin": [0.0, 0.0, 0.0], "occupied_thresh": 0.65, "negate": 0}))
    base = root / "scenarios" / "normal_a"
    base.mkdir(parents=True)
    doc = {
        "robots": [ROBOT],
        "dynamic": [
            {"name": "normal_1", "model": "female_adult_medical_01", "pose": "reception", "agent": {"agent_type": "elder", "desired_velocity": 0.7}, "waypoints": ["reception", "open_work_area"], "waypoint_mode": "reverse"},
            {"name": "normal_2", "model": "male_adult_medical_01", "pose": "open_work_area", "agent": {"agent_type": "adult", "desired_velocity": 1.1}, "waypoints": ["open_work_area", "reception"], "waypoint_mode": "reverse"},
            {"name": "normal_3", "model": "female_adult_police_01", "pose": "reception", "agent": {"agent_type": "hurried", "desired_velocity": 1.6}, "waypoints": ["reception", "open_work_area"], "waypoint_mode": "reverse"},
        ],
    }
    (base / "scenario.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    return tmp_path / "worlds", doc, base


@pytest.fixture
def toy(tmp_path: Path) -> tuple[Path, dict, Path, WorldContext]:
    worlds, doc, base = _world(tmp_path)
    world = WorldContext.load(worlds / "toy", doc)
    return worlds, doc, base, world


def _prompt(text, targets=(), **oracle):
    return P.Prompt(id="t_001", text=text, targets=tuple(targets), oracle=dict(oracle))


def test_tiers_scale_speed_by_profile() -> None:
    assert tier_value("speed", "high", prof=profile("child")) < tier_value("speed", "high", prof=profile("hurried"))
    assert tier_value("vision_range", "low") == 2.0


def test_aliases_resolve() -> None:
    assert profile("old man").name == "elder" and profile("kid").name == "child"
    with pytest.raises(ProfileError):
        profile("dragon")


def test_derive_pins_point_masses_and_refuses_bt_params_on_plain_types() -> None:
    doc = derive({"extends": "adult"}, {"vision_range": "low", "speed": "high"}, name="x", prof=profile("adult"))
    assert doc["perception"]["vision_range"] == {"mean": 2.0, "std": 0.0, "clip_low": 2.0, "clip_high": 2.0}
    assert doc["desired_velocity"]["mean"] == 1.7
    with pytest.raises(ProfileError, match="behaviour-tree"):
        derive({"extends": "adult"}, {"need_decay": "high"}, name="y")
    assert derived_name("./talker.yaml", {"repulsion": "low"}) == "talker__repulsion_low"


@pytest.mark.parametrize(
    "text, case",
    [
        ("The fire alarm goes off and everyone heads for the exit.", "evacuation"),
        ("A power cut leaves the floor in the dark.", "blackout"),
        ("The break is over and everyone surges back to work all at once.", "stampede"),
        ("It is visiting hours and the place is packed.", "busy"),
    ],
)
def test_lexicon_matches_the_taxonomy(text, case) -> None:
    got = match(text)
    assert got is not None and got.case == case


def test_time_phrases_are_read_from_the_prompt() -> None:
    from arena_evaluation.generation.promptgen.lexicon import when_in

    assert when_in("Everyone is on their phone, after half a minute.").as_dict() == {"at": 30.0}
    assert when_in("The lights go out, after a while. The elderly are slow, once the robot is halfway.", exclude_cases=("blackout",)).as_dict() == {"route_fraction": 0.5}
    assert when_in("A nurse stops in front of the robot for a while.") is None
    assert match("The lights go out, twenty seconds in.").when.as_dict() == {"at": 20.0}
    # A semicolon does not end the sentence: the phrase after it still belongs to the evacuation.
    assert match("An evacuation is called; everybody makes for the exit, as soon as the robot sets off.").when.as_dict() == {"route_fraction": 0.1}


def test_lexicon_ignores_the_unrelated() -> None:
    assert match("An old man walks straight at the robot.") is None


def test_busy_reads_its_multiplier() -> None:
    assert match("The floor is very busy today.").k == 3.0


def test_when_phrases_and_mappings() -> None:
    from arena_evaluation.generation.promptgen.spec import When

    assert parse_when("as the robot arrives", default=When(at=0.0), where="x").robot_within == 3.0
    assert parse_when({"at": 4}, default=When(), where="x").at == 4.0
    with pytest.raises(SchemaError):
        parse_when({"at": 1, "robot_within": 2, "route_fraction": 0.5}, default=When(), where="x")


def test_decisions_reject_numbers_where_tiers_are_asked() -> None:
    with pytest.raises(SchemaError, match="not one of"):
        parse_b({"case": "intrinsic", "select": {"amount": "half"}, "params": {"speed": 0.5}})
    with pytest.raises(SchemaError, match="allowed parameter"):
        parse_b({"case": "intrinsic", "select": {"amount": "half"}, "params": {"mass": "low"}})
    with pytest.raises(SchemaError, match="B.params"):  # a model once answered a string here; that is a complaint, not a crash
        parse_b({"case": "intrinsic", "select": {"amount": "half"}, "params": "speed: low"})
    assert parse_c({"case": "approach", "angle": 450}).angle == 90.0
    with pytest.raises(SchemaError, match="two to"):
        parse_d({"case": "formation_break", "members": ["adult"]})
    assert parse_d({"case": "formation_break", "members": ["adult", "adult"]}).when.robot_within == 3.0


def test_world_plans_the_route_and_places_on_it(toy) -> None:
    _worlds, _doc, _base, world = toy
    # The 60 s episode budget truncates the toy's 28 m route at MAX_LEG_M (20 m).
    assert world.route[0] == (2.0, 4.0) and abs(world.route[-1][0] - 22.0) < 1.0
    from arena_evaluation.generation.promptgen.world import MAX_LEG_M, _length

    assert _length(world.route) <= MAX_LEG_M * 1.2
    pt, heading = world.point_at(0.5)
    assert pt == pytest.approx((12.0, 4.0), abs=0.3) and heading == pytest.approx((1.0, 0.0), abs=0.05)
    width, _l, _r = world.passage_width(pt, heading)
    assert 1.6 < width < 2.4
    assert world.valid((16.0, 4.0)) is not None and world.valid((16.0, 4.0), on_route=True) is None
    assert world.valid((4.0, 7.0)) is None and world.zone_of((4.0, 7.0)) == "reception"
    assert world.exit_zone() == "central_hallway" and world.work_zone() == "open_work_area"


def test_the_shipped_prompts_round_trip_through_the_lexicon() -> None:
    prompts = P.load_sets("")
    assert len(prompts) == 13 and len({p.id for p in prompts}) == 13
    for prompt in prompts:
        wanted = [t for t in prompt.targets if t.startswith("A:")]
        if wanted:
            got = match(prompt.text)
            assert got is not None and f"A:{got.case}" == wanted[0], prompt.text
    assert {len(p.targets) for p in prompts} >= {1, 2}


def _zones_of(world, steps: dict) -> list[str]:
    return [world.zone_of((s["target_pose"]["x"], s["target_pose"]["y"])) for s in steps.values()]


def test_blackout_bakes_each_walkers_route_into_a_tree_tuned_at_the_onset(toy) -> None:
    worlds, doc, base, world = toy
    result = generate(_prompt("The lights go out and everyone slows down.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok" and result.record.steps == ["A:blackout"]
    (stimulus,) = result.effects
    assert stimulus == {"type": "stimulus", "label": "A:blackout", "scope": ["normal_1", "normal_2", "normal_3"], "need": "blackout", "when": {"at": 10.0}}
    agents = {a["name"]: a for a in result.doc["dynamic"]}
    # pose, waypoints and spawn speed stay the base's; only the type changes, and only the final types are written
    assert all({k: v for k, v in agents[a["name"]].items() if k != "agent"} == {k: v for k, v in a.items() if k != "agent"} for a in doc["dynamic"])
    assert agents["normal_1"]["agent"]["desired_velocity"] == 0.7
    assert set(result.files.docs) == {a["agent"]["agent_type"] for a in agents.values()}
    elder = result.files.docs[agents["normal_1"]["agent"]["agent_type"]]
    assert elder["extends"] == "elder" and elder["mode"] == "behavior_tree" and elder["sequences"]["walk"]["then"] == "walk"
    assert _zones_of(world, elder["sequences"]["walk"]["steps"]) == ["reception", "open_work_area"]
    assert elder["tunes"] == [{"when": {"blackout": {"above": 50.0}}, "tune": {"perception": {"vision_range": 2.0}, "desired_velocity": {"mul": pytest.approx(0.6 / 1.1, abs=0.01)}}}]

    loader = pytest.importorskip("arena_humansim.core.agents.loader")
    out = write_result(result, worlds / "toy")
    loaded = loader.load_agent_type_from_file(out / agents["normal_1"]["agent"]["agent_type"])
    assert loaded.mode == "behavior_tree" and len(loaded.sequences["walk"].steps) == 2


def test_talkers_no_step_touches_keep_their_type_file(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("A child crosses the robot's path from the right right in the robot's way.", ["C:approach"], C={"case": "approach", "profile": "child", "angle": 90, "route_fraction": 0.5})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=_with_talkers(doc, base), base_dir=base, oracle=True)
    out = write_result(result, worlds / "toy")
    assert {a["agent"]["agent_type"] for a in result.doc["dynamic"] if a["name"].startswith("talker_")} == {"./talker.yaml"}
    assert (out / "talker.yaml").read_text() == (base / "talker.yaml").read_text()


def test_walk_route_bakes_the_waypoint_mode(toy) -> None:
    from arena_evaluation.generation.promptgen.step_a import walk_route

    worlds, doc, base, world = toy
    walker = {"name": "w", "waypoints": ["reception", "open_work_area", [20.0, 4.0, 0.0]], "waypoint_mode": "reverse"}
    there_and_back = walk_route(walker, world)
    assert [world.zone_of(p) for p in there_and_back] == ["reception", "open_work_area", "central_hallway", "open_work_area"]
    assert there_and_back[2] == (20.0, 4.0) and walk_route(walker, world) == there_and_back
    shuffled = walk_route({**walker, "waypoint_mode": "random"}, world)
    stops = [world.zone_of(p) for p in shuffled]
    assert len(stops) == 6 and all(a != b for a, b in zip(stops, stops[1:]))


def test_evacuation_switches_the_crowd_present_at_step_a(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("The fire alarm goes off and everyone heads for the exit. A child crosses the robot's path from the right right in the robot's way.", ["A:evacuation", "C:approach"], C={"case": "approach", "profile": "child", "angle": 90, "route_fraction": 0.5})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok"
    stimulus, intercept = result.effects
    assert stimulus == {"type": "stimulus", "label": "A:evacuation", "scope": ["normal_1", "normal_2", "normal_3"], "need": "alarm", "when": {"at": 8.0}}
    # Everyone hurries from the same moment, with the engine's speed cap lifted so the hurry is a run, to the exit.
    for agent in result.doc["dynamic"][:3]:
        evacuate = result.files.docs[agent["agent"]["agent_type"]]["sequences"]["evacuate"]
        assert evacuate["tune"]["max_velocity"] == 2.6 and evacuate["tune"]["desired_velocity"]["mul"] > 1.0
        assert _zones_of(world, {"leave": evacuate["steps"]["leave"]}) == ["central_hallway"]
    assert intercept["type"] == "intercept" and intercept["angle"] == 90.0 and intercept["profile"] == "child"
    assert result.record.owned["C"] == ["edge_0"]


def _with_talkers(doc: dict, base: Path) -> dict:
    from arena_evaluation.generation.scenario.populate import bt_type_source

    (base / "talker.yaml").write_text(bt_type_source().read_text())
    talkers = [{"name": f"talker_{i}", "model": "male_adult_medical_01", "pose": "reception", "agent": {"agent_type": "./talker.yaml", "desired_velocity": 1.1}, "waypoints": ["reception"]} for i in (1, 2)]
    return {**doc, "dynamic": [*doc["dynamic"], *talkers]}


def test_evacuation_switches_talkers_and_walkers_by_one_stimulus(toy, tmp_path: Path) -> None:
    worlds, doc, base, world = toy
    doc = _with_talkers(doc, base)
    result = generate(_prompt("The fire alarm goes off and everyone heads for the exit.", ["A:evacuation"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok"
    (stimulus,) = result.effects
    assert stimulus == {"type": "stimulus", "label": "A:evacuation", "scope": ["normal_1", "normal_2", "normal_3", "talker_1", "talker_2"], "need": "alarm", "when": {"at": 8.0}}
    # each agent its own type and its own point in the exit zone
    refs = {a["name"]: a["agent"]["agent_type"] for a in result.doc["dynamic"]}
    ref = refs["talker_1"]
    assert ref == "./types/talker__evacuate_talker_1.yaml" and refs["talker_2"] == "./types/talker__evacuate_talker_2.yaml"
    leaves = [result.files.docs[r]["sequences"]["evacuate"]["steps"]["leave"]["target_pose"] for r in refs.values()]
    assert len({(p["x"], p["y"]) for p in leaves}) == len(refs) and set(_zones_of(world, {str(i): {"target_pose": p} for i, p in enumerate(leaves)})) == {"central_hallway"}
    switched = result.files.docs[ref]
    assert switched["needs"]["alarm"] == {"initial": 0.0, "decay_rate": 0.0}
    assert all({"when": {"alarm": {"above": 50.0}}, "goto": "evacuate"} in switched["sequences"][s]["transitions"] for s in ("chat", "step_aside"))
    assert _zones_of(world, {"leave": switched["sequences"]["evacuate"]["steps"]["leave"]}) == ["central_hallway"]
    assert switched["sequences"]["evacuate"]["tune"] == {"max_velocity": 2.6, "desired_velocity": {"mul": pytest.approx(1.7 / 1.1, abs=0.01)}}
    walker = result.files.docs[result.doc["dynamic"][1]["agent"]["agent_type"]]
    assert {"when": {"alarm": {"above": 50.0}}, "goto": "evacuate"} in walker["sequences"]["walk"]["transitions"]

    loader = pytest.importorskip("arena_humansim.core.agents.loader")
    out = write_result(result, worlds / "toy")
    assert loader.load_agent_type_from_file(out / ref).sequences["evacuate"].tune.fields["max_velocity"] == (2.6,)
    assert all(loader.load_agent_type_from_file(out / a["agent"]["agent_type"]).mode == "behavior_tree" for a in result.doc["dynamic"])
    block = pytest.importorskip("task_generator.tasks.obstacles.edge_case.scenario_block")
    assert [e.kind for e in block.read(out / "scenario.yaml").effects] == ["stimulus"]


@pytest.mark.parametrize(
    ("text", "targets", "oracle", "need", "tune"),
    [
        ("The lights go out and everyone slows down.", ["A:blackout"], {}, "blackout", {"perception": {"vision_range": 2.0}, "desired_velocity": {"mul": pytest.approx(0.6 / 1.1, abs=0.01)}}),
        ("Half the people here cannot see far.", ["B:intrinsic"], {"B": {"case": "intrinsic", "select": {"amount": "all"}, "params": {"vision_range": "low"}}}, "intrinsic", {"perception": {"vision_range": 2.0}}),
    ],
)
def test_trees_hold_a_tune_while_the_need_is_up(toy, text: str, targets: list[str], oracle: dict, need: str, tune: dict) -> None:
    worlds, doc, base, world = toy
    result = generate(_prompt(text, targets, **oracle), world=world, base_id="normal_a", base_doc=_with_talkers(doc, base), base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    (stimulus,) = result.effects
    assert stimulus["scope"] == ["normal_1", "normal_2", "normal_3", "talker_1", "talker_2"] and stimulus["need"] == need
    ref = next(a["agent"]["agent_type"] for a in result.doc["dynamic"] if a["name"] == "talker_1")
    tuned = result.files.docs[ref]
    assert tuned["needs"][need] == {"initial": 0.0, "decay_rate": 0.0}
    assert tuned["tunes"] == [{"when": {need: {"above": 50.0}}, "tune": tune}]

    loader = pytest.importorskip("arena_humansim.core.agents.loader")
    out = write_result(result, worlds / "toy")
    assert loader.load_agent_type_from_file(out / ref).tunes[0].tune.fields["perception.vision_range"] == (2.0,)


def test_a_stampede_switches_everyone_instead_of_rerouting(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("The break is over and the whole crowd rushes back to work all at once.", ["A:stampede"])
    result = generate(prompt, world=world, base_id="normal_a", base_doc=_with_talkers(doc, base), base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    assert [e["type"] for e in result.effects] == ["stimulus"]
    rush = result.files.docs["./types/talker__rush_talker_1.yaml"]["sequences"]["rush"]
    assert rush["tune"] == {"max_velocity": 2.6, "local_planner_params": {"repulsion_strength": 0.5}, "desired_velocity": {"mul": pytest.approx(1.7 / 1.1, abs=0.01)}}
    assert _zones_of(world, {"leave": rush["steps"]["leave"]}) == ["open_work_area"]


def test_busy_injects_and_stampede_runs_on_a_small_crowd(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("It is visiting hours and the place is packed.", ["A:busy"])
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert len(result.doc["dynamic"]) > 3 and result.record.owned["A"][0].startswith("busy_")
    prompt = _prompt("The break is over and the whole crowd rushes back to work all at once.", ["A:stampede"])
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok"
    (stimulus,) = result.effects
    assert stimulus["need"] == "alarm" and len(stimulus["scope"]) == len(result.doc["dynamic"]) >= 6
    assert all(result.files.docs[a["agent"]["agent_type"]]["sequences"]["rush"]["tune"]["local_planner_params"] == {"repulsion_strength": 0.5} for a in result.doc["dynamic"])


def test_blocking_line_spans_the_corridor(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("Two nurses stand side by side across the corridor right in the robot's way, chatting, and do not move aside.", ["D:interaction_unbroken"], D={"case": "interaction_unbroken", "variant": "blocking", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    members = [a for a in result.doc["dynamic"] if a["name"].startswith("d_")]
    assert len(members) >= 2
    # Until the onset they walk their own route from a room; at it a stimulus sends each to its spot, where it stands and stops yielding.
    assert all(isinstance(a["pose"], str) and a["waypoint_mode"] == "reverse" for a in members)
    (form,) = result.effects
    assert form["type"] == "stimulus" and form["label"].endswith(":form") and form["need"] == "form" and form["when"] == {"at": 8.0}
    assert form["scope"] == sorted(a["name"] for a in members)
    types = [result.files.docs[a["agent"]["agent_type"]] for a in members]
    assert all(t["extends"] == profile("nurse").base and "walk" in t["sequences"] for t in types)
    spots = [t["sequences"]["form"]["steps"]["form"]["target_pose"] for t in types]
    ys = sorted(pt["y"] for pt in spots)
    assert ys[-1] - ys[0] >= 0.5 * (len(members) - 1)
    assert all(abs(pt["x"] - world.point_at(0.5)[0][0]) < 1.5 for pt in spots)
    assert all(t["sequences"]["form"]["tune"] == {"local_planner_params": {"repulsion_strength": 0.5}} for t in types)
    assert result.record.formation["variant"] == "blocking" and abs(result.record.formation["centre"][0] - world.point_at(0.5)[0][0]) < 1.5


def test_a_talking_pair_stands_in_the_corridor_too_close_to_drive_between(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("Two nurses stand talking right in the robot's way.", ["D:interaction_unbroken"], D={"case": "interaction_unbroken", "variant": "standing", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    spots = [result.files.docs[a["agent"]["agent_type"]]["sequences"]["form"]["steps"]["form"]["target_pose"] for a in result.doc["dynamic"] if a["name"].startswith("d_")]
    (a, b) = [(s["x"], s["y"]) for s in spots]
    # the 2 m corridor could not hold the old 1.5 m pair, which stood beside the route in a room
    assert math.dist(a, b) == pytest.approx(0.9, abs=0.05)
    assert world.route_clearance(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)) < 0.5


def test_formation_break_scatters_on_approach(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("Three colleagues stand in a circle in the reception; they scatter as the robot arrives.", ["D:formation_break"], D={"case": "formation_break", "variant": "circle", "members": ["adult", "adult", "adult"], "place": {"zone": "reception"}, "when": {"robot_within": 3.0}, "break": "scatter"})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    form, scatter = result.effects
    # They walk first, form the circle at 8 s, and break when the robot comes near the circle.
    assert form["type"] == "rally" and form["when"] == {"at": 8.0} and len(form["targets"]) == 3
    assert scatter["type"] == "scatter" and scatter["when"]["robot_within"] >= 3.0 and len(scatter["scope"]) == 3
    centre = tuple(scatter["when"]["of"])
    assert centre == tuple(result.record.formation["centre"]) and world.zone_of(centre) == "reception"
    members = [a for a in result.doc["dynamic"] if a["name"] in scatter["scope"]]
    assert all(isinstance(a["pose"], str) and a["pose"] != "reception" for a in members)


def test_stubborn_is_a_head_on_intercept_that_does_not_yield(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("An old man walks towards the robot and refuses to give way.", ["D:stubborn"], D={"case": "stubborn", "profile": "elder", "route_fraction": 0.4})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    (effect,) = result.effects
    assert effect["type"] == "intercept" and effect["angle"] == 180.0 and "repulsion_low" in effect["profile"]
    assert effect["speed"] < 1.0


def test_b_tunes_only_unowned_agents(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("The elderly people are slow to react and do not get out of the way.", ["B:intrinsic"], B={"case": "intrinsic", "select": {"profile": "elder"}, "params": {"reaction": "high", "repulsion": "low"}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.owned["B"] == ["normal_1"]
    elder = next(a for a in result.doc["dynamic"] if a["name"] == "normal_1")
    (stimulus,) = result.effects
    assert stimulus == {"type": "stimulus", "label": "B:intrinsic", "scope": ["normal_1"], "need": "intrinsic", "when": {"at": 10.0}}
    tuned = result.files.docs[elder["agent"]["agent_type"]]
    assert tuned["extends"] == "elder" and tuned["tunes"] == [{"when": {"intrinsic": {"above": 50.0}}, "tune": {"local_planner_params": {"relaxation_time": 1.0, "repulsion_strength": 0.5}}}]
    assert all(a["agent"]["agent_type"] in ("adult", "hurried") for a in result.doc["dynamic"] if a["name"] != "normal_1")


def test_an_untunable_parameter_holds_from_the_start(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("The elderly people are wide and slow to react.", ["B:intrinsic"], B={"case": "intrinsic", "select": {"profile": "elder"}, "params": {"radius": "high", "reaction": "high"}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    assert [e["type"] for e in result.effects] == ["stimulus"]
    elder = next(a for a in result.doc["dynamic"] if a["name"] == "normal_1")
    tuned = result.files.docs[elder["agent"]["agent_type"]]
    assert tuned["agent_radius"]["mean"] == 0.4 and tuned["tunes"][0]["tune"] == {"local_planner_params": {"relaxation_time": 1.0}}


def test_erratic_route_is_a_tree_sequence_for_walkers_and_trees_alike(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("Everyone keeps changing where they are going.", ["B:erratic_route"], B={"case": "erratic_route", "select": {"amount": "all"}, "period": 3.0})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=_with_talkers(doc, base), base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    (stimulus,) = result.effects
    assert stimulus["type"] == "stimulus" and stimulus["need"] == "erratic" and stimulus["scope"] == ["normal_1", "normal_2", "normal_3", "talker_1", "talker_2"]
    for agent in result.doc["dynamic"]:
        erratic = result.files.docs[agent["agent"]["agent_type"]]["sequences"]["erratic"]
        assert erratic["then"] == erratic["on_failure"] == "erratic" and len(erratic["steps"]) >= 2
        assert all(step["patience"] == {"mean": 3.0} for step in erratic["steps"].values())

    loader = pytest.importorskip("arena_humansim.core.agents.loader")
    out = write_result(result, worlds / "toy")
    written = read_scenario(out / "scenario.yaml")
    for agent in written["dynamic"]:
        assert loader.load_agent_type_from_file(out / agent["agent"]["agent_type"]).mode == "behavior_tree"
    radius = {a["name"]: a["agent"]["radius"] for a in written["dynamic"]}
    assert radius["normal_1"] == pytest.approx(0.23) and radius["normal_2"] == pytest.approx(0.25)


def test_a_reused_agent_without_a_slot_goes_back_unchanged(toy) -> None:
    worlds, doc, base, world = toy
    doc = _with_talkers(doc, base)
    reuse = ["normal_1", "normal_2", "normal_3", "talker_1"]
    prompt = _prompt("Two nurses stand side by side across the corridor right in the robot's way.", ["D:interaction_unbroken"], D={"case": "interaction_unbroken", "variant": "blocking", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}, "reuse": reuse})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    # a 2 m corridor holds three at LINE_SPACING, so the fourth reused agent has no slot
    assert sorted(result.record.owned["D"]) == ["normal_1", "normal_2", "normal_3"]
    (talker,) = [a for a in result.doc["dynamic"] if a["name"] == "talker_1"]
    assert talker == next(a for a in doc["dynamic"] if a["name"] == "talker_1")


def test_c_conflicting_with_d_is_a_recorded_failure(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt(
        "Two nurses stand talking right in the robot's way. A visitor stands still right in the robot's way, blocking it.",
        ["D:interaction_unbroken", "C:blocking"],
        D={"case": "interaction_unbroken", "variant": "standing", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}},
        C={"case": "blocking", "mode": "stand", "route_fraction": 0.5},
    )
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    # The oracle cannot answer a complaint, so C is moved along the route instead.
    assert result.record.status == "ok", result.record.failures
    assert result.record.c is not None and result.record.c.route_fraction != 0.5
    assert any("moved to route fraction" in w for w in result.record.warnings)


def test_written_scenario_is_self_contained(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("The lights go out. A nurse stops right in front of the robot halfway along the robot's route and stares at it for a while.", ["A:blackout", "C:approach"], C={"case": "approach", "profile": "nurse", "angle": 180, "route_fraction": 0.5, "after": "stop_facing", "duration": 8})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    out = write_result(result, worlds / "toy")
    assert out is not None and (out / "scenario.yaml").is_file() and (out / "edge_case.yaml").is_file() and (out / "record.json").is_file()
    written = read_scenario(out / "scenario.yaml")
    block = written["edge_case"]
    assert block["prompt_id"] == "t_001" and block["steps"] == ["A:blackout", "C:approach"]
    assert block["effects"][-1]["after"] == "stop_facing"
    refs = [a["agent"]["agent_type"] for a in written["dynamic"]] + [e["profile"] for e in block["effects"] if e.get("profile")]
    assert any(ref.startswith("./types/") for ref in refs)
    for ref in refs:
        if ref.startswith("./"):
            assert (out / ref[2:]).is_file(), ref
    record = json.loads((out / "record.json").read_text())
    assert record["status"] == "ok" and record["dedup_key"] == result.record.dedup_key()


def test_robot_route_is_pinned_to_the_planned_points(toy) -> None:
    worlds, doc, base, world = toy
    result = generate(_prompt("The lights go out.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    (robot,) = result.doc["robots"]
    assert robot["start"][:2] == [round(world.start[0], 2), round(world.start[1], 2)] and len(robot["start"]) == 3
    assert robot["phases"] == [{"goto": [round(world.goal[0], 2), round(world.goal[1], 2)]}]
    assert doc["robots"][0] == ROBOT and len(ROBOT["phases"]) == 2  # the base is untouched
    assert any("route pinned" in w for w in result.record.warnings)


def test_a_needs_a_crowd(toy) -> None:
    worlds, doc, base, world = toy
    empty = {"robots": doc["robots"], "dynamic": []}
    prompt = _prompt("The fire alarm goes off and everyone heads for the exit.", ["A:evacuation"])
    result = generate(prompt, world=world, base_id="empty", base_doc=empty, base_dir=base, oracle=True)
    assert result.record.status == "denied" and result.record.failures[0].kind == "denied"
    d_slots = {"case": "interaction_unbroken", "variant": "standing", "members": ["adult", "adult", "adult"], "place": {"route_fraction": 0.5}}
    both = _prompt("The fire alarm goes off and everyone heads for the exit. Three colleagues stand talking halfway along the robot's route and will not stop.", ["A:evacuation", "D:interaction_unbroken"], D=d_slots)
    result = generate(both, world=world, base_id="empty", base_doc=empty, base_dir=base, oracle=True)
    assert result.record.status == "partial" and result.record.steps == ["D:interaction_unbroken"]
    assert result.record.failures[0].kind == "denied"


def test_empty_base_is_denied_before_any_model_call(toy) -> None:
    worlds, doc, base, world = toy
    empty = {"robots": doc["robots"], "dynamic": []}
    prompt = _prompt("Half the people here are rushing.", ["B:intrinsic"], B={"case": "intrinsic", "select": {"amount": "half"}, "params": {"speed": "high"}})
    result = generate(prompt, world=world, base_id="empty", base_doc=empty, base_dir=base, oracle=True)
    assert result.record.status == "denied" and not result.record.steps
    assert result.record.failures[0].kind == "denied"


def test_d_generates_its_members_on_an_empty_base(toy) -> None:
    worlds, doc, base, world = toy
    empty = {"robots": doc["robots"], "dynamic": []}
    prompt = _prompt("Three colleagues stand talking halfway along the robot's route and will not stop.", ["D:interaction_unbroken"], D={"case": "interaction_unbroken", "variant": "standing", "members": ["adult", "adult", "adult"], "place": {"route_fraction": 0.5}})
    result = generate(prompt, world=world, base_id="empty", base_doc=empty, base_dir=base, oracle=True)
    assert result.record.status == "ok" and len(result.record.owned["D"]) == 3
    assert len(result.doc["dynamic"]) == 3


def test_dedup_key_ignores_wording(toy) -> None:
    worlds, doc, base, world = toy
    a = generate(_prompt("The lights go out.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    b = generate(_prompt("A power cut leaves the floor in the dark.", ["A:blackout"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert a.record.dedup_key() == b.record.dedup_key()
    assert math.isfinite(world.route_length)


def test_busy_quantity_beats_intensifier() -> None:
    assert match("The floor is busy today, twice as many people as usual.").k == 2.0
    assert match("The floor is very busy today, twice as many people as usual.").k == 2.0


def test_system_prompts_document_every_phrase() -> None:
    from arena_evaluation.generation.promptgen.spec import PHRASES

    root = Path(P.__file__).parent
    bc = (root / "system_bc.md").read_text().lower()
    d = (root / "system_d.md").read_text().lower()
    for phrase, _when in PHRASES:
        assert phrase in bc, phrase
        assert phrase in d, phrase
    assert "up to 8" in d


def test_wc_and_storage_placements_are_refused(toy) -> None:
    from arena_evaluation.generation.promptgen.step_a import StepError
    from arena_evaluation.generation.promptgen.step_d import _place

    d = parse_d({"case": "interaction_unbroken", "variant": "standing", "members": ["adult", "adult"], "place": {"zone": "wc_west"}})
    with pytest.raises(StepError, match="nobody is placed"):
        _place(d, toy)


def test_slightly_busy_reads_k_one_and_a_half() -> None:
    got = match("The floor is slightly busier than usual.")
    assert got is not None and got.case == "busy" and got.k == 1.5


def test_place_vocabulary_is_documented() -> None:
    root = Path(P.__file__).parent
    bc = (root / "system_bc.md").read_text()
    d = (root / "system_d.md").read_text()
    for phrase, _fraction in P.PLACE_ROUTE:
        assert phrase in bc, phrase
        assert phrase in d, phrase


def test_member_limits_follow_the_variant() -> None:
    assert parse_d({"case": "interaction_unbroken", "variant": "blocking", "members": ["adult"] * 6}) is not None
    with pytest.raises(SchemaError):
        parse_d({"case": "interaction_unbroken", "variant": "standing", "members": ["adult"] * 6})


def test_kb_system_prompts_are_committed_builds() -> None:
    from arena_evaluation.generation.promptgen import build_kb

    built = build_kb.build()
    root = Path(P.__file__).parent
    for name, text in built.items():
        assert (root / name).read_text() == text, f"{name} is stale - run python3 -m arena_evaluation.generation.promptgen.build_kb"


def test_step_a_llm_parses_and_tolerates_typos() -> None:
    from arena_evaluation.generation.promptgen.llm import Oracle
    from arena_evaluation.generation.promptgen.spec import Record
    from arena_evaluation.generation.promptgen.step_a_llm import decide, who_line

    rec = Record(prompt_id="t", prompt="", world="w", base="b")
    answer = Oracle({"case": "evacuation", "when": {"at": 10}, "who": [{"mention": "a visiter", "profile": "visitor", "count": 1}, {"mention": "three colleagues", "profile": "adult", "count": 3}, {"mention": "a dragon", "profile": "dragon", "count": 1}]})
    a, who, warnings = decide("An evacation is caled.", rec, cache_dir=None, oracle=answer)
    assert a is not None and a.case == "evacuation" and a.when.at == 10.0
    assert [w["profile"] for w in who] == ["adult", "adult"] and "dragon" in warnings[0]
    assert who_line(who).startswith("who (actors read from the prompt): 1 x adult")


@pytest.mark.skipif(client.available(), reason="steps D and B/C would call the configured model for real")
def test_llm_a_overrides_a_silent_lexicon(toy, tmp_path: Path) -> None:
    from arena_evaluation.generation.promptgen import step_a_llm
    from arena_evaluation.generation.propose.client import prompt_sha256

    worlds, doc, base, world = toy
    text = "An evacation is caled; evrybody hurrys out."
    cache = tmp_path / "llm_cache"
    cache.mkdir()
    answer = json.dumps({"case": "evacuation", "who": [{"mention": "a visiter", "profile": "adult", "count": 1}]})
    (cache / f"{prompt_sha256(step_a_llm.A_MODEL, step_a_llm.system_prompt(), step_a_llm.user_prompt(text))}.json").write_text(json.dumps({"answer": answer}))
    result = generate(_prompt(text, ["A:evacuation"]), world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=False, cache_dir=cache)
    assert result.record.a is not None and result.record.a.case == "evacuation"
    assert any("lexicon silent" in w for w in result.record.warnings)
    assert result.record.provenance["who"] == [{"mention": "a visiter", "profile": "adult", "count": 1}]
