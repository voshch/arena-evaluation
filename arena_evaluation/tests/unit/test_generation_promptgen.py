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
    (root / "0").mkdir(parents=True)
    (root / "0" / "world.yaml").write_text(yaml.safe_dump({"zones": ZONES}))
    res = 0.1
    w, h = int(32 / res), int(8 / res)
    px = np.full((h, w), 0, dtype=np.uint8)  # occupied everywhere...

    def free(x0, y0, x1, y1):
        px[h - int(y1 / res) : h - int(y0 / res), int(x0 / res) : int(x1 / res)] = 255

    free(0.3, 0.3, 7.7, 7.7)
    free(7.7, 3.0, 24.3, 5.0)
    free(24.3, 0.3, 31.7, 7.7)
    Image.fromarray(px, mode="L").save(root / "0" / "map.png")
    (root / "0" / "map.yaml").write_text(yaml.safe_dump({"image": "map.png", "resolution": res, "origin": [0.0, 0.0, 0.0], "occupied_thresh": 0.65, "negate": 0}))
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
    world = WorldContext.load(worlds, "toy", doc)
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


def test_blackout_retunes_speed_and_vision_at_the_onset(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("The lights go out and everyone slows down.", ["A:blackout"])
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok" and result.record.steps == ["A:blackout"]
    # The population is untouched until the onset: base types, base speeds.
    agents = result.doc["dynamic"]
    assert not any(a["agent"]["agent_type"].startswith("./types/") for a in agents)
    assert agents == doc["dynamic"]
    retunes = [e for e in result.effects if e["type"] == "retune"]
    assert retunes and all(e["when"] == {"at": 10.0} and "vision_range_low" in e["profile"] and e["speed_scale"] == pytest.approx(0.6 / 1.1, abs=0.01) for e in retunes)
    assert sorted(n for e in retunes for n in e["scope"]) == sorted(a["name"] for a in agents)
    assert all(ref in result.files.docs for ref in {e["profile"] for e in retunes})


def test_evacuation_rallies_the_crowd_present_at_step_a(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("The fire alarm goes off and everyone heads for the exit. A child crosses the robot's path from the right right in the robot's way.", ["A:evacuation", "C:approach"], C={"case": "approach", "profile": "child", "angle": 90, "route_fraction": 0.5})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok"
    *retunes, rally, intercept = result.effects
    assert rally["type"] == "rally" and rally["target"] == "central_hallway" and rally["scope"] == ["normal_1", "normal_2", "normal_3"]
    assert rally["when"] == {"at": 8.0}
    # Everyone hurries from the same moment, with the engine's speed cap lifted so the hurry is a run.
    assert retunes and all(e["type"] == "retune" and e["when"] == {"at": 8.0} and e["speed_scale"] > 1.0 and "max_speed_high" in e["profile"] for e in retunes)
    assert result.files.docs[retunes[0]["profile"]]["max_velocity"]["mean"] == 2.6
    assert intercept["type"] == "intercept" and intercept["angle"] == 90.0 and intercept["profile"] == "child"
    assert result.record.owned["C"] == ["edge_0"]


def test_busy_injects_and_stampede_runs_on_a_small_crowd(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("It is visiting hours and the place is packed.", ["A:busy"])
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert len(result.doc["dynamic"]) > 3 and result.record.owned["A"][0].startswith("busy_")
    prompt = _prompt("The break is over and the whole crowd rushes back to work all at once.", ["A:stampede"])
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok"
    *retunes, rally = result.effects
    assert rally["type"] == "rally" and rally["target"] == "open_work_area" and len(result.doc["dynamic"]) >= 6
    assert retunes and all("repulsion_low" in e["profile"] and e["speed_scale"] > 1.0 for e in retunes)


def test_blocking_line_spans_the_corridor(toy) -> None:
    worlds, doc, base, world = toy
    prompt = _prompt("Two nurses stand side by side across the corridor right in the robot's way, chatting, and do not move aside.", ["D:interaction_unbroken"], D={"case": "interaction_unbroken", "variant": "blocking", "members": ["nurse", "nurse"], "place": {"route_fraction": 0.5}})
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    members = [a for a in result.doc["dynamic"] if a["name"].startswith("d_")]
    assert len(members) >= 2
    # Until the onset they are ordinary walkers on their base profile, starting in a room.
    assert all(a["agent"]["agent_type"] == profile("nurse").base and isinstance(a["pose"], str) and a["waypoint_mode"] == "reverse" for a in members)
    form, retune = result.effects
    assert form["type"] == "rally" and form["label"].endswith(":form") and form["when"] == {"at": 8.0} and form["scope"] == sorted(a["name"] for a in members)
    ys = sorted(pt[1] for pt in form["targets"])
    assert ys[-1] - ys[0] >= 0.5 * (len(members) - 1)
    assert all(abs(pt[0] - world.point_at(0.5)[0][0]) < 1.5 for pt in form["targets"])
    assert retune["type"] == "retune" and "repulsion_low" in retune["profile"] and retune["when"] == form["when"] and retune["scope"] == form["scope"]
    assert result.record.formation["variant"] == "blocking" and abs(result.record.formation["centre"][0] - world.point_at(0.5)[0][0]) < 1.5


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
    assert elder["agent"]["agent_type"] == "elder"  # unchanged until the onset
    (retune,) = result.effects
    assert retune["type"] == "retune" and retune["scope"] == ["normal_1"] and retune["when"] == {"at": 10.0}
    assert retune["profile"] == "./types/elder__reaction_high_repulsion_low.yaml"
    assert result.files.docs[retune["profile"]]["local_planner_params"]["relaxation_time"]["mean"] == 1.0


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


def test_empty_base_is_denied_before_any_model_call(toy, monkeypatch) -> None:
    import arena_evaluation.generation.promptgen.pipeline as pl

    worlds, doc, base, world = toy
    empty = {"robots": doc["robots"], "dynamic": []}
    monkeypatch.setattr(pl, "decide_d", lambda *a, **k: pytest.fail("Step D was consulted on a denied slot"))
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


def test_llm_a_overrides_a_silent_lexicon(toy, monkeypatch) -> None:
    import arena_evaluation.generation.promptgen.pipeline as pl
    from arena_evaluation.generation.promptgen.spec import ADecision

    worlds, doc, base, world = toy
    from arena_evaluation.generation.promptgen import step_a_llm

    monkeypatch.setattr(step_a_llm, "decide", lambda text, rec, **k: (ADecision(case="evacuation"), [{"mention": "a visiter", "profile": "adult", "count": 1}], []))
    called = {}
    real = pl.decide_d
    monkeypatch.setattr(pl, "decide_d", lambda *a, **k: called.setdefault("who", k.get("who")) and None or real(*a, **k))
    prompt = _prompt("An evacation is caled; evrybody hurrys out.", ["A:evacuation"])
    result = generate(prompt, world=world, base_id="normal_a", base_doc=doc, base_dir=base, oracle=False)
    assert result.record.a is not None and result.record.a.case == "evacuation"
    assert any("lexicon silent" in w for w in result.record.warnings)
    assert "1 x adult" in (called.get("who") or "")
