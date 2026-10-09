from __future__ import annotations

import random
from pathlib import Path

import pytest
import yaml
from arena_evaluation.generation.scenario import TALKERS, build
from arena_evaluation.generation.scenario.populate import KINEMATICS, MIX, ONLY_IN, mix_for, speed_for, world_kind


def zone(name: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"name": name, "corners": [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}]}


HOSPITAL = [
    zone("patient_ward", 2.0, 2.0, 12.0, 12.0),
    zone("nurse_station", 16.0, 2.0, 22.0, 8.0),
    zone("waiting_area", 16.0, 10.0, 22.0, 16.0),
    zone("pharmacy", 2.0, 20.0, 10.0, 26.0),
    zone("main_hallway", 12.5, 2.0, 15.5, 26.0),
]
OFFICE = [
    zone("open_work_area", 2.0, 2.0, 12.0, 12.0),
    zone("meeting_room_1", 16.0, 2.0, 22.0, 8.0),
    zone("kitchen", 16.0, 10.0, 22.0, 16.0),
    zone("reception", 2.0, 20.0, 10.0, 26.0),
    zone("main_hallway", 12.5, 2.0, 15.5, 26.0),
]
FLAT = [
    zone("living_room", 0.0, 0.0, 8.0, 8.0),
    zone("kitchen", 10.0, 0.0, 14.0, 4.0),
    zone("bedroom_1", 10.0, 6.0, 14.0, 10.0),
    zone("bathroom", 0.0, 10.0, 3.0, 13.0),
]
ROBOT = {"start": [4.0, 16.0, 0.0], "goal": [24.0, 16.0, 0.0]}
FAR = {"start": [30.0, 30.0, 0.0], "goal": [31.0, 30.0, 0.0]}

HUMANSIM_TYPES = Path(__file__).resolve().parents[4] / "humansim" / "arena_humansim" / "config" / "agent_types"


def _walkers(zones: list, profile: str = "normal", seed: int = 0, robot: dict = ROBOT) -> list:
    doc = build(profile, zones=zones, robot=robot, seed=seed, world="w")
    return [a for a in doc["dynamic"] if not a["agent"]["agent_type"].startswith("./")]


def test_the_mirrored_kinematics_match_humansims_own_files() -> None:
    if not HUMANSIM_TYPES.is_dir():
        pytest.skip("humansim not beside this package")
    for name, (mean, std, lo, hi) in KINEMATICS.items():
        raw = yaml.safe_load((HUMANSIM_TYPES / f"{name}.yaml").read_text())
        d = raw["desired_velocity"]
        assert (d["mean"], d.get("std", 0.0), d["clip_low"], d["clip_high"]) == (mean, std, lo, hi), name


def test_every_mixed_type_exists_and_every_kinematic_type_is_mixed_somewhere() -> None:
    mixed = {t for w in MIX.values() for t in w}
    assert mixed <= set(KINEMATICS)
    assert mixed == set(KINEMATICS)


def test_world_kind_reads_the_vocabulary() -> None:
    assert world_kind(HOSPITAL) == "hospital"
    assert world_kind(FLAT) == "residential"
    assert world_kind(OFFICE) == "office"


def test_specialised_types_stay_in_their_buildings() -> None:
    assert "child" in mix_for("break", "residential") and "child" not in mix_for("break", "office")
    for t, kinds in ONLY_IN.items():
        for profile in MIX:
            for kind in ("hospital", "residential", "office"):
                if kind not in kinds:
                    assert t not in mix_for(profile, kind), (t, profile, kind)


def test_a_hospital_population_has_no_children_and_no_phantom_types() -> None:
    # Wheelchairs and carts were dropped 2026-08-31: Arena cannot embody them.
    kinds = {a["agent"]["agent_type"] for s in range(30) for p in ("normal", "night") for a in _walkers(HOSPITAL, p, seed=s)}
    assert "child" not in kinds
    assert not kinds & {"cart", "wheelchair"}
    assert kinds <= set(KINEMATICS)


def test_a_flat_has_children_and_no_gurneys() -> None:
    kinds = {a["agent"]["agent_type"] for s in range(40) for p in ("break", "normal") for a in _walkers(FLAT, p, seed=s, robot=FAR)}
    assert "child" in kinds
    assert not kinds & {"cart", "wheelchair"}


def test_an_office_has_neither() -> None:
    kinds = {a["agent"]["agent_type"] for s in range(8) for a in _walkers(OFFICE, "normal", seed=s)}
    assert not kinds & {"cart", "wheelchair", "child"}
    assert len(kinds) >= 3, "an office crowd should still be mixed"


def test_speeds_come_from_the_types_own_range() -> None:
    for profile in ("normal", "night"):
        for a in _walkers(HOSPITAL, profile, seed=2):
            _, _, lo, hi = KINEMATICS[a["agent"]["agent_type"]]
            assert lo <= a["agent"]["desired_velocity"] <= hi, a


def test_night_is_slower_than_normal_for_the_same_type() -> None:
    # Same draw, different profile scale: the only thing that may differ is the scale.
    for kind in KINEMATICS:
        assert speed_for(kind, random.Random(7), 0.8) <= speed_for(kind, random.Random(7), 1.0)
    assert speed_for("adult", random.Random(7), 0.8) < speed_for("adult", random.Random(7), 1.0)


def test_no_radius_is_written_the_type_decides() -> None:
    for a in _walkers(HOSPITAL, "normal", seed=1):
        assert "radius" not in a["agent"] and "agent_radius" not in a["agent"]


def test_talkers_are_untouched_by_the_mix() -> None:
    from arena_evaluation.generation.scenario.populate import cast_split

    doc = build("normal", zones=HOSPITAL, robot=ROBOT, seed=1, world="w")
    talkers = [a for a in doc["dynamic"] if a["agent"]["agent_type"].startswith("./")]
    assert len(talkers) == cast_split(len(doc["dynamic"]))[0] and len(talkers) % TALKERS == 0 and talkers


def test_speed_for_clips_to_the_type() -> None:
    rng = random.Random(0)
    for kind, (_, _, lo, hi) in KINEMATICS.items():
        for scale in (0.5, 1.0, 2.0):
            for _ in range(50):
                assert lo <= speed_for(kind, rng, scale) <= hi


def test_generation_is_still_byte_identical() -> None:
    assert build("normal", zones=HOSPITAL, robot=ROBOT, seed=4, world="w") == build("normal", zones=HOSPITAL, robot=ROBOT, seed=4, world="w")


def test_every_base_has_an_elder_to_select() -> None:
    for s in range(10):
        for p in ("normal", "break", "night"):
            kinds = {a["agent"]["agent_type"] for a in _walkers(HOSPITAL, p, seed=s)}
            assert "elder" in kinds, (p, s)
