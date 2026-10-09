from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_evaluation.generation.propose import client
from arena_evaluation.generation.propose.zone_roles import (
    Proposal,
    build_prompt,
    classify_tail,
    load_overrides,
    merge,
    parse,
    unknown_names,
)
from arena_evaluation.generation.scenario.zones import Role, zones_by_role


def zone(name, contents=(), corners=((0, 0), (4, 0), (4, 4), (0, 4))):
    return {
        "name": name,
        "corners": [{"x": x, "y": y} for x, y in corners],
        "entities": {"static": [{"name": f"e{i}", "model": m} for i, m in enumerate(contents)]},
    }


ZONES = [
    zone("main_hallway"),
    zone("kitchen"),
    zone("top_room_0", contents=["Common/Furniture/desk"]),
    zone("3_study"),
]


def test_only_the_tail_is_asked_about() -> None:
    # The substring rules cover 99 of the tree's 116 names for free; asking a model to confirm
    # that `kitchen` is a break zone is the expensive random number generator to avoid.
    assert unknown_names(ZONES) == ["3_study", "top_room_0"]


def test_a_reviewed_name_is_never_asked_again() -> None:
    assert unknown_names(ZONES, {"3_study": "work"}) == ["top_room_0"]


def test_the_prompt_carries_the_whole_vocabulary_but_asks_only_the_tail() -> None:
    prompt = build_prompt(ZONES, ["top_room_0"], world="w")
    # The rest of the world is the evidence: `top_room_0` beside `main_hallway` reads
    # differently from `top_room_0` beside `bedroom_1`.
    assert "main_hallway" in prompt
    assert "transit" in prompt
    assert "desk" in prompt
    assert prompt.count("3_study") == 1


def test_a_zones_size_and_contents_reach_the_prompt() -> None:
    prompt = build_prompt(ZONES, ["top_room_0"], world="w")
    assert "size_m" in prompt
    assert "contents" in prompt


GOOD = '{"top_room_0": {"role": "work", "why": "holds a desk"}}'


def test_a_well_formed_answer_parses() -> None:
    proposals, complaints = parse(GOOD, ["top_room_0"], model="m", sha="s")
    assert not complaints
    assert proposals == [Proposal(zone="top_room_0", role=Role.WORK, why="holds a desk", model="m", prompt_sha="s")]


def test_a_fenced_answer_parses() -> None:
    proposals, _ = parse(f"```json\n{GOOD}\n```", ["top_room_0"], model="m", sha="s")
    assert [p.zone for p in proposals] == ["top_room_0"]


def test_an_unasked_zone_is_dropped_not_accepted() -> None:
    proposals, complaints = parse(
        '{"kitchen": {"role": "work"}}',
        ["top_room_0"],
        model="m",
        sha="s",
    )
    assert proposals == []
    assert any("not one of the zones asked" in c for c in complaints)
    assert any("no answer returned" in c for c in complaints)


def test_an_invented_role_is_refused_not_coerced() -> None:
    proposals, complaints = parse(
        '{"top_room_0": {"role": "storage_cupboard"}}',
        ["top_room_0"],
        model="m",
        sha="s",
    )
    assert proposals == []
    assert any("is not a role" in c for c in complaints)


def test_prose_instead_of_json_is_a_complaint_not_a_crash() -> None:
    proposals, complaints = parse("I think it is a work zone!", ["top_room_0"], model="m", sha="s")
    assert proposals == []
    assert any("not JSON" in c for c in complaints)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "zone_roles.yaml"


def test_a_confident_proposal_is_written_with_its_provenance(store) -> None:
    written, _ = merge([Proposal(zone="a", role=Role.WORK, why="w", model="m", prompt_sha="s")], store=store)
    assert written == 1
    entry = yaml.safe_load(store.read_text())["zones"]["a"]
    assert entry["role"] == "work"
    assert entry["source"] == "llm"
    assert entry["model"] == "m"
    assert entry["prompt_sha"] == "s"


def test_an_unknown_answer_is_left_for_a_human(store) -> None:
    written, kept = merge([Proposal(zone="a", role=Role.UNKNOWN, model="m")], store=store)
    assert written == 0
    assert any("left for a human" in k for k in kept)
    assert not store.exists()


def test_a_human_entry_is_never_overwritten(store) -> None:
    store.write_text(yaml.safe_dump({"zones": {"a": {"role": "private", "source": "human"}}}))
    written, kept = merge([Proposal(zone="a", role=Role.WORK, model="m")], store=store)
    assert written == 0
    assert any("reviewed by a human" in k for k in kept)
    assert load_overrides(store) == {"a": "private"}


def test_rerunning_with_the_same_answer_writes_nothing(store) -> None:
    proposals = [Proposal(zone="a", role=Role.WORK, model="m", prompt_sha="s")]
    assert merge(proposals, store=store)[0] == 1
    assert merge(proposals, store=store)[0] == 0


def test_a_missing_store_is_an_empty_mapping_not_an_error(store) -> None:
    assert load_overrides(store) == {}


def test_the_reviewed_mapping_reaches_the_role_split(store) -> None:
    store.write_text(yaml.safe_dump({"zones": {"top_room_0": {"role": "work", "source": "llm"}}}))
    by_role = zones_by_role(ZONES, load_overrides(store))
    assert "top_room_0" in by_role[Role.WORK]
    assert "top_room_0" not in by_role[Role.UNKNOWN]


def test_nothing_to_ask_makes_no_call(monkeypatch) -> None:
    # `classify_tail` must not reach the network when the rules already cover everything.
    monkeypatch.setattr(client, "_client", lambda: pytest.fail("should not have called the model"))
    assert classify_tail([zone("kitchen")], world="w", names=[]) == ([], [])


def test_no_configured_model_is_a_clear_refusal_not_a_guess(monkeypatch) -> None:
    for var in ("LLM_API_ENDPOINT", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert not client.available()
    with pytest.raises(client.Unavailable, match="no LLM configured"):
        client.ask("s", "u")


def test_the_prompt_hash_changes_with_the_prompt() -> None:
    assert client.prompt_sha256("a", "b") == client.prompt_sha256("a", "b")
    assert client.prompt_sha256("a", "b") != client.prompt_sha256("ab", "")
