"""Unit tests for arena_evaluation.storage.planner_names.split_planner_name.

Splits "local/inter" planner strings:

  "teb"                  -> ("teb", "none")
  "teb-dwa"              -> ("teb", "dwa")
  "a-local-inter"        -> ("local", "inter")     # first segment dropped
  "a-local-inter-extra"  -> ("local", "inter-extra")
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from arena_evaluation.storage.planner_names import split_planner_name


# ---------------------------------------------------------------------------
# falsy inputs
# ---------------------------------------------------------------------------


def test_none_returns_unknown_pair():
    assert split_planner_name(None) == ("unknown", "unknown")


def test_empty_string_returns_unknown_pair():
    assert split_planner_name("") == ("unknown", "unknown")


# ---------------------------------------------------------------------------
# one part
# ---------------------------------------------------------------------------


def test_single_part_returns_none_inter():
    assert split_planner_name("teb") == ("teb", "none")


def test_non_string_input_coerced_to_str():
    assert split_planner_name(123) == ("123", "none")


def test_whitespace_only_name_is_truthy_single_part():
    assert split_planner_name("   ") == ("   ", "none")


# ---------------------------------------------------------------------------
# two parts
# ---------------------------------------------------------------------------


def test_two_parts_split():
    assert split_planner_name("teb-dwa") == ("teb", "dwa")


def test_two_parts_second_empty():
    assert split_planner_name("teb-") == ("teb", "")


def test_two_parts_first_empty():
    assert split_planner_name("-dwa") == ("", "dwa")


# ---------------------------------------------------------------------------
# three or more parts: first segment dropped, remainder joined
# ---------------------------------------------------------------------------


def test_three_parts_drop_first():
    assert split_planner_name("contestant-local-inter") == ("local", "inter")


def test_three_parts_join_remainder():
    assert split_planner_name("team-teb-dwb-extra") == ("teb", "dwb-extra")


def test_five_parts_join_remainder():
    assert split_planner_name("a-b-c-d-e") == ("b", "c-d-e")


# ---------------------------------------------------------------------------
# type contract
# ---------------------------------------------------------------------------


def test_returns_tuple_of_two_strings():
    result = split_planner_name("nav2-teb-smac")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(part, str) for part in result)


# ---------------------------------------------------------------------------
# hypothesis properties
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, derandomize=True)
@given(st.text(max_size=20))
def test_split_planner_name_always_returns_string_pair(name):
    local, inter = split_planner_name(name)
    assert isinstance(local, str)
    assert isinstance(inter, str)
    if not name:
        assert (local, inter) == ("unknown", "unknown")


@settings(max_examples=100, deadline=None, derandomize=True)
@given(st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=6))
def test_split_planner_name_matches_join_spec(parts):
    name = "-".join(parts)
    local, inter = split_planner_name(name)
    if len(parts) == 1:
        assert (local, inter) == (parts[0], "none")
    elif len(parts) == 2:
        assert (local, inter) == (parts[0], parts[1])
    else:
        assert (local, inter) == (parts[1], "-".join(parts[2:]))
