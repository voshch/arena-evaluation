"""Regression tests for the MCP argv builders and the removed runner options (B-002, R-028)."""

from __future__ import annotations

import argparse
import inspect
import sys

import pytest

pytest.importorskip("arena_mcp")

from mcp_fixtures import arena_data


def _mcp_tools_taking(option: str) -> list[str]:
    """Names of the evaluation tools in the runs module whose signature has the option."""
    from arena_evaluation.mcp import runs

    return [
        name
        for name, fn in vars(runs).items()
        if name.startswith("evaluation_")
        and callable(fn)
        and option in inspect.signature(fn).parameters
    ]


def test_the_helper_sees_the_pipeline_tools():
    """Sanity check: the signature scan finds the report tool, so an empty result means something."""
    assert "evaluation_report" in _mcp_tools_taking("profile")


@pytest.mark.parametrize("option", ["generate_gifs", "ros_domain_id"])
def test_no_mcp_tool_offers_a_removed_runner_option(option):
    """The runner no longer takes these options, so no MCP tool offers them."""
    assert _mcp_tools_taking(option) == []


def _arena_evaluation_parser(monkeypatch):
    """The real argparse parser of the arena evaluation CLI, captured from its own entry point."""
    import arena_evaluation.cli as cli

    original_parse_args = argparse.ArgumentParser.parse_args
    captured: dict[str, argparse.ArgumentParser] = {}

    class _Captured(Exception):
        pass

    def capture(self, args=None, namespace=None):
        captured["parser"] = self
        raise _Captured

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(sys, "argv", ["arena", "report"])
    with pytest.raises(_Captured):
        cli._main_impl()
    return captured["parser"], original_parse_args


@pytest.mark.parametrize("verb", ["extract", "process", "run", "report", "plot"])
def test_every_pipeline_argv_is_accepted_by_the_cli(tmp_path, monkeypatch, verb):
    """Each argv the pipeline tools build parses with the real arena evaluation CLI."""
    from arena_evaluation.mcp.runs import pipeline_argv

    from mcp_fixtures import build_run

    build_run(tmp_path / "data" / "benchmarks", "run-1")
    parser, original_parse_args = _arena_evaluation_parser(monkeypatch)
    with arena_data(tmp_path):
        argv = pipeline_argv(
            verb,
            ["run-1"],
            None,
            output_dir=str(tmp_path / "out"),
            profile=True,
            workers=2,
            report_manifest="reviewed",
            force_extract=verb == "process",
        )

    parsed = original_parse_args(parser, argv)

    assert parsed.command == verb
