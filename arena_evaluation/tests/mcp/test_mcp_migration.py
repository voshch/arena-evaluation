"""Round-2 migration tools: the MCP adapter over the L4 legacy converter."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys
import types

import pytest

from arena_mcp.core import ToolError

from mcp_fixtures import build_runs

from arena_evaluation.mcp import migration

_CONVERTER = "arena_evaluation.presentation.legacy_manifest"


@dataclasses.dataclass(frozen=True, slots=True)
class _Manifest:
    name: str

    def model_copy(self, update: dict[str, object]) -> _Manifest:
        return dataclasses.replace(self, **update)


@dataclasses.dataclass(frozen=True, slots=True)
class _Migration:
    source: pathlib.Path
    manifest: _Manifest
    entries: tuple[dict[str, object], ...]
    metrics: tuple[dict[str, object], ...]


def _stub_module() -> types.ModuleType:
    module = types.ModuleType(_CONVERTER)

    def convert_legacy_file(
        path: pathlib.Path,
        *,
        metric_map: object,
        known_metrics: object,
        metric_tables: object,
    ) -> _Migration:
        return _Migration(
            source=pathlib.Path(path),
            manifest=_Manifest(name="social"),
            entries=(
                {
                    "id": "success_rate",
                    "legacy_type": "bar",
                    "status": "converted",
                    "new_id": "bar_success",
                    "reason": "",
                },
                {
                    "id": "invented_plot",
                    "legacy_type": "radar",
                    "status": "todo",
                    "new_id": None,
                    "reason": "metric 'invented_metric' is not in the registry",
                },
            ),
            metrics=(
                {
                    "legacy": "success",
                    "new": "success",
                    "status": "exact",
                    "note": "",
                },
                {
                    "legacy": "invented_metric",
                    "new": None,
                    "status": "missing",
                    "note": "not in the metric dictionary",
                },
            ),
        )

    def render_manifest_yaml(mig: _Migration) -> str:
        return f"name: {mig.manifest.name}\nplots: {len(mig.entries)}\n"

    def render_migration_report(mig: _Migration) -> dict[str, object]:
        return {
            "source": str(mig.source),
            "manifest": mig.manifest.name,
            "counts": {
                "total": len(mig.entries),
                "converted": 1,
                "dropped": 0,
                "todo": 1,
            },
            "plots": list(mig.entries),
            "metrics": list(mig.metrics),
            "carried_notes": ["legacy title kept"],
        }

    module.convert_legacy_file = convert_legacy_file
    module.load_metric_map = lambda path=None: object()
    module.render_manifest_yaml = render_manifest_yaml
    module.render_migration_report = render_migration_report
    return module


@pytest.fixture
def run_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    data_dir = tmp_path / "data"
    run = build_runs(data_dir / "benchmarks")
    os.environ["ARENA_DATA_DIR"] = str(data_dir)
    return run


def _legacy_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "legacy_social.yaml"
    path.write_text("plots: []\n", encoding="utf-8")
    return path


def test_migrate_manifest_refuses_with_a_hint_when_the_converter_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Until the L4 converter ships, the tool must refuse loudly instead of inventing a migration."""
    monkeypatch.delitem(sys.modules, _CONVERTER, raising=False)
    with pytest.raises(ToolError, match="legacy manifest converter"):
        migration.evaluation_migrate_manifest(str(_legacy_file(tmp_path)))


def test_migrate_manifest_dry_run_returns_skeleton_and_report_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, run_dir: pathlib.Path
) -> None:
    monkeypatch.setitem(sys.modules, _CONVERTER, _stub_module())
    source = _legacy_file(tmp_path)

    result = migration.evaluation_migrate_manifest(
        str(source), run_id=run_dir.name, dry_run=True
    )

    assert result["summary"] == {
        "total": 2,
        "converted": 1,
        "dropped": 0,
        "todo": 1,
    }
    assert "name: social" in result["skeleton_yaml"]
    assert result["written"] is None
    assert result["migration_report"]["plots"][1]["status"] == "todo"
    assert "evaluation_lint_declarations(deep=True)" in result["hint"]
    assert not (run_dir / "analysis" / "reports").exists()


def test_migrate_manifest_writes_into_the_run_analysis_reports_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, run_dir: pathlib.Path
) -> None:
    monkeypatch.setitem(sys.modules, _CONVERTER, _stub_module())
    source = _legacy_file(tmp_path)

    result = migration.evaluation_migrate_manifest(
        str(source),
        target="social_migrated",
        run_id=run_dir.name,
        dry_run=False,
    )

    report_path = pathlib.Path(result["written"]["report"])
    migration_path = pathlib.Path(result["written"]["migration_report"])
    assert (
        report_path
        == run_dir / "analysis" / "reports" / "social_migrated" / "report.yaml"
    )
    assert report_path.is_file()
    assert migration_path.is_file()
    assert "name: social_migrated" in report_path.read_text(encoding="utf-8")
    assert "todo: 1" in migration_path.read_text(encoding="utf-8")

    with pytest.raises(ToolError, match="already exists"):
        migration.evaluation_migrate_manifest(
            str(source),
            target="social_migrated",
            run_id=run_dir.name,
            dry_run=False,
        )
    replaced = migration.evaluation_migrate_manifest(
        str(source),
        target="social_migrated",
        run_id=run_dir.name,
        dry_run=False,
        overwrite=True,
    )
    assert replaced["written"]["report"] == str(report_path)


def test_migrate_manifest_requires_a_run_id_to_write_and_refuses_escaping_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, run_dir: pathlib.Path
) -> None:
    monkeypatch.setitem(sys.modules, _CONVERTER, _stub_module())
    source = _legacy_file(tmp_path)

    with pytest.raises(ToolError, match="needs a run id"):
        migration.evaluation_migrate_manifest(str(source), dry_run=False)
    with pytest.raises(ToolError, match="one path component"):
        migration.evaluation_migrate_manifest(
            str(source), target="../escape", run_id=run_dir.name, dry_run=False
        )
    with pytest.raises(ToolError, match="no legacy manifest file"):
        migration.evaluation_migrate_manifest(str(tmp_path / "missing.yaml"))


def test_manifest_mappings_returns_metric_and_plot_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setitem(sys.modules, _CONVERTER, _stub_module())

    mappings = migration.evaluation_manifest_mappings(str(_legacy_file(tmp_path)))

    assert mappings["manifest"] == "social"
    assert mappings["metrics"][0] == {
        "legacy_metric": "success",
        "new_metric": "success",
        "status": "exact",
        "note": "",
    }
    assert mappings["plots"][1]["legacy_plot"] == "invented_plot"
    assert mappings["plots"][1]["new_plot"] is None
    assert mappings["summary"]["todo"] == 1
    assert mappings["carried_notes"] == ["legacy title kept"]
