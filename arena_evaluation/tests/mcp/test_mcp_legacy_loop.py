"""End-to-end declaration migration on the merged tree (FIX11, item 7)."""

from __future__ import annotations

import pathlib

import yaml

from star_benchmark import write_benchmark

from arena_evaluation.mcp import migration, reporting, schema

LEGACY_SOCIAL = pathlib.Path(__file__).parents[1] / "unit" / "fixtures" / "legacy" / "social.yaml"
SOCIAL_COLUMNS = {
    "personal_space_intrusion_integral": 1.5,
    "total_time_looking_at_pedestrians": 0.4,
    "time_in_intimate_zone": 0.2,
    "time_in_personal_zone": 0.3,
    "time_in_social_zone": 0.4,
    "time_in_public_zone": 0.5,
    "psi_intimate_events": 1,
    "psi_personal_events": 2,
    "psi_social_events": 3,
    "max_speed_intimate_zone": 0.1,
    "max_speed_personal_zone": 0.2,
    "max_speed_social_zone": 0.3,
    "max_speed_public_zone": 0.4,
    "social_force_ped_max": 1.0,
    "social_force_ped_mean": 0.5,
    "ci_max": 0.3,
    "ci_mean": 0.2,
    "ped_path_deflection_m": 0.1,
    "aeps_linear_s": 0.2,
    "ped_max_exposure_dba": 55.0,
    "ped_leq_exposure_dba": 50.0,
    "ped_max_startle_factor": 1.0,
    "passing_rule_compliance": 1.0,
    "path_length_m": 9.0,
}


def _social_run(tmp_path: pathlib.Path) -> pathlib.Path:
    from arena_evaluation.processing.metrics import MetricRegistry

    # The fact schema grows from the shared metric store; load it before writing the rows.
    MetricRegistry.discover_calculators_cls()
    run_dir = tmp_path / "data" / "benchmarks" / "legacy_loop"
    write_benchmark(
        run_dir,
        [
            {
                "episode_id": 0,
                "planner": "dwb",
                "success": True,
                "time_to_goal": 10.0,
                **SOCIAL_COLUMNS,
            },
            {
                "episode_id": 1,
                "planner": "teb",
                "success": False,
                "time_to_goal": 12.0,
                **SOCIAL_COLUMNS,
            },
        ],
    )
    return run_dir


def test_legacy_manifest_migrates_lints_renders_and_assembles(tmp_path: pathlib.Path, monkeypatch) -> None:
    run_dir = _social_run(tmp_path)
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path / "data"))

    dry = migration.evaluation_migrate_manifest(str(LEGACY_SOCIAL), dry_run=True)
    assert dry["summary"]["total"] == 31
    assert dry["summary"]["converted"] > 0
    assert dry["written"] is None

    written = migration.evaluation_migrate_manifest(str(LEGACY_SOCIAL), target="social", run_id=run_dir.name, dry_run=False)
    report_path = pathlib.Path(written["written"]["report"])
    assert report_path.is_file()

    lint = schema.evaluation_lint_declarations(run_id=run_dir.name, deep=True)
    social = [row for row in lint["plots"] if row["name"].startswith("report 'social' plot")]
    assert social
    passed = [row for row in social if row["passed"]]
    assert passed, [row["reason"] for row in social if not row["passed"]][:5]

    draft = reporting.evaluation_report_draft(run_id=run_dir.name, from_manifest="social")
    manifest = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    approved: list[str] = []
    for plot in manifest["plots"]:
        result = reporting.evaluation_report_plot(run_id=run_dir.name, draft_id=draft["draft_id"], plot=plot)
        if result["status"] != "rendered":
            continue
        reporting.evaluation_report_review(
            run_id=run_dir.name,
            draft_id=draft["draft_id"],
            plot_id=plot["id"],
            verdict="approved",
            reason="binds the fixture's social columns",
            by="reporter",
        )
        approved.append(plot["id"])
    assert len(approved) >= 2, approved

    assembled = reporting.evaluation_report_assemble(run_id=run_dir.name, draft_id=draft["draft_id"], name="social_legacy")
    assert assembled["complete"] is True
    assert pathlib.Path(assembled["report_path"]).is_file()
    assert set(approved) <= set(assembled["plots_used"])
