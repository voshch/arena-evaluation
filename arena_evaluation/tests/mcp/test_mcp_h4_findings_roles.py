"""Contract tests: verified findings, derived tables, roles, audit, and write guard."""

from __future__ import annotations

import json
import multiprocessing
import os
import pathlib

import polars as pl
import pytest
import yaml


from arena_mcp.core import ToolError
from mcp_fixtures import run_session, server_env
from synthetic_benchmark import write_synthetic_benchmark

from arena_evaluation.mcp import analysis, base, findings, query, reporting, roles


def test_h4_structured_number_rejected_when_not_in_evidence(
    tmp_path: pathlib.Path,
) -> None:
    """Creating a finding whose structured number is absent from its evidence raises ToolError."""
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "h4_run"
    write_synthetic_benchmark(run_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    res = analysis.evaluation_analysis(
        procedure="compare",
        run_id="h4_run",
        params={
            "planner_a": "planner_a",
            "planner_b": "planner_b",
            "metric": "time_to_goal",
        },
    )
    ev_id = res["evidence_id"]

    with pytest.raises(ToolError, match="not found in evidence"):
        findings.evaluation_finding_create(
            run_id="h4_run",
            claim="Fake claim",
            evidence_ids=[ev_id],
            structured_numbers=[{"evidence_id": ev_id, "column": "time_to_goal", "value": 99999.0}],
            author="analyst",
        )


def test_h4_verify_gate_author_separation_and_attach_gate(
    tmp_path: pathlib.Path,
) -> None:
    """Author cannot verify own finding; unverified finding cannot attach; verifier succeeds after verify."""
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "h4_gate"
    write_synthetic_benchmark(run_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    q_res = query.evaluation_query(
        run_id="h4_gate",
        query="SELECT planner, AVG(time_to_goal) AS mean_ttg FROM v_episode_summary GROUP BY planner ORDER BY planner",
        pin_evidence=True,
    )
    ev_id = q_res["evidence_id"]
    pa_val = next(r["mean_ttg"] for r in q_res["data"] if r["planner"] == "planner_a")

    created = findings.evaluation_finding_create(
        run_id="h4_gate",
        claim="planner_a achieves lower mean time_to_goal",
        evidence_ids=[ev_id],
        structured_numbers=[{"evidence_id": ev_id, "column": "mean_ttg", "value": float(pa_val)}],
        author="analyst",
    )
    fid = created.finding_id
    assert created.status == findings.FindingStatus.PROPOSED

    # Proposed finding cannot be attached
    with pytest.raises(ToolError):
        findings.evaluation_finding_attach(run_id="h4_gate", finding_id=fid, report="standard")

    # Mechanical verification passes
    ver = findings.evaluation_finding_verify(run_id="h4_gate", finding_id=fid)
    assert ver["mechanical_passed"] is True

    # Author cannot verify own finding
    with pytest.raises(ToolError, match="author"):
        findings.evaluation_finding_status(run_id="h4_gate", finding_id=fid, status="verified", by="analyst")

    # Different role (verifier) can set verified
    updated = findings.evaluation_finding_status(run_id="h4_gate", finding_id=fid, status="verified", by="verifier")
    assert updated.status == findings.FindingStatus.VERIFIED

    # Now attach succeeds
    attached = findings.evaluation_finding_attach(run_id="h4_gate", finding_id=fid, report="standard")
    assert attached.finding.finding_id == fid


def test_h4_derived_table_lifecycle_stale_and_refresh_and_version_backup(
    tmp_path: pathlib.Path,
) -> None:
    """Derived tables save under analysis/derived, query as derived_<name>, go stale on reprocess, and refresh."""
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "h4_derived"
    write_synthetic_benchmark(run_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    saved = query.evaluation_query(
        run_id="h4_derived",
        query="SELECT planner, COUNT(*) AS n FROM v_episode_summary GROUP BY planner",
        save_as="planner_counts",
    )
    assert saved["derived_table"] == "derived_planner_counts"

    # Queryable as derived_planner_counts
    q2 = query.evaluation_query(
        run_id="h4_derived",
        query="SELECT planner, n FROM derived_planner_counts ORDER BY planner",
    )
    assert len(q2["data"]) == 4

    # Replace keeps v1 backup
    query.evaluation_query(
        run_id="h4_derived",
        query="SELECT planner, COUNT(*) AS n FROM v_episode_summary GROUP BY planner",
        save_as="planner_counts",
    )
    assert (run_dir / "analysis" / "derived" / "planner_counts.v1.parquet").is_file()

    # Fresh before reprocess
    listed = analysis.evaluation_tables(run_id="h4_derived")
    assert listed["tables"][0]["status"] == "fresh"

    # Modify an episode parquet -> snapshot changes -> derived table turns stale
    ep0 = run_dir / "episodes" / "episode_000" / "metrics" / "fact_robots.parquet"
    pl.read_parquet(ep0).with_columns(pl.lit(42.0).alias("energy_total_wh")).write_parquet(ep0)

    listed_stale = analysis.evaluation_tables(run_id="h4_derived")
    assert listed_stale["tables"][0]["status"] == "stale"

    refreshed = analysis.evaluation_refresh_table(run_id="h4_derived", name="planner_counts")
    assert refreshed["status"] == "fresh"


def test_h4_write_guard_refuses_episodes_and_audit_logs_mutations(
    tmp_path: pathlib.Path,
) -> None:
    """Write guard refuses paths under episodes/ and audit.jsonl records mutating calls."""
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "h4_guard"
    write_synthetic_benchmark(run_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    with pytest.raises(ToolError, match="protected"):
        base.assert_safe_analysis_write(
            run_dir,
            run_dir / "episodes" / "episode_000" / "metrics" / "fact_robots.parquet",
        )

    query.evaluation_query(
        run_id="h4_guard",
        query="SELECT planner FROM v_episode_summary LIMIT 1",
        save_as="audit_check",
    )
    audit_path = run_dir / "analysis" / "audit.jsonl"
    assert audit_path.is_file()
    entries = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(e["tool"] == "evaluation_query" for e in entries)


def _shift_time_to_goal(run_dir: pathlib.Path, delta_s: float) -> None:
    for path in (run_dir / "episodes").glob("*/metrics/fact_robots.parquet"):
        frame = pl.read_parquet(path)
        frame.with_columns((pl.col("time_to_goal") + delta_s).alias("time_to_goal")).write_parquet(path)


def test_h4_failed_verification_moves_verified_finding_to_needs_work(
    tmp_path: pathlib.Path,
) -> None:
    """A stale or failed verification is not attachable, and a failed verify moves the finding."""
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "h4_stale"
    write_synthetic_benchmark(run_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)
    pinned = query.evaluation_query(
        run_id="h4_stale",
        query=("SELECT planner, AVG(time_to_goal) AS val FROM v_episode_summary GROUP BY planner ORDER BY planner"),
        pin_evidence=True,
    )
    value = float(pinned["data"][0]["val"])
    created = findings.evaluation_finding_create(
        run_id="h4_stale",
        claim="claim tied to the recorded average",
        evidence_ids=[pinned["evidence_id"]],
        structured_numbers=[
            {
                "evidence_id": pinned["evidence_id"],
                "column": "val",
                "value": value,
            }
        ],
        author="analyst",
    )
    finding_id = created.finding_id
    first = findings.evaluation_finding_verify(run_id="h4_stale", finding_id=finding_id)
    assert first["mechanical_passed"] is True
    findings.evaluation_finding_status(
        run_id="h4_stale",
        finding_id=finding_id,
        status="verified",
        by="verifier",
    )
    draft = reporting.evaluation_report_draft(run_id="h4_stale", title="stale finding")
    verified_facts = {path: path.read_bytes() for path in (run_dir / "episodes").glob("*/metrics/fact_robots.parquet")}
    _shift_time_to_goal(run_dir, 500.0)

    with pytest.raises(ToolError, match="snapshot"):
        findings.evaluation_finding_attach(run_id="h4_stale", finding_id=finding_id)
    left_out = reporting.evaluation_report_assemble(
        run_id="h4_stale",
        draft_id=draft["draft_id"],
        name="stale_report",
    )
    assert left_out["findings_used"] == []
    assert [entry["finding_id"] for entry in left_out["findings_left_out"]] == [finding_id]

    second = findings.evaluation_finding_verify(run_id="h4_stale", finding_id=finding_id)
    assert second["mechanical_passed"] is False
    stored = next(finding for finding in findings.evaluation_findings(run_id="h4_stale").findings if finding.finding_id == finding_id)
    assert stored.status is findings.FindingStatus.NEEDS_WORK
    with pytest.raises(ToolError, match="needs_work"):
        findings.evaluation_finding_attach(run_id="h4_stale", finding_id=finding_id)

    for path, content in verified_facts.items():
        path.write_bytes(content)
    restored = findings.evaluation_finding_verify(run_id="h4_stale", finding_id=finding_id)
    assert restored["mechanical_passed"] is True
    findings.evaluation_finding_status(
        run_id="h4_stale",
        finding_id=finding_id,
        status="verified",
        by="verifier",
    )
    assembled = reporting.evaluation_report_assemble(
        run_id="h4_stale",
        draft_id=draft["draft_id"],
        name="fresh_report",
    )
    assert assembled["findings_used"] == [finding_id]
    attached = findings.evaluation_finding_attach(run_id="h4_stale", finding_id=finding_id, report="fresh_report")
    assert attached.finding.status is findings.FindingStatus.VERIFIED

    audit_path = run_dir / "analysis" / "audit.jsonl"
    audit_entries = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(entry["tool"] == "evaluation_finding_status" and entry["produced_ids"] == [finding_id] for entry in audit_entries)


def test_h4_roles_resource_and_allowlists_valid(tmp_path: pathlib.Path) -> None:
    """Every tool named by a role exists in the domain, destructive tools are operator-only, and resource serves roles."""
    from arena_evaluation.mcp import domain

    by_name = {t.name: t for t in domain.tools}
    for role_name, spec in roles.ROLES.items():
        for tool_name in spec["tools"]:
            assert tool_name in by_name, f"role {role_name} names unknown tool {tool_name}"
            if by_name[tool_name].annotations.destructive_hint:
                assert role_name == "operator" or (tool_name in roles.AGENT_AREA_TOOLS), f"destructive tool {tool_name} in non-operator role {role_name}"
    for tool_name, area in roles.AGENT_AREA_TOOLS.items():
        assert tool_name in by_name, f"AGENT_AREA_TOOLS names unregistered tool {tool_name}"
        assert by_name[tool_name].annotations.destructive_hint, f"AGENT_AREA_TOOLS tool {tool_name} is not destructive"
        assert area in {"analysis", "reports"}, f"AGENT_AREA_TOOLS tool {tool_name} has unexpected area {area!r}"

    async def body(client):
        res = await client.read_resource("arena://evaluation/roles")
        return json.loads(res.contents[0].text)

    out = run_session(body, server_env(tmp_path), ("evaluation",))
    assert "verifier" in out["roles"]
    assert "operator" in out["roles"]


def test_h4_derived_table_replace_and_describe_metric_null_fields(
    tmp_path: pathlib.Path,
) -> None:
    """Saving a derived table twice leaves only active/v1/yaml files; NULL dim_metrics fields return None."""
    from arena_evaluation.analytics.derived import derived_dir, save_derived_table

    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "h4_r9"
    write_synthetic_benchmark(run_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    df = pl.DataFrame({"planner": ["planner_a"], "val": [1.0]})
    save_derived_table(
        run_dir,
        name="r9_tbl",
        frame=df,
        definition={"kind": "sql", "sql": "SELECT 1"},
        snapshot_id="snap-1",
        author="analyst",
    )
    save_derived_table(
        run_dir,
        name="r9_tbl",
        frame=df.with_columns(pl.lit(2.0).alias("val")),
        definition={"kind": "sql", "sql": "SELECT 2"},
        snapshot_id="snap-2",
        author="analyst",
    )
    names = sorted(p.name for p in derived_dir(run_dir).iterdir() if not p.name.endswith(".lock"))
    assert names == ["r9_tbl.parquet", "r9_tbl.v1.parquet", "r9_tbl.yaml"]

    # Non-mapping derived sidecar raises ValueError on replace
    (derived_dir(run_dir) / "r9_tbl.yaml").write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a YAML mapping"):
        save_derived_table(
            run_dir,
            name="r9_tbl",
            frame=df,
            definition={"kind": "sql", "sql": "SELECT 3"},
            snapshot_id="snap-3",
            author="analyst",
        )

    # NULL producer / fact_table in dim_metrics returns None (not "")
    dm_path = run_dir / "dim_metrics.parquet"
    dm = pl.read_parquet(dm_path).with_columns(
        pl.when(pl.col("metric") == "time_to_goal").then(pl.lit(None, dtype=pl.String)).otherwise(pl.col("producer")).alias("producer"),
        pl.when(pl.col("metric") == "time_to_goal").then(pl.lit(None, dtype=pl.String)).otherwise(pl.col("fact_table")).alias("fact_table"),
    )
    dm.write_parquet(dm_path)
    desc = analysis.evaluation_describe_metric("time_to_goal", run_id="h4_r9")
    assert desc["producer"] is None
    assert desc["fact_table"] is None


def test_h4_verify_finding_checks_integrity_rerun_reprocess_and_same_seed_flag(
    tmp_path: pathlib.Path,
) -> None:
    """verify_finding checks content integrity, reruns on unchanged/reprocessed data, and flags same-seed overlap."""
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "h4_r12"
    write_synthetic_benchmark(run_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    q_res = query.evaluation_query(
        run_id="h4_r12",
        query="SELECT planner, AVG(time_to_goal) AS mean_ttg FROM v_episode_summary GROUP BY planner ORDER BY planner",
        pin_evidence=True,
    )
    ev_id = q_res["evidence_id"]
    pa_val = next(float(r["mean_ttg"]) for r in q_res["data"] if r["planner"] == "planner_a")
    created = findings.evaluation_finding_create(
        run_id="h4_r12",
        claim="planner_a mean_ttg matches pinned query",
        evidence_ids=[ev_id],
        structured_numbers=[
            {
                "evidence_id": ev_id,
                "column": "mean_ttg",
                "value": pa_val,
                "tolerance": 1e-4,
            }
        ],
        author="analyst",
    )

    # (c) Untouched run passes with equal hashes
    v_ok = findings.evaluation_finding_verify(run_id="h4_r12", finding_id=created.finding_id)
    assert v_ok["mechanical_passed"] is True
    rerun_chk = next(c for c in v_ok["checks"] if c["check"] == "evidence_rerun")
    assert rerun_chk["mode"] == "hash"
    assert rerun_chk["stored_hash"] == rerun_chk["rerun_hash"]

    # (a) Edited evidence file fails evidence_integrity
    ev_parquet = run_dir / "analysis" / "evidence" / f"{ev_id}.parquet"
    ev_orig = pl.read_parquet(ev_parquet)
    ev_edited = ev_orig.with_columns(pl.when(pl.col("planner") == "planner_b").then(999.0).otherwise(pl.col("mean_ttg")).alias("mean_ttg"))
    ev_edited.write_parquet(ev_parquet)
    v_tamper = findings.evaluation_finding_verify(run_id="h4_r12", finding_id=created.finding_id)
    assert v_tamper["mechanical_passed"] is False
    assert any(c["check"] == "evidence_integrity" and not c["passed"] for c in v_tamper["checks"])
    ev_orig.write_parquet(ev_parquet)

    # (d) Reprocess leaving planner_a intact passes and records new snapshot_id
    planner_b_eps = pl.read_parquet(run_dir / "dim_episodes.parquet").filter(pl.col("planner") == "planner_b")["episode_id"].to_list()
    for rob_path in (run_dir / "episodes").glob("*/metrics/fact_robots.parquet"):
        df = pl.read_parquet(rob_path).with_columns(pl.when(pl.col("episode_id").is_in(planner_b_eps)).then(pl.col("time_to_goal") + 10.0).otherwise(pl.col("time_to_goal")).alias("time_to_goal"))
        df.write_parquet(rob_path)
    v_reproc_ok = findings.evaluation_finding_verify(run_id="h4_r12", finding_id=created.finding_id)
    assert v_reproc_ok["mechanical_passed"] is False
    snapshot_chk = next(c for c in v_reproc_ok["checks"] if c.get("mode") == "snapshot_changed")
    assert snapshot_chk["passed"] is False
    assert v_reproc_ok["snapshot_id"] != q_res["snapshot_id"]

    # (b) Changed data under claimed number fails and shows rerun_value
    for rob_path in (run_dir / "episodes").glob("*/metrics/fact_robots.parquet"):
        df = pl.read_parquet(rob_path).with_columns((pl.col("time_to_goal") + 500.0).alias("time_to_goal"))
        df.write_parquet(rob_path)
    v_changed = findings.evaluation_finding_verify(run_id="h4_r12", finding_id=created.finding_id)
    assert v_changed["mechanical_passed"] is False
    num_chk = next(c for c in v_changed["checks"] if c["check"] == "structured_number")
    assert num_chk["passed"] is False
    assert num_chk["rerun_value"] is not None and num_chk["rerun_value"] > 400.0

    # (e) same-seed overlap raises flag without failing
    ev1 = query.evaluation_query(
        run_id="h4_r12",
        query="SELECT seed, AVG(time_to_goal) AS val FROM v_episode_summary WHERE seed IN (1, 2) GROUP BY seed ORDER BY seed",
        pin_evidence=True,
    )
    ev2 = query.evaluation_query(
        run_id="h4_r12",
        query="SELECT seed, AVG(time_to_goal) AS val FROM v_episode_summary WHERE seed IN (2, 3) GROUP BY seed ORDER BY seed",
        pin_evidence=True,
    )
    f_seed = findings.evaluation_finding_create(
        run_id="h4_r12",
        claim="overlapping seed pair",
        evidence_ids=[ev1["evidence_id"], ev2["evidence_id"]],
        author="analyst",
    )
    v_seed = findings.evaluation_finding_verify(run_id="h4_r12", finding_id=f_seed.finding_id)
    assert v_seed["mechanical_passed"] is True
    seed_chk = next(c for c in v_seed["checks"] if c["check"] == "same_seed_overlap")
    assert seed_chk["flagged"] is True
    assert seed_chk["passed"] is True


_SAVES_PER_WORKER = 20


def _concurrent_process_worker(run_dir_str: str, worker_id: int) -> None:
    from arena_evaluation.analytics.derived import save_derived_table
    from arena_evaluation.analytics.findings import create_finding

    rdir = pathlib.Path(run_dir_str)
    for idx in range(_SAVES_PER_WORKER):
        create_finding(
            rdir,
            claim=f"worker {worker_id} finding {idx}",
            author="analyst",
        )
        save_derived_table(
            rdir,
            name="shared_tbl",
            frame=pl.DataFrame({"worker": [worker_id], "step": [idx]}),
            definition={"kind": "sql", "sql": "SELECT 1"},
            snapshot_id=f"snap-{worker_id}-{idx}",
            author="analyst",
        )


def test_h4_two_processes_concurrent_findings_and_derived_tables(
    tmp_path: pathlib.Path,
) -> None:
    """Two processes save one derived table concurrently and every save is counted."""
    from arena_evaluation.analytics.derived import derived_dir
    from arena_evaluation.analytics.findings import load_findings

    run_dir = tmp_path / "data" / "benchmarks" / "h4_mp"
    write_synthetic_benchmark(run_dir, include_swap=False)

    ctx = multiprocessing.get_context("spawn")
    p1 = ctx.Process(target=_concurrent_process_worker, args=(str(run_dir), 1))
    p2 = ctx.Process(target=_concurrent_process_worker, args=(str(run_dir), 2))
    p1.start()
    p2.start()
    p1.join(timeout=30)
    p2.join(timeout=30)
    assert p1.exitcode == 0
    assert p2.exitcode == 0

    stored = load_findings(run_dir)
    assert len(stored) == 2 * _SAVES_PER_WORKER
    derived = derived_dir(run_dir)
    assert (derived / "shared_tbl.parquet").is_file()
    assert (derived / "shared_tbl.v1.parquet").is_file()
    sidecar_path = derived / "shared_tbl.yaml"
    assert sidecar_path.is_file()
    sidecar = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["version"] == 2 * _SAVES_PER_WORKER
