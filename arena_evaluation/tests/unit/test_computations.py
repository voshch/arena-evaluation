"""Golden, schema-contract, and EvidenceStore round-trip tests for analysis computations."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import BaseModel

from arena_evaluation.analytics import (
    COMPUTATIONS,
    BenchmarkCatalog,
    EvidenceStore,
    list_computations,
    run_computation,
)
from arena_evaluation.analytics.computations import (
    resolve_metric_directions,
    resolve_metric_units,
)
from arena_evaluation.storage.registry import dimensions
from arena_evaluation.storage.registry.analysis import (
    CompositeRankingAnalysis,
    CorrelationMatrixAnalysis,
    GroupSummaryAnalysis,
    PairedDiffAnalysis,
    RadarProfileAnalysis,
)
from arena_evaluation.storage.registry.table import Table
from star_benchmark import write_benchmark


def _write_dim_metrics(root: pathlib.Path, *, null_direction: str | None = None) -> None:
    dm_rows = [
        {
            "metric": "success",
            "scope": "robots",
            "fact_table": "fact_robots",
            "dtype": "Boolean",
            "unit": "",
            "description": "Episode succeeded.",
            "direction": None if null_direction == "success" else "higher",
            "producer": "outcome",
            "null_reason": None,
        },
        {
            "metric": "time_to_goal",
            "scope": "robots",
            "fact_table": "fact_robots",
            "dtype": "Float64",
            "unit": "s",
            "description": "Time to reach goal.",
            "direction": None if null_direction == "time_to_goal" else "lower",
            "producer": "time",
            "null_reason": None,
        },
        {
            "metric": "path_length_m",
            "scope": "robots",
            "fact_table": "fact_robots",
            "dtype": "Float64",
            "unit": "m",
            "description": "Path length.",
            "direction": None if null_direction == "path_length_m" else "lower",
            "producer": "path",
            "null_reason": None,
        },
        {
            "metric": "mean_velocity_mps",
            "scope": "robots",
            "fact_table": "fact_robots",
            "dtype": "Float64",
            "unit": "m/s",
            "description": "Mean velocity.",
            "direction": "higher",
            "producer": "kinematics",
            "null_reason": None,
        },
    ]
    dm_path = root / dimensions.DIMENSION_TABLES["dim_metrics"].relative_path()
    pl.DataFrame(dm_rows, schema=dimensions.DIMENSION_TABLES["dim_metrics"].polars_schema()).write_parquet(dm_path)


def _make_benchmark(
    root: pathlib.Path,
    rows: list[dict[str, object]],
    *,
    null_direction: str | None = None,
) -> pathlib.Path:
    write_benchmark(root, rows)
    _write_dim_metrics(root, null_direction=null_direction)
    return root


def _default_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ep_id = 0
    for planner, ttg_vals, path_vals, succ_vals in (
        (
            "dwb",
            [10.0, 12.0, 14.0, 16.0],
            [10.0, 10.0, 10.0, 10.0],
            [True, True, True, True],
        ),
        (
            "teb",
            [20.0, 22.0, 24.0, 26.0],
            [20.0, 22.0, 24.0, 26.0],
            [True, True, False, False],
        ),
    ):
        for idx, (ttg, pl_m, succ) in enumerate(zip(ttg_vals, path_vals, succ_vals, strict=True)):
            rows.append(
                {
                    "episode_id": ep_id,
                    "planner": planner,
                    "stage": "s0" if idx < 2 else "s1",
                    "seed": (idx % 2) + 1,
                    "success": succ,
                    "time_to_goal": ttg,
                    "path_length_m": pl_m,
                    "mean_velocity_mps": 0.0,
                    "is_reference": False,
                }
            )
            ep_id += 1
    rows.append(
        {
            "episode_id": ep_id,
            "planner": "ref",
            "stage": "s0",
            "seed": 1,
            "success": True,
            "time_to_goal": 999.0,
            "path_length_m": 999.0,
            "mean_velocity_mps": 99.0,
            "is_reference": True,
        }
    )
    return rows


def test_registry_metadata_and_evidence_defaults(tmp_path: pathlib.Path) -> None:
    expected = (
        "cluster_bootstrap_ci",
        "composite_ranking",
        "correlation_matrix",
        "distribution_bins",
        "group_summary",
        "paired_diff",
        "radar_profile",
        "series_profile",
        "top_n_ranking",
    )
    assert list_computations() == expected
    for name in expected:
        comp = COMPUTATIONS[name]
        assert issubclass(comp.params, BaseModel)
        assert issubclass(comp.output, Table)

    bench = _make_benchmark(tmp_path / "bench", _default_rows())
    catalog = BenchmarkCatalog.open(bench)
    store = EvidenceStore(benchmark_dir=bench)

    r_short = run_computation(
        "distribution_bins",
        catalog,
        {"metric": "time_to_goal"},
        evidence_store=store,
    )
    r_full = run_computation(
        "distribution_bins",
        catalog,
        {
            "metric": "time_to_goal",
            "group_cols": (),
            "nbins": 15,
            "density": False,
            "exclude_references": True,
        },
        evidence_store=store,
    )
    assert r_short.evidence_id == r_full.evidence_id
    rec = store.get(r_short.evidence_id)
    assert rec.params == {
        "source": "v_episode_summary",
        "filter": None,
        "exclude_references": True,
        "valid_only": True,
        "directions": None,
        "metric": "time_to_goal",
        "group_cols": [],
        "nbins": 15,
        "density": False,
    }
    assert_frame_equal(store.read_frame(r_short.evidence_id), r_short.summary)


def test_group_summary_wilson_numeric_trim_and_n_lt_2_nulls(
    tmp_path: pathlib.Path,
) -> None:
    rows = _default_rows() + [
        {
            "episode_id": 99,
            "planner": "solo",
            "stage": "s0",
            "seed": 1,
            "success": True,
            "time_to_goal": 9.0,
            "path_length_m": 8.0,
            "mean_velocity_mps": 0.0,
            "is_reference": False,
        },
        {
            "episode_id": 100,
            "planner": "ghost",
            "stage": "s0",
            "seed": 1,
            "success": False,
            "time_to_goal": None,
            "path_length_m": None,
            "mean_velocity_mps": None,
            "is_reference": False,
        },
    ]
    bench = _make_benchmark(tmp_path / "bench", rows)
    catalog = BenchmarkCatalog.open(bench)

    res_ci95 = run_computation(
        "group_summary",
        catalog,
        {
            "group_cols": ["planner"],
            "metrics": ["success", "time_to_goal"],
            "error_mode": "ci95",
        },
    )
    assert res_ci95.excluded_rows == {
        "references": 1,
        "time_to_goal:not_recorded": 1,
    }
    assert res_ci95.summary.columns == [
        "planner",
        *GroupSummaryAnalysis.polars_schema().names(),
    ]

    dwb_succ = res_ci95.summary.filter((pl.col("planner") == "dwb") & (pl.col("metric") == "success")).row(0, named=True)
    assert dwb_succ["mean"] == pytest.approx(1.0)
    assert dwb_succ["err_plus"] == pytest.approx(0.0)
    assert dwb_succ["err_minus"] == pytest.approx(0.48990002040399916)

    teb_succ = res_ci95.summary.filter((pl.col("planner") == "teb") & (pl.col("metric") == "success")).row(0, named=True)
    assert teb_succ["mean"] == pytest.approx(0.5)
    assert teb_succ["err_plus"] == pytest.approx(0.3499642911798285)
    assert teb_succ["err_minus"] == pytest.approx(0.3499642911798285)

    dwb_ttg_ci95 = res_ci95.summary.filter((pl.col("planner") == "dwb") & (pl.col("metric") == "time_to_goal")).row(0, named=True)
    assert dwb_ttg_ci95["mean"] == pytest.approx(13.0)
    assert dwb_ttg_ci95["std"] == pytest.approx(2.581988897471611)
    assert dwb_ttg_ci95["sem"] == pytest.approx(1.2909944487358056)
    assert dwb_ttg_ci95["err_plus"] == pytest.approx(2.530349119522179)

    res_sem = run_computation(
        "group_summary",
        catalog,
        {"group_cols": ["planner"], "metrics": ["time_to_goal"], "error_mode": "sem"},
    )
    dwb_sem = res_sem.summary.filter(pl.col("planner") == "dwb").row(0, named=True)
    assert dwb_sem["err_plus"] == pytest.approx(1.2909944487358056)

    solo_ttg = res_ci95.summary.filter((pl.col("planner") == "solo") & (pl.col("metric") == "time_to_goal")).row(0, named=True)
    assert solo_ttg["n"] == 1
    assert solo_ttg["mean"] == pytest.approx(9.0)
    assert solo_ttg["std"] is None
    assert solo_ttg["ci_low"] is None
    assert solo_ttg["err_plus"] is None

    ghost_ttg = res_ci95.summary.filter((pl.col("planner") == "ghost") & (pl.col("metric") == "time_to_goal")).row(0, named=True)
    assert ghost_ttg["n"] == 0
    assert ghost_ttg["mean"] is None
    assert ghost_ttg["std"] is None

    trim_rows = [
        {
            "episode_id": i,
            "planner": "dwb",
            "seed": i + 1,
            "time_to_goal": v,
            "is_reference": False,
        }
        for i, v in enumerate([1.0, 10.0, 14.0, 100.0])
    ]
    trim_cat = BenchmarkCatalog.open(_make_benchmark(tmp_path / "trim_bench", trim_rows))
    res_trim = run_computation(
        "group_summary",
        trim_cat,
        {
            "group_cols": ["planner"],
            "metrics": ["time_to_goal"],
            "trim_percentile": 0.4,
        },
    )
    assert res_trim.summary.row(0, named=True)["mean"] == pytest.approx(12.0)


def test_radar_profile_branches_log_scale_all_zero_and_null_group(
    tmp_path: pathlib.Path,
) -> None:
    rows = [
        {
            "episode_id": 0,
            "planner": "p1",
            "seed": 1,
            "time_to_goal": 0.0,
            "path_length_m": 1.0,
            "mean_velocity_mps": 0.0,
            "is_reference": False,
        },
        {
            "episode_id": 1,
            "planner": "p2",
            "seed": 1,
            "time_to_goal": 5.0,
            "path_length_m": 3.0,
            "mean_velocity_mps": 0.0,
            "is_reference": False,
        },
        {
            "episode_id": 2,
            "planner": "ghost",
            "seed": 1,
            "time_to_goal": None,
            "path_length_m": None,
            "mean_velocity_mps": 0.0,
            "is_reference": False,
        },
    ]
    cat = BenchmarkCatalog.open(_make_benchmark(tmp_path / "bench", rows))
    res = run_computation(
        "radar_profile",
        cat,
        {"metrics": ["time_to_goal", "path_length_m", "mean_velocity_mps"]},
    )
    assert res.summary.columns == [
        "planner",
        *RadarProfileAnalysis.polars_schema().names(),
    ]
    assert RadarProfileAnalysis.arrow_schema().field("normalized_score").nullable is True
    casted = res.summary.select(RadarProfileAnalysis.polars_schema().names()).to_arrow().cast(RadarProfileAnalysis.arrow_schema())
    assert casted.num_rows == res.summary.height
    by_key = {(r["planner"], r["metric"]): r["normalized_score"] for r in res.summary.to_dicts()}
    assert by_key[("p1", "time_to_goal")] == pytest.approx(1.0)
    assert by_key[("p2", "time_to_goal")] == pytest.approx(0.0)
    assert by_key[("ghost", "time_to_goal")] is None
    assert by_key[("p1", "path_length_m")] == pytest.approx(1.0)
    assert by_key[("p2", "path_length_m")] == pytest.approx(1.0 / 3.0)
    assert by_key[("p1", "mean_velocity_mps")] == pytest.approx(1.0)
    assert by_key[("p2", "mean_velocity_mps")] == pytest.approx(1.0)

    res_log = run_computation(
        "radar_profile",
        cat,
        {"metrics": ["path_length_m"], "log_scale": True},
    )
    log_by_p = {r["planner"]: r["normalized_score"] for r in res_log.summary.to_dicts()}
    assert log_by_p["p1"] == pytest.approx(1.0)
    assert log_by_p["p2"] == pytest.approx(0.5)
    assert log_by_p["ghost"] is None

    # A partial dim_metrics must not silently decide a ranking direction: the error names every
    # unresolved metric with its count, and a caller that knows the direction supplies it.
    partial_cat = BenchmarkCatalog.open(_make_benchmark(tmp_path / "bad_bench", rows, null_direction="time_to_goal"))
    with pytest.raises(ValueError, match=r"1 metric\(s\) have no 'higher'/'lower' direction"):
        resolve_metric_directions(partial_cat, ["time_to_goal"])
    declared = resolve_metric_directions(partial_cat, ["time_to_goal"], declared={"time_to_goal": "lower"})
    assert declared.by_metric == {"time_to_goal": "lower"}
    assert declared.from_override == ("time_to_goal",)
    assert declared.diagnostics[0].startswith("directions_from_param=1:")

    unknown_cat = BenchmarkCatalog.open(_make_benchmark(tmp_path / "unknown_bench", rows))
    with pytest.raises(ValueError, match="have no 'higher'/'lower' direction"):
        resolve_metric_directions(unknown_cat, ["no_such_metric"])


def test_composite_and_top_n_ranking_unrounded_precision(
    tmp_path: pathlib.Path,
) -> None:
    rows = [
        {
            "episode_id": 0,
            "planner": "a_second",
            "seed": 1,
            "time_to_goal": 10.0004,
            "path_length_m": 10.0,
            "is_reference": False,
        },
        {
            "episode_id": 1,
            "planner": "z_first",
            "seed": 1,
            "time_to_goal": 10.0000,
            "path_length_m": 10.0,
            "is_reference": False,
        },
        {
            "episode_id": 2,
            "planner": "slow",
            "seed": 1,
            "time_to_goal": 15.0,
            "path_length_m": 20.0,
            "is_reference": False,
        },
        {
            "episode_id": 3,
            "planner": "fourth",
            "seed": 1,
            "time_to_goal": 20.0,
            "path_length_m": 30.0,
            "is_reference": False,
        },
    ]
    cat = BenchmarkCatalog.open(_make_benchmark(tmp_path / "bench", rows))
    res = run_computation(
        "composite_ranking",
        cat,
        {
            "metrics": ["time_to_goal", "path_length_m"],
            "weights": {"time_to_goal": 1.0, "path_length_m": 1.0},
        },
    )
    assert res.summary.columns == [
        "planner",
        *CompositeRankingAnalysis.polars_schema().names(),
    ]
    ranks = {r["planner"]: (r["rank"], r["composite_score"]) for r in res.summary.to_dicts()}
    assert ranks["z_first"] == (1, pytest.approx(1.0))
    assert ranks["a_second"] == (2, pytest.approx(0.99998))
    assert ranks["slow"] == (3, pytest.approx(0.5))

    top3 = run_computation(
        "top_n_ranking",
        cat,
        {"metrics": ["time_to_goal", "path_length_m"]},
    )
    assert top3.summary["planner"].unique(maintain_order=True).to_list() == [
        "z_first",
        "a_second",
        "slow",
    ]


def test_correlation_paired_diff_and_cluster_bootstrap_ci(
    tmp_path: pathlib.Path,
) -> None:
    corr_rows = [
        {
            "episode_id": 0,
            "planner": "dwb",
            "seed": 1,
            "time_to_goal": 1.0,
            "path_length_m": 1.0,
            "is_reference": False,
        },
        {
            "episode_id": 1,
            "planner": "dwb",
            "seed": 2,
            "time_to_goal": 2.0,
            "path_length_m": 4.0,
            "is_reference": False,
        },
        {
            "episode_id": 2,
            "planner": "dwb",
            "seed": 3,
            "time_to_goal": 3.0,
            "path_length_m": 2.0,
            "is_reference": False,
        },
        {
            "episode_id": 3,
            "planner": "dwb",
            "seed": 4,
            "time_to_goal": 4.0,
            "path_length_m": 5.0,
            "is_reference": False,
        },
        {
            "episode_id": 4,
            "planner": "dwb",
            "seed": 5,
            "time_to_goal": None,
            "path_length_m": 6.0,
            "is_reference": False,
        },
        {
            "episode_id": 5,
            "planner": "dwb",
            "seed": 6,
            "time_to_goal": 5.0,
            "path_length_m": None,
            "is_reference": False,
        },
    ]
    corr_cat = BenchmarkCatalog.open(_make_benchmark(tmp_path / "corr_bench", corr_rows))
    corr = run_computation(
        "correlation_matrix",
        corr_cat,
        {"metrics": ["time_to_goal", "path_length_m"]},
    )
    assert corr.summary.columns == list(CorrelationMatrixAnalysis.polars_schema().names())
    by_pair = {(r["metric_x"], r["metric_y"]): (r["pearson_r"], r["n_pairs"]) for r in corr.summary.to_dicts()}
    assert by_pair[("time_to_goal", "time_to_goal")] == (pytest.approx(1.0), 5)
    assert by_pair[("path_length_m", "path_length_m")] == (pytest.approx(1.0), 5)
    assert by_pair[("time_to_goal", "path_length_m")] == (
        pytest.approx(0.7071067811865475),
        4,
    )

    bench_cat = BenchmarkCatalog.open(_make_benchmark(tmp_path / "bench", _default_rows()))
    pdiff = run_computation(
        "paired_diff",
        bench_cat,
        {"metric": "time_to_goal", "treatment": "s1", "control": "s0", "n_boot": 50},
    )
    assert pdiff.summary.columns == list(PairedDiffAnalysis.polars_schema().names())
    assert pdiff.summary["mean_diff"].to_list() == pytest.approx([4.0])

    cci = run_computation(
        "cluster_bootstrap_ci",
        bench_cat,
        {"metric": "time_to_goal", "group_cols": ["planner"], "n_boot": 50, "seed": 42},
    )
    dwb_cci = cci.summary.filter(pl.col("planner") == "dwb").row(0, named=True)
    assert dwb_cci["estimate"] == pytest.approx(13.0)
    assert dwb_cci["se"] == pytest.approx(0.637597531554897)
    assert dwb_cci["ci_low"] == pytest.approx(12.0)
    assert dwb_cci["ci_high"] == pytest.approx(14.0)
    assert resolve_metric_units(bench_cat, ["time_to_goal"]) == {"time_to_goal": "s"}


@pytest.mark.parametrize(
    ("name", "params"),
    [
        (
            "group_summary",
            {
                "group_cols": ["local_planner"],
                "metrics": ["time_to_goal", "path_length_m"],
            },
        ),
        (
            "distribution_bins",
            {"group_cols": ["local_planner"], "metric": "time_to_goal"},
        ),
        (
            "radar_profile",
            {
                "group_col": "local_planner",
                "metrics": ["time_to_goal", "path_length_m"],
            },
        ),
        (
            "composite_ranking",
            {
                "group_col": "local_planner",
                "metrics": ["time_to_goal", "path_length_m"],
            },
        ),
        (
            "cluster_bootstrap_ci",
            {
                "group_cols": ["local_planner"],
                "metric": "time_to_goal",
                "n_boot": 30,
                "seed": 42,
            },
        ),
    ],
)
def test_grouped_computations_preserve_null_group_key(
    tmp_path: pathlib.Path,
    name: str,
    params: dict[str, object],
) -> None:
    rows = [
        {
            "episode_id": 0,
            "planner": "dwb",
            "seed": 1,
            "time_to_goal": 10.0,
            "path_length_m": 20.0,
            "is_reference": False,
        },
        {
            "episode_id": 1,
            "planner": "dwb",
            "seed": 2,
            "time_to_goal": 12.0,
            "path_length_m": 22.0,
            "is_reference": False,
        },
        {
            "episode_id": 2,
            "planner": "teb",
            "seed": 1,
            "time_to_goal": 14.0,
            "path_length_m": 24.0,
            "is_reference": False,
        },
        {
            "episode_id": 3,
            "planner": "teb",
            "seed": 2,
            "time_to_goal": 16.0,
            "path_length_m": 26.0,
            "is_reference": False,
        },
        {
            "episode_id": 4,
            "planner": "",
            "seed": 1,
            "time_to_goal": 12.0,
            "path_length_m": 22.0,
            "is_reference": False,
        },
        {
            "episode_id": 5,
            "planner": "",
            "seed": 2,
            "time_to_goal": 14.0,
            "path_length_m": 24.0,
            "is_reference": False,
        },
    ]
    cat = BenchmarkCatalog.open(_make_benchmark(tmp_path / "null_bench", rows))
    res = run_computation(name, cat, params)
    assert set(res.summary.get_column("local_planner").to_list()) == {
        "dwb",
        "teb",
        None,
    }
    if name == "radar_profile":
        null_scores = {r["metric"]: r["normalized_score"] for r in res.summary.filter(pl.col("local_planner").is_null()).to_dicts()}
        assert null_scores["time_to_goal"] == pytest.approx(11.0 / 13.0)
        assert null_scores["path_length_m"] == pytest.approx(21.0 / 23.0)
