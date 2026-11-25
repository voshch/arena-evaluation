"""Golden, schema-contract, and EvidenceStore tests for series-grain analysis computations."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.analytics import (
    COMPUTATIONS,
    BenchmarkCatalog,
    EvidenceStore,
    list_computations,
    run_computation,
)
from arena_evaluation.analytics.computations import SeriesProfileParams
from star_benchmark import write_benchmark
from arena_evaluation.storage.registry import series as series_reg
from arena_evaluation.storage.registry.analysis import SeriesProfileAnalysis


def _write_series_benchmark(root: pathlib.Path) -> pathlib.Path:
    write_benchmark(
        root,
        [
            {
                "episode_id": 0,
                "planner": "dwb",
                "local_planner": "dwb",
                "seed": 1,
                "is_reference": False,
            },
            {
                "episode_id": 1,
                "planner": "teb",
                "local_planner": "teb",
                "seed": 2,
                "is_reference": False,
            },
            {
                "episode_id": 2,
                "planner": "",
                "local_planner": "",
                "seed": 3,
                "is_reference": False,
            },
            {
                "episode_id": 3,
                "planner": "ref_p",
                "local_planner": "ref_p",
                "seed": 99,
                "is_reference": True,
            },
        ],
    )

    # Series rows covering: idle/linear phases, bin_x=0.1, NULL x, NULL y, n=1 cell, acoustic dba
    rows = [
        # episode 0 (dwb, seed 1)
        (0, "linear", 0.5, 0.12, 10.0, 60.0),
        (0, "linear", 0.5, 0.14, 14.0, 64.0),
        (0, "linear", 0.5, 0.21, 20.0, 70.0),
        (0, "linear", 0.5, 0.23, 24.0, 72.0),
        (0, "linear", 0.5, 0.31, 30.0, 66.0),
        (0, "linear", 0.5, 0.33, 36.0, 68.0),
        (0, "linear", 0.5, 0.41, 50.0, 74.0),  # n=1 cell at x=0.4
        (0, "linear", 0.5, None, 99.0, 80.0),  # NULL x dropped
        (0, "linear", 0.5, 0.51, None, None),  # NULL y -> n=0
        (0, "idle", 0.0, 0.01, 4.0, 40.0),
        (0, "idle", 0.0, 0.02, 6.0, 42.0),
        # episode 1 (teb, seed 2)
        (1, "linear", 0.5, 0.11, 18.0, 62.0),
        (1, "linear", 0.5, 0.14, 22.0, 64.0),
        (1, "linear", 0.5, 0.21, 26.0, 66.0),
        (1, "linear", 0.5, 0.24, 30.0, 68.0),
        (1, "idle", 0.0, 0.01, 8.0, 44.0),
        # episode 2 (planner="", local_planner=None, seed 3)
        (2, "linear", 0.5, 0.11, 12.0, 58.0),
        (2, "linear", 0.5, 0.14, 16.0, 62.0),
        (2, "idle", 0.0, 0.01, 5.0, 41.0),
        # episode 3 (reference -> excluded by default)
        (3, "linear", 0.5, 0.11, 999.0, 99.0),
    ]
    tbl = series_reg.SeriesCharacterization
    for ep_id in (0, 1, 2, 3):
        ep_rows = [r for r in rows if r[0] == ep_id]
        base_cols = {
            "episode_id": [r[0] for r in ep_rows],
            "phase_kind": [r[1] for r in ep_rows],
            "speed_target_mps": [r[2] for r in ep_rows],
            "vx_achieved_mps": [r[3] for r in ep_rows],
            "power_total_w": [r[4] for r in ep_rows],
            "dba": [r[5] for r in ep_rows],
        }
        full_schema = tbl.polars_schema()
        defaults = {"run_id": "run", "robot_id": "robot_0", "env_id": "env_0"}
        for col_name in full_schema:
            if col_name not in base_cols:
                base_cols[col_name] = [defaults.get(col_name)] * len(ep_rows)
        ep_df = pl.DataFrame(base_cols, schema=full_schema)
        out_path = root / f"episodes/episode_{ep_id:03d}" / tbl.relative_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ep_df.write_parquet(out_path)
    return root


def test_series_profile_bin_trim_reduce_error_and_smoothing_golden(
    tmp_path: pathlib.Path,
) -> None:
    cat = BenchmarkCatalog.open(_write_series_benchmark(tmp_path / "bench"))

    res_std = run_computation(
        "series_profile",
        cat,
        {
            "source": "v_series_characterization",
            "filter": {"local_planner": "dwb", "phase_kind": "linear"},
            "x": "vx_achieved_mps",
            "metric": "power_total_w",
            "bin_x": 0.1,
            "reduce": "mean",
            "error_mode": "std",
        },
    )
    assert res_std.summary.columns == list(SeriesProfileAnalysis.polars_schema().names())
    by_x = {round(r["x"], 1): r for r in res_std.summary.to_dicts()}
    assert set(by_x) == {0.1, 0.2, 0.3, 0.4, 0.5}
    # bin 0.1: [10.0, 14.0] -> mean=12.0, sample std=2.8284271247461903, n=2
    assert by_x[0.1]["n"] == 2
    assert by_x[0.1]["value"] == pytest.approx(12.0)
    assert by_x[0.1]["err_plus"] == pytest.approx(2.8284271247461903)
    assert by_x[0.1]["err_minus"] == pytest.approx(2.8284271247461903)
    # bin 0.4: n=1 -> value=50.0, err is None
    assert by_x[0.4]["n"] == 1
    assert by_x[0.4]["value"] == pytest.approx(50.0)
    assert by_x[0.4]["err_plus"] is None
    assert by_x[0.4]["err_minus"] is None
    # bin 0.5: only NULL y -> n=0, value/err None
    assert by_x[0.5]["n"] == 0
    assert by_x[0.5]["value"] is None
    assert by_x[0.5]["err_plus"] is None

    # sem and ci95 at bin 0.1 (n=2, std=sqrt(8)=2.8284271247461903):
    # sem = sqrt(8)/sqrt(2) = 2.0, ci95 = 1.96 * 2.0 = 3.92
    res_sem = run_computation(
        "series_profile",
        cat,
        {
            "source": "v_series_characterization",
            "filter": {"local_planner": "dwb", "phase_kind": "linear"},
            "x": "vx_achieved_mps",
            "metric": "power_total_w",
            "bin_x": 0.1,
            "reduce": "mean",
            "error_mode": "sem",
        },
    )
    row_sem = res_sem.summary.sort("x").row(0, named=True)
    assert row_sem["err_plus"] == pytest.approx(2.0)

    res_ci95 = run_computation(
        "series_profile",
        cat,
        {
            "source": "v_series_characterization",
            "filter": {"local_planner": "dwb", "phase_kind": "linear"},
            "x": "vx_achieved_mps",
            "metric": "power_total_w",
            "bin_x": 0.1,
            "reduce": "mean",
            "error_mode": "ci95",
        },
    )
    row_ci95 = res_ci95.summary.sort("x").row(0, named=True)
    assert row_ci95["err_plus"] == pytest.approx(3.92)

    # reduce="leq" & "max" on dba:
    # bin 0.1 dba=[60.0, 64.0] -> leq = 10*log10((10^6 + 10^6.4)/2) = 62.44505355586191
    res_leq = run_computation(
        "series_profile",
        cat,
        {
            "source": "v_series_characterization",
            "filter": {"local_planner": "dwb", "phase_kind": "linear"},
            "x": "vx_achieved_mps",
            "metric": "dba",
            "bin_x": 0.1,
            "reduce": "leq",
        },
    )
    leq_01 = res_leq.summary.sort("x").row(0, named=True)
    assert leq_01["value"] == pytest.approx(62.44505355586191)
    assert leq_01["err_plus"] is None

    res_max = run_computation(
        "series_profile",
        cat,
        {
            "source": "v_series_characterization",
            "filter": {"local_planner": "dwb", "phase_kind": "linear"},
            "x": "vx_achieved_mps",
            "metric": "dba",
            "bin_x": 0.1,
            "reduce": "max",
        },
    )
    assert res_max.summary.sort("x").row(0, named=True)["value"] == pytest.approx(64.0)

    res_smooth = run_computation(
        "series_profile",
        cat,
        {
            "source": "v_series_characterization",
            "filter": {"local_planner": "dwb", "phase_kind": ("linear",)},
            "x": "vx_achieved_mps",
            "metric": "power_total_w",
            "bin_x": 0.1,
            "trim_percentile": 0.05,
            "reduce": "mean",
            "error_mode": "sem",
            "smooth_window": 3,
        },
    )
    smooth_by_x = {round(r["x"], 1): r for r in res_smooth.summary.to_dicts()}
    # After trim_percentile=0.05 on 2-element bins, NULL y at 0.5 is dropped -> bins [0.1, 0.2, 0.3, 0.4]
    # Unsmoothed values: [12.0, 22.0, 33.0, 50.0]; errors: [2.0, 2.0, 3.0, None]
    # Smoothed value at 0.1: (12 + 12 + 22) / 3 = 15.333333333333334; at 0.2: (12+22+33)/3 = 22.333333333333332
    assert smooth_by_x[0.1]["value"] == pytest.approx((12.0 + 12.0 + 22.0) / 3.0)
    assert smooth_by_x[0.2]["value"] == pytest.approx((12.0 + 22.0 + 33.0) / 3.0)
    assert smooth_by_x[0.1]["err_plus"] == pytest.approx((2.0 + 2.0 + 2.0) / 3.0)
    # 0.3 window includes 0.4 (err=None) -> smoothed err is None; n stays unsmoothed (n=2)
    assert smooth_by_x[0.3]["err_plus"] is None
    assert smooth_by_x[0.3]["n"] == 2


def test_series_source_validation_null_group_and_bar_summary_constants(
    tmp_path: pathlib.Path,
) -> None:
    cat = BenchmarkCatalog.open(_write_series_benchmark(tmp_path / "bench"))

    with pytest.raises(ValueError, match="Unsupported computation source"):
        run_computation(
            "series_profile",
            cat,
            {
                "source": "nonexistent_view",
                "x": "vx_achieved_mps",
                "metric": "power_total_w",
            },
        )
    with pytest.raises(ValueError, match="Unsupported computation source"):
        run_computation(
            "series_profile",
            cat,
            {
                "source": "dim_episodes",
                "x": "vx_achieved_mps",
                "metric": "power_total_w",
            },
        )
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        run_computation(
            "series_profile",
            cat,
            {
                "source": "v_series_characterization",
                "filter": {"no_such_col": "x"},
                "x": "vx_achieved_mps",
                "metric": "power_total_w",
            },
        )

    res_prof_null = run_computation(
        "series_profile",
        cat,
        {
            "source": "v_series_characterization",
            "filter": {"phase_kind": "linear"},
            "x": "vx_achieved_mps",
            "metric": "power_total_w",
            "group_cols": ["local_planner"],
            "bin_x": 0.1,
        },
    )
    assert set(res_prof_null.summary.get_column("local_planner").to_list()) == {
        "dwb",
        "teb",
        None,
    }

    res_grp_null = run_computation(
        "group_summary",
        cat,
        {
            "source": "v_series_characterization",
            "filter": {"phase_kind": "linear"},
            "group_cols": ["local_planner"],
            "metrics": ["power_total_w"],
        },
    )
    assert set(res_grp_null.summary.get_column("local_planner").to_list()) == {
        "dwb",
        "teb",
        None,
    }

    bar_res = run_computation(
        "group_summary",
        cat,
        {
            "source": "v_series_characterization",
            "filter": {"phase_kind": "idle"},
            "group_cols": ["local_planner"],
            "metrics": ["power_total_w"],
            "error_mode": "std",
        },
    )
    bar_by_p = {r["local_planner"]: r for r in bar_res.summary.to_dicts()}
    assert bar_by_p["dwb"]["mean"] == pytest.approx(5.0)
    assert bar_by_p["dwb"]["std"] == pytest.approx(1.4142135623730951)
    assert bar_by_p["teb"]["mean"] == pytest.approx(8.0)
    assert bar_by_p["teb"]["std"] is None
    assert bar_by_p[None]["mean"] == pytest.approx(5.0)

    summary_res = run_computation(
        "group_summary",
        cat,
        {
            "source": "v_series_characterization",
            "group_cols": ["phase_kind", "speed_target_mps"],
            "metrics": ["power_total_w"],
        },
    )
    sum_by_k = {(r["phase_kind"], r["speed_target_mps"]): r["mean"] for r in summary_res.summary.to_dicts()}
    assert sum_by_k[("idle", 0.0)] == pytest.approx((4.0 + 6.0 + 8.0 + 5.0) / 4.0)

    assert "series_profile" in list_computations()
    assert COMPUTATIONS["series_profile"].params is SeriesProfileParams
    assert COMPUTATIONS["series_profile"].output is SeriesProfileAnalysis

    store = EvidenceStore(tmp_path / "bench")
    r_min = run_computation(
        "series_profile",
        cat,
        {
            "source": "v_series_characterization",
            "x": "vx_achieved_mps",
            "metric": "power_total_w",
        },
        evidence_store=store,
    )
    r_full = run_computation(
        "series_profile",
        cat,
        {
            "source": "v_series_characterization",
            "filter": None,
            "exclude_references": True,
            "x": "vx_achieved_mps",
            "metric": "power_total_w",
            "group_cols": [],
            "bin_x": None,
            "reduce": "mean",
            "error_mode": "std",
            "trim_percentile": 0.0,
            "smooth_window": None,
        },
        evidence_store=store,
    )
    assert r_min.evidence_id == r_full.evidence_id
    assert len(store.list_records()) == 1
