"""Analysis functions from tools.py on synthetic Polars DataFrames."""

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("mcp")
pytest.importorskip("arena_evaluation_mcp")

# Metric declarations as calculators would publish them (PRIMARY_OUTPUTS /
# OUTPUT_DIRECTIONS), passed straight into the frame-level functions.
_DECLARATIONS = {
    "collision_metrics": {
        "primary_outputs": ["success", "collision_amount"],
        "output_directions": {"success": "higher", "collision_amount": "lower"},
    },
    "path_metrics": {
        "primary_outputs": ["time_to_goal"],
        "output_directions": {"time_to_goal": "lower"},
    },
    "smoothness": {
        "primary_outputs": ["jerk_mean"],
        "output_directions": {"jerk_mean": "lower"},
    },
}


@pytest.fixture
def sample_df() -> "pl.DataFrame":
    return pl.DataFrame(
        {
            "local_planner": ["dwb"] * 6 + ["teb"] * 6 + ["dwa"] * 6,
            "stage": ["stage_a", "stage_b"] * 9,
            "map": ["map_empty"] * 18,
            "success": [1.0] * 5 + [0.0] + [0.9] * 4 + [0.0, 0.0] + [0.8] * 6,
            "time_to_goal": [45.0, 48.0, 42.0, 47.0, 44.0, 0.0, 52.0, 55.0, 50.0, 53.0, 0.0, 0.0, 60.0, 62.0, 58.0, 61.0, 59.0, 63.0],
            "collision_amount": [0.1, 0.0, 0.2, 0.1, 0.0, 0.0, 0.3, 0.2, 0.4, 0.3, 0.0, 0.0, 0.5, 0.6, 0.4, 0.5, 0.6, 0.5],
            "jerk_mean": [0.01, 0.02, 0.01, 0.02, 0.01, 0.0, 0.03, 0.04, 0.03, 0.04, 0.0, 0.0, 0.05, 0.06, 0.05, 0.06, 0.05, 0.04],
        }
    )


class TestComparePlanners:
    def test_ranks_planners(self, sample_df):
        from arena_evaluation_mcp.tools import _compare_planners_frame

        result = _compare_planners_frame(
            sample_df,
            metrics=["success", "time_to_goal", "collision_amount"],
            declarations=_DECLARATIONS,
        )
        rankings = result["rankings"]
        assert len(rankings) == 3  # dwb, teb, dwa
        # dwb ranks first: highest success, lowest time/collisions
        assert rankings[0]["local_planner"] == "dwb"
        assert rankings[0]["rank"] == 1
        assert rankings[0]["composite_score"] > rankings[-1]["composite_score"]

    def test_filters_to_specified_planners(self, sample_df):
        from arena_evaluation_mcp.tools import _compare_planners_frame

        result = _compare_planners_frame(
            sample_df,
            metrics=["success"],
            declarations=_DECLARATIONS,
            planners=["dwb", "teb"],
        )
        rankings = result["rankings"]
        assert len(rankings) == 2
        planners = [r["local_planner"] for r in rankings]
        assert "dwa" not in planners

    def test_normalization_bounds(self, sample_df):
        from arena_evaluation_mcp.tools import _compare_planners_frame

        result = _compare_planners_frame(
            sample_df,
            metrics=["success", "time_to_goal", "collision_amount"],
            declarations=_DECLARATIONS,
        )
        for r in result["rankings"]:
            for k in r:
                if k.endswith("_norm"):
                    assert 0.0 <= r[k] <= 1.0, f"{k} = {r[k]} out of bounds"

    def test_group_by_stage(self, sample_df):
        from arena_evaluation_mcp.tools import _compare_planners_frame

        result = _compare_planners_frame(
            sample_df,
            metrics=["success"],
            declarations=_DECLARATIONS,
            group_by_stage=True,
        )
        rankings = result["rankings"]
        # 3 planners x 2 stages = 6 rows
        assert len(rankings) == 6
        stages = {r.get("stage") for r in rankings}
        assert "stage_a" in stages

    def test_handles_empty_metrics(self, sample_df):
        from arena_evaluation_mcp.tools import _compare_planners_frame

        result = _compare_planners_frame(
            sample_df,
            metrics=["nonexistent"],
            declarations=_DECLARATIONS,
        )
        assert "error" in result


class TestFindTopN:
    def test_returns_exact_count(self, sample_df):
        from arena_evaluation_mcp.tools import _find_top_n_frame

        result = _find_top_n_frame(
            sample_df,
            metrics=["success"],
            declarations=_DECLARATIONS,
            n=2,
        )
        assert len(result["top_n"]) == 2

    def test_best_planner_first(self, sample_df):
        from arena_evaluation_mcp.tools import _find_top_n_frame

        result = _find_top_n_frame(
            sample_df,
            metrics=["success", "time_to_goal"],
            declarations=_DECLARATIONS,
            n=3,
            weights=[1.0, 1.0],
        )
        # dwb has the highest mean success and the lowest mean time
        assert result["top_n"][0]["planner"] == "dwb"

    def test_custom_weights(self, sample_df):
        from arena_evaluation_mcp.tools import _find_top_n_frame

        result = _find_top_n_frame(
            sample_df,
            metrics=["success", "time_to_goal"],
            declarations=_DECLARATIONS,
            n=3,
            weights=[0.0, 1.0],
        )
        # only time_to_goal weighted (lower is better): teb's 0.0 outliers
        # give it the lowest mean (35.0 vs dwb 37.7 vs dwa 60.5)
        assert result["top_n"][0]["planner"] == "teb"

    def test_single_metric(self, sample_df):
        from arena_evaluation_mcp.tools import _find_top_n_frame

        result = _find_top_n_frame(
            sample_df,
            metrics=["collision_amount"],
            declarations=_DECLARATIONS,
            n=1,
        )
        assert len(result["top_n"]) == 1


class TestCorrelation:
    def test_correlation_bounds(self, sample_df):
        pytest.importorskip("pandas")
        from arena_evaluation_mcp.tools import _correlation_frame

        metrics = ["success", "time_to_goal", "collision_amount", "jerk_mean"]
        result = _correlation_frame(sample_df, metrics)
        matrix = result["correlation_matrix"]
        for mi in metrics:
            for mj in metrics:
                v = matrix[mi][mj]
                assert -1.0 <= v <= 1.0, f"corr({mi}, {mj}) = {v} out of bounds"

    def test_self_correlation_is_one(self, sample_df):
        pytest.importorskip("pandas")
        from arena_evaluation_mcp.tools import _correlation_frame

        metrics = ["success", "time_to_goal"]
        result = _correlation_frame(sample_df, metrics)
        matrix = result["correlation_matrix"]
        for m in metrics:
            assert abs(matrix[m][m] - 1.0) < 0.001, f"{m} self-corr != 1"

    def test_too_few_metrics_errors(self, sample_df):
        from arena_evaluation_mcp.tools import _correlation_frame

        result = _correlation_frame(sample_df, ["success"])
        assert "error" in result


class TestLowerBetter:
    def test_directions_come_from_declarations(self):
        from arena_evaluation_mcp.tools import _is_lower_better

        assert _is_lower_better("success", _DECLARATIONS) is False
        assert _is_lower_better("time_to_goal", _DECLARATIONS) is True

    def test_unlisted_defaults_to_lower(self):
        from arena_evaluation_mcp.tools import _is_lower_better

        assert _is_lower_better("zz_unknown_metric", _DECLARATIONS) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
