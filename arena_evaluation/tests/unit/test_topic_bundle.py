"""Unit tests for TopicBundle.available() and the native-topic collection of the pipeline."""

import pytest

pytest.importorskip("polars")

import polars as pl
from arena_evaluation.storage.schemas import TopicBundle


def test_topic_bundle_available_reports_only_populated_fields():
    odom = pl.DataFrame({"t_ns": [0, 1], "odom_x": [0.0, 1.0]})
    schedule = pl.DataFrame({"phase_label": ["a"], "phase_kind": ["linear"]})
    bundle = TopicBundle(robot_odom=odom, robot_characterization_schedule=schedule)
    assert bundle.available() == {"robot_odom", "robot_characterization_schedule"}

