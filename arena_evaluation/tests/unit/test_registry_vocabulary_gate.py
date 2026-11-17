"""Repo-wide gate: the registry vocabulary is the only evaluation vocabulary.

The rename closed L1-015/L1-031 in one pass: the pipeline publishes the aligned frame under the
registry's own column names and the calculators read the registry's own topic keys. This test keeps
the removed names from creeping back. ``pos_x``/``pos_y`` are deliberately not in the list: they
remain the registry columns of ``robot_initialpose``, ``robot_goal_pose`` and ``robot_task_pose``.
Fleet and task-pose readers therefore keep spelling them, while the aligned robot track is ``x``/``y``.
"""

from __future__ import annotations

import re
from pathlib import Path


import polars as pl

from arena_evaluation.processing.pipeline import (
    _align_bundle,
    _calculator_topic_names,
)
from arena_evaluation.storage.schemas import TopicBundle

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_FILE = Path(__file__).resolve()

#: Removed vocabulary: (label, pattern). Every pattern is a complete legacy spelling, so the
#: registry names that merely contain a removed word (``scan_min_m``, ``uncertainty_1sigma_dba``,
#: ``..._anyone``/``..._robot``) never match.
FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "identity rename map",
        re.compile(
            r"\b(_TOPIC_NAMES|_PRIMARY|_GROUND_TRUTH_ALIASES|_SCAN|_COLLISIONS|_ACOUSTICS)\b"
        ),
    ),
    (
        "aligned column",
        re.compile(
            r"(pos_x_gt|pos_y_gt|yaw_gt|\bvel_linear\b|\bvel_angular\b|\bscan_range_max\b"
            r"|\bn_rays\b|\bcollision_wall\b|\bcollision_static\b|\bcollision_pedestrian\b"
            r"|\bscan_min\b)"
        ),
    ),
    (
        "calculator topic key",
        re.compile(
            r'("pedestrian_tracks"|"pedestrian_tracks_physics"|"collision_event"|"collision_contacts")'
        ),
    ),
    ("collision topic key", re.compile(r'"collision_events"')),
    (
        "metric key",
        re.compile(
            r'("used_elevator_during_alarm"|"time_waiting_at_doors"|"elevator_rides")'
        ),
    ),
    ("series flag", re.compile(r"\bis_worst_frame\b")),
    ("acoustics column", re.compile(r"\buncertainty_dba\b")),
)

#: Files that legitimately name a legacy input or upstream schema.
#: - the ROS ingress spec and its test name the ``/collision_events`` topic (not a calculator key);
#: - the MuSoHu adapter preserves its upstream dataset's scan field names.
ALLOWED: dict[tuple[str, str], str] = {
    (
        "arena_evaluation/ingestion/topics.py",
        "collision topic key",
    ): "the ROS ingress spec keeps /collision_events",
    (
        "tests/unit/test_ingestion_topics.py",
        "collision topic key",
    ): "the ingress test names the ROS topic key",
    (
        "tests/test_adapter_musohu.py",
        "aligned column",
    ): "the MuSoHu adapter preserves the upstream scan_min field",
}


def _scan_suffixes(path: Path) -> bool:
    return path.suffix in {".py", ".yaml", ".yml", ".json", ".toml"}


def _files() -> list[Path]:
    roots = (
        REPO_ROOT / "arena_evaluation",
        REPO_ROOT / "tests",
        REPO_ROOT / "config",
        REPO_ROOT / "configs",
    )
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        found.extend(p for p in root.rglob("*") if p.is_file() and _scan_suffixes(p))
    return found


def _violations(path: Path) -> list[str]:
    rel = path.relative_to(REPO_ROOT).as_posix()
    if path == GATE_FILE:
        return []
    try:
        lines = path.read_text().splitlines()
    except UnicodeDecodeError:
        return []
    found: list[str] = []
    for label, pattern in FORBIDDEN:
        if (rel, label) in ALLOWED:
            continue
        for line_number, line in enumerate(lines, start=1):
            if pattern.search(line):
                found.append(f"{rel}:{line_number}: {label}: {line.strip()[:140]}")
    return found


def test_no_removed_legacy_name_remains_in_code_tests_or_configs() -> None:
    violations = [violation for path in _files() for violation in _violations(path)]
    assert violations == [], "\n".join(violations)


def test_the_aligned_frame_speaks_registry_names() -> None:
    tracks = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "yaw": [0.0, 0.0],
            "speed_mps": [1.0, 1.0],
            "omega_z": [0.0, 0.0],
        }
    )
    scan = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "scan_min_m": [1.0, 1.0],
            "range_max_m": [9.0, 9.0],
            "n_valid_rays": [5, 5],
        }
    )
    collisions = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "event_count": [0, 0],
            "wall_count": [0, 0],
            "static_count": [0, 0],
            "ped_count": [0, 0],
        }
    )
    aligned = _align_bundle(
        TopicBundle(
            robot_tracks=tracks,
            robot_scan_summary=scan,
            robot_collision_events=collisions,
        )
    )
    assert aligned is not None
    assert {"t_ns", "x", "y", "yaw", "speed_mps", "omega_z"} <= set(aligned.columns)
    assert {
        "scan_min_m",
        "range_max_m",
        "n_valid_rays",
        "event_count",
        "wall_count",
        "static_count",
        "ped_count",
    } <= set(aligned.columns)
    assert not {
        "pos_x",
        "pos_y",
        "vel_linear",
        "vel_angular",
        "scan_min",
        "range_max",
        "collision_event",
    } & set(aligned.columns)

    assert _calculator_topic_names(
        TopicBundle(
            robot_tracks=tracks,
            pedestrian_arena_peds=pl.DataFrame(
                {"t_ns": [0], "ped_id": ["p"], "x": [0.0], "y": [0.0]}
            ),
            pedestrian_agent_states=pl.DataFrame(
                {"t_ns": [0], "ped_id": ["p"], "x": [0.0], "y": [0.0]}
            ),
            robot_collision_events=collisions,
            robot_collision_contacts=pl.DataFrame({"t_ns": [], "obstacle_id": []}),
        )
    ) >= {
        "robot_tracks",
        "pedestrian_arena_peds",
        "pedestrian_agent_states",
        "robot_collision_events",
        "robot_collision_contacts",
    }
