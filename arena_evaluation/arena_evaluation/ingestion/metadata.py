from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

from ..storage.planner_names import split_planner_name
from ..storage.schemas import RunMetadata


class IngestionMetadata:
    """Helper to generate metadata during ingestion (recording)."""

    @staticmethod
    def get_git_sha(workspace_dir: str) -> str | None:
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace_dir, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except Exception:
            return None

    @staticmethod
    def is_git_dirty(workspace_dir: str) -> bool:
        try:
            result = subprocess.run(["git", "status", "--porcelain"], cwd=workspace_dir, capture_output=True, text=True, check=True)
            return len(result.stdout.strip()) > 0
        except Exception:
            return False

    @staticmethod
    def resolve_git_workspace(workspace_dir: str) -> str:
        """Resolve an Arena checkout from either a checkout or colcon workspace.

        Container recordings normally run below ``/opt/arena_ws`` while the
        Git repository itself is ``/opt/arena_ws/src/Arena``.  Native installs
        use the same colcon layout.  Falling back to the supplied directory
        preserves the previous null-SHA behavior for non-Git deployments.
        """
        start = Path(workspace_dir).expanduser().resolve()
        ancestry = [start, *start.parents]
        candidates = [candidate for parent in ancestry for candidate in (parent, parent / "src" / "Arena")]
        for candidate in candidates:
            if (candidate / ".git").exists():
                return str(candidate)
        return str(start)

    @staticmethod
    def create_episode_metadata(
        benchmark_id: str,
        planner: str,
        stage: str,
        map_name: str,
        episode_id: int,
        robot_model: str,
        workspace_dir: str = "/opt/arena_ws",
        env_ns_root: str | None = None,
        is_reference: bool = False,
        reference_type: str | None = None,
        suite_name: str = "",
        contest_name: str = "",
        episodes_requested: int = 0,
        local_planner: str | None = None,
        inter_planner: str | None = None,
        task_generator_episode_id: int | None = None,
        agent_name: str = "",
    ) -> RunMetadata:
        """Create metadata for a single episode (new flat structure)."""

        fallback_lp, fallback_ip = split_planner_name(planner)
        git_workspace = IngestionMetadata.resolve_git_workspace(workspace_dir)
        return RunMetadata(
            benchmark_id=benchmark_id,
            planner=planner,
            robot_model=[robot_model],
            map=map_name,
            stage=stage,
            episode_id=episode_id,
            episodes_requested=episodes_requested,
            suite_name=suite_name,
            contest_name=contest_name,
            local_planner=local_planner if local_planner else fallback_lp,
            inter_planner=inter_planner if inter_planner else fallback_ip,
            agent_name=agent_name,
            task_generator_episode_id=task_generator_episode_id,
            recording_started_at=datetime.datetime.now(datetime.UTC).isoformat(),
            arena_git_sha=IngestionMetadata.get_git_sha(git_workspace),
            arena_git_dirty=IngestionMetadata.is_git_dirty(git_workspace),
            python_version=sys.version.split()[0],
            ros_distro=os.environ.get("ROS_DISTRO", "unknown"),
            env_ns_root=env_ns_root,
            is_reference=is_reference,
            reference_type=reference_type,
        )
