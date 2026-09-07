from __future__ import annotations

import os
import pathlib
from ament_index_python.packages import get_package_share_directory

from .schemas import RunDescriptor, EpisodeDescriptor
from .manifest import MetadataWriter


class FolderManager:
    """Manages directory paths within data_root for evaluation runs."""

    def __init__(self, data_root: pathlib.Path | None = None):
        if data_root is None:
            self.data_root = pathlib.Path(
                get_package_share_directory("arena_evaluation")
            ) / "data"
        else:
            self.data_root = pathlib.Path(data_root).resolve()

        if not self.data_root.exists():
            self.data_root.mkdir(parents=True, exist_ok=True)

    def _safe_resolve(self, target: pathlib.Path) -> pathlib.Path:
        """Resolve path and ensure it remains within data_root."""
        resolved = target.resolve()
        try:
            resolved.relative_to(self.data_root)
        except ValueError:
            raise ValueError(f"Path {resolved} is outside data_root {self.data_root}")
        return resolved

    def episodes_dir(self, benchmark_id: str) -> pathlib.Path:
        """Return the episodes directory for a benchmark or run."""
        for cand in [
            self.data_root / benchmark_id / "episodes",
            self.data_root / "runs" / benchmark_id / "episodes",
            self.data_root / "benchmarks" / benchmark_id / "episodes",
        ]:
            if cand.exists():
                return self._safe_resolve(cand)
        return self._safe_resolve(self.data_root / benchmark_id / "episodes")

    def episode_dir(self, benchmark_id: str, episode_id: int) -> pathlib.Path:
        """Return the directory for a specific episode."""
        return self._safe_resolve(self.episodes_dir(benchmark_id) / f"episode_{episode_id:03d}")

    def discover_episodes(self, benchmark_id: str) -> list[EpisodeDescriptor]:
        """Discover valid episode directories and their metadata."""
        eps_dir = self.episodes_dir(benchmark_id)
        if not eps_dir.exists() or not eps_dir.is_dir():
            return []

        episodes: list[EpisodeDescriptor] = []
        for ep_dir in sorted(eps_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            try:
                ep_id = int(ep_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue

            yaml_path = ep_dir / f"{ep_dir.name}.yaml"
            if not yaml_path.exists():
                yaml_path = ep_dir / "metadata.yaml"
            if not yaml_path.exists():
                continue

            try:
                meta = MetadataWriter.read(yaml_path)
                episodes.append(EpisodeDescriptor(
                    episode_dir=str(ep_dir),
                    benchmark_id=benchmark_id,
                    episode_id=ep_id,
                    planner=meta.planner,
                    stage=meta.stage,
                    map=meta.map,
                    is_reference=meta.is_reference,
                    reference_type=meta.reference_type,
                ))
            except Exception:
                continue

        return episodes

    def mcap_path_for_episode(self, episode_dir: pathlib.Path) -> pathlib.Path:
        """Get MCAP file path for an episode directory."""
        canonical = episode_dir / f"{episode_dir.name}.mcap"
        if canonical.exists():
            return self._safe_resolve(canonical)
        candidates = sorted(p for p in episode_dir.glob("*.mcap") if p.stat().st_size > 0)
        if candidates:
            return self._safe_resolve(candidates[0])
        return self._safe_resolve(canonical)

    def extracted_topics_path_for_episode(self, episode_dir: pathlib.Path) -> pathlib.Path:
        """Return topics directory for an episode."""
        return self._safe_resolve(episode_dir / "topics")

    def combined_metrics_path(self, benchmark_id: str) -> pathlib.Path:
        """Get path for combined benchmark metrics Parquet file."""
        for cand in [
            self.data_root / benchmark_id / "combined_metrics.parquet",
            self.data_root / "runs" / benchmark_id / "combined_metrics.parquet",
            self.data_root / "benchmarks" / benchmark_id / "combined_metrics.parquet",
        ]:
            if cand.parent.exists():
                return self._safe_resolve(cand)
        return self._safe_resolve(self.data_root / benchmark_id / "combined_metrics.parquet")

