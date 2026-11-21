from __future__ import annotations

import os
import pathlib
from collections.abc import Callable, Mapping

from ament_index_python.packages import get_package_share_directory

from .data_root import benchmarks_root
from .manifest import MetadataWriter
from .schemas import EpisodeDescriptor


class FolderManager:
    """Paths of the benchmarks root, the directory whose children are the `benchmarks/<id>` runs."""

    def __init__(
        self,
        data_root: pathlib.Path | str | None = None,
        *,
        environ: Mapping[str, str] = os.environ,
        share_directory: Callable[[str], str] = get_package_share_directory,
    ):
        root = pathlib.Path(data_root) if data_root is not None else benchmarks_root(environ=environ, share_directory=share_directory)
        self.data_root = root.resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)

    def _safe_resolve(self, target: pathlib.Path) -> pathlib.Path:
        """Resolve path and ensure it remains within data_root."""
        resolved = target.resolve()
        try:
            resolved.relative_to(self.data_root)
        except ValueError:
            raise ValueError(f"Path {resolved} is outside data_root {self.data_root}") from None
        return resolved

    def benchmark_dir(self, benchmark_id: str) -> pathlib.Path:
        """Return the benchmark's directory under the benchmarks root."""
        return self._safe_resolve(self.data_root / benchmark_id)

    def episodes_dir(self, benchmark_id: str) -> pathlib.Path:
        """Return the episodes directory of a benchmark."""
        return self._safe_resolve(self.benchmark_dir(benchmark_id) / "episodes")

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
            ep = self.describe_episode(ep_dir, benchmark_id)
            if ep is not None:
                episodes.append(ep)
        return episodes

    @staticmethod
    def describe_episode(ep_dir: pathlib.Path, benchmark_id: str | None = None) -> EpisodeDescriptor | None:
        """Descriptor of one `episode_NNN` directory, None when it is not a readable episode."""
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
            return None
        try:
            ep_id = int(ep_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            return None

        yaml_path = ep_dir / f"{ep_dir.name}.yaml"
        if not yaml_path.exists():
            yaml_path = ep_dir / "metadata.yaml"
        if not yaml_path.exists():
            return None

        meta = MetadataWriter.read(yaml_path)
        return EpisodeDescriptor(
            episode_dir=str(ep_dir),
            benchmark_id=benchmark_id or meta.benchmark_id or ep_dir.parent.parent.name,
            episode_id=ep_id,
            planner=meta.planner,
            stage=meta.stage,
            map=meta.map,
            is_reference=meta.is_reference,
            reference_type=meta.reference_type,
        )

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
