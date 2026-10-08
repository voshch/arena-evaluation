"""Per-step frames of evaluation recorder episodes, read through the evaluation pipeline's extraction and alignment."""

from __future__ import annotations

import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    import polars as pl

    from arena_evaluation.storage.schemas import EpisodeDescriptor, RunMetadata, TopicBundle


class RecordedEpisodes:
    """The episodes of one run directory, recorded from an Arena robot or imported."""

    def __init__(self, run: Path):
        from arena_evaluation.storage.folder_manager import FolderManager

        self.run = Path(run).resolve()
        self._folders = FolderManager(self.run.parent)
        self.episodes: dict[int, EpisodeDescriptor] = {episode.episode_id: episode for episode in self._folders.discover_episodes(self.run.name)}

    def __len__(self) -> int:
        return len(self.episodes)

    def episode(self, episode_id: int = 0) -> EpisodeDescriptor:
        if episode_id not in self.episodes:
            raise IndexError(f"no episode {episode_id} in {self.run}, have {sorted(self.episodes)}")
        return self.episodes[episode_id]

    def metadata(self, episode_id: int = 0) -> RunMetadata:
        from arena_evaluation.storage.manifest import MetadataWriter

        directory = Path(self.episode(episode_id).episode_dir)
        return MetadataWriter.read(directory / f"{directory.name}.yaml")

    def robots(self, episode_id: int = 0) -> dict[str, TopicBundle]:
        from arena_evaluation.processing.pipeline import ProcessingPipeline

        bundles = ProcessingPipeline(self._folders, workers=1).extract_episode(self.episode(episode_id)) or {}
        return {name: bundle for name, bundle in bundles.items() if bundle.odom is not None}

    def frames(self, episode_id: int = 0, robot: str | None = None) -> pl.DataFrame:
        """One row per odom sample of the episode, the other topics joined as the evaluation pipeline joins them."""
        from arena_evaluation.processing.episode_splitter import EpisodeSplitter
        from arena_evaluation.processing.topic_aligner import TopicAligner

        robots = self.robots(episode_id)
        if robot is None and len(robots) == 1:
            robot = next(iter(robots))
        if robot not in robots:
            raise KeyError(f"episode {episode_id} in {self.run} has robots {sorted(robots)}, pick one with robot=")
        recorded = self.metadata(episode_id).task_generator_episode_id
        spans = [span for span in EpisodeSplitter(TopicAligner()).split(robots[robot], robot) if recorded is None or span.episode_id == recorded]
        if not spans:
            raise ValueError(f"episode {episode_id} in {self.run} has no span of {robot} with enough odom samples")
        return spans[-1].data
