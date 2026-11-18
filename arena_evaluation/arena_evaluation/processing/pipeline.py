from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import multiprocessing
import os
import pathlib
import shutil
import time
import typing


from arena_evaluation.benchmark.profiler import PipelineProfiler
from arena_evaluation.processing.mcap_reader import MCAPReader, load_bundles
from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.dimension_rows import episode_dimensions
from arena_evaluation.storage.contracts import enforce_dimension_contract
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.star_schema import stack, write_benchmark_dimensions
from arena_evaluation.storage.schemas import (
    EpisodeDescriptor,
    TopicBundle,
)

_log = logging.getLogger(__name__)


def _worker_init():
    import os
    import signal

    os.environ["POLARS_MAX_THREADS"] = "1"
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

def _shutdown_executor_cleanly(executor: concurrent.futures.ProcessPoolExecutor):
    """Force terminate all child worker processes without blocking on wait=True."""
    try:
        processes = list(executor._processes.values())
        for proc in processes:
            try:
                proc.kill()
            except Exception:
                pass
    except Exception:
        pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass

def _extract_worker(
    data_root_str: str,
    ep: EpisodeDescriptor,
    force_extract: bool,
    status_dict: typing.MutableMapping[int, tuple[str, str, str, int, int, float]] | None = None,
) -> int:
    import pathlib
    import time

    from arena_evaluation.processing.pipeline import ProcessingPipeline
    from arena_evaluation.storage.folder_manager import FolderManager

    if status_dict is not None:
        try:
            status_dict[ep.episode_id] = (ep.planner, ep.stage, "Extracting MCAP / Topics", 1, 1, time.perf_counter())
        except Exception:
            pass
    fm = FolderManager(data_root=pathlib.Path(data_root_str))
    pipeline = ProcessingPipeline(fm, profiler=None, workers=1)
    pipeline.extract_episode(ep, force_extract=force_extract)
    if status_dict is not None:
        try:
            status_dict.pop(ep.episode_id, None)
        except Exception:
            pass
    return ep.episode_id


class ProcessingPipeline:
    """
    Orchestrates the data processing pipeline:
    MCAP -> Extract -> Align -> Split -> Metrics -> Parquet
    """

    def __init__(self, folder_manager: FolderManager, profiler: PipelineProfiler | None = None, workers: int | None = None):
        self.folder_manager = folder_manager
        self.profiler = profiler
        # -1 (or None) = auto-detect the CPU count; any value < 1 falls back to it.
        self.workers = (os.cpu_count() or 1) if (workers is None or workers < 1) else workers

    def extract_episode(self, ep: EpisodeDescriptor, force_extract: bool = False) -> dict[str, TopicBundle] | None:
        """Extract topics from a single episode MCAP into episode_dir/topics/."""
        episode_dir = pathlib.Path(ep.episode_dir)
        mcap_path = self.folder_manager.mcap_path_for_episode(episode_dir)
        topics_dir = self.folder_manager.extracted_topics_path_for_episode(episode_dir)

        if not mcap_path.exists():
            _log.warning(f"episode_{ep.episode_id:03d} ({ep.planner}/{ep.stage}): no MCAP found, so it is not extracted")
            return None

        if not force_extract and TopicParquetStore.is_current(topics_dir):
            bundles = TopicParquetStore.read(topics_dir)
            if bundles:
                return bundles

        _ctx = self.profiler.phase("extract") if self.profiler else contextlib.nullcontext()
        with _ctx:
            if topics_dir.exists():
                shutil.rmtree(topics_dir)
            bundles = MCAPReader(mcap_path).read(topics_dir)
            TopicParquetStore.mark_complete(topics_dir)
            return bundles

    def _describe_run_dir(self, run_dir: pathlib.Path) -> EpisodeDescriptor:
        ep = self.folder_manager.describe_episode(pathlib.Path(run_dir))
        if ep is None:
            raise FileNotFoundError(f"{run_dir} is not an episode directory (episode_NNN with episode_NNN.yaml)")
        return ep

    def extract_run_dir(self, run_dir: pathlib.Path, force_extract: bool = False) -> dict[str, TopicBundle] | None:
        """Extract one `episode_NNN` directory into its topics/ folder."""
        return self.extract_episode(self._describe_run_dir(run_dir), force_extract=force_extract)

    def extract_benchmark(self, benchmark_id: str, force_extract: bool = False) -> None:
        """Extract all episodes of a benchmark, then write its dimensions once."""
        episodes = self.folder_manager.discover_episodes(benchmark_id)
        if not episodes:
            print(f"No episodes found for benchmark '{benchmark_id}'")
            return
        self._extract_episodes(benchmark_id, episodes, force_extract)
        self._write_benchmark_dimensions(benchmark_id, episodes)

    def _write_benchmark_dimensions(self, benchmark_id: str, episodes: list[EpisodeDescriptor]) -> None:
        """Stack the dimension rows of every extracted episode and write each dimension once at the benchmark root."""
        contributions = []
        for ep in episodes:
            episode_dir = pathlib.Path(ep.episode_dir)
            topics = episode_dir / "topics"
            if not topics.is_dir():
                _log.warning(f"episode_{ep.episode_id:03d}: no extracted topics, so its dimension rows are missing")
                continue
            metadata = MetadataWriter.read(episode_dir / f"{episode_dir.name}.yaml")
            bundles = load_bundles(topics, {})
            contributions.append(episode_dimensions(run_id=benchmark_id, episode_id=ep.episode_id, metadata=metadata, bundles=bundles))
        dimensions = stack(contributions)
        enforce_dimension_contract(dimensions)
        write_benchmark_dimensions(dimensions, self.folder_manager.benchmark_dir(benchmark_id))

    def _extract_episodes(self, benchmark_id: str, episodes: list[EpisodeDescriptor], force_extract: bool) -> None:
        from .progress_display import PipelineProgressDisplay

        data_root_str = str(self.folder_manager.data_root)
        if self.workers <= 1:
            local_status: dict[int, tuple[str, str, str, int, int, float]] = {}
            with PipelineProgressDisplay(
                f"Phase 1: Extracting MCAP topics for {benchmark_id}",
                len(episodes),
                1,
                status_dict=local_status,
            ) as display:
                for ep in episodes:
                    t_start = time.perf_counter()
                    try:
                        ep_id = _extract_worker(data_root_str, ep, force_extract, local_status)
                        elapsed = time.perf_counter() - t_start
                        display.log_completed(ep_id, f"{ep.planner}/{ep.stage}", elapsed)
                    except Exception as e:
                        display.log_error(ep.episode_id, str(e))
            return

        mp_ctx = multiprocessing.get_context("spawn")
        manager = mp_ctx.Manager()
        status_dict = manager.dict()

        with PipelineProgressDisplay(
            f"Phase 1: Extracting MCAP topics for {benchmark_id}",
            len(episodes),
            self.workers,
            status_dict=status_dict,
        ) as display:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp_ctx,
                initializer=_worker_init,
            )
            try:
                futures = {executor.submit(_extract_worker, data_root_str, ep, force_extract, status_dict): (ep, time.perf_counter()) for ep in episodes}
                for future in concurrent.futures.as_completed(futures):
                    ep, t_start = futures[future]
                    try:
                        ep_id = future.result()
                        elapsed = time.perf_counter() - t_start
                        display.log_completed(ep_id, f"{ep.planner}/{ep.stage}", elapsed)
                    except Exception as e:
                        display.log_error(ep.episode_id, str(e))
            except KeyboardInterrupt:
                _shutdown_executor_cleanly(executor)
                raise
            finally:
                _shutdown_executor_cleanly(executor)
