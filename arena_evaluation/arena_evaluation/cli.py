import argparse
import contextlib
import os
import pathlib
import sys

from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.storage.data_root import latest_benchmark
from arena_evaluation.storage.folder_manager import FolderManager

_PATH_ARG_SUBDIRS: dict[str, tuple[str, ...]] = {
    "run_dir": ("recordings", "recording"),
    "benchmark_dir": ("benchmarks", "benchmark"),
}


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    search_roots = []

    arena_data_dir_env = os.environ.get("ARENA_DATA_DIR")
    if arena_data_dir_env:
        search_roots.append(pathlib.Path(arena_data_dir_env))

    cwd = pathlib.Path.cwd()
    search_roots.append(cwd / "data")

    for parent in cwd.parents:
        data_candidate = parent / "data"
        if data_candidate.is_dir():
            search_roots.append(data_candidate)

    try:
        from ament_index_python.packages import get_package_share_directory

        pkg_data = (
            pathlib.Path(get_package_share_directory("arena_evaluation")) / "data"
        )
        if pkg_data.is_dir():
            search_roots.append(pkg_data)
    except Exception:
        pass

    unique_search_roots = []
    seen = set()
    for root in search_roots:
        resolved_root = root.resolve()
        if resolved_root not in seen and resolved_root.is_dir():
            seen.add(resolved_root)
            unique_search_roots.append(resolved_root)

    def _resolve_single_path(p: pathlib.Path, subdirs: tuple[str, ...]) -> pathlib.Path:
        if p.exists():
            return p.resolve()
        for root in unique_search_roots:
            for sub in subdirs:
                candidate = root / sub / p
                if candidate.is_dir():
                    return candidate.resolve()
        return p

    for dest, value in list(vars(args).items()):
        subdirs = _PATH_ARG_SUBDIRS.get(dest)
        if subdirs is None or value is None:
            continue
        if isinstance(value, list):
            setattr(args, dest, [_resolve_single_path(p, subdirs) for p in value])
        else:
            setattr(args, dest, _resolve_single_path(value, subdirs))

    return args


def _migrate_targets(run_dir: pathlib.Path) -> list[pathlib.Path]:
    """The episode folders of a run root, or the folder itself when it already is one episode."""
    for container in (run_dir / "episodes", run_dir):
        episodes = sorted(path for path in container.glob("episode_*") if path.is_dir())
        if episodes:
            return episodes
    return [run_dir]


def _validate_config(path: pathlib.Path, kind: str) -> tuple[str, tuple[str, ...]]:
    """Parse one suite or contest bundle; return its name and the suite stages without an explicit seed."""
    import yaml

    from arena_evaluation.benchmark.config import Contest, Suite

    config_path = path / f"{kind}.yaml" if path.is_dir() else path
    if not config_path.is_file():
        raise ValueError(f"{path}: no {kind}.yaml")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = (
        config_path.parent.name
        if config_path.name == f"{kind}.yaml"
        else config_path.stem
    )
    if kind == "contest":
        return Contest.parse(name, data).name, ()
    if not isinstance(data, dict) or not isinstance(data.get("stages"), list):
        raise ValueError(f"{config_path}: a suite needs a mapping with a stages list")
    if not all(isinstance(stage, dict) for stage in data["stages"]):
        raise ValueError(f"{config_path}: every suite stage must be a mapping")
    suite = Suite.parse(name, data)
    return suite.name, tuple(stage.name for stage in suite.stages if stage.seed is None)


def _main_impl(argv: list[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(
        description="Arena Evaluation Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single episodes (any episode_NNN directory of a benchmark):
  evaluation process --run-dir /opt/arena_ws/data/benchmarks/my_benchmark/episodes/episode_004

  # Process all runs in a benchmark:
  evaluation process --benchmark-dir /opt/arena_ws/data/my_benchmark

  # Full pipeline (process + report) for a benchmark:
  evaluation run --benchmark-dir /opt/arena_ws/data/my_benchmark

  # Generate report from multiple already-processed benchmarks:
  evaluation report --benchmark-dir /opt/arena_ws/data/bench1 /opt/arena_ws/data/bench2 --output-dir ./merged_report

  # With no input dir, operate on the most recent benchmark under $ARENA_DATA_DIR/benchmarks:
  evaluation run

  # Migrate a run recorded before the native layout, one episode folder at a time:
  evaluation migrate /opt/arena_ws/data/benchmarks/my_benchmark

  # Check a suite or contest definition without running it:
  evaluation validate --suite configs/suites/basic
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser.set_defaults(run_dir=None, list_manifests=False)

    run_parent = argparse.ArgumentParser(add_help=False)
    run_group = run_parent.add_mutually_exclusive_group(required=False)
    run_group.add_argument(
        "--benchmark-dir",
        type=pathlib.Path,
        nargs="+",
        metavar="DIR",
        help="Path to one or more benchmark root directories (contains multiple planner/stage runs)",
    )
    run_group.add_argument(
        "--run-dir",
        type=pathlib.Path,
        nargs="+",
        metavar="DIR",
        help="Path to one or more episode directories (episode_NNN with episode_NNN.yaml and the MCAP). process and extract handle each alone, report and plot merge their Star Schema tables",
    )
    run_parent.add_argument(
        "--output-dir",
        type=pathlib.Path,
        metavar="DIR",
        help="Optional path to output directory for reports and plots. Defaults to the first input directory.",
    )
    run_parent.add_argument(
        "--profile",
        action="store_true",
        help="Enable resource profiling. Writes pipeline_profile.yaml with per-phase CPU, GPU, RAM, and duration stats.",
    )
    run_parent.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of worker processes for parallel extraction and processing (-1 = auto-detect CPU count).",
    )
    run_parent.add_argument(
        "--report-manifest",
        type=str,
        default=None,
        metavar="NAME|PATH|{...}",
        help="Report manifest: a name from configs/reports/, a path to a YAML file, or inline {...} YAML. Used by run/report/plot; ignored otherwise.",
    )
    run_parent.add_argument(
        "--list-manifests",
        action="store_true",
        help="List the available named report manifests and exit.",
    )
    run_parent.set_defaults(force_extract=False)

    subparsers.add_parser(
        "extract",
        parents=[run_parent],
        help="Layer 1: Extract topics from MCAP into fast Parquet files (cache).",
    )
    subparsers.add_parser(
        "run",
        parents=[run_parent],
        help="Full pipeline: Extract MCAP (overwrite) -> Process -> Parquet -> HTML report.",
    )
    process_parser = subparsers.add_parser(
        "process",
        parents=[run_parent],
        help="Layer 2: Compute metrics and write the Star Schema tables (uses cached extraction by default).",
    )
    process_parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of MCAP files, overwriting the topic cache.",
    )
    subparsers.add_parser(
        "report",
        parents=[run_parent],
        help="Layer 5: Generate an interactive HTML report from the existing Star Schema tables.",
    )
    subparsers.add_parser(
        "plot",
        parents=[run_parent],
        help="Layer 5: Generate static PNG plots only (no HTML report).",
    )

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Convert a run recorded before the native layout (old metadata and MCAP) to the current layout, in place.",
    )
    migrate_parser.add_argument(
        "run_dir",
        type=pathlib.Path,
        help="Run directory, or one episode folder, to migrate. A run directory is migrated episode by episode.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Check a suite or contest definition without running it.",
    )
    validate_group = validate_parser.add_mutually_exclusive_group(required=True)
    validate_group.add_argument(
        "--suite",
        type=pathlib.Path,
        metavar="FILE",
        help="Suite bundle directory or suite.yaml to validate.",
    )
    validate_group.add_argument(
        "--contest",
        type=pathlib.Path,
        metavar="FILE",
        help="Contest bundle directory or contest.yaml to validate.",
    )

    from arena_evaluation.cli_acoustic import add_acoustic_commands

    add_acoustic_commands(subparsers)

    args = parser.parse_args(argv)

    if args.list_manifests:
        from arena_evaluation.presentation.manifest_registry import available_manifests

        print("Available report manifests:")
        for name in available_manifests():
            print(f"  - {name}")
        return 0

    args = resolve_paths(args)

    if args.command == "validate":
        kind = "suite" if args.suite is not None else "contest"
        config_path = args.suite if args.suite is not None else args.contest
        try:
            name, seedless = _validate_config(config_path, kind)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        if seedless:
            print(
                f"warning: {len(seedless)} stage(s) without an explicit seed: {', '.join(seedless)}",
                file=sys.stderr,
            )
        print(f"valid {kind}: {name}")
        return 0

    if args.command == "migrate":
        from arena_evaluation.ingestion.migrate import MigrationError, migrate_run

        migrated: list[pathlib.Path] = []
        try:
            for target in _migrate_targets(args.run_dir):
                migrated.extend(migrate_run(target))
        except MigrationError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        print(f"Migrated {len(migrated)} episode folder(s) to the native layout")
        return 0

    # acoustic subcommand handler
    if args.command == "acoustic":
        from arena_evaluation.cli_acoustic import handle_acoustic

        handle_acoustic(args)
        return 0

    if args.benchmark_dir is None and args.run_dir is None:
        latest = latest_benchmark()
        if latest is None:
            print("Error: No input directory given and no benchmark runs found.")
            sys.exit(1)
        print(f"Using latest benchmark: {latest}")
        args.benchmark_dir = [latest]

    target_dirs = args.benchmark_dir or args.run_dir
    assert target_dirs is not None
    if isinstance(target_dirs, pathlib.Path):
        target_dirs = [target_dirs]

    for d in target_dirs:
        if not d.exists() or not d.is_dir():
            print(f"Error: directory does not exist: {d}")
            sys.exit(1)

    profiler = None
    if args.profile:
        from arena_evaluation.benchmark.profiler import PipelineProfiler as _PP

        output_dir = args.output_dir or target_dirs[0]
        profiler = _PP(output_dir=output_dir, sample_hz=2.0)

    failed: list[pathlib.Path] = []
    if args.command in ("extract", "run", "process"):
        force_extract = args.force_extract
        if args.command == "run":
            force_extract = True

        if args.run_dir:
            for run_dir in args.run_dir:
                fm = FolderManager(data_root=run_dir.parent)
                pipeline = ProcessingPipeline(
                    fm, profiler=profiler, workers=args.workers
                )

                if args.command == "extract":
                    print(f"Extracting single run: {run_dir}")
                    pipeline.extract_run_dir(run_dir)
                else:
                    print(f"Processing single run: {run_dir}")
                    out = pipeline.process_run_dir(run_dir, force_extract=force_extract)
                    if out:
                        print(f"Star Schema tables written to: {out}")
                    else:
                        print(f"Processing failed for {run_dir} - see errors above.")
                        failed.append(run_dir)

        else:
            for benchmark_dir in target_dirs:
                fm = FolderManager(data_root=benchmark_dir.parent)
                pipeline = ProcessingPipeline(
                    fm, profiler=profiler, workers=args.workers
                )

                if args.command == "extract":
                    print(f"Extracting benchmark: {benchmark_dir.name}")
                    pipeline.extract_benchmark(benchmark_dir.name)
                else:
                    print(f"Processing benchmark: {benchmark_dir.name}")
                    if (
                        pipeline.process_benchmark(
                            benchmark_dir.name, force_extract=force_extract
                        )
                        is None
                    ):
                        print(
                            f"Processing failed for {benchmark_dir} - no Star Schema tables written."
                        )
                        failed.append(benchmark_dir)

    if args.command in ("run", "report", "plot"):
        output_dir = args.output_dir
        if not output_dir:
            output_dir = target_dirs[0]

        from arena_evaluation.presentation.manifest_registry import resolve_manifest
        from arena_evaluation.presentation.report_builder import ReportBuilder

        manifest_obj = resolve_manifest(
            args.report_manifest, benchmark_dir=target_dirs[0]
        )

        print(
            f"Building report/plots from {len(target_dirs)} sources into: {output_dir}"
        )
        _ctx = profiler.phase("report") if profiler else contextlib.nullcontext()
        with _ctx:
            builder = ReportBuilder.from_dirs(
                target_dirs,
                output_dir=output_dir,
                manifest=manifest_obj,
            )
            builder.build()
        print("Report generation complete.")

    if profiler is not None:
        profiler.write_summary()
    return 1 if failed else 0


def main():
    try:
        sys.exit(_main_impl() or 0)
    except KeyboardInterrupt:
        print("\n\nEvaluation cancelled by user (Ctrl+C). Exiting.\n", flush=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
