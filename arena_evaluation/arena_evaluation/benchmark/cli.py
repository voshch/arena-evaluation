from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time


from arena_evaluation.storage.data_root import benchmarks_root


def _data_root() -> pathlib.Path:
    return benchmarks_root()


def _resolve_run(data_root: pathlib.Path, run_id: str | None) -> pathlib.Path:
    if run_id:
        path = data_root / run_id
        if not path.is_dir():
            raise SystemExit(f"no run found at {path}")
        return path
    if not data_root.is_dir():
        raise SystemExit(f"data root does not exist: {data_root}")
    runs = sorted([p for p in data_root.iterdir() if p.is_dir()], reverse=True)
    if not runs:
        raise SystemExit(f"no benchmark runs in {data_root}")
    return runs[0]


def _count_by_status(steps: dict) -> dict[str, int]:
    counts: dict[str, int] = {
        "ok": 0,
        "partial": 0,
        "failed": 0,
        "skipped": 0,
        "in_progress": 0,
    }
    for step in steps.values():
        s = step.status
        if s in counts:
            counts[s] += 1
    return counts


def _cmd_list(args: argparse.Namespace) -> int:
    from .state import Manifest, StateFile

    data_root = pathlib.Path(args.data_root) if args.data_root else _data_root()

    if not data_root.is_dir():
        print(f"no benchmark runs in {data_root}")
        return 0

    runs = sorted([p for p in data_root.iterdir() if p.is_dir()], reverse=True)
    if not runs:
        print(f"no benchmark runs in {data_root}")
        return 0

    rows: list[tuple[str, str, str, int, int, int, int, int, int, str]] = []
    for run_path in runs:
        manifest_path = run_path / "manifest.yaml"
        if not manifest_path.exists():
            continue
        try:
            manifest = Manifest.from_yaml(manifest_path.read_text())
        except Exception:
            continue
        state = StateFile.open(run_path)
        counts = _count_by_status(state.steps)
        total = len(manifest.steps)
        created = manifest.created_at[:16].replace("T", " ") if manifest.created_at else ""
        rows.append((
            manifest.run_id,
            manifest.suite_name,
            manifest.contest_name,
            total,
            counts["ok"],
            counts["partial"],
            counts["failed"],
            counts["skipped"],
            counts["in_progress"],
            created,
        ))

    if not rows:
        print(f"no benchmark runs in {data_root}")
        return 0

    col_widths = [
        max(len("RUN_ID"), max(len(r[0]) for r in rows)),
        max(len("SUITE"), max(len(r[1]) for r in rows)),
        max(len("CONTEST"), max(len(r[2]) for r in rows)),
        len("STEPS"),
        len("OK"),
        len("PARTIAL"),
        len("FAILED"),
        len("SKIPPED"),
        len("IN_FLIGHT"),
        len("CREATED"),
    ]

    def _row(
        run_id: str,
        suite: str,
        contest: str,
        total: int | str,
        ok: int | str,
        partial: int | str,
        failed: int | str,
        skipped: int | str,
        in_flight: int | str,
        created: str,
    ) -> str:
        return (
            f"{str(run_id):<{col_widths[0]}}  "
            f"{str(suite):<{col_widths[1]}}  "
            f"{str(contest):<{col_widths[2]}}  "
            f"{str(total):>{col_widths[3]}}  "
            f"{str(ok):>{col_widths[4]}}  "
            f"{str(partial):>{col_widths[5]}}  "
            f"{str(failed):>{col_widths[6]}}  "
            f"{str(skipped):>{col_widths[7]}}  "
            f"{str(in_flight):>{col_widths[8]}}  "
            f"{str(created)}"
        )

    print(_row("RUN_ID", "SUITE", "CONTEST", "STEPS", "OK", "PARTIAL", "FAILED", "SKIPPED", "IN_FLIGHT", "CREATED"))
    for r in rows:
        print(_row(*r))

    return 0


def _format_status_block(
    run_id: str,
    suite: str,
    contest: str,
    simulator: str,
    env_n: int,
    headless: bool,
    created_at: str,
    steps_total: int,
    ok: int,
    partial: int,
    failed: int,
    skipped: int,
    in_flight: int,
    active: list[tuple[str, str | None]],
    failed_steps: list[tuple[str, str | None, str | None]],
) -> str:
    pending = steps_total - ok - partial - failed - skipped - in_flight
    lines = [
        f"run: {run_id}",
        f"suite/contest: {suite}/{contest}",
        f"simulator: {simulator}    env_n: {env_n}    headless: {headless}",
        f"created: {created_at}",
        "",
        f"steps: {steps_total}    ok: {ok}  partial: {partial}  failed: {failed}  skipped: {skipped}  in_flight: {in_flight}  pending: {pending}",
    ]
    if active:
        lines.append("")
        lines.append("active:")
        for key, started in active:
            if started is not None:
                lines.append(f"  {key} (started {started})")
            else:
                lines.append(f"  {key}")
    if failed_steps:
        lines.append("")
        lines.append("failed:")
        for key, kind, detail in failed_steps:
            lines.append(f"  {key}: {kind or 'unknown'}: {detail or ''}")
    return "\n".join(lines)


def _ago(ts: float | None) -> str | None:
    if ts is None:
        return None
    delta = int(time.time() - ts)
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    return f"{delta // 3600}h ago"


def _status_from_disk(data_root: pathlib.Path, run_id: str | None) -> str:
    from .state import Manifest, StateFile

    run_path = _resolve_run(data_root, run_id)
    manifest_path = run_path / "manifest.yaml"
    manifest = Manifest.from_yaml(manifest_path.read_text())
    state = StateFile.open(run_path)
    counts = _count_by_status(state.steps)

    active: list[tuple[str, str | None]] = []
    failed_steps: list[tuple[str, str | None, str | None]] = []
    for step in state.steps.values():
        if step.status == "in_progress":
            active.append((step.key, _ago(step.started_at)))
        elif step.status == "failed":
            kind = step.error_kind.value if step.error_kind is not None else None
            failed_steps.append((step.key, kind, step.error_detail))

    return _format_status_block(
        run_id=manifest.run_id,
        suite=manifest.suite_name,
        contest=manifest.contest_name,
        simulator=manifest.simulator or "",
        env_n=manifest.env_n,
        headless=manifest.headless,
        created_at=manifest.created_at,
        steps_total=len(manifest.steps),
        ok=counts["ok"],
        partial=counts["partial"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        in_flight=counts["in_progress"],
        active=active,
        failed_steps=failed_steps,
    )


def _cmd_status(args: argparse.Namespace) -> int:
    data_root = pathlib.Path(args.data_root) if args.data_root else _data_root()
    run_id: str | None = args.run_id

    if not args.watch:
        print(_status_from_disk(data_root, run_id))
        return 0

    import rclpy
    from arena_evaluation_msgs.msg import BenchmarkState
    from arena_rclpy_mixins import ArenaMixinNode, run_main

    class _WatchNode(ArenaMixinNode):
        def __init__(self) -> None:
            super().__init__("evaluation_cli_watch")
            self.create_subscription(
                BenchmarkState,
                "/arena/benchmark/state",
                self._on_state,
                rclpy.qos.QoSProfile(
                    depth=1,
                    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                ),
            )

        def _on_state(self, msg: BenchmarkState) -> None:
            active = [(k, None) for k in msg.active_keys]
            block = _format_status_block(
                run_id=msg.run_id,
                suite=msg.suite,
                contest=msg.contest,
                simulator=msg.simulator,
                env_n=msg.env_n,
                headless=msg.headless,
                created_at="",
                steps_total=msg.steps_total,
                ok=msg.steps_done,
                partial=msg.steps_partial,
                failed=msg.steps_failed,
                skipped=msg.steps_skipped,
                in_flight=msg.steps_in_flight,
                active=active,
                failed_steps=[],
            )
            print("\033[2J\033[H", end="")
            print(block)

    try:
        run_main(_WatchNode)
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_tail(args: argparse.Namespace) -> int:
    data_root = pathlib.Path(args.data_root) if args.data_root else _data_root()
    run_path = _resolve_run(data_root, args.run_id)
    csv_path = run_path / "progress.csv"

    while not csv_path.exists():
        print(f"progress.csv not yet created at {csv_path}; waiting...")
        time.sleep(1)

    try:
        subprocess.run(["tail", "-n", "50", "-F", str(csv_path)], check=False)
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_ps(args: argparse.Namespace) -> int:
    """List arena-related processes (benchmark runner, sim, nodes)."""
    from .debug import running_processes

    rows = running_processes()
    if not rows:
        print("no arena processes running")
        return 0

    def _elapsed(secs: float | None) -> str:
        if secs is None:
            return "?"
        secs = int(secs)
        if secs < 60:
            return f"{secs}s"
        if secs < 3600:
            return f"{secs // 60}m{secs % 60:02d}s"
        return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"

    w_pid = max(len("PID"), max(len(str(r["pid"])) for r in rows))
    w_kind = max(len("KIND"), max(len(r["kind"]) for r in rows))
    w_el = max(len("ELAPSED"), max(len(_elapsed(r["elapsed_s"])) for r in rows))

    print(f"{'PID':<{w_pid}}  {'KIND':<{w_kind}}  {'ELAPSED':<{w_el}}  CMD")
    for r in rows:
        print(
            f"{r['pid']:<{w_pid}}  {r['kind']:<{w_kind}}  "
            f"{_elapsed(r['elapsed_s']):<{w_el}}  {r['command']}"
        )
    return 0


def _cmd_console(args: argparse.Namespace) -> int:
    """Tail a benchmark run's console log."""
    from .debug import console_log_path, tail_console

    run_id = args.run_id
    if not run_id:
        # Resolve the run id from the most recent run directory
        run_path = _resolve_run(_data_root(), None)
        manifest_path = run_path / "manifest.yaml"
        from .state import Manifest

        run_id = Manifest.from_yaml(manifest_path.read_text()).run_id

    if args.follow:
        seen = 0
        printed_header = False
        while True:
            res = tail_console(run_id, lines=0)  # 0 = all lines
            if not printed_header:
                print(f"console log: {res['path']}")
                printed_header = True
            if res["exists"]:
                for ln in res["lines"][seen:]:
                    print(ln)
                seen = len(res["lines"])
            if not res["alive"]:
                # Runner gone; flush anything new once more, then stop
                if res["exists"]:
                    for ln in res["lines"][seen:]:
                        print(ln)
                print("runner exited")
                return 0
            time.sleep(1)

    res = tail_console(run_id, lines=args.lines)
    if not res["exists"]:
        print(f"no console log for run '{run_id}' (expected at {res['path']})")
        print("hint: the benchmark writes runner.log as soon as it creates the run directory")
        return 1
    state = "running" if res["alive"] else "finished"
    print(f"run: {run_id}  [{state}]  pid: {res['pid']}  log: {res['path']}")
    if res.get("truncated"):
        print(f"(last {args.lines} of {args.lines} lines, use --lines to see more)")
    for ln in res["lines"]:
        print(ln)
    return 0


def _cmd_kill(args: argparse.Namespace) -> int:
    from arena_evaluation.benchmark.debug import kill_processes

    results = kill_processes(pids=args.pids or None, force=args.force, kind=args.kind)
    if not results:
        if args.pids:
            print("no matching processes found for specified PIDs")
        else:
            print("no running arena processes found")
        return 0

    for r in results:
        cmd_short = r.get("command", "")
        if len(cmd_short) > 60:
            cmd_short = cmd_short[:57] + "..."
        print(f"[{r['status']}] pid {r['pid']} ({r['kind']}) - {cmd_short}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation_cli", epilog="benchmark exit codes: 0 ok, 1 systemic abort, 2 config error or crash, 3 lockstep, strict or efficacy verdict, 4 runner hung (deadman), 130 interrupted")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list benchmark runs")
    p_list.add_argument("--data-root", default=None, metavar="PATH")

    p_status = sub.add_parser("status", help="show run status")
    p_status.add_argument("--data-root", default=None, metavar="PATH")
    p_status.add_argument("--watch", action="store_true", help="subscribe to live topic")
    p_status.add_argument("run_id", nargs="?", default=None)

    p_tail = sub.add_parser("tail", help="tail progress.csv of a run")
    p_tail.add_argument("--data-root", default=None, metavar="PATH")
    p_tail.add_argument("run_id", nargs="?", default=None)

    p_ps = sub.add_parser("ps", help="list running arena processes (benchmark runner, sim, nodes)")
    p_ps.add_argument("--data-root", default=None, metavar="PATH")

    p_kill = sub.add_parser("kill", help="terminate running arena benchmark and simulation processes")
    p_kill.add_argument("pids", nargs="*", type=int, default=None, help="specific PIDs to kill (default: all arena processes)")
    p_kill.add_argument("-9", "--force", action="store_true", help="send SIGKILL immediately")
    p_kill.add_argument("--kind", choices=["benchmark_runner", "simulation", "arena_node", "world_generator", "arena_cli"], default=None, help="filter by process kind")

    p_console = sub.add_parser("console", help="tail a benchmark run's console log")
    p_console.add_argument("--data-root", default=None, metavar="PATH")
    p_console.add_argument("--lines", type=int, default=200, help="tail lines (default 200)")
    p_console.add_argument("--follow", action="store_true", help="follow new output until the runner exits")
    p_console.add_argument("run_id", nargs="?", default=None)

    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            return _cmd_list(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "tail":
            return _cmd_tail(args)
        if args.command == "ps":
            return _cmd_ps(args)
        if args.command == "kill":
            return _cmd_kill(args)
        if args.command == "console":
            return _cmd_console(args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Exiting.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
