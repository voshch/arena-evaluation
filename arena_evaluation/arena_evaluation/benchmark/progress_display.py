from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
import typing

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


class BenchmarkProgressDisplay:
    """Live interactive terminal progress display for Arena Benchmark Runner."""

    def __init__(
        self,
        title: str,
        total_steps: int,
        env_n: int,
        run_id: str,
        sim_now: typing.Callable[[], float] | None = None,
    ):
        self.title = title
        self.sim_now = sim_now
        self.total_steps = total_steps
        self.env_n = env_n
        self.run_id = run_id
        self.completed_steps = 0
        self.ok_steps = 0
        self.failed_steps = 0
        self.partial_steps = 0
        self.skipped_steps = 0

        self.console = Console() if _HAS_RICH else None
        self.live: Live | None = None
        self._running = False
        self._lock = threading.RLock()
        self.start_time = time.perf_counter()

        self.active_slots: dict[int, dict[str, typing.Any]] = {}

    def __enter__(self) -> typing.Self:
        if _HAS_RICH and sys.stdout.isatty():
            self.console.print(f"[bold cyan]{self.title}[/bold cyan] [dim](run_id: {self.run_id}, env_n: {self.env_n})[/dim]")
            self.live = Live(
                get_renderable=self._render,
                console=self.console,
                refresh_per_second=4,
                transient=True,
            )
            self.live.__enter__()
            self._running = True
        else:
            print(
                f"{self.title} [run_id: {self.run_id}] ({self.total_steps} steps, {self.env_n} envs)...",
                flush=True,
            )
        return self

    def stop(self):
        with self._lock:
            self._running = False
            if self.live is not None:
                try:
                    self.live.stop()
                except Exception:
                    pass
                self.live = None
            if self.console is not None:
                try:
                    self.console.show_cursor(True)
                except Exception:
                    pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self.live is not None:
            try:
                self.live.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass
            self.live = None
        self.stop()
        if self.console is not None:
            elapsed = time.perf_counter() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            if exc_type is not None and issubclass(exc_type, (asyncio.CancelledError, KeyboardInterrupt)):
                self.console.print(f"\n[bold yellow]⚠ Benchmark interrupted after {mins:02d}:{secs:02d}[/bold yellow] (completed: {self.completed_steps}/{self.total_steps}, ok: [green]{self.ok_steps}[/green], failed: [red]{self.failed_steps}[/red], partial: [yellow]{self.partial_steps}[/yellow])")
            else:
                self.console.print(f"\n[bold green]✔ Benchmark completed in {mins:02d}:{secs:02d}[/bold green] (ok: [green]{self.ok_steps}[/green], failed: [red]{self.failed_steps}[/red], partial: [yellow]{self.partial_steps}[/yellow], skipped: [dim]{self.skipped_steps}[/dim])")

    def update_slot(
        self,
        slot_index: int,
        env_id: int | None,
        contestant: str,
        stage: str,
        step_key: str,
        ep_idx: int,
        ep_total: int,
        state: str = "RUNNING",
        sim_start: float | None = None,
    ):
        with self._lock:
            self.active_slots[slot_index] = {
                "env_id": env_id,
                "contestant": contestant,
                "stage": stage,
                "step_key": step_key,
                "ep_idx": ep_idx,
                "ep_total": ep_total,
                "state": state,
                "start_time": time.perf_counter(),
                "sim_start": sim_start,
            }

    def update_slot_state(self, slot_index: int, state: str, ep_idx: int | None = None):
        with self._lock:
            slot = self.active_slots.get(slot_index)
            if slot is not None:
                slot["state"] = state
                if ep_idx is not None:
                    slot["ep_idx"] = ep_idx

    def clear_slot(self, slot_index: int):
        with self._lock:
            self.active_slots.pop(slot_index, None)

    def log_step_completed(
        self,
        step_key: str,
        status: str,
        contestant: str,
        stage: str,
        episodes_run: int,
        episodes_total: int,
        episodes_failed: int,
        elapsed_sec: float,
        error_detail: str | None = None,
        sim_sec: float | None = None,
    ):
        with self._lock:
            self.completed_steps += 1
            if status == "ok":
                self.ok_steps += 1
                icon = "[bold green]✔[/bold green]"
                status_colored = f"[green]ok ({episodes_run}/{episodes_total} eps)[/green]"
            elif status == "partial":
                self.partial_steps += 1
                icon = "[bold yellow]⚠[/bold yellow]"
                status_colored = f"[yellow]partial ({episodes_run - episodes_failed}/{episodes_total} eps ok)[/yellow]"
            elif status == "skipped":
                self.skipped_steps += 1
                icon = "[dim]⊘[/dim]"
                status_colored = "[dim]skipped[/dim]"
            else:
                self.failed_steps += 1
                icon = "[bold red]✘[/bold red]"
                status_colored = f"[red]failed ({episodes_failed}/{episodes_total} eps failed)[/red]"

            timing = f"in {elapsed_sec:.1f}s wall" + (f", {sim_sec:.1f}s sim" if sim_sec is not None else "")
            line = f"{icon} [{self.completed_steps:2d}/{self.total_steps:2d}] [bold]{contestant}[/bold] • [cyan]{stage}[/cyan] [{status_colored}] {timing}"
            if error_detail and status == "failed":
                line += f" • [dim red]{error_detail}[/dim red]"
            has_live = self.live is not None

        if has_live:
            self.live.console.print(line)
            self.live.refresh()
        else:
            print(
                f"[{self.completed_steps}/{self.total_steps}] {step_key} ({status}) {timing}",
                flush=True,
            )

    def _render(self) -> Table:
        with self._lock:
            completed = self.completed_steps
            total = self.total_steps
            ok = self.ok_steps
            failed = self.failed_steps
            partial = self.partial_steps
            skipped = self.skipped_steps
            slots_snapshot = {k: dict(v) for k, v in self.active_slots.items()}

        table = Table.grid(padding=(0, 1))

        # Overall Progress Bar
        pct = (completed / total * 100) if total > 0 else 0
        filled = int(30 * (completed / total)) if total > 0 else 0
        bar = "█" * filled + "░" * (30 - filled)
        elapsed = time.perf_counter() - self.start_time
        mins, secs = divmod(int(elapsed), 60)

        table.add_row(f"[{bar}] [bold green]{completed}/{total}[/bold green] steps ({pct:.0f}%) | ok: [green]{ok}[/green] fail: [red]{failed}[/red] part: [yellow]{partial}[/yellow] | Elapsed: [yellow]{mins:02d}:{secs:02d}[/yellow]")
        table.add_row("")

        # Active Envs Table
        if slots_snapshot:
            worker_table = Table(show_header=True, header_style="bold blue", box=None, padding=(0, 2), expand=False)
            worker_table.add_column("Env", style="cyan", no_wrap=True)
            worker_table.add_column("Contestant / Stage", style="white", no_wrap=True, overflow="ellipsis", max_width=60)
            worker_table.add_column("State", style="yellow", no_wrap=True)
            worker_table.add_column("Episode", justify="right", style="magenta", no_wrap=True)
            worker_table.add_column("Sim / Wall", justify="right", style="green", no_wrap=True)

            now = time.perf_counter()
            sim_now = self.sim_now() if self.sim_now is not None else None
            for slot_idx in sorted(slots_snapshot.keys()):
                info = slots_snapshot[slot_idx]
                env_label = f"env_{info['env_id']}" if info.get("env_id") is not None else "-"
                ep_str = f"({info['ep_idx'] + 1}/{info['ep_total']})" if info.get("ep_total") else "-"
                w_elapsed = max(now - info.get("start_time", now), 0.0)
                sim_start = info.get("sim_start")
                sim_str = f"{max(sim_now - sim_start, 0.0):.1f}s" if sim_now is not None and sim_start is not None else "-"
                worker_table.add_row(
                    env_label,
                    f"{info.get('contestant', '')} / {info.get('stage', '')}",
                    str(info.get("state", "RUNNING")),
                    ep_str,
                    f"{sim_str} / {w_elapsed:.1f}s",
                )
            table.add_row(worker_table)
        else:
            table.add_row("[dim]Spawning environments...[/dim]")

        return table
