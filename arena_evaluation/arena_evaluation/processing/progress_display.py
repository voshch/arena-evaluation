from __future__ import annotations

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


class PipelineProgressDisplay:
    """Multi-worker live terminal progress display with active metric tracking."""

    def __init__(
        self,
        title: str,
        total_items: int,
        num_workers: int,
        status_dict: typing.MutableMapping[int, tuple[str, str, str, int, int, float]] | None = None,
    ):
        self.title = title
        self.total_items = total_items
        self.num_workers = num_workers
        self.status_dict = status_dict
        self.completed_items = 0
        self.console = Console() if _HAS_RICH else None
        self.live = None
        self._running = False
        self._lock = threading.Lock()
        self.start_time = time.perf_counter()
        self._monitor_thread = None

    def __enter__(self) -> typing.Self:
        # Only use rich Live if stdout is a real terminal
        if _HAS_RICH and sys.stdout.isatty():
            self.console.print(f"[bold cyan]{self.title}[/bold cyan]")
            self.live = Live(
                self._render(),
                console=self.console,
                refresh_per_second=10,
                transient=True,
            )
            self.live.__enter__()
            self._running = True

            if self.status_dict is not None:
                self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
                self._monitor_thread.start()
        else:
            print(
                f"{self.title} ({self.total_items} items, {self.num_workers} workers)...",
                flush=True,
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self._running = False
        if self.live is not None:
            self.live.__exit__(exc_type, exc_val, exc_tb)
            elapsed = time.perf_counter() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            self.console.print(f"[bold green]✔ All {self.total_items} episodes completed in {mins:02d}:{secs:02d}[/bold green]")

    def _monitor_loop(self):
        while self._running:
            time.sleep(0.1)
            if self.live is not None:
                try:
                    self.live.update(self._render())
                except Exception:
                    pass

    def log_completed(
        self,
        ep_id: int,
        label: str,
        elapsed_sec: float,
        extra: str = "",
    ):
        with self._lock:
            self.completed_items += 1
            line = f"[bold green]✔[/bold green] [{self.completed_items:2d}/{self.total_items:2d}] episode_{ep_id:03d} ({label}) completed in {elapsed_sec:.2f}s"
            if extra:
                line += f" • {extra}"

            if self.live is not None:
                self.console.print(line)
                self.live.update(self._render())
            else:
                print(
                    f"[{self.completed_items}/{self.total_items}] episode_{ep_id:03d} ({label}) in {elapsed_sec:.2f}s",
                    flush=True,
                )

    def log_error(self, ep_id: int, message: str):
        with self._lock:
            err_line = f"[bold red]✘[/bold red] episode_{ep_id:03d} error: {message}"
            if self.live is not None:
                self.console.print(err_line)
                self.live.update(self._render())
            else:
                print(f"Error in episode_{ep_id:03d}: {message}", flush=True)

    def _render(self) -> Table:
        table = Table.grid(padding=(0, 1))

        # Overall Progress Bar
        pct = (self.completed_items / self.total_items) * 100 if self.total_items > 0 else 0
        filled = int(30 * (self.completed_items / self.total_items)) if self.total_items > 0 else 0
        bar = "█" * filled + "░" * (30 - filled)
        elapsed = time.perf_counter() - self.start_time
        mins, secs = divmod(int(elapsed), 60)

        table.add_row(f"[{bar}] [bold green]{self.completed_items}/{self.total_items}[/bold green] episodes ({pct:.0f}%) | Elapsed: [yellow]{mins:02d}:{secs:02d}[/yellow]")
        table.add_row("")

        # Active Workers Table
        active_items = {}
        if self.status_dict is not None:
            try:
                active_items = dict(self.status_dict)
            except Exception:
                active_items = {}

        if active_items:
            worker_table = Table(show_header=True, header_style="bold blue", box=None, padding=(0, 2))
            worker_table.add_column("Episode", style="cyan", no_wrap=True)
            worker_table.add_column("Planner / Stage", style="white")
            worker_table.add_column("Active Step", style="yellow")
            worker_table.add_column("Progress", justify="right", style="magenta")
            worker_table.add_column("Time", justify="right", style="green")

            now = time.perf_counter()
            for ep_id in sorted(active_items.keys()):
                info = active_items[ep_id]
                # format: (planner, stage, task_name, idx, total, start_time)
                if isinstance(info, (list, tuple)) and len(info) >= 6:
                    planner, stage, task, idx, total, t_start = info[:6]
                    step_str = f"({idx:2d}/{total:2d})" if total > 0 else "..."
                    w_elapsed = max(now - t_start, 0.0)
                    worker_table.add_row(
                        f"episode_{ep_id:03d}",
                        f"{planner} / {stage}",
                        str(task),
                        step_str,
                        f"{w_elapsed:.1f}s",
                    )
            table.add_row(worker_table)
        else:
            table.add_row("[dim]Waiting for workers...[/dim]")

        return table
