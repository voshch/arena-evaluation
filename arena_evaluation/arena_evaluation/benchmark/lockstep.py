"""Per-window lockstep bookkeeping from /arena/state/lockstep, and the soak verdict."""

from __future__ import annotations

import dataclasses
import threading
import typing

if typing.TYPE_CHECKING:
    from arena_runtime_msgs.msg import LockstepStatus

STALL_FAIL_S = 5.0
# the scheduler publishes a stall only after this long at the gate, so observed stalls start late by it
STALL_REPORT_LAG_S = 1.0
BEAT_PREFIXES = ("nav/", "planner/")


@dataclasses.dataclass(frozen=True)
class LockstepSummary:
    active: bool = False
    stalls: int = 0
    stall_s: float = 0.0
    max_stall_s: float = 0.0
    rtf: float = 0.0
    channels: tuple[str, ...] = ()
    stalled_on: tuple[str, ...] = ()

    @property
    def beat_seen(self) -> bool:
        return any(ch.startswith(BEAT_PREFIXES) for ch in self.channels)

    @property
    def verdict(self) -> str:
        if not self.active:
            return "inactive"
        if self.max_stall_s >= STALL_FAIL_S or not self.beat_seen:
            return "fail"
        return "pass"

    def merge(self, other: LockstepSummary) -> LockstepSummary:
        rtfs = [r for r in (self.rtf, other.rtf) if r > 0.0]
        return LockstepSummary(
            active=self.active or other.active,
            stalls=self.stalls + other.stalls,
            stall_s=self.stall_s + other.stall_s,
            max_stall_s=max(self.max_stall_s, other.max_stall_s),
            rtf=sum(rtfs) / len(rtfs) if rtfs else 0.0,
            channels=tuple(sorted(set(self.channels) | set(other.channels))),
            stalled_on=tuple(sorted(set(self.stalled_on) | set(other.stalled_on))),
        )

    def short(self) -> str:
        beats = ",".join(ch for ch in self.channels if ch.startswith(BEAT_PREFIXES)) or "-"
        return f"lockstep {self.verdict}: stalls={self.stalls} max={self.max_stall_s:.1f}s rtf={self.rtf:.2f} beats={beats}"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LockstepSummary:
        return cls(
            active=bool(d.get("active", False)),
            stalls=int(d.get("stalls", 0)),
            stall_s=float(d.get("stall_s", 0.0)),
            max_stall_s=float(d.get("max_stall_s", 0.0)),
            rtf=float(d.get("rtf", 0.0)),
            channels=tuple(d.get("channels", ())),
            stalled_on=tuple(d.get("stalled_on", ())),
        )


@dataclasses.dataclass
class _Window:
    active: bool = False
    stalls: int = 0
    stall_s: float = 0.0
    max_stall_s: float = 0.0
    rtf_samples: list[float] = dataclasses.field(default_factory=list)
    channels: set[str] = dataclasses.field(default_factory=set)
    stalled_on: set[str] = dataclasses.field(default_factory=set)
    stall_since: float | None = None

    def end_stall(self, now: float) -> None:
        if self.stall_since is None:
            return
        duration = now - self.stall_since + STALL_REPORT_LAG_S
        self.stall_s += duration
        self.max_stall_s = max(self.max_stall_s, duration)
        self.stall_since = None


class LockstepMonitor:
    """Feed every LockstepStatus with its wall time, open a window per episode, close it for a summary.
    A window opened mid-stall inherits the stall from its open time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: dict[object, _Window] = {}
        self._active = False
        self._channels: set[str] = set()
        self._waiting: set[str] = set()

    def observe(self, msg: LockstepStatus, now: float) -> None:
        waiting = set(msg.waiting_on)
        channels = {ch.name for reg in msg.registrations for ch in reg.channels if ch.hard}
        with self._lock:
            self._active = bool(msg.active)
            self._channels = channels
            self._waiting = waiting
            for w in self._windows.values():
                w.active = w.active or bool(msg.active)
                w.channels |= channels
                if waiting:
                    w.stalled_on |= waiting
                    if w.stall_since is None:
                        w.stall_since = now
                        w.stalls += 1
                else:
                    w.end_stall(now)
                    if msg.active and msg.measured_rtf > 0.0:
                        w.rtf_samples.append(float(msg.measured_rtf))

    def open(self, key: object, now: float) -> None:
        with self._lock:
            w = _Window(active=self._active, channels=set(self._channels))
            if self._waiting:
                w.stalled_on |= self._waiting
                w.stall_since = now
                w.stalls = 1
            self._windows[key] = w

    def close(self, key: object, now: float) -> LockstepSummary:
        with self._lock:
            w = self._windows.pop(key)
            w.end_stall(now)
            return LockstepSummary(
                active=w.active,
                stalls=w.stalls,
                stall_s=w.stall_s,
                max_stall_s=w.max_stall_s,
                rtf=sum(w.rtf_samples) / len(w.rtf_samples) if w.rtf_samples else 0.0,
                channels=tuple(sorted(w.channels)),
                stalled_on=tuple(sorted(w.stalled_on)),
            )


def format_table(header: typing.Sequence[str], table: typing.Sequence[typing.Sequence[str]]) -> str:
    """Aligned, left-justified column printer shared by the lockstep and efficacy reports."""
    widths = [max(len(str(r[i])) for r in (header, *table)) for i in range(len(header))]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)).rstrip() for row in (header, *table)]
    return "\n".join(lines)


def format_report(rows: typing.Sequence[tuple[str, str, LockstepSummary]]) -> str:
    """One line per (contestant, stage): verdict, stalls, longest stall, mean rtf, beats, stalled channels."""
    header = ("contestant", "stage", "verdict", "stalls", "max_stall_s", "rtf", "beats", "stalled_on")
    table = [
        (
            contestant,
            stage,
            s.verdict,
            str(s.stalls),
            f"{s.max_stall_s:.1f}",
            f"{s.rtf:.2f}",
            ",".join(ch for ch in s.channels if ch.startswith(BEAT_PREFIXES)) or "-",
            ",".join(s.stalled_on) or "-",
        )
        for contestant, stage, s in rows
    ]
    return format_table(header, table)
