"""Arena Evaluation Pipeline Profiler."""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import datetime
import logging
import os
import pathlib
import threading
import time
import typing

import yaml

_log = logging.getLogger(__name__)

_nvml_available: bool = False
_nvml_initialised: bool = False
_nvml_lock = threading.Lock()


def _ensure_nvml() -> bool:
    """Try to initialise NVML once. Thread-safe, idempotent."""
    global _nvml_available, _nvml_initialised
    if _nvml_initialised:
        return _nvml_available
    with _nvml_lock:
        if _nvml_initialised:
            return _nvml_available
        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml_available = True
            _log.debug("pynvml initialised successfully")
        except Exception:
            _nvml_available = False
            _log.debug("pynvml not available - GPU metrics will be null")
        _nvml_initialised = True
    return _nvml_available


@dataclasses.dataclass(slots=True)
class _GpuSnapshot:
    """Single GPU reading."""

    util_percent: float
    vram_used_mb: float
    vram_total_mb: float


@dataclasses.dataclass(slots=True)
class _ResourceSnapshot:
    """One point-in-time reading of system resources."""

    timestamp: float
    cpu_percent_total: float
    cpu_percent_per_core: list[float]
    ram_used_mb: float
    ram_percent: float
    gpu: _GpuSnapshot | None
    disk_read_bytes: int
    disk_write_bytes: int
    process_read_bytes: int = 0
    process_write_bytes: int = 0


class _RunningStats:
    """Thread-safe running min/max/sum/count accumulator for a single scalar."""

    __slots__ = ("_min", "_max", "_sum", "_count", "_lock")

    def __init__(self) -> None:
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._sum: float = 0.0
        self._count: int = 0
        self._lock = threading.Lock()

    def update(self, value: float) -> None:
        with self._lock:
            if value < self._min:
                self._min = value
            if value > self._max:
                self._max = value
            self._sum += value
            self._count += 1

    def snapshot(self) -> dict[str, float | None]:
        with self._lock:
            if self._count == 0:
                return {"max": None, "mean": None}
            return {
                "max": round(self._max, 2),
                "mean": round(self._sum / self._count, 2),
            }


class _PerCoreAccumulator:
    """Track per-core CPU max and mean values."""

    __slots__ = ("_maxes", "_sums", "_count", "_lock")

    def __init__(self) -> None:
        self._maxes: list[float] = []
        self._sums: list[float] = []
        self._count = 0
        self._lock = threading.Lock()

    def update(self, per_core: list[float]) -> None:
        with self._lock:
            self._count += 1
            if not self._maxes:
                self._maxes = list(per_core)
                self._sums = list(per_core)
            else:
                for i, v in enumerate(per_core):
                    if i < len(self._maxes):
                        if v > self._maxes[i]:
                            self._maxes[i] = v
                        self._sums[i] += v
                    else:
                        self._maxes.append(v)
                        self._sums.append(v)

    def snapshot(self) -> dict[str, list[float]]:
        with self._lock:
            if not self._maxes or self._count == 0:
                return {"max": [], "mean": []}
            return {
                "max": [round(v, 1) for v in self._maxes],
                "mean": [round(s / self._count, 1) for s in self._sums],
            }


class SystemSampler:
    """Low-level system resource sampler using /proc."""

    def __init__(self) -> None:
        self._prev_cpu_total: list[int] | None = None
        self._prev_cpu_idle: list[int] | None = None
        self._prev_cpu_total_all: int | None = None
        self._prev_cpu_idle_all: int | None = None
        self._prev_disk_read: int = 0
        self._prev_disk_write: int = 0
        self._prev_ts: float = 0.0

    @staticmethod
    def _read_proc_stat() -> tuple[list[int], list[int], int, int]:
        """Parse /proc/stat for per-core and aggregate CPU jiffies."""
        per_core_totals: list[int] = []
        per_core_idles: list[int] = []
        total_all = 0
        idle_all = 0

        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("cpu"):
                    parts = line.split()
                    if parts[0] == "cpu":
                        vals = [int(x) for x in parts[1:]]
                        total_all = sum(vals)
                        idle_all = vals[3] if len(vals) > 3 else 0
                    else:
                        vals = [int(x) for x in parts[1:]]
                        per_core_totals.append(sum(vals))
                        per_core_idles.append(vals[3] if len(vals) > 3 else 0)
                else:
                    break
        return per_core_totals, per_core_idles, total_all, idle_all

    def _sample_cpu(self) -> tuple[float, list[float]]:
        """Return (total_cpu_percent, [per_core_percent, ...])."""
        totals, idles, total_all, idle_all = self._read_proc_stat()

        if self._prev_cpu_total_all is None:
            self._prev_cpu_total = totals
            self._prev_cpu_idle = idles
            self._prev_cpu_total_all = total_all
            self._prev_cpu_idle_all = idle_all
            return 0.0, [0.0] * len(totals)

        dt = total_all - self._prev_cpu_total_all
        di = idle_all - self._prev_cpu_idle_all
        cpu_pct = ((dt - di) / dt * 100.0) if dt > 0 else 0.0

        per_core: list[float] = []
        prev_t = self._prev_cpu_total or []
        prev_i = self._prev_cpu_idle or []
        for i in range(len(totals)):
            if i < len(prev_t):
                core_dt = totals[i] - prev_t[i]
                core_di = idles[i] - prev_i[i]
                pct = ((core_dt - core_di) / core_dt * 100.0) if core_dt > 0 else 0.0
            else:
                pct = 0.0
            per_core.append(pct)

        self._prev_cpu_total = totals
        self._prev_cpu_idle = idles
        self._prev_cpu_total_all = total_all
        self._prev_cpu_idle_all = idle_all

        return cpu_pct, per_core

    @staticmethod
    def _sample_ram() -> tuple[float, float]:
        """Return (used_mb, percent_used) from /proc/meminfo."""
        mem: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    mem[key] = int(parts[1])
                if len(mem) >= 4:
                    break

        total_kb = mem.get("MemTotal", 0)
        available_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
        used_kb = total_kb - available_kb
        used_mb = used_kb / 1024.0
        percent = (used_kb / total_kb * 100.0) if total_kb > 0 else 0.0
        return used_mb, percent

    @staticmethod
    def _sample_gpu() -> _GpuSnapshot | None:
        """Read GPU utilisation and VRAM via pynvml. Returns None if unavailable."""
        if not _ensure_nvml():
            return None
        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return _GpuSnapshot(
                util_percent=float(util.gpu),
                vram_used_mb=mem.used / (1024 * 1024),
                vram_total_mb=mem.total / (1024 * 1024),
            )
        except Exception:
            return None

    @staticmethod
    def _read_diskstats() -> tuple[int, int]:
        """Sum read/write sectors across all block devices from /proc/diskstats.

        Sector size is assumed to be 512 bytes (Linux standard).
        """
        total_read = 0
        total_write = 0
        with open("/proc/diskstats") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 14:
                    continue
                dev = parts[2]
                if any(c.isdigit() for c in dev) and not dev.startswith("nvme"):
                    continue
                if dev.startswith("nvme") and "p" in dev:
                    continue
                if dev.startswith("loop") or dev.startswith("ram") or dev.startswith("dm-"):
                    continue
                total_read += int(parts[5])
                total_write += int(parts[9])
        return total_read * 512, total_write * 512

    def _sample_disk(self) -> tuple[int, int]:
        """Return cumulative (read_bytes, write_bytes) since boot."""
        return self._read_diskstats()

    @staticmethod
    def _read_process_io() -> tuple[int, int]:
        """Read cumulative rchar and wchar from /proc/self/io."""
        rchar = 0
        wchar = 0
        try:
            with open("/proc/self/io") as f:
                for line in f:
                    if line.startswith("rchar:"):
                        rchar = int(line.split()[1])
                    elif line.startswith("wchar:"):
                        wchar = int(line.split()[1])
        except Exception:
            pass
        return rchar, wchar

    def sample(self) -> _ResourceSnapshot:
        """Take one full resource snapshot."""
        ts = time.monotonic()
        cpu_pct, cpu_per_core = self._sample_cpu()
        ram_mb, ram_pct = self._sample_ram()
        gpu = self._sample_gpu()
        disk_r, disk_w = self._sample_disk()
        proc_r, proc_w = self._read_process_io()

        return _ResourceSnapshot(
            timestamp=ts,
            cpu_percent_total=cpu_pct,
            cpu_percent_per_core=cpu_per_core,
            ram_used_mb=ram_mb,
            ram_percent=ram_pct,
            gpu=gpu,
            disk_read_bytes=disk_r,
            disk_write_bytes=disk_w,
            process_read_bytes=proc_r,
            process_write_bytes=proc_w,
        )


class SimulationProfiler:
    """Background profiler for live simulation runs."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        sample_hz: float = 2.0,
    ) -> None:
        self._output_dir = pathlib.Path(output_dir)
        self._sample_interval = 1.0 / max(0.1, sample_hz)
        self._sample_hz = sample_hz
        self._sampler = SystemSampler()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopped = False

        self._cpu = _RunningStats()
        self._ram_mb = _RunningStats()
        self._ram_pct = _RunningStats()
        self._gpu_util = _RunningStats()
        self._vram_mb = _RunningStats()
        self._disk_read_mbps = _RunningStats()
        self._disk_write_mbps = _RunningStats()
        self._per_core = _PerCoreAccumulator()
        self._sample_count = 0

        self._started_at: str = ""
        self._start_mono: float = 0.0

        self._prev_disk_read: int = 0
        self._prev_disk_write: int = 0
        self._prev_disk_ts: float = 0.0

    def start(self) -> None:
        """Begin background sampling."""
        if self._thread is not None:
            return

        self._started_at = datetime.datetime.now(datetime.UTC).isoformat()
        self._start_mono = time.monotonic()

        snap = self._sampler.sample()
        self._prev_disk_read = snap.disk_read_bytes
        self._prev_disk_write = snap.disk_write_bytes
        self._prev_disk_ts = snap.timestamp

        self._thread = threading.Thread(
            target=self._sampling_loop,
            name="simulation-profiler",
            daemon=True,
        )
        self._thread.start()

        atexit.register(self.stop)

        _log.info(
            "SimulationProfiler started (%.1f Hz, output=%s)",
            self._sample_hz,
            self._output_dir,
        )

    def _sampling_loop(self) -> None:
        """Daemon thread main loop."""
        while not self._stop_event.is_set():
            try:
                snap = self._sampler.sample()
                self._accumulate(snap)
            except Exception:
                pass
            self._stop_event.wait(timeout=self._sample_interval)

    def _accumulate(self, snap: _ResourceSnapshot) -> None:
        """Update running accumulators from a snapshot."""
        self._cpu.update(snap.cpu_percent_total)
        self._ram_mb.update(snap.ram_used_mb)
        self._ram_pct.update(snap.ram_percent)
        self._per_core.update(snap.cpu_percent_per_core)

        if snap.gpu is not None:
            self._gpu_util.update(snap.gpu.util_percent)
            self._vram_mb.update(snap.gpu.vram_used_mb)

        dt = snap.timestamp - self._prev_disk_ts
        if dt > 0:
            read_rate = (snap.disk_read_bytes - self._prev_disk_read) / dt / (1024 * 1024)
            write_rate = (snap.disk_write_bytes - self._prev_disk_write) / dt / (1024 * 1024)
            self._disk_read_mbps.update(max(0.0, read_rate))
            self._disk_write_mbps.update(max(0.0, write_rate))

        self._prev_disk_read = snap.disk_read_bytes
        self._prev_disk_write = snap.disk_write_bytes
        self._prev_disk_ts = snap.timestamp
        self._sample_count += 1

    def stop(self) -> None:
        """Stop sampling and write simulation_profile.yaml."""
        if self._stopped:
            return
        self._stopped = True

        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        with contextlib.suppress(Exception):
            atexit.unregister(self.stop)

        ended_at = datetime.datetime.now(datetime.UTC).isoformat()
        duration = time.monotonic() - self._start_mono if self._start_mono else 0.0

        self._write_yaml(ended_at, duration)
        _log.info(
            "SimulationProfiler stopped (%d samples, %.1fs)",
            self._sample_count,
            duration,
        )

    def _write_yaml(self, ended_at: str, duration: float) -> None:
        """Write simulation_profile.yaml to output_dir."""
        cpu = self._cpu.snapshot()
        ram_mb = self._ram_mb.snapshot()
        ram_pct = self._ram_pct.snapshot()
        gpu = self._gpu_util.snapshot()
        vram = self._vram_mb.snapshot()
        disk_r = self._disk_read_mbps.snapshot()
        disk_w = self._disk_write_mbps.snapshot()
        per_core = self._per_core.snapshot()

        data: dict[str, typing.Any] = {
            "simulation_profile": {
                "started_at": self._started_at,
                "ended_at": ended_at,
                "duration_s": round(duration, 1),
                "sample_hz": self._sample_hz,
                "samples_collected": self._sample_count,
                "cpu": {
                    "percent_max": cpu["max"],
                    "percent_mean": cpu["mean"],
                    "per_core_max": per_core["max"],
                    "per_core_mean": per_core["mean"],
                },
                "ram": {
                    "MB_max": ram_mb["max"],
                    "MB_mean": ram_mb["mean"],
                    "percent_max": ram_pct["max"],
                },
                "gpu": (
                    {
                        "percent_max": gpu["max"],
                        "vram_MB_max": vram["max"],
                    }
                    if gpu["max"] is not None
                    else None
                ),
                "disk_io": {
                    "read_MBps_max": disk_r["max"],
                    "read_MBps_mean": disk_r["mean"],
                    "write_MBps_max": disk_w["max"],
                    "write_MBps_mean": disk_w["mean"],
                },
            }
        }

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / "simulation_profile.yaml"
        tmp_path = out_path.with_suffix(".yaml.tmp")
        tmp_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        os.replace(tmp_path, out_path)
        _log.info("Wrote %s", out_path)


@dataclasses.dataclass
class _PhaseStats:
    """Accumulated stats for one pipeline phase."""

    duration_s: float = 0.0
    cpu: _RunningStats = dataclasses.field(default_factory=_RunningStats)
    per_core: _PerCoreAccumulator = dataclasses.field(default_factory=_PerCoreAccumulator)
    ram_mb: _RunningStats = dataclasses.field(default_factory=_RunningStats)
    gpu_util: _RunningStats = dataclasses.field(default_factory=_RunningStats)
    vram_mb: _RunningStats = dataclasses.field(default_factory=_RunningStats)
    disk_read_mbps: _RunningStats = dataclasses.field(default_factory=_RunningStats)
    disk_write_mbps: _RunningStats = dataclasses.field(default_factory=_RunningStats)
    _has_gpu: bool = False

    def to_dict(self) -> dict[str, typing.Any]:
        cpu = self.cpu.snapshot()
        ram = self.ram_mb.snapshot()
        per_core = self.per_core.snapshot()
        result: dict[str, typing.Any] = {
            "duration_s": round(self.duration_s, 2),
            "cpu_percent_max": cpu["max"],
            "cpu_percent_mean": cpu["mean"],
            "cpu_per_core_max": per_core["max"],
            "cpu_per_core_mean": per_core["mean"],
            "ram_MB_max": ram["max"],
            "ram_MB_mean": ram["mean"],
        }

        disk_r = self.disk_read_mbps.snapshot()
        disk_w = self.disk_write_mbps.snapshot()
        if disk_r["max"] is not None:
            result["disk_read_MBps_max"] = disk_r["max"]
            result["disk_read_MBps_mean"] = disk_r["mean"]
            result["disk_write_MBps_max"] = disk_w["max"]
            result["disk_write_MBps_mean"] = disk_w["mean"]
        else:
            result["disk_read_MBps_max"] = None
            result["disk_read_MBps_mean"] = None
            result["disk_write_MBps_max"] = None
            result["disk_write_MBps_mean"] = None
        if self._has_gpu:
            gpu = self.gpu_util.snapshot()
            vram = self.vram_mb.snapshot()
            result["gpu_percent_max"] = gpu["max"]
            result["gpu_percent_mean"] = gpu["mean"]
            result["vram_MB_max"] = vram["max"]
        else:
            result["gpu_percent_max"] = None
            result["vram_MB_max"] = None
        return result


class PipelineProfiler:
    """Per-phase profiler for offline evaluation pipeline steps."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        sample_hz: float = 2.0,
    ) -> None:
        self._output_dir = pathlib.Path(output_dir)
        self._sample_hz = sample_hz
        self._sample_interval = 1.0 / max(0.1, sample_hz)
        self._phases: dict[str, _PhaseStats] = {}
        self._active_phases: set[str] = set()
        self._total_start: float = time.monotonic()

    @contextlib.contextmanager
    def phase(self, name: str) -> typing.Generator[None, None, None]:
        """Context manager that samples resources during a named phase."""
        if name in self._active_phases:
            yield
            return

        if name not in self._phases:
            self._phases[name] = _PhaseStats()
        stats = self._phases[name]

        self._active_phases.add(name)
        sampler = SystemSampler()
        stop_event = threading.Event()

        def _sample_loop() -> None:
            snap = sampler.sample()
            prev_disk_r = snap.process_read_bytes
            prev_disk_w = snap.process_write_bytes
            prev_disk_ts = snap.timestamp

            while not stop_event.is_set():
                try:
                    snap = sampler.sample()
                    stats.cpu.update(snap.cpu_percent_total)
                    stats.per_core.update(snap.cpu_percent_per_core)
                    stats.ram_mb.update(snap.ram_used_mb)
                    if snap.gpu is not None:
                        stats.gpu_util.update(snap.gpu.util_percent)
                        stats.vram_mb.update(snap.gpu.vram_used_mb)
                        stats._has_gpu = True

                    dt = snap.timestamp - prev_disk_ts
                    if dt > 0:
                        read_rate = (snap.process_read_bytes - prev_disk_r) / dt / (1024 * 1024)
                        write_rate = (snap.process_write_bytes - prev_disk_w) / dt / (1024 * 1024)
                        stats.disk_read_mbps.update(max(0.0, read_rate))
                        stats.disk_write_mbps.update(max(0.0, write_rate))

                    prev_disk_r = snap.process_read_bytes
                    prev_disk_w = snap.process_write_bytes
                    prev_disk_ts = snap.timestamp
                except Exception:
                    pass
                stop_event.wait(timeout=self._sample_interval)

        thread = threading.Thread(
            target=_sample_loop,
            name=f"pipeline-profiler-{name}",
            daemon=True,
        )

        t0 = time.monotonic()
        thread.start()
        _log.info("PipelineProfiler: phase '%s' started", name)

        try:
            yield
        finally:
            stop_event.set()
            thread.join(timeout=3.0)
            self._active_phases.discard(name)
            stats.duration_s += time.monotonic() - t0
            _log.info(
                "PipelineProfiler: phase '%s' ended (%.1fs)",
                name,
                stats.duration_s,
            )

    def write_summary(self) -> None:
        """Write pipeline_profile.yaml to output_dir."""
        total_duration = time.monotonic() - self._total_start

        phases_dict: dict[str, typing.Any] = {}
        for name, stats in self._phases.items():
            phases_dict[name] = stats.to_dict()

        data: dict[str, typing.Any] = {
            "pipeline_profile": {
                "total_duration_s": round(total_duration, 2),
                "sample_hz": self._sample_hz,
                "phases": phases_dict,
            }
        }

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / "pipeline_profile.yaml"
        tmp_path = out_path.with_suffix(".yaml.tmp")
        tmp_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        os.replace(tmp_path, out_path)
        _log.info("Wrote %s", out_path)
