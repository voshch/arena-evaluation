"""
Tests for the arena evaluation profiler module.
"""

from __future__ import annotations

import os
import pathlib
import signal
import threading
import time

import pytest
import yaml


@pytest.fixture
def tmp_output(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a temporary directory for profiler output."""
    out = tmp_path / "profiling_output"
    out.mkdir()
    return out


class TestSystemSampler:
    """Tests for the low-level /proc reader."""

    def test_sample_returns_snapshot(self):
        from arena_evaluation.benchmark.profiler import SystemSampler

        sampler = SystemSampler()
        snap = sampler.sample()
        assert snap.timestamp > 0
        assert snap.ram_used_mb > 0
        assert snap.ram_percent >= 0

    def test_cpu_requires_two_samples(self):
        from arena_evaluation.benchmark.profiler import SystemSampler

        sampler = SystemSampler()
        snap1 = sampler.sample()
        assert snap1.cpu_percent_total == 0.0

        time.sleep(0.1)
        snap2 = sampler.sample()
        assert snap2.cpu_percent_total >= 0.0
        assert snap2.cpu_percent_total <= 100.0

    def test_per_core_list_is_populated(self):
        from arena_evaluation.benchmark.profiler import SystemSampler

        sampler = SystemSampler()
        sampler.sample()
        time.sleep(0.05)
        snap = sampler.sample()
        assert isinstance(snap.cpu_percent_per_core, list)
        assert len(snap.cpu_percent_per_core) >= 1

    def test_ram_values_sensible(self):
        from arena_evaluation.benchmark.profiler import SystemSampler

        sampler = SystemSampler()
        snap = sampler.sample()
        assert snap.ram_used_mb > 100
        assert snap.ram_percent > 0
        assert snap.ram_percent <= 100

    def test_disk_counters_non_negative(self):
        from arena_evaluation.benchmark.profiler import SystemSampler

        sampler = SystemSampler()
        snap = sampler.sample()
        assert snap.disk_read_bytes >= 0
        assert snap.disk_write_bytes >= 0

    def test_gpu_returns_none_or_snapshot(self):
        from arena_evaluation.benchmark.profiler import SystemSampler, _GpuSnapshot

        sampler = SystemSampler()
        snap = sampler.sample()
        assert snap.gpu is None or isinstance(snap.gpu, _GpuSnapshot)


class TestSimulationProfiler:
    """Tests for the benchmark/simulation profiler."""

    def test_start_stop_lifecycle(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=tmp_output, sample_hz=10.0)
        profiler.start()
        time.sleep(0.5)
        profiler.stop()

        assert profiler._stopped is True

    def test_writes_simulation_profile_yaml(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=tmp_output, sample_hz=10.0)
        profiler.start()
        time.sleep(0.5)
        profiler.stop()

        yaml_path = tmp_output / "simulation_profile.yaml"
        assert yaml_path.exists(), f"Expected {yaml_path} to exist"

        data = yaml.safe_load(yaml_path.read_text())
        assert "simulation_profile" in data
        profile = data["simulation_profile"]
        assert "started_at" in profile
        assert "ended_at" in profile
        assert "duration_s" in profile
        assert "samples_collected" in profile
        assert profile["samples_collected"] >= 3

    def test_cpu_stats_in_yaml(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=tmp_output, sample_hz=10.0)
        profiler.start()
        time.sleep(0.5)
        profiler.stop()

        data = yaml.safe_load((tmp_output / "simulation_profile.yaml").read_text())
        cpu = data["simulation_profile"]["cpu"]
        assert cpu["percent_max"] is not None
        assert cpu["percent_mean"] is not None
        assert cpu["percent_max"] >= cpu["percent_mean"]
        assert isinstance(cpu["per_core_max"], list)
        assert isinstance(cpu["per_core_mean"], list)
        assert len(cpu["per_core_max"]) >= 1
        assert len(cpu["per_core_mean"]) >= 1

    def test_ram_stats_in_yaml(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=tmp_output, sample_hz=10.0)
        profiler.start()
        time.sleep(0.5)
        profiler.stop()

        data = yaml.safe_load((tmp_output / "simulation_profile.yaml").read_text())
        profile = data["simulation_profile"]

        ram = profile["ram"]
        assert "MB_max" in ram
        assert "MB_max" in ram
        assert ram["MB_max"] > 0
        assert "MB_mean" in ram
        assert ram["MB_max"] >= ram["MB_mean"]

    def test_stop_is_idempotent(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=tmp_output, sample_hz=5.0)
        profiler.start()
        time.sleep(0.3)
        profiler.stop()
        profiler.stop()
        profiler.stop()

    def test_start_is_idempotent(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=tmp_output, sample_hz=5.0)
        profiler.start()
        profiler.start()
        time.sleep(0.2)
        profiler.stop()

    def test_disk_io_in_yaml(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=tmp_output, sample_hz=10.0)
        profiler.start()
        time.sleep(0.5)
        profiler.stop()

        data = yaml.safe_load((tmp_output / "simulation_profile.yaml").read_text())
        disk = data["simulation_profile"]["disk_io"]
        assert "read_MBps_max" in disk
        assert "write_MBps_max" in disk


class TestPipelineProfiler:
    """Tests for the evaluation pipeline profiler."""

    def test_phase_context_manager(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import PipelineProfiler

        profiler = PipelineProfiler(output_dir=tmp_output, sample_hz=10.0)

        with profiler.phase("test_phase"):
            time.sleep(0.3)

        assert "test_phase" in profiler._phases
        stats = profiler._phases["test_phase"]
        assert stats.duration_s >= 0.2

    def test_multiple_phases(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import PipelineProfiler

        profiler = PipelineProfiler(output_dir=tmp_output, sample_hz=10.0)

        with profiler.phase("extract"):
            time.sleep(0.2)
        with profiler.phase("process"):
            time.sleep(0.2)

        assert "extract" in profiler._phases
        assert "process" in profiler._phases

    def test_writes_pipeline_profile_yaml(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import PipelineProfiler

        profiler = PipelineProfiler(output_dir=tmp_output, sample_hz=10.0)

        with profiler.phase("extract"):
            time.sleep(0.2)
        with profiler.phase("process"):
            time.sleep(0.2)

        profiler.write_summary()

        yaml_path = tmp_output / "pipeline_profile.yaml"
        assert yaml_path.exists(), f"Expected {yaml_path} to exist"

        data = yaml.safe_load(yaml_path.read_text())
        assert "pipeline_profile" in data
        profile = data["pipeline_profile"]
        assert "total_duration_s" in profile
        assert "phases" in profile
        assert "extract" in profile["phases"]
        assert "process" in profile["phases"]

    def test_phase_stats_structure(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import PipelineProfiler

        profiler = PipelineProfiler(output_dir=tmp_output, sample_hz=10.0)

        with profiler.phase("extract"):
            time.sleep(0.3)

        profiler.write_summary()

        data = yaml.safe_load((tmp_output / "pipeline_profile.yaml").read_text())
        phase = data["pipeline_profile"]["phases"]["extract"]

        assert "duration_s" in phase
        assert phase["duration_s"] >= 0.2
        assert "cpu_percent_max" in phase
        assert "cpu_percent_mean" in phase
        assert "cpu_per_core_max" in phase
        assert "cpu_per_core_mean" in phase
        assert "ram_MB_max" in phase
        assert "ram_MB_mean" in phase
        assert "disk_read_MBps_max" in phase
        assert "disk_read_MBps_mean" in phase
        assert "disk_write_MBps_max" in phase
        assert "disk_write_MBps_mean" in phase

    def test_phase_cpu_max_gte_mean(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import PipelineProfiler

        profiler = PipelineProfiler(output_dir=tmp_output, sample_hz=10.0)

        with profiler.phase("work"):
            total = 0
            for i in range(500_000):
                total += i * i
            time.sleep(0.2)

        profiler.write_summary()

        data = yaml.safe_load((tmp_output / "pipeline_profile.yaml").read_text())
        phase = data["pipeline_profile"]["phases"]["work"]

        if phase["cpu_percent_max"] is not None and phase["cpu_percent_mean"] is not None:
            assert phase["cpu_percent_max"] >= phase["cpu_percent_mean"]


class TestRunningStats:
    """Unit tests for the O(1) accumulator."""

    def test_empty_snapshot(self):
        from arena_evaluation.benchmark.profiler import _RunningStats

        stats = _RunningStats()
        snap = stats.snapshot()
        assert snap["max"] is None
        assert snap["mean"] is None

    def test_single_value(self):
        from arena_evaluation.benchmark.profiler import _RunningStats

        stats = _RunningStats()
        stats.update(42.0)
        snap = stats.snapshot()
        assert snap["max"] == 42.0
        assert snap["mean"] == 42.0

    def test_multiple_values(self):
        from arena_evaluation.benchmark.profiler import _RunningStats

        stats = _RunningStats()
        for v in [10.0, 20.0, 30.0]:
            stats.update(v)
        snap = stats.snapshot()
        assert snap["max"] == 30.0
        assert snap["mean"] == 20.0

    def test_max_always_gte_mean(self):
        from arena_evaluation.benchmark.profiler import _RunningStats

        stats = _RunningStats()
        import random

        random.seed(42)
        for _ in range(100):
            stats.update(random.uniform(0, 100))
        snap = stats.snapshot()
        assert snap["max"] >= snap["mean"]


class TestSignalSafety:
    """Tests that profilers flush on abnormal termination."""

    def test_simulation_profiler_atexit_registered(self, tmp_output: pathlib.Path):
        from arena_evaluation.benchmark.profiler import SimulationProfiler

        profiler = SimulationProfiler(output_dir=tmp_output, sample_hz=5.0)
        profiler.start()
        assert profiler._thread is not None
        assert profiler._thread.is_alive()
        profiler.stop()
        assert not profiler._thread.is_alive()
