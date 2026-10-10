"""LockstepMonitor windows, verdicts and the report table, fed with plain status records."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from arena_evaluation.benchmark.lockstep import STALL_FAIL_S, STALL_REPORT_LAG_S, LockstepMonitor, LockstepSummary, format_report


def _status(
    *,
    active: bool = True,
    waiting: tuple[str, ...] = (),
    rtf: float = 0.0,
    channels: tuple[str, ...] = ("nav/jackal", "engine"),
    ticks: int = 0,
    counts: tuple[tuple[str, str, int, int], ...] = (),
) -> SimpleNamespace:
    regs = [SimpleNamespace(caller="x", env="", channels=[SimpleNamespace(name=n, hard=True) for n in channels])]
    return SimpleNamespace(
        active=active,
        waiting_on=list(waiting),
        measured_rtf=rtf,
        registrations=regs,
        gated_ticks=ticks,
        counts=[SimpleNamespace(name=n, topic=t, due=d, received=r) for n, t, d, r in counts],
    )


_ENV0 = "/task_generator_node/env_0"
_ENV1 = "/task_generator_node/env_1"


def _counts(n: int, *, env1_received: int | None = None) -> tuple[tuple[str, str, int, int], ...]:
    return (
        ("engine", "/arena/env_0/agent_states", n, n),
        ("planner/jackal", f"{_ENV0}/jackal/lockstep/planner", n, n),
        ("planner/jackal", f"{_ENV1}/jackal/lockstep/planner", n, n if env1_received is None else env1_received),
    )


def test_clean_window_passes() -> None:
    m = LockstepMonitor()
    m.observe(_status(rtf=2.0), 0.0)
    m.open("ep0", 1.0)
    m.observe(_status(rtf=3.0), 2.0)
    m.observe(_status(rtf=1.0), 3.0)
    s = m.close("ep0", 4.0)
    assert s.active and s.stalls == 0 and s.max_stall_s == 0.0
    assert s.rtf == pytest.approx(2.0)
    assert s.channels == ("engine", "nav/jackal")
    assert s.beat_seen and s.verdict == "pass"


def test_stall_accounting_includes_report_lag() -> None:
    m = LockstepMonitor()
    m.open("ep0", 0.0)
    m.observe(_status(waiting=("nav/jackal",)), 10.0)
    m.observe(_status(waiting=("nav/jackal",)), 20.0)
    m.observe(_status(rtf=2.0), 13.0 + 10.0)
    m.observe(_status(waiting=("engine",)), 30.0)
    m.observe(_status(), 31.0)
    s = m.close("ep0", 40.0)
    assert s.stalls == 2
    assert s.max_stall_s == pytest.approx(13.0 + STALL_REPORT_LAG_S)
    assert s.stall_s == pytest.approx(14.0 + 2 * STALL_REPORT_LAG_S)
    assert s.stalled_on == ("engine", "nav/jackal")
    assert s.verdict == "fail"


def test_window_opened_mid_stall_inherits_it_and_open_stall_closes_with_window() -> None:
    m = LockstepMonitor()
    m.observe(_status(waiting=("planner/jackal",)), 5.0)
    m.open("ep1", 6.0)
    s = m.close("ep1", 8.0)
    assert s.stalls == 1
    assert s.max_stall_s == pytest.approx(2.0 + STALL_REPORT_LAG_S)
    assert s.stalled_on == ("planner/jackal",)


def test_no_beat_fails_and_inactive_is_not_a_verdict() -> None:
    m = LockstepMonitor()
    m.open("a", 0.0)
    m.observe(_status(channels=("engine",), rtf=1.0), 1.0)
    assert m.close("a", 2.0).verdict == "fail"
    idle = LockstepMonitor()
    idle.open("b", 0.0)
    idle.observe(_status(active=False, channels=()), 1.0)
    assert idle.close("b", 2.0).verdict == "inactive"
    assert LockstepSummary().verdict == "inactive"


def test_stall_just_under_threshold_passes() -> None:
    m = LockstepMonitor()
    m.open("a", 0.0)
    m.observe(_status(waiting=("nav/jackal",)), 1.0)
    m.observe(_status(rtf=1.0), 1.0 + STALL_FAIL_S - STALL_REPORT_LAG_S - 0.1)
    assert m.close("a", 10.0).verdict == "pass"


def test_merge_and_roundtrip() -> None:
    a = LockstepSummary(active=True, stalls=1, stall_s=2.0, max_stall_s=2.0, rtf=2.0, channels=("nav/jackal",), stalled_on=("nav/jackal",), ticks=10, beats=9, beats_due=10)
    b = LockstepSummary(active=True, stalls=0, stall_s=0.0, max_stall_s=0.0, rtf=4.0, channels=("engine",), ticks=5, beats=5, beats_due=5)
    merged = a.merge(b)
    assert merged.stalls == 1 and merged.max_stall_s == 2.0 and merged.rtf == pytest.approx(3.0)
    assert (merged.ticks, merged.beats, merged.beats_due) == (15, 14, 15)
    assert merged.channels == ("engine", "nav/jackal")
    assert LockstepSummary.from_dict(merged.to_dict()) == merged
    assert a.merge(LockstepSummary()).rtf == pytest.approx(2.0)


def test_format_report_columns() -> None:
    rows = [
        ("drlvo", "soak", LockstepSummary(active=True, rtf=2.5, channels=("engine", "planner/jackal"), ticks=30, beats=30, beats_due=30)),
        ("teb", "soak", LockstepSummary(active=True, stalls=2, max_stall_s=7.25, rtf=1.0, channels=("nav/jackal",), stalled_on=("nav/jackal",))),
    ]
    text = format_report(rows)
    lines = text.splitlines()
    assert lines[0].split() == ["contestant", "stage", "verdict", "stalls", "max_stall_s", "rtf", "ticks", "beats", "beat_channels", "stalled_on"]
    assert lines[1].split() == ["drlvo", "soak", "pass", "0", "0.0", "2.50", "30", "30/30", "planner/jackal", "-"]
    assert lines[2].split() == ["teb", "soak", "fail", "2", "7.2", "1.00", "0", "0/0", "nav/jackal", "nav/jackal"]


def test_beat_counts_are_window_deltas_scoped_to_the_env() -> None:
    m = LockstepMonitor()
    m.observe(_status(ticks=100, counts=_counts(100)), 0.0)
    m.open("ep0", 0.5, env=_ENV1)
    m.observe(_status(ticks=130, counts=_counts(130, env1_received=125)), 1.0)
    s = m.close("ep0", 1.5)
    assert (s.ticks, s.beats, s.beats_due) == (30, 25, 30)


def test_unscoped_window_counts_every_env() -> None:
    m = LockstepMonitor()
    m.open("ep0", 0.0)
    m.observe(_status(ticks=10, counts=_counts(10)), 1.0)
    s = m.close("ep0", 2.0)
    assert (s.ticks, s.beats, s.beats_due) == (10, 20, 20)


def test_counter_reset_folds_the_finished_run_into_the_window() -> None:
    m = LockstepMonitor()
    m.observe(_status(ticks=40, counts=_counts(40)), 0.0)
    m.open("ep0", 0.5, env=_ENV0)
    m.observe(_status(ticks=50, counts=_counts(50)), 1.0)
    m.observe(_status(active=False, channels=()), 2.0)
    m.observe(_status(ticks=7, counts=_counts(7)), 3.0)
    s = m.close("ep0", 4.0)
    assert (s.ticks, s.beats, s.beats_due) == (17, 17, 17)


def test_summary_roundtrip_keeps_counts() -> None:
    s = LockstepSummary(active=True, channels=("planner/jackal",), ticks=3, beats=2, beats_due=3)
    assert LockstepSummary.from_dict(s.to_dict()) == s
    assert "ticks=3 beats=2/3 beat_channels=planner/jackal" in s.short()
