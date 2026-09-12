"""Two-person contact interactions (hug, handshake): stalls, release-time windows, passes-between.

The episode is split at the pair's lifecycle edges (humansim `interaction_events`):
approach = start -> HOLD_ONSET, hold = -> RELEASED, post = the POST_S after release, rest = after that.
Every stall is bucketed by the window its onset falls in, next to that window's share of episode time.
"""

from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

CONTACT_TYPES = {10: "hug", 11: "shake_hand"}  # arena_humansim InteractionType
EVENT_HOLD_ONSET = "HOLD_ONSET"
EVENT_RELEASED = "RELEASED"
EVENT_ACTIVATED = "ACTIVATED"

STALL_SPEED = 0.05  # m/s, below this the robot counts as stopped
STALL_MIN_S = 3.0  # a stop shorter than this is not a stall (manuscript §4.1: at least 3 s)
SPEED_WINDOW_S = 0.25  # moving-average window of the speed signal
MOVING_SPEED = 0.1  # stops before the robot first exceeds this are start-up latency, not stalls
GOAL_TOLERANCE = 0.5  # m, stops this close to the goal are arrival
POST_S = 5.0  # post-release window length

WINDOWS = ("approach", "hold", "post", "rest")


def smoothed_speed(t_s: np.ndarray, x: np.ndarray, y: np.ndarray, window_s: float = SPEED_WINDOW_S) -> np.ndarray:
    """Planar speed from positions, moving-averaged over window_s."""
    n = len(t_s)
    if n < 2:
        return np.zeros(n)
    dt = np.diff(t_s)
    dt[dt <= 0.0] = np.nan
    v = np.hypot(np.diff(x), np.diff(y)) / dt
    v = np.nan_to_num(np.append(v, v[-1]), nan=0.0)
    step = float(np.nanmedian(np.diff(t_s))) if n > 2 else window_s
    k = max(1, int(round(window_s / step))) if step > 0 else 1
    return np.convolve(v, np.ones(k) / k, mode="same")


def stall_intervals(
    t_s: np.ndarray,
    speed: np.ndarray,
    dist_to_goal: np.ndarray | None = None,
    *,
    stall_speed: float = STALL_SPEED,
    min_s: float = STALL_MIN_S,
    moving_speed: float = MOVING_SPEED,
    goal_tolerance: float = GOAL_TOLERANCE,
) -> list[tuple[float, float]]:
    """(onset, end) of every stop at least min_s long, after the robot first moved and away from the goal."""
    if len(t_s) == 0:
        return []
    moved = np.flatnonzero(speed > moving_speed)
    if len(moved) == 0:
        return []
    stopped = speed < stall_speed
    stopped[: moved[0]] = False
    if dist_to_goal is not None:
        stopped &= dist_to_goal > goal_tolerance
    out: list[tuple[float, float]] = []
    edges = np.diff(np.concatenate(([0], stopped.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        onset, end = float(t_s[start]), float(t_s[stop - 1])
        if end - onset >= min_s:
            out.append((onset, end))
    return out


def window_bounds(t_end: float, hold_onset: float | None, release: float | None, post_s: float = POST_S) -> dict[str, tuple[float, float]]:
    """Episode-relative [start, end) of each window, clipped to [0, t_end]; absent edges collapse the later windows."""

    def clip(a: float, b: float) -> tuple[float, float]:
        a, b = min(max(a, 0.0), t_end), min(max(b, 0.0), t_end)
        return (a, max(a, b))

    if hold_onset is None:
        return {"approach": clip(0.0, t_end), "hold": clip(t_end, t_end), "post": clip(t_end, t_end), "rest": clip(t_end, t_end)}
    hold_end = t_end if release is None else release
    post_end = hold_end if release is None else release + post_s
    return {
        "approach": clip(0.0, hold_onset),
        "hold": clip(hold_onset, hold_end),
        "post": clip(hold_end, post_end),
        "rest": clip(post_end, t_end),
    }


def bucket(onsets: typing.Iterable[float], windows: dict[str, tuple[float, float]]) -> dict[str, int]:
    """Count onsets per window, [start, end) except the last non-empty window which also takes its end."""
    counts = dict.fromkeys(WINDOWS, 0)
    last = max((w for w in WINDOWS if windows[w][1] > windows[w][0]), key=lambda w: windows[w][1], default=None)
    for t in onsets:
        for w in WINDOWS:
            a, b = windows[w]
            if a <= t < b or (w == last and t == b):
                counts[w] += 1
                break
    return counts


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:, 0] - o[:, 0]) * (b[:, 1] - o[:, 1]) - (a[:, 1] - o[:, 1]) * (b[:, 0] - o[:, 0])


def segment_crossings(robot: np.ndarray, ped_a: np.ndarray, ped_b: np.ndarray) -> np.ndarray:
    """Per step i: does the robot's move robot[i]->robot[i+1] cross the pair segment a[i]-b[i].

    A point on a line counts to its non-negative side, so a sample landing exactly on the
    segment is one crossing, not zero (strict signs) or two (inclusive signs).
    """
    if len(robot) < 2:
        return np.zeros(0, dtype=bool)
    p, q = robot[:-1], robot[1:]
    a, b = ped_a[:-1], ped_b[:-1]
    d1, d2 = _cross(a, b, p), _cross(a, b, q)
    d3, d4 = _cross(p, q, a), _cross(p, q, b)
    moved = np.any(p != q, axis=1)
    hit = ((d1 >= 0) != (d2 >= 0)) & ((d3 >= 0) != (d4 >= 0)) & moved
    return hit & np.isfinite(d1) & np.isfinite(d2) & np.isfinite(d3) & np.isfinite(d4)


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.where(denom > 0, np.einsum("ij,ij->i", p - a, ab) / np.where(denom > 0, denom, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p - (a + ab * t[:, None]), axis=1)


def pair_track(peds_ids: list, peds_positions: list, pid: int) -> np.ndarray:
    """(N, 2) position of ped pid per aligned row, NaN where it is absent."""
    out = np.full((len(peds_ids), 2), np.nan)
    for i, (ids, pos) in enumerate(zip(peds_ids, peds_positions, strict=True)):
        if ids is None or pos is None:
            continue
        ids = list(ids)
        if pid in ids:
            j = ids.index(pid)
            out[i] = (pos[3 * j], pos[3 * j + 1])
    return out


class ContactInteractionCalculator(BaseMetricCalculator):
    """Stalls bucketed by the pair's approach / hold / post-release windows, and passes between the pair."""

    NAME = "hhi_contact"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    REQUIRED_TOPICS = ["odom", "peds", "interaction_events"]

    UNITS = {
        "hhi_hold_onset_s": "s",
        "hhi_release_s": "s",
        "hhi_episode_s": "s",
        "hhi_stall_count": "",
        "hhi_stall_time_s": "s",
        "hhi_passes_between": "",
        "hhi_min_pair_clearance": "m",
        "hhi_pair_gap_hold": "m",
        **{f"hhi_stalls_{w}": "" for w in WINDOWS},
        **{f"hhi_share_{w}": "" for w in WINDOWS},
    }
    PRIMARY_OUTPUTS = ["hhi_stall_count", "hhi_passes_between"]
    OUTPUT_DIRECTIONS = {"hhi_stall_count": "lower", "hhi_stall_time_s": "lower", "hhi_min_pair_clearance": "higher"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "hhi_interaction",
            "hhi_hold_onset_s",
            "hhi_release_s",
            "hhi_episode_s",
            "hhi_stall_count",
            "hhi_stall_time_s",
            "hhi_stall_onsets_s",
            *(f"hhi_stalls_{w}" for w in WINDOWS),
            *(f"hhi_share_{w}" for w in WINDOWS),
            "hhi_passes_between",
            "hhi_min_pair_clearance",
            "hhi_pair_gap_hold",
        ]

    @staticmethod
    def _contact_edges(events: pl.DataFrame | None, t0_ns: int) -> tuple[int | None, list[int], float | None, float | None]:
        """(type, participants, hold onset s, release s) of the episode's first contact interaction."""
        if events is None or events.height == 0:
            return None, [], None, None
        contact = events.filter(pl.col("interaction_type").is_in(list(CONTACT_TYPES))).sort("time_ns")
        activated = contact.filter(pl.col("event_name").is_in([EVENT_ACTIVATED, EVENT_HOLD_ONSET]))
        if activated.height == 0:
            return None, [], None, None
        first = activated.row(0, named=True)
        iid = first["interaction_id"]
        mine = contact.filter(pl.col("interaction_id") == iid)

        def at(name: str) -> float | None:
            rows = mine.filter(pl.col("event_name") == name)
            return None if rows.height == 0 else (int(rows["time_ns"][0]) - t0_ns) / 1e9

        return int(first["interaction_type"]), [int(p) for p in first["participants"]], at(EVENT_HOLD_ONSET), at(EVENT_RELEASED)

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        out: dict[str, typing.Any] = dict.fromkeys(self.output_keys())
        data = episode.data
        if data is None or data.height < 2 or "time_ns" not in data.columns:
            return out
        topics = episode.topics or {}
        pos_x, pos_y, _, _, _, _ = self.resolve_robot_pose(episode)
        data = episode.data  # resolve_robot_pose may trim it to the pose segment
        time_ns = data["time_ns"].to_numpy()
        n = min(len(time_ns), len(pos_x))
        if n < 2:
            return out
        time_ns, pos_x, pos_y = time_ns[:n], pos_x[:n], pos_y[:n]
        t0_ns = int(time_ns[0])
        t_s = (time_ns - t0_ns) / 1e9
        t_end = float(t_s[-1])

        itype, participants, hold_onset, release = self._contact_edges(topics.get("interaction_events"), t0_ns)
        out["hhi_interaction"] = CONTACT_TYPES.get(itype) if itype is not None else None
        out["hhi_hold_onset_s"] = hold_onset
        out["hhi_release_s"] = release
        out["hhi_episode_s"] = t_end

        dist_to_goal = None
        if episode.goal_pos and len(episode.goal_pos) >= 2:
            dist_to_goal = np.hypot(pos_x - episode.goal_pos[0], pos_y - episode.goal_pos[1])
        stalls = stall_intervals(t_s, smoothed_speed(t_s, pos_x, pos_y), dist_to_goal)
        onsets = [a for a, _ in stalls]
        out["hhi_stall_count"] = len(stalls)
        out["hhi_stall_time_s"] = float(sum(b - a for a, b in stalls))
        out["hhi_stall_onsets_s"] = onsets

        windows = window_bounds(t_end, hold_onset, release)
        for w, count in bucket(onsets, windows).items():
            out[f"hhi_stalls_{w}"] = count
        for w, (a, b) in windows.items():
            out[f"hhi_share_{w}"] = (b - a) / t_end if t_end > 0 else None

        if len(participants) >= 2 and "peds_ids" in data.columns and "peds_positions" in data.columns:
            ids, positions = data["peds_ids"].to_list()[:n], data["peds_positions"].to_list()[:n]
            a, b = pair_track(ids, positions, participants[0]), pair_track(ids, positions, participants[1])
            robot = np.column_stack([pos_x, pos_y])
            present = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            if hold_onset is not None:
                # only the hold: before it the partners can stand metres apart on either side of the path
                h0, h1 = windows["hold"]
                in_hold = (t_s >= h0) & (t_s <= h1) & present
                if len(in_hold) > 1:
                    out["hhi_passes_between"] = int(np.sum(segment_crossings(robot, a, b) & in_hold[:-1] & in_hold[1:]))
                if in_hold.any():
                    out["hhi_min_pair_clearance"] = float(np.min(point_segment_distance(robot[in_hold], a[in_hold], b[in_hold])))
                    out["hhi_pair_gap_hold"] = float(np.median(np.linalg.norm(a[in_hold] - b[in_hold], axis=1)))
        return out
