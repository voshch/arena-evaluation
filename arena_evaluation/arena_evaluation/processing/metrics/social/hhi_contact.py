"""Two-person contact interactions (hug, handshake): stalls, release-time windows, passes-between."""

from __future__ import annotations

import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
)
from arena_evaluation.processing.pedestrian_frames import pedestrians_at
from arena_evaluation.storage.registry import Metric, MetricScope, define_metric
from arena_evaluation.storage.registry.series import SeriesHhiContact
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

CONTACT_TYPES: Final[dict[int, str]] = {10: "hug", 11: "shake_hand"}
EVENT_HOLD_ONSET: Final[str] = "HOLD_ONSET"
EVENT_RELEASED: Final[str] = "RELEASED"
EVENT_ACTIVATED: Final[str] = "ACTIVATED"

STALL_SPEED: Final[float] = 0.05
STALL_MIN_S: Final[float] = 3.0
SPEED_WINDOW_S: Final[float] = 0.25
MOVING_SPEED: Final[float] = 0.1
GOAL_TOLERANCE: Final[float] = 0.5
POST_S: Final[float] = 5.0

WINDOWS: Final[tuple[str, ...]] = ("approach", "hold", "post", "rest")

STALL_SERIES_SCHEMA: Final[pl.Schema] = SeriesHhiContact.polars_schema()


def _episode_metric(name: str, dtype: pl.DataType | type[pl.DataType], unit: str, description: str) -> Metric:
    """Episode-scope output of the two-person interaction, with a null reason."""
    return define_metric(
        name,
        dtype,
        scopes=(MetricScope.EPISODE,),
        unit=unit,
        description=description,
        null_reason=True,
    )


HHI_INTERACTION = _episode_metric(
    "hhi_interaction",
    pl.String,
    "",
    "First two-person contact type of the interaction episode: hug or shake_hand.",
)
HHI_HOLD_ONSET_S = _episode_metric(
    "hhi_hold_onset_s",
    pl.Float64,
    "s",
    "Seconds from the episode's start to the pair's hold onset.",
)
HHI_RELEASE_S = _episode_metric(
    "hhi_release_s",
    pl.Float64,
    "s",
    "Seconds from the episode's start to the pair's release.",
)
HHI_EPISODE_S = _episode_metric(
    "hhi_episode_s",
    pl.Float64,
    "s",
    "Seconds of the episode's robot time axis the interaction windows are measured over.",
)
HHI_STALL_COUNT = _episode_metric(
    "hhi_stall_count",
    pl.Int64,
    "",
    "Number of robot stalls of at least 3 s after first motion and away from the goal.",
)
HHI_STALL_TIME_S = _episode_metric(
    "hhi_stall_time_s",
    pl.Float64,
    "s",
    "Total seconds the robot spent in those stalls.",
)
HHI_STALLS_APPROACH = _episode_metric(
    "hhi_stalls_approach",
    pl.Int64,
    "",
    "Stalls whose onset fell in the approach window.",
)
HHI_STALLS_HOLD = _episode_metric(
    "hhi_stalls_hold",
    pl.Int64,
    "",
    "Stalls whose onset fell in the hold window.",
)
HHI_STALLS_POST = _episode_metric(
    "hhi_stalls_post",
    pl.Int64,
    "",
    "Stalls whose onset fell in the post-release window.",
)
HHI_STALLS_REST = _episode_metric(
    "hhi_stalls_rest",
    pl.Int64,
    "",
    "Stalls whose onset fell in the rest window.",
)
HHI_SHARE_APPROACH = _episode_metric(
    "hhi_share_approach",
    pl.Float64,
    "1",
    "Share of the episode's time covered by the approach window.",
)
HHI_SHARE_HOLD = _episode_metric(
    "hhi_share_hold",
    pl.Float64,
    "1",
    "Share of the episode's time covered by the hold window.",
)
HHI_SHARE_POST = _episode_metric(
    "hhi_share_post",
    pl.Float64,
    "1",
    "Share of the episode's time covered by the post-release window.",
)
HHI_SHARE_REST = _episode_metric(
    "hhi_share_rest",
    pl.Float64,
    "1",
    "Share of the episode's time covered by the rest window.",
)
HHI_PASSES_BETWEEN = _episode_metric(
    "hhi_passes_between",
    pl.Int64,
    "",
    "Number of robot steps that passed between the two interacting pedestrians during the hold.",
)
HHI_MIN_PAIR_CLEARANCE = _episode_metric(
    "hhi_min_pair_clearance",
    pl.Float64,
    "m",
    "Minimum distance of the robot to the segment between the two interacting pedestrians during the hold.",
)
HHI_PAIR_GAP_HOLD = _episode_metric(
    "hhi_pair_gap_hold",
    pl.Float64,
    "m",
    "Median distance between the two interacting pedestrians during the hold.",
)


def smoothed_speed(t_s: np.ndarray, x: np.ndarray, y: np.ndarray, window_s: float = SPEED_WINDOW_S) -> np.ndarray:
    """Planar speed from positions, moving-averaged over window_s."""
    n = len(t_s)
    if n < 2:
        return np.zeros(n)
    step_s = np.diff(t_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(step_s > 0.0, np.hypot(np.diff(x), np.diff(y)) / step_s, 0.0)
    speed = np.append(speed, speed[-1])
    step = float(np.median(step_s)) if n > 2 else window_s
    kernel = max(1, int(round(window_s / step))) if step > 0 else 1
    return np.convolve(speed, np.ones(kernel) / kernel, mode="same")


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


def window_bounds(
    t_end: float,
    hold_onset: float | None,
    release: float | None,
    post_s: float = POST_S,
) -> dict[str, tuple[float, float]]:
    """Episode-relative [start, end) of each window, clipped to [0, t_end]; absent edges collapse the later windows."""

    def clip(a: float, b: float) -> tuple[float, float]:
        a, b = min(max(a, 0.0), t_end), min(max(b, 0.0), t_end)
        return (a, max(a, b))

    if hold_onset is None:
        return dict.fromkeys(WINDOWS, clip(t_end, t_end)) | {"approach": clip(0.0, t_end)}
    hold_end = t_end if release is None else release
    post_end = hold_end if release is None else release + post_s
    return {
        "approach": clip(0.0, hold_onset),
        "hold": clip(hold_onset, hold_end),
        "post": clip(hold_end, post_end),
        "rest": clip(post_end, t_end),
    }


def _last_window(windows: dict[str, tuple[float, float]]) -> str | None:
    """Last window with a non-zero extent; its end also takes an onset exactly on the boundary."""
    return max(
        (window for window in WINDOWS if windows[window][1] > windows[window][0]),
        key=lambda window: windows[window][1],
        default=None,
    )


def _window_containing(onset: float, windows: dict[str, tuple[float, float]], last: str | None) -> str | None:
    """The window whose [start, end) holds the onset, or the last window's closed end."""
    for window in WINDOWS:
        start, end = windows[window]
        if start <= onset < end or (window == last and onset == end):
            return window
    return None


def bucket(onsets: typing.Iterable[float], windows: dict[str, tuple[float, float]]) -> dict[str, int]:
    """Count onsets per window, [start, end) except the last non-empty window which also takes its end."""
    counts: dict[str, int] = dict.fromkeys(WINDOWS, 0)
    last = _last_window(windows)
    for onset in onsets:
        window = _window_containing(onset, windows, last)
        if window is not None:
            counts[window] += 1
    return counts


def _cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (first[:, 0] - origin[:, 0]) * (second[:, 1] - origin[:, 1]) - (first[:, 1] - origin[:, 1]) * (second[:, 0] - origin[:, 0])


def segment_crossings(robot: np.ndarray, ped_a: np.ndarray, ped_b: np.ndarray) -> np.ndarray:
    """Per step i: whether the move robot[i]->robot[i+1] crosses the pair segment a[i]-b[i], a sample on the segment counting once."""
    if len(robot) < 2:
        return np.zeros(0, dtype=bool)
    start, end = robot[:-1], robot[1:]
    a, b = ped_a[:-1], ped_b[:-1]
    d1, d2 = _cross(a, b, start), _cross(a, b, end)
    d3, d4 = _cross(start, end, a), _cross(start, end, b)
    moved = np.any(start != end, axis=1)
    hit = ((d1 >= 0) != (d2 >= 0)) & ((d3 >= 0) != (d4 >= 0)) & moved
    return hit & np.isfinite(d1) & np.isfinite(d2) & np.isfinite(d3) & np.isfinite(d4)


def point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = b - a
    denominator = np.einsum("ij,ij->i", delta, delta)
    projection = np.where(
        denominator > 0,
        np.einsum("ij,ij->i", point - a, delta) / np.where(denominator > 0, denominator, 1.0),
        0.0,
    )
    projection = np.clip(projection, 0.0, 1.0)
    return np.linalg.norm(point - (a + delta * projection[:, None]), axis=1)


def pair_track(held: pl.DataFrame, t_ns: np.ndarray, ped_id: str) -> np.ndarray:
    """(N, 2) position of one pedestrian at each robot time, NaN where it is not in the frame held at that time."""
    positions = np.full((len(t_ns), 2), np.nan)
    rows = held.filter(pl.col("ped_id") == ped_id).select("t_ns", "x", "y")
    if rows.height == 0:
        return positions
    held_times = rows.get_column("t_ns").to_numpy().astype(np.int64)
    positions[np.searchsorted(t_ns.astype(np.int64), held_times)] = rows.select("x", "y").to_numpy().astype(np.float64)
    return positions


class ContactInteractionCalculator(BaseMetricCalculator):
    """Stalls bucketed by the pair's approach / hold / post-release windows, and passes between the pair."""

    NAME = "hhi_contact"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    REQUIRED_TOPICS = [
        "robot_tracks",
        "pedestrian_arena_peds",
        "pedestrian_interaction_events",
    ]

    OUTPUTS = (
        HHI_INTERACTION,
        HHI_HOLD_ONSET_S,
        HHI_RELEASE_S,
        HHI_EPISODE_S,
        HHI_STALL_COUNT,
        HHI_STALL_TIME_S,
        HHI_STALLS_APPROACH,
        HHI_STALLS_HOLD,
        HHI_STALLS_POST,
        HHI_STALLS_REST,
        HHI_SHARE_APPROACH,
        HHI_SHARE_HOLD,
        HHI_SHARE_POST,
        HHI_SHARE_REST,
        HHI_PASSES_BETWEEN,
        HHI_MIN_PAIR_CLEARANCE,
        HHI_PAIR_GAP_HOLD,
    )
    PRIMARY_OUTPUTS = ["hhi_stall_count", "hhi_passes_between"]
    DIRECTIONS = {"hhi_min_pair_clearance": Direction.HIGHER}

    @staticmethod
    def _contact_edges(events: pl.DataFrame | None, t0_ns: int) -> tuple[int | None, list[str], float | None, float | None]:
        """(type, participant ped_ids, hold onset s, release s) of the episode's first contact interaction."""
        if events is None or events.height == 0:
            return None, [], None, None
        contact = events.filter(pl.col("interaction_type").is_in(list(CONTACT_TYPES))).sort("t_ns")
        activated = contact.filter(pl.col("event_name").is_in([EVENT_ACTIVATED, EVENT_HOLD_ONSET]))
        if activated.height == 0:
            return None, [], None, None
        first = activated.row(0, named=True)
        interaction_id = first["interaction_id"]
        mine = contact.filter(pl.col("interaction_id") == interaction_id)
        same_event = mine.filter((pl.col("t_ns") == first["t_ns"]) & (pl.col("event_name") == first["event_name"]))
        participants = [str(v) for v in same_event.get_column("ped_id").drop_nulls().to_numpy()]

        def at(name: str) -> float | None:
            rows = mine.filter(pl.col("event_name") == name)
            return None if rows.height == 0 else (int(rows.get_column("t_ns")[0]) - t0_ns) / 1e9

        return (
            int(first["interaction_type"]),
            participants,
            at(EVENT_HOLD_ONSET),
            at(EVENT_RELEASED),
        )

    def _robot_time_axis(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        """Times, x, y and the offset of the episode view's robot samples, none when it has fewer than two."""
        data = episode.data
        if data is None or data.height < 2:
            return None
        pos_x, pos_y, _yaw, _odom_x, _odom_y, _odom_yaw = self.resolve_robot_pose(episode)
        t_ns = data.get_column("t_ns").to_numpy().astype(np.int64)
        length = min(len(t_ns), len(pos_x))
        if length < 2:
            return None
        return (
            t_ns[:length],
            pos_x[:length],
            pos_y[:length],
            float(t_ns[0]),
        )

    def _stall_intervals(
        self,
        episode: AlignedEpisodeBundle,
        axis: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None,
    ) -> list[tuple[float, float]] | None:
        """(onset, end) seconds of every stall of the episode's robot, none without a resolved axis."""
        axis = self._robot_time_axis(episode) if axis is None else axis
        if axis is None:
            return None
        t_ns, pos_x, pos_y, t0_ns = axis
        t_s = (t_ns - t0_ns) / 1e9
        dist_to_goal = None
        if episode.goal_pos and len(episode.goal_pos) >= 2:
            dist_to_goal = np.hypot(pos_x - episode.goal_pos[0], pos_y - episode.goal_pos[1])
        return stall_intervals(t_s, smoothed_speed(t_s, pos_x, pos_y), dist_to_goal)

    def _metrics_for_axis(
        self,
        episode: AlignedEpisodeBundle,
        axis: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    ) -> dict[str, typing.Any]:
        """Every hhi output over one robot time axis and the episode view's own interaction stream."""
        t_ns, pos_x, pos_y, t0_ns = axis
        t_s = (t_ns - t0_ns) / 1e9
        t_end = float(t_s[-1])
        out: dict[str, typing.Any] = dict.fromkeys(self.output_keys())

        topics = episode.topics or {}
        events = topics.get("pedestrian_interaction_events")
        if events is None or events.height == 0:
            # Without the human-human interaction stream no hold/release window exists: the stall
            # counters and window shares are unknown, not zero. A recorded stream with no contact
            # event is different and keeps its honest zeros.
            out["hhi_episode_s"] = t_end
            return out
        interaction_type, participants, hold_onset, release = self._contact_edges(events, int(t0_ns))
        out["hhi_interaction"] = CONTACT_TYPES.get(interaction_type) if interaction_type is not None else None
        out["hhi_hold_onset_s"] = hold_onset
        out["hhi_release_s"] = release
        out["hhi_episode_s"] = t_end

        stalls = self._stall_intervals(episode, axis) or []
        onsets = [onset for onset, _ in stalls]
        out["hhi_stall_count"] = len(stalls)
        out["hhi_stall_time_s"] = float(sum(end - onset for onset, end in stalls))

        windows = window_bounds(t_end, hold_onset, release)
        for window, count in bucket(onsets, windows).items():
            out[f"hhi_stalls_{window}"] = count
        for window, (start_s, end_s) in windows.items():
            out[f"hhi_share_{window}"] = (end_s - start_s) / t_end if t_end > 0 else None

        peds_df = self.native_ped_frame(episode)
        if len(participants) >= 2 and peds_df is not None:
            held = pedestrians_at(pl.Series("t_ns", t_ns), peds_df)
            pair_a = pair_track(held, t_ns, participants[0])
            pair_b = pair_track(held, t_ns, participants[1])
            robot = np.column_stack([pos_x, pos_y])
            present = np.isfinite(pair_a).all(axis=1) & np.isfinite(pair_b).all(axis=1)
            if hold_onset is not None:
                hold_start_s, hold_end_s = windows["hold"]
                in_hold = (t_s >= hold_start_s) & (t_s <= hold_end_s) & present
                if len(in_hold) > 1:
                    out["hhi_passes_between"] = int(np.sum(segment_crossings(robot, pair_a, pair_b) & in_hold[:-1] & in_hold[1:]))
                if in_hold.any():
                    out["hhi_min_pair_clearance"] = float(np.min(point_segment_distance(robot[in_hold], pair_a[in_hold], pair_b[in_hold])))
                    out["hhi_pair_gap_hold"] = float(np.median(np.linalg.norm(pair_a[in_hold] - pair_b[in_hold], axis=1)))
        return out

    def missing_input_provenance(self, episode: AlignedEpisodeBundle) -> dict[str, str]:
        """Stall windows need the interaction stream; without it their counters are unknown, not zero."""
        events = (episode.topics or {}).get("pedestrian_interaction_events")
        if events is None or events.height == 0:
            return {key: f"{key}:topic_missing:pedestrian_interaction_events" for key in self.output_keys()}
        return {}

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        axis = self._robot_time_axis(episode)
        if axis is None:
            return dict.fromkeys(self.output_keys())
        return self._metrics_for_axis(episode, axis)

    SERIES_TABLE_STEM = "hhi_contact"
    SERIES_KEY = ("t_ns",)
    # The stall series derives from the robot's own motion; the interaction stream only labels the
    # windows, so it is not a series requirement.
    SERIES_REQUIRED_TOPICS = ("robot_tracks",)

    def series_table(
        self,
        episode: AlignedEpisodeBundle,
        run_id: str = "",
        robot_id: str = "",
    ) -> pl.DataFrame:
        """1NF stall series: one row per stall onset (natural key ``t_ns``) with its duration and window.

        The stalls are derived from the robot's own motion; the interaction stream only labels the
        approach/hold/post/rest windows. The table is therefore emitted whenever robot ground truth
        exists, independently of whether ``pedestrian_interaction_events`` was recorded.
        """
        axis = self._robot_time_axis(episode)
        stalls = self._stall_intervals(episode, axis) or []
        if not stalls or axis is None:
            return pl.DataFrame(schema=STALL_SERIES_SCHEMA)
        t_ns, _pos_x, _pos_y, t0_ns = axis
        t_end = float((t_ns[-1] - t0_ns) / 1e9)
        events = (episode.topics or {}).get("pedestrian_interaction_events")
        _interaction_type, _participants, hold_onset, release = self._contact_edges(events, int(t0_ns))
        windows = window_bounds(t_end, hold_onset, release)
        last_window = _last_window(windows)
        window_of = {onset: _window_containing(onset, windows, last_window) for onset, _end in stalls}
        rows = [
            {
                "run_id": run_id,
                "episode_id": int(episode.episode_id),
                "robot_id": robot_id or (episode.robot_name or "robot"),
                "t_ns": int(t0_ns + round(onset * 1e9)),
                "onset_s": onset,
                "duration_s": end - onset,
                "window": window_of.get(onset),
            }
            for onset, end in stalls
        ]
        return pl.DataFrame(rows, schema=STALL_SERIES_SCHEMA)
