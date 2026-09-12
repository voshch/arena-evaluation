"""Runtime blending quality, read back from recorded episodes (manuscript §3.2 playback gate, applied in the loop).

Every recorded pedestrian's joint state goes through the rig's own FK, and the per-tick step of the wrist,
elbow and head links is compared against the playback gate's 0.25 m per 50 ms. Ticks are split into
*transition* windows (a gesture slot fading in or out, a clip installed, the base animation switching
between gait and a canned clip) and *steady* playback, from the recorded `animation_states`.

`--source replay` (default) re-runs the recorded layer inputs (each ped's gesture channels, locomotion state,
velocity and heading, all on arena_peds) through the current AnimationManager + GestureLayer, so the numbers
describe the layer as it is now, independent of the build the episodes were recorded with. `--source recorded`
reads the joint states as published.

    python -m arena_evaluation.experiments.blend_quality --benchmark-dir <data_root>/<run_id> [--source replay] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

GATE_LINK_STEP_M = 0.25  # task_generator gestures/qa.py MAX_LINK_STEP_M, per 50 ms tick
TICK_S = 0.05
LINKS = ("l_wrist", "r_wrist", "l_elbow", "r_elbow", "head")
WINDOW_BEFORE_S = 0.1
WINDOW_AFTER_S = 0.9  # FADE_S 0.25 plus the longest move/transition (MOVE_MAX_S 0.9)


def _messages(mcap_path: pathlib.Path, suffixes: tuple[str, ...]):
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        # decode only the wanted channels, the lidar and tf traffic dominates an episode
        topics = [c.topic for c in reader.get_summary().channels.values() if c.topic.endswith(suffixes)]
        for _schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=topics):
            yield channel.topic, message.log_time, ros_msg


def _yaw(q: object) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def episode_tracks(mcap_path: pathlib.Path) -> tuple[dict[int, list], dict[int, list], dict[int, list]]:
    """Per ped: [(t_s, angles)] as published, [(t_s, frozenset of slot keys, base)] from animation_states,
    and [(t_s, layer input)] with what publish_arena_peds fed the animation layer."""
    joints: dict[int, list] = defaultdict(list)
    anim: dict[int, list] = defaultdict(list)
    inputs: dict[int, list] = defaultdict(list)
    for topic, t_ns, msg in _messages(mcap_path, ("/arena_peds", "/animation_states")):
        t = t_ns / 1e9
        if topic.endswith("/arena_peds"):
            for p in msg.pedestrians:
                js = p.joint_state
                if js.name:
                    joints[int(p.id)].append((t, dict(zip(js.name, js.position, strict=False))))
                yaw = _yaw(p.pose.orientation)
                channels = tuple((g.slot, (g.at.x, g.at.y, g.at.z), g.clip, g.hand, bool(g.render_pose_override)) for g in p.gestures)
                inputs[int(p.id)].append((t, int(p.animation_state), p.twist.linear.x * math.cos(yaw) + p.twist.linear.y * math.sin(yaw), (p.pose.position.x, p.pose.position.y, yaw), channels))
        else:
            for p in msg.peds:
                keys = frozenset((s.slot, s.animation, s.phase) for s in p.slots)
                anim[int(p.id)].append((t, keys, p.base))
    return joints, anim, inputs


def replay(inputs: dict[int, list]) -> dict[int, list]:
    """Joint angles the current layer produces from the recorded inputs, the way publish_arena_peds computes them."""
    from task_generator.simulators.human import animation_mananager as am
    from task_generator.simulators.human.gestures import Channel, GestureLayer, GestureRequest

    log = _Log()
    mgr = am.AnimationManager(pathlib.Path(am.__file__).resolve().parent / "animations", logger=log, fps=20.0)
    layer = GestureLayer(mgr, log)
    mgr.gesture_hook = layer
    moving_states = (1, 2)  # Pedestrian.WALKING, RUNNING
    out: dict[int, list] = defaultdict(list)
    samples = sorted((t, pid, state, speed, pose, chans) for pid, rows in inputs.items() for t, state, speed, pose, chans in rows)
    prev: dict[int, float] = {}
    for t, pid, state, speed, pose, chans in samples:
        dt = 0.05 if pid not in prev else max(0.0, t - prev[pid])
        prev[pid] = t
        req = GestureRequest(channels=tuple(Channel(slot=c[0], at=c[1], clip=c[2], hand=c[3], lock=c[4]) for c in chans), pose=pose, moving=state in moving_states)
        out[pid].append((t, mgr.compute(pid, state, speed, dt, gesture=req)))
    return out


class _Log:
    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass


def transition_times(states: list) -> list[float]:
    """Times the animation layer changes what it plays: a slot appears, leaves, changes clip or phase, or the base switches."""
    out = []
    prev_keys, prev_base = None, None
    for t, keys, base in states:
        if prev_keys is not None and (keys != prev_keys or base != prev_base):
            out.append(t)
        prev_keys, prev_base = keys, base
    return out


def link_steps(samples: list, fk, body) -> tuple[np.ndarray, np.ndarray]:
    """(t, worst link step in m per tick) between consecutive joint-state samples, rescaled to a 50 ms tick."""
    ts, pos = [], []
    for t, angles in samples:
        p, _ = fk(angles, body)
        ts.append(t)
        pos.append(np.stack([p[k] for k in LINKS]))
    ts, pos = np.asarray(ts), np.asarray(pos)
    if len(ts) < 2:
        return np.zeros(0), np.zeros(0)
    dt = np.diff(ts)
    ok = dt > 1e-4
    step = np.linalg.norm(np.diff(pos, axis=0), axis=-1).max(axis=1)
    return ts[1:][ok], (step[ok] * TICK_S / dt[ok])


def analyze_episode(mcap_path: pathlib.Path, fk, body, source: str = "replay") -> dict:
    joints, anim, inputs = episode_tracks(mcap_path)
    if source == "replay":
        joints = replay(inputs)
    trans_steps, steady_steps, events, events_over = [], [], 0, 0
    for pid, samples in joints.items():
        full = []
        for t, a in samples:
            full.append((t, {**dict.fromkeys(_joint_names(), 0.0), **a}))
        t, step = link_steps(full, fk, body)
        if len(t) == 0:
            continue
        in_window = np.zeros(len(t), dtype=bool)
        for te in transition_times(anim.get(pid, [])):
            w = (t >= te - WINDOW_BEFORE_S) & (t <= te + WINDOW_AFTER_S)
            in_window |= w
            if w.any():
                events += 1
                events_over += int(step[w].max() > GATE_LINK_STEP_M)
        trans_steps.append(step[in_window])
        steady_steps.append(step[~in_window])
    trans = np.concatenate(trans_steps) if trans_steps else np.zeros(0)
    steady = np.concatenate(steady_steps) if steady_steps else np.zeros(0)
    return {"episode": mcap_path.parent.name, "peds": len(joints), "events": events, "events_over_gate": events_over, "trans": trans, "steady": steady}


_NAMES: list[str] = []


def _joint_names() -> list[str]:
    if not _NAMES:
        from task_generator.simulators.human.pointing.contract import ROS_JOINT_ORDER

        _NAMES.extend(ROS_JOINT_ORDER)
    return _NAMES


def summarize(results: list[dict]) -> dict:
    trans = np.concatenate([r["trans"] for r in results]) if results else np.zeros(0)
    steady = np.concatenate([r["steady"] for r in results]) if results else np.zeros(0)

    def stats(x: np.ndarray) -> dict:
        if len(x) == 0:
            return {"ticks": 0}
        return {"ticks": int(len(x)), "median_m": float(np.median(x)), "p99_m": float(np.percentile(x, 99)), "max_m": float(x.max()), "over_gate": int((x > GATE_LINK_STEP_M).sum())}

    events = sum(r["events"] for r in results)
    over = sum(r["events_over_gate"] for r in results)
    return {
        "episodes": len(results),
        "transitions": events,
        "transitions_over_gate": over,
        "transition_pass_rate": (1.0 - over / events) if events else None,
        "transition_ticks": stats(trans),
        "steady_ticks": stats(steady),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N episodes")
    ap.add_argument("--source", choices=("replay", "recorded"), default="replay")
    args = ap.parse_args(argv)
    from task_generator.simulators.human.gestures import BODY_HEIGHT
    from task_generator.simulators.human.pointing import skeleton as S

    body = S.Body(BODY_HEIGHT)
    mcaps = sorted(args.benchmark_dir.glob("episodes/*/*.mcap"))[: args.limit]
    results = []
    for m in mcaps:
        try:
            results.append(analyze_episode(m, S.fk, body, args.source))
        except Exception as e:  # an episode still being written or truncated
            print(f"skip {m.parent.name}: {e!r}", file=sys.stderr)
    summary = {"source": args.source, **summarize(results)}
    out = args.out or args.benchmark_dir / "layer"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"blend_quality_{args.source}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
