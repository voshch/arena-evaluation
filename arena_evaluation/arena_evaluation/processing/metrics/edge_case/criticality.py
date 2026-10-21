"""Criticality of a sampled episode against chosen pedestrians: clearance, TTC, PET, freezes, collisions, formations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import attrs

Point = tuple[float, float]

FREEZE_SPEED = 0.05
PERSONAL_SPACE = 0.5
SETTLE_S = 2.0
ENCOUNTER_NEAR_M = 1.5


@attrs.frozen
class Sample:
    t: float
    robot: Point
    robot_vel: Point = (0.0, 0.0)
    robot_radius: float = 0.3
    peds: dict[str, tuple[Point, Point, float]] = attrs.Factory(dict)


@attrs.frozen
class Criticality:
    """Per-episode criticality, None where nothing was observed, never zero."""

    min_ttc_s: float | None = None
    min_ttc_with: str = ""
    min_ttc_at: float | None = None

    min_clearance_m: float | None = None
    min_clearance_with: str = ""
    min_clearance_at: float | None = None

    pet_s: float | None = None
    pet_with: str = ""
    freeze_duration_s: float = 0.0
    intrusion_time_s: float = 0.0

    collided: bool = False
    collision_fault: str = ""
    collision_with: str = ""
    collision_at: float | None = None
    collision_robot_closing: float | None = None
    collision_ped_closing: float | None = None
    samples: int = 0
    duration_s: float = 0.0

    @property
    def observed(self) -> bool:
        return self.samples > 0 and self.min_clearance_m is not None


def time_to_collision(p_a: Point, v_a: Point, r_a: float, p_b: Point, v_b: Point, r_b: float) -> float | None:
    """Seconds until two discs on constant velocities touch, 0.0 when they already do, None when they never will."""
    dpx, dpy = p_b[0] - p_a[0], p_b[1] - p_a[1]
    dvx, dvy = v_b[0] - v_a[0], v_b[1] - v_a[1]
    r = r_a + r_b

    a = dvx * dvx + dvy * dvy
    if a < 1e-12:
        return None
    b = 2.0 * (dpx * dvx + dpy * dvy)
    c = dpx * dpx + dpy * dpy - r * r

    if c <= 0.0:
        return 0.0

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    for root in ((-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)):
        if root >= 0.0:
            return root
    return None


def _speed(v: Point) -> float:
    return math.hypot(v[0], v[1])


def post_encroachment_time(samples: Sequence[Sample], encounter: Point, agent: str) -> float | None:
    """Time gap between the robot and `agent` passing `encounter`, None when either never comes within ENCOUNTER_NEAR_M of it."""

    def crossing_time(track: Iterable[tuple[float, Point]]) -> float | None:
        best_t, best_d = None, math.inf
        for t, p in track:
            d = math.dist(p, encounter)
            if d < best_d:
                best_t, best_d = t, d
        return best_t if best_d <= ENCOUNTER_NEAR_M else None

    t_robot = crossing_time((s.t, s.robot) for s in samples)
    t_agent = crossing_time((s.t, s.peds[agent][0]) for s in samples if agent in s.peds)
    if t_robot is None or t_agent is None:
        return None
    return abs(t_agent - t_robot)


def score(samples: Sequence[Sample], *, only: Sequence[str] | None = None, encounters: Sequence[tuple[str, Point]] = (), settle_s: float = SETTLE_S) -> Criticality:
    """Criticality of `samples` against the pedestrians in `only`, every pedestrian when it is empty, with PET at the designed `encounters`."""
    if not samples:
        return Criticality()

    t0 = samples[0].t
    wanted = set(only) if only else None

    min_ttc: float | None = None
    ttc_with, ttc_at = "", None
    min_clear: float | None = None
    clear_with, clear_at = "", None
    freeze = 0.0
    intrusion = 0.0
    collided = False
    fault, fault_with, fault_at = "", "", None
    fault_rc: float | None = None
    fault_pc: float | None = None

    for i, s in enumerate(samples):
        dt = (samples[i + 1].t - s.t) if i + 1 < len(samples) else 0.0
        dt = max(0.0, dt)

        if s.t - t0 >= settle_s and _speed(s.robot_vel) < FREEZE_SPEED:
            freeze += dt

        intruding = False
        for name, (p, v, r) in s.peds.items():
            if wanted is not None and name not in wanted:
                continue

            clearance = math.dist(s.robot, p) - (s.robot_radius + r)
            if min_clear is None or clearance < min_clear:
                min_clear, clear_with, clear_at = clearance, name, s.t - t0
            if clearance <= 0.0:
                if not collided:
                    dx, dy = p[0] - s.robot[0], p[1] - s.robot[1]
                    norm = math.hypot(dx, dy) or 1.0
                    ux, uy = dx / norm, dy / norm
                    rc = s.robot_vel[0] * ux + s.robot_vel[1] * uy
                    pc = -(v[0] * ux + v[1] * uy)
                    r_moving, p_moving = rc >= FREEZE_SPEED, pc >= FREEZE_SPEED
                    fault = "mutual" if r_moving and p_moving else "robot" if r_moving else "pedestrian" if p_moving else "static"
                    fault_with, fault_at, fault_rc, fault_pc = name, s.t - t0, rc, pc
                collided = True
            if clearance < PERSONAL_SPACE:
                intruding = True

            ttc = time_to_collision(s.robot, s.robot_vel, s.robot_radius, p, v, r)
            if ttc is not None and (min_ttc is None or ttc < min_ttc):
                min_ttc, ttc_with, ttc_at = ttc, name, s.t - t0

        if intruding:
            intrusion += dt

    pet: float | None = None
    pet_with = ""
    for agent, point in encounters:
        gap = post_encroachment_time(samples, point, agent)
        if gap is not None and (pet is None or gap < pet):
            pet, pet_with = gap, agent
    return Criticality(
        min_ttc_s=min_ttc,
        min_ttc_with=ttc_with,
        min_ttc_at=ttc_at,
        min_clearance_m=min_clear,
        min_clearance_with=clear_with,
        min_clearance_at=clear_at,
        pet_s=pet,
        pet_with=pet_with,
        freeze_duration_s=freeze,
        intrusion_time_s=intrusion,
        collided=collided,
        collision_fault=fault,
        collision_with=fault_with,
        collision_at=fault_at,
        collision_robot_closing=fault_rc,
        collision_ped_closing=fault_pc,
        samples=len(samples),
        duration_s=samples[-1].t - t0,
    )


def formation_hold(samples: Sequence[Sample], members: Sequence[str], *, radius: float = 1.5, centre: Point | None = None) -> dict[str, Any]:
    """When the members first stood within `radius` of the centre (their centroid without one), how long they held and when they broke."""
    wanted = {str(m) for m in members}
    out: dict[str, Any] = {"members": sorted(wanted), "seen": False, "formed_at": None, "held_s": None, "broke_at": None, "arrived": 0}
    if not wanted or not samples:
        return out
    t0 = samples[0].t
    ref: Point | None = centre
    formed_at: float | None = None
    held_until: float | None = None
    for sample in samples:
        present = {n: v[0] for n, v in sample.peds.items() if n in wanted}
        if not present:
            continue
        out["seen"] = True
        if ref is not None:
            out["arrived"] = max(out["arrived"], sum(1 for p in present.values() if math.dist(p, ref) <= radius))
        if len(present) < len(wanted) and formed_at is None:
            continue
        if ref is None:
            ref = (sum(p[0] for p in present.values()) / len(present), sum(p[1] for p in present.values()) / len(present))
        together = all(math.dist(p, ref) <= radius for p in present.values())
        if formed_at is None:
            if together:
                formed_at = sample.t - t0
                held_until = formed_at
            continue
        if not together:
            out["broke_at"] = sample.t - t0
            break
        held_until = sample.t - t0
    if formed_at is not None:
        out["formed_at"] = formed_at
        out["held_s"] = (held_until or formed_at) - formed_at
    return out
