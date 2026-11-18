"""Criticality of a sampled episode against chosen pedestrians: clearance, TTC, PET, freezes, collisions, formations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Final

import attrs
from arena_simulation_setup.shared.pathing import Point

FREEZE_SPEED: Final[float] = 0.05
#: Personal-space edge clearance, metres (centre distance minus both body radii), the proxemic boundary.
PERSONAL_SPACE: Final[float] = 0.5
SETTLE_S: Final[float] = 2.0
#: How near a body must pass a designed encounter point to count as passing it, centre-to-centre metres.
ENCOUNTER_NEAR_CENTRE_M: Final[float] = 1.5
FORMATION_RADIUS_M: Final[float] = 1.5


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
    collision_with: str = ""
    collision_at: float | None = None
    collision_robot_closing: float | None = None
    collision_ped_closing: float | None = None
    samples: int = 0
    duration_s: float = 0.0

    @property
    def observed(self) -> bool:
        return self.samples > 0 and self.min_clearance_m is not None


def time_to_collision(
    p_a: Point, v_a: Point, r_a: float, p_b: Point, v_b: Point, r_b: float
) -> float | None:
    """Seconds until two discs on constant velocities touch, 0.0 when they already do, None when they never will."""
    dpx, dpy = p_b[0] - p_a[0], p_b[1] - p_a[1]
    dvx, dvy = v_b[0] - v_a[0], v_b[1] - v_a[1]
    r = r_a + r_b

    c = dpx * dpx + dpy * dpy - r * r
    if c <= 0.0:
        return 0.0

    a = dvx * dvx + dvy * dvy
    if a < 1e-12:
        return None
    b = 2.0 * (dpx * dvx + dpy * dvy)

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


def post_encroachment_time(
    samples: Sequence[Sample], encounter: Point, agent: str
) -> float | None:
    """Time gap between the robot and `agent` passing `encounter`, None when either never comes within ENCOUNTER_NEAR_CENTRE_M of it."""

    def crossing_time(track: Iterable[tuple[float, Point]]) -> float | None:
        best_t, best_d = None, math.inf
        for t, p in track:
            d = math.dist(p, encounter)
            if d < best_d:
                best_t, best_d = t, d
        return best_t if best_d <= ENCOUNTER_NEAR_CENTRE_M else None

    t_robot = crossing_time((s.t, s.robot) for s in samples)
    t_agent = crossing_time((s.t, s.peds[agent][0]) for s in samples if agent in s.peds)
    if t_robot is None or t_agent is None:
        return None
    return abs(t_agent - t_robot)


def _min_pet(
    samples: Sequence[Sample], encounters: Sequence[tuple[str, Point]]
) -> tuple[float | None, str]:
    pet: float | None = None
    pet_with = ""
    for agent, point in encounters:
        gap = post_encroachment_time(samples, point, agent)
        if gap is not None and (pet is None or gap < pet):
            pet, pet_with = gap, agent
    return pet, pet_with


def _first_collision(
    s: Sample, p: Point, v: Point, t0: float, name: str
) -> tuple[str, float, float, float]:
    """The agent touched, the time of contact, and the robot and pedestrian closing speeds along the line of sight."""
    dx, dy = p[0] - s.robot[0], p[1] - s.robot[1]
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    rc = s.robot_vel[0] * ux + s.robot_vel[1] * uy
    pc = -(v[0] * ux + v[1] * uy)
    return name, s.t - t0, rc, pc


def _sample_ped_stats(
    s: Sample, t0: float, wanted: set[str] | None
) -> tuple[
    tuple[float, str, float] | None,
    tuple[float, str, float] | None,
    tuple[str, float, float, float] | None,
    bool,
]:
    best_clear: tuple[float, str, float] | None = None
    best_ttc: tuple[float, str, float] | None = None
    first_collision: tuple[str, float, float, float] | None = None
    intruding = False
    for name, (p, v, r) in s.peds.items():
        if wanted is not None and name not in wanted:
            continue
        clearance = math.dist(s.robot, p) - (s.robot_radius + r)
        if best_clear is None or clearance < best_clear[0]:
            best_clear = (clearance, name, s.t - t0)
        if clearance <= 0.0 and first_collision is None:
            first_collision = _first_collision(s, p, v, t0, name)
        if clearance < PERSONAL_SPACE:
            intruding = True
        ttc = time_to_collision(s.robot, s.robot_vel, s.robot_radius, p, v, r)
        if ttc is not None and (best_ttc is None or ttc < best_ttc[0]):
            best_ttc = (ttc, name, s.t - t0)
    return best_clear, best_ttc, first_collision, intruding


def score(
    samples: Sequence[Sample],
    *,
    only: Sequence[str] | None = None,
    encounters: Sequence[tuple[str, Point]] = (),
    settle_s: float = SETTLE_S,
) -> Criticality:
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
    collision_with = ""
    collision_at: float | None = None
    collision_rc: float | None = None
    collision_pc: float | None = None

    for i, s in enumerate(samples):
        dt = max(0.0, (samples[i + 1].t - s.t) if i + 1 < len(samples) else 0.0)
        if s.t - t0 >= settle_s and _speed(s.robot_vel) < FREEZE_SPEED:
            freeze += dt

        sample_clear, sample_ttc, sample_collision, intruding = _sample_ped_stats(
            s, t0, wanted
        )
        if sample_clear is not None and (
            min_clear is None or sample_clear[0] < min_clear
        ):
            min_clear, clear_with, clear_at = sample_clear
        if sample_ttc is not None and (min_ttc is None or sample_ttc[0] < min_ttc):
            min_ttc, ttc_with, ttc_at = sample_ttc
        if sample_collision is not None and not collided:
            collision_with, collision_at, collision_rc, collision_pc = sample_collision
            collided = True
        if intruding:
            intrusion += dt

    pet, pet_with = _min_pet(samples, encounters)
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
        collision_with=collision_with,
        collision_at=collision_at,
        collision_robot_closing=collision_rc,
        collision_ped_closing=collision_pc,
        samples=len(samples),
        duration_s=samples[-1].t - t0,
    )


@attrs.frozen
class FormationHold:
    """How the recorded members of a formation came together around its center, how long they held and when they broke."""

    members: tuple[str, ...] = ()
    absent: tuple[str, ...] = ()
    seen: bool = False
    formed_at: float | None = None
    held_s: float | None = None
    broke_at: float | None = None
    arrived: int = 0


def _centroid(points: Sequence[Point]) -> Point:
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def formation_hold(
    samples: Sequence[Sample],
    members: Sequence[str],
    *,
    radius: float = FORMATION_RADIUS_M,
    center: Point | None = None,
    form_at: float | None = None,
) -> FormationHold:
    """When the recorded members stood within `radius` of the center (else their centroid) from `form_at` on, and how long."""
    wanted = tuple(sorted({str(m) for m in members}))
    recorded = {name for sample in samples for name in sample.peds}
    scored = [name for name in wanted if name in recorded]
    absent = tuple(name for name in wanted if name not in recorded)
    if not scored:
        return FormationHold(members=wanted, absent=absent)
    t0 = samples[0].t
    arrived = 0
    formed_at: float | None = None
    held_until: float | None = None
    broke_at: float | None = None
    for sample in samples:
        if form_at is not None and sample.t - t0 < form_at:
            continue
        present = [sample.peds[name][0] for name in scored if name in sample.peds]
        if formed_at is not None and len(present) < len(scored):
            broke_at = sample.t - t0
            break
        if not present:
            continue
        ref = center if center is not None else _centroid(present)
        within = sum(1 for p in present if math.dist(p, ref) <= radius)
        arrived = max(arrived, within)
        together = within == len(scored)
        if formed_at is None:
            if together:
                formed_at = held_until = sample.t - t0
            continue
        if not together:
            broke_at = sample.t - t0
            break
        held_until = sample.t - t0
    held_s = None if formed_at is None or held_until is None else held_until - formed_at
    return FormationHold(
        members=wanted,
        absent=absent,
        seen=True,
        formed_at=formed_at,
        held_s=held_s,
        broke_at=broke_at,
        arrived=arrived,
    )
