"""Collision and pedestrian-interaction telemetry, normalized so that every participant list is its own table."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from arena_evaluation.processing.telemetry.messages import (
    Envelope,
    env_row,
    pedestrian_id,
    robot_row,
)
from arena_evaluation.processing.telemetry.writer import TableSink

if TYPE_CHECKING:
    from arena_humansim_msgs.msg import InteractionFrame
    from arena_people_msgs.msg import AnimationSlot, AnimationState, AnimationStates
    from arena_robots_msgs.msg import CollisionEvent, CollisionEvents
    from nav2_msgs.msg import CollisionMonitorState

WALL_KIND = "wall"
STATIC_KIND = "static"
PEDESTRIAN_KIND = "pedestrian"
MEMBER_ROLE_NAMES = {0: "participant", 1: "queue"}
INTERACTION_EVENT_NAMES = {
    0: "ACTIVATED",
    1: "HOLD_ONSET",
    2: "RELEASED",
    3: "INTERRUPTED",
    4: "CANCELED",
}


def record_collision_events(
    sink: TableSink, envelope: Envelope[CollisionEvents]
) -> None:
    events = envelope.body.events
    sink.append(
        "robot_collision_events",
        robot_row(
            envelope,
            event_count=len(events),
            wall_count=_count_kind(events, WALL_KIND),
            static_count=_count_kind(events, STATIC_KIND),
            ped_count=_count_kind(events, PEDESTRIAN_KIND),
        ),
    )
    for event in events:
        x, y, z = _position(event)
        sink.append(
            "robot_collision_contacts",
            robot_row(
                envelope,
                obstacle_id=str(event.obstacle_id),
                kind=str(event.kind),
                obstacle_x=x,
                obstacle_y=y,
                obstacle_z=z,
                distance=float(event.distance),
            ),
        )


def record_collision_monitor(
    sink: TableSink, envelope: Envelope[CollisionMonitorState]
) -> None:
    monitor = envelope.body
    sink.append(
        "robot_collision_monitor",
        robot_row(
            envelope,
            action_type=int(monitor.action_type),
            polygon_name=str(monitor.polygon_name),
        ),
    )


def record_interactions(sink: TableSink, envelope: Envelope[InteractionFrame]) -> None:
    frame = envelope.body
    offsets = [int(offset) for offset in frame.member_offset]
    member_ids = [int(member) for member in frame.member_id]
    member_roles = bytes(frame.member_role)
    interaction_types = bytes(frame.interaction_type)
    outcomes = bytes(frame.outcome)
    object_ids = [str(obj) for obj in frame.object_id]
    durations = [float(dur) for dur in frame.duration]
    arrived = [bool(arr) for arr in frame.arrived]
    holding = [bool(hld) for hld in frame.holding]
    for index, interaction_id in enumerate(frame.interaction_id):
        interaction = int(interaction_id)
        sink.append(
            "pedestrian_interactions",
            env_row(
                envelope,
                interaction_id=interaction,
                interaction_type=int(interaction_types[index]),
                outcome=int(outcomes[index]),
                object_id=object_ids[index],
                duration=durations[index],
                arrived=arrived[index],
                holding=holding[index],
            ),
        )
        start, end = offsets[index], offsets[index + 1]
        for member, role in zip(
            member_ids[start:end], member_roles[start:end], strict=True
        ):
            role_name = MEMBER_ROLE_NAMES.get(role)
            if role_name is not None:
                sink.append(
                    "pedestrian_interaction_members",
                    env_row(
                        envelope,
                        interaction_id=interaction,
                        ped_id=pedestrian_id(member),
                        member_role=role_name,
                    ),
                )


def record_interaction_events(
    sink: TableSink, envelope: Envelope[InteractionFrame]
) -> None:
    for event in envelope.body.events:
        event_code = int(event.event)
        participants: list[int | None] = [
            int(participant) for participant in event.participants
        ] or [None]
        for participant in participants:
            sink.append(
                "pedestrian_interaction_events",
                env_row(
                    envelope,
                    interaction_id=int(event.interaction_id),
                    interaction_type=int(event.interaction_type),
                    event=event_code,
                    event_name=INTERACTION_EVENT_NAMES.get(event_code, str(event_code)),
                    ped_id=None if participant is None else pedestrian_id(participant),
                ),
            )


def record_animations(sink: TableSink, envelope: Envelope[AnimationStates]) -> None:
    for ped in envelope.body.peds:
        ped_id = pedestrian_id(int(ped.id))
        for slot in list(ped.slots) or [None]:
            sink.append(
                "pedestrian_animations", _animation_row(envelope, ped_id, ped, slot)
            )


def _count_kind(events: Iterable[CollisionEvent], kind: str) -> int:
    return sum(1 for event in events if event.kind == kind)


def _position(event: CollisionEvent) -> tuple[float, float, float]:
    position = event.obstacle_position
    return float(position.x), float(position.y), float(position.z)


def _animation_row(
    envelope: Envelope, ped_id: str, ped: AnimationState, slot: AnimationSlot | None
) -> dict[str, object]:
    base = {
        "ped_id": ped_id,
        "ped_name": str(ped.name),
        "base": str(ped.base),
        "base_phase": float(ped.base_phase),
    }
    if slot is None:
        return env_row(
            envelope,
            **base,
            slot="",
            clip="",
            playhead=None,
            duration=None,
            weight=None,
            loop=None,
        )
    return env_row(
        envelope,
        **base,
        slot=str(slot.slot),
        clip=str(slot.clip),
        playhead=float(slot.playhead),
        duration=float(slot.duration),
        weight=float(slot.weight),
        loop=bool(slot.loop),
    )
