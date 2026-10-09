"""Base scenarios for a world: zone roles, the population, the robot's route and the zone screen."""

from __future__ import annotations

from .occupancy import world_statics
from .populate import (
    MIN_ROUTE_CLEARANCE,
    PROFILES,
    ROUTE_STOPS,
    TALKERS,
    basis_for,
    build,
    clearance,
    robot_zones,
    route_points,
    work_zones,
)
from .route import MAX_LEGS, MIN_EPISODE_S, ROBOT_SPEED_MPS, ROUTE_MARGIN, robot_for, route_length, route_seconds
from .zone_screen import Verdict, ref_safe_zones, screen_zones
from .zones import Role, classify, classify_all, zones_by_role

__all__ = [
    "MIN_EPISODE_S",
    "MIN_ROUTE_CLEARANCE",
    "PROFILES",
    "ROUTE_STOPS",
    "TALKERS",
    "ROBOT_SPEED_MPS",
    "Role",
    "Verdict",
    "basis_for",
    "build",
    "classify",
    "classify_all",
    "clearance",
    "ref_safe_zones",
    "robot_for",
    "route_length",
    "ROUTE_MARGIN",
    "MAX_LEGS",
    "robot_zones",
    "route_points",
    "route_seconds",
    "screen_zones",
    "work_zones",
    "world_statics",
    "zones_by_role",
]
