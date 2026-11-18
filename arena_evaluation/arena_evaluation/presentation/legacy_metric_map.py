"""Built-in legacy metric map used when no external map file is configured.

The overhaul's config roots admit only the five registered bundle kinds (`configs/README.md`),
so the map ships with the converter; an operator can still point `--metric-map` at a YAML
file with the same `renamed:`/`removed:` shape (and `load_metric_map` prefers one found in a
config root at `legacy/legacy_metric_map.yaml`, so it can move there once that kind exists).
"""

from __future__ import annotations

from typing import Final

__all__ = ["LEGACY_METRIC_MAP"]

LEGACY_METRIC_MAP: Final[dict[str, dict[str, str]]] = {
    "renamed": {
        "spl": "spl_geodesic",
        "path_length": "path_length_m",
        "total_energy_wh": "energy_total_wh",
        "energy_wh": "energy_total_wh",
        "acceleration_max": "max_acceleration_mps2",
        "jerk_max": "jerk_max_mps3",
        "velocity_mean": "mean_velocity_mps",
        "total_contacts": "contact_count",
        "time_to_goal_s": "time_to_goal",
    },
    "removed": {
        "social_force_mean": (
            "D-01/PR2: stored social-force scalars were dropped; social disturbance is "
            "reported through the directional collision index (ci_mean/ci_max) and "
            "pedestrian deflection."
        ),
        "social_force_max": (
            "D-01/PR2: stored social-force scalars were dropped; see social_force_mean."
        ),
        "social_force_ped_mean": (
            "D-01/PR2: stored social-force scalars were dropped; see social_force_mean."
        ),
        "social_force_ped_max": (
            "D-01/PR2: stored social-force scalars were dropped; see social_force_mean."
        ),
        "total_sound_energy_joules": (
            "No equivalent registry metric: acoustic exposure is published as "
            "sound_exposure_dose_pa2s / the acoustic_* scalars; confirm the intended "
            "replacement before porting this plot (TODO in the migration report)."
        ),
    },
}
