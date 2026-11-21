from __future__ import annotations

import dataclasses
import typing

import numpy as np
import polars as pl
from arena_simulation_setup.shared.judge import Sample, WallGrid

from arena_evaluation.processing.metrics.common.alignment import (
    TRACK_ALIGN_TOLERANCE_NS,
    asof_position,
)
from arena_evaluation.processing.metrics.common.rosters import ped_aliases
from arena_evaluation.processing.metrics.common.zones import ZoneGeometry

if typing.TYPE_CHECKING:
    import shapely


def _entity_field_series(events: pl.DataFrame, entity: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    """Sorted (event times, values) for one recorded entity and field."""
    rows = events.filter((pl.col("entity") == entity) & (pl.col("field") == field)).sort("t_ns")
    return rows["t_ns"].to_numpy(), rows["current"].to_numpy()


def _entity_family_series(events: pl.DataFrame, entity: str, family: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Sorted (event times, values) of every `<family>.<member>` field of one recorded entity, keyed by member."""
    prefix = f"{family}."
    rows = events.filter((pl.col("entity") == entity) & (pl.col("field").str.starts_with(prefix))).sort("t_ns")
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (field,), group in rows.group_by(["field"], maintain_order=True):
        ordered = group.sort("t_ns")
        series[str(field).removeprefix(prefix)] = (
            ordered["t_ns"].to_numpy(),
            ordered["current"].to_numpy(),
        )
    return series


def _step_value(times: np.ndarray, values: typing.Sequence[str] | np.ndarray, t_ns: int) -> str | None:
    val = values[max(asof_position(times, t_ns), 0)]
    return None if val is None else str(val)


def value_at(times: np.ndarray, values: list[str], t_ns: int) -> str:
    """Stepwise value at `t_ns`, holding the earliest known (seed) value before it."""
    return typing.cast(str, _step_value(times, values, t_ns))


@dataclasses.dataclass(frozen=True, slots=True)
class EvalContext:
    """Recorded inputs of one robot's episode on one time axis, the ego robot being `robot`.

    The two lookup caches are dicts that fill in lazily; the context itself is never rebound, so a
    reader can share it across the metric boundary.
    """

    events: pl.DataFrame
    t_ns: np.ndarray
    pos_x: np.ndarray
    pos_y: np.ndarray
    data: pl.DataFrame
    zones_by_name: dict[str, ZoneGeometry]
    entity_roster: dict[str, str]
    ped_roster: dict[str, str]
    pose_valid: bool = True
    yaw: np.ndarray | None = None
    fleet: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = dataclasses.field(default_factory=dict)
    walls: WallGrid | None = None
    clips: list[dict[str, frozenset[str]]] | None = None
    _fields: dict[tuple[str, str], tuple[np.ndarray, np.ndarray] | None] = dataclasses.field(default_factory=dict, init=False, repr=False)
    _families: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]] | None] = dataclasses.field(default_factory=dict, init=False, repr=False)
    peds_frames: list[pl.DataFrame] = dataclasses.field(default_factory=list)
    peds_times: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def field_at(self, entity: str, field: str, t_ns: int) -> str | None:
        """Stepwise recorded value of one entity field at `t_ns`, None when the entity is unknown."""
        key = (entity, field)
        if key not in self._fields:
            recorded = self.entity_roster.get(entity)
            series = None if recorded is None else _entity_field_series(self.events, recorded, field)
            self._fields[key] = None if series is None or len(series[0]) == 0 else series
        series = self._fields[key]
        if series is None:
            return None
        return _step_value(series[0], series[1], t_ns)

    def family_at(self, entity: str, family: str, t_ns: int) -> dict[str, str] | None:
        """Stepwise recorded members of one entity field family at `t_ns`, None when the entity is unknown."""
        key = (entity, family)
        if key not in self._families:
            recorded = self.entity_roster.get(entity)
            self._families[key] = None if recorded is None else _entity_family_series(self.events, recorded, family)
        members = self._families[key]
        if members is None:
            return None
        return typing.cast(
            dict[str, str],
            {member: _step_value(times, values, t_ns) for member, (times, values) in members.items()},
        )

    def peds_at(self, i: int) -> dict[str, tuple[float, float]]:
        """Pedestrians of the latest frame at or before sample i, within the alignment tolerance."""
        if len(self.peds_times) == 0:
            return {}
        idx = asof_position(self.peds_times, int(self.t_ns[i]), TRACK_ALIGN_TOLERANCE_NS)
        if idx < 0:
            return {}
        peds: dict[str, tuple[float, float]] = {}
        frame = self.peds_frames[idx]
        for name, x, y in zip(
            frame["ped_name"].to_numpy(),
            frame["x"].to_numpy(),
            frame["y"].to_numpy(),
            strict=True,
        ):
            for alias in ped_aliases(str(name)):
                if self.ped_roster.get(alias, name) == name:
                    peds[alias] = (float(x), float(y))
        return peds

    def sample(self, i: int) -> Sample:
        """Judge sample of the ego robot at axis index i."""
        robots: dict[str, tuple[float, float, float]] = {}
        if self.pose_valid:
            robots["robot"] = (
                float(self.pos_x[i]),
                float(self.pos_y[i]),
                float(self.yaw[i]) if self.yaw is not None else 0.0,
            )
        for name, (xs, ys, yaws) in self.fleet.items():
            robots[name] = (float(xs[i]), float(ys[i]), float(yaws[i]))
        t_ns = int(self.t_ns[i])
        known = frozenset(self.ped_roster) | frozenset(self.fleet)
        clips = self.clips[i] if self.clips is not None else {}
        return Sample(
            t=t_ns / 1e9,
            robots=robots,
            peds=self.peds_at(i),
            field=lambda entity, field: self.field_at(entity, field, t_ns),
            known=known,
            clips=clips,
            walls=self.walls,
            family=lambda entity, family: self.family_at(entity, family, t_ns),
        )

    @property
    def polygons(self) -> dict[str, shapely.Polygon | shapely.MultiPolygon]:
        """Zone polygons keyed by zone name."""
        return {name: zone.polygon for name, zone in self.zones_by_name.items()}
