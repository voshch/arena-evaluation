"""World furniture footprints, read offline from `world.yaml` and the assets' `annotation.yaml`.

`map.png` is rasterised from walls alone, so furniture is not in the occupancy grid and comes from here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..formats import first_level

#: Footprint radius, metres, of an asset with no annotation. Chairs and small props dominate the
#: shipped furniture, so this is deliberately not generous.
DEFAULT_FOOTPRINT = 0.4


def asset_annotation(model: str, assets_root: Path) -> dict[str, Any] | None:
    """A model's `annotation.yaml`, found the way the runtime resolves the model: the world names
    `Hospital/SM_ReceptionDesk_01a`, the asset lives under `Hospital/Object/SM_ReceptionDesk_01a`."""
    parts = [p for p in model.split("/") if p]
    if not parts:
        return None
    candidates = [assets_root.joinpath(*parts) / "annotation.yaml"]
    if len(parts) == 2:
        candidates.append(assets_root / parts[0] / "Object" / parts[1] / "annotation.yaml")
    candidates.extend(sorted(assets_root.glob(f"*/*/{parts[-1]}/annotation.yaml")))
    for ann in candidates:
        if ann.is_file():
            try:
                doc = yaml.safe_load(ann.read_text()) or {}
            except Exception:
                return None
            return doc if isinstance(doc, dict) else None
    return None


Box = tuple[float, float, float, float]


def asset_box(model: str, assets_root: Path) -> Box | None:
    """Model-frame `(x_lo, x_hi, y_lo, y_hi)` from the annotation's `bounding_box`. Not centred on
    the model's origin as a rule: the reception desk's runs 0..2.42 m in x."""
    ann = asset_annotation(model, assets_root)
    try:
        box = (ann or {})["bounding_box"]
        (x_lo, x_hi), (y_lo, y_hi) = box[0], box[1]
        return float(x_lo), float(x_hi), float(y_lo), float(y_hi)
    except Exception:
        return None


def asset_footprint(model: str, assets_root: Path) -> float:
    """Circumscribing radius of the model's x-y box around its origin; over-blocks where only a radius fits."""
    box = asset_box(model, assets_root)
    if box is None:
        return DEFAULT_FOOTPRINT
    x_lo, x_hi, y_lo, y_hi = box
    return math.hypot(max(abs(x_lo), abs(x_hi)), max(abs(y_lo), abs(y_hi)))


def _yaw(orientation: Mapping[str, object] | None) -> float:
    try:
        w, x, y, z = (float(orientation[k]) for k in ("w", "x", "y", "z"))
    except Exception:
        return 0.0
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def static_polygon(obj: dict[str, Any]) -> list[tuple[float, float]] | None:
    """World-frame footprint rectangle (`bbox` rotated by `yaw` about `pose`), the segments the
    human simulator is given. None without a box; the caller falls back to the radius."""
    pose = obj.get("pose")
    box = obj.get("bbox")
    if not isinstance(pose, (list, tuple)) or len(pose) < 2 or not box or len(box) != 4:
        return None
    x_lo, x_hi, y_lo, y_hi = (float(v) for v in box)
    px, py = float(pose[0]), float(pose[1])
    c, s = math.cos(float(obj.get("yaw") or 0.0)), math.sin(float(obj.get("yaw") or 0.0))
    return [(px + c * lx - s * ly, py + s * lx + c * ly) for lx, ly in ((x_lo, y_lo), (x_lo, y_hi), (x_hi, y_hi), (x_hi, y_lo))]


def find_assets(world_root: Path) -> Path:
    """The first `_assets/default` walking up from the world (`src/Arena/_assets/default` for a shipped world)."""
    # a fixed two levels up found none, so every static was a 0.4 m disc (2026-08-29)
    for parent in (world_root, *world_root.parents):
        candidate = parent / "_assets" / "default"
        if candidate.is_dir():
            return candidate
    return world_root.parents[1] / "_assets" / "default"


def world_statics(world_root: str | Path, assets_root: Path | None = None) -> list[dict[str, Any]]:
    """Every static entity a world declares, with a footprint `radius` and, when annotated, a `bbox`."""
    root = Path(world_root)
    assets = assets_root or find_assets(root)
    try:
        doc = first_level(root) or {}
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for zone in doc.get("zones") or []:
        if not isinstance(zone, dict):
            continue
        for ent in (zone.get("entities") or {}).get("static") or []:
            pos = ((ent or {}).get("pose") or {}).get("position") or {}
            if "x" not in pos or "y" not in pos:
                continue
            model = str(ent.get("model", ""))
            box = asset_box(model, assets) if model else None
            item: dict[str, Any] = {
                "name": ent.get("name", "?"),
                "model": model,
                "pose": [float(pos["x"]), float(pos["y"])],
                "yaw": _yaw(((ent or {}).get("pose") or {}).get("orientation")),
                "radius": asset_footprint(model, assets) if model else DEFAULT_FOOTPRINT,
            }
            if box is not None:
                item["bbox"] = list(box)
            out.append(item)
    return out
