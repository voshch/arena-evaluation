"""Cast helpers: naming, meshes, per-agent entries, and the derived profile files written beside a scenario (`types/<name>.yaml`)."""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs
import yaml

from .profiles import PROFILES, Profile, ProfileError, derive, derived_name, profile

Point = tuple[float, float]

TYPES_DIR = "types"


def agent_type_of(agent: Mapping[str, Any]) -> str:
    block = agent.get("agent")
    if isinstance(block, str):
        return block
    if isinstance(block, Mapping):
        return str(block.get("agent_type", "adult"))
    return "adult"


def unique_names(doc: Mapping[str, Any], prefix: str, n: int) -> list[str]:
    taken = {str(a.get("name")) for a in (doc.get("dynamic") or []) if isinstance(a, Mapping)}
    out, i = [], 1
    while len(out) < n:
        name = f"{prefix}_{i}"
        if name not in taken:
            out.append(name)
            taken.add(name)
        i += 1
    return out


def mesh_for(prof: Profile, rng: random.Random) -> str:
    return rng.choice(list(prof.meshes))


def entry(name: str, prof: Profile, pose: Point | str, waypoints: Sequence[Point | str], *, speed: float, mode: str, agent_type: str, rng: random.Random, yaw: float = 0.0) -> dict[str, Any]:
    def pt(p: Point | str) -> str | list[float]:
        return p if isinstance(p, str) else [round(p[0], 2), round(p[1], 2), 0.0]

    return {
        "name": name,
        "model": mesh_for(prof, rng),
        "pose": pose if isinstance(pose, str) else [round(pose[0], 2), round(pose[1], 2), round(yaw, 3)],
        "agent": {"agent_type": agent_type, "desired_velocity": round(speed, 2)},
        "waypoints": [pt(p) for p in waypoints],
        "waypoint_mode": mode,
    }


@attrs.define
class TypeFiles:
    base_dir: Path
    docs: dict[str, dict[str, Any]] = attrs.Factory(dict)
    #: Files copied verbatim from the base scenario directory (a `./talker.yaml`).
    copies: dict[str, Path] = attrs.Factory(dict)

    def base_doc(self, ref: str) -> dict[str, Any]:
        """The document a derivation starts from: `{extends: builtin}` or a scenario-local file."""
        if ref.startswith("./"):
            path = self.base_dir / ref[2:]
            if not path.is_file():
                raise ProfileError(f"{ref}: not found beside the base scenario ({path})")
            self.copies.setdefault(ref, path)
            return yaml.safe_load(path.read_text()) or {}
        return {"extends": ref}

    def derived(self, ref: str, params: Mapping[str, str], *, prof: Profile | None = None) -> str:
        """The ref of `ref` with `params` pinned, writing the document once per distinct combination."""
        if not params:
            return ref
        name = derived_name(ref, params)
        out = f"./{TYPES_DIR}/{name}.yaml"
        if out not in self.docs:
            base = self.base_doc(ref)
            base.pop("name", None)
            self.docs[out] = derive(base, params, name=name, prof=prof)
        return out

    def write(self, scenario_dir: Path) -> list[Path]:
        written = []
        for ref, doc in self.docs.items():
            path = scenario_dir / ref[2:]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))
            written.append(path)
        for ref, src in self.copies.items():
            dst = scenario_dir / ref[2:]
            if not dst.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src.read_text())
                written.append(dst)
        return written


def profile_of_agent(agent: Mapping[str, Any]) -> Profile | None:
    """Which library profile an existing agent most resembles: its agent type, refined by its
    mesh - an adult in a medical uniform is a nurse."""
    ref = agent_type_of(agent)
    stem = ref.rsplit("/", 1)[-1].removesuffix(".yaml").split("__", 1)[0]
    try:
        base = profile(stem)
    except ProfileError:
        return None
    mesh = str(agent.get("model") or "")
    for staff in ("nurse", "doctor"):
        prof = PROFILES[staff]
        if base.name == prof.base and mesh in prof.meshes:
            return prof
    return base


def clone(agent: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(agent))
