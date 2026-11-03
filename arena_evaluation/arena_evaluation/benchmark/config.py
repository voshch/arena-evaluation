from __future__ import annotations

import hashlib
import itertools
import json
import re
import typing

import attrs

if typing.TYPE_CHECKING:
    from arena_runtime.env.task_layer import TaskLayer, TaskModes

_DUR_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(ms|s|m|h)?')

_DEFAULT_TIMEOUT_S = 60.0


def _parse_duration(s: str) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    total = 0.0
    for n, unit in _DUR_RE.findall(s):
        total += float(n) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "": 1}[unit]
    if total == 0:
        raise ValueError(f"unparseable duration: {s!r}")
    return total


def _yamlable(v: object) -> object:
    if isinstance(v, (str, int, float, bool)):
        return v
    return repr(v)


_STAGE_OWNED_KEYS = frozenset({"world", "robot", "run_seed", "task_layer"})
_STAGE_OWNED_PREFIXES = ("task.", "record.")

_SUITE_METRIC_KEYS = frozenset({"max_collisions", "barn"})

_LEGACY_MODE_PREFIX = "tm_"

_IMPLICIT_SEED_EXCLUDES = ("timeout", "timeout_peds", "optim")


def _layer(task_layer_key: str = "") -> TaskLayer:
    from arena_runtime.env.task_layer import task_layer

    return task_layer(task_layer_key)


def _expand_modes(obj: dict, layer: TaskLayer) -> tuple[dict, dict]:
    """obj with its validated `modes` selection, and the same selection under legacy `tm_<axis>` keys in the input's key order."""
    out: dict = {}
    legacy: dict = {}
    modes: dict[str, str] = {}
    for key, value in obj.items():
        if key == "modes":
            if not isinstance(value, dict):
                raise ValueError(f"invalid modes type: {type(value)}")
            pairs = [(str(axis), name) for axis, name in value.items()]
        elif key.startswith(_LEGACY_MODE_PREFIX):
            pairs = [(key.removeprefix(_LEGACY_MODE_PREFIX), value)]
        else:
            out[key] = value
            legacy[key] = value
            continue
        for axis, name in pairs:
            if not isinstance(name, str):
                raise ValueError(f"invalid {axis} type: {type(name)}")
            if axis in modes:
                raise ValueError(f"stage sets the {axis!r} mode twice")
            given = [part.strip() for part in name.split(",") if part.strip()]
            canonical = ",".join(layer.validate_modes({axis: given})[axis])
            modes[axis] = canonical
            legacy[f"{_LEGACY_MODE_PREFIX}{axis}"] = canonical
    if modes:
        out["modes"] = modes
    return out, legacy


def stage_modes(stage: Suite.Stage, vocabulary: TaskModes) -> TaskModes:
    """The mode names the stage selects on every axis of the vocabulary, empty for an axis it leaves out."""
    return {axis: [name for name in stage.modes.get(axis, "").split(",") if name] for axis in vocabulary}


class Suite(typing.NamedTuple):
    @classmethod
    def parse(cls, name: str, obj: dict) -> Suite:
        task_layer_key = str(obj.get("task_layer") or "")
        layer = _layer(task_layer_key)
        return cls(
            name=name,
            stages=[cls.Stage.parse(stage, layer) for stage in obj["stages"]],
            launch_args=cls._parse_launch(obj.get("launch") or {}),
            references=bool(obj.get("references", False)),
            metrics=cls._parse_metrics(obj.get("metrics") or {}),
            task_layer=task_layer_key,
        )

    @staticmethod
    def _parse_launch(obj: dict) -> dict[str, str]:
        launch = {str(k): str(v) for k, v in obj.items()}
        owned = [k for k in launch if k in _STAGE_OWNED_KEYS or k.startswith(_STAGE_OWNED_PREFIXES)]
        if owned:
            raise ValueError(f"suite launch: {owned} are set per stage, not suite-wide")
        return launch

    @staticmethod
    def _parse_metrics(obj: dict) -> dict[str, typing.Any]:
        unknown = sorted(set(obj) - _SUITE_METRIC_KEYS)
        if unknown:
            raise ValueError(f"suite metrics: unknown keys {unknown}, known: {sorted(_SUITE_METRIC_KEYS)}")
        metrics: dict[str, typing.Any] = {}
        if "max_collisions" in obj:
            v = int(obj["max_collisions"])
            if v < 1:
                raise ValueError(f"suite metrics: max_collisions must be >= 1, got {v}")
            metrics["max_collisions"] = v
        if "barn" in obj:
            from arena_evaluation.processing.metrics.performance.barn_metrics import parse_reference

            try:
                metrics["barn"] = parse_reference(obj["barn"])
            except ValueError as e:
                raise ValueError(f"suite metrics: {e}") from None
        return metrics

    class Index(int):
        pass

    class Stage(typing.NamedTuple):
        name: str
        episodes: int
        robot: str
        map: str
        modes: dict[str, str]
        config: dict
        seed: int
        timeout: float
        optim: dict | None = None
        timeout_peds: float | None = None

        @classmethod
        def _make_serializable(cls, item: object) -> object:
            if isinstance(item, dict):
                return {k: cls._make_serializable(v) for k, v in item.items()}
            if isinstance(item, (list, tuple)):
                return [cls._make_serializable(i) for i in item]
            return item

        @classmethod
        def hash(cls, obj: dict) -> int:
            hashable_obj = {k: v for k, v in obj.items() if k != "config"}
            hashable_obj = cls._make_serializable(hashable_obj)
            try:
                return 0x7FFFFFFF & int.from_bytes(
                    hashlib.sha1(json.dumps(hashable_obj).encode()).digest()[-4:],
                    byteorder="big",
                )
            except Exception:
                return 0

        @classmethod
        def parse(cls, obj: dict, layer: TaskLayer | None = None) -> Suite.Stage:
            obj, legacy = _expand_modes(obj, _layer() if layer is None else layer)
            obj.setdefault("config", {})
            raw_timeout, raw_timeout_peds, optim = (obj.pop(key, None) for key in _IMPLICIT_SEED_EXCLUDES)
            timeout_f = _DEFAULT_TIMEOUT_S if raw_timeout is None else _parse_duration(str(raw_timeout))
            timeout_peds_f = _parse_duration(str(raw_timeout_peds)) if raw_timeout_peds is not None else None
            modes = obj.pop("modes", {})
            if "seed" not in obj:
                obj["seed"] = cls.hash({key: value for key, value in legacy.items() if key not in _IMPLICIT_SEED_EXCLUDES})
            return cls(timeout=timeout_f, optim=optim, timeout_peds=timeout_peds_f, modes=modes, **obj)

    name: str
    stages: list[Suite.Stage]
    launch_args: dict[str, str] = {}
    references: bool = False
    metrics: dict[str, typing.Any] = {}
    task_layer: str = ""

    @property
    def min_index(self) -> Suite.Index:
        return self.Index()

    @property
    def max_index(self) -> Suite.Index:
        return self.Index(len(self.stages) - 1)

    def config(self, index: Suite.Index) -> Suite.Stage:
        return self.stages[index]


def robot_with_parts(robot: str, parts: typing.Mapping[str, str]) -> str:
    """`robot:=` value whose every entry carries `parts` as bracket items, in place of its own items of the same key."""
    if not parts:
        return robot
    from arena_runtime.env.robot_manager.robots_manager import split_robot_arg  # noqa: PLC0415

    entries = []
    for entry in split_robot_arg(robot):
        model, _, body = entry.partition("[")
        kept = [item for item in split_robot_arg(body.rpartition("]")[0]) if item.partition("=")[0].strip() not in parts]
        entries.append(f"{model.strip()}[{','.join([*kept, *(f'{k}={v}' for k, v in parts.items())])}]")
    return ",".join(entries)


def _parse_parts(name: str, obj: object) -> dict[str, str]:
    if obj is None:
        return {}
    if not isinstance(obj, dict) or not all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in obj.values()):
        raise ValueError(f"contestant {name!r}: parts must map a part key to one value, e.g. {{camera: vln_rgbd}}, got {obj!r}")
    return {str(k): str(v) for k, v in obj.items()}


@attrs.frozen
class Contestant:
    name: str
    args: dict[str, typing.Any] = attrs.field(factory=dict)
    parts: dict[str, str] = attrs.field(factory=dict)


@attrs.frozen
class Contest:
    name: str
    description: str | None
    contestants: list[Contestant]

    class Index(int):
        pass

    # Keep Contestant accessible as Contest.Contestant for backward compat with runner imports.
    Contestant = Contestant

    @property
    def min_index(self) -> Contest.Index:
        return self.Index()

    @property
    def max_index(self) -> Contest.Index:
        return self.Index(len(self.contestants) - 1)

    def config(self, index: int) -> Contestant:
        return self.contestants[index]

    @classmethod
    def parse(cls, name: str, obj: list | dict) -> Contest:
        if isinstance(obj, list):
            return cls._parse_list(name, obj)
        if isinstance(obj, dict):
            if "contestants" in obj and isinstance(obj["contestants"], list):
                contest_name = str(obj.get("name") or name)
                description = obj.get("description")
                sub = cls._parse_list(contest_name, obj["contestants"])
                return cls(name=contest_name, description=description, contestants=sub.contestants)
            return cls._parse_sweep(name, obj)
        raise ValueError(f"contest must be list or dict, got {type(obj).__name__}")

    @classmethod
    def _parse_list(cls, name: str, items: list[dict]) -> Contest:
        contestants = []
        for item in items:
            if "name" not in item:
                raise ValueError(f"contestant in list-form requires 'name': {item!r}")
            entry_name = item["name"]
            args = {k: v for k, v in item.items() if k not in ("name", "parts")}
            contestants.append(Contestant(name=entry_name, args=args, parts=_parse_parts(entry_name, item.get("parts"))))
        cls._reject_duplicate_names(contestants)
        return cls(name=name, description=None, contestants=contestants)

    @classmethod
    def _parse_sweep(cls, name: str, spec: dict) -> Contest:
        spec = dict(spec)
        description = spec.pop("description", None)
        prefix = spec.pop("name", None)
        robot_parts = _parse_parts(prefix or name, spec.pop("parts", None))

        axes: list[tuple[str, str | None, list]] = []
        consts: dict[str, typing.Any] = {}
        cap_templates: dict[str, dict] = {}

        for k, v in spec.items():
            if isinstance(v, list):
                axes.append((k, None, v))
            elif isinstance(v, dict):
                inner_consts: dict[str, typing.Any] = {}
                for ik, iv in v.items():
                    if isinstance(iv, list):
                        axes.append((k, ik, iv))
                    else:
                        inner_consts[ik] = iv
                cap_templates[k] = inner_consts
            else:
                consts[k] = v

        if axes:
            combos = list(itertools.product(*[vs for _, _, vs in axes]))
        else:
            combos = [()]

        varying_idx = [i for i, (_, _, vs) in enumerate(axes) if len({_yamlable(v) for v in vs}) > 1]

        contestants = []
        for combo in combos:
            args: dict[str, typing.Any] = dict(consts)
            cap_dicts: dict[str, dict] = {k: dict(v) for k, v in cap_templates.items()}

            for (outer, inner, _), value in zip(axes, combo, strict=True):
                if inner is None:
                    args[outer] = value
                else:
                    cap_dicts.setdefault(outer, {})[inner] = value

            for cap_key, cap_dict in cap_dicts.items():
                args[cap_key] = cap_dict

            if varying_idx:
                parts = [str(combo[i]) for i in varying_idx]
                derived = "-".join(parts)
            else:
                derived = "single"
            entry_name = f"{prefix}-{derived}" if prefix else derived
            contestants.append(Contestant(name=entry_name, args=args, parts=robot_parts))

        cls._reject_duplicate_names(contestants)
        return cls(name=name, description=description, contestants=contestants)

    @staticmethod
    def _reject_duplicate_names(contestants: list[Contestant]) -> None:
        seen: set[str] = set()
        for c in contestants:
            if c.name in seen:
                raise ValueError(f"duplicate contestant name: {c.name!r}")
            seen.add(c.name)
