from __future__ import annotations

import hashlib
import itertools
import json
import re
import typing

import attrs
from task_generator.constants import Constants

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


_STAGE_OWNED_KEYS = frozenset({"world", "robot", "run_seed"})
_STAGE_OWNED_PREFIXES = ("task.", "record.")

_SUITE_METRIC_KEYS = frozenset({"max_collisions"})


class Suite(typing.NamedTuple):
    @classmethod
    def parse(cls, name: str, obj: dict) -> Suite:
        return cls(
            name=name,
            stages=[cls.Stage.parse(stage) for stage in obj["stages"]],
            launch_args=cls._parse_launch(obj.get("launch") or {}),
            references=bool(obj.get("references", False)),
            metrics=cls._parse_metrics(obj.get("metrics") or {}),
        )

    @staticmethod
    def _parse_launch(obj: dict) -> dict[str, str]:
        launch = {str(k): str(v) for k, v in obj.items()}
        owned = [k for k in launch if k in _STAGE_OWNED_KEYS or k.startswith(_STAGE_OWNED_PREFIXES)]
        if owned:
            raise ValueError(f"suite launch: {owned} are set per stage, not suite-wide")
        return launch

    @staticmethod
    def _parse_metrics(obj: dict) -> dict[str, int]:
        unknown = sorted(set(obj) - _SUITE_METRIC_KEYS)
        if unknown:
            raise ValueError(f"suite metrics: unknown keys {unknown}, known: {sorted(_SUITE_METRIC_KEYS)}")
        metrics: dict[str, int] = {}
        if "max_collisions" in obj:
            v = int(obj["max_collisions"])
            if v < 1:
                raise ValueError(f"suite metrics: max_collisions must be >= 1, got {v}")
            metrics["max_collisions"] = v
        return metrics

    class Index(int):
        pass

    class Stage(typing.NamedTuple):
        name: str
        episodes: int
        robot: str
        map: str
        tm_robots: Constants.TaskMode.TM_Robots
        tm_obstacles: Constants.TaskMode.TM_Obstacles
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
            if isinstance(item, (Constants.TaskMode.TM_Robots, Constants.TaskMode.TM_Obstacles)):
                return item.value
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
        def parse(cls, obj: dict) -> Suite.Stage:
            obj = dict(obj)
            if "tm_robots" in obj:
                v = obj["tm_robots"]
                if isinstance(v, str):
                    obj["tm_robots"] = Constants.TaskMode.TM_Robots[v.upper()]
                elif isinstance(v, Constants.TaskMode.TM_Robots):
                    pass
                else:
                    raise ValueError(f"invalid tm_robots type: {type(v)}")
            if "tm_obstacles" in obj:
                v = obj["tm_obstacles"]
                if isinstance(v, str):
                    obj["tm_obstacles"] = Constants.TaskMode.TM_Obstacles[v.upper()]
                elif isinstance(v, Constants.TaskMode.TM_Obstacles):
                    pass
                else:
                    raise ValueError(f"invalid tm_obstacles type: {type(v)}")
            obj.setdefault("config", {})
            raw_timeout = obj.pop("timeout", None)
            timeout_f = _DEFAULT_TIMEOUT_S if raw_timeout is None else _parse_duration(str(raw_timeout))
            raw_timeout_peds = obj.pop("timeout_peds", None)
            timeout_peds_f = _parse_duration(str(raw_timeout_peds)) if raw_timeout_peds is not None else None
            optim = obj.pop("optim", None)
            obj.setdefault("seed", cls.hash(obj))
            return cls(timeout=timeout_f, optim=optim, timeout_peds=timeout_peds_f, **obj)

    name: str
    stages: list[Suite.Stage]
    launch_args: dict[str, str] = {}
    references: bool = False
    metrics: dict[str, int] = {}

    @property
    def min_index(self) -> Suite.Index:
        return self.Index()

    @property
    def max_index(self) -> Suite.Index:
        return self.Index(len(self.stages) - 1)

    def config(self, index: Suite.Index) -> Suite.Stage:
        return self.stages[index]


@attrs.frozen
class Contestant:
    name: str
    args: dict[str, typing.Any] = attrs.field(factory=dict)


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
            args = {k: v for k, v in item.items() if k != "name"}
            contestants.append(Contestant(name=entry_name, args=args))
        cls._reject_duplicate_names(contestants)
        return cls(name=name, description=None, contestants=contestants)

    @classmethod
    def _parse_sweep(cls, name: str, spec: dict) -> Contest:
        spec = dict(spec)
        description = spec.pop("description", None)
        prefix = spec.pop("name", None)

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
            contestants.append(Contestant(name=entry_name, args=args))

        cls._reject_duplicate_names(contestants)
        return cls(name=name, description=description, contestants=contestants)

    @staticmethod
    def _reject_duplicate_names(contestants: list[Contestant]) -> None:
        seen: set[str] = set()
        for c in contestants:
            if c.name in seen:
                raise ValueError(f"duplicate contestant name: {c.name!r}")
            seen.add(c.name)
