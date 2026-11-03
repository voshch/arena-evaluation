from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping

if typing.TYPE_CHECKING:
    from arena_runtime.env.task_layer import TaskModes


def legacy_modes(*, robots: str | None, obstacles: str | None, modules: Iterable[str] | None) -> TaskModes:
    """`modes` of a record made before modes: the tm_* fields it carried, an absent field leaves its axis out."""
    from arena_runtime.env.task_layer import TM_AXES

    given = dict(zip(TM_AXES, (robots, obstacles, modules), strict=True))
    return {axis: [name for name in ([value] if isinstance(value, str) else value) if name] for axis, value in given.items() if value is not None}


def legacy_params(modes: TaskModes, robots_params: Mapping[str, typing.Any], obstacles_params: Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    """Full-name `params` of a record made before modes: a leaf param of the robots or obstacles mode `m` was `task.<m>.<leaf>`."""
    from arena_runtime.env.task_layer import TM_AXES

    robots_axis, obstacles_axis, _ = TM_AXES
    return {f"task.{mode}.{leaf}": value for axis, leaves in ((robots_axis, robots_params), (obstacles_axis, obstacles_params)) for mode in modes.get(axis, [])[:1] for leaf, value in leaves.items()}


def flatten_params(nested: Mapping[str, typing.Any], prefix: str = "") -> dict[str, typing.Any]:
    """Dotted leaf names of a nested params mapping."""
    flat: dict[str, typing.Any] = {}
    for key, value in nested.items():
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def read_metadata(data: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Run metadata with the tm_* fields and nested leaf params of files written before modes mapped to `modes` and `params`."""
    from arena_runtime.env.task_layer import TM_AXES

    data = dict(data)
    robots, obstacles, modules = (data.pop(f"tm_{axis}", None) for axis in TM_AXES)
    robots_params = flatten_params(data.pop("robots_params", None) or {})
    obstacles_params = flatten_params(data.pop("obstacles_params", None) or {})
    data["modes"] = data.get("modes") or legacy_modes(robots=robots, obstacles=obstacles, modules=modules)
    data["params"] = data.get("params") or legacy_params(data["modes"], robots_params, obstacles_params)
    return data
