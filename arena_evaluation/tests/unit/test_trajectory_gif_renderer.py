"""Trajectory renderer: the animated GIF export restored behind the generate_gifs option."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.presentation.static_renderer import StaticRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def _spec(*, generate_gifs: bool) -> PlotSpec:
    options = {"generate_gifs": True} if generate_gifs else {}
    return PlotSpec(
        id="trajectory_gif",
        type="trajectory",
        title="Trajectories",
        data={"source": "v_episode_summary"},
        channels=PlotChannels(x="x", y="y"),
        options=options,
    )


def _frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for robot, ys in (("robot_a", 0.0), ("robot_b", 2.0)):
        for step in range(6):
            rows.append(
                {
                    "robot_id": robot,
                    "t_ns": step * 100_000_000,
                    "x": float(step),
                    "y": ys + 0.1 * step,
                }
            )
    return pl.DataFrame(rows)


def test_trajectory_gif_is_written_when_requested(tmp_path: pathlib.Path) -> None:
    png = tmp_path / "trajectory.png"

    StaticRenderer().render(_spec(generate_gifs=True), _frame(), png)

    gif = tmp_path / "trajectory.gif"
    assert png.exists()
    assert gif.exists()
    assert gif.stat().st_size > 1024
    from PIL import Image

    with Image.open(gif) as image:
        assert image.format == "GIF"
        assert getattr(image, "n_frames", 1) > 1


