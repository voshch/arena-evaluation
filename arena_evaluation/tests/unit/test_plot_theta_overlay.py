import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from arena_evaluation.presentation.plot_types.trajectory import THETA_OVERLAY_ROBOT_RADIUS_M, TrajectoryRenderer
from arena_evaluation.processing.path import static_environment
from arena_evaluation.processing.path.theta_star import StaticGrid


def test_theta_overlay_draws_the_route_planned_on_the_static_grid(monkeypatch) -> None:
    lethal = np.zeros((40, 40), dtype=bool)
    lethal[0:30, 20] = True
    grid = StaticGrid(origin_xy=(0.0, 0.0), resolution_m=0.1, lethal=lethal)
    radii: list[float] = []

    def grid_for_map(map_name, run_dir, robot_radius_m):
        radii.append(robot_radius_m)
        return grid

    monkeypatch.setattr(static_environment, "grid_for_map", grid_for_map)
    frame = pd.DataFrame({"map": ["wall_room"], "start": [(1.03, 1.0)], "goal": [(3.07, 1.0)]})
    ax = Figure().add_subplot()

    TrajectoryRenderer._draw_theta_star(ax, frame, None, None)

    (line,) = [drawn for drawn in ax.get_lines() if drawn.get_label() == "Theta* Reference"]
    xs = np.asarray(line.get_xdata())
    ys = np.asarray(line.get_ydata())
    assert (xs[0], ys[0]) == (1.03, 1.0)
    assert (xs[-1], ys[-1]) == (3.07, 1.0)
    assert ys.max() >= 3.0
    assert radii == [THETA_OVERLAY_ROBOT_RADIUS_M]


def test_theta_overlay_draws_nothing_for_an_unnamed_map(monkeypatch) -> None:
    def grid_for_map(map_name, run_dir, robot_radius_m):
        raise AssertionError("an unnamed map has no grid to plan on")

    monkeypatch.setattr(static_environment, "grid_for_map", grid_for_map)
    frame = pd.DataFrame({"map": [None], "start": [(1.0, 1.0)], "goal": [(3.0, 1.0)]})
    ax = Figure().add_subplot()

    TrajectoryRenderer._draw_theta_star(ax, frame, None, None)

    assert ax.get_lines() == []
