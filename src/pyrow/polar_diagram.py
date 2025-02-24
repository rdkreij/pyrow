import copy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import pyrow.physical_model as physical_model
import pyrow.shapes as shapes


def calculate_mat_polar_diagram(
    boat_properties: physical_model.BoatProperties,
    range_medium_perp: Tuple[float, float] = (-10, 10),
    range_medium_par: Tuple[float, float] = (-10, 10),
    n: int = 50,
    row: bool = True,
    medium: str = "air",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v_medium_perp_vals = np.linspace(*range_medium_perp, n)
    v_medium_par_vals = np.linspace(*range_medium_par, n)
    v_medium_perp_mat, v_medium_par_mat = np.meshgrid(
        v_medium_perp_vals, v_medium_par_vals
    )

    vperp_mat_boat = np.empty([n, n])
    vpar_mat_boat = np.empty([n, n])

    orientation = np.pi / 2

    for i in range(n):
        for j in range(n):
            v_medium_perp, v_medium_par = (
                v_medium_perp_mat[j, i],
                v_medium_par_mat[j, i],
            )
            V_medium = np.array([v_medium_perp, v_medium_par])
            if medium == "water":
                V_water = V_medium
                V_air = np.array([0, 0])
            elif medium == "air":
                V_water = np.array([0, 0])
                V_air = V_medium
            V_boat = physical_model.solve_boat_velocity(
                boat_properties, V_water, V_air, orientation, row
            )
            vperp_mat_boat[j, i], vpar_mat_boat[j, i] = V_boat
    return v_medium_perp_mat, v_medium_par_mat, vperp_mat_boat, vpar_mat_boat


def calculate_boat_velocity_difference_row_rest(
    boat_properties: physical_model.BoatProperties,
    range_medium_perp: Tuple[float, float] = (-10, 10),
    range_medium_par: Tuple[float, float] = (-10, 10),
    n: int = 50,
    medium: str = "air",
):
    boat_properties = copy.deepcopy(boat_properties)
    boat_properties.wind_correction = None

    v_medium_perp_mat, v_medium_par_mat, vperp_mat_boat_row, vpar_mat_boat_row = (
        calculate_mat_polar_diagram(
            boat_properties, range_medium_perp, range_medium_par, n, True, medium
        )
    )
    _, _, vperp_mat_boat_rest, vpar_mat_boat_rest = calculate_mat_polar_diagram(
        boat_properties, range_medium_perp, range_medium_par, n, False, medium
    )

    vperp_mat_boat_diff = vperp_mat_boat_row - vperp_mat_boat_rest
    vpar_mat_boat_diff = vpar_mat_boat_row - vpar_mat_boat_rest
    return v_medium_perp_mat, v_medium_par_mat, vperp_mat_boat_diff, vpar_mat_boat_diff


def plot_polar_diagram_matplotlib(
    boat_properties: physical_model.BoatProperties,
    v_medium_perp_mat: np.ndarray,
    v_medium_par_mat: np.ndarray,
    vpar_mat_boat: np.ndarray,
    contour_step=0.5,
    vmin=None,
    vmax=None,
    cmap="seismic",
    xlabel="perpendicular velocity (m/s)",
    ylabel="parallel velocity (m/s)",
    title="boat velocity polar diagram",
    colorbar_label="boat velocity (m/s)",
    draw_row_zone=False,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    plot_lim_val = np.max(np.abs(vpar_mat_boat))
    if vmin is None:
        vmin = -plot_lim_val
    if vmax is None:
        vmax = plot_lim_val

    im = ax.imshow(
        vpar_mat_boat,
        extent=[
            v_medium_perp_mat.min(),
            v_medium_perp_mat.max(),
            v_medium_par_mat.min(),
            v_medium_par_mat.max(),
        ],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    plt.colorbar(im, label=colorbar_label)

    levels = np.arange(
        np.floor(np.min(vpar_mat_boat)), np.ceil(np.max(vpar_mat_boat)), contour_step
    )
    contour = ax.contour(
        v_medium_perp_mat,
        v_medium_par_mat,
        vpar_mat_boat,
        levels,
        colors=["k"],
    )
    plt.clabel(contour, inline=True, fontsize=10, fmt="%.1f")

    if draw_row_zone:
        if boat_properties.wind_correction is not None:
            v_perp_boat, v_par_boat = shapes.calculate_zone_coords(
                **boat_properties.wind_correction
            )
            plt.plot(v_perp_boat, v_par_boat, "r-")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax


def plot_polar_diagram(
    boat_properties: physical_model.BoatProperties,
    range_medium_perp: Tuple[float, float] = (-10, 10),
    range_medium_par: Tuple[float, float] = (-10, 10),
    n: int = 50,
    medium: str = "air",
    boat_direction: str = "parallel",
    plot_config: dict | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    v_medium_perp_mat, v_medium_par_mat, vperp_mat_boat, vpar_mat_boat = (
        calculate_mat_polar_diagram(
            boat_properties,
            range_medium_perp=range_medium_perp,
            range_medium_par=range_medium_par,
            n=n,
            medium=medium,
        )
    )
    if plot_config is None:
        plot_config = {}

    if boat_direction == "parallel":
        vdir_mat_boat = vpar_mat_boat
    elif boat_direction == "perpendicular":
        vdir_mat_boat = vperp_mat_boat

    if "title" not in plot_config:
        plot_config["title"] = f"{boat_direction} velocity - {medium} polar diagram"
    if "xlabel" not in plot_config:
        plot_config["xlabel"] = f"perpendicular {medium} velocity (m/s)"
    if "ylabel" not in plot_config:
        plot_config["ylabel"] = f"parallel {medium} velocity (m/s)"
    if "colorbar_label" not in plot_config:
        plot_config["colorbar_label"] = f"boat {boat_direction} velocity (m/s)"

    fig, ax = plot_polar_diagram_matplotlib(
        boat_properties,
        v_medium_perp_mat,
        v_medium_par_mat,
        vdir_mat_boat,
        **plot_config,
    )
    return fig, ax


def plot_polar_diagram_row_rest(
    boat_properties: physical_model.BoatProperties,
    range_medium_perp: Tuple[float, float] = (-10, 10),
    range_medium_par: Tuple[float, float] = (-10, 10),
    n: int = 50,
    medium: str = "air",
    boat_direction: str = "parallel",
    plot_config: dict | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    v_medium_perp_mat, v_medium_par_mat, vperp_mat_boat, vpar_mat_boat = (
        calculate_boat_velocity_difference_row_rest(
            boat_properties,
            range_medium_perp=range_medium_perp,
            range_medium_par=range_medium_par,
            n=n,
            medium=medium,
        )
    )
    if plot_config is None:
        plot_config = {}

    if boat_direction == "parallel":
        vdir_mat_boat = vpar_mat_boat
    elif boat_direction == "perpendicular":
        vdir_mat_boat = vperp_mat_boat

    if "title" not in plot_config:
        plot_config["title"] = (
            f"{boat_direction} velocity - {medium} polar diagram row - rest"
        )
    if "xlabel" not in plot_config:
        plot_config["xlabel"] = f"perpendicular {medium} velocity (m/s)"
    if "ylabel" not in plot_config:
        plot_config["ylabel"] = f"parallel {medium} velocity (m/s)"
    if "colorbar_label" not in plot_config:
        plot_config["colorbar_label"] = f"boat {boat_direction} velocity (m/s)"
    if "cmap" not in plot_config:
        plot_config["cmap"] = "viridis"

    fig, ax = plot_polar_diagram_matplotlib(
        boat_properties,
        v_medium_perp_mat,
        v_medium_par_mat,
        vdir_mat_boat,
        **plot_config,
    )
    return fig, ax


if __name__ == "__main__":
    boat_properties = physical_model.BoatProperties(
        area_anchor=6,
        area_water_front=0.329,
        area_water_side=1.87 + 0.196 + 0.184,
        area_air_front=1.89,
        area_air_side=5.81,
        drag_coefficient_air=0.2,
        drag_coefficient_water=0.0075,
        speed_perfect=2,
        wind_correction={"correction_par": 2, "window_par": 6, "window_perp": 4},
    )
    fig, ax = plot_polar_diagram(
        boat_properties, medium="air", boat_direction="parallel"
    )
    fig, ax = plot_polar_diagram(
        boat_properties, medium="air", boat_direction="perpendicular"
    )
    fig, ax = plot_polar_diagram(
        boat_properties, medium="water", boat_direction="parallel"
    )
    fig, ax = plot_polar_diagram(
        boat_properties, medium="water", boat_direction="perpendicular"
    )
    fig, ax = plot_polar_diagram_row_rest(
        boat_properties, medium="air", boat_direction="parallel"
    )
    fig, ax = plot_polar_diagram_row_rest(
        boat_properties, medium="water", boat_direction="parallel"
    )
    plt.show()
