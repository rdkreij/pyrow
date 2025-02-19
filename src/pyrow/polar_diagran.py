import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

import physical_model
import shapes


def plot_polar_diagram(
    boat_properties: physical_model.BoatProperties,
    range_air_perp: Tuple[float, float] = (-10, 10),
    range_air_par: Tuple[float, float] = (-10, 10),
    n: int = 50,
):
    """Plots a polar diagram for boat velocity based on varying air velocities."""
    v_air_perp_vals = np.linspace(*range_air_perp, n)
    v_air_par_vals = np.linspace(*range_air_par, n)
    v_air_perp_mat, v_air_par_mat = np.meshgrid(v_air_perp_vals, v_air_par_vals)

    vperp_mat_boat = np.empty([n, n])
    vpar_mat_boat = np.empty([n, n])

    orientation = np.pi / 2
    V_water = np.array([0, 0])
    row = True

    for i in range(n):
        for j in range(n):
            v_air_perp, v_air_par = v_air_perp_mat[j, i], v_air_par_mat[j, i]
            V_air = np.array([v_air_perp, v_air_par])
            V_boat = physical_model.solve_boat_velocity(
                boat_properties, V_water, V_air, orientation, row
            )
            vperp_mat_boat[j, i], vpar_mat_boat[j, i] = V_boat

    fig, ax = plt.subplots()
    plot_lim_val = np.max(np.abs(vpar_mat_boat))
    im = ax.imshow(
        vpar_mat_boat,
        extent=[*range_air_perp, *range_air_par],
        origin="lower",
        vmin=-plot_lim_val,
        vmax=plot_lim_val,
        cmap="seismic",
    )
    plt.colorbar(im, label="Boat Velocity (m/s)")

    levels = np.arange(
        np.floor(np.min(vpar_mat_boat)), np.ceil(np.max(vpar_mat_boat)), 0.5
    )
    contour = plt.contour(
        vpar_mat_boat, levels, extent=[*range_air_perp, *range_air_par], colors=["k"]
    )
    plt.clabel(contour, inline=True, fontsize=10, fmt="%.1f")

    # if boat_properties.wind_correction is not None:
    #     v_perp_boat, v_par_boat = shapes.calculate_zone_coords(**boat_properties.wind_correction)
    #     plt.plot(v_perp_boat, v_par_boat,'k:')

    ax.set_xlabel("Wind Perpendicular Velocity (m/s)")
    ax.set_ylabel("Wind Parallel Velocity (m/s)")
    ax.set_title("Boat Velocity Polar Diagram")

    plt.show()
    return fig, ax


if __name__ == "__main__":
    boat_properties = physical_model.BoatProperties(
        area_anchor=6,
        area_air_front=4,
        area_water_front=1,
        area_water_side=7,
        area_air_side=8,
        drag_coefficient_air=0.4,
        drag_coefficient_water=20,
        speed_perfect=1,
        wind_correction={"correction_par": 3, "window_par": 7, "window_perp": 5},
    )

    plot_polar_diagram(boat_properties)
