from dataclasses import dataclass, field

import numpy as np
import scipy
import scipy.optimize

import pyrow.spatial_operations as spatial_operations


@dataclass
class BoatProperties:
    area_anchor: float
    area_water_front: float
    area_water_side: float
    area_air_front: float
    area_air_side: float
    area_anchor: float
    drag_coefficient_air: float
    drag_coefficient_water: float
    speed_perfect: float
    force_row: float = field(init=False)
    wind_correction: None | dict = None
    # min_anchor_speed: None | float = None

    def __post_init__(self):
        self.force_row = calculate_row_force_perfect_conditions(
            self.area_air_front,
            self.area_water_front,
            self.drag_coefficient_air,
            self.drag_coefficient_water,
            self.speed_perfect,
        )


@dataclass
class BoatState:
    coords: tuple[float]
    time: np.datetime64
    orientation: float = 0.0
    V_boat: np.ndarray = field(default_factory=lambda: np.zeros(2))
    V_water: np.ndarray = field(default_factory=lambda: np.zeros(2))
    V_air: np.ndarray = field(default_factory=lambda: np.zeros(2))
    row: bool = True
    anchor: bool = False

    # @property
    # def heading_boat(self) -> float:
    #     return np.arctan2(self.v_boat, self.u_boat)

    # @property
    # def heading_air(self) -> float:
    #     return np.arctan2(self.v_air, self.u_air)

    # @property
    # def heading_water(self) -> float:
    #     return np.arctan2(self.v_water, self.u_water)


def calculate_row_force_perfect_conditions(
    area_air_front: float,
    area_water_front: float,
    drag_coefficient_air: float,
    drag_coefficient_water: float,
    speed_perfect: float,
) -> float:
    # TODO use drag functions below for this!
    drag_air = 0.5 * 1.225 * area_air_front * drag_coefficient_air * speed_perfect**2
    drag_water = (
        0.5 * 1025 * area_water_front * drag_coefficient_water * speed_perfect**2
    )
    return drag_air + drag_water


def calculate_effective_area(
    area_front: float, area_side: float, angle: float
) -> float:
    return area_front * np.cos(angle) ** 2 + area_side * np.sin(angle) ** 2


def calculate_drag_air(
    boat_properties: BoatProperties,
    V_air: np.ndarray,
    V_boat: np.ndarray,
    orientation: float,
) -> np.ndarray:
    V_relative = V_air - V_boat
    speed_relative = np.linalg.norm(V_relative)
    heading_relative = np.arctan2(V_relative[1], V_relative[0])
    heading_reference = heading_relative - orientation
    effective_area = calculate_effective_area(
        boat_properties.area_air_front, boat_properties.area_air_side, heading_reference
    )
    # drag_force = (
    #     0.5 * 1.225 * boat_properties.drag_coefficient_air * speed_relative**2 * effective_area
    # )
    # drag_parallel = drag_force * np.cos(heading_reference)

    # return drag_parallel * np.array([np.cos(orientation), np.sin(orientation)])
    return (
        0.5
        * 1.225
        * boat_properties.drag_coefficient_air
        * speed_relative**2
        * effective_area
    ) * np.array([np.cos(heading_relative), np.sin(heading_relative)])


def calculate_drag_water(
    boat_properties: BoatProperties,
    V_water: np.ndarray,
    V_boat: np.ndarray,
    orientation: float,
    force_anchor: bool = False,
) -> np.ndarray:
    V_relative = V_water - V_boat
    speed_relative = np.linalg.norm(V_relative)
    heading_relative = np.arctan2(V_relative[1], V_relative[0])
    heading_reference = heading_relative - orientation
    effective_area = calculate_effective_area(
        boat_properties.area_water_front,
        boat_properties.area_water_side,
        heading_reference,
    )

    # Drag due to boat drag
    drag_force = (
        0.5
        * 1025
        * boat_properties.drag_coefficient_water
        * speed_relative**2
        * effective_area
    ) * np.array([np.cos(heading_relative), np.sin(heading_relative)])

    # # Drag due to anchor drag
    # if boat_properties.min_anchor_speed is not None:
    #     boat_par_speed, _ = decompose_vector(V_boat, orientation)
    #     if boat_par_speed < boat_properties.min_anchor_speed:
    #         force_anchor = True

    if force_anchor:
        drag_force += (
            0.5
            * 1025
            * boat_properties.drag_coefficient_water
            * speed_relative**2
            * boat_properties.area_anchor
        ) * np.array([np.cos(heading_relative), np.sin(heading_relative)])

    return drag_force


def is_wind_in_row_zone(
    v_par_air, v_perp_air, correction_par, window_par, window_perp
) -> bool:
    return ((v_par_air - correction_par) ** 2) / window_par**2 + (
        (v_perp_air) ** 2
    ) / window_perp**2 <= 1


def decompose_vector(vec, angle):
    if np.allclose(vec, 0):
        parallel_mag = 0
        perpendicular_mag = 0
        return parallel_mag, perpendicular_mag
    else:
        parallel_mag = np.dot(vec, [np.cos(angle), np.sin(angle)])
        perpendicular_mag = np.linalg.norm(vec) * np.sqrt(
            1 - (parallel_mag / np.linalg.norm(vec)) ** 2
        )
        return parallel_mag, perpendicular_mag


def calculate_row_force(
    boat_properties: BoatProperties, V_air: np.ndarray, orientation: float, row: bool
) -> np.ndarray:
    if not row:
        return np.zeros(2)

    row_in_wind = True
    if boat_properties.wind_correction is not None:
        v_par_air, v_perp_air = decompose_vector(V_air, orientation)
        row_in_wind = is_wind_in_row_zone(
            v_par_air, v_perp_air, **boat_properties.wind_correction
        )

    return (
        boat_properties.force_row * np.array([np.cos(orientation), np.sin(orientation)])
        if row_in_wind
        else np.zeros(2)
    )


def calculate_total_force(
    V_boat: np.ndarray,
    V_water: np.ndarray,
    V_air: np.ndarray,
    orientation: float,
    row: bool,
    boat_properties: BoatProperties,
    force_anchor: bool = False,
) -> np.ndarray:
    drag_air = calculate_drag_air(boat_properties, V_air, V_boat, orientation)
    drag_water = calculate_drag_water(
        boat_properties, V_water, V_boat, orientation, force_anchor
    )
    force_row = calculate_row_force(boat_properties, V_air, orientation, row)
    return force_row + drag_air + drag_water


def solve_boat_velocity(
    boat_properties: BoatProperties,
    V_water: np.ndarray,
    V_air: np.ndarray,
    orientation: float,
    row: bool,
    force_anchor: bool = False,
    tol: float = 1e-5,
) -> tuple[float, float]:
    V_boat_guess = V_water + boat_properties.speed_perfect * np.array(
        [np.cos(orientation), np.sin(orientation)]
    )

    args = (V_water, V_air, orientation, row, boat_properties, force_anchor)

    methods = ["hybr", "lm", "diagbroyden"]
    for method in methods:
        V_boat_solution = scipy.optimize.root(
            calculate_total_force,
            V_boat_guess,
            args=args,
            tol=tol,
            method=method,
        )
        if V_boat_solution.success:
            break
    if not V_boat_solution.success:
        raise ValueError("Optimization failed to find boat velocity")
    return V_boat_solution.x


def move_boat(boat_state: BoatState, time_step: float) -> BoatState:
    x = boat_state.V_boat[0] * time_step
    y = boat_state.V_boat[1] * time_step
    new_coords = spatial_operations.displace_coordinates_pyproj(boat_state.coords, x, y)
    new_time = boat_state.time + np.timedelta64(int(time_step), "s")
    return BoatState(coords=new_coords, time=new_time)


if __name__ == "__main__":
    boat_state = BoatState(
        coords=(10, 20),
        orientation=0,
        time=np.datetime64("now"),
        V_boat=np.array([0, 0]),
        V_air=np.array([0, 0]),
        V_water=np.array([0, 0]),
        row=True,
    )
    print(boat_state)

    boat_properties = BoatProperties(
        area_anchor=6,
        area_air_front=4,
        area_water_front=1,
        area_water_side=7,
        area_air_side=8,
        drag_coefficient_air=0.4,
        drag_coefficient_water=20,
        speed_perfect=2,
        wind_correction={"correction_par": 1, "window_par": 3, "window_perp": 1},
    )
    print(boat_properties)

    row_force = calculate_row_force(
        boat_properties, boat_state.V_air, boat_state.orientation, boat_state.row
    )
    print(f"{row_force[0]=:.2f}, {row_force[1]=:.2f}")

    effective_area_air = calculate_effective_area(
        boat_properties.area_air_front,
        boat_properties.area_air_side,
        boat_state.orientation,
    )
    print(f"{effective_area_air=:.2f}")

    drag_air = calculate_drag_air(
        boat_properties,
        boat_state.V_air,
        boat_state.V_boat,
        boat_state.orientation,
    )
    print(f"{drag_air[0]=:.2f}, {drag_air[1]=:.2f}")

    drag_water = calculate_drag_water(
        boat_properties, boat_state.V_water, boat_state.V_boat, boat_state.orientation
    )
    print(f"{drag_water[0]=:.2f}, {drag_water[1]=:.2f}")

    total_force = calculate_total_force(
        boat_state.V_boat,
        boat_state.V_water,
        boat_state.V_air,
        boat_state.orientation,
        boat_state.row,
        boat_properties,
    )
    print(f"{total_force[0]=:.2f}, {total_force[1]=:.2f}")

    V_boat = solve_boat_velocity(
        boat_properties,
        boat_state.V_water,
        boat_state.V_air,
        boat_state.orientation,
        boat_state.row,
    )
    print(f"{V_boat[0]=:.2f}, {V_boat[1]=:.2f}")

    boat_state.V_boat = V_boat

    new_boat_state = move_boat(boat_state, 3600 * 24)
    print(new_boat_state)
