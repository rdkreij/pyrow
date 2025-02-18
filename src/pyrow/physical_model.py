from dataclasses import dataclass, field

import numpy as np
import scipy
import scipy.optimize
from geopy.distance import geodesic

@dataclass
class BoatProperties:
    area_anchor: float
    area_water_front: float
    area_water_side: float
    area_air_front: float
    area_air_side: float
    drag_coefficient_air: float
    drag_coefficient_water: float
    speed_perfect: float
    force_row: float = field(init=False)

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
    lon: float
    lat: float
    time: np.datetime64
    orientation: float = 0.0
    u_boat: float = 0.0
    v_boat: float = 0.0
    u_air: float = 0.0
    v_air: float = 0.0
    u_water: float = 0.0
    v_water: float = 0.0
    row: bool = True
    anchor: bool = False

    @property
    def heading_boat(self) -> float:
        return np.arctan2(self.v_boat, self.u_boat)

    @property
    def heading_air(self) -> float:
        return np.arctan2(self.v_air, self.u_air)

    @property
    def heading_water(self) -> float:
        return np.arctan2(self.v_water, self.u_water)


def calculate_row_force_perfect_conditions(
    area_air_front: float,
    area_water_front: float,
    drag_coefficient_air: float,
    drag_coefficient_water: float,
    speed_perfect: float,
) -> float:
    drag_air = 0.5 * area_air_front * drag_coefficient_air * speed_perfect**2
    drag_water = 0.5 * area_water_front * drag_coefficient_water * speed_perfect**2
    return drag_air + drag_water


def calculate_effective_area(
    area_front: float, area_side: float, angle: float
) -> float:
    return area_front * np.cos(angle) ** 2 + area_side * np.sin(angle) ** 2


def calculate_drag_air(
    boat_properties: BoatProperties,
    u_air: float,
    v_air: float,
    u_boat: float,
    v_boat: float,
    orientation: float,
) -> tuple[float, float]:
    u_relative = u_air - u_boat
    v_relative = v_air - v_boat
    speed_relative = np.sqrt(u_relative**2 + v_relative**2)
    heading_relative = np.arctan2(v_relative, u_relative)
    heading_reference = heading_relative - orientation
    effective_area = calculate_effective_area(
        boat_properties.area_air_front, boat_properties.area_air_side, heading_reference
    )
    drag_force = (
        0.5 * boat_properties.drag_coefficient_air * speed_relative**2 * effective_area
    )
    drag_parallel = drag_force * np.cos(heading_reference)
    drag_air_x = drag_parallel * np.cos(orientation)
    drag_air_y = drag_parallel * np.sin(orientation)
    return drag_air_x, drag_air_y


def calculate_drag_water_no_currents(
    boat_properties: BoatProperties,
    u_boat: float,
    v_boat: float,
    orientation: float,
) -> tuple[float, float]:
    speed = np.sqrt(u_boat**2 + v_boat**2)
    u_relative = -u_boat
    v_relative = -v_boat
    heading_relative = np.arctan2(v_relative, u_relative)
    heading_reference = heading_relative - orientation
    effective_area = calculate_effective_area(
        boat_properties.area_water_front,
        boat_properties.area_water_side,
        heading_reference,
    )
    drag_force = (
        0.5 * boat_properties.drag_coefficient_water * speed**2 * effective_area
    )
    drag_water_x = drag_force * np.cos(heading_relative)
    drag_water_y = drag_force * np.sin(heading_relative)
    return drag_water_x, drag_water_y


def calculate_row_force(
    boat_properties: BoatProperties, orientation: float, row: bool
) -> tuple[float, float]:
    force_row_x = boat_properties.force_row * np.cos(orientation) * int(row)
    force_row_y = boat_properties.force_row * np.sin(orientation) * int(row)
    return force_row_x, force_row_y


def calculate_total_force_no_currents(
    velocity_boat: float,
    u_air: float,
    v_air: float,
    orientation: float,
    row: bool,
    boat_properties: BoatProperties,
) -> list[float, float]:
    u_boat, v_boat = velocity_boat
    drag_air_x, drag_air_y = calculate_drag_air(
        boat_properties, u_air, v_air, u_boat, v_boat, orientation
    )
    drag_water_x, drag_water_y = calculate_drag_water_no_currents(
        boat_properties, u_boat, v_boat, orientation
    )
    force_row_x, force_row_y = calculate_row_force(boat_properties, orientation, row)
    total_force_x = force_row_x + drag_air_x + drag_water_x
    total_force_y = force_row_y + drag_air_y + drag_water_y
    total_force = [total_force_x, total_force_y]
    return total_force


def calculate_boat_velocity(
    boat_properties: BoatProperties, boat_state: BoatState
) -> tuple[float, float]:
    velocity_boat_relative_guess = [
        boat_state.u_water
        + boat_properties.speed_perfect * np.cos(boat_state.orientation),
        boat_state.v_water
        + boat_properties.speed_perfect * np.sin(boat_state.orientation),
    ]
    u_air_relative = boat_state.u_air - boat_state.u_water
    v_air_relative = boat_state.v_air - boat_state.v_water
    args = (
        u_air_relative,
        v_air_relative,
        boat_state.orientation,
        boat_state.row,
        boat_properties,
    )
    velocity_boat_relative_solution = scipy.optimize.root(
        calculate_total_force_no_currents,
        velocity_boat_relative_guess,
        args=args,
        tol=1e-4,
    )
    if not velocity_boat_relative_solution.success:
        raise ValueError(
            f"Optimization failed: {velocity_boat_relative_solution.message}"
        )
    u_boat_relative, v_boat_relative = velocity_boat_relative_solution.x
    u_boat = u_boat_relative + boat_state.u_water
    v_boat = v_boat_relative + boat_state.v_water
    return u_boat, v_boat


def displace_lon_lat_geopy(lon, lat, x, y):
    start = (lat, lon)
    new_point = geodesic(meters=y).destination(start, 0)  # North
    new_lat, new_lon = new_point.latitude, new_point.longitude
    new_point = geodesic(meters=x).destination((new_lat, new_lon), 90)  # East
    return new_point.latitude, new_point.longitude


def move_boat(boat_state: BoatState, time_step: float) -> BoatState:
    x = boat_state.u_boat * time_step
    y = boat_state.v_boat * time_step
    new_lat, new_lon = displace_lon_lat_geopy(boat_state.lon, boat_state.lat, x, y)
    new_time = boat_state.time + np.timedelta64(int(time_step), "s")
    return BoatState(lon=new_lon, lat=new_lat, time=new_time)


if __name__ == "__main__":
    boat_state = BoatState(
        lon=10,
        lat=20,
        orientation=0,
        time=np.datetime64("now"),
        u_boat=0,
        v_boat=0,
        u_air=10,
        v_air=0,
        u_water=0,
        v_water=0,
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
    )
    print(boat_properties)

    row_force_x, row_force_y = calculate_row_force(
        boat_properties, boat_state.orientation, boat_state.row
    )
    print(f"{row_force_x=:.2f}, {row_force_y=:.2f}")

    effective_area_air = calculate_effective_area(
        boat_properties.area_air_front,
        boat_properties.area_air_side,
        boat_state.orientation,
    )
    print(f"{effective_area_air=:.2f}")

    drag_air_x, drag_air_y = calculate_drag_air(
        boat_properties,
        boat_state.u_air,
        boat_state.v_air,
        boat_state.u_boat,
        boat_state.v_boat,
        boat_state.orientation,
    )
    print(f"{drag_air_x=:.2f}, {drag_air_y=:.2f}")

    drag_water_x, drag_water_y = calculate_drag_water_no_currents(
        boat_properties, boat_state.u_boat, boat_state.v_boat, boat_state.orientation
    )
    print(f"{drag_water_x=:.2f}, {drag_water_y=:.2f}")

    total_force_x, total_force_y = calculate_total_force_no_currents(
        (boat_state.u_boat, boat_state.v_boat),
        boat_state.u_air,
        boat_state.v_air,
        boat_state.orientation,
        boat_state.row,
        boat_properties,
    )
    print(f"{total_force_x=:.2f}, {total_force_y=:.2f}")

    u_boat, v_boat = calculate_boat_velocity(boat_properties, boat_state)
    print(f"{u_boat=:.2f}, {v_boat=:.2f}")

    boat_state.u_boat = u_boat
    boat_state.v_boat = v_boat

    new_boat_state = move_boat(boat_state, 3600 * 24)
    print(new_boat_state)
