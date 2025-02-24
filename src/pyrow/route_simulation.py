from dataclasses import dataclass

import numpy as np
import xarray as xr

import pyrow.physical_model as physical_model
import pyrow.spatial_operations as spatial_operations
import pyrow.wheather_interpolation as wheather_interpolation


@dataclass
class BoatSimInfo:
    state: physical_model.BoatState
    id: int | None = None
    parent_id: int | None = None


class RouteSimulation:
    def __init__(
        self,
        boat_properties: physical_model.BoatProperties,
        start_coords: tuple[float],
        target_coords: tuple[float],
        min_distance_target: float,
        start_time: np.datetime64,
        time_step: float,
        ds_air: xr.Dataset,
        ds_water: xr.Dataset,
        algorithm: str = "single",
        interpolation: str = "nearest",
        finish_time_step_scale: float | None = None,
    ):
        self.boat_properties = boat_properties
        self.start_coords = start_coords
        self.target_coords = target_coords
        self.min_distance_target = min_distance_target
        self.time_step = time_step
        self.start_time = start_time
        self.ds_air = ds_air
        self.ds_water = ds_water
        self.algortihm = algorithm
        self.interpolation = interpolation
        self.finish_time_step_scale = finish_time_step_scale

        if self.finish_time_step_scale is not None:
            self.finish_zone = (
                self.finish_time_step_scale
                * self.boat_properties.speed_perfect
                * self.time_step
            )
        else:
            self.finish_zone = None

        self.store_boat_info = self.run_simulation()
        self.fastest_route_info = self.find_fastest_route_info(self.store_boat_info)
        if self.interpolation != "zero":
            for ds in [self.ds_air, self.ds_water]:
                wheather_interpolation.process_dataset(
                    ds, {"time", "lon", "lat"}, {"u", "v"}
                )

    def calculate_time_step_flex(self, target_distance):
        if self.finish_time_step_scale is not None:
            if np.any(target_distance <= self.finish_zone):
                time_step_flex = np.min(target_distance) / (
                    self.boat_properties.speed_perfect * self.finish_time_step_scale
                )
                return min(time_step_flex, self.time_step)
            else:
                return self.time_step
        else:
            return self.time_step

    def run_simulation(self):
        boat_state_start = physical_model.BoatState(self.start_coords, self.start_time)
        boat_info_start = BoatSimInfo(boat_state_start, 0, None)
        list_boat_info = [boat_info_start]
        store_boat_info = [list_boat_info]

        list_coords = [boat_info.state.coords for boat_info in list_boat_info]
        target_distance = calculate_target_distance_list(
            list_coords, self.target_coords
        )
        i = 0
        while np.all(target_distance > self.min_distance_target):
            new_list_boat_info = []
            time_step_instance = self.calculate_time_step_flex(target_distance)
            print(time_step_instance)
            id = 0
            for info in list_boat_info:
                if self.algortihm == "single":
                    info.id = id
                    info.state.orientation = spatial_operations.bearing_east_rad(
                        info.state.coords, self.target_coords
                    )
                    info.state.V_air = wheather_interpolation.get_velocity_from_dataset(
                        self.ds_air,
                        info.state.time,
                        info.state.coords,
                        self.interpolation,
                    )
                    info.state.V_water = (
                        wheather_interpolation.get_velocity_from_dataset(
                            self.ds_water,
                            info.state.time,
                            info.state.coords,
                            self.interpolation,
                        )
                    )
                    info.state.row = True
                    info.state.V_boat = physical_model.solve_boat_velocity(
                        self.boat_properties,
                        info.state.V_water,
                        info.state.V_air,
                        info.state.orientation,
                        row=info.state.row,
                    )
                    new_boat_state = physical_model.move_boat(
                        info.state, time_step_instance
                    )
                    new_info = BoatSimInfo(new_boat_state, parent_id=id)

                    id += 1
                new_list_boat_info.append(new_info)
            list_boat_info = new_list_boat_info

            store_boat_info.append(list_boat_info)

            list_coords = [info.state.coords for info in list_boat_info]
            target_distance = calculate_target_distance_list(
                list_coords, self.target_coords
            )
            i += 1
        return store_boat_info

    def find_fastest_route_info(self, store_boat_info: list[list[BoatSimInfo]]):
        idx_time_last = len(store_boat_info) - 1
        info_list_end = store_boat_info[idx_time_last]
        list_coords = [info.state.coords for info in info_list_end]
        target_distance = calculate_target_distance_list(
            list_coords, self.target_coords
        )
        idx_winner = np.argmin(target_distance)
        info_winner = info_list_end[idx_winner]
        parent_id = info_winner.parent_id

        fastest_route_info = [info_winner]
        for idx_time in range(idx_time_last - 1, -1, -1):
            info_list = store_boat_info[idx_time]
            id_arr = np.array([info.id for info in info_list])
            idx_parent = np.where(id_arr == parent_id)[0][0]
            info_parent = info_list[idx_parent]
            fastest_route_info.append(info_parent)

        fastest_route_info = fastest_route_info[::-1]

        return fastest_route_info


def spatial_filter_radial_list(
    list_coords: list[tuple[float]], target_coords: tuple[float]
) -> list[tuple[float]]:
    pass


# def load_climate_velocity_list(
#     list_coords: list[tuple[float]], ds: xr.Dataset
# ) -> list[tuple[float]]:
#     pass


def calculate_target_distance_list(
    list_coords: list[tuple[float]], target_coords
) -> np.ndarray:
    return np.array(
        [
            spatial_operations.calculate_distance_till_between_points(
                coords, target_coords
            )
            for coords in list_coords
        ]
    )


if __name__ == "__main__":
    boat_properties = physical_model.BoatProperties(
        area_anchor=6,
        area_water_front=0.329,
        area_water_side=1.87 + 0.196 + 0.184,
        area_air_front=1.89,
        area_air_side=5.81,
        drag_coefficient_air=0.2,
        drag_coefficient_water=0.0075,
        speed_perfect=4,
        wind_correction={"correction_par": 2, "window_par": 6, "window_perp": 4},
    )

    route_simulation = RouteSimulation(
        boat_properties,
        start_coords=(80, 40),
        target_coords=(-80, -40),
        min_distance_target=1e3,
        start_time=np.datetime64("now"),
        time_step=3600,
        ds_air=None,
        ds_water=None,
        algorithm="single",
        interpolation="zero",
        finish_time_step_scale=4,
    )

    fastest_route_info = route_simulation.fastest_route_info
    coords_list = [info.state.coords for info in fastest_route_info]
    lon = [i[0] for i in coords_list]
    lat = [i[1] for i in coords_list]

    print(len(coords_list))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(lon, lat)
    plt.show()
