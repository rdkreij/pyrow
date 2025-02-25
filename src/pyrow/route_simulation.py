from dataclasses import dataclass

import numpy as np
import xarray as xr

import pyrow.load_dataset as load_dataset
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
        algorithm_options: dict | None = None,
        interpolation: str = "nearest",
        interpolation_options: dict | None = None,
        filter: str | None = None,
        filter_options: dict | None = None,
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
        self.algortihm_options = algorithm_options
        self.interpolation = interpolation
        self.interpolation_options = interpolation_options
        self.filter = filter
        self.filter_options = filter_options
        self.finish_time_step_scale = finish_time_step_scale

        if self.filter_options is None:
            self.filter_options = {}

        if self.interpolation_options is None:
            self.interpolation_options = {}

        if self.algortihm_options is None:
            self.algortihm_options = {}

        if self.finish_time_step_scale is not None:
            self.finish_zone = (
                self.finish_time_step_scale
                * self.boat_properties.speed_perfect
                * self.time_step
            )
        else:
            self.finish_zone = None

        self.store_boat_info = self.run_simulation()
        self.fastest_route_info = find_fastest_route_info(
            self.store_boat_info, self.target_coords
        )
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
            time_step_instance = self.calculate_time_step_flex(target_distance)
            if self.algortihm == "single":
                list_boat_info = self.update_boat_info_algorithm_single(
                    list_boat_info, time_step_instance
                )
            if self.algortihm == "multi":
                list_boat_info = self.update_boat_info_algorithm_single(
                    list_boat_info, time_step_instance, **self.algortihm_options
                )
            store_boat_info.append(list_boat_info)

            if self.filter == "radial":
                store_boat_info = self.spatial_filter_radial_list(
                    store_boat_info, **self.filter_options
                )

            list_coords = [info.state.coords for info in list_boat_info]
            target_distance = calculate_target_distance_list(
                list_coords, self.target_coords
            )
            i += 1
        return store_boat_info

    def update_boat_info_algorithm_single(
        self, list_boat_info: list[BoatSimInfo], time_step_instance: float
    ):
        new_list_boat_info = []
        id = 0
        for info in list_boat_info:
            info.id = id
            info.state.orientation = spatial_operations.bearing_east_rad(
                info.state.coords, self.target_coords
            )
            info.state.V_air = wheather_interpolation.get_velocity_from_dataset(
                self.ds_air,
                info.state.time,
                info.state.coords,
                self.interpolation,
                self.interpolation_options,
            )
            info.state.V_water = wheather_interpolation.get_velocity_from_dataset(
                self.ds_water,
                info.state.time,
                info.state.coords,
                self.interpolation,
                self.interpolation_options,
            )
            info.state.row = True
            info.state.V_boat = physical_model.solve_boat_velocity(
                self.boat_properties,
                info.state.V_water,
                info.state.V_air,
                info.state.orientation,
                row=info.state.row,
            )
            new_boat_state = physical_model.move_boat(info.state, time_step_instance)
            new_info = BoatSimInfo(new_boat_state, parent_id=id)
            id += 1
            new_list_boat_info.append(new_info)
        return new_list_boat_info

    def update_boat_info_algorithm_multi(
        self, list_boat_info: list[BoatSimInfo], time_step_instance: float
    ):
        # TODO To be implemented, also add algorithm options (see in class)
        pass

    def spatial_filter_radial_list(
        self, store_boat_info: list[BoatSimInfo]
    ) -> list[BoatSimInfo]:
        # TODO To be implemented, also add filter options (see in class)
        pass


def find_fastest_route_info(
    store_boat_info: list[list[BoatSimInfo]], target_coords=tuple[float]
):
    idx_time_last = len(store_boat_info) - 1
    info_list_end = store_boat_info[idx_time_last]
    list_coords = [info.state.coords for info in info_list_end]
    target_distance = calculate_target_distance_list(list_coords, target_coords)
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
    import pyrow.config as config
    import pyrow.route_plot as route_plot

    boat_properties = config.BOAT_PROPERTIES
    ds_air = load_dataset.load_air_dataset(
        "data/CERSAT-GLO-REP_WIND_L4-OBS_FULL_TIME_SERIE_1648775425193.nc"
    )
    ds_water = load_dataset.load_water_dataset("data/nemo_monthly_mean.nc")

    route_simulation = RouteSimulation(
        boat_properties,
        start_coords=config.START_COORDS["Fremantle"],
        target_coords=config.TARGET_COORDS["North Madagascar"],
        min_distance_target=1e4,
        start_time=np.datetime64("2009-06-01T12:00:00"),
        time_step=3600,
        ds_air=ds_air,
        ds_water=ds_water,
        algorithm="single",
        interpolation="nearest",
        interpolation_options={"ignore_nan": True},
        finish_time_step_scale=4,
    )

    fastest_route_info = route_simulation.fastest_route_info
    coords_list = [info.state.coords for info in fastest_route_info]
    time = np.array([info.state.time for info in fastest_route_info])
    print((time[-1] - time[0]) // np.timedelta64(1, "D"))
    lon = [i[0] for i in coords_list]
    lat = [i[1] for i in coords_list]

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    fig, ax = route_plot.plot_cartopy_globe(
        show_image=True,
        focus_coords=(75, -10),
        extent=[20, 130, -40, +0],
        grid_color="r",
        background_color="k",
    )
    ax.plot(lon, lat, "r-", transform=ccrs.PlateCarree())

    # fig, ax = plt.subplots()
    # ax.plot(lon, lat)
    plt.show()
