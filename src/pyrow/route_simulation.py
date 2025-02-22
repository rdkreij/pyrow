import numpy as np
import xarray as xr
from geopy.distance import geodesic

import pyrow.physical_model as physical_model


class RouteSimulation:
    def __init__(
        self,
        boat_properties: physical_model.BoatProperties,
        start_coords: tuple[float],
        target_coords: tuple[float],
        distance_till_target: float,
        start_time: np.datetime64,
        time_step: float,
        ds_air: xr.Dataset,
        ds_water: xr.Dataset,
        algorithm: str = "single",
    ):
        self.boat_properties = boat_properties
        self.start_coords = start_coords
        self.target_coords = target_coords
        self.distance_till_target = distance_till_target
        self.time_step = time_step
        self.start_time = start_time
        self.ds_air = ds_air
        self.ds_water = ds_water
        self.algortihm = algorithm

        self.store_boat_state = []

    def run_simulation(self):
        start_boat_state = physical_model.BoatState(self.start_coords, self.start_time)
        list_boat_state = [start_boat_state]
        self.store_boat_state.append(list_boat_state)
        pass


def spatial_filter_radial_list(
    list_coords: list[tuple[float]], target_coords: tuple[float]
) -> list[tuple[float]]:
    pass


def load_climate_velocity_list(
    list_coords: list[tuple[float]], ds: xr.Dataset
) -> list[tuple[float]]:
    pass


def calculate_distance_till_between_points(coords_1, coords_2) -> float:
    return geodesic(coords_1, coords_2).m


def calculate_distance_till_target_list(
    list_coords: list[tuple[float]], target_coords
) -> list[float]:
    return [
        calculate_distance_till_between_points(coords, target_coords)
        for coords in list_coords
    ]
