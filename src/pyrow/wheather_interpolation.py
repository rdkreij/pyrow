import xarray as xr
import numpy as np


def process_dataset(
    ds: xr.Dataset, required_coords: set[str], required_vars: set[str]
) -> None:
    """
    Process an xarray Dataset, ensuring it has the required coordinates and variables.
    """
    missing_coords = required_coords - set(ds.coords)
    missing_vars = required_vars - set(ds.data_vars)

    if missing_coords:
        raise ValueError(f"Missing required coordinates: {missing_coords}")
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")


def get_velocity_from_dataset(
    ds: xr.Dataset,
    target_time: np.datetime64,
    target_coords: tuple[float],
    type: str = "nearest",
    interpolation_options: dict | None = None,
) -> tuple[float]:
    if interpolation_options is None:
        interpolation_options = {}

    if type == "nearest":
        return get_nearest_value(
            ds, target_time, target_coords, **interpolation_options
        )
    elif type == "zero":
        return (0, 0)


def get_nearest_value(
    ds: xr.Dataset,
    target_time: np.datetime64,
    target_coords: tuple[float],
    ignore_nan: bool = False,
) -> tuple[float]:
    time_idx = np.argmin(np.abs(ds["time"].values - target_time))
    lon_idx = np.argmin(np.abs(ds["lon"].values - target_coords[0]))
    lat_idx = np.argmin(np.abs(ds["lat"].values - target_coords[1]))

    u = ds["u"].isel(time=time_idx, lon=lon_idx, lat=lat_idx).values
    v = ds["v"].isel(time=time_idx, lon=lon_idx, lat=lat_idx).values

    if ignore_nan:
        if np.isnan(u):
            u = 0
        if np.isnan(v):
            v = 0
    return (u, v)
