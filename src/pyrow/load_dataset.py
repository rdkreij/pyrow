import xarray as xr

#     air   2007-05-01 ... 2020-12-01
#     water 1993-01-16 ... 2020-05-16


def load_air_dataset(file: str):
    ds_air_o = xr.open_dataset(file)
    ds_air = xr.Dataset(
        data_vars=dict(
            u=(["time", "lat", "lon"], ds_air_o.eastward_wind.isel(depth=0).values),
            v=(["time", "lat", "lon"], ds_air_o.northward_wind.isel(depth=0).values),
        ),
        coords=dict(
            lon=ds_air_o.longitude.values,
            lat=ds_air_o.latitude.values,
            time=ds_air_o.time.values,
        ),
    )
    return ds_air


def load_water_dataset(file: str):
    ds_water_o = xr.open_dataset(file)
    ds_water = xr.Dataset(
        data_vars=dict(
            u=(["time", "lat", "lon"], ds_water_o.uo.isel(depth=0).values),
            v=(["time", "lat", "lon"], ds_water_o.vo.isel(depth=0).values),
        ),
        coords=dict(
            lon=ds_water_o.longitude.values,
            lat=ds_water_o.latitude.values,
            time=ds_water_o.time.values,
        ),
    )
    return ds_water


if __name__ == "__main__":
    ds_air = load_air_dataset(
        "data/CERSAT-GLO-REP_WIND_L4-OBS_FULL_TIME_SERIE_1648775425193.nc"
    )
    print(ds_air)

    ds_water = load_water_dataset("data/nemo_monthly_mean.nc")
    print(ds_water)
