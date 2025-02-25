from pyproj import Geod
import numpy as np

geod = Geod(ellps="WGS84")


def displace_coordinates_pyproj(coords, x, y):
    lon, lat = coords
    lon_new, lat_new, _ = geod.fwd(lon, lat, 90, x)  # Move east by x meters
    lon_new, lat_new, _ = geod.fwd(lon_new, lat_new, 0, y)  # Move north by y meters
    return (lon_new, lat_new)


def bearing_east_rad(coords_1, coords_2):
    lon1, lat1 = coords_1
    lon2, lat2 = coords_2
    azi_12 = geod.inv(lon1, lat1, lon2, lat2)[0]
    theta_east = (90 - azi_12) % 360
    return np.deg2rad(theta_east)


def calculate_distance_till_between_points(coords_1, coords_2) -> float:
    lon1, lat1 = coords_1
    lon2, lat2 = coords_2
    return geod.inv(lon1, lat1, lon2, lat2)[2]


def calculate_geodesic_line(coords1: tuple[float], coords2: tuple[float], n: int = 100):
    lon1, lat1 = coords1
    lon2, lat2 = coords2

    points = geod.npts(lon1, lat1, lon2, lat2, n)
    lon = np.array([lon1] + [p[0] for p in points] + [lon2])
    lat = np.array([lat1] + [p[1] for p in points] + [lat2])
    return lon, lat
