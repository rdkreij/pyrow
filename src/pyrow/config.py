from pyrow import physical_model

BOAT_PROPERTIES = physical_model.BoatProperties(
    area_anchor=8,
    area_water_front=0.329,
    area_water_side=1.87 + 0.196 + 0.184,
    area_air_front=1.89,
    area_air_side=5.81,
    drag_coefficient_air=0.10,
    drag_coefficient_water=0.0075,
    speed_perfect=2,
    drag_air_only_parallel=False,
    wind_correction={"correction_par": 2, "window_par": 15, "window_perp": 10},
)

START_COORDS = {
    "Fremantle": (115.743889, -32.056946),
    "Geraldton": (114.607513, -28.782387),
    "Kalbarri": (114.164536, -27.711053),
    "Carnarvon": (113.656956, -24.883717),
    "Exmouth": (114.122389, -21.930724),
    "Broome": (122.244327, -17.951221),
}

TARGET_COORDS = {
    "Mombassa": (39.6747, -4.0935),
    "North Madagascar": (49.5358965, -11.3225573),
}
