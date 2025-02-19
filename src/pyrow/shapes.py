import numpy as np

def calculate_zone_coords(correction_par, window_par, window_perp, n=100):
    t = np.linspace(0, 2 * np.pi, n)
    v_perp_boat = window_perp * np.cos(t)
    v_par_boat = window_par * np.sin(t) + correction_par
    return v_perp_boat, v_par_boat

if __name__ == '__main__':    
    print(calculate_zone_coords(2,1,2))
