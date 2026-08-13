import numpy as np

def dynamics(t, x, u):
    # Ball and Beam dynamics (only ball states)
    # States:
    #   x[0]: ball position r (m)
    #   x[1]: ball velocity dr (m/s)
    # Input u: beam angle alpha (rad) - motor dynamics removed

    g = 9.8                  # gravity (m/s²)

    r, dr = x[0], x[1]
    alpha = u[0] if isinstance(u, (list, np.ndarray)) else u

    # Ball rolling dynamics (exact match to MATLAB model)
    ddr = (5/7) * g * np.sin(alpha)

    return np.array([dr, ddr])