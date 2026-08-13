import numpy as np

def dynamics(t, x, u):
    # DC Motor armature-controlled model
    # States: x[0] = omega_m (rad/s), x[1] = i_a (A)
    # Input: u[0] = V_a (V)

    Jm = 0.01
    bm = 0.001
    Kt = 0.1
    Ke = 0.1
    R = 1
    L = 0.01

    omega = x[0]
    ia = x[1]

    Va = u[0] if isinstance(u, (list, np.ndarray)) else u

    didt = (-R * ia - Ke * omega + Va) / L
    domegadt = (Kt * ia - bm * omega) / Jm

    return np.array([domegadt, didt])