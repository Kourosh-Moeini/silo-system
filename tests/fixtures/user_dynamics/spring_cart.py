import numpy as np

def dynamics(t, x, u):
    # Mass-Spring-Damper (Spring Cart) System
    # States: x[0] = position (m), x[1] = velocity (m/s)
    # Input: u[0] = applied force (N)
    
    m = 1.0  # Mass (kg)
    b = 0.5  # Damping coefficient (N*s/m)
    k = 2.0  # Spring constant (N/m)

    pos = x[0]
    vel = x[1]
    
    force = u[0] if isinstance(u, (list, np.ndarray)) else u

    # Compute acceleration: a = (F - b*v - k*x) / m
    accel = (force - b * vel - k * pos) / m

    return np.array([vel, accel])