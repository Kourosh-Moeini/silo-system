import numpy as np

def dynamics(t, x, u):
    m = 0.1  # mass (kg)
    l = 0.5  # length (m)
    b = 0.1  # damping
    g = 9.81 # gravity
    theta, theta_dot = x[0], x[1]
    u_scalar = u[0] if isinstance(u, (list, np.ndarray)) else u
    theta_dotdot = (u_scalar - m * g * l * np.sin(theta) - b * theta_dot) / (m * l**2)
    return np.array([theta_dot, theta_dotdot])