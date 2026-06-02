"""Integrators module.

Generic numerical methods to advance ordinary differential equations.
"""

from collections.abc import Callable


# (x, y) state of the heart.
type HeartState = tuple[float, float]
# VelocityVector == (dx/dt, dy/dt)      
type VelocityVector = tuple[float, float]       
# A field: given a point (x, y), returns the velocity at that point.
type Differentiator = Callable[ [float, float], VelocityVector]


def rk4_step(diff : Differentiator, x : float, y : float, dt : float) -> HeartState:
    """Advance one Runge-Kutta 4 step of size ``dt``.
    -----
    RK4 samples the field four times across the step and combines them in
    a weighted average. The two midpoint samples are weighted double
    because they best approximate the average velocity over the interval.
    """

    # Velocity vector at the starting point.
    k1x, k1y = diff(x, y)

    # Velocity at trial points along the step (not the heart's real state):
    # each midpoint is reached using the previous estimate.
    k2x, k2y = diff(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
    k3x, k3y = diff(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
    k4x, k4y = diff(x + dt * k3x,       y + dt * k3y)

    new_x = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    new_y = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    
    return new_x, new_y