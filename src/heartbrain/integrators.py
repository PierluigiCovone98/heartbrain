'''Utility module that exposes function to integrate ODEs'''

from collections.abc import Callable


# Aliases for complex type signatures
type HeartState = tuple[float, float]           # HeartState == (x,y)
type VelocityVector = tuple[float, float]       # VelocityVector == (dx/dt, dy/dt)      

type Differentiator = Callable[ [float, float], VelocityVector]


def rk4_step(diff : Differentiator, x : float, y : float, dt : float) -> HeartState:
    """
    Advance one RK4 step of size dt, from the state (x, y), under the given differentiator.
    Return a new state (x', y').
    """

    # Velocity vector at starting state (x,y)
    k1x, k1y = diff(x, y)

    # Velocity vectors at intermediate-states 
    k2x, k2y = diff(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
    k3x, k3y = diff(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
    k4x, k4y = diff(x + dt * k3x,       y + dt * k3y)

    # Final position & velocity
    new_x = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    new_y = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    
    return new_x, new_y