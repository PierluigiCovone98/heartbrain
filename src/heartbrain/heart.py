"""Heart module.

Models the heart as a self-sustained nonlinear oscillator (a modified
Van der Pol oscillator) that can be coupled to the brain module.
"""
from heartbrain import integrators

class Heart:
    """ A self-sustained nonlinear oscillator (modified Van der Pol). 
    
    The state is a pair ``(x,y)`` that evolves according to:

        dx/dt = y
        dy/dt = mu * (1-x^2) * y - x + sigma
        
    Here:
        -   ``y`` is the velocity of  ``x``; 
        -   ``-x`` is the linear restoring term; 
        -   ``mu * (1 - x**2) * y`` is the nonlinear damping that 
            pumps small oscillations and damps large ones, producing 
            a stable limit cycle;
        -   ``sigma`` is the input coming from the brain, scaled by the
            intensity value ``k_bh``.

    The state ``(current_x, current_y)`` evolves at every step. 
    The configuration parameter ``mu`` is fixed for the 
    life of the object and are therefore not part of the state.

    Attributes
    ----------
    current_x : float
        First state variable (the heart signal).
    current_y : float
        Second state variable (the velocity of ``current_x``).
    mu : float
        Nonlinearity parameter; strictly positive.
    sigma:
        Signal coming from the brain, scaled by the intensity value ``k_bh``.
    """    
    
    def __init__(self, initial_x: float = 0.1,
                        initial_y: float = 0.0, 
                        mu: float = 1.0) -> None:
        """Initialize a heart with a given state and configuration.
        
        Raises a ``ValueError`` if the initial state is the origin,
        or if ``mu <= 0``. 
        """

        self._validate_state(initial_x, initial_y)

        # A limit cycle (self-sustained oscillation) exists only for mu > 0.
        if mu <= 0:
            raise ValueError("Cannot have mu <= 0; " \
            "this breaks the self-supported requirement.")

        self._initial_x = initial_x
        self._initial_y = initial_y

        self.current_x = initial_x
        self.current_y = initial_y
        self.mu = mu


    def step(self, sigma: float, dt: float) -> None:
        """Advance the state by one time step of size ``dt`` (RK4).

        Parameters
        ----------
        sigma : float
            Brain signal for this step, previously scaled by ``k_bh``.
        dt : float
            Step size.

        Notes
        -----
        ``sigma`` is frozen for the whole step: all four RK4 evaluations
        use the same value.
        """

        # Closure
        def differentiator(x: float, y: float) -> integrators.VelocityVector:
            return self._differentiate(x, y, sigma)

        # Update the state
        self.current_x, self.current_y = integrators.rk4_step(differentiator, self.current_x, self.current_y, dt)


    def get_initial_state(self) -> integrators.HeartState:
        """Return the initial state (x, y)."""
        return (self._initial_x, self._initial_y)


    def get_state(self) -> integrators.HeartState:
        """Return a snapshot of the heart state."""
        return (self.current_x, self.current_y)
    

    def set_state(self, heart_state: integrators.HeartState) -> None:
        """Restore the heart state from a snapshot.
        
        Raises a ``ValueError`` if the restored state is the origin
        """

        x = heart_state[0]
        y = heart_state[1]

        self._validate_state(x, y)
        
        self.current_x = x
        self.current_y = y


    def reset_state(self) -> None:
        """Restore the heart state with initial values."""
        self.current_x = self._initial_x
        self.current_y = self._initial_y


    def _differentiate(self, x: float, y: float, sigma: float) -> integrators.VelocityVector:
        """Evaluate the vector field ``(dx/dt, dy/dt)`` at a point.

        Evaluated at an arbitrary ``(x, y)`` — not necessarily the current
        state — because the integrator probes intermediate points along a
        step.

        Parameters
        ----------
        x, y : float
            Point at which to evaluate the field.
        sigma : float
            Brain signal, scaled by the ``k_bh`` intensity value.

        Returns
        -------
        VelocityVector
            The pair ``(dx/dt, dy/dt)`` at ``(x, y)``.
        """
        dx = y
        dy = self.mu * (1 - x**2) * y - x + sigma
        
        return dx, dy


    def _validate_state(self, x: float, y: float) -> None:
        """Check that ``(x, y)`` is a valid heart state.
        
        Raises a ``ValueError`` if the state is the origin.
        """
        
        if x == 0.0 and y == 0.0:
            raise ValueError("A Heart cannot be at the origin (x=0, y=0): " \
            "it is the unstable fixed point.")