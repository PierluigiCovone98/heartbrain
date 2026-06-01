from heartbrain import integrators

class Heart:
    ''' .. Appropiate introduction here .. '''
    
    def __init__(self, initial_x : float = 0.1,
                        initial_y : float = 0.0, 
                        mu : float = 1.0,
                        k_bh : float = 0.0) -> None:

        # A trajectory that starts from an unstable fixed point remains blocked. 
        if initial_x == 0.0 and initial_y == 0.0:
            raise ValueError("Cannot start at the origin (x=0, y=0): " \
            "it is the unstable fixed point.")
        
        # The only condition to have a self-supported heart is: mu > 0.
        if mu <= 0:
            raise ValueError("Cannot have mu <= 0; " \
            "this breaks the self-supported requirement.")

        self.current_x = initial_x
        self.current_y = initial_y
        self.mu = mu
        self.k_bh = k_bh


    
    def _differentiate(self, x : float, y : float, sigma : float) -> integrators.VelocityVector:
        '''
        [...]
            -   dx/dt = y
            -   dy/dt = mu(1-x^2)y - x + (k_bh * sigma)
        '''

        dx = y
        dy = self.mu * (1 - x**2) * y - x + (self.k_bh * sigma)
        
        return dx, dy


    def step(self, sigma : float, dt : float) -> None:
        '''
        Update the current state [...complete the comment here...]
        '''

        # Closure
        def differentiator(x : float, y : float) -> integrators.VelocityVector:
            return self._differentiate(x, y, sigma)

        # Update the state
        self.current_x, self.current_y = integrators.rk4_step(differentiator, self.current_x, self.current_y, dt)