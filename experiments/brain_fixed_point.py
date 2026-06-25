"""Convergence to a fixed point in a Vanilla RNN cell.

A Vanilla RNN cell with fixed parameters and a fixed input ``x`` is
iterated for ``N_STEPS`` starting from ``h_0 = 0``. At each step the
state is updated as ``h <- cell_forward(x, h, params)`` and the
distance from the previous state ``||h_t - h_{t-1}||`` is tracked.

With small initialization scale (``1/sqrt(N)``), the trajectory is
expected to converge to a fixed point: the inter-step distance should
decay to zero, meaning ``h_t`` eventually satisfies
``h_t = tanh(W_xh @ x + W_hh @ h_t + b_h)``.

The fixed point depends on the input ``x`` and on the parameters: a
different ``x`` (or different weights) yields a different fixed point,
or possibly no convergence at all.
"""
import numpy as np

from heartbrain import brain

# Constants
INPUT_SIZE = 3
HIDDEN_SIZE = 4

SEED = 42

def main(): 
    """First experiment on the Vanilla RNN: [...]."""
   
    # Vector of zeros by convention.
    h = brain.init_h0(HIDDEN_SIZE)

    rng = np.random.default_rng(SEED)
    
    # Input
    x = rng.normal( size=INPUT_SIZE )     # (INPUT_SIZE,)

    # Random weights
    params = brain.init_params(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, rng=rng)

    # Experiment
    h_prime = brain.cell_forward(x, h, params)

    # Log 
    print(f"{h_prime=}")


if __name__ == '__main__':
    main()