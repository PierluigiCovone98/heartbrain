"""Experiments on the Vanilla RNN module."""
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
    params = brain.init_params(input_size=INPUT_SIZE, output_size=HIDDEN_SIZE, rng=rng)

    # Experiment
    h_prime = brain.cell_forward(x, h, params)

    # Log 
    print(f"{h_prime=}")


if __name__ == '__main__':
    main()