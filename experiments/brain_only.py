"""Experiments on the Vanilla RNN module."""
import numpy as np

from heartbrain import brain


# Constants
INPUT_SIZE = 3
HIDDEN_SIZE = 4

SEED = 42

def main(): 
    """First experiment on the Vanilla RNN: create the first hidden state ``h'``."""
    # Vector of zeros by convention.
    h = np.zeros(HIDDEN_SIZE)       # (HIDDEN_SIZE,)

    
    rng = np.random.default_rng(SEED)
    
    # Input
    x = rng.normal( size=INPUT_SIZE )     # (INPUT_SIZE,)

    # Random weights
    W_xh = rng.normal( scale=_weight_std(INPUT_SIZE), size=(HIDDEN_SIZE,INPUT_SIZE) )
    W_hh = rng.normal( scale=_weight_std(HIDDEN_SIZE), size=(HIDDEN_SIZE, HIDDEN_SIZE) )
    b_h = rng.normal( size=HIDDEN_SIZE )

    params = brain.VanillaRNNParams(W_xh, W_hh, b_h)

    # Experiment
    h_prime = brain.cell_forward(x, h, params)

    # Log 
    print(f"{h_prime=}")


def _weight_std(dimension: int) -> float:
    """Inversely propotional scale of weights sdt. w.r.t. input dimension of the layer."""
    return 1 / np.sqrt(dimension)


if __name__ == '__main__':
    main()