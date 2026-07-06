"""Raw run of the coupled components of the framework.

More details later...
"""
import numpy as np

from heartbrain import brain, heart

# Constants
SEED = 42

INPUT_SIZE = 3
N =64

G = 2.04    # Edge of chaos (?)


def main():

    # Two different normal random number generators
    # does not bind future reproductions of the experiment
    # to specific "call sequences".
    brain_rng = np.random.default_rng(SEED)
    coupled_rng = np.random.default_rng(SEED)

    # === Vanilla RNN setup ===
    #
    # We use the same order used in the ``gain_sweep``
    # experiment such that the dynamics for a given 
    # value of ``g`` is well known.  
    x = brain_rng.normal(size=INPUT_SIZE)
    h0 = brain.init_h0(hidden_size=N)
    
    rnn = brain.VanillaRNN(
        hidden_size=N,
        input_size=INPUT_SIZE,
        h0=h0,
        rng=brain_rng
    )

    rnn.scale_W_hh(g=G)

    # === Heart setup ===
    h = heart.Heart()


    # === Coupling Them ===


if __name__=="__main__":
    main()