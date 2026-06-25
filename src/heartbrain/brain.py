"""Brain module.

The "brain" of this framework is a "Vanilla RNN".

Notation convention
-------------------
We follow the mathematical convention ``W @ x``:
- A weight matrix ``W`` has shape ``(output_size, input_size)``.
- Row ``i`` of ``W`` contains the weights of neuron ``i``.

This differs from the "batch-first" convention ``x @ W`` common in
deep learning code. Here we use ``W @ x`` because it aligns with
the dynamical-systems literature .
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class VanillaRNNParams:
    """Parameters of the Vanilla RNN.

    Notice that the dimension of the (hidden) state space is 
    not properly a parameter of the recurrent neural network.

    When I will refactor the code to create the proper class,
    there will be the proper field for that information, that 
    we will name: "hidden_size".
    """
    W_xh: np.ndarray
    W_hh: np.ndarray
    b_h: np.ndarray


def init_params(input_size: int, hidden_size: int, rng: np.random.Generator) -> VanillaRNNParams:
    """Random scaled initialization for Vanilla RNN parameters.
    
    ``hidden_size`` is the size of the (hidden) state space.
    """
    return VanillaRNNParams(
        W_xh= rng.normal( size = (hidden_size, input_size), scale = _weight_std(input_size) ),
        W_hh= rng.normal( size = (hidden_size, hidden_size), scale = _weight_std(hidden_size) ),
        b_h= rng.normal( size = hidden_size )
    )


def init_h0(hidden_size: int) -> np.ndarray:
    """Initialzie the first state ``h0`` of dimension (hidden_size,).

    Now ``h0`` is initialized as a vector of zeros (following the convention);
    in the future we will abstract this convention simply by introducing a bool
    parameter ``convention``.  
    """
    return np.zeros(hidden_size, dtype= np.float64)


def cell_forward(x: np.ndarray, h: np.ndarray, params: VanillaRNNParams) -> np.ndarray:
    """Implements the state transition function of the Vanilla RNN, such that:
    
            h' = tanh( W_xh @ x + W_hh @ h + b_h)
        
        where:
            -   ``h'`` is the hidden state at the current time ``t``;
            -   ``x`` is the input vector;
            -   ``h`` is the hidden state at the previous time ``t-1``;
            -   ``b_h`` is the bias (global offset);
    
    Notice that state space has a dimension of ``hidden_size``.
    
    Returns
    -------
    np.ndarray
        The new hidden state ``h'``, shape ``(hidden_size,)``.
    """
    # Transform the input vector ``x`` into a vector in the hidden state space.
    x_contribution = params.W_xh @ x

    # Transform the previous hidden state ``h`` into a vector in the hidden state space again.
    h_contribution = params.W_hh @ h

    return np.tanh( x_contribution + h_contribution + params.b_h)


# === Utility Functions ===
def _weight_std(input_size: int) -> float:
    """Inversely propotional scale of weights standard deviation. w.r.t. input dimension of the layer."""
    return 1 / np.sqrt(input_size)
