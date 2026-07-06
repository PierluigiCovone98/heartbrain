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
        W_xh= rng.normal( size = (hidden_size, input_size), scale = weight_std(input_size) ),
        W_hh= rng.normal( size = (hidden_size, hidden_size), scale = weight_std(hidden_size) ),
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


class VanillaRNN:
    """A Vanilla Recurrent Neural Network as a dynamical system.

    The network is characterized by its hidden state ``h``, which evolves
    in time according to the recurrence:

        h_{t+1} = tanh(W_xh @ x + W_hh @ h_t + b_h)

    The class encapsulates the parameters (``W_xh``, ``W_hh``, ``b_h_baseline``)
    and the current state (``h``). The state evolves through the ``step`` method.
    The bias can be modulated by an external signal via ``apply_bias_perturbation``;
    this is used, for example, when the network is coupled to a heart oscillator
    (see the ``coupling`` module).

    Weights are initialized from a normal distribution scaled by ``1/sqrt(fan_in)``,
    with ``W_hh`` further scaled by the gain ``g`` if required.
    This scaling ensures the argument of tanh stays in the informative regime,
    and ``g`` controls the dynamical regime of the network.
    """

    def __init__(self, hidden_size: int, input_size: int, h0: np.ndarray, rng: np.random.Generator) -> None:
        """Initialize the network with random weights and a given initial state.

        Parameters
        ----------
        hidden_size : int
            Dimension of the hidden state space (``N``).
        input_size : int
            Dimension of the input vector.
        h0 : np.ndarray
            Initial hidden state, shape ``(hidden_size,)``.
        rng : np.random.Generator
            Normal random number generator to initiliaze network parameters.
        """
    
        self._W_xh = rng.normal( size = (hidden_size, input_size), scale = weight_std(input_size) )
        # TODO: implement a ``_W_xh_baseline`` version if required later.

        self._W_hh_baseline = rng.normal( size = (hidden_size, hidden_size), scale = weight_std(hidden_size) )
        self._W_hh = self._W_hh_baseline.copy()

        self._b_h_baseline = rng.normal( size = hidden_size ) 
        self._b_h = self._b_h_baseline.copy()

        # Inital hidden state is set by the caller.
        self._initial_state = h0
        self._state = self._initial_state.copy()


    def scale_W_hh(self, g: float) -> None:
        """Scale the layer ``W_hh`` by g.
        
        Parameters
        ----------
        g : float
            Gain parameter that scales ``W_hh``. Controls the dynamical regime:
            ``g < 1`` produces convergence, ``g ≈ 1`` is the critical threshold,
            ``g > 1`` produces rich dynamics (limit cycles, chaos).
        """
        self._W_hh = g * self._W_hh_baseline


    def apply_bias_perturbation(self, perturbation: np.ndarray) -> None:
        """Update the current bias as ``b_h_baseline + perturbation``.

        This is the entry point for external modulation of the network. The
        ``perturbation`` is added to a fixed baseline; the baseline itself
        never changes. Passing zeros restores the unperturbed dynamics.

        Parameters
        ----------
        perturbation : np.ndarray
            A vector of shape ``(hidden_size,)`` that is added to the baseline
            bias to produce the current bias used in the next call to ``step``.
        """
        self._b_h = self._b_h_baseline + perturbation


    def step(self, x: np.ndarray) -> None:
        """Advance the hidden state by one step.

        Uses the current bias, which may have been modulated
        by a previous call to ``apply_bias_perturbation``.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape ``(input_size,)``.
        """
        self._state = np.tanh( self._W_xh @ x + self._W_hh @ self._state + self._b_h )


    def reset_state(self) -> None:
        """Reset the current state to the initial state."""
        self._state = self._initial_state.copy()


    # === Getters ===
    @property
    def W_xh(self) -> np.ndarray: 
        """Reads safely the input projection layer."""
        return self._W_xh.copy()
    
    @property
    def W_hh_baseline(self) -> np.ndarray: 
        """Reads safely the baseline state projection layer."""
        return self._W_hh_baseline.copy()

    @property
    def W_hh(self) -> np.ndarray: 
        """Reads safely the scaled state projection layer."""
        return self._W_hh.copy()
    
    @property
    def b_h_baseline(self) -> np.ndarray: 
        """Reads safely the baseline bias b_h."""
        return self._b_h_baseline.copy()

    @property
    def b_h(self) -> np.ndarray: 
        """Reads safely the bias b_h."""
        return self._b_h.copy()

    @property
    def initial_state(self) -> np.ndarray: 
        """Reads safely the initial state h0."""
        return self._initial_state.copy()

    @property
    def state(self) -> np.ndarray: 
        """The current hidden state ``h``, shape ``(hidden_size,)``.

        Read-only. Modify via ``step``.
        """
        return self._state.copy()
    

# === Utility Functions ===
def weight_std(input_size: int) -> float:
    """Inversely propotional scale of weights standard deviation. w.r.t. input dimension of the layer."""
    return 1 / np.sqrt(input_size)
