"""The aim of the following experiment is to find the critical threshold 
where the Vanilla RNN changes its dynamics from a qualitative point of view.

We fix the following parameters:
    - rng seed
    - input ``x``
    - initial state ``h_0`` (set to zero by convention)
    - layer ``W_xh``
    - bias ``b_h``.

We define a base weight matrix W_base for the recurrent connection, generated 
as a standard scaled initialization with 
        sigma_base = 1 / sqrt(hidden_size). 
This corresponds to the case g = 1 (i.e., no rescaling)."mension value.

Then, for each value of g in the sweep, we scale the W_base matrix by a factor 
of g and run the cell for N_STEPS steps. For example, when g = 1.25: 
        
        W_(g=1.25) = 1.25 * W_base.

Results are stored in a list of (g, trajectory) tuples and then logged.
"""
import numpy as np

from heartbrain import brain


# Constants
SEED = 45

INPUT_SIZE = 3
HIDDEN_SIZE = 4

G_MAX = 2.5
G_MIN = 0.5
NUM_G = 9

# Notice:
#   When the system starts to have a chaoitic dynamics
#   it is important to study the trajectory for many steps.
N_STEPS = 100


def main():
    
    rng = np.random.default_rng(SEED)

    # === Set up invariants ===
    x = rng.normal(size=INPUT_SIZE)
    h0 = brain.init_h0(hidden_size=HIDDEN_SIZE)

    # W_xh and W_hh are both initialized with normal-scaled weights,
    # where the standard deviation is
    #       sigma =  1/sqrt(fan_in).
    # 
    # Here fan_in = INPUT_SIZE for W_xh, and fan_in = HIDDEN_SIZE for W_hh.
    params = brain.init_params(input_size=INPUT_SIZE,
                                hidden_size=HIDDEN_SIZE, 
                                rng=rng)

    # ``W_hh`` is our ``W_base`` (just to be clear).
    W_base = params.W_hh


    # === Sweep g ===
    g_values = np.linspace(start=G_MIN, stop=G_MAX, num=NUM_G)


    # === Run The Cell ===
    g_trajectories = []
    for g in g_values:
        
        # The scaled ``W_hh = g * W_base`` is computed directly in the constructor.
        new_params = brain.VanillaRNNParams(W_xh=params.W_xh, 
                                            W_hh= g * W_base, 
                                            b_h=params.b_h)
        trajectory = _run_single_experiment(x=x,params=new_params, h_0=h0, n_steps=N_STEPS)
        g_trajectories.append( (g, trajectory) )

    # Logs
    for i, (g, trajectory) in enumerate(g_trajectories):
        _print_trajectory(i, len(g_trajectories), g, trajectory)

        _print_summary(x, g_trajectories)

 


def _run_single_experiment(x: np.ndarray, params: brain.VanillaRNNParams, h_0: np.ndarray, n_steps: int) -> list[np.ndarray]:
    """Run the cell for n_steps steps with fixed parameters (encapsulated in params). 
    Returns the full trajectory as a list of states, starting with h_0.
    """
    trajectory = [h_0]
    h = h_0
    for _ in range(1, n_steps):
        h = brain.cell_forward(x, h, params)
        trajectory.append(h)
    return trajectory


# === Log functions ===
def _print_trajectory(index: int, total: int, g: float, trajectory: list[np.ndarray]) -> None:
    """Print one experiment's trajectory with a header line."""
    print(f"\n==== Experiment {index+1}/{total} — g = {g:.4f} ====")
    for t, h in enumerate(trajectory):
        h_str = ", ".join(f"{hi:+.4f}" for hi in h)
        if t == 0:
            distance_str = "-"
        else:
            distance_str = f"{np.linalg.norm(h - trajectory[t-1]):.6f}"
        print(f"t={t:3d}   h = [{h_str}]   |dh| = {distance_str}")


def _print_summary(x: np.ndarray, g_trajectories: list[tuple[float, list[np.ndarray]]]) -> None:
    """Print a compact summary mapping each g to its final hidden state and convergence status."""
    print(f"\n==== Summary (fixed x = {np.array2string(x, precision=4)}) ====")
    for g, trajectory in g_trajectories:
        h_final = trajectory[-1]
        last_dh = np.linalg.norm(trajectory[-1] - trajectory[-2])
        converged = last_dh < 1e-4
        status = "converged" if converged else "NOT converged"
        h_str = np.array2string(h_final, precision=4)
        print(f"g = {g:.4f}   h_final = {h_str}   last |dh| = {last_dh:.6f}   [{status}]")


if __name__ == '__main__':
    main()