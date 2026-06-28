"""Showing how different values for the initial state ``h0``
converge to the same fixed point ``h*``.
The differences between each trajectories are:
    1.  The trajectory begin
    2.  (Partially) The speed the trajectory converges.

The point 2. is simply motivated by a space closeness:
``h0``s that are closer to fixed points, converge faster 
than those that are further away. 
"""
import numpy as np
import matplotlib.pyplot as plt

from heartbrain import brain

import os


# Constrants
SEED = 42

INPUT_SIZE = 3
HIDDEN_SIZE = 2     # Easier to plot

N_STEPS = 50

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
PLOT_PATH = os.path.join(OUTPUT_DIR, "brain_h0_sensitivity_phase_portrait.png")


def main():
    rng = np.random.default_rng(SEED)

    # One fixed input
    x = rng.normal( size=INPUT_SIZE )   # (INPUT_SIZE,)

    # Random weights
    params = brain.init_params(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, rng=rng)

    # Design choice: manually h0's
    h0s = []
    h0s.append( np.array( [+0.90, +0.90], dtype=np.float64 ) )
    h0s.append( np.array( [+0.90, -0.90], dtype=np.float64 ) )
    h0s.append( np.array( [-0.90, +0.90], dtype=np.float64 ) )
    h0s.append( np.array( [-0.90, -0.90], dtype=np.float64 ) )
    h0s.append( np.array( [0, 0.], dtype=np.float64 ) )

    # Experiment
    h0s_trajectories = []
    for h0 in h0s:
        trajectory = _run_single_experiment(x, params, h0, N_STEPS)
        h0s_trajectories.append( [h0, trajectory] )

    # Logs
    for i, (h_0, trajectory) in enumerate(h0s_trajectories):
        _print_trajectory(i, len(h0s), h_0, trajectory)

    _print_summary(x, h0s_trajectories)
    _plot_phase_portrait(x, h0s_trajectories)


def _run_single_experiment(x: np.ndarray, params: brain.VanillaRNNParams, h0: np.ndarray, n_steps: int) -> list[np.ndarray]:
    """Run the cell from fixed ``h_0`` for ``n_steps`` with ``x``. Returns the trajectory."""
    trajectory = [h0]
    h = h0
    for _ in range(1, n_steps):
        h = brain.cell_forward(x, h, params)
        trajectory.append(h)
    return trajectory


# === Log Functions ===
def _print_trajectory(index: int, total: int, h_0: np.ndarray, trajectory: list[np.ndarray]) -> None:
    """Print one experiment's trajectory with a header line."""
    print(f"\n==== Experiment {index+1}/{total} — h_0 = {np.array2string(h_0, precision=4)} ====")
    for t, h in enumerate(trajectory):
        h_str = ", ".join(f"{hi:+.4f}" for hi in h)
        if t == 0:
            distance_str = "-"
        else:
            distance_str = f"{np.linalg.norm(h - trajectory[t-1]):.6f}"
        print(f"t={t:3d}   h = [{h_str}]   |dh| = {distance_str}")


def _print_summary(x: np.ndarray, h0s_trajectories: list[list]) -> None:
    """Print a compact summary mapping each initial state to its final hidden state."""
    print(f"\n==== Summary (fixed x = {np.array2string(x, precision=4)}) ====")
    for i, (h_0, trajectory) in enumerate(h0s_trajectories):
        h_star = trajectory[-1]
        h0_str = np.array2string(h_0, precision=4)
        h_str = np.array2string(h_star, precision=4)
        print(f"h0_{i} = {h0_str}   ->   h* = {h_str}")
        

# === Plotting Functions ===
def _plot_phase_portrait(x: np.ndarray, h0s_trajectories: list[list]) -> None:
    """Phase plot of all trajectories in the (h[0], h[1]) plane.
    
    Each trajectory is drawn as a line with markers at each step,
    starting from a colored circle (h_0) and ending at a black X (h*).
    All trajectories share the same fixed input ``x``.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10.colors

    for i, (h_0, trajectory) in enumerate(h0s_trajectories):
        trajectory_array = np.array(trajectory)  # shape (N_STEPS, 2)
        h0_dim0 = trajectory_array[:, 0]
        h0_dim1 = trajectory_array[:, 1]
        color = colors[i % len(colors)]

        # Trajectory line with small markers at each step
        plt.plot(h0_dim0, h0_dim1, color=color, lw=1.2, marker="o", ms=3, alpha=0.7,
                 label=f"h_0 = [{h_0[0]:+.1f}, {h_0[1]:+.1f}]")
        # Start point: filled circle of the same color
        plt.plot(h0_dim0[0], h0_dim1[0], "o", color=color, ms=12, mec="black", mew=1.2)

    # End point: single black X (all trajectories converge here)
    final_h = h0s_trajectories[0][1][-1]
    plt.plot(final_h[0], final_h[1], "x", color="black", ms=15, mew=2.5, label=f"h* (shared)")

    plt.title(f"Phase portrait — all trajectories converge to h*\n(fixed x = {np.array2string(x, precision=3)})")
    plt.xlabel("h[0]")
    plt.ylabel("h[1]")
    plt.axhline(0, color="gray", lw=0.4)
    plt.axvline(0, color="gray", lw=0.4)
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.legend(fontsize=9, loc="lower left")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=130)
    plt.close()


if __name__=='__main__':
    main()
