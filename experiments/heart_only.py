"""Run the autonomous Heart module and visualize its dynamics.

Produces two figures in output/:
  * the phase portrait: convergence to the limit cycle from several starts;
  * the time series x(t): the warm-up settling onto the cycle.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from heartbrain.heart import Heart

DT = 0.01
N = 3000                 # ~30 time units
OUTPUT_DIR = "output"


def run(initial_x: float, initial_y: float, n: int = N, dt: float = DT):
    """Run an autonomous heart (sigma=0) and record its trajectory."""
    heart = Heart(initial_x=initial_x, initial_y=initial_y, mu=1.0)
    xs, ys = np.empty(n), np.empty(n)
    for i in range(n):
        xs[i], ys[i] = heart.current_x, heart.current_y
        heart.step(sigma=0.0, dt=dt)
    return xs, ys


def plot_phase_portrait(path: str) -> None:
    """Several starts converging onto the same limit cycle."""
    plt.figure(figsize=(6, 6))
    starts = [(0.1, 0.0), (3.0, 0.0), (-2.5, 2.0)]
    colors = ["#c0392b", "#2980b9", "#27ae60"]
    for (sx, sy), c in zip(starts, colors):
        xs, ys = run(sx, sy)
        plt.plot(xs, ys, color=c, lw=1.0, label=f"start ({sx}, {sy})")
        plt.plot(sx, sy, "o", color=c, ms=5)
    plt.plot(0, 0, "kx", ms=9, mew=2, label="unstable fixed point")
    plt.title("Autonomous heart: convergence to the limit cycle")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(fontsize=8)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_time_series(path: str) -> None:
    """x(t) starting small: the amplitude grows, then locks onto the cycle."""
    xs, _ = run(0.1, 0.0)
    t = np.arange(N) * DT
    plt.figure(figsize=(10, 3.5))
    plt.plot(t, xs, color="#c0392b", lw=1.3)
    plt.axhline(0, color="gray", lw=0.6)
    plt.title("Autonomous heart: x(t) settling onto the cycle (warm-up)")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_phase_portrait(os.path.join(OUTPUT_DIR, "heart_phase_portrait.png"))
    plot_time_series(os.path.join(OUTPUT_DIR, "heart_time_series.png"))
    print(f"Figures written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
