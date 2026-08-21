"""Does the heart-brain locking hold steady at (1,1), or does it fluctuate?

The locking value characterised so far is a single number over a whole run, and
it was measured in the *monodirectional* sweep (k_bh = 0), where it saturates
above |k_hb| ~ 0.15. Neither fact answers the question this experiment asks. A
single number is an average, and an average of 0.99 is equally compatible with
rigid locking at 0.99 and with an excursion between 0.95 and 1.0 — opposite
situations for anything that wants to use the locking as a modulating signal.

The measurement is therefore the same chain, split at the last step only: phases
are extracted and trimmed exactly as in ``measure_plv``, and the final circular
average is computed on consecutive windows instead of on the whole valid range.
The windowed values stay comparable to the global one, and the global value is
printed alongside as a check.

The window is one cardiac cycle (T = 762.5 steps, measured at this operating
point): the locking "of one beat". Two other window lengths are reported as a
robustness check, since the window is the only free parameter here — if the
spread collapses when the window grows, the fluctuation was sub-cycle structure
rather than a slow drift of the locking itself.

No explicit warmup: the transient is dropped by the same trimming used in the
monodirectional experiment, so that the numbers remain comparable to it.

What is read: not the mean, but the excursion — min, max, standard deviation, and
the shape over time.

Pre-registered expectation: the user declined to predict, preferring to look at
the result first. Recorded as such.
"""
import numpy as np

from heartbrain import heart, coupling, coupled_system
from heartbrain.infra import analysis, plotting
from heartbrain.infra.persistence import networks, experiments


# Constants
GAIN_SWEEP_EXP = "gain_sweep"
RNN_BASELINE_NAME = "gain_sweep_baseline"

SEED2 = 54
SEED3 = 77

# === brain
G = 2.5
m
# === heart
DT = 0.01

# === coupled
N_STEPS = 90000
K_HB = 1.0
K_BH = 1.0
SUBDIR = "coupled_plv"

# === brain signal filtering (unchanged from the monodirectional chain)
CUTOFF_PERIOD = 100

# === measurement window (unchanged from the monodirectional chain)
TRANSIENT_STEPS = 1000
BORDER_STEPS = 500

# === windowing
# Cardiac period at (1,1), measured: 762.5 +- 5.9 steps.
CARDIAC_PERIOD = 762
WINDOW_LENGTH = CARDIAC_PERIOD            # one cycle: the locking "of one beat"
ROBUSTNESS_WINDOWS = [CARDIAC_PERIOD // 2, CARDIAC_PERIOD, 2 * CARDIAC_PERIOD]


def main():

    # === Vanilla RNN setup ===
    rnn = networks.load_network(RNN_BASELINE_NAME)
    rnn.scale_W_hh(g=G)
    print(f"OK - {RNN_BASELINE_NAME} network loaded and scaled (g={G}).")    # Log

    _, arrays = experiments.load_experiment(GAIN_SWEEP_EXP)
    x = arrays["x"]
    N = rnn.N

    oscillator = heart.Heart()


    # === Coupling components ===
    heart_to_brain_rng = np.random.default_rng(SEED2)
    K_baseline = coupling.create_K_baseline(rng=heart_to_brain_rng, N=N, fan_in=2)
    K_baseline_x = coupling.extract_K_baseline_x(K_baseline)
    K = coupling.build_K(k_hb=K_HB, K_baseline=K_baseline)

    brain_to_heart_rng = np.random.default_rng(SEED3)
    D_baseline = coupling.create_D_baseline(rng=brain_to_heart_rng, N=N)


    # === Coupled run at the operating point ===
    heart_time_series, brain_time_series, _ = coupled_system.run_coupled(
        rnn=rnn,
        oscillator=oscillator,
        x=x,
        K=K,
        K_baseline_x=K_baseline_x,
        D_baseline=D_baseline,
        k_bh=K_BH,
        n_steps=N_STEPS,
        dt=DT,
    )
    print(f"OK - coupled run done (k_hb={K_HB}, k_bh={K_BH}, n_steps={N_STEPS}).")    # Log


    # === Global PLV: the number comparable to the monodirectional one ===
    plv_global = analysis.measure_plv(heart_series=heart_time_series,
                                      brain_series=brain_time_series,
                                      cutoff_period=CUTOFF_PERIOD,
                                      transient_steps=TRANSIENT_STEPS,
                                      border_steps=BORDER_STEPS)
    print(f"\nPLV global (k_hb={K_HB}, k_bh={K_BH}) = {plv_global:.4f}")    # Log


    # === PLV over windows: what the global average hides ===
    plv_windows = analysis.plv_over_windows(heart_series=heart_time_series,
                                            brain_series=brain_time_series,
                                            cutoff_period=CUTOFF_PERIOD,
                                            transient_steps=TRANSIENT_STEPS,
                                            border_steps=BORDER_STEPS,
                                            window_length=WINDOW_LENGTH)

    print(f"\nwindow = {WINDOW_LENGTH} steps (1 cardiac cycle), "
          f"{len(plv_windows)} windows")    # Log
    print(f"  mean  = {plv_windows.mean():.4f}")    # Log
    print(f"  sd    = {plv_windows.std():.4f}")    # Log
    print(f"  min   = {plv_windows.min():.4f}")    # Log
    print(f"  max   = {plv_windows.max():.4f}")    # Log
    print(f"  range = {plv_windows.max() - plv_windows.min():.4f}")    # Log


    # === Robustness: does the excursion survive a change of window? ===
    print(f"\n{'window':>8}  {'cycles':>7}  {'n':>4}  {'mean':>7}  {'sd':>7}  "
          f"{'min':>7}  {'max':>7}")    # Log
    for window_length in ROBUSTNESS_WINDOWS:
        values = analysis.plv_over_windows(heart_series=heart_time_series,
                                           brain_series=brain_time_series,
                                           cutoff_period=CUTOFF_PERIOD,
                                           transient_steps=TRANSIENT_STEPS,
                                           border_steps=BORDER_STEPS,
                                           window_length=window_length)
        print(f"{window_length:>8}  {window_length / CARDIAC_PERIOD:>7.2f}  "
              f"{len(values):>4}  {values.mean():>7.4f}  {values.std():>7.4f}  "
              f"{values.min():>7.4f}  {values.max():>7.4f}")    # Log


    # === Plot: the shape over time, at one cycle per window ===
    window_index = np.arange(len(plv_windows))
    plotting.plot_curve(x_values=window_index,
                        y_values=plv_windows,
                        name="coupled_plv_over_time",
                        subdir=SUBDIR,
                        x_label="window (1 cardiac cycle each)",
                        y_label="PLV")


if __name__ == "__main__":
    main()
