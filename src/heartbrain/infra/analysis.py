"""Analysis module.

Signal-analysis tools for measuring the coupled system. 
Nothing here changes how the system evolves: these functions 
run on the data it produces.
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert


# Filter design constants (see module discussion for the rationale).
_FILTER_ORDER = 4  # Butterworth order: sharp enough, numerically stable.


def low_pass_filter(signal: np.ndarray, cutoff_period: float) -> np.ndarray:
    """Low-pass filter a signal, keeping components slower than a threshold.

    Separates scales by period: components with period longer than
    ``cutoff_period`` (i.e. slower) pass through; shorter (faster) ones are
    removed. The threshold is given in steps, matching how the signals are
    reasoned about, and converted to frequency internally.

    Uses a Butterworth filter applied forward-and-backward (zero-phase), so the
    output stays time-aligned with the input — essential, since the result feeds
    a phase measurement and any filter-induced delay would corrupt it.

    Note: zero-phase filtering can leave small artifacts at the very start and
    end of the signal; callers should discard the borders before measuring
    (as the transient is already discarded).

    Parameters
    ----------
    signal : np.ndarray
        The 1-D signal to filter, one sample per step.
    cutoff_period : float
        Threshold period in steps. Components slower than this are kept; faster
        ones removed. Choose it in the gap between the two scales (e.g. between
        the network's fast cycle and the heart's slow one).

    Returns
    -------
    np.ndarray
        The filtered signal, same length as the input, time-aligned with it.
    """
    # Sampling rate is one sample per step, so Nyquist (the max representable
    # frequency) is 0.5 cycles/step. Filter cutoff frequency is 1 / period.
    nyquist = 0.5
    cutoff_freq = 1.0 / cutoff_period
    normalized_cutoff = cutoff_freq / nyquist

    # Design the Butterworth low-pass and apply it forward-and-backward.
    # Notice that the type checker has been explicitly silenced because unable
    # to make the correct type inference.
    b, a = butter(N=_FILTER_ORDER, Wn=normalized_cutoff, btype="low", output='ba')   # type: ignore[misc] 
    return filtfilt(b, a, signal)


def _center_on_zero(signal: np.ndarray) -> np.ndarray:
    """Subtract the mean so the signal oscillates around zero.
 
    A precondition for the Hilbert transform, whose phase is only meaningful for
    a signal centered on zero. Kept as its own function to name the operation
    ("bring to zero mean") independently of who needs it.
    """
    return signal - np.mean(signal)


def measure_plv(heart_series: np.ndarray, brain_series: np.ndarray, 
                cutoff_period: int, transient_steps: int, border_steps: int) -> float:
    """Measure the phase locking value between the heart and brain signals.

    The brain signal is low-pass filtered first, to isolate the slow scale that 
    carries the locking (its fast chaotic component would spoil the phase); the 
    heart signal is already slow and clean, so it goes straight to phase extraction. 
    Both phases are then trimmed to the valid window — transient plus border at the
    start, border at the end — before the locking value is computed.

    The window is applied last: filtering and phase extraction need the whole
    signal (they use surrounding context), so trimming earlier would only create
    fresh artifacts at the new borders.

    Parameters
    ----------
    heart_series : np.ndarray
        The heart signal ``x(t)``, one value per step.
    brain_series : np.ndarray
        The network's projected signal ``s(t)``, one value per step.
    cutoff_period : int
        Low-pass cutoff for the brain signal, in steps.
    transient_steps : int
        Steps dropped at the start for the system's transient.
    border_steps : int
        Steps dropped at each end for filter/Hilbert border artifacts.

    Returns
    -------
    float
        The phase locking value in ``[0, 1]``: 0 no locking, 1 perfect locking.
    """
    heart_phase = instantaneous_phase(heart_series)

    # Brain series needs to be pre-processed to become phase
    brain_slow = low_pass_filter(signal=brain_series, cutoff_period=cutoff_period)
    brain_phase = instantaneous_phase(brain_slow)

    n_start = transient_steps + border_steps
    heart_phase_valid = discard_borders(heart_phase, n_start=n_start, n_end=border_steps)
    brain_phase_valid = discard_borders(brain_phase, n_start=n_start, n_end=border_steps)

    return phase_locking_value(heart_phase_valid, brain_phase_valid)


def instantaneous_phase(signal: np.ndarray) -> np.ndarray:
    """Instantaneous phase of an oscillating signal, via the Hilbert transform.
 
    Centers the signal on zero, builds the analytic signal (the original as the
    real part, its 90°-shifted version as the imaginary part — the two
    coordinates of the rotating point), and returns the angle of that point at
    each step: the instantaneous phase.
 
    The phase is wrapped in ``(-π, π]`` — it rises then jumps back by ``2π``
    each cycle (a sawtooth). This is the natural form for a periodic angle and is
    exactly what a phase-locking measure consumes; it is not unwrapped here.
 
    The signal must be narrow-band (a single dominant oscillation) for the
    phase to be well defined — hence apply ``low_pass_filter`` first to isolate
    one scale. Border artifacts (from filtering and from Hilbert) affect the
    start/end of the output; discarding the borders is the caller's job, at
    measurement time.
 
    Parameters
    ----------
    signal : np.ndarray
        The 1-D oscillating signal, one sample per step. Expected to be already
        band-limited (e.g. low-pass filtered).
 
    Returns
    -------
    np.ndarray
        The wrapped instantaneous phase in radians, same length as the input.
    """
    centered = _center_on_zero(signal)
    # Specify the type such that no "type-inference" problems raises.
    analytic: np.ndarray = hilbert(centered)    # type: ignore[assignment]
    return np.angle(analytic) 


def discard_borders(series: np.ndarray, n_start: int, n_end: int) -> np.ndarray:
    """Drop samples from the start and the end of a time series.
 
    Border samples are often unusable: transformations that need surrounding
    context (filtering, analytic-signal construction) have none at the extremes
    and distort the output there, and a run's opening samples may still be a
    transient. Both are spurious — they reflect the procedure, not the system —
    so they are dropped before measuring.
 
    The two amounts are independent because the two ends usually need different
    trims.
 
    Parameters
    ----------
    series : np.ndarray
        The 1-D time series to trim.
    n_start : int
        Number of samples to drop from the beginning.
    n_end : int
        Number of samples to drop from the end.
 
    Returns
    -------
    np.ndarray
        The interior of the series, of length ``len(series) - n_start - n_end``.
 
    Raises
    ------
    ValueError
        If the requested trims leave nothing.
    """
    if n_start + n_end >= len(series):
        raise ValueError(
            f"Trimming {n_start} + {n_end} samples leaves nothing of a "
            f"{len(series)}-sample series."
        )
 
    # ``n_end`` counts from the end; slicing with a negative stop, or with None
    # when nothing is dropped (series[a:-0] would be empty).
    stop = -n_end if n_end > 0 else None
    return series[n_start:stop]
 
 
def phase_locking_value(phase_a: np.ndarray, phase_b: np.ndarray) -> float:
    """Phase locking value between two instantaneous-phase series.
 
    Measures how stable the phase difference between the two signals is, i.e.
    how strongly they are locked, as a number in ``[0, 1]``.
 
    Phases are angles, so ordinary statistics do not apply (the mean of 179° and
    -179° is not 0°). Instead each phase difference is treated as a unit arrow on
    the circle pointing in its direction; the arrows are averaged and the length
    of the resulting arrow is the result. All pointing the same way (constant
    difference) reinforce => length 1; scattered ones cancel out => length close to 0.
    Complex numbers are only the convenient encoding of those arrows.
 
    Symmetric in its arguments: swapping them flips the sign of the difference,
    which does not change the resulting length.
 
    Border artifacts are not handled here — pass series already trimmed to a
    valid window (see ``discard_borders``).
 
    Parameters
    ----------
    phase_a : np.ndarray
        Instantaneous phase of the first signal, in radians.
    phase_b : np.ndarray
        Instantaneous phase of the second signal, in radians. Same length as
        ``phase_a``.
 
    Returns
    -------
    float
        The phase locking value: 0 means no locking, 1 means perfect locking.
 
    Raises
    ------
    ValueError
        If the two phase series have different lengths.
    """
    if len(phase_a) != len(phase_b):
        raise ValueError(
            f"Phase length mismatch: {len(phase_a)} vs {len(phase_b)}."
        )
 
    # Numpy difference means "difference element-wise".
    phase_difference = phase_b - phase_a
 
    # Each difference becomes a unit arrow; average them and take the length.
    unit_arrows = np.exp(1j * phase_difference)
 
    return np.abs(np.mean(unit_arrows)).item()