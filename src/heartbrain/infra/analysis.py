"""Analysis module.

Signal-analysis tools for measuring the coupled system. 
Nothing here changes how the system evolves: these functions 
run on the data it produces.
"""
import numpy as np
from scipy.signal import butter, filtfilt


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
