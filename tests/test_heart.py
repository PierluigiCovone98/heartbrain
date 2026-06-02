"""Tests for the Heart module.

Run with: pytest tests/ -v
"""
import numpy as np
import pytest

from heartbrain.heart import Heart

DT = 0.01


def _run(heart: Heart, n: int, sigma: float = 0.0, dt: float = DT):
    """Step ``heart`` ``n`` times, returning the recorded ``(xs, ys)`` arrays."""
    xs = np.empty(n)
    ys = np.empty(n)
    for i in range(n):
        xs[i], ys[i] = heart.current_x, heart.current_y
        heart.step(sigma=sigma, dt=dt)
    return xs, ys


# --- limit cycle -----------------------------------------------------------

@pytest.mark.parametrize("start", [(0.1, 0.0), (3.0, 0.0), (-2.5, 2.0), (0.5, -1.0)])
def test_converges_to_limit_cycle(start):
    """From any non-origin start, the amplitude settles near the Van der Pol
    limit-cycle value (|x|max ~= 2.0 for mu=1)."""
    heart = Heart(initial_x=start[0], initial_y=start[1], mu=1.0)
    xs, _ = _run(heart, n=6000)          # ~60 time units: well past warm-up
    amplitude = np.abs(xs[-1000:]).max()  # over the last ~10 time units
    assert amplitude == pytest.approx(2.0, abs=0.05)


def test_limit_cycle_independent_of_initial_conditions():
    """Two hearts from different starts reach the same cycle amplitude
    (the 'forgetting' of initial conditions)."""
    h1 = Heart(initial_x=0.1, initial_y=0.0, mu=1.0)
    h2 = Heart(initial_x=3.0, initial_y=-1.0, mu=1.0)
    a1 = np.abs(_run(h1, 6000)[0][-1000:]).max()
    a2 = np.abs(_run(h2, 6000)[0][-1000:]).max()
    assert a1 == pytest.approx(a2, abs=0.02)


def test_autonomous_heart_oscillates():
    """An autonomous heart keeps crossing zero: it does not settle to a point."""
    xs, _ = _run(Heart(mu=1.0), 2000)
    assert (xs[1000:] > 0).any() and (xs[1000:] < 0).any()


# --- state snapshot --------------------------------------------------------

def test_state_roundtrip_is_exact():
    """A continuous run equals a run that is snapshotted and restored midway."""
    # path A: 1500 steps, uninterrupted
    a = Heart(mu=1.0)
    _run(a, 1500)

    # path B: 1000 steps, snapshot, restore into a fresh heart, 500 more steps
    b = Heart(mu=1.0)
    _run(b, 1000)
    snapshot = b.get_state()
    c = Heart(mu=1.0)
    c.set_state(snapshot)
    _run(c, 500)

    assert a.get_state() == c.get_state()


def test_determinism():
    """Same parameters and same inputs produce the same trajectory."""
    h1 = Heart(initial_x=0.2, initial_y=0.1, mu=1.5, k_bh=0.3)
    h2 = Heart(initial_x=0.2, initial_y=0.1, mu=1.5, k_bh=0.3)
    for _ in range(1000):
        h1.step(sigma=0.1, dt=DT)
        h2.step(sigma=0.1, dt=DT)
    assert h1.get_state() == h2.get_state()


# --- coupling --------------------------------------------------------------

def test_brain_signal_changes_trajectory():
    """A nonzero brain signal (with k_bh > 0) changes the evolution."""
    autonomous = Heart(mu=1.0, k_bh=0.5)
    driven = Heart(mu=1.0, k_bh=0.5)
    for _ in range(500):
        autonomous.step(sigma=0.0, dt=DT)
        driven.step(sigma=0.8, dt=DT)
    assert autonomous.get_state() != driven.get_state()


# --- guards ----------------------------------------------------------------

def test_origin_rejected_at_init():
    with pytest.raises(ValueError):
        Heart(initial_x=0.0, initial_y=0.0)


@pytest.mark.parametrize("bad_mu", [0.0, -1.0])
def test_nonpositive_mu_rejected(bad_mu):
    with pytest.raises(ValueError):
        Heart(mu=bad_mu)


def test_origin_rejected_at_set_state():
    heart = Heart(mu=1.0)
    with pytest.raises(ValueError):
        heart.set_state((0.0, 0.0))
