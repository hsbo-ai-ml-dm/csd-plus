"""Tests for negative_rate_at.

The point of the function is that a rate is only meaningful together with the
configuration it was measured at, so the tests pin the two mechanisms that make
that true: the rate rises with the artist count because the cross term is a
maximum over competitors, and it rises as the anchor depth falls because that
maximum runs over noisy medians.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csd_plus import discrimination_gap, negative_rate_at  # noqa: E402


def _corpus(n_artists=40, n_works=40, dim=32, spread=0.35, seed=0):
    """Artists as isotropic clusters around random centres."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_artists, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    X = np.repeat(centres, n_works, axis=0)
    X = X + spread * rng.normal(size=X.shape)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.repeat(np.arange(n_artists), n_works)
    return X.astype(np.float32), y


def test_default_reproduces_discrimination_gap():
    X, y = _corpus()
    rows = discrimination_gap(X, y)
    expected = sum(1 for r in rows if r["gap"] < 0) / len(rows)
    got = negative_rate_at(X, y)
    assert got["negative_rate_mean"] == pytest.approx(expected, abs=1e-12)
    # The unique configuration needs no resampling.
    assert got["reps"] == 1
    assert got["negative_rate_std"] == 0.0


def test_rate_grows_with_artist_count():
    # Tight clusters would put every rate at zero, so the corpus is built with
    # enough overlap that the maximum over competitors can actually bind.
    X, y = _corpus(n_artists=60, n_works=30, spread=0.75, seed=1)
    rates = [negative_rate_at(X, y, n_artists=k, reps=40, seed=1)
             ["negative_rate_mean"] for k in (10, 20, 40, 60)]
    assert rates == sorted(rates), rates
    assert rates[-1] > rates[0]


def test_thin_anchor_pools_inflate_the_rate():
    X, y = _corpus(n_artists=40, n_works=60, spread=0.75, seed=2)
    deep = negative_rate_at(X, y, works_per_artist=60, reps=40, seed=2)
    thin = negative_rate_at(X, y, works_per_artist=10, reps=40, seed=2)
    assert thin["negative_rate_mean"] > deep["negative_rate_mean"]


def test_non_binding_cap_is_exact():
    X, y = _corpus(n_artists=20, n_works=25, seed=3)
    # A cap at or above the largest pool leaves the configuration unique.
    got = negative_rate_at(X, y, works_per_artist=25, reps=99)
    assert got["reps"] == 1
    assert got["negative_rate_mean"] == pytest.approx(
        negative_rate_at(X, y)["negative_rate_mean"], abs=1e-12)


def test_reported_configuration_is_echoed():
    X, y = _corpus(n_artists=30, n_works=20, seed=4)
    got = negative_rate_at(X, y, n_artists=12, works_per_artist=8, reps=5)
    assert got["n_artists"] == 12
    assert got["works_per_artist"] == 8
    assert got["n_eligible_artists"] == 30
    assert got["reps"] == 5


def test_min_works_filters_artists():
    X, y = _corpus(n_artists=10, n_works=12, seed=5)
    assert negative_rate_at(X, y, min_works=12)["n_eligible_artists"] == 10
    with pytest.raises(ValueError, match="at least 2 artists"):
        negative_rate_at(X, y, min_works=13)


def test_too_many_artists_requested():
    X, y = _corpus(n_artists=8, n_works=12, seed=6)
    with pytest.raises(ValueError, match="exceeds 8 eligible"):
        negative_rate_at(X, y, n_artists=9)
