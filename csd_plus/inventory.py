"""Negative-gap rate at a stated measurement configuration.

A raw negative-gap count, or the rate derived from it, is not comparable
between corpora, and the reason is in the definition of the cross term rather
than in any property of the embeddings.

The cross term is a maximum over competitors. For a fixed artist,
P(c_k > w_k) is the probability that a maximum over K-1 competitors clears one
fixed threshold, so it grows with the artist count K whatever the competitors
look like. Both the flagged count and the flagged rate therefore rise with the
inventory by arithmetic, and a corpus of 1639 artists will report a higher rate
than one of 89 even if the representation is equally good on both.

Both terms are also medians over pairs, and thin per-artist sampling makes both
noisier. They are not affected alike: the within term is a single median while
the cross term is a maximum over K-1 of them, and a maximum over noisy
estimates is biased upward. Anchor pools of ten to twenty works, the usual size
in style-fidelity evaluation, therefore inflate the rate substantially relative
to pools of hundreds.

Neither effect is representational, and neither is a reason to distrust the
diagnostic: they are reasons to state the configuration. Two rates are
comparable when they were measured at the same artist count and the same anchor
depth, which is what ``negative_rate_at`` produces.

Measured on the three corpora of the paper at K=89, with the artist count and
the works per artist swept jointly, the rate ranges from about 11% at hundreds
of works per artist to about 25% at ten, and the three corpora agree to within
a few points once both are matched.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .csls import _l2_normalize, _per_artist_indices

__all__ = ["negative_rate_at"]


def negative_rate_at(
    X: np.ndarray,
    y: np.ndarray | Sequence[int],
    n_artists: int | None = None,
    works_per_artist: int | None = None,
    reps: int = 200,
    min_works: int = 10,
    seed: int = 0,
) -> dict:
    """Pairwise negative-gap rate at a stated (artist count, anchor depth).

    Artists and works are resampled jointly on each repetition, so the reported
    standard deviation covers both sources of variation.

    Args:
        X: (n, d) embeddings; rows need not be L2-normalised.
        y: (n,) integer artist labels.
        n_artists: how many artists the cross-term maximum runs over. Defaults
            to every artist with at least ``min_works`` works, which reproduces
            the plain ``discrimination_gap`` rate.
        works_per_artist: cap on the anchors per artist. None uses all of them.
            Artists holding fewer than the cap are used at their own size, so a
            cap above the corpus median has little effect.
        reps: resampling repetitions. Ignored when neither argument binds,
            since the configuration is then unique and the rate exact.
        min_works: minimum works for an artist to be eligible at all.
        seed: seed for the resampling.

    Returns:
        Dict with n_artists, works_per_artist, negative_rate_mean,
        negative_rate_std, n_eligible_artists and reps. When neither cap binds,
        std is 0.0 and reps is 1.

    Raises:
        ValueError: if fewer than two artists are eligible, or if n_artists
            exceeds the number of eligible artists.
    """
    Xn = _l2_normalize(np.asarray(X, dtype=np.float32))
    y = np.asarray(y)
    by_artist = _per_artist_indices(y)
    eligible = [a for a in sorted(by_artist) if by_artist[a].size >= min_works]
    if len(eligible) < 2:
        raise ValueError(
            f"need at least 2 artists with >= {min_works} works, got "
            f"{len(eligible)}")
    if n_artists is not None and n_artists > len(eligible):
        raise ValueError(
            f"n_artists={n_artists} exceeds {len(eligible)} eligible artists")

    k_target = len(eligible) if n_artists is None else int(n_artists)
    max_works = max(by_artist[a].size for a in eligible)
    binds = (k_target < len(eligible)
             or (works_per_artist is not None and works_per_artist < max_works))
    n_rep = reps if binds else 1

    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_rep):
        sel = (eligible if k_target == len(eligible) else
               [eligible[i] for i in rng.choice(len(eligible), size=k_target,
                                                replace=False)])
        idx = []
        for a in sel:
            i = by_artist[a]
            idx.append(i if works_per_artist is None or i.size <= works_per_artist
                       else rng.choice(i, size=works_per_artist, replace=False))
        w = np.empty(k_target)
        C = np.zeros((k_target, k_target))
        for p in range(k_target):
            S = Xn[idx[p]] @ Xn[idx[p]].T
            w[p] = float(np.median(S[np.triu_indices(len(idx[p]), 1)]))
        for p in range(k_target):
            for q in range(p + 1, k_target):
                v = float(np.median(Xn[idx[p]] @ Xn[idx[q]].T))
                C[p, q] = C[q, p] = v
        neg = sum((w[p] - np.delete(C[p], p).max()) < 0 for p in range(k_target))
        rates.append(neg / k_target)

    return {
        "n_artists": k_target,
        "works_per_artist": works_per_artist,
        "negative_rate_mean": float(np.mean(rates)),
        "negative_rate_std": float(np.std(rates)),
        "n_eligible_artists": len(eligible),
        "reps": n_rep,
    }
