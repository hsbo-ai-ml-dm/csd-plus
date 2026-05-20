"""Discrimination-gap diagnostic.

For each artist k:
    w_k = median of cos(a, b) over off-diagonal pairs in k's anchors
    c_k = max over j != k of median cos(a, b) for a in k's, b in j's anchors
    g_k = w_k - c_k

A negative g_k means the absolute same-versus-different reading of raw
cosine is order-inverted on artist k against at least one other artist
on the candidate corpus.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)


def _per_artist_indices(y: np.ndarray) -> dict[int, np.ndarray]:
    return {int(c): np.where(y == c)[0] for c in np.unique(y)}


def _within_median(C: np.ndarray, idx: np.ndarray) -> float:
    """Median cosine over off-diagonal pairs within a single artist."""
    if idx.size < 2:
        return float("nan")
    sub = C[np.ix_(idx, idx)]
    iu = np.triu_indices(idx.size, k=1)
    return float(np.median(sub[iu]))


def _cross_median(C: np.ndarray, idx_k: np.ndarray, idx_j: np.ndarray) -> float:
    """Median cosine of the full cross-pair set between artists k and j."""
    if idx_k.size == 0 or idx_j.size == 0:
        return float("nan")
    return float(np.median(C[np.ix_(idx_k, idx_j)]))


def discrimination_gap(
    X: np.ndarray,
    y: np.ndarray | Sequence[int],
    names: Sequence[str] | None = None,
) -> list[dict]:
    """Compute the discrimination gap g_k = w_k - c_k for every artist.

    Args:
        X: (n, d) embedding matrix; rows need not be L2-normalised.
        y: (n,) artist label, integer-coded.
        names: optional length-K mapping from artist id to name; if given,
            output rows include the human-readable name.

    Returns:
        A list of dicts, one per artist, with keys: artist_id, name (if
        provided), n_anchors, w_k, c_k, gap, worst_other_id, worst_other_name.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    Xn = _l2_normalize(X)
    C = Xn @ Xn.T

    by_artist = _per_artist_indices(y)
    artist_ids = sorted(by_artist.keys())

    out = []
    for k in artist_ids:
        idx_k = by_artist[k]
        w_k = _within_median(C, idx_k)
        worst_j = None
        worst_val = -np.inf
        for j in artist_ids:
            if j == k:
                continue
            idx_j = by_artist[j]
            v = _cross_median(C, idx_k, idx_j)
            if v > worst_val:
                worst_val = v
                worst_j = j
        c_k = float(worst_val)
        g_k = w_k - c_k
        row = {
            "artist_id": k,
            "n_anchors": int(idx_k.size),
            "w_k": float(w_k),
            "c_k": float(c_k),
            "gap": float(g_k),
            "worst_other_id": int(worst_j) if worst_j is not None else None,
        }
        if names is not None:
            row["name"] = str(names[k])
            if worst_j is not None:
                row["worst_other_name"] = str(names[worst_j])
        out.append(row)
    return out


def bootstrap_gap_ci(
    X: np.ndarray,
    y: np.ndarray | Sequence[int],
    names: Sequence[str] | None = None,
    n_resamples: int = 100,
    seed: int = 0,
    ci: float = 0.95,
) -> list[dict]:
    """Per-artist bootstrap 95% CI on the discrimination gap.

    Resamples each artist's anchors with replacement at the same per-artist
    size. Cross-class pools are resampled jointly.

    Returns rows with: artist_id, name (if given), n_anchors, gap (point
    estimate), gap_ci_lo, gap_ci_hi, classification (one of
    'robust_negative', 'ambiguous', 'robust_positive').
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    point = {row["artist_id"]: row for row in discrimination_gap(X, y, names)}
    by_artist = _per_artist_indices(y)
    artist_ids = sorted(by_artist.keys())

    samples: dict[int, list[float]] = {k: [] for k in artist_ids}
    for _ in range(n_resamples):
        new_idx = []
        new_y = []
        for k in artist_ids:
            ids = by_artist[k]
            chosen = rng.choice(ids, size=ids.size, replace=True)
            new_idx.append(chosen)
            new_y.append(np.full(ids.size, k))
        new_idx = np.concatenate(new_idx)
        new_y = np.concatenate(new_y)
        rows = discrimination_gap(X[new_idx], new_y)
        for r in rows:
            samples[r["artist_id"]].append(r["gap"])

    alpha = (1 - ci) / 2.0
    out = []
    for k in artist_ids:
        arr = np.asarray(samples[k])
        lo = float(np.quantile(arr, alpha))
        hi = float(np.quantile(arr, 1 - alpha))
        if hi < 0:
            cls = "robust_negative"
        elif lo > 0:
            cls = "robust_positive"
        else:
            cls = "ambiguous"
        row = dict(point[k])
        row.update({
            "gap_ci_lo": lo,
            "gap_ci_hi": hi,
            "classification": cls,
        })
        out.append(row)
    return out
