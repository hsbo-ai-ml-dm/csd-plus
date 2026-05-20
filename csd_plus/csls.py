"""CSLS readout for visual style discrimination.

CSLS (Conneau et al. 2018) was originally introduced for cross-lingual
lexicon induction. The readout removes the inflation of cosine values
in regions of the embedding where many points are mutually close — the
exact failure mode the discrimination diagnostic exposes.

For two embeddings x, y on a corpus X with k-nearest-neighbour mean
cosine r_k(z) = (1/k) sum_{w in N_k(z)} <z, w>,

    csls(x, y) = 2 * cos(x, y) - r_k(x) - r_k(y)

This module provides a pairwise CSLS matrix and a wrapper that returns
per-artist within / cross / gap statistics, matching the API of
:mod:`diagnostic`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)


def csls_pairwise_matrix(X: np.ndarray, k: int = 15) -> np.ndarray:
    """Compute the pairwise CSLS matrix on a single embedding pool.

    Args:
        X: (n, d) embeddings.
        k: number of neighbours for the local-density estimator.

    Returns:
        (n, n) CSLS matrix C with C[i, j] = 2*cos(x_i, x_j) - r_k(x_i) - r_k(x_j).
        The diagonal is set to 2 - 2*r_k(x_i).
    """
    X = _l2_normalize(np.asarray(X, dtype=np.float32))
    cos = X @ X.T
    n = cos.shape[0]
    # r_k(x) = mean cosine to top-k neighbours, excluding self.
    eff_k = min(k, n - 1)
    if eff_k <= 0:
        raise ValueError(f"need at least 2 points for CSLS, got n={n}")
    # mask self by setting diagonal very low temporarily
    cos_off = cos.copy()
    np.fill_diagonal(cos_off, -np.inf)
    # top-k along each row
    topk = -np.partition(-cos_off, eff_k - 1, axis=1)[:, :eff_k]
    r = topk.mean(axis=1)
    csls = 2.0 * cos - r[:, None] - r[None, :]
    return csls


def _per_artist_indices(y: np.ndarray) -> dict[int, np.ndarray]:
    return {int(c): np.where(y == c)[0] for c in np.unique(y)}


def _within_median(C: np.ndarray, idx: np.ndarray) -> float:
    if idx.size < 2:
        return float("nan")
    sub = C[np.ix_(idx, idx)]
    iu = np.triu_indices(idx.size, k=1)
    return float(np.median(sub[iu]))


def _cross_median(C: np.ndarray, idx_k: np.ndarray, idx_j: np.ndarray) -> float:
    if idx_k.size == 0 or idx_j.size == 0:
        return float("nan")
    return float(np.median(C[np.ix_(idx_k, idx_j)]))


def csls_readout(
    X: np.ndarray,
    y: np.ndarray | Sequence[int],
    k: int = 15,
    names: Sequence[str] | None = None,
) -> list[dict]:
    """Per-artist CSLS-corrected within / cross / gap statistics.

    Same contract as :func:`diagnostic.discrimination_gap` but on the
    CSLS-transformed pairwise matrix. The reference pool over which local
    densities are estimated is the candidate corpus itself.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    C = csls_pairwise_matrix(X, k=k)

    by_artist = _per_artist_indices(y)
    artist_ids = sorted(by_artist.keys())

    out = []
    for art_k in artist_ids:
        idx_k = by_artist[art_k]
        w_k = _within_median(C, idx_k)
        worst_j = None
        worst_val = -np.inf
        for art_j in artist_ids:
            if art_j == art_k:
                continue
            idx_j = by_artist[art_j]
            v = _cross_median(C, idx_k, idx_j)
            if v > worst_val:
                worst_val = v
                worst_j = art_j
        c_k = float(worst_val)
        g_k = w_k - c_k
        row = {
            "artist_id": art_k,
            "n_anchors": int(idx_k.size),
            "w_k": float(w_k),
            "c_k": float(c_k),
            "gap": float(g_k),
            "worst_other_id": int(worst_j) if worst_j is not None else None,
            "k_csls": int(k),
        }
        if names is not None:
            row["name"] = str(names[art_k])
            if worst_j is not None:
                row["worst_other_name"] = str(names[worst_j])
        out.append(row)
    return out
