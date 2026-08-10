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


def csls_pairwise_matrix(
    X: np.ndarray,
    k: int = 15,
    y: np.ndarray | Sequence[int] | None = None,
    exclude_class: bool = False,
    balance_pool: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Compute the pairwise CSLS matrix on a single embedding pool.

    The density term r_k(x) is the mean cosine to x's k nearest reference
    neighbours. Only x itself is excluded by default, so when the reference
    pool is the evaluation corpus the neighbourhood can contain other works
    by x's own artist. It typically does: on a corpus with ~19 works per
    artist and k=15, roughly 40 percent of a work's nearest neighbours share
    its artist, and r_k(x) then measures within-artist cohesion as well as
    cross-artist local density. Keep that in mind when interpreting a
    reduction in negative gaps as hubness removal.

    ``exclude_class`` drops works by the query's own artist from its density
    term, which isolates the cross-artist component; ``balance_pool``
    additionally subsamples the reference pool to an equal number of works
    per artist, removing pool-size effects. Both need ``y`` and both need the
    query's label, so they are instruments for decomposing the effect rather
    than readouts you can apply to an unlabelled query.

    Args:
        X: (n, d) embeddings.
        k: number of neighbours for the local-density estimator.
        y: (n,) artist labels; required if exclude_class or balance_pool.
        exclude_class: remove same-artist works from each density term.
        balance_pool: additionally equalise works per artist in the pool.
        seed: subsampling seed for balance_pool.

    Returns:
        (n, n) CSLS matrix C with C[i, j] = 2*cos(x_i, x_j) - r_k(x_i) - r_k(x_j).
        The diagonal is set to 2 - 2*r_k(x_i).
    """
    X = _l2_normalize(np.asarray(X, dtype=np.float32))
    cos = X @ X.T
    n = cos.shape[0]
    eff_k = min(k, n - 1)
    if eff_k <= 0:
        raise ValueError(f"need at least 2 points for CSLS, got n={n}")
    if (exclude_class or balance_pool) and y is None:
        raise ValueError("exclude_class and balance_pool require y")

    # Reference mask for the density term: self is always excluded.
    cos_off = cos.copy()
    np.fill_diagonal(cos_off, -np.inf)
    if exclude_class or balance_pool:
        y = np.asarray(y)
        for c in np.unique(y):
            ic = np.where(y == c)[0]
            cos_off[np.ix_(ic, ic)] = -np.inf
    if balance_pool:
        rng = np.random.default_rng(seed)
        per = [np.where(y == c)[0] for c in np.unique(y)]
        m = min(len(i) for i in per)
        ref = np.concatenate([rng.permutation(i)[:m] for i in per])
        masked = np.full_like(cos_off, -np.inf)
        masked[:, ref] = cos_off[:, ref]
        cos_off = masked

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
    exclude_class: bool = False,
    balance_pool: bool = False,
    seed: int = 0,
) -> list[dict]:
    """Per-artist CSLS-corrected within / cross / gap statistics.

    Same contract as :func:`diagnostic.discrimination_gap` but on the
    CSLS-transformed pairwise matrix. The reference pool over which local
    densities are estimated is the candidate corpus itself.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    C = csls_pairwise_matrix(X, k=k, y=y, exclude_class=exclude_class,
                             balance_pool=balance_pool, seed=seed)

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
