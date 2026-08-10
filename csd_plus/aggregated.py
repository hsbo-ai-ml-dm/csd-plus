"""Aggregated-pool discrimination gap.

Two regimes are in circulation and they give different numbers, so the
distinction matters when reproducing a figure.

*Pairwise* (``diagnostic.discrimination_gap``) compares individual artwork
pairs: the within term is the median cosine over pairs inside an artist, the
cross term the median over pairs between two artists.

*Aggregated* is what style-fidelity evaluations actually compute. A candidate
image is scored against an artist's whole anchor pool at once,

    s(x, A) = mean_{a in A} cos(x, a),

with the within-class score computed leave-one-out so that cos(x, x) = 1 does
not enter. The gap is then the median of an artist's own scores minus the
largest median score any other artist's works achieve against that pool.

Averaging over the pool cancels part of the per-pair variance, so the
aggregated regime flags fewer artists than the pairwise one on the same
corpus. The headline reduction reported in the paper (12 of 89 under raw
cosine, 4 of 89 under the CSLS readout) is aggregated; ``csls_readout``
returns the pairwise counterpart and will not match it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .csls import _local_density, _l2_normalize, _per_artist_indices

__all__ = ["aggregated_gap"]


def aggregated_gap(
    X: np.ndarray,
    y: np.ndarray | Sequence[int],
    names: Sequence[str] | None = None,
    readout: str = "cosine",
    k: int = 15,
    exclude_class: bool = False,
) -> list[dict]:
    """Per-artist aggregated-pool gap.

    Args:
        X: (n, d) embeddings; rows need not be L2-normalised.
        y: (n,) integer artist labels.
        names: optional length-K artist names for the output rows.
        readout: ``"cosine"`` for the plain pooled score or ``"csls"`` for the
            density-corrected one.
        k: neighbourhood size of the density term, used when readout="csls".
        exclude_class: drop same-artist works from each density term. Isolates
            the cross-artist component of the correction; needs the query
            label and is therefore a decomposition instrument rather than a
            readout. See ``csls.csls_pairwise_matrix``.

    Returns:
        One dict per artist with artist_id, name (if given), n_anchors, w_k,
        c_k, gap, worst_other_id and worst_other_name — the same fields as
        ``diagnostic.discrimination_gap``, computed in the aggregated regime.
    """
    if readout not in ("cosine", "csls"):
        raise ValueError(f"readout must be 'cosine' or 'csls', got {readout!r}")
    Xn = _l2_normalize(np.asarray(X, dtype=np.float32))
    y = np.asarray(y)
    C = Xn @ Xn.T
    by_artist = _per_artist_indices(y)
    artist_ids = sorted(by_artist.keys())

    r = None
    if readout == "csls":
        r = _local_density(C, y, k=k, exclude_class=exclude_class)

    out = []
    for a in artist_ids:
        ia = by_artist[a]
        if ia.size < 2:
            continue
        # within: every anchor of a scored against the rest of a's pool
        within = []
        for i in ia:
            pool = ia[ia != i]
            mean_cos = float(C[i, pool].mean())
            if r is None:
                within.append(mean_cos)
            else:
                within.append(2.0 * mean_cos - r[i] - float(r[pool].mean()))
        w_k = float(np.median(within))

        r_pool = None if r is None else float(r[ia].mean())
        worst_j, worst_val = None, -np.inf
        for b in artist_ids:
            if b == a:
                continue
            ib = by_artist[b]
            if ib.size == 0:
                continue
            mean_cos = C[np.ix_(ib, ia)].mean(axis=1)
            scores = (mean_cos if r is None
                      else 2.0 * mean_cos - r[ib] - r_pool)
            v = float(np.median(scores))
            if v > worst_val:
                worst_val, worst_j = v, b

        row = {
            "artist_id": a,
            "n_anchors": int(ia.size),
            "w_k": w_k,
            "c_k": float(worst_val),
            "gap": float(w_k - worst_val),
            "worst_other_id": int(worst_j) if worst_j is not None else None,
        }
        if names is not None:
            row["name"] = str(names[a])
            if worst_j is not None:
                row["worst_other_name"] = str(names[worst_j])
        out.append(row)
    return out
