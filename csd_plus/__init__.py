"""csd_plus — discrimination-gap diagnostic and CSLS readout for CSD.

Reference implementation accompanying *When Style Similarity Scores Fail*
(Frochte 2026, arXiv:2605.09030).

Public API:

    discrimination_gap(X, y, names=None) -> DataFrame
        Per-artist within/cross median cosines and the discrimination gap.

    csls_readout(X, y, k=15) -> DataFrame
        Same per-artist statistics under CSLS-corrected pairwise cosines.
        The density term r_k(x) excludes only x itself, so when the reference
        pool is the evaluation corpus it also contains other works by x's own
        artist and measures within-artist cohesion alongside cross-artist
        local density. Pass exclude_class=True (optionally balance_pool=True)
        to isolate the cross-artist component; both need the query label and
        are decomposition instruments rather than readouts.

    bootstrap_gap_ci(X, y, n_resamples=1000, seed=0) -> DataFrame
        Per-artist bootstrap 95% confidence intervals on the gap.

    CSDBackbone
        Thin wrapper around the CSD ViT-L/14 checkpoint of Somepalli et al.
        Provides .embed(image) -> np.ndarray (768,) on a CUDA or CPU device.

Typical use on a third-party corpus is six lines:

    >>> from csd_plus import CSDBackbone, discrimination_gap, csls_readout
    >>> backbone = CSDBackbone(device='cuda')
    >>> X = np.stack([backbone.embed(img) for img in my_images])
    >>> y = np.array(my_artist_labels)
    >>> gaps = discrimination_gap(X, y)
    >>> csls = csls_readout(X, y, k=15)
"""
from .diagnostic import discrimination_gap, bootstrap_gap_ci
from .csls import csls_readout, csls_pairwise_matrix
from .csd_backbone import CSDBackbone

__all__ = [
    "discrimination_gap",
    "bootstrap_gap_ci",
    "csls_readout",
    "csls_pairwise_matrix",
    "CSDBackbone",
]

__version__ = "1.0.0"
