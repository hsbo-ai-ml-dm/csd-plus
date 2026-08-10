"""csd_plus — discrimination-gap diagnostic and CSLS readout for CSD.

Reference implementation accompanying *When Style Similarity Scores Fail*
(Frochte 2026, arXiv:2605.09030).

Which function reproduces which published number:

    paper                                    call
    -------------------------------------    ------------------------------
    pairwise, raw cosine        20 of 89     discrimination_gap
    pairwise, CSLS               8 of 89     csls_readout
    aggregated, raw cosine      12 of 89     aggregated_gap(readout='cosine')
    aggregated, CSLS             4 of 89     aggregated_gap(readout='csls')
    aggregated, class-excluded   9 of 89     aggregated_gap(readout='csls',
                                                 exclude_class=True)

The abstract quotes the aggregated regime, because that is what a
style-fidelity evaluation computes when it scores an image against a whole
anchor pool. Reading a pairwise number against an aggregated claim is the
easiest way to conclude, wrongly, that the numbers do not reproduce.

Not in this library: the pair-verification harness of the paper's §8, that
is, artist-disjoint splitting, pair sampling, the inductive CSLS reference
gallery and the pair-feature logistic regression. It is evaluation
scaffolding rather than a metric, and it ships with the paper's
supplementary material instead.

Public API:

    discrimination_gap(X, y, names=None) -> list[dict]
        Per-artist within/cross median cosines and the pairwise
        discrimination gap.

    aggregated_gap(X, y, names=None, readout='cosine', k=15) -> list[dict]
        The same statistics in the aggregated-pool regime, where each work is
        scored against an artist's entire pool. This is the regime the paper's
        headline reduction is reported in.

    csls_readout(X, y, k=15) -> list[dict]
        Per-artist statistics under CSLS-corrected pairwise cosines.
        The density term r_k(x) excludes only x itself, so when the reference
        pool is the evaluation corpus it also contains other works by x's own
        artist and measures within-artist cohesion alongside cross-artist
        local density. Pass exclude_class=True (optionally balance_pool=True)
        to isolate the cross-artist component; both need the query label and
        are decomposition instruments rather than readouts.

    bootstrap_gap_ci(X, y, n_resamples=1000, seed=0) -> list[dict]
        Per-artist bootstrap 95% confidence intervals on the gap.

    CSDBackbone
        Thin wrapper around the CSD ViT-L/14 checkpoint of Somepalli et al.
        Provides .embed(image, input_size=224) -> np.ndarray (768,) on a CUDA
        or CPU device. input_size=336 gives the pos-interp variant of the
        paper: the positional embeddings are resampled to the 24x24 patch grid
        for the duration of the call. Any multiple of the patch size 14 works;
        280 performs on par with 336 at two thirds of the cost. A larger input
        samples the same centre crop more finely and does not widen the field
        of view.

Typical use on a third-party corpus is six lines:

    >>> from csd_plus import CSDBackbone, discrimination_gap, csls_readout
    >>> backbone = CSDBackbone(device='cuda')
    >>> X = np.stack([backbone.embed(img) for img in my_images])
    >>> y = np.array(my_artist_labels)
    >>> gaps = discrimination_gap(X, y)          # pairwise
    >>> agg = aggregated_gap(X, y, readout='csls')  # the headline regime
"""
from .diagnostic import discrimination_gap, bootstrap_gap_ci
from .csls import csls_readout, csls_pairwise_matrix
from .aggregated import aggregated_gap
from .csd_backbone import CSDBackbone

__all__ = [
    "discrimination_gap",
    "aggregated_gap",
    "bootstrap_gap_ci",
    "csls_readout",
    "csls_pairwise_matrix",
    "CSDBackbone",
]

__version__ = "1.1.0"
