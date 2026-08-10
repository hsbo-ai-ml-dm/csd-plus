"""The pairwise and aggregated regimes must stay distinguishable.

The library first shipped only the pairwise gap while the paper's headline
reduction is aggregated, so a user reproducing the abstract got a different
number with no hint why. These tests pin that both regimes exist, that they
are not accidentally the same computation, and that the CSLS readout only
ever moves the gap in the direction it is meant to.
"""
import numpy as np
import pytest

from csd_plus import aggregated_gap, csls_readout, discrimination_gap


@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(0)
    K, per, d = 8, 12, 32
    centres = rng.normal(size=(K, d))
    X = np.concatenate([centres[k] + 0.6 * rng.normal(size=(per, d))
                        for k in range(K)])
    y = np.repeat(np.arange(K), per)
    return X, y


def test_both_regimes_return_one_row_per_artist(corpus):
    X, y = corpus
    assert len(discrimination_gap(X, y)) == len(np.unique(y))
    assert len(aggregated_gap(X, y)) == len(np.unique(y))


def test_regimes_are_different_computations(corpus):
    X, y = corpus
    pw = {r["artist_id"]: r["gap"] for r in discrimination_gap(X, y)}
    ag = {r["artist_id"]: r["gap"] for r in aggregated_gap(X, y)}
    assert any(abs(pw[k] - ag[k]) > 1e-6 for k in pw), \
        "aggregated must not collapse to the pairwise computation"


def test_aggregated_gap_larger_than_pairwise_on_average(corpus):
    # Pooling averages out per-pair variance, so the aggregated within term
    # sits above the pairwise one and gaps are typically wider.
    X, y = corpus
    pw = np.mean([r["gap"] for r in discrimination_gap(X, y)])
    ag = np.mean([r["gap"] for r in aggregated_gap(X, y)])
    assert ag > pw


def test_names_are_passed_through(corpus):
    X, y = corpus
    names = [f"artist {i}" for i in range(len(np.unique(y)))]
    row = aggregated_gap(X, y, names=names)[0]
    assert row["name"] == "artist 0"
    assert "worst_other_name" in row


def test_unknown_readout_is_rejected(corpus):
    X, y = corpus
    with pytest.raises(ValueError, match="readout"):
        aggregated_gap(X, y, readout="euclidean")


def test_exclude_class_changes_the_density_term(corpus):
    X, y = corpus
    a = [r["gap"] for r in aggregated_gap(X, y, readout="csls")]
    b = [r["gap"] for r in aggregated_gap(X, y, readout="csls",
                                          exclude_class=True)]
    assert not np.allclose(a, b)


def test_csls_readout_stays_pairwise(corpus):
    X, y = corpus
    cs = {r["artist_id"]: r["gap"] for r in csls_readout(X, y, k=5)}
    ag = {r["artist_id"]: r["gap"] for r in aggregated_gap(X, y, readout="csls",
                                                          k=5)}
    assert any(abs(cs[k] - ag[k]) > 1e-6 for k in cs)
