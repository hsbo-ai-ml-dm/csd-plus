---
layout: default
title: "When Style Similarity Scores Fail"
---

# When Style Similarity Scores Fail
### Diagnosing Raw CSD Cosine in Artist-Style Evaluation

**Jörg Frochte** — [Hochschule Bochum](https://www.hochschule-bochum.de/), Department of Electrical Engineering and Computer Science

[![arXiv](https://img.shields.io/badge/arXiv-2605.09030-b31b1b.svg)](https://arxiv.org/abs/2605.09030)
[![Code](https://img.shields.io/badge/code-GitHub-181717.svg)](https://github.com/hsbo-ai-ml-dm/csd-plus)
[![License](https://img.shields.io/badge/license-MIT-E2001A.svg)](https://github.com/hsbo-ai-ml-dm/csd-plus/blob/main/LICENSE)

---

## TL;DR

Raw cosine in the 768-dimensional output space of the Contrastive Style
Descriptor (CSD) is now widely read as an **absolute, calibrated**
style-fidelity score for text-to-image and style-imitation evaluation. We
show that this absolute reading is corpus-dependent and fails for a
substantial fraction of artists drawn from the same art-historical
tradition. We give:

- a **corpus-internal diagnostic** — the *discrimination gap* — that
  detects when raw cosine is order-inverted on a candidate corpus, with no
  prototypes and no threshold tuning;
- a **zero-training readout fix** — CSLS, imported from the cross-lingual
  word-embedding literature — that lifts most of the failure modes on a
  frozen backbone;
- a **cross-backbone replication** on CLIP-ViT-L/14, SigLIP-large and
  DINOv2-Large showing the residual failure pattern is shared across
  contrastive vision backbones rather than CSD-specific.

We call the diagnostic-driven readout protocol on the frozen backbone
*CSD+*. CSD+ is **not a new encoder**.

## Headline numbers (corpus: 1645 artworks, 89 artists)

| Setting | Negative-gap artists (aggregated-pool) | Pair-verification AUC |
|---|---|---|
| Raw CSD cosine | 12 / 89 | 0.891 |
| CSLS readout (k = 15) | 4 / 89 | 0.905 |
| CSLS + pos-interp 336 | — | **0.911** |

Across 25 artist-disjoint splits. The count reduction replicates on two
corpora we did not build (10 → 3 on the WikiArt dump, 479 → 298 on
ArtBench-10), and the verification gain concentrates in the
within-tradition pair regime the diagnostic flags (+0.019 to +0.054 AUC
across four backbones) rather than on randomly drawn pairs.

### Which call reproduces which number

The paper reports two regimes and they give different counts, so pick the one
the figure you are checking uses.

| paper | count | call |
|---|---|---|
| pairwise, raw cosine | 20 / 89 | `discrimination_gap` |
| pairwise, CSLS | 8 / 89 | `csls_readout` |
| aggregated, raw cosine | 12 / 89 | `aggregated_gap(readout='cosine')` |
| aggregated, CSLS | 4 / 89 | `aggregated_gap(readout='csls')` |
| aggregated, class-excluded | 9 / 89 | `aggregated_gap(readout='csls', exclude_class=True)` |

The headline numbers quoted above and in the abstract are **aggregated**: each
work is scored against an artist's whole anchor pool, which is what a
style-fidelity evaluation actually does. The pairwise regime compares
individual pairs and flags more artists on the same corpus.

The pair-verification harness of the paper (artist-disjoint splits, pair
sampling, the inductive CSLS reference gallery, the pair-feature logistic
regression) is not part of this library — it is evaluation scaffolding rather
than a metric, and it ships with the paper's supplementary material.

### Input resolution

`CSDBackbone.embed(image, input_size=...)` runs the checkpoint above its
training resolution by resampling the positional embeddings to the patch grid
the input implies, which is the *pos-interp* variant of the paper. Any
multiple of the patch size 14 is accepted; 336 is what the paper reports and
280 performs on par with it at two thirds of the cost. Which of the two leads
depends on the readout — 280 under CSLS, 336 under raw cosine, both by about
0.001 AUC — so there is no optimum to chase between them. Beyond 336 the grid
is far enough from its trained 16x16 size that every measure degrades.

A larger input does **not** widen the field of view: the preprocessing
resizes the shortest side and takes a square centre crop, so the visible
fraction of the artwork is the same at every input size. Only the sampling
density changes.

### What the CSLS density term corrects for

`r_k(x)` is the mean cosine to `x`'s k nearest reference neighbours, and only
`x` itself is excluded. When the reference pool is the evaluation corpus, the
neighbourhood therefore also holds other works by `x`'s own artist — at
k = 15 and ~19 works per artist, about 40 percent of them. `r_k(x)` then
normalises within-artist cohesion as well as cross-artist local density, and
roughly half of the reduction from 12 to 4 negative gaps comes from the
former. Passing `exclude_class=True` isolates the cross-artist component and
leaves 9 of 89; adding `balance_pool=True` gives 8 of 89. Both need the
query's artist label, so they decompose the effect rather than replace the
readout: in a real query the artist is what you are trying to establish. In
the paper's inductive verification protocol the reference gallery holds only
training artists, so the same-artist share is zero and the reported AUC gains
are purely cross-artist.

### Why a rate needs its configuration attached

Do not compare a negative-gap rate with one from another corpus unless both
were measured at the same artist count and the same number of works per
artist. Two effects make a bare rate incomparable, and neither has anything to
do with how good the embedding is.

The cross term is a maximum over competitors. For a fixed artist,
`P(c_k > w_k)` is the probability that a maximum over `K-1` competitors clears
one fixed threshold, so it grows with the artist count `K` whatever the
competitors look like. Holding a corpus fixed and letting the maximum run over
random subsets of its own artists shows the size of it: on the WikiArt dump the
rate goes 4.5 % at 32 artists, 10.9 % at 89, 14.8 % at all 128, and on
ArtBench-10 it continues to 39.4 % at 1639. The gap between ArtBench's 39.4 %
and our 22.5 % is mostly this.

Both terms are medians over pairs, and thin per-artist sampling makes both
noisier — but not alike. The within term is one median; the cross term is a
maximum over `K-1` of them, and a maximum over noisy estimates is biased
upward. At a fixed 89 artists, WikiArt reads 27.5 % with ten works per artist,
19.3 % with 19 and 11.9 % with all of them. The inflating end is the end that
style-fidelity evaluation actually operates at, since anchor pools there hold
tens of works rather than hundreds.

Match both and the three corpora agree to within a few points: 25.4 %, 27.5 %
and 19.4 % at 89 artists with ten works each. `negative_rate_at` computes a
rate at a configuration you state:

```python
from csd_plus import negative_rate_at

negative_rate_at(X, y)                                    # your corpus as it is
negative_rate_at(X, y, n_artists=89, works_per_artist=10) # comparable to the above
```

What does survive both controls is how densely traditions are sampled. At a
constant artist count, constant anchor depth and inside one corpus, drawing
artists from a single style class rather than ten moves the rate from 29.9 % to
16.4 %. That is the property you control when assembling an evaluation pool.

## Practical implication

Before reporting CSD cosine as an absolute style-fidelity score for a
text-to-image evaluation, run the diagnostic on the candidate corpus. No
published rate transfers to it, for the reasons above, so this step cannot be
skipped by citing one. If a non-trivial fraction of artists exhibit negative
gaps, CSLS is the minimal correction. A positive gap means the necessary
median-order condition holds; it is not by itself a certificate of
calibration.

## Reference implementation

The companion library `csd_plus` provides the diagnostic, the CSLS
readout, and a thin loader for the CSD backbone. Six lines on your own
corpus:

```python
import numpy as np
from PIL import Image
from csd_plus import CSDBackbone, discrimination_gap, csls_readout

backbone = CSDBackbone()                       # CUDA if available, else CPU
images   = [Image.open(p).convert("RGB") for p in my_paths]
X        = np.stack([backbone.embed(im) for im in images])
y        = np.array(my_artist_labels)          # ints, one per image

raw  = discrimination_gap(X, y, names=my_artist_names)
csls = csls_readout(X, y, k=15, names=my_artist_names)
```

A self-contained demo on five public-domain Wikimedia artists runs in
about a minute after a one-time CSD checkpoint download:

```bash
git clone https://github.com/hsbo-ai-ml-dm/csd-plus.git
cd csd-plus
pip install -r requirements.txt
python -m examples.demo
```

## Citation

```bibtex
@article{frochte2026csd,
  title  = {When Style Similarity Scores Fail: Diagnosing Raw CSD Cosine
            in Artist-Style Evaluation},
  author = {Frochte, J\"org},
  journal= {arXiv preprint arXiv:2605.09030},
  year   = {2026}
}
```

## License

Code is released under MIT. The CSD weights are downloaded from the
community mirror `yuxi-liu-wired/CSD` on HuggingFace and remain under
their respective license; this project does not redistribute them.
