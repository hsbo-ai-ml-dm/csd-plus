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

## Practical implication

Before reporting CSD cosine as an absolute style-fidelity score for a
text-to-image evaluation, run the diagnostic on the candidate corpus. If a
non-trivial fraction of artists exhibit negative gaps, CSLS is the
minimal correction. A positive gap means the necessary median-order
condition holds; it is not by itself a certificate of calibration.

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
