# csd_plus — discrimination-gap diagnostic and CSLS readout for CSD

Reference implementation accompanying *When Style Similarity Scores Fail:
Diagnosing Raw CSD Cosine in Artist-Style Evaluation* (Frochte 2026,
arXiv:2605.09030).

## What this is

The Contrastive Style Descriptor (CSD) of Somepalli et al. 2024 is widely
used as an absolute style-fidelity score for text-to-image evaluation. The
paper shows that this absolute reading is corpus-dependent and fails for a
substantial fraction of artists in pairs from the same art-historical
tradition. It also shows that CSLS, a hubness-correction readout from the
cross-lingual word-embedding literature, transfers to visual style
discrimination and recovers most of the failure modes without retraining.

This repository is a small, focused library for testing and applying both:

- `discrimination_gap(X, y)` — the corpus-internal validity diagnostic.
- `csls_readout(X, y, k=15)` — the zero-training readout correction.
- `bootstrap_gap_ci(X, y)` — per-artist confidence intervals on the gap.
- `CSDBackbone` — a thin loader for the CSD ViT-L/14 checkpoint of Somepalli
  et al. (Hugging Face mirror), so you can produce embeddings to feed into
  the diagnostic.

## Quick start

```bash
git clone https://github.com/hsbo-ai-ml-dm/csd-plus.git
cd csd-plus

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# End-to-end demo: fetches 30 images from Wikimedia (Hokusai, Hiroshige,
# Monet, Goya, Vermeer), embeds with CSD, prints the gap and CSLS readout.
python -m examples.demo
```

The demo runs in ~1 minute on a small GPU after a one-time CSD checkpoint
download (~1.2 GB, cached under `~/.cache/huggingface/`). It demonstrates
the negative-gap behaviour the paper studies on a deliberately small,
intra-tradition-overlapping corpus.

## Apply on your own corpus

The library expects an embedding matrix `X` (shape `(n, 768)`, can be
unnormalised) and an integer-coded artist label vector `y` (shape `(n,)`).
You can use `CSDBackbone` to compute `X` from a list of PIL images, or
plug in any embeddings you already have.

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

for r in raw:
    print(f"{r['name']}: gap={r['gap']:+.3f} (worst-other: {r['worst_other_name']})")
```

A negative `gap` means raw CSD cosine is order-inverted on that artist
against at least one other artist on the corpus you passed in — i.e. the
absolute same-versus-different reading is not interpretable for them on
this corpus. CSLS attempts to lift such artists; the residual that survives
both is the shared-tradition limit the paper analyses.

## What this is *not*

- Not a new style encoder. CSD is from Somepalli et al. 2024 and is loaded
  unchanged.
- Not a downstream style-transfer or evaluation pipeline. We provide the
  diagnostic and the readout; the caller decides how to use them.
- Not a polished production library. This is a research artefact intended
  to make the paper's methods easy to reuse.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA strongly recommended for embedding; the diagnostic
  itself is CPU-fine)
- `pip install -r requirements.txt`

## License

Code: MIT. CSD weights are downloaded from the Hugging Face mirror
`yuxi-liu-wired/CSD` and remain under their respective license; this
repository does not redistribute them.

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
