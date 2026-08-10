"""Regression tests for the pos-interp input variant.

The library once documented an ``embed_posinterp336`` helper that did not
exist, while ``embed(input_size=336)`` raised a tensor-shape error because the
positional embeddings were left at their trained 16x16 grid. These tests pin
the behaviour that replaced it.
"""
import numpy as np
import pytest
from PIL import Image

from csd_plus.csd_backbone import _PATCH, _TRAINED_INPUT, CSDBackbone


@pytest.fixture(scope="module")
def backbone():
    return CSDBackbone(device="cpu")


@pytest.fixture(scope="module")
def image():
    rng = np.random.default_rng(0)
    return Image.fromarray(
        rng.integers(0, 255, (320, 480, 3), dtype=np.uint8), mode="RGB")


@pytest.mark.parametrize("size", [_TRAINED_INPUT, 280, 336])
def test_embed_runs_and_is_normalised(backbone, image, size):
    v = backbone.embed(image, input_size=size)
    assert v.shape == (768,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-4)


def test_resolutions_differ(backbone, image):
    a = backbone.embed(image, input_size=_TRAINED_INPUT)
    b = backbone.embed(image, input_size=336)
    assert float(a @ b) < 0.999, "336 must not silently equal the 224 pipeline"


def test_positional_embeddings_are_restored(backbone, image):
    before = backbone._model.backbone.positional_embedding
    backbone.embed(image, input_size=336)
    after = backbone._model.backbone.positional_embedding
    assert after is before, "the 224 grid must be restored after the call"


def test_non_multiple_of_patch_size_is_rejected(backbone, image):
    with pytest.raises(ValueError, match=str(_PATCH)):
        backbone.embed(image, input_size=_TRAINED_INPUT + 1)
