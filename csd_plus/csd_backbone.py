"""Thin wrapper around the CSD ViT-L/14 checkpoint.

Loads the Somepalli-et-al.-2024 weights from the community safetensors
mirror at ``yuxi-liu-wired/CSD`` on HuggingFace (~1.2 GB on first run,
cached under ``~/.cache/huggingface/hub`` thereafter), and exposes a
single ``embed(image)`` method returning a 768-dim L2-normalised vector.

This is the same backbone evaluated in the paper. Default input is the
standard 224x224 centre-crop pipeline of Somepalli et al.; for the
pos-interp 336 variant studied in §5/§A2, see ``embed_posinterp336``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image as PImage

from . import _csd_model


_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)


def _resize_shortest(img: PImage.Image, target: int) -> PImage.Image:
    w, h = img.size
    scale = target / min(w, h)
    return img.resize((int(round(w * scale)), int(round(h * scale))), PImage.BICUBIC)


def _center_crop(img: PImage.Image, size: int) -> PImage.Image:
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _to_tensor(img: PImage.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - _CLIP_MEAN) / _CLIP_STD
    return torch.from_numpy(arr).unsqueeze(0)


class CSDBackbone:
    """Frozen CSD ViT-L/14 backbone with a simple ``embed(img)`` method.

    Args:
        device: ``"cuda"``, ``"cuda:0"``, ``"cpu"``, or a ``torch.device``.
            Defaults to CUDA if available, else CPU.
        local_path: optional path to a local CSD ``.safetensors`` file.
            If omitted, downloads from HuggingFace (cached).

    Example::

        backbone = CSDBackbone()
        img = PImage.open('foo.jpg').convert('RGB')
        z = backbone.embed(img)        # np.ndarray, shape (768,), L2-normalised
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        local_path: Optional[Path] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)
        sd = _csd_model.load_csd_state_dict(local_path=local_path)
        self._model = _csd_model.build_csd_from_state_dict(sd).to(self._device).eval()

    @torch.no_grad()
    def embed(self, image: PImage.Image, input_size: int = 224) -> np.ndarray:
        """Return the L2-normalised 768-dim style embedding of ``image``.

        Args:
            image: PIL RGB image (any size).
            input_size: 224 for the vanilla pipeline (default), 336 for the
                pos-interp variant. Note: pos-interp 336 requires the
                positional embeddings to be bilinearly interpolated to a
                24x24 grid; this method does not do that, see
                :func:`embed_posinterp336`.
        """
        img = image.convert("RGB")
        img = _resize_shortest(img, input_size)
        img = _center_crop(img, input_size)
        t = _to_tensor(img).to(self._device)
        z = self._model(t)  # already L2-normed by the style head
        return z.detach().cpu().float().numpy()[0]
