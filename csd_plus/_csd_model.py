"""Vendored CSD (Contrastive Style Descriptor) model for style-similarity scoring.

CSD is a ViT-L/14 visual tower (OpenAI-CLIP key-structure) with a style
projection head. The checkpoint we load is the community safetensors mirror
at ``yuxi-liu-wired/CSD`` on HuggingFace, which is the Somepalli-et-al.-2024
weights (https://arxiv.org/abs/2404.01292) re-packaged.

We vendor the architecture instead of depending on OpenAI's ``clip`` package
for two reasons:

1. ``pip install git+https://github.com/openai/CLIP.git`` is not available in
   this environment and we want the repo self-contained.
2. The OpenAI CLIP package downloads ~900 MB of vanilla CLIP weights purely
   to be immediately overwritten by CSD's state_dict. Skipping that is a
   net win.

The CLIP ViT-L/14 visual tower is a well-known architecture; the state_dict
key-names below match OpenAI CLIP exactly so the CSD safetensors load
cleanly with ``strict=True``.

**Only the style branch** is evaluated in forward — CSD's ``last_layer_content``
weight is loaded but not used. For our sieve we treat the style embedding
as the "how Van-Gogh-ish is this image" signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential()
        self.mlp.add_module("c_fc", nn.Linear(d_model, d_model * 4))
        self.mlp.add_module("gelu", QuickGELU())
        self.mlp.add_module("c_proj", nn.Linear(d_model * 4, d_model))
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """OpenAI-CLIP ViT-L/14 visual tower.

    Default hyperparams match CLIP ViT-L/14:
      input_resolution=224, patch_size=14, width=1024, layers=24, heads=16,
      output_dim=768.

    The ``proj`` parameter is set to ``None`` when used inside CSD_CLIP so
    that style/content heads can project from the raw 1024-dim feature.
    """

    def __init__(
        self,
        input_resolution: int = 224,
        patch_size: int = 14,
        width: int = 1024,
        layers: int = 24,
        heads: int = 16,
        output_dim: int = 768,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        n_patches = (input_resolution // patch_size) ** 2
        self.positional_embedding = nn.Parameter(scale * torch.randn(n_patches + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj: Optional[nn.Parameter] = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)                          # [B, 1024, 16, 16]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, 1024, 256]
        x = x.permute(0, 2, 1)                     # [B, 256, 1024]
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device,
        )
        x = torch.cat([cls, x], dim=1)             # [B, 257, 1024]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)                     # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                     # LND -> NLD
        x = self.ln_post(x[:, 0, :])               # CLS row
        if self.proj is not None:
            x = x @ self.proj
        return x


class CSD_CLIP(nn.Module):
    """Minimal CSD model matching yuxi-liu-wired/CSD safetensors keys.

    ``forward(x)`` returns the L2-normalized **style** embedding (768-dim).
    We deliberately drop the (feature, content_output, style_output) tuple
    from the upstream module because only the style projection is needed for
    the sieve.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = VisionTransformer()
        # Initialize style/content with scaled Gaussian matching the
        # backbone.proj scale so random-weight forwards produce finite
        # non-zero L2-normalized outputs (needed for unit tests; real
        # weights overwrite these at load time).
        scale = 1024 ** -0.5
        self.last_layer_style = nn.Parameter(scale * torch.randn(1024, 768))
        self.last_layer_content = nn.Parameter(scale * torch.randn(1024, 768))
        self.backbone.proj = None

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(pixel_values)            # [B, 1024]
        style = feature @ self.last_layer_style          # [B, 768]
        return F.normalize(style, dim=-1, p=2)


_CSD_REPO = "yuxi-liu-wired/CSD"
_CSD_FILE = "model.safetensors"


def load_csd_state_dict(
    repo_id: str = _CSD_REPO,
    filename: str = _CSD_FILE,
    local_path: Optional[Path] = None,
) -> dict:
    """Load a CSD state_dict from either a local file or the HF hub.

    When ``local_path`` is given it must point at a ``.safetensors`` file.
    Otherwise the weights are fetched via ``hf_hub_download`` which caches
    under ``~/.cache/huggingface/hub/`` and returns a stable local path.
    """
    from safetensors.torch import load_file

    if local_path is not None:
        path = Path(local_path)
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=repo_id, filename=filename))

    sd = load_file(str(path))
    # Some mirrors prefix keys with "model." — strip if present
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    return sd


def build_csd_from_state_dict(sd: dict) -> CSD_CLIP:
    """Instantiate CSD_CLIP and load weights, tolerating ``backbone.proj``
    being present (upstream save) or absent (our convention)."""
    model = CSD_CLIP()
    # If the checkpoint still carries backbone.proj, drop it — we don't use it.
    sd = {k: v for k, v in sd.items() if k != "backbone.proj"}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # backbone.proj is intentionally missing (set to None in __init__)
    missing = [k for k in missing if k != "backbone.proj"]
    if missing:
        raise RuntimeError(f"CSD load: missing keys {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"CSD load: unexpected keys {unexpected[:10]}")
    return model
