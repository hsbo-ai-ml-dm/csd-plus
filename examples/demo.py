"""End-to-end demo: fetch -> embed -> discrimination gap -> CSLS readout.

Picks five public-domain artists from Wikimedia Commons (Hokusai, Hiroshige,
Monet, Goya, Vermeer), pulls six images per artist, embeds them with CSD,
and prints the per-artist discrimination gap before and after CSLS.

Run::

    python -m examples.demo

First run downloads ~30 images (<20 MB) and the CSD checkpoint (~1.2 GB,
cached under ~/.cache/huggingface). Subsequent runs use the cache and finish
in under a minute on a small GPU.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
from PIL import Image as PImage

from csd_plus import CSDBackbone, csls_readout, discrimination_gap


CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Five Wikimedia Commons categories. The first per artist is the primary
# category; if it returns too few usable items we try the fallback.
DEMO_ARTISTS = [
    {
        "name": "Hokusai",
        "categories": [
            "Ukiyo-e prints by Katsushika Hokusai",
            "Paintings by Katsushika Hokusai",
        ],
    },
    {
        "name": "Hiroshige",
        "categories": [
            "Ukiyo-e prints by Utagawa Hiroshige",
            "Paintings by Utagawa Hiroshige",
        ],
    },
    {
        "name": "Monet",
        "categories": [
            "Paintings by Claude Monet",
        ],
    },
    {
        "name": "Goya",
        "categories": [
            "Paintings by Francisco de Goya",
        ],
    },
    {
        "name": "Vermeer",
        "categories": [
            "Paintings by Johannes Vermeer",
        ],
    },
]

PER_ARTIST = 6
MIN_IMG_SIDE = 400        # skip tiny / icon-sized files
MAX_BYTES = 8 * 1024 ** 2  # 8 MB cap per file


def _wikimedia_list_files(category: str, limit: int = 30) -> list[dict]:
    """Return list of {title, url, width, height} for files in a Commons category."""
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}",
        "gcmtype": "file",
        "gcmlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
    }
    headers = {"User-Agent": "csd-plus-demo/1.0 (research; contact via repo)"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        info = page.get("imageinfo", [{}])[0]
        mime = info.get("mime", "")
        if mime not in ("image/jpeg", "image/png"):
            continue
        out.append({
            "title": page.get("title", ""),
            "url": info.get("url", ""),
            "width": info.get("width", 0),
            "height": info.get("height", 0),
            "size": info.get("size", 0),
        })
    return out


def _safe_filename(title: str) -> str:
    h = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    suffix = ".jpg" if title.lower().endswith((".jpg", ".jpeg")) else ".png"
    return f"{h}{suffix}"


def _download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        headers = {"User-Agent": "csd-plus-demo/1.0 (research; contact via repo)"}
        resp = requests.get(url, headers=headers, timeout=60, stream=True)
        resp.raise_for_status()
        total = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    f.close()
                    dest.unlink(missing_ok=True)
                    return False
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ! download failed: {e}", flush=True)
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def fetch_artist(artist: dict, target: int = PER_ARTIST) -> list[Path]:
    """Download up to ``target`` usable images for one artist into the cache."""
    name = artist["name"]
    artist_dir = CACHE_DIR / name.lower().replace(" ", "_")
    artist_dir.mkdir(parents=True, exist_ok=True)
    have = sorted(p for p in artist_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    if len(have) >= target:
        return have[:target]

    saved: list[Path] = list(have)
    for cat in artist["categories"]:
        if len(saved) >= target:
            break
        try:
            entries = _wikimedia_list_files(cat, limit=4 * target)
        except Exception as e:
            print(f"  ! category lookup failed for {cat}: {e}", flush=True)
            continue
        # Prefer larger, non-tiny images
        entries = [e for e in entries
                   if e["width"] >= MIN_IMG_SIDE and e["height"] >= MIN_IMG_SIDE
                   and 0 < e["size"] <= MAX_BYTES]
        for ent in entries:
            if len(saved) >= target:
                break
            dest = artist_dir / _safe_filename(ent["title"])
            if dest in saved:
                continue
            if _download(ent["url"], dest):
                saved.append(dest)
                print(f"  + {name}: {ent['title']}", flush=True)
            time.sleep(0.2)  # be polite to Wikimedia
    return saved[:target]


def main() -> int:
    print("=== csd_plus demo: fetch -> embed -> diagnostic -> CSLS ===\n")

    # 1) Fetch
    print(f"Step 1/4: fetching up to {PER_ARTIST} images per artist from Wikimedia Commons")
    print(f"          cache: {CACHE_DIR}")
    image_paths: list[Path] = []
    artist_names: list[str] = []
    for art in DEMO_ARTISTS:
        print(f"  [{art['name']}]")
        files = fetch_artist(art)
        if len(files) < 3:
            print(f"  ! only {len(files)} usable images for {art['name']}; "
                  f"skipping (the demo needs at least 3 per artist)")
            continue
        for p in files:
            image_paths.append(p)
            artist_names.append(art["name"])

    if not image_paths:
        print("\nFailed to fetch any images. Check your internet connection.")
        return 1
    print(f"\nFetched {len(image_paths)} images across "
          f"{len(set(artist_names))} artists.\n")

    # 2) Embed
    print("Step 2/4: loading CSD ViT-L/14 (cached under ~/.cache/huggingface)")
    backbone = CSDBackbone()
    print(f"          device: {backbone._device}")
    print("Step 3/4: embedding images")
    embs = []
    t0 = time.time()
    for i, p in enumerate(image_paths):
        with PImage.open(p) as im:
            z = backbone.embed(im)
        embs.append(z)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(image_paths)} in {time.time() - t0:.1f}s")
    X = np.stack(embs).astype(np.float32)
    print(f"  embedded {len(image_paths)} images in {time.time() - t0:.1f}s")

    # Map artist names to integer ids
    unique_names = sorted(set(artist_names))
    name_to_id = {n: i for i, n in enumerate(unique_names)}
    y = np.array([name_to_id[n] for n in artist_names], dtype=np.int64)

    # 3) Discrimination gap (raw cosine) and 4) CSLS
    print("\nStep 4/4: computing discrimination gap (raw cosine) and CSLS readout\n")
    raw = discrimination_gap(X, y, names=unique_names)
    csls = csls_readout(X, y, k=5, names=unique_names)
    csls_by_id = {r["artist_id"]: r for r in csls}

    print("Per-artist discrimination gap g_k = w_k - c_k")
    print(f"{'Artist':<12} {'n':>3} {'w_k':>7} {'c_k':>7} {'gap_raw':>9} "
          f"{'worst-other':<14} {'gap_CSLS':>9}")
    print("-" * 72)
    for r in sorted(raw, key=lambda r: r["gap"]):
        c = csls_by_id[r["artist_id"]]
        wo = r.get("worst_other_name", "?")
        flag_raw = "*" if r["gap"] < 0 else " "
        flag_csls = "*" if c["gap"] < 0 else " "
        print(f"{r['name']:<12} {r['n_anchors']:>3} {r['w_k']:>+7.3f} {r['c_k']:>+7.3f} "
              f"{r['gap']:>+8.3f}{flag_raw} {wo:<14} {c['gap']:>+8.3f}{flag_csls}")
    n_neg_raw = sum(1 for r in raw if r["gap"] < 0)
    n_neg_csls = sum(1 for r in csls if r["gap"] < 0)
    print("-" * 72)
    print(f"  negative-gap artists: raw cosine {n_neg_raw}/{len(raw)},"
          f" CSLS {n_neg_csls}/{len(csls)}")
    print("  '*' marks negative-gap (raw / CSLS columns)")
    print()
    print("Reading: a negative gap_raw means raw cosine misorders the artist "
          "against\nat least one other artist on this corpus. CSLS is expected "
          "to lift such\nartists toward zero or positive when the failure is "
          "readout-corrigible.")

    # Persist for inspection
    out = CACHE_DIR.parent / "demo_output.json"
    out.write_text(json.dumps({"raw": raw, "csls": csls}, indent=2, ensure_ascii=False))
    print(f"\nWrote per-artist results to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
