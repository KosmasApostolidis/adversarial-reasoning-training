"""Content hashing helpers for caching gold trajectories + config fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    """Hash a JSON-serializable object with sorted keys for stable output."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def sha256_image(image: Image.Image) -> str:
    """Hash a PIL image deterministically across PIL/libpng versions.

    The previous implementation rounded the image through PIL's PNG encoder
    and hashed the resulting bytes — but PNG output drifts with libpng and
    PIL versions (chunk ordering, default text metadata), which silently
    invalidates the cache for collaborators on different toolchains. We
    now hash the raw RGB pixel buffer plus the (width, height) so two
    visually-identical images always produce the same key regardless of
    encoder version. NB: this is a different hash than the previous
    implementation; existing gold caches will rebuild on first miss.
    """
    rgb = image.convert("RGB")
    h = hashlib.sha256()
    h.update(f"{rgb.size[0]}x{rgb.size[1]}|".encode("ascii"))
    h.update(rgb.tobytes())
    return h.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def gold_cache_key(
    task_id: str,
    sample_id: str,
    prompt: str,
    image: Image.Image,
    oracle_version: str,
) -> str:
    """Canonical cache key for rule-based gold trajectories.

    Any change in the prompt template, image content, or oracle version
    produces a different key — preventing silent staleness.
    """
    parts = {
        "task_id": task_id,
        "sample_id": sample_id,
        "prompt": sha256_text(prompt),
        "image": sha256_image(image),
        "oracle_version": oracle_version,
    }
    return sha256_json(parts)
