"""Unit tests for utils/hashing — content-addressed cache key correctness."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from adversarial_reasoning_training.utils.hashing import (
    gold_cache_key,
    sha256_bytes,
    sha256_file,
    sha256_image,
    sha256_json,
    sha256_text,
)


def test_sha256_bytes_known_vector() -> None:
    # Echo of the canonical SHA-256("") test vector — guards against
    # any accidental swap to a different hash function.
    assert sha256_bytes(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_text_utf8_roundtrip() -> None:
    # Non-ASCII must hash by UTF-8 bytes, not by Python str repr.
    text = "héllo"
    assert sha256_text(text) == sha256_bytes(text.encode("utf-8"))


def test_sha256_json_key_order_invariant() -> None:
    a = sha256_json({"a": 1, "b": [2, 3], "c": {"d": 4}})
    b = sha256_json({"c": {"d": 4}, "b": [2, 3], "a": 1})
    assert a == b


def test_sha256_json_value_change_invalidates() -> None:
    base = sha256_json({"k": 1})
    other = sha256_json({"k": 2})
    assert base != other


def test_sha256_json_unicode_preserved() -> None:
    # ensure_ascii=False — accented strings are hashed by their UTF-8
    # byte form, not their \uXXXX-escaped JSON form.
    h = sha256_json({"x": "café"})
    expected = sha256_text(json.dumps({"x": "café"}, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    assert h == expected


def _img(seed: int, size: int = 16) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8))


def test_sha256_image_deterministic() -> None:
    img = _img(0)
    assert sha256_image(img) == sha256_image(img.copy())


def test_sha256_image_distinct_images_distinct_hashes() -> None:
    assert sha256_image(_img(0)) != sha256_image(_img(1))


def test_sha256_image_mode_normalization() -> None:
    # RGBA and RGB of the same pixels must hash identically — the helper
    # converts to RGB before hashing to keep the cache key stable across
    # alpha-channel quirks in upstream loaders.
    rgb = _img(2)
    rgba = rgb.convert("RGBA")
    assert sha256_image(rgb) == sha256_image(rgba)


def test_sha256_file_matches_bytes(tmp_path) -> None:
    payload = b"file-content-blob" * 4096
    p = tmp_path / "blob.bin"
    p.write_bytes(payload)
    assert sha256_file(p) == sha256_bytes(payload)


def test_sha256_file_chunk_size_invariant(tmp_path) -> None:
    payload = bytes(range(256)) * 100
    p = tmp_path / "chunked.bin"
    p.write_bytes(payload)
    assert sha256_file(p, chunk_size=17) == sha256_file(p, chunk_size=1 << 20)


def test_gold_cache_key_changes_on_prompt() -> None:
    img = _img(3)
    a = gold_cache_key("task", "s1", "prompt-A", img, "v1")
    b = gold_cache_key("task", "s1", "prompt-B", img, "v1")
    assert a != b


def test_gold_cache_key_changes_on_oracle_version() -> None:
    img = _img(4)
    a = gold_cache_key("task", "s1", "p", img, "v1")
    b = gold_cache_key("task", "s1", "p", img, "v2")
    assert a != b


def test_gold_cache_key_changes_on_image() -> None:
    a = gold_cache_key("task", "s1", "p", _img(5), "v1")
    b = gold_cache_key("task", "s1", "p", _img(6), "v1")
    assert a != b


def test_gold_cache_key_stable_across_invocations() -> None:
    img = _img(7)
    a = gold_cache_key("task", "s1", "p", img, "v1")
    b = gold_cache_key("task", "s1", "p", img, "v1")
    assert a == b
