# Copyright 2026 Adobe
# All Rights Reserved.
#
# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

"""Exercise native/Python removal parity across all variants and crop/feather edges."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from trustmark import TrustMark


PYTHON_DIR = Path(__file__).resolve().parent
RUST_DIR = PYTHON_DIR.parent / "rust"


class ReferenceRemover(TrustMark):
    """Use the Python removal pipeline without loading encoder/decoder models."""

    def __init__(self, variant, cache):
        self.model_type = variant
        self.model_resolution_remove = 256
        self.concentrate_wm_region = 1.0
        self.aspect_ratio_lim = 0 if variant == "P" else 2.0
        self.removal = self.load_model(
            str(cache / f"trustmark_rm_{variant}.yaml"),
            str(cache / f"trustmark_rm_{variant}.ckpt"),
            "cpu", 100, part="remover",
        )
        if self.removal is None:
            raise RuntimeError("Unable to load Python reference remover")


def compare_removal(reference, input_path, actual_path):
    """Check dimensions, alpha, and RGB parity using the shared tolerances."""
    with Image.open(input_path) as image:
        expected = reference.remove_watermark(image.convert("RGB"))
        original_alpha = np.asarray(image.getchannel("A")) if image.mode == "RGBA" else None
    with Image.open(actual_path) as image:
        if image.size != expected.size:
            raise AssertionError(f"Dimensions changed: {expected.size} -> {image.size}")
        if original_alpha is not None:
            np.testing.assert_array_equal(np.asarray(image.getchannel("A")), original_alpha)
        actual = image.convert("RGB")
    difference = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    print(f"{reference.model_type} / {expected.size}: max pixel difference={difference.max()}, mean={difference.mean():.4f}")
    # Native raster decoding/resizing and Pillow differ slightly. Check both the
    # worst channel deviation and the average, rather than requiring bit equality.
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=4, rtol=0)
    if difference.mean() > 0.5:
        raise AssertionError("Mean RGB difference exceeds half an 8-bit level")


SIZES = [
    (1, 1),           # Smallest possible region; overlapping feather strips.
    (17, 19),         # Odd, sub-model-resolution dimensions.
    (101, 10),        # Wide, with a very small crop.
    (10, 101),        # Tall, with a very small crop.
    (257, 263),       # Odd resize and feather dimensions.
    (513, 257),       # Just below the Q/B/C square-crop threshold.
    (515, 257),       # Just above it, with an odd crop offset.
    (257, 515),       # The corresponding tall case.
    (1017, 257),      # A wide centred crop and untouched outer pixels.
    (257, 1017),      # A tall centred crop and untouched outer pixels.
]


def patterned_image(width, height, with_alpha):
    """Make repeatable spatial gradients without requiring image fixtures."""
    x = np.broadcast_to(np.linspace(0, 1, width)[None, :], (height, width))
    y = np.broadcast_to(np.linspace(0, 1, height)[:, None], (height, width))
    rgb = np.rint(np.stack([40 + 180 * x, 40 + 180 * y, 40 + 90 * (x + y)], axis=2)).astype(np.uint8)
    if with_alpha:
        alpha = np.rint(255 * x).astype(np.uint8)
        return Image.fromarray(np.dstack([rgb, alpha]))
    return Image.fromarray(rgb)


class RemovalParityTests(unittest.TestCase):
    def test_native_matches_reference(self):
        for variant in OPTIONS.variant or ["B", "C", "P", "Q"]:
            reference = ReferenceRemover(variant, OPTIONS.cache)
            for width, height in SIZES:
                for with_alpha in [False, True]:
                    with self.subTest(variant=variant, size=(width, height), alpha=with_alpha):
                        with tempfile.TemporaryDirectory(prefix="trustmark-parity-", dir=OPTIONS.temp_dir) as directory:
                            input_path = Path(directory) / "input.png"
                            actual_path = Path(directory) / "actual.png"
                            patterned_image(width, height, with_alpha).save(input_path)
                            result = subprocess.run(
                                [str(OPTIONS.cli), "-m", str(OPTIONS.models), "remove",
                                 "-i", str(input_path), "-o", str(actual_path), "--variant", variant],
                                capture_output=True, text=True, timeout=60,
                            )
                            self.assertEqual(result.returncode, 0, result.stderr)
                            compare_removal(reference, input_path, actual_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=RUST_DIR / "target" / "debug" / "trustmark")
    parser.add_argument("--models", type=Path, default=RUST_DIR / "models")
    parser.add_argument("--cache", type=Path, default=PYTHON_DIR / "trustmark" / "models")
    parser.add_argument("--variant", choices=["B", "C", "P", "Q"], action="append")
    parser.add_argument("--temp-dir", type=Path)
    OPTIONS = parser.parse_args()
    OPTIONS.cli = OPTIONS.cli.resolve()
    OPTIONS.models = OPTIONS.models.resolve()
    if not OPTIONS.cli.is_file():
        parser.error(f"Rust CLI not found at {OPTIONS.cli}; run cargo build -p trustmark-cli from rust/")
    torch.set_num_threads(8)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
