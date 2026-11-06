# Copyright 2026 Adobe
# All Rights Reserved.
#
# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

"""Export the Python TrustMark remover checkpoints for native Rust inference."""

import argparse
import hashlib
import inspect
import os
from pathlib import Path
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
from omegaconf import OmegaConf
import torch

from trustmark.trustmark import MODEL_CHECKSUMS, MODEL_REMOTE_HOST, instantiate_from_config


PYTHON_DIR = Path(__file__).resolve().parent


def download_model(filename, cache):
    """Reuse only checksum-verified model files; never replace an existing file."""
    path = cache / filename
    if not path.exists():
        with urllib.request.urlopen(MODEL_REMOTE_HOST + filename) as response:
            contents = response.read()
        if hashlib.md5(contents).hexdigest() != MODEL_CHECKSUMS[filename]:
            raise ValueError(f"Checksum mismatch downloading {filename}")
        with path.open("xb") as file:
            file.write(contents)
    if hashlib.md5(path.read_bytes()).hexdigest() != MODEL_CHECKSUMS[filename]:
        raise ValueError(f"Checksum mismatch for {path}; remove it before retrying")
    return path


def export_remover(variant, output, cache):
    destination = output / f"remover_{variant}.onnx"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    config = download_model(f"trustmark_rm_{variant}.yaml", cache)
    checkpoint = download_model(f"trustmark_rm_{variant}.ckpt", cache)
    # Load only the denoiser, not the watermark encoder/decoder or training modules.
    model = instantiate_from_config(OmegaConf.load(config).model.params.denoise_config)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = weights.get("state_dict", weights)
    weights = {key.removeprefix("denoise."): value for key, value in weights.items()
               if key.startswith("denoise.")}
    model.load_state_dict(weights, strict=True)
    model.eval()
    example = torch.linspace(-1, 1, 3 * 256 * 256).reshape(1, 3, 256, 256)
    # A temporary export is published only after ONNX/PyTorch parity succeeds.
    temporary = destination.with_suffix(".onnx.partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {temporary}")
    try:
        with torch.no_grad():
            # Keep TorchScript export across PyTorch versions, including versions
            # predating the dynamo option and versions where dynamo is the default.
            options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
            torch.onnx.export(model, example, temporary, input_names=["image"],
                              output_names=["output"], opset_version=17, **options)
            expected = model(example).numpy()
        onnx.checker.check_model(str(temporary))
        session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"image": example.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
        # Publish atomically without replacing an export made by another process.
        # Both names are in the same directory/filesystem.
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Exported and verified {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["B", "C", "P", "Q"], action="append")
    parser.add_argument("--output", type=Path, default=PYTHON_DIR.parent / "rust" / "models")
    parser.add_argument("--cache", type=Path, default=PYTHON_DIR / "trustmark" / "models")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    for variant in args.variant or ["B", "C", "P", "Q"]:
        export_remover(variant, args.output, args.cache)


if __name__ == "__main__":
    main()
