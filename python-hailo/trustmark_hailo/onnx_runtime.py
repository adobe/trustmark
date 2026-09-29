# Copyright 2026 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


# onnxruntime backend - the CPU/float counterpart to hailo_runtime.py's
# HailoModel
from __future__ import annotations

from pathlib import Path


class OnnxModel:
    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def close(self) -> None:
        pass  # onnxruntime InferenceSession has no explicit resource to release


def make_encoder_infer(onnx_path: str | Path):
    """Returns an `encoder_infer(cover_nchw, secret) -> stego_nchw` callable
    (see trustmark_hailo.pipeline.EncoderInfer), running the float ONNX
    encoder graph directly - no Secret2Image split, no quantization, matches
    the real PyTorch encoder's output up to onnxruntime's own numerics."""
    model = OnnxModel(onnx_path)
    in1, in2 = [i.name for i in model.session.get_inputs()]
    out = model.session.get_outputs()[0].name

    def encoder_infer(cover_nchw, secret):
        return model.session.run([out], {in1: cover_nchw, in2: secret})[0]

    return encoder_infer, model


def make_decoder_infer(onnx_path: str | Path):
    """Returns a `decoder_infer(image_nchw) -> logits` callable (see
    trustmark_hailo.pipeline.DecoderInfer), running the float ONNX decoder
    graph directly."""
    model = OnnxModel(onnx_path)
    in1 = model.session.get_inputs()[0].name
    out = model.session.get_outputs()[0].name

    def decoder_infer(image_nchw):
        return model.session.run([out], {in1: image_nchw})[0]

    return decoder_infer, model
