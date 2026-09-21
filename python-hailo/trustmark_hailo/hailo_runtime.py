# Copyright 2026 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


# HailoRT wrapper - only importable on the Pi (aarch64 Linux with hailort's
# Python bindings, `hailo_platform`, installed via `sudo apt install hailo-all`).
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

# Hailo's native vstream layout for a spatial input is NHWC (e.g. shape (256,256,3)), NOT NCHW like the
# source ONNX graph - the DFC parser transposes during compilation. 


_shared_vdevice = None


def get_shared_vdevice():
    global _shared_vdevice
    if _shared_vdevice is None:
        from hailo_platform import VDevice

        _shared_vdevice = VDevice()
    return _shared_vdevice


class HailoModel:

    def __init__(self, hef_path: str | Path):
        from hailo_platform import (
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InputVStreamParams,
            OutputVStreamParams,
        )

        self._hef = HEF(str(hef_path))
        self._device = get_shared_vdevice()
        configure_params = ConfigureParams.create_from_hef(self._hef, interface=HailoStreamInterface.PCIe)
        self._network_group = self._device.configure(self._hef, configure_params)[0]
        self._network_group_params = self._network_group.create_params()

        self._input_infos = self._hef.get_input_vstream_infos()
        self._output_infos = self._hef.get_output_vstream_infos()
        self._input_params = InputVStreamParams.make(self._network_group, quantized=False, format_type=FormatType.FLOAT32)
        self._output_params = OutputVStreamParams.make(self._network_group, quantized=False, format_type=FormatType.FLOAT32)
        # name -> native vstream shape, e.g. (256,256,3) for a spatial input/output,
        # (100,) for a flat vector - used to auto-detect NCHW<->NHWC conversion needs.
        self._native_shapes = {i.name: tuple(i.shape) for i in list(self._input_infos) + list(self._output_infos)}

    @property
    def input_names(self) -> list[str]:
        return [i.name for i in self._input_infos]

    @property
    def output_names(self) -> list[str]:
        return [o.name for o in self._output_infos]

    def infer_multi(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        from hailo_platform import InferVStreams

        # Order matters: open InferVStreams *then* activate inside it, matching
        # Hailo's own HRT_2_Infer_Pipeline_Inference_Tutorial.ipynb
        with InferVStreams(self._network_group, self._input_params, self._output_params) as pipeline:
            with self._network_group.activate(self._network_group_params):
                return pipeline.infer(inputs)

    def infer(self, input_arr: np.ndarray) -> np.ndarray:
        name = self.input_names[0]
        out_name = self.output_names[0]
        result = self.infer_multi({name: input_arr})
        out = result[out_name]
        if out.ndim == len(self._native_shapes[out_name]):
            out = out[None]  # add back the batch dim HailoRT drops for flat (N,) vstreams
        return out

    def close(self) -> None:
        # Don't release self._device here - it's the process-wide shared
        # VDevice (get_shared_vdevice), other HailoModels may still be using
        pass


def _nchw_float_neg1_1_to_nhwc_float_0_255(x: np.ndarray) -> np.ndarray:
    """(1,3,H,W) float32 in [-1,1] -> (1,H,W,3) float32 in [0,255], matching
    both the NHWC-native vstream layout and the in-graph normalization baked
    in by compile/compile_common.py's NORMALIZATION_MODEL_SCRIPT. Used for the
    decoder only - the compiled encoder has no in-graph normalization (see
    compile/compile_encoder.py), so it uses _nchw_to_nhwc (layout only) below."""
    hwc = x[0].transpose(1, 2, 0)
    u8_range = np.clip((hwc + 1.0) * 127.5, 0, 255)
    return np.ascontiguousarray(u8_range.astype(np.float32)[None])


def _nchw_to_nhwc(x: np.ndarray) -> np.ndarray:
    """(1,3,H,W) -> (1,H,W,3), layout only, no value rescale."""
    return np.ascontiguousarray(x[0].transpose(1, 2, 0)[None])


def _nhwc_to_nchw(x: np.ndarray) -> np.ndarray:
    """(1,H,W,3) -> (1,3,H,W), layout only, no value rescale."""
    return np.ascontiguousarray(x[0].transpose(2, 0, 1)[None])


class Secret2ImageCPU:
    """CPU-side replacement for the first two ops of the encoder's
    Secret2Image submodule (Gemm/Linear + Reshape) - the Hailo parser can't
    translate the Reshape (`UnsupportedShuffleLayerError`), so
    compile/compile_encoder.py starts the Hailo graph one step later, at this
    op's output, and does the upsample-to-256x256 + ReLU on the NPU instead."""

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.weight = data["weight"]  # (768, 100)
        self.bias = data["bias"]  # (768,)
        # reshape_shape is [-1, 3, 16, 16] (NCHW, batch dim -1) as exported from PyTorch
        self.reshape_shape = tuple(int(d) for d in data["reshape_shape"])

    def __call__(self, secret: np.ndarray) -> np.ndarray:
        """secret: (1,100) float32 in {0.,1.} -> (1,16,16,3) float32 NHWC,
        matching the Hailo-compiled encoder's second input layout."""
        y = secret.astype(np.float32) @ self.weight.T + self.bias  # (1,768)
        batch = secret.shape[0]
        nchw = y.reshape((batch,) + self.reshape_shape[1:])  # (1,3,16,16)
        return np.ascontiguousarray(nchw.transpose(0, 2, 3, 1))  # (1,16,16,3) NHWC


def make_encoder_infer(hef_path: str | Path, secret2image_npz: str | Path):
    """Returns an `encoder_infer(cover_nchw, secret) -> stego_nchw` callable
    (see trustmark_hailo.pipeline.EncoderInfer) backed by a HailoModel.

    The compiled encoder's second input is NOT the raw secret bits - the
    Hailo parser can't translate the Secret2Image submodule's Reshape op, so
    compile/compile_encoder.py splits the graph after it. The raw (1,100)
    secret this callable receives (matching trustmark_hailo.pipeline's
    contract) is projected to the (1,16,16,3) map the HEF actually expects via
    Secret2ImageCPU (CPU-side, using weights extracted from the same ONNX by
    scripts/extract_secret2image.py) before being sent to the NPU. See
    docs/ARCHITECTURE.md's "Encoder: splitting Secret2Image off the NPU graph".

    Also unlike the decoder, the compiled encoder has no in-graph
    normalization (see compile_encoder.py for why), so image data is sent
    already-normalized in [-1,1] (layout-only NCHW<->NHWC conversion), not
    rescaled to [0,255]."""
    model = HailoModel(hef_path)
    secret2image = Secret2ImageCPU(secret2image_npz)

    image_name = next(n for n in model.input_names if model._native_shapes[n] == (256, 256, 3))
    secretmap_name = next(n for n in model.input_names if model._native_shapes[n] == (16, 16, 3))
    out_name = model.output_names[0]

    def encoder_infer(cover_nchw: np.ndarray, secret: np.ndarray) -> np.ndarray:
        image_in = _nchw_to_nhwc(cover_nchw)  # already [-1,1], layout only
        secretmap_in = secret2image(secret)  # (1,16,16,3) float32
        result = model.infer_multi({image_name: image_in, secretmap_name: secretmap_in})
        return _nhwc_to_nchw(result[out_name])

    return encoder_infer, model


def make_decoder_infer(hef_path: str | Path):
    """Returns a `decoder_infer(image_nchw) -> logits` callable
    (see trustmark_hailo.pipeline.DecoderInfer) backed by a HailoModel.
    Verified against real hardware - see docs/ARCHITECTURE.md."""
    model = HailoModel(hef_path)

    def decoder_infer(image_nchw: np.ndarray) -> np.ndarray:
        image_in = _nchw_float_neg1_1_to_nhwc_float_0_255(image_nchw)
        return model.infer(image_in)

    return decoder_infer, model
