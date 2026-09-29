# Copyright 2026 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image

from .datalayer import DataLayer

CONCENTRATE_WM_REGION = 1.0
ASPECT_RATIO_LIM = 2.0
FEATHERING_RESIDUAL = 0.01
# Matches upstream trustmark.py's own model_resolution_max_detect - localize()
# first downsizes to this before running the detector, same as the real model.
MODEL_RESOLUTION_MAX_DETECT = 1600
# Unlike upstream's real Faster R-CNN (which runs at whatever size the image
# happens to be, up to max_detect), the NPU-compiled bbox trunk needs a fixed
# square canvas (Hailo compiles static shapes only) - see
# compile/export_bbox_trunk.py / compile_bbox.py. localize() below resizes
# (non-aspect-preserving) to this square size for NPU inference, then rescales
# detected boxes back into the max_detect-scaled image's own coordinate space
# before normalizing - a documented simplification versus upstream's
# arbitrary-aspect-ratio handling.
BBOX_CANVAS_SIZE = 640

# Both ONNX graphs (encoder_C.onnx / decoder_C.onnx) were empirically verified
# to take/return NCHW float32 in [-1, 1] at 256x256 - see
# scripts/validate_onnx_parity.py. Pixel-format helpers used by both the
# onnxruntime (dev machine) and HailoRT (Pi) backends, via encode()/decode()
# below - upstream does the equivalent conversion inline with
# `transforms.ToTensor()`, not needed here since there's no torch dependency.
MODEL_RESOLUTION = 256


def pil_to_model_input(img: Image.Image, resolution: int = MODEL_RESOLUTION) -> np.ndarray:
    """PIL RGB image -> (1,3,res,res) float32 NCHW in [-1, 1]."""
    resized = img.resize((resolution, resolution), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0 * 2.0 - 1.0
    return arr.transpose(2, 0, 1)[None].copy()


def model_output_to_stego_arr(stego_nchw: np.ndarray) -> np.ndarray:
    """(1,3,H,W) float32 in [-1,1] -> (H,W,3) float32 in [-1,1]."""
    return stego_nchw[0].transpose(1, 2, 0)

# EncoderInfer: (cover_nchw[-1,1] float32 (1,3,256,256), secret (1,secret_len) float32 0/1) -> stego_nchw (1,3,256,256) float32
EncoderInfer = Callable[[np.ndarray, np.ndarray], np.ndarray]
# DecoderInfer: (image_nchw[-1,1] float32 (1,3,H,W)) -> logits (1,secret_len) float32.
# H,W is 256 for variants C/Q/B (decoder_{C,Q,B}.onnx bake an internal resize to
# their true resolution, unlike upstream's own 245 - see docs/ARCHITECTURE.md's
# "Model I/O"), but 224 for variant P (decoder_P.onnx takes 224 directly) - see
# `__init__`'s model_type dispatch below. Verified via
# scripts/validate_onnx_parity.py / scripts/decode_existing.py.
DecoderInfer = Callable[[np.ndarray], np.ndarray]
# BboxInfer: (image_hwc float32 (BBOX_CANVAS_SIZE,BBOX_CANVAS_SIZE,3), values in [0,255]) ->
# (boxes_xyxy (N,4) float32 pixel coords in that same canvas, scores (N,) float32).
BboxInfer = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


def _resize_float_hwc(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an (H,W,3) float array (arbitrary range) with bilinear
    interpolation, channel-by-channel via PIL's 'F' mode (avoids requiring
    torch or opencv just for this one resize)."""
    w, h = size
    out = np.empty((h, w, arr.shape[2]), dtype=np.float32)
    for c in range(arr.shape[2]):
        band = Image.fromarray(arr[:, :, c].astype(np.float32), mode="F")
        out[:, :, c] = np.asarray(band.resize((w, h), Image.BILINEAR))
    return out


class TrustMarkHailo:
    """Same encode()/decode() contract as trustmark.TrustMark, but the two
    neural net forward passes are delegated to injected callables so this
    class has no PyTorch/Hailo dependency of its own."""

    def __init__(
        self,
        encoder_infer: Optional[EncoderInfer],
        decoder_infer: Optional[DecoderInfer],
        secret_len: int = 100,
        encoding_type: int = 1,  # DataLayer/BCH schema id; 1 == BCH_5 (61-bit payload), matches TrustMark's default
        model_type: str = "Q",
        concentrate_wm_region: float = CONCENTRATE_WM_REGION,
        bbox_infer: Optional[BboxInfer] = None,
    ):
        self.encoder_infer = encoder_infer
        self.decoder_infer = decoder_infer
        self.bbox_infer = bbox_infer
        self.secret_len = secret_len
        self.ecc = DataLayer(secret_len, encoding_mode=encoding_type)
        self.model_type = model_type
        self.concentrate_wm_region = concentrate_wm_region
        self.encoder_resolution = MODEL_RESOLUTION  # 256 for every variant (C/Q/B/P)
        self.aspect_ratio_lim = ASPECT_RATIO_LIM

        # Mirrors ../python/trustmark/trustmark.py's own model_type dispatch in
        # __init__ - same shape, one real difference: decoder_resolution is 256
        # here for C/Q/B, not upstream's 245, because decoder_{C,Q,B}.onnx bakes
        # in an internal Resize to 256 (see docs/ARCHITECTURE.md's "Model I/O") -
        # not a departure to fix, a genuine difference in what these exported
        # graphs actually expect.
        if model_type == "P":
            self.decoder_resolution = 224
            self.aspect_ratio_lim = 0  # always force to centre square crop
        else:
            self.decoder_resolution = MODEL_RESOLUTION

    def schemaCapacity(self) -> int:
        return self.ecc.schemaCapacity(self.ecc.encoding_mode)

    def get_the_image_for_processing(self, in_image: Image.Image) -> Image.Image:
        scale = self.concentrate_wm_region
        width, height = in_image.size

        # Compute aspect ratio (>= 1.0)
        if width > height:
            aspect_ratio = width / height
        else:
            aspect_ratio = height / width

        # Make a copy of the image (PIL)
        out_im = in_image.copy()

        if aspect_ratio > self.aspect_ratio_lim:
            # We do a center-square approach, but scaled
            square_size = min(width, height)  # largest possible square dimension
            scaled_size = int(square_size * scale)  # scale that dimension

            # Compute bounding box
            left = (width - scaled_size) // 2
            top = (height - scaled_size) // 2
            right = left + scaled_size
            bottom = top + scaled_size

            out_im = out_im.crop((left, top, right, bottom))

        else:
            # The aspect ratio is normal, so we consider
            # the *entire* image dimension. Then scale that region
            scaled_w = int(width * scale)
            scaled_h = int(height * scale)

            # Center the smaller (or bigger) rectangle
            left = (width - scaled_w) // 2
            top = (height - scaled_h) // 2
            right = left + scaled_w
            bottom = top + scaled_h

            out_im = out_im.crop((left, top, right, bottom))

        return out_im

    def put_the_image_after_processing(self, wm_image: np.ndarray, cover_im: np.ndarray, feather: bool = True) -> np.ndarray:
        scale = self.concentrate_wm_region
        cover_h, cover_w, _ = cover_im.shape

        if cover_w > cover_h:
            aspect_ratio = cover_w / cover_h
        else:
            aspect_ratio = cover_h / cover_w

        out_im = cover_im.copy()

        if aspect_ratio > self.aspect_ratio_lim:
            # Square region, scaled
            square_size = min(cover_w, cover_h)
            scaled_size = int(square_size * scale)

            left = (cover_w - scaled_size) // 2
            top = (cover_h - scaled_size) // 2
            right = left + scaled_size
            bottom = top + scaled_size

            region_w = scaled_size
            region_h = scaled_size

        else:
            # Normal ratio, scaled
            scaled_w = int(cover_w * scale)
            scaled_h = int(cover_h * scale)
            left = (cover_w - scaled_w) // 2
            top = (cover_h - scaled_h) // 2
            right = left + scaled_w
            bottom = top + scaled_h

            region_w = scaled_w
            region_h = scaled_h

        if feather:
            feather_size = int(min(region_w, region_h) * FEATHERING_RESIDUAL)

            feather_size = max(1, feather_size)
            feather_size = min(feather_size, 50)

            self.feather_paste(out_im, cover_im, wm_image, top, bottom, left, right, feather_size=feather_size)
        else:
            out_im[top:bottom, left:right, :] = wm_image

        return out_im

    def feather_paste(
        self,
        out_im: np.ndarray,  # Output image (modified in-place)
        cover_im: np.ndarray,  # Original cover image (same shape)
        wm_image: np.ndarray,  # Watermarked patch to paste
        top: int,
        bottom: int,
        left: int,
        right: int,
        feather_size: int = 9,
    ) -> None:
        out_im[top:bottom, left:right, :] = wm_image
        alpha_vals = [(i + 1) / feather_size for i in range(feather_size)]

        feather_size = min(feather_size, (bottom - top), (right - left))

        for i in range(feather_size):
            alpha = alpha_vals[i]
            row = top + i
            # Blend that entire row from left..right
            out_im[row, left:right, :] = alpha * wm_image[i, :, :] + (1.0 - alpha) * cover_im[row, left:right, :]

        for i in range(feather_size):
            alpha = alpha_vals[i]
            row = bottom - 1 - i
            wm_row = (bottom - top - 1) - i
            out_im[row, left:right, :] = alpha * wm_image[wm_row, :, :] + (1.0 - alpha) * cover_im[row, left:right, :]

        for i in range(feather_size):
            alpha = alpha_vals[i]
            col = left + i
            out_im[top:bottom, col, :] = alpha * wm_image[:, i, :] + (1.0 - alpha) * cover_im[top:bottom, col, :]

        for i in range(feather_size):
            alpha = alpha_vals[i]
            col = right - 1 - i
            wm_col = (right - left - 1) - i
            out_im[top:bottom, col, :] = alpha * wm_image[:, wm_col, :] + (1.0 - alpha) * cover_im[top:bottom, col, :]

    def encode(self, in_cover_image: Image.Image, string_secret: str, MODE: str = "binary", WM_STRENGTH: float = 1.0) -> Image.Image:
        if self.encoder_infer is None:
            raise RuntimeError("No encoder backend configured on this TrustMarkHailo instance")

        if self.model_type == "P":
            WM_STRENGTH = WM_STRENGTH * 1.25

        secret = self.ecc.encode_binary([string_secret]) if MODE == "binary" else self.ecc.encode_text([string_secret])
        secret = secret.astype(np.float32)  # (1, secret_len), values in {0.0, 1.0}

        cover_image = self.get_the_image_for_processing(in_cover_image)
        w, h = cover_image.size
        cover_nchw = pil_to_model_input(cover_image, self.encoder_resolution)

        stego_nchw = self.encoder_infer(cover_nchw, secret)

        stego_hwc = model_output_to_stego_arr(np.clip(stego_nchw, -1, 1))
        cover_hwc = model_output_to_stego_arr(cover_nchw)
        residual = stego_hwc - cover_hwc
        residual = residual - residual.mean(axis=(0, 1), keepdims=True)  # remove per-channel color shift

        residual_full = _resize_float_hwc(residual, (w, h))
        cover_arr = np.asarray(cover_image).astype(np.float32)
        stego_full = np.clip(residual_full * WM_STRENGTH + cover_arr / 127.5 - 1.0, -1, 1) * 127.5 + 127.5

        stego = self.put_the_image_after_processing(stego_full, np.asarray(in_cover_image).astype(np.uint8))
        return Image.fromarray(stego.astype(np.uint8))

    def subimage_decode(self, stego_image: Image.Image, MODE: str = "binary"):
        if self.decoder_infer is None:
            raise RuntimeError("No decoder backend configured on this TrustMarkHailo instance")

        stego_nchw = pil_to_model_input(stego_image, self.decoder_resolution)
        logits = self.decoder_infer(stego_nchw)
        secret_binaryarray = (logits > 0).astype(np.uint8)
        secret_pred, detected, version = self.ecc.decode_bitstream(secret_binaryarray, MODE)[0]
        return secret_pred, detected, version

    def localize(self, in_stego_image: Image.Image, return_all: bool = False):
        """Same contract as upstream trustmark.TrustMark.localize(): returns
        None if nothing is detected, otherwise a single normalized [x1,y1,x2,y2]
        box (top score) or a list of them if return_all=True. No confidence
        threshold - matches upstream, which returns top-1 unconditionally
        whenever the detector produces any box at all."""
        if self.bbox_infer is None:
            raise RuntimeError("BBox detector model is not loaded. Initialize TrustMark passing loadBBoxDetector=True.")

        w, h = in_stego_image.size
        long_side = max(w, h)
        if long_side > MODEL_RESOLUTION_MAX_DETECT:
            scale = MODEL_RESOLUTION_MAX_DETECT / long_side
            sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
            scaled = in_stego_image.resize((sw, sh), Image.BILINEAR)
        else:
            scaled = in_stego_image

        canvas = scaled.convert("RGB").resize((BBOX_CANVAS_SIZE, BBOX_CANVAS_SIZE), Image.BILINEAR)
        canvas_arr = np.asarray(canvas).astype(np.float32)  # (640,640,3) in [0,255]
        boxes, scores = self.bbox_infer(canvas_arr)
        if boxes.shape[0] == 0:
            return None

        # Boxes are in the square NPU canvas's own pixel space; since the
        # canvas is a uniform (non-aspect-preserving) stretch of the whole
        # `scaled` image, a box's fractional position within the canvas is
        # identical to its fractional position within `scaled` - no need to
        # go back through (sw,sh) explicitly.
        def normalize_box(b):
            x1, y1, x2, y2 = (float(v) / BBOX_CANVAS_SIZE for v in b)
            return [max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)), max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))]

        if return_all:
            return [normalize_box(b) for b in boxes]
        top_idx = int(np.argmax(scores))
        return normalize_box(boxes[top_idx])

    def decode(self, in_stego_image: Image.Image, MODE: str = "binary", DETECTFIRST: bool = False, ROTATION: bool = False):
        angleset = [0, 90, 180, 270] if ROTATION else [0]

        for angle in angleset:
            if angle == 0:
                rotated = in_stego_image
            elif angle == 90:
                rotated = in_stego_image.transpose(Image.ROTATE_90)
            elif angle == 180:
                rotated = in_stego_image.transpose(Image.ROTATE_180)
            else:
                rotated = in_stego_image.transpose(Image.ROTATE_270)

            if DETECTFIRST and self.bbox_infer is not None:
                boxes_pred = self.localize(rotated, return_all=True)
                if boxes_pred is None:
                    return "", False, -1
                w, h = rotated.size
                for box in boxes_pred:
                    x1, y1, x2, y2 = box
                    px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
                    px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))
                    if px2 <= px1 or py2 <= py1:
                        continue
                    cropped = rotated.crop((px1, py1, px2, py2))
                    secret_pred, detected, version = self.subimage_decode(cropped, MODE)
                    if detected:
                        return secret_pred, detected, version
                # no box on this rotation decoded successfully - fall through to next angle
            else:
                stego_image = self.get_the_image_for_processing(rotated)
                secret_pred, detected, version = self.subimage_decode(stego_image, MODE)
                if detected:
                    return secret_pred, detected, version

        return "", False, -1
