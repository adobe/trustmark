# Copyright 2026 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


# bbox-trunk NPU inference + CPU postprocess, mirroring hailo_runtime.py's
# make_encoder_infer/make_decoder_infer pattern.
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .bbox_postprocess import postprocess
from .hailo_runtime import HailoModel

CANVAS_SIZE = 640

# The trunk HEF has 15 outputs (feat0-4, cls0-4, bbox0-4 - see
# rpi-trustmark/compile/export_bbox_trunk.py). Hailo's compiler renames
# output vstreams (e.g. to "output_layer13"), so outputs are matched by their
# unique native shape rather than by name - every one of these 15 shapes is
# distinct for a 640x640 input.
_OUTPUT_SHAPES = {
    "feat0": (160, 160, 256), "feat1": (80, 80, 256), "feat2": (40, 40, 256), "feat3": (20, 20, 256), "feat4": (10, 10, 256),
    "cls0": (160, 160, 3), "cls1": (80, 80, 3), "cls2": (40, 40, 3), "cls3": (20, 20, 3), "cls4": (10, 10, 3),
    "bbox0": (160, 160, 12), "bbox1": (80, 80, 12), "bbox2": (40, 40, 12), "bbox3": (20, 20, 12), "bbox4": (10, 10, 12),
}


def make_bbox_infer(hef_path: str | Path, box_head_npz: str | Path):
    """Returns a `bbox_infer(image_hwc_0_255) -> (boxes_xyxy, scores)`
    callable. `image_hwc_0_255` is (H,W,3) float32/uint8 raw pixel values
    (the trunk HEF has in-graph ImageNet normalization baked in, matching
    compile/compile_bbox.py's NORMALIZATION_MODEL_SCRIPT - same convention as
    the decoder's `_nchw_float_neg1_1_to_nhwc_float_0_255` path).

    Everything downstream of the NPU trunk (anchor decode, NMS, ROIAlign, the
    tiny box head) runs on the CPU via bbox_postprocess.postprocess - see that
    module's docstring for why (proposal counts are data-dependent, so this
    stage can't be a fixed-shape NPU graph)."""
    model = HailoModel(hef_path)
    box_head_weights = dict(np.load(box_head_npz))

    name_by_role = {}
    for role, shape in _OUTPUT_SHAPES.items():
        matches = [n for n in model.output_names if model._native_shapes[n] == shape]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one bbox-trunk output with shape {shape} (role {role}), found {len(matches)}")
        name_by_role[role] = matches[0]
    image_name = model.input_names[0]

    def bbox_infer(image_hwc_0_255: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image_in = np.ascontiguousarray(image_hwc_0_255.astype(np.float32)[None])  # (1,H,W,3)
        result = model.infer_multi({image_name: image_in})
        trunk_outputs = {role: result[name][0] for role, name in name_by_role.items()}  # drop batch dim
        return postprocess(trunk_outputs, (CANVAS_SIZE, CANVAS_SIZE), box_head_weights)

    return bbox_infer, model
