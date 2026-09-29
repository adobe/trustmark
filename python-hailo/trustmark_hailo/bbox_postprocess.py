from __future__ import annotations

import numpy as np

# Anchor generator config (RegionProposalNetwork.anchor_generator, confirmed
# against the real loaded model): one size per FPN level, same 3 aspect
# ratios at every level.
ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))
ANCHOR_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * 5

# RPN (RegionProposalNetwork, confirmed against the real loaded model).
RPN_BOX_CODER_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
RPN_NMS_THRESH = 0.7
RPN_SCORE_THRESH = 0.0  # real model's default - effectively a no-op after sigmoid
RPN_MIN_SIZE = 0.001
RPN_PRE_NMS_TOP_N = 1000  # eval/"testing" mode value
RPN_POST_NMS_TOP_N = 1000

# RoIHeads (confirmed against the real loaded model).
ROI_BOX_CODER_WEIGHTS = (10.0, 10.0, 5.0, 5.0)  # different from RPN's - torchvision's RoIHeads default
ROI_SCORE_THRESH = 0.05
ROI_NMS_THRESH = 0.5
ROI_DETECTIONS_PER_IMG = 4
ROI_MIN_BOX_SIZE = 1e-2
ROI_OUTPUT_SIZE = 7
ROI_SAMPLING_RATIO = 2
ROI_CANONICAL_SCALE = 224
ROI_CANONICAL_LEVEL = 4

BBOX_XFORM_CLIP = float(np.log(1000.0 / 16))

# Order the 5-level NPU trunk outputs correspond to (feat0..4/cls0..4/bbox0..4
# = P2,P3,P4,P5,"pool"/P6 - see rpi-trustmark/compile/export_bbox_trunk.py).
# Only the first 4 (P2-P5) feed ROIAlign; all 5 feed anchor generation/RPN.
NUM_LEVELS = 5
ROI_LEVELS = 4


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def generate_base_anchors(sizes=ANCHOR_SIZES, aspect_ratios=ANCHOR_ASPECT_RATIOS):
    """Per-level zero-centered anchor templates, shape (num_anchors_per_loc, 4)
    each. Direct port of AnchorGenerator.generate_anchors."""
    base_anchors = []
    for scales, ars in zip(sizes, aspect_ratios):
        scales = np.asarray(scales, dtype=np.float64)
        ars = np.asarray(ars, dtype=np.float64)
        h_ratios = np.sqrt(ars)
        w_ratios = 1.0 / h_ratios
        ws = (w_ratios[:, None] * scales[None, :]).reshape(-1)
        hs = (h_ratios[:, None] * scales[None, :]).reshape(-1)
        anchors = np.stack([-ws, -hs, ws, hs], axis=1) / 2.0
        base_anchors.append(np.round(anchors).astype(np.float32))
    return base_anchors


def grid_anchors(feature_shapes, image_size, base_anchors):
    """feature_shapes: list of (H,W) per level. image_size: (H,W) of the
    (unpadded) input. Returns a single (total_anchors, 4) array, ordered
    level-by-level then row-major within each level - matching the trunk's
    own per-level output ordering. Direct port of AnchorGenerator.grid_anchors."""
    all_anchors = []
    for (gh, gw), anchors in zip(feature_shapes, base_anchors):
        stride_h = image_size[0] // gh
        stride_w = image_size[1] // gw
        shifts_x = np.arange(0, gw, dtype=np.int32) * stride_w
        shifts_y = np.arange(0, gh, dtype=np.int32) * stride_h
        shift_y, shift_x = np.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts = np.stack([shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)], axis=1)
        # (grid, 1, 4) + (1, num_anchors, 4) -> (grid, num_anchors, 4) -> (grid*num_anchors, 4)
        level_anchors = (shifts[:, None, :] + anchors[None, :, :]).reshape(-1, 4)
        all_anchors.append(level_anchors.astype(np.float32))
    return all_anchors  # kept per-level; caller concatenates as needed


def decode_boxes(rel_codes, boxes, weights, bbox_xform_clip=BBOX_XFORM_CLIP):
    """boxes: (N,4) reference boxes. rel_codes: (N, 4*num_classes) - for RPN
    num_classes=1 (plain (N,4)); for RoIHeads num_classes=3, since
    box_regression predicts a separate box per class. weights: (wx,wy,ww,wh).
    Returns (N, 4*num_classes), matching det_utils.BoxCoder.decode_single's
    strided multi-class handling exactly (rel_codes[:, 0::4] etc)."""
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = rel_codes[:, 0::4] / wx  # (N, num_classes)
    dy = rel_codes[:, 1::4] / wy
    dw = rel_codes[:, 2::4] / ww
    dh = rel_codes[:, 3::4] / wh

    dw = np.clip(dw, a_min=None, a_max=bbox_xform_clip)
    dh = np.clip(dh, a_min=None, a_max=bbox_xform_clip)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = np.exp(dw) * widths[:, None]
    pred_h = np.exp(dh) * heights[:, None]

    pred_boxes = np.stack(
        [
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ],
        axis=2,
    )  # (N, num_classes, 4)
    return pred_boxes.reshape(rel_codes.shape[0], -1).astype(np.float32)  # (N, 4*num_classes)


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def clip_boxes_to_image(boxes, size):
    h, w = size
    out = boxes.copy()
    out[:, 0] = np.clip(out[:, 0], 0, w)
    out[:, 2] = np.clip(out[:, 2], 0, w)
    out[:, 1] = np.clip(out[:, 1], 0, h)
    out[:, 3] = np.clip(out[:, 3], 0, h)
    return out


def remove_small_boxes(boxes, min_size):
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    return np.where((ws >= min_size) & (hs >= min_size))[0]


def nms(boxes, scores, iou_threshold):
    """Standard greedy NMS. Returns indices to keep, sorted by score descending.
    Suppression tracked via a fixed-size boolean mask rather than rebuilding a
    Python list each iteration (~3-5x faster in practice, bit-identical
    output - verified against the naive list-based version)."""
    order = np.argsort(-scores)
    boxes = boxes[order]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    n = len(order)
    suppressed = np.zeros(n, dtype=bool)
    keep = []
    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(i)
        if i + 1 >= n:
            break
        xx1 = np.maximum(x1[i], x1[i + 1 :])
        yy1 = np.maximum(y1[i], y1[i + 1 :])
        xx2 = np.minimum(x2[i], x2[i + 1 :])
        yy2 = np.minimum(y2[i], y2[i + 1 :])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[i + 1 :] - inter + 1e-12)
        suppressed[i + 1 :] |= iou > iou_threshold
    return order[np.asarray(keep, dtype=np.int64)]


def batched_nms(boxes, scores, idxs, iou_threshold):
    """NMS run independently per distinct value of idxs, via the coordinate-
    offset trick (torchvision's default for small N)."""
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    max_coord = boxes.max() if boxes.size else 0.0
    offsets = idxs.astype(np.float64) * (max_coord + 1.0)
    boxes_for_nms = boxes + offsets[:, None]
    return nms(boxes_for_nms, scores, iou_threshold)


def level_mapper(box_areas_sqrt, k_min, k_max, canonical_scale=ROI_CANONICAL_SCALE, canonical_level=ROI_CANONICAL_LEVEL, eps=1e-6):
    target = np.floor(canonical_level + np.log2(box_areas_sqrt / canonical_scale + eps))
    target = np.clip(target, k_min, k_max)
    return (target.astype(np.int64) - k_min)


def roi_align_single_level(feature, boxes, spatial_scale, output_size=ROI_OUTPUT_SIZE, sampling_ratio=ROI_SAMPLING_RATIO):
    """feature: (C,H,W) float32. boxes: (K,4) in the ORIGINAL image coordinate
    space (spatial_scale converts). aligned=False (matches the real model's
    MultiScaleRoIAlign, which doesn't pass aligned=True). Direct numpy port
    of torchvision's pure-python _roi_align reference kernel.

    NOTE: measured on real Raspberry Pi 5 (Cortex-A76) hardware, fully
    vectorizing this across all K ROIs at once (or in chunks) is SLOWER than
    the per-ROI loop below, not faster - confirmed via direct benchmarking
    (e.g. K=250 @ 256x160x160: loop=355ms vs fully-vectorized=515ms, and
    every chunk size in between was also slower than the loop). This is
    memory-bandwidth/cache-bound, not Python-overhead-bound, on this
    platform: gathering many scattered feature-map locations into one large
    buffer thrashes cache worse than many small per-ROI gathers do. Do not
    "optimize" this into a vectorized-over-K form without re-benchmarking on
    real target hardware first."""
    C, H, W = feature.shape
    K = boxes.shape[0]
    if K == 0:
        return np.zeros((0, C, output_size, output_size), dtype=np.float32)

    roi_start_w = boxes[:, 0] * spatial_scale
    roi_start_h = boxes[:, 1] * spatial_scale
    roi_end_w = boxes[:, 2] * spatial_scale
    roi_end_h = boxes[:, 3] * spatial_scale

    roi_width = np.maximum(roi_end_w - roi_start_w, 1.0)
    roi_height = np.maximum(roi_end_h - roi_start_h, 1.0)

    bin_size_h = roi_height / output_size
    bin_size_w = roi_width / output_size

    grid_h = sampling_ratio
    grid_w = sampling_ratio
    count = max(grid_h * grid_w, 1)

    ph = np.arange(output_size, dtype=np.float32)
    pw = np.arange(output_size, dtype=np.float32)
    iy = np.arange(grid_h, dtype=np.float32)
    ix = np.arange(grid_w, dtype=np.float32)

    # y: (K, output_size, grid_h), x: (K, output_size, grid_w)
    y = (
        roi_start_h[:, None, None]
        + ph[None, :, None] * bin_size_h[:, None, None]
        + (iy[None, None, :] + 0.5) * (bin_size_h[:, None, None] / grid_h)
    )
    x = (
        roi_start_w[:, None, None]
        + pw[None, :, None] * bin_size_w[:, None, None]
        + (ix[None, None, :] + 0.5) * (bin_size_w[:, None, None] / grid_w)
    )

    y = np.clip(y, 0, None)
    x = np.clip(x, 0, None)
    y_low = np.floor(y).astype(np.int64)
    x_low = np.floor(x).astype(np.int64)
    y_high = np.where(y_low >= H - 1, H - 1, y_low + 1)
    y_low = np.where(y_low >= H - 1, H - 1, y_low)
    y = np.where(y_low >= H - 1, y.astype(feature.dtype), y)
    x_high = np.where(x_low >= W - 1, W - 1, x_low + 1)
    x_low = np.where(x_low >= W - 1, W - 1, x_low)
    x = np.where(x_low >= W - 1, x.astype(feature.dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx

    # Per-ROI loop (K is small: <=1000 before RPN NMS, <=4 after final NMS -
    # far cheaper than the bookkeeping needed to vectorize the 6D gather
    # correctly). Everything inside is fully vectorized over (PH,PW,IY,IX).
    out = np.empty((K, C, output_size, output_size), dtype=np.float32)
    for k in range(K):
        yl, yh, xl, xh = y_low[k], y_high[k], x_low[k], x_high[k]  # each (output_size, grid)
        # y_idx/x_idx broadcast to (PH,PW,IY,IX); since axis 0 is a plain slice and the
        # two advanced-index axes are contiguous, numpy places the broadcast result
        # in-place (C stays first) -> v#: (C,PH,PW,IY,IX) directly, no transpose needed.
        v1 = feature[:, yl[:, None, :, None], xl[None, :, None, :]]
        v2 = feature[:, yl[:, None, :, None], xh[None, :, None, :]]
        v3 = feature[:, yh[:, None, :, None], xl[None, :, None, :]]
        v4 = feature[:, yh[:, None, :, None], xh[None, :, None, :]]

        hy_k, ly_k, hx_k, lx_k = hy[k], ly[k], hx[k], lx[k]  # each (output_size, grid)
        w1 = (hy_k[:, None, :, None] * hx_k[None, :, None, :])  # (PH,PW,IY,IX)
        w2 = (hy_k[:, None, :, None] * lx_k[None, :, None, :])
        w3 = (ly_k[:, None, :, None] * hx_k[None, :, None, :])
        w4 = (ly_k[:, None, :, None] * lx_k[None, :, None, :])
        val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4  # (C,PH,PW,IY,IX)
        out[k] = val.sum(axis=(-1, -2)) / count

    return out


def linear(x, weight, bias):
    return x @ weight.T + bias


def box_head_forward(pooled, weights):
    """pooled: (K, C, 7, 7) -> flattened -> fc6 -> relu -> fc7 -> relu -> logits.
    weights: dict from box_head_{TYPE}.npz. Direct port of TwoMLPHead +
    FastRCNNPredictor."""
    x = pooled.reshape(pooled.shape[0], -1)
    x = np.maximum(linear(x, weights["fc6_w"], weights["fc6_b"]), 0.0)
    x = np.maximum(linear(x, weights["fc7_w"], weights["fc7_b"]), 0.0)
    class_logits = linear(x, weights["cls_w"], weights["cls_b"])
    box_regression = linear(x, weights["bbox_w"], weights["bbox_b"])
    return class_logits, box_regression


def postprocess(trunk_outputs, image_size, box_head_weights):
    """trunk_outputs: dict with keys feat0..4 (H,W,256 NHWC), cls0..4 (H,W,3),
    bbox0..4 (H,W,12) - the raw NPU trunk outputs for ONE image. image_size:
    (H,W) of the image actually fed to the trunk (e.g. (640,640)).
    box_head_weights: dict loaded from box_head_{TYPE}.npz.

    Returns (boxes, scores) as numpy arrays, boxes in (x1,y1,x2,y2) pixel
    coords in the same space as image_size, already through the real
    thresholds/NMS/top-k - equivalent to torchvision RoIHeads' final output
    with the background class already removed. Empty arrays if nothing
    survives (mirrors the real model - no artificial fallback detection).
    """
    feats_nchw = [np.transpose(trunk_outputs[f"feat{i}"], (2, 0, 1)) for i in range(NUM_LEVELS)]
    cls_nchw = [np.transpose(trunk_outputs[f"cls{i}"], (2, 0, 1)) for i in range(NUM_LEVELS)]
    bbox_nchw = [np.transpose(trunk_outputs[f"bbox{i}"], (2, 0, 1)) for i in range(NUM_LEVELS)]
    feature_shapes = [f.shape[1:] for f in feats_nchw]  # (H,W) per level

    base_anchors = generate_base_anchors()
    anchors_per_level = grid_anchors(feature_shapes, image_size, base_anchors)
    num_anchors_per_level = [a.shape[0] for a in anchors_per_level]
    anchors = np.concatenate(anchors_per_level, axis=0).astype(np.float32)

    # objectness/bbox_deltas: (C,H,W) NCHW -> (H*W*A,) / (H*W*A,4), A=3 anchors/loc,
    # matching concat_box_prediction_layers' permute(0,2,3,1).reshape(-1, ...).
    objectness_parts, deltas_parts = [], []
    for cls, bbx in zip(cls_nchw, bbox_nchw):
        A, H, W = cls.shape
        objectness_parts.append(cls.transpose(1, 2, 0).reshape(-1))
        deltas_parts.append(bbx.transpose(1, 2, 0).reshape(-1, 4))
    objectness = np.concatenate(objectness_parts, axis=0)
    pred_bbox_deltas = np.concatenate(deltas_parts, axis=0)

    proposals = decode_boxes(pred_bbox_deltas, anchors, RPN_BOX_CODER_WEIGHTS)

    # --- RPN.filter_proposals ---
    offset = 0
    top_n_idx_parts = []
    for n in num_anchors_per_level:
        level_scores = objectness[offset : offset + n]
        k = min(RPN_PRE_NMS_TOP_N, n)
        top_local = np.argpartition(-level_scores, k - 1)[:k]
        top_local = top_local[np.argsort(-level_scores[top_local])]
        top_n_idx_parts.append(top_local + offset)
        offset += n
    top_n_idx = np.concatenate(top_n_idx_parts, axis=0)

    levels = np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(num_anchors_per_level)])
    sel_objectness = objectness[top_n_idx]
    sel_levels = levels[top_n_idx]
    sel_boxes = proposals[top_n_idx]
    sel_scores = _sigmoid(sel_objectness)

    sel_boxes = clip_boxes_to_image(sel_boxes, image_size)
    keep = remove_small_boxes(sel_boxes, RPN_MIN_SIZE)
    sel_boxes, sel_scores, sel_levels = sel_boxes[keep], sel_scores[keep], sel_levels[keep]
    keep = np.where(sel_scores >= RPN_SCORE_THRESH)[0]
    sel_boxes, sel_scores, sel_levels = sel_boxes[keep], sel_scores[keep], sel_levels[keep]
    keep = batched_nms(sel_boxes, sel_scores, sel_levels, RPN_NMS_THRESH)
    keep = keep[:RPN_POST_NMS_TOP_N]
    proposal_boxes = sel_boxes[keep]

    if proposal_boxes.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # --- RoIHeads: ROIAlign across the 4 finest levels (P2-P5), matching
    # MultiScaleRoIAlign's featmap_names=['0','1','2','3'] ---
    roi_feature_shapes = feature_shapes[:ROI_LEVELS]
    scales = [2.0 ** round(np.log2(fs[0] / image_size[0])) for fs in roi_feature_shapes]
    lvl_min = int(round(-np.log2(scales[0])))
    lvl_max = int(round(-np.log2(scales[-1])))
    box_sqrt_area = np.sqrt(box_area(proposal_boxes))
    assigned_levels = level_mapper(box_sqrt_area, lvl_min, lvl_max)

    C = feats_nchw[0].shape[0]
    pooled = np.zeros((proposal_boxes.shape[0], C, ROI_OUTPUT_SIZE, ROI_OUTPUT_SIZE), dtype=np.float32)
    for lvl in range(ROI_LEVELS):
        idx = np.where(assigned_levels == lvl)[0]
        if idx.size == 0:
            continue
        pooled[idx] = roi_align_single_level(feats_nchw[lvl], proposal_boxes[idx], scales[lvl])

    class_logits, box_regression = box_head_forward(pooled, box_head_weights)

    num_classes = class_logits.shape[-1]
    pred_boxes = decode_boxes(box_regression, proposal_boxes, ROI_BOX_CODER_WEIGHTS)  # (N, 4*num_classes)
    pred_boxes = pred_boxes.reshape(-1, num_classes, 4)  # (N, num_classes, 4), matches torch decode()'s wrapper reshape
    pred_scores = _softmax(class_logits, axis=-1)  # (N, num_classes)

    pred_boxes = clip_boxes_to_image(pred_boxes.reshape(-1, 4), image_size).reshape(-1, num_classes, 4)
    # drop background (class 0)
    boxes = pred_boxes[:, 1:, :].reshape(-1, 4)
    scores = pred_scores[:, 1:].reshape(-1)
    num_classes_minus_bg = num_classes - 1
    labels = np.tile(np.arange(1, num_classes_minus_bg + 1), proposal_boxes.shape[0])

    keep = np.where(scores > ROI_SCORE_THRESH)[0]
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = remove_small_boxes(boxes, ROI_MIN_BOX_SIZE)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    keep = batched_nms(boxes, scores, labels, ROI_NMS_THRESH)
    keep = keep[:ROI_DETECTIONS_PER_IMG]
    return boxes[keep], scores[keep]
