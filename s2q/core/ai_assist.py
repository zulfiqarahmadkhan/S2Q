"""AI-assisted adding of missed cells: ``segment_all`` adds every missed cell in a box or
selected area; ``segment_box`` (below) finds one missed cell for a click.

Strategy (first success wins):
  1. Cellpose-SAM on the box padded by 100 % of its size (context matters).
  2. Same crop, telling the model the expected cell diameter (box size).
  3. Bigger context (box padded by 200 %).
  4. Fallback without AI: Otsu threshold of the brightness inside the box.

The chosen object is the one covering the most *unsegmented* area inside the
box, so neighbouring cells that already exist are never picked.
"""
from __future__ import annotations

import logging

import numpy as np

from s2q.core.mask_edit import pick_object

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_GPU = None      # device the cached model was loaded on


def get_model():
    """Cellpose-SAM, loaded once per session (again if the GPU/CPU choice changed). Call from the main thread."""
    global _MODEL, _MODEL_GPU
    from s2q.core.device import use_gpu
    gpu = use_gpu()
    if _MODEL is None or _MODEL_GPU != gpu:
        from cellpose import models
        _MODEL = models.CellposeModel(gpu=gpu, pretrained_model="cpsam")
        _MODEL_GPU = gpu
    return _MODEL


def model_loaded() -> bool:
    from s2q.core.device import use_gpu
    return _MODEL is not None and _MODEL_GPU == use_gpu()


def _crop(image, box, pad_frac):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    pad = max(32, int(pad_frac * max(x1 - x0, y1 - y0)))
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(w, x1 + pad + 1), min(h, y1 + pad + 1)
    return cx0, cy0, cx1, cy1


def _brightness_fallback(intensity, free, box):
    """Largest Otsu component inside the box that lies on unsegmented pixels."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    x0, y0, x1, y1 = box
    sub = intensity[y0:y1 + 1, x0:x1 + 1]
    if sub.size < 25 or sub.max() == sub.min():
        return None
    t = threshold_otsu(sub)
    # cells are the minority class in a tight box: pick bright or dark accordingly
    fg = sub > t if (sub > t).mean() <= 0.5 else sub <= t
    fg &= free[y0:y1 + 1, x0:x1 + 1]
    lab, n = ndi.label(fg)
    if not n:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    best = int(np.argmax(sizes)) + 1
    if sizes[best - 1] < 20:
        return None
    obj = ndi.binary_fill_holes(lab == best)
    full = np.zeros(intensity.shape, bool)
    full[y0:y1 + 1, x0:x1 + 1] = obj
    return full


def segment_all(model, image: np.ndarray, free: np.ndarray, box, region: np.ndarray | None = None,
                min_size: int = 20) -> np.ndarray:
    """Every missed cell in *box* = (x0, y0, x1, y1) (and inside *region*, a full-size bool mask, if given).

    Cellpose-SAM runs once on the box plus some context. A predicted cell is kept when its centre
    lies in the box / region and most of it is on unsegmented pixels (*free*), so cells that
    already exist are not added again. Returns an int32 label image (full size, 0 = nothing new).
    """
    from scipy import ndimage as ndi
    h, w = free.shape
    x0, y0, x1, y1 = box
    cx0, cy0, cx1, cy1 = _crop(image, box, 0.25)
    crop = (image[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)
    if crop.shape[2] == 1:
        crop = crop[:, :, 0]
    labels = np.asarray(model.eval(crop)[0]).astype(np.int32)
    out = np.zeros((h, w), np.int32)
    if labels.max() == 0:
        return out
    fr = free[cy0:cy1, cx0:cx1]
    inside = np.zeros(labels.shape, bool)
    inside[y0 - cy0:y1 - cy0 + 1, x0 - cx0:x1 - cx0 + 1] = True
    if region is not None:
        inside &= region[cy0:cy1, cx0:cx1]
    ids = np.arange(1, labels.max() + 1)
    sizes = np.bincount(labels.ravel(), minlength=len(ids) + 1)[1:]
    new_px = np.bincount(labels[fr].ravel(), minlength=len(ids) + 1)[1:]
    cents = ndi.center_of_mass(np.ones_like(labels), labels, ids)
    n = 0
    sub = out[cy0:cy1, cx0:cx1]
    for i, (cy, cx), size, new in zip(ids, cents, sizes, new_px):
        if size == 0 or new < min_size or new < 0.5 * size:
            continue                                   # tiny, or mostly an existing cell
        if not inside[min(int(round(cy)), inside.shape[0] - 1), min(int(round(cx)), inside.shape[1] - 1)]:
            continue                                   # centre outside the box / area
        n += 1
        sub[(labels == i) & fr] = n
    return out


def segment_box(model, image: np.ndarray, free: np.ndarray, box, from_click: bool = False):
    """Find the missed cell in *box* = (x0, y0, x1, y1).

    *image* is float (H, W, C) in [0, 1]; *free* is True where the current mask
    is background. Returns (boolean object mask in full-image coordinates,
    method string) or (None, reason).
    """
    x0, y0, x1, y1 = box
    attempts = [(1.0, None), (2.0, None)]
    if not from_click:
        attempts.insert(1, (1.0, max(x1 - x0, y1 - y0) * 0.8))
    for pad_frac, diameter in attempts:
        cx0, cy0, cx1, cy1 = _crop(image, box, pad_frac)
        crop = (image[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)
        if crop.shape[2] == 1:
            crop = crop[:, :, 0]
        kw = {"diameter": diameter} if diameter else {}
        try:
            labels = np.asarray(model.eval(crop, **kw)[0])
        except Exception as e:
            logger.warning(f"AI attempt failed: {e}")
            continue
        obj = pick_object(labels, (x0 - cx0, y0 - cy0, x1 - cx0 + 1, y1 - cy0 + 1),
                          free[cy0:cy1, cx0:cx1])
        if obj is not None and (obj & free[cy0:cy1, cx0:cx1]).sum() >= 20:
            full = np.zeros(free.shape, bool)
            full[cy0:cy1, cx0:cx1] = obj
            return full, "Cellpose-SAM"
    obj = _brightness_fallback(image.mean(axis=2), free, box)
    if obj is not None:
        return obj, "brightness (AI found nothing)"
    return None, "no cell found"
