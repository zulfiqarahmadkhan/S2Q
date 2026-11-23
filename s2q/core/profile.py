"""Intensity profile along a line, for every channel of an image."""
from __future__ import annotations

import numpy as np


def line_profile(channels: list[tuple[str, np.ndarray]], p0, p1, width: int = 1, labels: np.ndarray | None = None,
                 pixel_size: float | None = None) -> dict:
    """Profile of each channel from *p0* to *p1* (image coordinates ``(x, y)``).

    *width* (px) is the thickness of the line: at each point the pixels across the
    line are averaged (the plotted profile) and summed (for the integrated intensity).

    Returns ``{"distance", "unit", "length", "channels": [{name, profile, mean, max, min, integrated}],
    "labels"}`` where *integrated* is the sum of all pixel values inside the line band and
    *labels* is the mask label under each point (0 = background) when a mask is given.
    """
    from skimage.measure import profile_line
    (x0, y0), (x1, y1) = p0, p1
    src, dst = (float(y0), float(x0)), (float(y1), float(x1))
    width = max(1, int(width))
    length_px = float(np.hypot(x1 - x0, y1 - y0))
    out = []
    for name, img in channels:
        a = np.asarray(img, dtype=np.float64)
        mean = profile_line(a, src, dst, linewidth=width, order=1, mode="reflect", reduce_func=np.mean)
        total = profile_line(a, src, dst, linewidth=width, order=1, mode="reflect", reduce_func=np.sum)
        out.append({"name": name, "profile": mean, "mean": float(mean.mean()), "max": float(mean.max()),
                    "min": float(mean.min()), "integrated": float(total.sum())})
    n = len(out[0]["profile"]) if out else 0
    scale = pixel_size or 1.0
    distance = np.linspace(0, length_px * scale, n) if n > 1 else np.zeros(n)
    lab = None
    if labels is not None and n:
        lab = np.ravel(profile_line(labels, src, dst, linewidth=1, order=0, mode="constant",
                                    reduce_func=np.max)).astype(np.int64)[:n]
    return {"distance": distance, "unit": "µm" if pixel_size else "px", "length": length_px * scale,
            "width": width, "channels": out, "labels": lab}
