"""Measure the cells of one mask on other images (channels, time points, other stainings).

``measure_on_images(mask, images)`` gives, per cell, the area, centre and — for every image
channel — the mean and the integrated intensity (sum of the original pixel values inside the
cell). With ``subtract_background`` the background (median of the pixels outside every cell)
is also reported and removed: corrected integrated = integrated − area × background
(the "corrected total cell fluorescence", CTCF).
"""
from __future__ import annotations

import numpy as np


def measure_on_images(mask: np.ndarray, images: list, subtract_background: bool = True,
                      pixel_size: float | None = None):
    """*images*: list of (image name, [(channel name, 2-D array), ...]) of the mask's size.
    *pixel_size* (µm per pixel) adds the area in µm² and the centre in µm.

    Returns (per-cell DataFrame, per-image-channel summary DataFrame).
    """
    import pandas as pd
    mask = np.asarray(mask)
    ids = np.unique(mask)
    ids = ids[ids > 0]
    flat = mask.ravel()
    n = int(mask.max()) + 1
    area = np.bincount(flat, minlength=n)
    yy, xx = np.indices(mask.shape)
    cy = np.bincount(flat, yy.ravel(), minlength=n)
    cx = np.bincount(flat, xx.ravel(), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        cells = pd.DataFrame({"Cell": ids, "Area (px)": area[ids],
                              "Centre X": np.round(cx[ids] / area[ids], 1),
                              "Centre Y": np.round(cy[ids] / area[ids], 1)})
        if pixel_size:
            k = float(pixel_size)
            cells["Area (µm²)"] = np.round(area[ids] * k * k, 4)
            cells["Centre X (µm)"] = np.round(cx[ids] / area[ids] * k, 3)
            cells["Centre Y (µm)"] = np.round(cy[ids] / area[ids] * k, 3)
    summary = []
    background = mask == 0
    for img_name, channels in images:
        for ch_name, data in channels:
            data = np.asarray(data, dtype=np.float64)
            if data.shape != mask.shape:
                raise ValueError(f"{img_name} is {data.shape[1]}×{data.shape[0]}, "
                                 f"the mask {mask.shape[1]}×{mask.shape[0]}.")
            col = img_name if len(channels) == 1 else f"{img_name} [{ch_name}]"
            total = np.bincount(flat, data.ravel(), minlength=n)[ids]
            with np.errstate(invalid="ignore", divide="ignore"):
                cells[f"{col}: Mean"] = total / area[ids]
            cells[f"{col}: Integrated"] = total
            bg = float(np.median(data[background])) if background.any() else 0.0
            row = {"Image": img_name, "Channel": ch_name, "Cells": len(ids),
                   "Total integrated": float(total.sum()),
                   "Mean integrated per cell": float(total.mean()) if len(ids) else np.nan,
                   "Mean intensity in cells": float(data[~background].mean()) if (~background).any() else np.nan}
            if subtract_background:
                corr = total - area[ids] * bg
                cells[f"{col}: Integrated − background"] = corr
                row.update({"Background (median outside cells)": bg,
                            "Total integrated − background": float(corr.sum()),
                            "Mean integrated − background per cell": float(corr.mean()) if len(ids) else np.nan})
            summary.append(row)
    return cells, pd.DataFrame(summary)


def save_measurements(path, cells, summary, info: dict | None = None):
    """Excel (sheets "Per cell", "Summary", "Info") or, for .csv, the per-cell table."""
    import pandas as pd
    path = str(path)
    if path.lower().endswith(".csv"):
        cells.to_csv(path, index=False)
        return
    with pd.ExcelWriter(path) as xw:
        cells.to_excel(xw, sheet_name="Per cell", index=False)
        summary.to_excel(xw, sheet_name="Summary", index=False)
        if info:
            pd.DataFrame(list(info.items()), columns=["Setting", "Value"]).to_excel(xw, sheet_name="Info",
                                                                                   index=False)
