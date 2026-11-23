"""Pixel → micrometre conversion.

``read_pixel_size(path)`` returns the µm per pixel a TIFF records (ImageJ calibration or OME
metadata), or None. Only calibrations that name a length unit are trusted: a plain "72 dpi"
resolution tag says nothing about the microscope.
"""
from __future__ import annotations

from pathlib import Path

# factor to µm for the length units files use
_UNITS = {"micron": 1.0, "microns": 1.0, "um": 1.0, "µm": 1.0, "\\u00b5m": 1.0, "μm": 1.0,
          "micrometer": 1.0, "micrometre": 1.0, "nm": 1e-3, "nanometer": 1e-3, "mm": 1e3,
          "millimeter": 1e3, "cm": 1e4, "m": 1e6}


def _unit_factor(unit) -> float | None:
    if unit is None:
        return None
    return _UNITS.get(str(unit).strip().lower())


def read_pixel_size(path) -> float | None:
    """µm per pixel recorded in *path*, or None when the file does not say."""
    try:
        p = Path(path)
        if p.suffix.lower() not in (".tif", ".tiff"):
            from s2q.core import microscopy
            if microscopy.is_microscopy(p):
                px = microscopy.info(p).pixel_size_um
                return float(px) if px else None
            return None
        import tifffile
        with tifffile.TiffFile(str(p)) as tf:
            page = tf.pages[0]
            # OME-TIFF
            if tf.is_ome and tf.ome_metadata:
                import re
                m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tf.ome_metadata)
                if m:
                    u = re.search(r'PhysicalSizeXUnit="([^"]+)"', tf.ome_metadata)
                    f = _unit_factor(u.group(1)) if u else 1.0
                    v = float(m.group(1)) * (f or 1.0)
                    return v if v > 0 else None
            # ImageJ calibration: XResolution = pixels per unit, unit in the ImageJ description
            ij = tf.imagej_metadata or {}
            f = _unit_factor(ij.get("unit"))
            tag = page.tags.get("XResolution")
            if f and tag is not None:
                num, den = tag.value
                if num > 0:
                    v = den / num * f
                    return v if v > 0 else None
            # resolution unit centimetre (TIFF ResolutionUnit = 3) is a real calibration
            ru = page.tags.get("ResolutionUnit")
            if tag is not None and ru is not None and int(ru.value) == 3:
                num, den = tag.value
                if num > 0:
                    v = den / num * 1e4
                    # a scanned page (e.g. 300 dpi ≈ 85 µm) is not a microscope image
                    return v if 0 < v < 50 else None
    except Exception:
        return None
    return None


def fmt_um(px_value: float, pixel_size: float | None, power: int = 1, digits: int = 2) -> str:
    """'123 px' or '123 px (4.56 µm)' for lengths (power 1) or areas (power 2)."""
    unit = "px" if power == 1 else "px²"
    txt = f"{px_value:,.0f} {unit}" if float(px_value).is_integer() else f"{px_value:,.1f} {unit}"
    if pixel_size:
        um = px_value * pixel_size ** power
        txt += f" ({um:,.{digits}f} µm{'²' if power == 2 else ''})"
    return txt
