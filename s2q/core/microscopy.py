"""Reading microscopy files: ND2, LIF, CZI, LSM, OME-TIFF and OME-Zarr.

These files can hold several series (positions), time points, channels and
Z-planes. :func:`info` describes a file; :func:`read_plane` returns one 2-D plane
(or an RGB composite of the first channels) with its original pixel values.

Reading uses *bioio* and its format plugins (bioio-nd2, bioio-lif, bioio-czi,
bioio-ome-tiff, bioio-ome-zarr, bioio-tifffile). Without them, LSM and OME-TIFF
are still read with *tifffile*; the other formats then ask for the plugins.

Masks of multi-plane files are stored per series / time point / Z-plane:
``<name>[_s<series>][_t<time>][_z<z>]_cp_masks.tif`` (only the parts that vary).
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MICROSCOPY_SUFFIXES = (".nd2", ".lif", ".czi", ".lsm", ".ome.tif", ".ome.tiff", ".ome.zarr", ".zarr")
FILE_FILTER = "Microscopy (*.nd2 *.lif *.czi *.lsm *.ome.tif *.ome.tiff)"
_PLANE_RE = re.compile(r"(?:_s(\d+))?(?:_t(\d+))?(?:_z(\d+))?$")

_lock = threading.RLock()
_cache: dict[str, object] = {}          # path -> opened BioImage (the last few)


def is_microscopy(path) -> bool:
    name = str(path).lower().rstrip("/\\")
    return name.endswith(MICROSCOPY_SUFFIXES)


def base_name(path) -> str:
    """File name without the (possibly double) extension: x.ome.tiff -> x."""
    name = Path(str(path).rstrip("/\\")).name
    low = name.lower()
    for suf in sorted(MICROSCOPY_SUFFIXES, key=len, reverse=True):
        if low.endswith(suf):
            return name[: -len(suf)]
    return Path(name).stem


@dataclass
class Plane:
    """Which plane to show. ``z=-1`` = maximum projection; ``c=-1`` = all channels (composite)."""
    scene: int = 0
    t: int = 0
    z: int = -1
    c: int = -1


@dataclass
class ImageInfo:
    scenes: list[str] = field(default_factory=lambda: ["0"])
    T: int = 1
    C: int = 1
    Z: int = 1
    Y: int = 0
    X: int = 0
    channel_names: list[str] = field(default_factory=list)
    rgb: bool = False
    pixel_size_um: float | None = None       # physical X pixel size, when the file records it

    @property
    def multi(self) -> bool:
        return len(self.scenes) > 1 or self.T > 1 or self.C > 1 or self.Z > 1


def _bioimage(path: str):
    with _lock:
        img = _cache.get(path)
        if img is None:
            try:
                from bioio import BioImage
            except ImportError as e:
                raise ImportError("Reading this format needs the bioio readers: "
                                  "pip install bioio bioio-nd2 bioio-lif bioio-czi bioio-ome-tiff "
                                  "bioio-ome-zarr bioio-tifffile") from e
            img = BioImage(path)
            if len(_cache) >= 4:
                _cache.pop(next(iter(_cache)))
            _cache[path] = img
        return img


def info(path, scene: int = 0) -> ImageInfo:
    path = str(path)
    try:
        img = _bioimage(path)
    except ImportError:
        return _tiff_info(path)
    with _lock:
        img.set_scene(scene)
        d = img.dims
        # files without channel names get placeholders like "Channel:0:1"; show those as "Channel 2"
        names = [f"Channel {int(m.group(1)) + 1}" if (m := re.fullmatch(r"Channel:\d+:(\d+)", str(n))) else str(n)
                 for n in (img.channel_names or [])]
        try:
            px = img.physical_pixel_sizes.X
            px = float(px) if px and px > 0 else None
        except Exception:
            px = None
        return ImageInfo(scenes=[str(s) for s in img.scenes], T=d.T, C=d.C, Z=d.Z, Y=d.Y, X=d.X,
                         channel_names=names, rgb="S" in d.order and getattr(d, "S", 1) > 1, pixel_size_um=px)


def read_channels(path, plane: Plane | None = None) -> list[tuple[str, np.ndarray]]:
    """Every channel of one plane as (name, 2-D array with original values) — for line profiles."""
    from copy import copy
    plane = plane or Plane()
    inf = info(path, plane.scene)
    names = inf.channel_names or [f"Channel {i + 1}" for i in range(inf.C)]
    out = []
    for c in range(inf.C):
        p = copy(plane)
        p.c = c
        arr = read_plane(path, p)
        if arr.ndim == 3:                      # RGB samples inside one channel
            out += [(f"{names[c]} {rgb}" if inf.C > 1 else rgb, arr[..., i]) for i, rgb in enumerate("RGB"[:arr.shape[-1]])]
        else:
            out.append((names[c] if c < len(names) else f"Channel {c + 1}", arr))
    return out


def read_plane(path, plane: Plane | None = None) -> np.ndarray:
    """One plane with its original values: (Y, X) or (Y, X, 3) for RGB / channel composites."""
    plane = plane or Plane()
    path = str(path)
    try:
        img = _bioimage(path)
    except ImportError:
        return _tiff_plane(path, plane)
    with _lock:
        img.set_scene(min(plane.scene, len(img.scenes) - 1))
        d = img.dims
        t = min(max(plane.t, 0), d.T - 1)
        rgb = "S" in d.order and getattr(d, "S", 1) > 1
        yx = "YXS" if rgb else "YX"

        def one(c):
            if plane.z < 0 and d.Z > 1:
                return img.get_image_data("Z" + yx, T=t, C=c).max(axis=0)
            return img.get_image_data(yx, T=t, C=c, Z=min(max(plane.z, 0), d.Z - 1))

        if rgb:
            return one(0 if plane.c < 0 else min(plane.c, d.C - 1))
        if plane.c >= 0 or d.C == 1:
            return one(min(max(plane.c, 0), d.C - 1))
        chans = [one(c) for c in range(min(3, d.C))]
        while len(chans) < 3:
            chans.append(np.zeros_like(chans[0]))
        return np.stack(chans, axis=-1)


# ── masks per plane ──────────────────────────────────────────────────────────

def plane_suffix(inf: ImageInfo, plane: Plane) -> str:
    s = ""
    if len(inf.scenes) > 1:
        s += f"_s{plane.scene}"
    if inf.T > 1:
        s += f"_t{plane.t}"
    if inf.Z > 1 and plane.z >= 0:
        s += f"_z{plane.z}"
    return s


def mask_name(image_path, inf: ImageInfo, plane: Plane, mask_suffix: str = "_cp_masks") -> str:
    return f"{base_name(image_path)}{plane_suffix(inf, plane)}{mask_suffix}.tif"


def plane_from_mask_name(image_path, mask_path, mask_suffix: str = "_cp_masks") -> Plane:
    """The plane a mask file belongs to, from its name (plane 0 / max projection if not encoded)."""
    stem = Path(mask_path).name
    if stem.endswith(mask_suffix + ".tif"):
        stem = stem[: -len(mask_suffix + ".tif")]
    rest = stem[len(base_name(image_path)):] if stem.startswith(base_name(image_path)) else ""
    m = _PLANE_RE.match(rest) if rest else None
    if not m or m.end() != len(rest):
        return Plane()
    s, t, z = m.groups()
    return Plane(scene=int(s or 0), t=int(t or 0), z=int(z) if z is not None else -1)


def strip_plane_suffix(stem: str) -> str:
    """'x_s0_t3' -> 'x' (to match a per-plane mask to its microscopy file)."""
    return _PLANE_RE.sub("", stem) or stem


# ── fallback without bioio: LSM / OME-TIFF with tifffile ─────────────────────

def _tiff_series(path):
    import tifffile
    low = path.lower()
    if not low.endswith((".lsm", ".tif", ".tiff")):
        raise ImportError(f"Reading {Path(path).suffix} files needs the bioio readers "
                          "(pip install bioio bioio-nd2 bioio-lif bioio-czi bioio-ome-zarr).")
    tf = tifffile.TiffFile(path)
    return tf, tf.series[0]


def _tiff_info(path) -> ImageInfo:
    tf, s = _tiff_series(path)
    try:
        ax = s.axes.upper()
        dims = dict(zip(ax, s.shape))
        return ImageInfo(T=dims.get("T", 1), C=dims.get("C", 1), Z=dims.get("Z", 1),
                         Y=dims.get("Y", 0), X=dims.get("X", 0), rgb="S" in ax)
    finally:
        tf.close()


def _tiff_plane(path, plane: Plane) -> np.ndarray:
    tf, s = _tiff_series(path)
    try:
        arr = s.asarray()
        ax = s.axes.upper()
    finally:
        tf.close()
    idx = []
    for a, n in zip(ax, arr.shape):
        if a in ("Y", "X", "S"):
            idx.append(slice(None))
        elif a == "T":
            idx.append(min(plane.t, n - 1))
        elif a == "Z":
            idx.append(slice(None) if plane.z < 0 else min(plane.z, n - 1))
        elif a == "C":
            idx.append(slice(None) if plane.c < 0 else min(plane.c, n - 1))
        else:
            idx.append(0)
    out = arr[tuple(idx)]
    kept = [a for a, i in zip(ax, idx) if isinstance(i, slice)]
    if "Z" in kept:
        out = out.max(axis=kept.index("Z"))
        kept.remove("Z")
    if "C" in kept:
        out = np.moveaxis(out, kept.index("C"), -1)[..., :3]
        if out.shape[-1] < 3:
            out = np.concatenate([out, np.zeros(out.shape[:-1] + (3 - out.shape[-1],), out.dtype)], -1)
    return out
