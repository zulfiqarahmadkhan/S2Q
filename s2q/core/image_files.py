"""File-naming conventions shared by the segmenters and the ROI step."""
from pathlib import Path

# every ordinary image format S2Q reads (OpenCV, scikit-image and tifffile all handle these)
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".bmp", ".dib", ".gif",
                    ".webp", ".jp2", ".j2k", ".pgm", ".ppm", ".pnm", ".pbm", ".sr", ".ras"}
# for file dialogs: "*.tif *.tiff *.png ..."
IMAGE_PATTERNS = " ".join(f"*{e}" for e in sorted(IMAGE_EXTENSIONS, key=lambda e: (e not in (".tif", ".tiff"), e)))
MASK_SUFFIX = "_cp_masks"

# Outputs written by S2Q / Cellpose that must never be segmented as inputs
_DERIVED_MARKERS = (MASK_SUFFIX, "_masks", "_flows", "_cellprob", "_cp_output", "_ROI")


def is_derived(path) -> bool:
    """Masks / overlays written by S2Q or Cellpose (judged by how the name ends, so raw
    images such as ``cell_ROI_3.tif`` are not mistaken for outputs)."""
    stem = Path(path).stem
    return MASK_SUFFIX in stem or stem.endswith(_DERIVED_MARKERS)


def channels_last(img):
    """(C, H, W) multichannel images (common for microscopy TIFFs) → (H, W, C)."""
    import numpy as np
    if img.ndim == 3 and img.shape[0] <= 4 < img.shape[-1]:
        return np.moveaxis(img, 0, -1)
    return img


def read_image(path):
    """Pixel values of an ordinary image file as (H, W) or (H, W, C), whatever its layout.

    TIFF stacks (Fiji / ImageJ hyperstacks, Z-stacks, time series, channel-first files) are reduced
    to one plane: first time point, maximum projection over Z, channels last. GIF / multi-frame
    files give their first frame; an alpha channel of PNG / WebP / GIF images is dropped.
    """
    import numpy as np
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            return _read_tiff(p)
        except Exception:
            pass                                   # odd TIFF: let scikit-image try
    from skimage.io import imread
    img = np.asarray(imread(str(p)))
    if img.ndim == 4:                              # frames of an animation / stack
        img = img[0]
    if img.ndim == 3 and img.shape[-1] > 4 and img.shape[0] <= 4:
        img = np.moveaxis(img, 0, -1)              # (C, H, W) -> (H, W, C)
    elif img.ndim == 3 and img.shape[-1] > 4:
        img = img[0]                               # (frames, H, W): first frame
    if img.ndim == 3 and suffix not in (".tif", ".tiff"):
        if img.shape[-1] == 2:                     # grey + alpha
            img = img[..., 0]
        elif img.shape[-1] == 4:                   # RGBA
            img = img[..., :3]
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    return img


def _read_tiff(path):
    """One plane of a TIFF using its axes (T -> first, Z -> max projection, C / S -> last)."""
    import numpy as np
    import tifffile
    with tifffile.TiffFile(str(path)) as tf:
        s = tf.series[0]
        arr = s.asarray()
        axes = list(s.axes.upper())
    if arr.ndim == 2:
        return arr
    # an unnamed axis (Q / I) of size 2-4 just before Y X is usually the channels
    if not any(a in ("C", "S") for a in axes) and len(axes) >= 3:
        i = len(axes) - 3
        if axes[i] in ("Q", "I") and 2 <= arr.shape[i] <= 4:
            axes[i] = "C"
    idx, kept = [], []
    for a, n in zip(axes, arr.shape):
        if a in ("Y", "X", "C", "S", "Z"):
            idx.append(slice(None))
            kept.append(a)
        elif a in ("Q", "I") and "Z" not in axes:
            idx.append(slice(None))                # an unnamed stack: treated as Z
            kept.append("Z")
        else:
            idx.append(0)                          # T and anything else: the first one
    out = arr[tuple(idx)]
    while "Z" in kept:                             # maximum projection
        i = kept.index("Z")
        out = out.max(axis=i)
        kept.pop(i)
    chans = [i for i, a in enumerate(kept) if a in ("C", "S")]
    if len(chans) == 2:                            # C and S (RGB per channel): first channel
        out = np.take(out, 0, axis=chans[0])
        kept.pop(chans[0])
        chans = [i for i, a in enumerate(kept) if a in ("C", "S")]
    if chans:
        out = np.moveaxis(out, chans[0], -1)
        if out.shape[-1] == 1:
            out = out[..., 0]
    return out


def label_dtype(labels):
    """uint16 holds up to 65,535 cells; more would wrap around and merge cells."""
    import numpy as np
    return np.uint32 if int(labels.max(initial=0)) > 65535 else np.uint16


def list_images(directory) -> list[str]:
    """Raw input images directly inside *directory* (masks and overlays excluded)."""
    return sorted(
        str(p) for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not is_derived(p)
    )


def list_microscopy(directory) -> list[str]:
    """ND2 / LIF / CZI / LSM / OME-TIFF files and OME-Zarr folders directly inside *directory*."""
    from s2q.core.microscopy import is_microscopy
    return sorted(str(p) for p in Path(directory).iterdir()
                  if is_microscopy(p) and (p.is_file() or p.suffix.lower() == ".zarr") and not is_derived(p))


def mask_path_for(image_path, output_dir) -> Path:
    return Path(output_dir) / f"{Path(image_path).stem}{MASK_SUFFIX}.tif"


def list_masks(directory) -> list[Path]:
    """Mask files (``<stem>_cp_masks.tif``, or .png / other image formats, e.g. from Cellpose)."""
    found = {}
    for p in Path(directory).glob(f"*{MASK_SUFFIX}.*"):
        if (p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.count(MASK_SUFFIX) == 1
                and p.stem.endswith(MASK_SUFFIX)):
            # the same mask in two formats: prefer TIFF (lossless, 16-bit)
            if p.stem not in found or p.suffix.lower() in (".tif", ".tiff"):
                found[p.stem] = p
    return sorted(found.values())


def original_stem(mask_path) -> str:
    return Path(mask_path).stem[: -len(MASK_SUFFIX)]
