"""Card pictures of the gallery start screen.

Pictures are stored as ``s2q/core/gallery/thumbs/<plot id>.png`` (made with
``python -m s2q.core.gallery.thumbs``). A missing picture is drawn from the plot's example data
the first time it is needed and kept in the user's cache folder.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

THUMB_DIR = Path(__file__).with_name("thumbs")
THUMB_W, THUMB_H = 520, 380            # stored size (cards show it at about half, sharp on HiDPI)


def cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(Path.home(), ".cache")
    d = Path(base) / "S2Q" / "gallery_thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def thumb_path(spec) -> Path | None:
    """The stored or cached picture of *spec*, or None if there is none yet."""
    for d in (THUMB_DIR, cache_dir()):
        p = d / f"{spec.id}.png"
        if p.exists():
            return p
    return None


def render_thumb_png(spec) -> bytes:
    """PNG bytes of *spec* drawn with its example data, fitted into THUMB_W × THUMB_H on white."""
    import warnings
    from PIL import Image
    from s2q.core.gallery.base import render
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, h = spec.size
        fig = render(spec, spec.example(), None, {"legend": "best"}, size=(w, h), dpi=100)
        buf = io.BytesIO()
        dpi = max(40, min(110, THUMB_W * 1.35 / w))
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    im.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), "white")
    canvas.paste(im, ((THUMB_W - im.width) // 2, (THUMB_H - im.height) // 2))
    out = io.BytesIO()
    # 256-colour palette: a third of the size, no visible difference at card size
    canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=256).save(out, format="PNG", optimize=True)
    return out.getvalue()


def make_missing(spec) -> Path:
    """Draw and cache the picture of *spec* (safe to call from a worker thread)."""
    p = cache_dir() / f"{spec.id}.png"
    p.write_bytes(render_thumb_png(spec))
    return p


def build_all(target: Path = THUMB_DIR, only_missing: bool = False):
    """Write the pictures of every plot type into *target* (run when plot types change)."""
    from s2q.core import gallery as G
    target.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for spec in G.REGISTRY.values():
        p = target / f"{spec.id}.png"
        if only_missing and p.exists():
            continue
        try:
            p.write_bytes(render_thumb_png(spec))
            done += 1
        except Exception as e:                      # one broken example must not stop the rest
            failed += 1
            print(f"FAIL {spec.id}: {e}", file=sys.stderr)
    print(f"{done} pictures written to {target}" + (f", {failed} failed" if failed else ""))
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if build_all(only_missing="--missing" in sys.argv) else 1)
