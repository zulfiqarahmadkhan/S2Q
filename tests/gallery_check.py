"""Render every gallery plot with its example data (and some stress variations).

    python tests/gallery_check.py [out_dir] [--module bars] [--sheets]

Prints one line per failure and a summary; with --sheets also writes one contact sheet PNG per
category into out_dir so the plots can be looked at quickly.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[0]) if args else Path("gallery_out")
    out.mkdir(parents=True, exist_ok=True)
    module = sys.argv[sys.argv.index("--module") + 1] if "--module" in sys.argv else None
    from s2q.core import gallery as G
    from s2q.core.gallery.base import auto_columns, render
    specs = [s for s in G.REGISTRY.values() if module is None or s.draw.__module__.endswith(module)]
    fails, slow = [], []
    for s in specs:
        t0 = time.time()
        try:
            df = s.example()
            cols = auto_columns(s, df)
            missing = [r.key for r in s.roles if not r.optional and not cols.get(r.key)]
            if missing:
                raise ValueError(f"auto_columns left required roles empty: {missing} (columns {list(df.columns)})")
            fig = render(s, df, cols)
            fig.savefig(out / f"{s.id}.png", dpi=55, facecolor="white")
            # stress: other legend placements, a different palette, user labels
            for o in ({"legend": "below", "palette": "Grey", "title": "T", "xlabel": "X", "ylabel": "Y"},
                      {"legend": "off", "font": 16}, {"legend": "outside right", "grid": False}):
                render(s, df, cols, o).savefig(out / "_tmp.png", dpi=30)
        except Exception as e:
            fails.append((s.id, f"{type(e).__name__}: {e}", traceback.format_exc(limit=-3)))
        dt = time.time() - t0
        if dt > 6:
            slow.append((s.id, round(dt, 1)))
    for fid, msg, tb in fails:
        print(f"FAIL {fid}: {msg}\n{tb}")
    print(f"{len(specs) - len(fails)}/{len(specs)} plot types rendered" + (f"; slow: {slow}" if slow else ""))
    if "--sheets" in sys.argv:
        sheets(G, out)
    return 1 if fails else 0


def sheets(G, out):
    """One contact sheet per category: every plot with its id and name."""
    from PIL import Image, ImageDraw
    for cat, specs in G.by_category().items():
        tiles = []
        for s in specs:
            p = out / f"{s.id}.png"
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((380, 300))
            tile = Image.new("RGB", (390, 330), "white")
            tile.paste(im, ((390 - im.width) // 2, 26))
            ImageDraw.Draw(tile).text((6, 6), f"{s.number} {s.name}"[:60], fill="black")
            tiles.append(tile)
        if not tiles:
            continue
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 390, rows * 330), "#cccccc")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 390, (i // cols) * 330))
        name = "".join(ch if ch.isalnum() else "_" for ch in cat)[:40]
        sheet.save(out / f"sheet_{name}.png")


if __name__ == "__main__":
    sys.exit(main())
