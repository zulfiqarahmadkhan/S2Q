"""Find badly placed text in the gallery plots.

    python tests/gallery_text_check.py [--module bars] [--only id1,id2] [--verbose] [--png out_dir]

Every plot is drawn with its example data at several figure sizes and font sizes. For each
drawing it counts
  * out   – text that sticks out of the figure,
  * over  – pairs of texts that overlap (tick labels, value labels, annotations, legend…),
and prints the plots with problems (worst first) plus a total. ``--png`` also saves the drawings
that have problems, with the offending texts outlined in red.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

VARIANTS = [("default", 1.0, 11), ("small", 0.7, 11), ("large", 1.5, 11), ("big font", 1.0, 16),
            ("small + big font", 0.7, 14)]


def _texts(fig, renderer):
    """(text, bbox) of every text that is actually drawn."""
    from matplotlib.text import Text
    out = []
    tick_labels = set()

    def all_axes(axes):
        for a in axes:
            yield a
            yield from all_axes(getattr(a, "child_axes", []))   # e.g. a colour bar inside a corner

    for ax in all_axes(fig.axes):
        for axis in [getattr(ax, n) for n in ("xaxis", "yaxis", "zaxis") if hasattr(ax, n)]:
            for t in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_labels.add(id(t.label1))
                tick_labels.add(id(t.label2))
            if not ax.get_visible() or not ax.axison or not axis.get_visible():
                continue                                # hidden axis: its tick labels are not drawn
            try:
                ticks = axis._update_ticks()
            except Exception:
                continue
            try:
                lo, hi = sorted(axis.get_view_interval())
            except Exception:
                lo, hi = -float("inf"), float("inf")
            eps = (hi - lo) * 1e-6 if hi > lo else 0
            for t in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_labels.add(id(t.label1))
                tick_labels.add(id(t.label2))
            for t in ticks:
                loc = t.get_loc()
                if loc is None or not (lo - eps <= loc <= hi + eps):
                    continue                        # computed but not drawn (outside the axis range)
                for lab in (t.label1, t.label2):
                    if lab.get_visible() and t.get_visible() and lab.get_text().strip():
                        out.append(lab)
    for t in fig.findobj(Text):
        if id(t) in tick_labels or not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_alpha() == 0:
            continue
        from matplotlib.text import Annotation
        if isinstance(t, Annotation):
            try:
                if not t._check_xy(renderer):      # its point is outside the plot: not drawn
                    continue
            except Exception:
                pass
        # skip texts whose parent artist is invisible (e.g. removed legend)
        out.append(t)
    res = []
    seen = set()
    for t in out:
        if id(t) in seen:
            continue
        seen.add(id(t))
        try:
            # the text itself (an annotation's own size includes its pointer line)
            bb = Text.get_window_extent(t, renderer)
        except Exception:
            continue
        if bb.width < 1 or bb.height < 1:
            continue
        # texts clipped to their axes and fully outside it are not drawn
        if t.get_clip_on() and t.axes is not None and t.get_clip_box() is not None:
            cb = t.get_clip_box()
            if not cb.overlaps(bb):
                continue
        res.append((t, bb))
    return res


def check_figure(fig):
    """(texts outside the figure, overlapping pairs) for one figure."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if not isinstance(fig.canvas, FigureCanvasAgg):
        FigureCanvasAgg(fig)
    for _ in range(3):                  # the app draws a figure several times: layouts must hold
        fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = _texts(fig, r)
    outs, overs = [], []
    for t, bb in texts:
        if bb.x0 < -2 or bb.y0 < -2 or bb.x1 > W + 2 or bb.y1 > H + 2:
            outs.append((t, bb))
    import math
    sizes = {}

    def upright_size(t):
        """Width / height of the text as if it were not turned."""
        if id(t) not in sizes:
            rot = t.get_rotation()
            if rot % 180 == 0:
                bb = t.get_window_extent(r)
                sizes[id(t)] = (bb.width, bb.height)
            else:
                t.set_rotation(0)
                bb = t.get_window_extent(r)
                t.set_rotation(rot)
                sizes[id(t)] = (bb.width, bb.height)
        return sizes[id(t)]

    for i in range(len(texts)):
        t1, b1 = texts[i]
        for j in range(i + 1, len(texts)):
            t2, b2 = texts[j]
            if not b1.overlaps(b2):
                continue
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix <= 1 or iy <= 1:
                continue
            rot = t1.get_rotation()
            if rot % 180 != 0 and abs(rot - t2.get_rotation()) < 1:
                # two labels turned the same way: compare them along the slant, not their boxes
                a = math.radians(rot)
                cx, cy = (b2.x0 + b2.x1 - b1.x0 - b1.x1) / 2, (b2.y0 + b2.y1 - b1.y0 - b1.y1) / 2
                u = cx * math.cos(a) + cy * math.sin(a)
                v = -cx * math.sin(a) + cy * math.cos(a)
                (w1, h1), (w2, h2) = upright_size(t1), upright_size(t2)
                ou, ov = (w1 + w2) / 2 - abs(u), (h1 + h2) / 2 - abs(v)
                if ou <= 1.5 or ov <= 1.5:
                    continue
            elif t1.get_rotation() % 180 != 0 or t2.get_rotation() % 180 != 0:
                # differently turned texts (e.g. around a circle): their boxes are only rough
                small = min(b1.width * b1.height, b2.width * b2.height)
                if ix * iy <= 0.25 * small:
                    continue
            elif ix <= 2 or iy <= 2:
                continue                       # strict: any visible touching of upright texts counts
            overs.append((t1, t2, b1, b2))
    return outs, overs


def main():
    out_png = Path(sys.argv[sys.argv.index("--png") + 1]) if "--png" in sys.argv else None
    module = sys.argv[sys.argv.index("--module") + 1] if "--module" in sys.argv else None
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    verbose = "--verbose" in sys.argv
    from s2q.core import gallery as G
    from s2q.core.gallery.base import render
    rows = []
    for s in G.REGISTRY.values():
        if module and not s.draw.__module__.endswith(module):
            continue
        if only and s.id not in only:
            continue
        df = s.example()
        total_out = total_over = 0
        notes = []
        for name, k, font in VARIANTS:
            w, h = s.size
            try:
                fig = render(s, df, None, {"font": font}, size=(w * k, h * k), dpi=100)
                outs, overs = check_figure(fig)
            except Exception as e:
                notes.append(f"{name}: ERROR {e}")
                continue
            total_out += len(outs)
            total_over += len(overs)
            if outs or overs:
                ex = [f"out '{t.get_text()[:18]}'" for t, _ in outs[:3]] + \
                     [f"'{a.get_text()[:14]}'×'{b.get_text()[:14]}'" for a, b, *_ in overs[:3]]
                notes.append(f"{name}: {len(outs)} out, {len(overs)} overlap  " + "; ".join(ex))
                if out_png is not None:
                    from matplotlib.patches import Rectangle
                    out_png.mkdir(parents=True, exist_ok=True)
                    for t, bb in outs:
                        fig.patches.append(Rectangle((bb.x0, bb.y0), bb.width, bb.height, fill=False,
                                                     ec="red", lw=1.5, transform=None))
                    for a, b, b1, b2 in overs:
                        for bb in (b1, b2):
                            fig.patches.append(Rectangle((bb.x0, bb.y0), bb.width, bb.height, fill=False,
                                                         ec="red", lw=1, transform=None))
                    fig.savefig(out_png / f"{s.id}__{name.replace(' ', '_')}.png", dpi=100)
        rows.append((total_out + total_over, s.id, total_out, total_over, notes))
    rows.sort(reverse=True)
    bad = [r for r in rows if r[0]]
    for score, sid, o, v, notes in bad:
        print(f"{sid:32s} out={o:3d} overlap={v:3d}")
        if verbose:
            for n in notes:
                print("     ", n)
    print(f"\n{len(bad)} of {len(rows)} plot types have misplaced text "
          f"({sum(r[2] for r in rows)} outside the figure, {sum(r[3] for r in rows)} overlaps) "
          f"over {len(VARIANTS)} sizes each")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
