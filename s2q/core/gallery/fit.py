"""Automatic text fitting for the gallery plots (run after every drawing).

Whatever the figure size and font size, text should neither collide nor leave the figure:

1. **Tick labels** – when neighbouring labels collide, x labels are turned (35° → 90°, long ones
   wrapped onto two lines) and only then thinned (every 2nd / 3rd …); y labels first get a
   smaller font (down to 70 %) and only then are thinned.
2. **Labels inside the plot** – annotations (labels with a pointer, e.g. gene names) are pushed
   apart until they no longer overlap; plain texts that still overlap get a smaller font.
3. **Text leaving the figure** – moved back inside.

Everything works in screen units on the drawn figure, so the result follows the actual size,
font and dpi.
"""
from __future__ import annotations

import math
import textwrap

import numpy as np


def _canvas(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if not isinstance(fig.canvas, FigureCanvasAgg):
        FigureCanvasAgg(fig)
    return fig.canvas


def _draw(fig):
    c = _canvas(fig)
    c.draw()
    return c.get_renderer()


def _is_plain_axes(ax) -> bool:
    return ax.name == "rectilinear" and ax.get_visible() and ax.axison


def _drawn_ticks(axis):
    """Ticks whose label is drawn (inside the axis range, visible, non-empty)."""
    try:
        ticks = axis._update_ticks()
    except Exception:
        return []
    lo, hi = sorted(axis.get_view_interval())
    eps = (hi - lo) * 1e-6 if hi > lo else 0
    out = []
    for t in ticks:
        loc = t.get_loc()
        if loc is None or not (lo - eps <= loc <= hi + eps) or not t.get_visible():
            continue
        lab = t.label1 if t.label1.get_visible() else (t.label2 if t.label2.get_visible() else None)
        if lab is not None and lab.get_text().strip():
            out.append((loc, lab))
    return out


def _wrap(text: str, width: int) -> str:
    if "\n" in text or len(text) <= width or " " not in text:
        return text
    return "\n".join(textwrap.wrap(text, width=max(8, width), break_long_words=False))


# ── 1. tick labels ───────────────────────────────────────────────────────────

def _fix_x_ticks(ax, r) -> bool:
    ticks = _drawn_ticks(ax.xaxis)
    if len(ticks) < 2:
        return False
    labs = [l for _, l in ticks]
    y_ref = ax.get_ylim()[0]                 # any valid y (0 would fail on a log axis)
    pos = sorted(ax.transData.transform((loc, y_ref))[0] for loc, _ in ticks)
    gaps = np.diff(np.asarray(pos))
    gaps = gaps[gaps > 0.5]
    if not len(gaps):
        return False
    dx = float(gaps.min())
    rot0 = labs[0].get_rotation() % 180
    # unrotated size of the widest / tallest label
    saved = [(l, l.get_rotation()) for l in labs]
    for l in labs:
        l.set_rotation(0)
    widths = [l.get_window_extent(r).width for l in labs]
    h = max(l.get_window_extent(r).height for l in labs)
    for l, rot in saved:
        l.set_rotation(rot)
    w = max(widths)

    def fits(theta):
        if theta == 0:
            return w + 4 <= dx
        return dx * math.sin(math.radians(theta)) >= h * 1.12

    if fits(rot0 if rot0 <= 90 else 180 - rot0):
        return False
    # long labels: wrap onto two lines before turning them (narrower and still readable)
    if rot0 == 0 and w > dx and any(" " in l.get_text() and len(l.get_text()) > 14 for l in labs):
        longest = max(len(l.get_text()) for l in labs)
        ordered = sorted(ticks, key=lambda t: t[0])
        wrapped = [_wrap(l.get_text(), math.ceil(longest / 2) + 1) for _, l in ordered]
        ax.set_xticks([loc for loc, _ in ordered], labels=wrapped)
        r2 = _draw(ax.figure)
        labs = [l for _, l in _drawn_ticks(ax.xaxis)]
        w = max(l.get_window_extent(r2).width for l in labs)
        h = max(l.get_window_extent(r2).height for l in labs)
        if fits(0):
            return True
    texts = [l.get_text() for _, l in sorted(ticks, key=lambda t: t[0])]
    locs = [loc for loc, _ in sorted(ticks, key=lambda t: t[0])]

    def apply(theta, keep_every=1):
        # fixed ticks + labels: matplotlib would otherwise rewrite the label texts on the next draw
        shown = [t if i % keep_every == 0 else "" for i, t in enumerate(texts)]
        ax.set_xticks(locs, labels=shown)
        ax.tick_params(axis="x", labelrotation=theta)
        for l in ax.get_xticklabels():
            l.set_horizontalalignment("right" if 0 < theta < 90 else "center")
            l.set_rotation_mode("anchor" if 0 < theta < 90 else "default")
            if 0 < theta < 90:
                l.set_verticalalignment("center" if theta >= 60 else "top")

    for theta in (35, 45, 60, 90):
        if theta > (rot0 if rot0 <= 90 else 0) and fits(theta):
            apply(theta)
            return True
    # even upright labels do not fit: upright, and only every k-th label
    apply(90, keep_every=max(2, math.ceil(h * 1.12 / dx)))
    return True


def _fix_y_ticks(ax, r) -> bool:
    ticks = _drawn_ticks(ax.yaxis)
    if len(ticks) < 2:
        return False
    labs = [l for _, l in ticks]
    x_ref = ax.get_xlim()[0]
    pos = sorted(ax.transData.transform((x_ref, loc))[1] for loc, _ in ticks)
    gaps = np.diff(np.asarray(pos))
    gaps = gaps[gaps > 0.5]
    if not len(gaps):
        return False
    dy = float(gaps.min())
    if labs[0].get_rotation() % 180 != 0:
        return False
    h = max(l.get_window_extent(r).height for l in labs)
    if h * 1.02 <= dy:
        return False
    size0 = labs[0].get_fontsize()
    scale = max(0.7, dy / (h * 1.05))
    ax.tick_params(axis="y", labelsize=size0 * scale)
    if h * scale * 1.02 <= dy:
        return True
    k = max(2, math.ceil(h * scale * 1.05 / dy))
    ordered = sorted(ticks, key=lambda t: t[0])
    ax.set_yticks([loc for loc, _ in ordered],
                  labels=[l.get_text() if i % k == 0 else "" for i, (_, l) in enumerate(ordered)])
    return True


# ── 2. labels inside the plot ────────────────────────────────────────────────

def _shown(t, r) -> bool:
    """Is this text drawn? (an annotation whose point lies outside its plot area is not)."""
    from matplotlib.text import Annotation
    if not t.get_visible() or not t.get_text().strip():
        return False
    if isinstance(t, Annotation):
        try:
            return bool(t._check_xy(r))
        except Exception:
            return True
    return True


def _text_box(t, r):
    """Box of the text itself (an annotation's own size would include its pointer line)."""
    from matplotlib.text import Text
    return Text.get_window_extent(t, r)


def _overlap(a, b, pad=1.0) -> bool:
    return (a.x0 < b.x1 + pad and b.x0 < a.x1 + pad and a.y0 < b.y1 + pad and b.y0 < a.y1 + pad)


def _move(t, dx_px: float, dy_px: float):
    """Move a text (or an annotation's text) by a number of pixels on screen."""
    from matplotlib.text import Annotation
    fig = t.figure
    if isinstance(t, Annotation):
        coords = t.anncoords if isinstance(t.anncoords, str) else None
        x, y = t.xyann
        if coords in ("offset points", "offset pixels"):
            f = 72.0 / fig.dpi if coords == "offset points" else 1.0
            t.xyann = (x + dx_px * f, y + dy_px * f)
            return
        tr = t._get_xy_transform(t.figure.canvas.get_renderer(), t.anncoords)
        p = tr.transform((x, y)) + (dx_px, dy_px)
        t.xyann = tuple(tr.inverted().transform(p))
        return
    tr = t.get_transform()
    p = tr.transform(t.get_position()) + (dx_px, dy_px)
    t.set_position(tuple(tr.inverted().transform(p)))


def _repel(ax, r, iterations: int = 80) -> bool:
    from matplotlib.text import Annotation
    texts = [t for t in ax.texts if _shown(t, r)]
    if len(texts) < 2:
        return False
    movable = [t for t in texts if isinstance(t, Annotation)]
    box = ax.get_window_extent(r)
    changed = False
    if movable:
        # spring layout: every step, each pointer label is pushed away from what it overlaps
        # (mostly up / down), then held inside the plot area; repeat until nothing overlaps
        mov_ids = {id(t) for t in movable}
        # the legend inside this plot is an obstacle too (labels must not hide under it)
        lg = ax.get_legend()
        fixed = [lg.get_window_extent(r)] if lg is not None and lg.get_visible() else []
        for _ in range(iterations):
            bbs = {id(t): _text_box(t, r) for t in texts}
            push = {id(t): [0.0, 0.0] for t in movable}
            alt = {id(t): [0.0, 0.0] for t in movable}      # sideways push, used if blocked
            any_overlap = False
            for t in movable:
                b = bbs[id(t)]
                others = [(id(o), bbs[id(o)]) for o in texts if o is not t] + [(-k - 1, fb) for k, fb in
                                                                                 enumerate(fixed)]
                for oid, ob in others:
                    if not _overlap(b, ob, pad=1.5):
                        continue
                    any_overlap = True
                    oy = min(b.y1, ob.y1) - max(b.y0, ob.y0) + 2
                    ox = min(b.x1, ob.x1) - max(b.x0, ob.x0) + 2
                    share = 0.5 if oid in mov_ids else 1.0
                    sy = 1 if (b.y0 + b.y1) > (ob.y0 + ob.y1) else -1
                    if (b.y0 + b.y1) == (ob.y0 + ob.y1):
                        sy = 1 if id(t) > oid else -1
                    sx = 1 if (b.x0 + b.x1) >= (ob.x0 + ob.x1) else -1
                    if oy <= ox * 2:
                        push[id(t)][1] += sy * oy * share
                        alt[id(t)][0] += sx * ox * share
                    else:
                        push[id(t)][0] += sx * ox * share
                        alt[id(t)][1] += sy * oy * share
            if not any_overlap:
                break
            for t in movable:
                dx, dy = push[id(t)]
                b = bbs[id(t)]
                # pushed against the edge of the plot: go the other way, or sideways
                if (dy > 0 and b.y1 + 1 >= box.y1) or (dy < 0 and b.y0 - 1 <= box.y0):
                    dy = -dy if (box.y0 <= b.y0 - dy and b.y1 - dy <= box.y1) else 0
                    if dy == 0 and not dx:
                        dx = alt[id(t)][0]
                        if not dx:                      # no hint: towards the side with more room
                            room_left, room_right = b.x0 - box.x0, box.x1 - b.x1
                            dx = (1 if room_right >= room_left else -1) * abs(push[id(t)][1])
                if (dx > 0 and b.x1 + 1 >= box.x1) or (dx < 0 and b.x0 - 1 <= box.x0):
                    dx = -dx if (box.x0 <= b.x0 - dx and b.x1 - dx <= box.x1) else 0
                    if dx == 0:
                        dy = dy or alt[id(t)][1]
                # hold inside the plot area
                if b.x0 + dx < box.x0:
                    dx = box.x0 - b.x0
                if b.x1 + dx > box.x1:
                    dx = box.x1 - b.x1
                if b.y0 + dy < box.y0:
                    dy = box.y0 - b.y0
                if b.y1 + dy > box.y1:
                    dy = box.y1 - b.y1
                if dx or dy:
                    _move(t, dx, dy)
                    changed = True
    # plain texts that still collide: smaller font (down to 70 %)
    for _ in range(6):
        bbs = [(t, _text_box(t, r)) for t in texts]
        hit = set()
        for i in range(len(bbs)):
            for j in range(i + 1, len(bbs)):
                if _overlap(bbs[i][1], bbs[j][1], pad=-1):
                    for t in (bbs[i][0], bbs[j][0]):
                        if not isinstance(t, Annotation):
                            hit.add(t)
        if not hit:
            break
        for t in hit:
            base = getattr(t, "_s2q_font0", None) or t.get_fontsize()
            t._s2q_font0 = base
            floor = 0.5 if len(t.get_text().strip()) <= 4 else 0.62    # counts may get smaller
            if t.get_fontsize() > base * floor:
                t.set_fontsize(t.get_fontsize() * 0.9)
                changed = True
    return changed


# ── 3. text leaving the figure ───────────────────────────────────────────────

def _keep_inside(fig, r) -> bool:
    from matplotlib.text import Text
    W, H = fig.bbox.width, fig.bbox.height
    changed = False
    for ax in fig.axes:
        for t in list(ax.texts):
            if not _shown(t, r):
                continue
            b = _text_box(t, r)
            dx = (2 - b.x0) if b.x0 < 0 else ((W - 2 - b.x1) if b.x1 > W else 0)
            dy = (2 - b.y0) if b.y0 < 0 else ((H - 2 - b.y1) if b.y1 > H else 0)
            if dx or dy:
                if b.width > W - 4:              # wider than the figure: smaller font instead
                    t.set_fontsize(t.get_fontsize() * (W - 4) / b.width)
                else:
                    _move(t, dx, dy)
                changed = True
        # axes titles (e.g. the two side titles of a butterfly chart) wider than the figure allows
        for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            if t is None or not t.get_visible() or not t.get_text().strip():
                continue
            b = t.get_window_extent(r)
            if b.x0 < 0 or b.x1 > W:
                # shift it back inside; if it is wider than the figure, shrink it instead
                if b.width <= W - 4:
                    _move(t, (2 - b.x0) if b.x0 < 0 else (W - 2 - b.x1), 0)
                else:
                    t.set_fontsize(t.get_fontsize() * max(0.6, (W - 4) / b.width))
                changed = True
    for t in fig.texts:
        if isinstance(t, Text) and t.get_visible() and t.get_text().strip():
            b = t.get_window_extent(r)
            if b.width > W - 4:
                t.set_fontsize(t.get_fontsize() * (W - 4) / b.width)
                changed = True
    # 3-D plots place their axis labels themselves: shrink one that sticks out
    for ax in fig.axes:
        if ax.name != "3d" or not ax.get_visible():
            continue
        for lab in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
            if not lab.get_visible() or not lab.get_text().strip():
                continue
            for _ in range(5):
                b = lab.get_window_extent(r)
                if b.x0 >= -1 and b.y0 >= -1 and b.x1 <= W + 1 and b.y1 <= H + 1:
                    break
                lab.set_fontsize(lab.get_fontsize() * 0.88)
                changed = True
    return changed


def _make_room(ax, r) -> bool:
    """Extend the axis range so value labels tied to data points (bar labels…) stay inside the plot."""
    from matplotlib.text import Annotation
    if ax.name != "rectilinear" or not ax.get_visible():
        return False
    box = ax.get_window_extent(r)
    if box.width < 10 or box.height < 10:
        return False
    need = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
    for t in ax.texts:
        if not _shown(t, r):
            continue
        in_data = (t.xycoords == "data") if isinstance(t, Annotation) else (t.get_transform() == ax.transData)
        if not in_data:
            continue
        b = _text_box(t, r)
        if b.width > box.width * 0.9:       # a label wider than the plot: not a value label
            continue
        need["top"] = max(need["top"], b.y1 - box.y1)
        need["bottom"] = max(need["bottom"], box.y0 - b.y0)
        need["right"] = max(need["right"], b.x1 - box.x1)
        need["left"] = max(need["left"], box.x0 - b.x0)
    changed = False
    for side, px in need.items():
        if px <= 0.5 or px > 0.6 * (box.height if side in ("top", "bottom") else box.width):
            continue
        vertical = side in ("top", "bottom")
        if (ax.get_yscale() if vertical else ax.get_xscale()) != "linear":
            continue
        lo, hi = ax.get_ylim() if vertical else ax.get_xlim()
        span_px = box.height if vertical else box.width
        extra = (hi - lo) * (px + 3) / max(span_px - px - 3, 1)
        inverted = hi < lo
        if side in ("top", "right"):
            hi = hi - extra if inverted else hi + extra
        else:
            lo = lo + extra if inverted else lo - extra
        (ax.set_ylim if vertical else ax.set_xlim)(lo, hi)
        changed = True
    return changed


def _wrap_text(t, lines: int = 2) -> bool:
    """Put a text on *lines* lines (split at spaces near the middle); False if it cannot be split."""
    s = t.get_text()
    if "\n" in s or " " not in s.strip():
        return False
    t.set_text(_wrap(s, math.ceil(len(s) / lines) + 2))
    return True


def _shrink_to(t, r, limit_px: float, along: str = "width", floor: float = 0.6):
    """Smaller font until the text's width / height fits *limit_px* (not below *floor* × original)."""
    b = _text_box(t, r)
    size = b.width if along == "width" else b.height
    if size <= limit_px or size <= 0:
        return
    base = getattr(t, "_s2q_font0", None) or t.get_fontsize()
    t._s2q_font0 = base
    t.set_fontsize(max(base * floor, t.get_fontsize() * limit_px / size * 0.98))


def _fit_labels_titles(fig, r) -> bool:
    """Axis titles longer than their axis, and panel titles running into each other."""
    changed = False
    for ax in fig.axes:
        if not ax.get_visible() or ax.name not in ("rectilinear", "polar"):
            continue
        box = ax.get_window_extent(r)
        xl, yl = ax.xaxis.label, ax.yaxis.label
        if ax.axison and xl.get_visible() and xl.get_text().strip():
            if _text_box(xl, r).width > box.width * 1.05:
                if _wrap_text(xl):
                    changed = True
                if _text_box(xl, r).width > box.width * 1.05:
                    _shrink_to(xl, r, box.width * 1.05)
                    changed = True
        if ax.axison and yl.get_visible() and yl.get_text().strip():
            if _text_box(yl, r).height > box.height * 1.05:   # turned 90°: its height is its length
                if _wrap_text(yl):
                    changed = True
                if _text_box(yl, r).height > box.height * 1.05:
                    _shrink_to(yl, r, box.height * 1.05, along="height")
                    changed = True
    # panel titles (centre, left or right aligned): wider than their panel, or into a neighbour's
    titles = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            if t is not None and t.get_visible() and t.get_text().strip():
                titles.append((ax, t))
    for ax, t in titles:
        box = ax.get_window_extent(r)
        others = [o for a, o in titles if o is not t]
        hits = [o for o in others if _overlap(_text_box(t, r), _text_box(o, r), pad=2)]
        if hits or _text_box(t, r).width > box.width * 1.25:
            if _wrap_text(t):
                changed = True
            if any(_overlap(_text_box(t, r), _text_box(o, r), pad=2) for o in hits) \
                    or _text_box(t, r).width > box.width * 1.25:
                _shrink_to(t, r, box.width * 1.02)
                changed = True
    return changed


def _legends(fig):
    out = list(fig.legends)
    for ax in fig.axes:
        if ax.get_legend() is not None:
            out.append(ax.get_legend())
    return [lg for lg in out if lg.get_visible()]


def _fit_legend_width(fig, r) -> bool:
    """A legend wider (or taller) than the figure: fewer columns, then a smaller font."""
    W, H = fig.bbox.width, fig.bbox.height
    changed = False
    for lg in _legends(fig):
        for _ in range(12):
            b = lg.get_window_extent(r)
            if b.width <= W - 4 and b.height <= H - 4:
                break
            ncols = getattr(lg, "_ncols", 1)
            if b.width > W - 4 and ncols > 1:
                lg.set_ncols(ncols - 1)
            else:
                for t in lg.get_texts():
                    t.set_fontsize(t.get_fontsize() * 0.9)
                if lg.get_title() is not None:
                    lg.get_title().set_fontsize(lg.get_title().get_fontsize() * 0.9)
            changed = True
            r = _draw(fig)
    return changed


def _move_legend_off_text(ax, r) -> bool:
    """An axes legend lying on labels (texts drawn in the plot): try the other corners."""
    lg = ax.get_legend()
    if lg is None or not lg.get_visible() or getattr(lg, "_bbox_to_anchor", None) is not None:
        return False
    texts = [_text_box(t, r) for t in ax.texts if _shown(t, r)]
    if not texts:
        return False

    def hits():
        b = lg.get_window_extent(r)
        return sum(1 for tb in texts if _overlap(b, tb, pad=1))

    if not hits():
        return False
    best, best_n = lg._loc, hits()
    for loc in (1, 2, 3, 4, 7, 6, 9, 8, 5, 10):            # corners first, then edges, centre
        lg._loc = loc
        _draw(ax.figure)
        n = hits()
        if n < best_n:
            best, best_n = loc, n
        if n == 0:
            return True
    lg._loc = best
    return True


def _fit_in_bars(ax, r) -> bool:
    """Labels written inside bars (e.g. % in stacked bars): as large as the bar allows, else hidden."""
    from matplotlib.patches import Rectangle
    bars = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_visible()
            and abs(p.get_width()) > 0 and abs(p.get_height()) > 0]
    if not bars or len(bars) > 3000:
        return False
    boxes = [p.get_window_extent(r) for p in bars]
    changed = False
    for t in ax.texts:
        if not _shown(t, r):
            continue
        tb = _text_box(t, r)
        cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
        # the bar holding the label: narrow (vertical bars) or thin (horizontal bars) next to it
        home = next((b for b in boxes if b.x0 <= cx <= b.x1 and b.y0 <= cy <= b.y1
                     and (b.width < tb.width * 3 or b.height < tb.height * 3)), None)
        if home is None:
            continue
        scale = min((home.width - 2) / tb.width if tb.width else 1, (home.height - 1) / tb.height if tb.height else 1)
        if scale >= 1:
            continue
        if scale < 0.55:
            t.set_visible(False)          # far too small a bar to carry a readable label
        else:
            t.set_fontsize(t.get_fontsize() * scale)
        changed = True
    return changed


def _is_colorbar(ax) -> bool:
    return hasattr(ax, "_colorbar") or ax.get_label() == "<colorbar>"


def _decorations_outside(fig, r) -> int:
    """How many tick labels / axis labels / titles stick out of the figure."""
    W, H = fig.bbox.width, fig.bbox.height
    n = 0
    for ax in fig.axes:
        if not ax.get_visible() or ax.name != "rectilinear" or not ax.axison:
            continue
        texts = [l for _, l in _drawn_ticks(ax.xaxis)] + [l for _, l in _drawn_ticks(ax.yaxis)]
        texts += [ax.xaxis.label, ax.yaxis.label, ax.title]
        for t in texts:
            if not t.get_visible() or not t.get_text().strip():
                continue
            b = t.get_window_extent(r)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
                n += 1
    return n


def _settle_layout(fig):
    """Pick a layout that stays put when the figure is drawn again, and draw until it settles.

    The constrained layout keeps shifting plots with fixed-aspect axes (heatmaps, pies, Venn…)
    a little on every drawing, until labels fall off the figure; the "compressed" variant made
    for such plots does not.
    """
    eng = fig.get_layout_engine()
    if eng is None or type(eng).__name__ != "ConstrainedLayoutEngine":
        return _draw(fig)
    fixed = [ax for ax in fig.axes if ax.get_visible() and not _is_colorbar(ax) and ax.name != "3d"
             and ax.get_aspect() != "auto"]
    if fixed and not getattr(eng, "_compress", False):
        fig.set_layout_engine("compressed")
    r = _draw(fig)
    r = _draw(fig)                            # second pass: the layout converges
    if _decorations_outside(fig, r):
        # safety net: try the other layout and keep whichever leaves less text outside
        before = _decorations_outside(fig, r)
        other = "constrained" if getattr(fig.get_layout_engine(), "_compress", False) else "compressed"
        current = "compressed" if other == "constrained" else "constrained"
        fig.set_layout_engine(other)
        r = _draw(fig)
        r = _draw(fig)
        if _decorations_outside(fig, r) >= before:
            fig.set_layout_engine(current)
            r = _draw(fig)
            r = _draw(fig)
    return r


def fit_text(fig):
    """Make the text of a drawn gallery figure fit (see the module docstring)."""
    try:
        r = _settle_layout(fig)
        # tick labels, axis / panel titles and legends all change the layout, which can crowd
        # another axis again: repeat (at most 3 rounds) until nothing changes
        for _ in range(3):
            changed = False
            for ax in fig.axes:
                if _is_plain_axes(ax):
                    changed |= _fix_x_ticks(ax, r)
                    changed |= _fix_y_ticks(ax, r)
            if changed:
                r = _draw(fig)
                r = _draw(fig)
            more = _fit_labels_titles(fig, r) | _fit_legend_width(fig, r)
            if more:
                r = _draw(fig)
                r = _draw(fig)
            if not (changed or more):
                break
        for ax in fig.axes:
            if ax.get_visible():
                _fit_in_bars(ax, r)
                _repel(ax, r)
        if any([_make_room(ax, r) for ax in fig.axes]):
            r = _draw(fig)
            r = _draw(fig)
        if any([_move_legend_off_text(ax, r) for ax in fig.axes if ax.get_visible()]):
            r = _draw(fig)
        _keep_inside(fig, r)
    except Exception:                        # fitting must never stop a plot from showing
        pass
    return fig
