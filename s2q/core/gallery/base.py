"""Plot gallery framework: every plot type is a registered ``PlotSpec``.

A plot declares
  * ``roles``   – the table columns it needs (numeric / categorical / any, one or several),
  * ``opts``    – its own options (numbers, switches, choices, text),
  * ``example`` – a function returning a small example table (so every plot can be tried at once
                  and the user sees what layout of data it expects),
  * ``draw``    – ``draw(fig, df, cols, opts, st)`` drawing onto a matplotlib Figure.

``cols`` maps role key → column name (or list of names for ``multi`` roles, "" when an optional role
is left empty). ``opts`` holds the plot's own options *and* the common ones (title, labels…).
``st`` is a ``Style`` with the palette and font size, and helpers that give every plot the same
clean, publication look (white background, light grid, readable fonts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ── palettes ──────────────────────────────────────────────────────────────────

PALETTES = {
    "S2Q": ["#7c6aef", "#e5609a", "#3b9ee8", "#f0a030", "#2bb673", "#e8553b", "#8d6e63", "#00acc1",
            "#c0ca33", "#5c6bc0"],
    "Nature (NPG)": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2",
                     "#DC0000", "#7E6148", "#B09C85"],
    "Science (AAAS)": ["#3B4992", "#EE0000", "#008B45", "#631879", "#008280", "#BB0021", "#5F559B",
                       "#A20056", "#808180", "#1B1919"],
    "Lancet": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6"],
    "NEJM": ["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD", "#FFDC91", "#EE4C97"],
    "JCO": ["#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67", "#8F7700", "#3B3B3B"],
    "Colour-blind safe (Okabe-Ito)": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
                                      "#CC79A7", "#000000"],
    "Tableau": ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
                "#9C755F", "#BAB0AC"],
    "Pastel": ["#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5",
               "#D9D9D9", "#BC80BD"],
    "Grey": ["#252525", "#636363", "#969696", "#bdbdbd", "#d9d9d9"],
}
COLORMAPS = ["RdBu_r", "viridis", "magma", "coolwarm", "RdYlBu_r", "YlOrRd", "Blues", "Reds", "Greens",
             "PuOr_r", "Spectral_r", "cividis", "plasma", "bwr"]

CATEGORIES = [
    "Bar charts",
    "Line & area",
    "Pie, donut & proportions",
    "Distribution (box, violin…)",
    "Scatter & correlation",
    "Heatmap, cluster & PCA",
    "Clinical & statistics",
    "Sets & flows (Venn, Sankey, chord)",
    "Transcriptome & enrichment",
    "Genome & sequence",
]


# ── declarations ─────────────────────────────────────────────────────────────

@dataclass
class Role:
    """A column the plot needs. kind: 'num' (numeric), 'cat' (labels / groups) or 'any'."""
    key: str
    label: str
    kind: str = "any"
    multi: bool = False         # several columns (e.g. the samples of a heatmap)
    optional: bool = False
    hint: str = ""


@dataclass
class Opt:
    """A plot option. type: 'int', 'float', 'bool', 'choice', 'str', 'cmap', 'color'."""
    key: str
    label: str
    type: str
    default: object
    choices: list | None = None
    lo: float = 0
    hi: float = 1e9
    step: float = 1


@dataclass
class PlotSpec:
    id: str
    name: str
    category: str
    draw: Callable
    example: Callable
    roles: list = field(default_factory=list)
    opts: list = field(default_factory=list)
    desc: str = ""
    number: str = ""            # number of the original plot type (e.g. "086")
    size: tuple = (7.0, 5.0)    # default figure size (inches)
    axis_labels: bool = True    # the common x / y label options make sense for this plot
    defaults: dict = field(default_factory=dict)   # other starting values for common options (e.g. cmap)


REGISTRY: dict[str, PlotSpec] = {}


def plot(id: str, name: str, category: str, *, roles=(), opts=(), example, desc="", number="",
         size=(7.0, 5.0), axis_labels=True, defaults=None):
    """Decorator registering a draw function as a gallery plot type."""
    assert category in CATEGORIES, category

    def deco(fn):
        if id in REGISTRY:
            raise ValueError(f"duplicate plot id {id}")
        REGISTRY[id] = PlotSpec(id, name, category, fn, example, list(roles), list(opts), desc, number,
                                tuple(size), axis_labels, dict(defaults or {}))
        return fn
    return deco


# ── common options (shown for every plot) ────────────────────────────────────

COMMON_OPTS = [
    Opt("title", "Title", "str", ""),
    Opt("xlabel", "X axis label", "str", ""),
    Opt("ylabel", "Y axis label", "str", ""),
    Opt("palette", "Colours", "choice", "Nature (NPG)", choices=list(PALETTES)),
    Opt("cmap", "Colour map", "cmap", "RdBu_r", choices=COLORMAPS),
    Opt("font", "Font size", "int", 11, lo=6, hi=30),
    Opt("legend", "Legend", "choice", "best", choices=["best", "outside right", "below", "off"]),
    Opt("grid", "Grid lines", "bool", True),
    Opt("fit", "Fit text automatically", "bool", True),
]


# ── style helpers used by the draw functions ─────────────────────────────────

class Style:
    """Palette, colour map and font size for one drawing, plus helpers for a consistent look."""

    def __init__(self, opts: dict):
        self.opts = opts
        self.palette = list(PALETTES.get(opts.get("palette"), PALETTES["Nature (NPG)"]))
        self.cmap = opts.get("cmap") or "RdBu_r"
        self.font = float(opts.get("font", 11))
        self.grid = bool(opts.get("grid", True))
        self.legend_mode = opts.get("legend", "best")

    def colors(self, n: int) -> list:
        """n distinct colours from the palette (repeating, then lighter shades)."""
        p = self.palette
        if n <= len(p):
            return p[:n]
        from matplotlib.colors import to_rgb
        out = []
        for i in range(n):
            r, g, b = to_rgb(p[i % len(p)])
            k = min(0.8, (i // len(p)) * 0.2)          # lighter shades for later rounds, never past white
            out.append((r + (1 - r) * k, g + (1 - g) * k, b + (1 - b) * k))
        return out

    def color(self, i: int = 0):
        return self.colors(i + 1)[i]

    def ax(self, ax, grid_axis: str = "both"):
        """Clean axes: no top/right spines, light dashed grid behind the data, font size."""
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#555555")
        ax.tick_params(labelsize=self.font * 0.85, colors="#333333")
        if self.grid and grid_axis:
            ax.grid(True, axis=grid_axis, color="#d9d9d9", linestyle="--", linewidth=0.6)
            ax.set_axisbelow(True)
        else:
            ax.grid(False)
        return ax

    def bare(self, ax):
        """No axes at all (pies, Venn diagrams, networks…)."""
        ax.set_axis_off()
        return ax

    def legend(self, ax, handles=None, labels=None, title=None, outside=None, **kw):
        """Legend of *ax* (an Axes), placed by the common 'Legend' option.

        *handles* / *labels* default to the labelled artists of *ax*. *outside* ("below" /
        "right") is where the plot prefers its legend in the default 'best' mode: under or beside
        the figure, with room made for it by the layout. Returns the legend or None.
        """
        if self.legend_mode == "off":
            return None
        if outside and self.legend_mode == "best":
            for k in ("loc", "bbox_to_anchor", "borderaxespad"):
                kw.pop(k, None)
            mode, self.legend_mode = self.legend_mode, ("below" if outside == "below" else "outside right")
            try:
                return self.legend(ax, handles, labels, title, **kw)
            finally:
                self.legend_mode = mode
        if handles is None:
            handles, labels = _handles([ax])
        elif labels is None:
            labels = [h.get_label() for h in handles]
        if not handles:
            return None
        kw.setdefault("fontsize", self.font * 0.8)
        kw.setdefault("frameon", False)
        if title:
            kw["title"] = title
            kw.setdefault("title_fontsize", self.font * 0.85)
        loc = kw.pop("loc", "best")
        if self.legend_mode == "outside right":
            kw.pop("ncol", None)
            kw.pop("bbox_to_anchor", None)
            return ax.figure.legend(handles, labels, loc="outside right center", **kw)
        if self.legend_mode == "below":
            kw.pop("bbox_to_anchor", None)
            if "ncol" not in kw:
                # as many columns as fit across the figure: ~0.6 × font size per character + the marker
                fig_w_pt = ax.figure.get_figwidth() * 72
                longest = max((len(str(l)) for l in labels), default=4)
                per = (longest * 0.6 + 4) * kw["fontsize"]
                kw["ncol"] = int(max(1, min(len(handles), fig_w_pt // max(per, 1))))
            return ax.figure.legend(handles, labels, loc="outside lower center", **kw)
        return ax.legend(handles, labels, loc=loc, **kw)

    def labels(self, ax, x: str = "", y: str = "", title: str = ""):
        """Axis labels / title: the user's text wins over the plot's default."""
        o = self.opts
        xl, yl = o.get("xlabel") or x, o.get("ylabel") or y
        if xl:
            ax.set_xlabel(xl, fontsize=self.font)
        if yl:
            ax.set_ylabel(yl, fontsize=self.font)
        t = o.get("title") or title
        if t:
            ax.set_title(t, fontsize=self.font * 1.15, fontweight="bold", pad=10)

    def suptitle(self, fig, default: str = ""):
        t = self.opts.get("title") or default
        if t:
            fig.suptitle(t, fontsize=self.font * 1.2, fontweight="bold")


def _handles(axes):
    hs, ls = [], []
    for a in axes:
        h, l = a.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in ls and not ll.startswith("_"):
                hs.append(hh)
                ls.append(ll)
    return hs, ls


# ── data helpers ─────────────────────────────────────────────────────────────

def num(df, col):
    """A column as float numbers (text that is not a number → NaN)."""
    import pandas as pd
    return pd.to_numeric(df[col], errors="coerce")


def cats(df, col) -> list:
    """Category labels in order of first appearance (missing → '(blank)')."""
    s = df[col].astype(object).where(df[col].notna(), "(blank)").astype(str)
    return list(dict.fromkeys(s))


def as_str(df, col):
    return df[col].astype(object).where(df[col].notna(), "(blank)").astype(str)


def p_stars(p: float) -> str:
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def rng(seed: int = 0):
    return np.random.default_rng(seed)


# ── column assignment ────────────────────────────────────────────────────────

def is_numeric(df, col) -> bool:
    import pandas as pd
    s = pd.to_numeric(df[col], errors="coerce")
    return s.notna().sum() >= max(1, 0.8 * df[col].notna().sum())


def auto_columns(spec: PlotSpec, df, fill_optional: bool = True) -> dict:
    """A first guess for every role: the next unused column of the right kind, left to right.

    With ``fill_optional=False`` (the user's own table) optional roles start empty, so a spare
    column is never used for something the user did not ask for.
    """
    used, out = set(), {}
    numeric = {c: is_numeric(df, c) for c in df.columns}
    for role in spec.roles:
        if role.optional and not fill_optional:
            out[role.key] = [] if role.multi else ""
            continue
        want = [c for c in df.columns if c not in used and
                (role.kind == "any" or (role.kind == "num") == numeric[c])]
        if not want and role.kind == "any":
            want = [c for c in df.columns if c not in used]
        if role.multi:
            # a multi role takes every remaining column of its kind (minus those later roles need)
            later = [r for r in spec.roles[spec.roles.index(role) + 1:] if not r.optional]
            pick = want[:max(1, len(want) - len(later))] if want else []
            out[role.key] = pick
            used.update(pick)
        elif want:
            out[role.key] = want[0]
            used.add(want[0])
        else:
            out[role.key] = ""
    return out


def default_opts(spec: PlotSpec) -> dict:
    out = {o.key: o.default for o in COMMON_OPTS + spec.opts}
    out.update(spec.defaults)
    return out


def render(spec: PlotSpec, df, cols: dict | None = None, opts: dict | None = None,
           size: tuple | None = None, dpi: int = 100):
    """Draw *spec* into a new white Figure (no pyplot; safe in worker threads)."""
    from matplotlib.figure import Figure
    o = default_opts(spec)
    o.update(opts or {})
    c = auto_columns(spec, df) if cols is None else cols
    w, h = size or spec.size
    fig = Figure(figsize=(w, h), dpi=dpi, facecolor="white", layout="constrained")
    import matplotlib as mpl
    with mpl.rc_context({"font.size": o.get("font", 11), "axes.titlesize": o.get("font", 11) * 1.15,
                         "font.family": "sans-serif", "axes.unicode_minus": False}):
        spec.draw(fig, df, c, o, Style(o))
        if o.get("fit", True):
            # turn / thin crowded tick labels, push colliding labels apart, keep text inside
            from s2q.core.gallery.fit import fit_text
            fit_text(fig)
    return fig
