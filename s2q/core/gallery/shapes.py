"""Lines, areas, pies, donuts, funnels, waffles, treemaps, radar, word cloud and other proportion plots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from s2q.core.gallery.base import Opt, Role, as_str, cats, num, p_stars, plot, rng  # noqa: F401

CL = "Line & area"
CP = "Pie, donut & proportions"


# ── small helpers ────────────────────────────────────────────────────────────

def _lst(v) -> list:
    if isinstance(v, (list, tuple)):
        return [x for x in v if x]
    return [v] if v else []


def _nodata(fig, st, msg="No usable (positive, numeric) values in the selected columns"):
    ax = st.bare(fig.add_subplot())
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=st.font, color="#777777",
            transform=ax.transAxes, wrap=True)
    st.labels(ax)
    return ax


def _fmt(v) -> str:
    v = float(v)
    if not np.isfinite(v):
        return ""
    if abs(v) < 1e15 and v == int(v):
        return f"{int(v):,}"
    if abs(v) >= 1e4:
        return f"{v:,.0f}"
    return f"{v:.4g}"


def _ppu(fig, xspan, yspan, wfrac=0.9, hfrac=0.84):
    """Approximate points per data unit for an equal-aspect axes filling the figure."""
    w, h = fig.get_size_inches()
    return 72.0 * min(w * wfrac / max(xspan, 1e-9), h * hfrac / max(yspan, 1e-9))


def _tw(text, fs) -> float:
    """Approximate width of *text* in points."""
    return max(len(s) for s in str(text).split("\n")) * fs * 0.56


def _th(text, fs) -> float:
    return (str(text).count("\n") + 1) * fs * 1.25


def _ink(col) -> str:
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(col)
    return "#222222" if 0.299 * r + 0.587 * g + 0.114 * b > 0.58 else "white"


def _shade(col, k):
    """Mix *col* with white (k > 0) or black (k < 0)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(col)
    if k >= 0:
        return (r + (1 - r) * k, g + (1 - g) * k, b + (1 - b) * k)
    k = -k
    return (r * (1 - k), g * (1 - k), b * (1 - k))


def _agg(df, lab, val, positive=True) -> pd.Series:
    """Sum of *val* per label (first-appearance order); only finite (and positive) values."""
    d = pd.DataFrame({"l": as_str(df, lab), "v": num(df, val)}).dropna()
    s = d.groupby("l", sort=False)["v"].sum()
    s = s[np.isfinite(s)]
    return s[s > 0] if positive else s


def _side_legend(ax, st, handles, labels, title=None):
    """Legend; for 'best' it goes just right of the (axis-less) plot so it never covers it."""
    kw = {}
    if st.legend_mode == "best":
        kw = dict(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return st.legend(ax, handles=list(handles), labels=list(labels), title=title, **kw)


def _frame(ax, xlim, ylim):
    """Equal-aspect view of at least *xlim* × *ylim*; the axes box keeps its size (extra room is
    added to the data limits), so constrained layout stays stable with legends beside the plot."""
    (x0, x1), (y0, y1) = xlim, ylim
    ax.ignore_existing_data_limits = True
    ax.update_datalim([(x0, y0), (x1, y1)])
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.autoscale(True)
    ax.set_aspect("equal", adjustable="datalim")


def _series(df, c, xk="x", yk="y") -> pd.DataFrame:
    """Wide table → index = sorted numeric x (duplicates averaged), one column per series."""
    d = pd.DataFrame({"__x": num(df, c[xk]).to_numpy()})
    for col in _lst(c[yk]):
        if col != c[xk]:
            d[col] = num(df, col).to_numpy()
    d = d.dropna(subset=["__x"]).groupby("__x", sort=True).mean()
    return d.loc[:, d.notna().any()]


# ── example tables ───────────────────────────────────────────────────────────

def _ex_time_series():
    r = rng(3)
    t = np.array([0, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48], float)
    il6 = 1 + 9 * (t / 6) * np.exp(1 - t / 6)
    tnf = 1 + 6 * (t / 2) * np.exp(1 - t / 2)
    cx = 1 + 7 * (1 - np.exp(-t / 10))
    f = lambda y: np.round(y + r.normal(0, 0.25, len(t)), 2)  # noqa: E731
    return pd.DataFrame({"Time_h": t, "IL6": f(il6), "TNF": f(tnf), "CXCL8": f(cx)})


def _ex_composition():
    r = rng(4)
    day = np.array([0, 3, 7, 10, 14, 21, 28], float)
    return pd.DataFrame({
        "Day": day,
        "T_cells": np.round(20 + 2.0 * day + r.normal(0, 2, len(day)), 1),
        "B_cells": np.round(15 + 0.6 * day + r.normal(0, 1.5, len(day)), 1),
        "Myeloid": np.round(40 - 0.8 * day + r.normal(0, 2, len(day)), 1),
        "NK_cells": np.round(8 + 0.3 * day + r.normal(0, 1, len(day)), 1),
    })


def _ex_calendar():
    r = rng(5)
    days = pd.date_range("2025-01-01", "2025-12-31")
    base = np.where(days.dayofweek < 5, 6.0, 1.0) * (1 + 0.4 * np.sin(np.arange(len(days)) / 58))
    v = r.poisson(base).astype(float)
    v[(days.month == 8) & (days.day < 20)] = 0      # summer break
    return pd.DataFrame({"Date": days.strftime("%Y-%m-%d"), "Samples_sequenced": v})


def _ex_radar():
    r = rng(6)
    axes = ["Proliferation", "Migration", "Invasion", "Apoptosis", "Stemness", "Hypoxia"]
    rows = []
    for name, mu in [("HeLa", 70), ("MCF7", 50), ("A549", 60), ("HEK293", 35)]:
        rows.append([name] + list(np.round(np.clip(r.normal(mu, 18, len(axes)), 5, 100), 1)))
    return pd.DataFrame(rows, columns=["Cell_line"] + axes)


def _ex_celltypes():
    return pd.DataFrame({"Cell_type": ["T cells", "Monocytes", "B cells", "Neutrophils", "NK cells",
                                       "Dendritic cells", "Platelets"],
                         "Cells": [420, 260, 180, 140, 95, 30, 12]})


def _ex_two_layer():
    rows = [("Immune", "T cell", 320), ("Immune", "B cell", 140), ("Immune", "Macrophage", 210),
            ("Immune", "NK cell", 60), ("Stromal", "Fibroblast", 230), ("Stromal", "Endothelial", 120),
            ("Epithelial", "Basal", 150), ("Epithelial", "Luminal", 190)]
    return pd.DataFrame(rows, columns=["Lineage", "Cell_type", "Cells"])


def _ex_pie_matrix():
    r = rng(7)
    rows = []
    for tissue in ["Liver", "Lung", "Kidney"]:
        for tp in ["Day 0", "Day 7", "Day 14", "Day 28"]:
            p = r.dirichlet([4, 2, 3, 1.5]) * r.integers(800, 1500)
            rows.append([tissue, tp] + list(np.round(p).astype(int)))
    return pd.DataFrame(rows, columns=["Tissue", "Timepoint", "T_cells", "B_cells", "Myeloid", "Stromal"])


def _ex_multi_pie():
    r = rng(8)
    ct = ["T cells", "B cells", "Monocytes", "NK cells", "Other"]
    d = {"Cell_type": ct}
    for p in ["Patient_1", "Patient_2", "Patient_3", "Patient_4"]:
        d[p] = np.round(r.dirichlet([5, 2, 3, 1.5, 1]) * 1000).astype(int)
    return pd.DataFrame(d)


def _ex_variants():
    return pd.DataFrame({"Variant_class": ["Missense", "Synonymous", "Frameshift", "Nonsense", "Splice site"],
                         "Count": [512, 268, 104, 71, 45]})


def _ex_funnel():
    return pd.DataFrame({"Step": ["Raw reads", "Quality filtered", "Adapter trimmed", "Aligned",
                                  "Uniquely mapped", "Deduplicated"],
                         "Reads_M": [48.2, 45.9, 44.7, 41.3, 36.8, 29.5]})


def _ex_circle_area():
    return pd.DataFrame({"Cell_type": ["Hepatocytes", "Kupffer cells", "Endothelial", "Stellate", "Cholangiocytes"],
                         "Cells": [5200, 1800, 1300, 600, 250]})


def _ex_nested():
    return pd.DataFrame({"Gene_set": ["Annotated genes", "Expressed", "Differentially expressed",
                                      "Up-regulated", "qPCR validated"],
                         "Genes": [20000, 14200, 2600, 1350, 120]})


def _ex_quadrants():
    return pd.DataFrame({"Compartment": ["Nucleus", "Cytoplasm", "Membrane", "Mitochondria"],
                         "Proteins": [3400, 2800, 1900, 850]})


def _ex_pathways():
    rows = [("Cell cycle", 124, "Cellular processes"), ("Apoptosis", 136, "Cellular processes"),
            ("Autophagy", 98, "Cellular processes"), ("p53 signaling", 73, "Cellular processes"),
            ("MAPK signaling", 294, "Signal transduction"), ("PI3K-Akt signaling", 354, "Signal transduction"),
            ("Wnt signaling", 166, "Signal transduction"), ("TNF signaling", 112, "Signal transduction"),
            ("Glycolysis", 67, "Metabolism"), ("TCA cycle", 30, "Metabolism"),
            ("Fatty acid degradation", 43, "Metabolism"), ("Oxidative phosphorylation", 133, "Metabolism"),
            ("T cell receptor", 104, "Immune system"), ("B cell receptor", 81, "Immune system"),
            ("Toll-like receptor", 104, "Immune system")]
    return pd.DataFrame(rows, columns=["Pathway", "Genes", "Category"])


def _ex_taxa():
    rows = [("Bacteroides", 820, "Bacteroidetes"), ("Prevotella", 460, "Bacteroidetes"),
            ("Alistipes", 210, "Bacteroidetes"), ("Parabacteroides", 120, "Bacteroidetes"),
            ("Faecalibacterium", 540, "Firmicutes"), ("Roseburia", 260, "Firmicutes"),
            ("Ruminococcus", 300, "Firmicutes"), ("Blautia", 180, "Firmicutes"),
            ("Lactobacillus", 70, "Firmicutes"), ("Clostridium", 150, "Firmicutes"),
            ("Bifidobacterium", 330, "Actinobacteria"), ("Collinsella", 90, "Actinobacteria"),
            ("Escherichia", 140, "Proteobacteria"), ("Sutterella", 60, "Proteobacteria"),
            ("Akkermansia", 190, "Verrucomicrobia")]
    return pd.DataFrame(rows, columns=["Genus", "Abundance", "Phylum"])


def _ex_words():
    words = ["Apoptosis", "Inflammation", "TP53", "Cell cycle", "Immune response", "Hypoxia", "MAPK",
             "Autophagy", "NF-kB", "Angiogenesis", "Metabolism", "EMT", "DNA repair", "Wnt", "mTOR",
             "Interferon", "Cytokine", "Proliferation", "Senescence", "Glycolysis", "Notch", "Hedgehog",
             "Oxidative stress", "Chemokine", "TGF-beta", "JAK-STAT", "Ferroptosis", "Ubiquitin",
             "Translation", "Splicing", "Chromatin", "Methylation", "Migration", "Adhesion", "Lipid",
             "Insulin", "Hippo", "PI3K-Akt", "T cell", "Macrophage"]
    r = rng(9)
    freq = np.round(200 * np.exp(-np.arange(len(words)) / 9) + r.uniform(3, 10, len(words))).astype(int)
    return pd.DataFrame({"Term": words, "Frequency": freq})


def _ex_flower():
    r = rng(10)
    s = [f"Strain_{ch}" for ch in "ABCDEFGH"]
    return pd.DataFrame({"Sample": s, "Unique_genes": r.integers(80, 600, len(s)),
                         "Core_genes": [2134] * len(s)})


# ── lines & areas ────────────────────────────────────────────────────────────

_WIDE_ROLES = [Role("x", "X (numeric, e.g. time)", "num"), Role("y", "Series (one column each)", "num", multi=True)]
_WIDE_DESC = ("Wide table: a numeric X column (e.g. time in hours), then one numeric column per series "
              "(e.g. one per gene); repeated X values are averaged.")


def _ylab(d):
    return str(d.columns[0]) if len(d.columns) == 1 else "Value"


@plot("smooth_line_001", "Smooth line", CL, number="001", example=_ex_time_series, roles=_WIDE_ROLES,
      opts=[Opt("degree", "Spline degree (1 = straight)", "int", 3, lo=1, hi=5),
            Opt("smooth", "Smoothness (points per curve)", "int", 300, lo=20, hi=3000, step=10),
            Opt("points", "Show the data points", "bool", True),
            Opt("lw", "Line width", "float", 2.0, lo=0.3, hi=8, step=0.1)],
      desc=_WIDE_DESC)
def smooth_line(fig, df, c, o, st):
    from scipy.interpolate import make_interp_spline
    d = _series(df, c)
    if d.empty or not len(d.columns):
        return _nodata(fig, st)
    ax = st.ax(fig.add_subplot())
    cols = st.colors(len(d.columns))
    for i, name in enumerate(d.columns):
        s = d[name].dropna()
        x, y = s.index.to_numpy(float), s.to_numpy(float)
        if not len(x):
            continue
        k = min(int(o["degree"]), len(x) - 1)
        if k >= 1:
            xs = np.linspace(x.min(), x.max(), max(int(o["smooth"]), 2))
            try:
                ys = make_interp_spline(x, y, k=k)(xs)
            except Exception:
                xs, ys = x, y
        else:
            xs, ys = x, y
        ax.plot(xs, ys, color=cols[i], lw=o["lw"], label=str(name), solid_capstyle="round")
        if o["points"] or len(x) == 1:
            ax.scatter(x, y, s=26, color=cols[i], edgecolor="white", linewidth=0.8, zorder=3)
    st.labels(ax, c["x"], _ylab(d))
    st.legend(ax)


@plot("line_marker_055", "Line with markers", CL, number="055", example=_ex_time_series, roles=_WIDE_ROLES,
      opts=[Opt("marker", "Marker", "choice", "mixed", choices=["mixed", "o", "s", "^", "D", "v", "none"]),
            Opt("ms", "Marker size", "float", 6.5, lo=1, hi=20, step=0.5),
            Opt("ls", "Line style", "choice", "solid", choices=["solid", "dashed", "dotted", "dashdot"]),
            Opt("lw", "Line width", "float", 1.6, lo=0.3, hi=8, step=0.1)],
      desc=_WIDE_DESC)
def line_marker(fig, df, c, o, st):
    d = _series(df, c)
    if d.empty or not len(d.columns):
        return _nodata(fig, st)
    ax = st.ax(fig.add_subplot())
    cols = st.colors(len(d.columns))
    mk = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    for i, name in enumerate(d.columns):
        s = d[name].dropna()
        m = mk[i % len(mk)] if o["marker"] == "mixed" else (None if o["marker"] == "none" else o["marker"])
        ax.plot(s.index, s.to_numpy(), color=cols[i], lw=o["lw"], ls=o["ls"], marker=m, ms=o["ms"],
                markeredgecolor="white", markeredgewidth=0.8, label=str(name))
    st.labels(ax, c["x"], _ylab(d))
    st.legend(ax)


@plot("area_plot_057", "Area plot", CL, number="057", example=_ex_time_series, roles=_WIDE_ROLES,
      opts=[Opt("alpha", "Fill opacity", "float", 0.35, lo=0.05, hi=1, step=0.05),
            Opt("lines", "Outline each area", "bool", True),
            Opt("base", "Fill down to", "choice", "zero", choices=["zero", "lowest value"])],
      desc=_WIDE_DESC)
def area_plot(fig, df, c, o, st):
    d = _series(df, c)
    if d.empty or not len(d.columns):
        return _nodata(fig, st)
    ax = st.ax(fig.add_subplot())
    cols = st.colors(len(d.columns))
    base = 0.0 if o["base"] == "zero" else float(np.nanmin(d.to_numpy()))
    # largest areas first so that small ones stay visible on top
    order = sorted(range(len(d.columns)), key=lambda i: -np.nansum(d.iloc[:, i].to_numpy() - base))
    for i in order:
        s = d.iloc[:, i].dropna()
        ax.fill_between(s.index, s.to_numpy(), base, color=cols[i], alpha=o["alpha"], linewidth=0,
                        label=str(d.columns[i]))
        if o["lines"]:
            ax.plot(s.index, s.to_numpy(), color=cols[i], lw=1.6)
    ax.set_xlim(d.index.min(), d.index.max()) if len(d) > 1 else None
    if base == 0 and np.nanmin(d.to_numpy()) >= 0:
        ax.set_ylim(bottom=0)
    st.labels(ax, c["x"], _ylab(d))
    h, lab = ax.get_legend_handles_labels()
    idx = sorted(range(len(lab)), key=lambda k: list(map(str, d.columns)).index(lab[k]))
    st.legend(ax, [h[k] for k in idx], [lab[k] for k in idx])


@plot("area_3d_058", "3D area plot", CL, number="058", example=_ex_time_series, roles=_WIDE_ROLES,
      opts=[Opt("elev", "View elevation", "float", 22, lo=-90, hi=90, step=2),
            Opt("azim", "View azimuth", "float", -62, lo=-180, hi=180, step=2),
            Opt("alpha", "Fill opacity", "float", 0.75, lo=0.1, hi=1, step=0.05)],
      desc=_WIDE_DESC + " Each series becomes one filled ribbon along the depth axis.", size=(7.5, 5.5))
def area_3d(fig, df, c, o, st):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    d = _series(df, c)
    if d.empty or not len(d.columns):
        return _nodata(fig, st)
    ax = fig.add_subplot(projection="3d")
    ax.set_in_layout(False)              # 3D axes confuse constrained layout; place it by hand
    ax.set_position([0.0, 0.02, 0.93, 0.9 if st.opts.get("title") else 0.96])
    cols = st.colors(len(d.columns))
    lo = min(0.0, float(np.nanmin(d.to_numpy())))
    for i, name in enumerate(d.columns):
        s = d[name].dropna()
        x, y = s.index.to_numpy(float), s.to_numpy(float)
        if not len(x):
            continue
        verts = [(x[0], i, lo)] + [(a, i, b) for a, b in zip(x, y)] + [(x[-1], i, lo)]
        poly = Poly3DCollection([verts], facecolor=cols[i], edgecolor=_shade(cols[i], -0.35),
                                linewidth=1.0, alpha=o["alpha"])
        ax.add_collection3d(poly)
    ax.set_xlim(d.index.min(), d.index.max() if len(d) > 1 else d.index.min() + 1)
    ax.set_ylim(-0.5, len(d.columns) - 0.5)
    ax.set_zlim(lo, float(np.nanmax(d.to_numpy())) * 1.05 or 1)
    ax.set_box_aspect((1.6, 1.0, 0.8))
    ax.set_yticks(range(len(d.columns)), [str(n) for n in d.columns], fontsize=st.font * 0.8)
    ax.tick_params(labelsize=st.font * 0.75)
    ax.view_init(elev=o["elev"], azim=o["azim"])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor("#cccccc")
        a._axinfo["grid"].update(color="#dddddd", linewidth=0.6)
    ax.grid(st.grid)
    st.labels(ax, c["x"], "")
    zl = ax.get_ylabel() or _ylab(d)      # the user's Y label belongs on the vertical (z) axis
    ax.set_ylabel("")
    ax.set_zlabel(zl, fontsize=st.font)


def _stack_data(df, c, percent):
    d = _series(df, c).fillna(0.0)
    if percent and len(d.columns):
        tot = d.sum(axis=1).replace(0, np.nan)
        d = d.div(tot, axis=0).fillna(0) * 100
    return d


@plot("area_stack_059", "Stacked area", CL, number="059", example=_ex_composition, roles=_WIDE_ROLES,
      opts=[Opt("percent", "Show as % of total", "bool", False),
            Opt("baseline", "Baseline", "choice", "zero", choices=["zero", "symmetric (stream)", "wiggle"]),
            Opt("alpha", "Fill opacity", "float", 0.9, lo=0.1, hi=1, step=0.05)],
      desc="Wide table: a numeric X column (e.g. day), then one numeric column per stacked component "
           "(e.g. cell-type abundance); missing values count as 0.")
def area_stack(fig, df, c, o, st):
    d = _stack_data(df, c, o["percent"])
    if d.empty or not len(d.columns):
        return _nodata(fig, st)
    ax = st.ax(fig.add_subplot())
    base = {"zero": "zero", "symmetric (stream)": "sym", "wiggle": "wiggle"}[o["baseline"]]
    ax.stackplot(d.index, d.to_numpy().T.clip(min=0), labels=[str(x) for x in d.columns],
                 colors=st.colors(len(d.columns)), alpha=o["alpha"], baseline=base,
                 edgecolor="white", linewidth=0.8)
    if len(d) > 1:
        ax.set_xlim(d.index.min(), d.index.max())
    if base == "zero":
        ax.set_ylim(0, 100 if o["percent"] else None)
    st.labels(ax, c["x"], "% of total" if o["percent"] else "Value (stacked)")
    h, lab = ax.get_legend_handles_labels()
    st.legend(ax, h[::-1], lab[::-1])


@plot("line_stack_060", "Stacked lines", CL, number="060", example=_ex_composition, roles=_WIDE_ROLES,
      opts=[Opt("percent", "Show as % of total", "bool", False),
            Opt("fill", "Light fill between lines", "bool", True),
            Opt("markers", "Markers", "bool", True)],
      desc="Wide table: a numeric X column, then one numeric column per component; each line is the "
           "running total of the components up to it.")
def line_stack(fig, df, c, o, st):
    d = _stack_data(df, c, o["percent"])
    if d.empty or not len(d.columns):
        return _nodata(fig, st)
    ax = st.ax(fig.add_subplot())
    cols = st.colors(len(d.columns))
    cum = np.cumsum(d.to_numpy(), axis=1)
    x = d.index.to_numpy(float)
    prev = np.zeros(len(x))
    for i, name in enumerate(d.columns):
        if o["fill"]:
            ax.fill_between(x, prev, cum[:, i], color=cols[i], alpha=0.15, linewidth=0)
        ax.plot(x, cum[:, i], color=cols[i], lw=1.8, marker="o" if o["markers"] else None, ms=5,
                markeredgecolor="white", label=str(name))
        prev = cum[:, i]
    if len(x) > 1:
        ax.set_xlim(x.min(), x.max())
    if np.nanmin(cum) >= 0:
        ax.set_ylim(bottom=0)
    st.labels(ax, c["x"], "Cumulative %" if o["percent"] else "Cumulative value")
    h, lab = ax.get_legend_handles_labels()
    st.legend(ax, h[::-1], lab[::-1])


@plot("calendar_heatmap_098", "Calendar heatmap", CL, number="098", example=_ex_calendar,
      roles=[Role("date", "Date", "any"), Role("value", "Value", "num")],
      opts=[Opt("agg", "Several rows on one day", "choice", "sum", choices=["sum", "mean", "max"]),
            Opt("week_start", "Week starts on", "choice", "Monday", choices=["Monday", "Sunday"]),
            Opt("years", "Show at most (years)", "int", 3, lo=1, hi=10),
            Opt("months", "Outline the months", "bool", True)],
      desc="Two columns: a date (e.g. 2025-03-14) and a numeric value per day; one calendar row per year, "
           "days without data stay grey.", size=(11, 2.6), axis_labels=False, defaults={"cmap": "YlOrRd"})
def calendar_heatmap(fig, df, c, o, st):
    import matplotlib
    from matplotlib.colors import ListedColormap
    dt = pd.to_datetime(df[c["date"]], errors="coerce")
    d = pd.DataFrame({"d": dt.dt.normalize(), "v": num(df, c["value"])}).dropna()
    if d.empty:
        return _nodata(fig, st, "No valid dates / values")
    agg = d.groupby("d")["v"].agg(o["agg"])
    years = sorted(set(agg.index.year))[-int(o["years"]):]
    agg = agg[agg.index.year.isin(years)]
    vmin, vmax = float(agg.min()), float(agg.max())
    if vmax <= vmin:
        vmax = vmin + 1
    cm = matplotlib.colormaps[st.cmap]
    sunday = o["week_start"] == "Sunday"
    names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"] if sunday else \
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes = [fig.add_subplot(len(years), 1, i + 1) for i in range(len(years))]
    st.labels(axes[0])
    mesh = None
    for ax, yr in zip(axes, years):
        days = pd.date_range(f"{yr}-01-01", f"{yr}-12-31")
        wd = days.dayofweek.to_numpy()
        if sunday:
            wd = (wd + 1) % 7
        week = (np.arange(len(days)) + wd[0]) // 7
        grid = np.full((7, week.max() + 1), np.nan)
        valid = np.full_like(grid, np.nan)
        grid[wd, week] = agg.reindex(days).to_numpy()
        valid[wd, week] = 1.0
        ax.pcolormesh(np.ma.masked_invalid(valid), cmap=ListedColormap(["#ebedf0"]), edgecolors="white",
                      linewidth=1.2)
        mesh = ax.pcolormesh(np.ma.masked_invalid(grid), cmap=cm, vmin=vmin, vmax=vmax, edgecolors="white",
                             linewidth=1.2)
        if o["months"]:
            for m in range(2, 13):          # boundary between month m-1 and m
                i0 = int(np.argmax(days.month == m))
                w0, r0 = week[i0], wd[i0]
                xs = [w0 + 1, w0 + 1, w0, w0] if r0 > 0 else [w0, w0]
                ys = [0, r0, r0, 7] if r0 > 0 else [0, 7]
                ax.plot(xs, ys, color="#333333", lw=1.6, solid_capstyle="butt")
        ax.set_aspect("equal")
        ax.set_xlim(0, week.max() + 1)
        ax.set_ylim(7, 0)
        rows = [names.index(n) for n in ("Mon", "Wed", "Fri")]
        ax.set_yticks([r + 0.5 for r in rows], ["Mon", "Wed", "Fri"])
        mpos = [week[int(np.argmax(days.month == m))] + 0.5 for m in range(1, 13)]
        ax.set_xticks(mpos, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.xaxis.tick_top()
        ax.tick_params(length=0, labelsize=st.font * 0.8, colors="#333333")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(str(yr), fontsize=st.font, fontweight="bold")
        ax.set_xlabel("")
    cb = fig.colorbar(mesh, ax=axes, shrink=0.85, pad=0.01, aspect=12 * len(years))
    cb.set_label(c["value"], fontsize=st.font * 0.85)
    cb.ax.tick_params(labelsize=st.font * 0.75)
    cb.outline.set_visible(False)


@plot("radar_052", "Radar (spider) chart", CL, number="052", example=_ex_radar,
      roles=[Role("sample", "Sample (one polygon per row)", "cat"),
             Role("vars", "Axes (one numeric column each)", "num", multi=True)],
      opts=[Opt("scale", "Scale", "choice", "raw values", choices=["raw values", "0–1 per axis"]),
            Opt("fill", "Fill polygons", "bool", True),
            Opt("alpha", "Fill opacity", "float", 0.15, lo=0, hi=1, step=0.05),
            Opt("max_n", "Show at most (samples)", "int", 8, lo=1, hi=30)],
      desc="One row per sample (first column = name), then one numeric column per radar axis (≥ 3); "
           "repeated names are averaged.", size=(7, 6), axis_labels=False)
def radar(fig, df, c, o, st):
    vars_ = [k for k in _lst(c["vars"]) if k != c["sample"]]
    d = pd.DataFrame({k: num(df, k).to_numpy() for k in vars_})
    d["__s"] = as_str(df, c["sample"]).to_numpy()
    d = d.groupby("__s", sort=False).mean()
    d = d.loc[:, d.notna().any()].dropna(how="all").head(int(o["max_n"]))
    if d.shape[1] < 3 or d.empty:
        return _nodata(fig, st, "A radar chart needs at least 3 numeric columns")
    v = d.to_numpy(float)
    if o["scale"] != "raw values":
        mn, mx = np.nanmin(v, 0), np.nanmax(v, 0)
        v = (v - mn) / np.where(mx > mn, mx - mn, 1)
    v = np.where(np.isnan(v), np.nanmin(v), v)
    n = v.shape[1]
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    cols = st.colors(len(d))
    for i, (name, row) in enumerate(zip(d.index, v)):
        a, r = np.r_[ang, ang[:1]], np.r_[row, row[:1]]
        ax.plot(a, r, color=cols[i], lw=1.8, marker="o", ms=4, label=str(name))
        if o["fill"] and o["alpha"] > 0:
            ax.fill(a, r, color=cols[i], alpha=o["alpha"])
    lo = min(0.0, float(np.nanmin(v)))
    ax.set_ylim(lo, float(np.nanmax(v)) * 1.08 if np.nanmax(v) > lo else lo + 1)
    ax.set_xticks(ang, [str(x) for x in d.columns])
    ax.tick_params(axis="x", labelsize=st.font * 0.9, pad=8, colors="#222222")
    ax.tick_params(axis="y", labelsize=st.font * 0.7, colors="#777777")
    ax.set_rlabel_position(180 / n)
    ax.grid(st.grid, color="#cccccc", linestyle="--", linewidth=0.7)
    ax.spines["polar"].set_color("#999999")
    st.labels(ax)
    h, lab = ax.get_legend_handles_labels()
    _side_legend(ax, st, h, lab)


# ── pies & donuts ────────────────────────────────────────────────────────────

def _wedges(ax, vals, colors, center=(0.0, 0.0), r=1.0, width=None, start=90.0, clockwise=True,
            explode=None, edge="white", lw=1.2):
    from matplotlib.patches import Wedge
    vals = np.asarray(vals, float)
    tot = vals.sum()
    out, a = [], float(start)
    for i, (v, col) in enumerate(zip(vals, colors)):
        span = 360.0 * v / tot
        t1, t2 = (a - span, a) if clockwise else (a, a + span)
        a = t1 if clockwise else t2
        cx, cy = center
        if explode is not None and explode[i]:
            m = np.deg2rad((t1 + t2) / 2)
            cx, cy = cx + explode[i] * np.cos(m), cy + explode[i] * np.sin(m)
        w = Wedge((cx, cy), r, t1, t2, width=width, facecolor=col, edgecolor=edge, linewidth=lw)
        ax.add_patch(w)
        out.append(w)
    return out


def _wedge_label(ax, w, text, fs, ppu, force=False, bold=False):
    """Text inside wedge *w* if its box fits inside the wedge (or *force*). Returns True when drawn."""
    if not text:
        return False
    a = np.deg2rad((w.theta1 + w.theta2) / 2)
    inner = (w.r - w.width) if w.width else 0.0
    cands = [(inner + w.r) / 2] if w.width else [0.62 * w.r, 0.55 * w.r, 0.7 * w.r, 0.45 * w.r]
    hw, hh = 0.5 * _tw(text, fs) / ppu, 0.5 * _th(text, fs) / ppu
    gx, gy = np.meshgrid(np.linspace(-hw, hw, 5), np.linspace(-hh, hh, 3))
    t1, span = np.deg2rad(w.theta1), np.deg2rad(w.theta2 - w.theta1)
    pos = None
    for rt in cands:
        x, y = rt * np.cos(a) + gx.ravel(), rt * np.sin(a) + gy.ravel()
        rr = np.hypot(x, y)
        ang = (np.arctan2(y, x) - t1) % (2 * np.pi)
        if (rr <= w.r * 0.99).all() and (rr >= inner + 0.01 * w.r).all() and (ang <= span).all():
            pos = rt
            break
    if pos is None:
        if not force:
            return False
        pos = cands[0]
    cx, cy = w.center
    ax.text(cx + pos * np.cos(a), cy + pos * np.sin(a), text, ha="center", va="center", fontsize=fs,
            color=_ink(w.get_facecolor()), fontweight="bold" if bold else "normal", linespacing=1.1)
    return True


def _outside_labels(ax, fig, wedges, texts, fs, r=1.0, set_limits=True):
    """Labels in two columns left/right of the pie with leader lines, spread so they never overlap."""
    items = []
    for w, t in zip(wedges, texts):
        if not t:
            continue
        a = np.deg2rad((w.theta1 + w.theta2) / 2)
        cx, cy = w.center
        items.append((cx + w.r * np.cos(a), cy + w.r * np.sin(a), np.cos(a), np.sin(a), t))
    wmax = max((_tw(t, fs) for *_, t in items), default=0.0)
    xh, yh = 1.15 * r, 1.15 * r
    for _ in range(4):
        ppu = _ppu(fig, 2 * xh, 2 * yh)
        xh = max(1.15 * r, 1.32 * r + wmax / ppu + 0.04 * r) if items else xh
    line = fs * 1.45 / ppu          # some margin: the final plot can come out a bit smaller
    for side in (1, -1):
        s = sorted([it for it in items if (it[2] >= 0) == (side > 0)], key=lambda it: it[3])
        if not s:
            continue
        hs = [line * (it[4].count("\n") + 1) for it in s]
        ys = [1.12 * r * it[3] for it in s]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + (hs[i] + hs[i - 1]) / 2)
        top = 1.2 * r
        if ys[-1] > top:
            ys[-1] = max(top, ys[-1] - (ys[-1] - top))
            for i in range(len(ys) - 2, -1, -1):
                ys[i] = min(ys[i], ys[i + 1] - (hs[i] + hs[i + 1]) / 2)
        for (xe, ye, ca, sa, t), yt, ht in zip(s, ys, hs):
            xk = xe + 0.06 * r * ca
            yk = ye + 0.06 * r * sa
            xt = side * 1.27 * r
            ax.plot([xe, xk, xt], [ye, yt if abs(yt - yk) > 0.5 * line else yk, yt], color="#777777", lw=0.7,
                    solid_capstyle="round")
            ax.text(xt + side * 0.03 * r, yt, t, ha="left" if side > 0 else "right", va="center", fontsize=fs,
                    linespacing=1.05)
            yh = max(yh, abs(yt) + ht / 2 + 0.03 * r)
    if set_limits:
        _frame(ax, (-xh, xh), (-yh, yh))
    return ppu


def _pie_core(fig, ax, st, names, vals, mode, show_pct, start, hole=0.0, explode=0.0, dec=1):
    vals = np.asarray(vals, float)
    cols = st.colors(len(vals))
    pct = 100 * vals / vals.sum()
    ex = None
    if explode > 0 and len(vals) > 1:
        ex = np.zeros(len(vals))
        ex[int(np.argmax(vals))] = explode
    ws = _wedges(ax, vals, cols, width=(1 - hole) if hole > 0 else None, start=start, explode=ex)
    fs = st.font * 0.8
    w_guess = max(_tw(f"{n} (00.0%)", fs) for n in names) if mode != "legend" else _tw("00.0%", fs)
    xh = 1.6
    for _ in range(3):
        ppu = _ppu(fig, 2 * xh, 2.6, wfrac=0.62 if mode == "legend" else 0.9)
        xh = 1.36 + w_guess / ppu
    out = []
    for w, n, p in zip(ws, names, pct):
        pt = f"{p:.{dec}f}%"
        o_txt = ""
        if mode == "inside":
            if not _wedge_label(ax, w, f"{n}\n{pt}" if show_pct else n, fs, ppu):
                o_txt = f"{n} ({pt})" if show_pct else n
        elif mode == "outside":
            o_txt = n
            if show_pct and not _wedge_label(ax, w, pt, fs, ppu):
                o_txt = f"{n} ({pt})"
        elif show_pct and not _wedge_label(ax, w, pt, fs, ppu):
            o_txt = pt
        out.append(o_txt)
    _outside_labels(ax, fig, ws, out, fs)
    if mode == "legend":
        _side_legend(ax, st, ws, names)
    return ws


_PIE_ROLES = [Role("label", "Categories", "cat"), Role("value", "Values (summed per category)", "num")]
_PIE_OPTS = [Opt("mode", "Category names", "choice", "outside", choices=["outside", "inside", "legend"]),
             Opt("pct", "Show percentages", "bool", True),
             Opt("dec", "Decimals of %", "int", 1, lo=0, hi=3),
             Opt("start", "Start angle (°)", "float", 90, lo=-360, hi=360, step=5),
             Opt("sort", "Sort slices", "choice", "descending", choices=["none", "descending", "ascending"])]


def _pie_series(df, c, o):
    s = _agg(df, c["label"], c["value"])
    if o.get("sort", "none") != "none":
        s = s.sort_values(ascending=o["sort"] == "ascending")
    return s


@plot("pie_2d_002", "2D pie", CP, number="002", example=_ex_celltypes, roles=_PIE_ROLES,
      opts=_PIE_OPTS + [Opt("explode", "Pull out the largest slice", "float", 0.06, lo=0, hi=0.5, step=0.02)],
      desc="Two columns: a category (e.g. cell type) and a positive value (e.g. cell count); rows of the same "
           "category are summed, zero / negative values are dropped.", axis_labels=False)
def pie_2d(fig, df, c, o, st):
    s = _pie_series(df, c, o)
    if s.empty:
        return _nodata(fig, st)
    ax = st.bare(fig.add_subplot())
    _pie_core(fig, ax, st, list(s.index), s.to_numpy(), o["mode"], o["pct"], o["start"],
              explode=o["explode"], dec=int(o["dec"]))
    st.labels(ax)


@plot("donut_051", "Donut chart", CP, number="051", example=_ex_celltypes, roles=_PIE_ROLES,
      opts=_PIE_OPTS + [Opt("hole", "Hole size (0–0.9)", "float", 0.55, lo=0.1, hi=0.9, step=0.05),
                        Opt("centre", "Centre text", "str", "Total"),
                        Opt("total", "Show the total in the centre", "bool", True)],
      desc="Two columns: a category and a positive value; one ring, the grand total is written in the hole.",
      axis_labels=False)
def donut(fig, df, c, o, st):
    s = _pie_series(df, c, o)
    if s.empty:
        return _nodata(fig, st)
    ax = st.bare(fig.add_subplot())
    _pie_core(fig, ax, st, list(s.index), s.to_numpy(), o["mode"], o["pct"], o["start"], hole=o["hole"],
              dec=int(o["dec"]))
    if o["total"]:
        txt = "\n".join(t for t in (o["centre"], _fmt(s.sum())) if t)
        ax.text(0, 0, txt, ha="center", va="center", fontsize=st.font * 1.1 * max(0.6, o["hole"] / 0.55) ** 0.5,
                fontweight="bold", color="#333333", linespacing=1.3)
    st.labels(ax)


@plot("pie_3d_020", "3D pie", CP, number="020", example=_ex_celltypes, roles=_PIE_ROLES,
      opts=[Opt("tilt", "Tilt (1 = flat, seen from above)", "float", 0.5, lo=0.2, hi=1, step=0.05),
            Opt("depth", "Thickness", "float", 0.16, lo=0, hi=0.5, step=0.02),
            Opt("explode", "Pull out the largest slice", "float", 0.08, lo=0, hi=0.5, step=0.02),
            Opt("start", "Start angle (°)", "float", 120, lo=-360, hi=360, step=5),
            Opt("pct", "Show percentages", "bool", True),
            Opt("sort", "Sort slices", "choice", "descending", choices=["none", "descending", "ascending"])],
      desc="Two columns: a category and a positive value; drawn as a tilted, shaded pie with category names in "
           "the legend.", axis_labels=False)
def pie_3d(fig, df, c, o, st):
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Ellipse, Patch, Polygon
    s = _pie_series(df, c, o)
    if s.empty:
        return _nodata(fig, st)
    ax = st.bare(fig.add_subplot())
    vals = s.to_numpy(float)
    names = list(s.index)
    cols = st.colors(len(vals))
    tilt, dep = float(o["tilt"]), float(o["depth"])
    fr = vals / vals.sum()
    a = np.deg2rad(o["start"])
    sl = []
    for i, f in enumerate(fr):
        t2, t1 = a, a - 2 * np.pi * f
        a = t1
        m = (t1 + t2) / 2
        e = o["explode"] if (i == int(np.argmax(vals)) and len(vals) > 1) else 0.0
        sl.append((t1, t2, e * np.cos(m), e * np.sin(m) * tilt, cols[i], f, m))
    ax.add_patch(Ellipse((0.03, -dep - 0.04), 2.25, 2.25 * tilt, color="black", alpha=0.08, lw=0))

    def pt(t, ox, oy, z=0.0):
        return np.column_stack([np.cos(t) + ox, tilt * np.sin(t) + oy - z])

    # side walls, back slices first
    for t1, t2, ox, oy, col, f, m in sorted(sl, key=lambda q: -np.sin(q[6])):
        if dep <= 0:
            break
        dark = _shade(col, -0.35)
        for t, front in ((t1, np.cos(t1) > 0), (t2, np.cos(t2) < 0)):      # radial cut faces
            if front and o["explode"] > 0:
                p0 = np.array([ox, oy])
                p1 = pt(np.array([t]), ox, oy)[0]
                ax.add_patch(Polygon([p0, p1, p1 - [0, dep], p0 - [0, dep]], closed=True, facecolor=dark,
                                     edgecolor=dark, lw=0.5))
        tt = np.linspace(t1, t2, max(3, int(f * 360)))
        quads, fcs = [], []
        for u, v in zip(tt[:-1], tt[1:]):
            if np.sin((u + v) / 2) >= 0:
                continue
            top = pt(np.array([u, v]), ox, oy)
            bot = top - [0, dep]
            quads.append([top[0], top[1], bot[1], bot[0]])
            k = 0.5 + 0.25 * np.cos((u + v) / 2 + 0.6)       # light from the front-left
            fcs.append(_shade(col, -(0.45 - 0.3 * k)))
        if quads:
            ax.add_collection(PolyCollection(quads, facecolors=fcs, edgecolors=fcs, linewidths=0.6))
    fs = st.font * 0.8
    outs = []
    for t1, t2, ox, oy, col, f, m in sl:
        tt = np.linspace(t1, t2, max(3, int(f * 360)))
        P = np.vstack([[ox, oy], pt(tt, ox, oy)])
        ax.add_patch(Polygon(P, closed=True, facecolor=col, edgecolor="white", lw=0.9))
        if o["pct"]:
            rr = 0.66
            x, y = ox + rr * np.cos(m), oy + rr * tilt * np.sin(m)
            room = 2 * rr * np.sin(min(np.pi, t2 - t1) / 2) * min(1, tilt * 2.2)
            ppu = _ppu(fig, 2.9, 2.4 * tilt + dep + 0.4, wfrac=0.65)
            if _tw("00.0%", fs) < room * ppu * 1.05:
                ax.text(x, y, f"{f * 100:.1f}%", ha="center", va="center", fontsize=fs, color=_ink(col),
                        fontweight="bold")
            else:
                xo, yo = ox + 1.13 * np.cos(m), oy + 1.13 * tilt * np.sin(m) - (dep if np.sin(m) < 0 else 0)
                outs.append([xo, yo, f"{f * 100:.1f}%"])
    ppu = _ppu(fig, 2.9, 2.4 * tilt + dep + 0.4, wfrac=0.65)
    line = fs * 1.45 / ppu          # some margin: the final plot can come out a bit smaller
    for side in (1, -1):                       # spread small-slice labels so they never overlap
        grp = sorted([q for q in outs if (q[0] >= 0) == (side > 0)], key=lambda q: q[1])
        for i in range(1, len(grp)):
            grp[i][1] = max(grp[i][1], grp[i - 1][1] + line)
        for xo, yo, t in grp:
            ax.text(xo, yo, t, ha="left" if side > 0 else "right", va="center", fontsize=fs)
    _frame(ax, (-1.45, 1.45), (-tilt - dep - 0.25, tilt + 0.2))
    st.labels(ax)
    _side_legend(ax, st, [Patch(facecolor=cc, edgecolor="white") for cc in cols], names)


def _two_layer(fig, df, c, o, st, donut_):
    from matplotlib.patches import Patch
    d = pd.DataFrame({"m": as_str(df, c["main"]), "s": as_str(df, c["sub"]), "v": num(df, c["value"])}).dropna()
    d = d[np.isfinite(d["v"]) & (d["v"] > 0)]
    if d.empty:
        return _nodata(fig, st)
    mains = list(dict.fromkeys(d["m"]))
    g = d.groupby(["m", "s"], sort=False)["v"].sum()
    mv, subs, sv, scol = [], [], [], []
    mcol = st.colors(len(mains))
    for i, m in enumerate(mains):
        part = g.loc[m]
        if o["sort"]:
            part = part.sort_values(ascending=False)
        mv.append(part.sum())
        ks = np.linspace(0, 0.6, len(part) + 1)[:-1] if len(part) > 1 else [0.0]
        for j, (sname, v) in enumerate(part.items()):
            subs.append(sname)
            sv.append(v)
            scol.append(_shade(mcol[i], ks[j]))
    tot = float(np.sum(mv))
    ax = st.bare(fig.add_subplot())
    r_in = 0.6
    inner = _wedges(ax, mv, mcol, r=r_in, width=(r_in - 0.24) if donut_ else None, start=o["start"], lw=1.5)
    outer = _wedges(ax, sv, scol, r=1.0, width=0.36, start=o["start"], lw=1.0)
    fs = st.font * 0.78
    names_out = [f"{n} ({100 * v / tot:.1f}%)" if o["pct"] else n for n, v in zip(subs, sv)]
    w_guess = max(_tw(t, fs) for t in names_out)
    xh = 1.6
    for _ in range(3):
        ppu = _ppu(fig, 2 * xh, 2.6, wfrac=0.7)
        xh = 1.36 + w_guess / ppu
    for w, m, v in zip(inner, mains, mv):
        p = f"{100 * v / tot:.1f}%"
        txt = {"name + %": f"{m}\n{p}" if o["pct"] else m, "name": m, "%": p, "none": ""}[o["inner"]]
        if not _wedge_label(ax, w, txt, fs, ppu, bold=True) and o["inner"] == "name + %":
            _wedge_label(ax, w, p if o["pct"] else "", fs, ppu, bold=True)
    texts = []
    for w, t, n in zip(outer, names_out, subs):
        if o["outer"] == "outside":
            texts.append(t)
        elif o["outer"] == "inside":
            texts.append("" if _wedge_label(ax, w, t.replace(" (", "\n("), fs, ppu) else t)
        else:
            texts.append("")
    _outside_labels(ax, fig, outer, texts, fs)
    if donut_ and o["total"]:
        ax.text(0, 0, f"Total\n{_fmt(tot)}", ha="center", va="center", fontsize=st.font * 0.95,
                fontweight="bold", color="#333333")
    st.labels(ax)
    _side_legend(ax, st, [Patch(facecolor=cc, edgecolor="white") for cc in mcol], mains, title=c["main"])


_TWO_ROLES = [Role("main", "Main category (inner)", "cat"), Role("sub", "Sub-category (outer)", "cat"),
              Role("value", "Values", "num")]
_TWO_OPTS = [Opt("inner", "Inner labels", "choice", "name + %", choices=["name + %", "name", "%", "none"]),
             Opt("outer", "Outer labels", "choice", "outside", choices=["outside", "inside", "none"]),
             Opt("pct", "Show percentages", "bool", True),
             Opt("start", "Start angle (°)", "float", 90, lo=-360, hi=360, step=5),
             Opt("sort", "Sort sub-categories by size", "bool", False)]
_TWO_DESC = ("Three columns: main category (e.g. lineage), sub-category (e.g. cell type) and a positive value; "
             "sub-categories are drawn in shades of their main category's colour.")


@plot("two_layer_pie_003", "Two-layer pie", CP, number="003", example=_ex_two_layer, roles=_TWO_ROLES,
      opts=_TWO_OPTS, desc=_TWO_DESC, size=(8, 5.5), axis_labels=False)
def two_layer_pie(fig, df, c, o, st):
    _two_layer(fig, df, c, o, st, False)


@plot("two_layer_donut_004", "Two-layer donut", CP, number="004", example=_ex_two_layer, roles=_TWO_ROLES,
      opts=_TWO_OPTS + [Opt("total", "Show the total in the centre", "bool", True)], desc=_TWO_DESC,
      size=(8, 5.5), axis_labels=False)
def two_layer_donut(fig, df, c, o, st):
    _two_layer(fig, df, c, o, st, True)


@plot("pie_matrix_033", "Pie matrix", CP, number="033", example=_ex_pie_matrix,
      roles=[Role("row", "Rows of the matrix", "cat"), Role("col", "Columns of the matrix", "cat", optional=True),
             Role("parts", "Components (one numeric column each)", "num", multi=True)],
      opts=[Opt("scale", "Pie size ∝ total", "bool", False),
            Opt("pct", "Show percentages", "bool", False),
            Opt("totals", "Write the total under each pie", "bool", False),
            Opt("hole", "Hole size (0 = pie)", "float", 0.0, lo=0, hi=0.85, step=0.05),
            Opt("start", "Start angle (°)", "float", 90, lo=-360, hi=360, step=5)],
      desc="One row per pie: a row label (e.g. tissue), an optional column label (e.g. time point), then one "
           "numeric column per component (e.g. cell-type counts).", size=(8, 6), axis_labels=False)
def pie_matrix(fig, df, c, o, st):
    parts = [k for k in _lst(c["parts"]) if k not in (c["row"], c.get("col"))]
    if not parts:
        return _nodata(fig, st)
    D = pd.DataFrame({k: num(df, k).clip(lower=0).to_numpy() for k in parts})
    D["__r"] = as_str(df, c["row"]).to_numpy()
    D["__c"] = as_str(df, c["col"]).to_numpy() if c.get("col") else ""
    G = D.groupby(["__r", "__c"], sort=False)[parts].sum(min_count=1).fillna(0)
    G = G[G.sum(axis=1) > 0]
    if G.empty:
        return _nodata(fig, st)
    rows = list(dict.fromkeys(G.index.get_level_values(0)))
    colsl = list(dict.fromkeys(G.index.get_level_values(1)))
    pc = st.colors(len(parts))
    ax = st.bare(fig.add_subplot())
    fs = st.font * 0.8
    tot_max = G.sum(axis=1).max()
    if c.get("col"):
        pos = {(rv, cv): (colsl.index(cv), rows.index(rv)) for rv, cv in G.index}
        ncol, nrow = len(colsl), len(rows)
    else:
        ncol = int(np.ceil(np.sqrt(len(G))))
        nrow = int(np.ceil(len(G) / ncol))
        pos = {k: (i % ncol, i // ncol) for i, k in enumerate(G.index)}
    gap = 1.0 if c.get("col") else 0.0
    yrow = 1.0 + (0.25 if (not c.get("col") or o["totals"]) else 0.0)
    hdr_w = max(_tw(r, fs) for r in rows) if c.get("col") else 0.0
    xs = ncol + 0.2
    for _ in range(3):
        ppu = _ppu(fig, xs, nrow * yrow + 0.6, wfrac=0.75)
        xs = ncol + 0.2 + hdr_w / ppu
    for (rv, cv), vals in G.iterrows():
        x, yi = pos[(rv, cv)]
        y = -yi * yrow
        v = vals.to_numpy(float)
        rad = 0.44 * (np.sqrt(v.sum() / tot_max) if o["scale"] else 1.0)
        ws = _wedges(ax, v, pc, center=(x, y), r=rad, width=rad * (1 - o["hole"]) if o["hole"] > 0 else None,
                     start=o["start"], lw=0.8)
        if o["pct"]:
            for w, p in zip(ws, 100 * v / v.sum()):
                if p > 0:
                    _wedge_label(ax, w, f"{p:.0f}%", fs * 0.85, ppu)
        sub = []
        if not c.get("col"):
            sub.append(str(rv))
        if o["totals"]:
            sub.append(_fmt(v.sum()))
        if sub:
            ax.text(x, y - 0.47, "\n".join(sub), ha="center", va="top", fontsize=fs, linespacing=1.1)
    if c.get("col"):
        hfs = st.font * 0.85
        rot = 35 if max(_tw(cv, hfs) for cv in colsl) > 0.95 * ppu else 0
        for j, cv in enumerate(colsl):
            ax.text(j, 0.55, cv, ha="left" if rot else "center", va="bottom", fontsize=hfs, fontweight="bold",
                    rotation=rot, rotation_mode="anchor")
        for i, rv in enumerate(rows):
            ax.text(-0.55, -i * yrow, rv, ha="right", va="center", fontsize=st.font * 0.85, fontweight="bold")
    left = -0.55 - (hdr_w / ppu + 0.05 if c.get("col") else 0)
    bottom = -(nrow - 1) * yrow - 0.5 - (0.3 if (not c.get("col") or o["totals"]) else 0)
    _frame(ax, (left, ncol - 0.45), (bottom, 0.55 + (fs * 1.6 / ppu if c.get("col") else 0)))
    st.labels(ax)
    _side_legend(ax, st, ws, [str(p) for p in parts])
    _ = gap


@plot("multi_pie_128", "Multiple pies", CP, number="128", example=_ex_multi_pie,
      roles=[Role("label", "Categories (slices)", "cat"),
             Role("values", "One numeric column per pie", "num", multi=True)],
      opts=[Opt("hole", "Hole size (0 = pie)", "float", 0.0, lo=0, hi=0.85, step=0.05),
            Opt("pct", "Show percentages", "bool", True),
            Opt("ncols", "Pies per row (0 = auto)", "int", 0, lo=0, hi=10),
            Opt("totals", "Write the total under each pie", "bool", True),
            Opt("start", "Start angle (°)", "float", 90, lo=-360, hi=360, step=5)],
      desc="First column = category (slice), then one numeric column per pie (e.g. one per patient); "
           "all pies share one legend.", size=(10, 4.4), axis_labels=False)
def multi_pie(fig, df, c, o, st):
    vc = [k for k in _lst(c["values"]) if k != c["label"]]
    if not vc:
        return _nodata(fig, st)
    D = pd.DataFrame({k: num(df, k).clip(lower=0).to_numpy() for k in vc})
    D["__l"] = as_str(df, c["label"]).to_numpy()
    D = D.groupby("__l", sort=False)[vc].sum(min_count=1).fillna(0)
    vc = [k for k in vc if D[k].sum() > 0]
    if not vc:
        return _nodata(fig, st)
    n = len(vc)
    ncol = int(o["ncols"]) or (n if n <= 5 else int(np.ceil(np.sqrt(n))))
    nrow = int(np.ceil(n / ncol))
    cols = st.colors(len(D))
    ax = st.bare(fig.add_subplot())
    fs = st.font * 0.75
    ppu = _ppu(fig, ncol * 2.5, nrow * 2.9, wfrac=0.8)
    ws = []
    for k, name in enumerate(vc):
        cx, cy = (k % ncol) * 2.5, -(k // ncol) * 2.9
        v = D[name].to_numpy(float)
        keep = v > 0
        w = _wedges(ax, v[keep], [cc for cc, kk in zip(cols, keep) if kk], center=(cx, cy),
                    width=(1 - o["hole"]) if o["hole"] > 0 else None, start=o["start"])
        if o["pct"]:
            for wi, p in zip(w, 100 * v[keep] / v.sum()):
                _wedge_label(ax, wi, f"{p:.0f}%", fs, ppu)
        ax.text(cx, cy - 1.12, str(name), ha="center", va="top", fontsize=st.font * 0.9, fontweight="bold")
        if o["totals"]:
            ax.text(cx, cy - 1.12 - st.font * 1.35 / ppu, f"n = {_fmt(v.sum())}", ha="center", va="top",
                    fontsize=fs, color="#555555")
        if keep.all() and len(w) > len(ws):
            ws = w
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=cc, edgecolor="white") for cc in cols]
    _frame(ax, (-1.15, (ncol - 1) * 2.5 + 1.15), (-(nrow - 1) * 2.9 - 1.75, 1.1))
    st.labels(ax)
    _side_legend(ax, st, handles, list(D.index), title=c["label"])


# ── waffles ──────────────────────────────────────────────────────────────────

def _largest_remainder(v, n):
    q = v / v.sum() * n
    k = np.floor(q).astype(int)
    for i in np.argsort(-(q - k))[: n - k.sum()]:
        k[i] += 1
    return k


def _waffle(fig, df, c, o, st, circle):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle, Patch, Rectangle
    s = _agg(df, c["label"], c["value"])
    if s.empty:
        return _nodata(fig, st)
    nr, nc = int(o["rows"]), int(o["cols"])
    N = nr * nc
    k = _largest_remainder(s.to_numpy(float), N)
    cid = np.repeat(np.arange(len(s)), k)
    cols = st.colors(len(s))
    g = float(o["gap"])
    patches, fc = [], []
    for i in range(N):
        if o["order"] == "by row":
            x, y = i % nc, nr - 1 - i // nc
        else:
            x, y = i // nr, i % nr
        if circle:
            patches.append(Circle((x + 0.5, y + 0.5), 0.5 - g / 2))
        else:
            patches.append(Rectangle((x + g / 2, y + g / 2), 1 - g, 1 - g))
        fc.append(cols[cid[i]] if i < len(cid) else "#eeeeee")
    ax = st.bare(fig.add_subplot())
    ax.add_collection(PatchCollection(patches, facecolors=fc, edgecolors="none"))
    _frame(ax, (-0.1, nc + 0.1), (-0.1, nr + 0.1))
    pct = 100 * s.to_numpy() / s.sum()
    labs = [f"{n}  {p:.1f}%" + (f"  ({_fmt(v)})" if o["values"] else "") for n, p, v in zip(s.index, pct, s)]
    st.labels(ax)
    _side_legend(ax, st, [Patch(facecolor=cc) for cc in cols], labs)


_WAFFLE_OPTS = [Opt("rows", "Rows", "int", 10, lo=1, hi=60), Opt("cols", "Columns", "int", 10, lo=1, hi=100),
                Opt("gap", "Gap between cells", "float", 0.12, lo=0, hi=0.6, step=0.02),
                Opt("order", "Fill order", "choice", "by row", choices=["by row", "by column"]),
                Opt("values", "Show values in the legend", "bool", True)]


@plot("waffle_050", "Waffle chart", CP, number="050", example=_ex_variants, roles=_PIE_ROLES, opts=_WAFFLE_OPTS,
      desc="Two columns: a category and a positive value; each of the rows × columns squares (100 by "
           "default, i.e. 1 % each) is coloured by category share.", size=(7.5, 5), axis_labels=False)
def waffle(fig, df, c, o, st):
    _waffle(fig, df, c, o, st, False)


@plot("circle_waffle_089", "Circle waffle", CP, number="089", example=_ex_variants, roles=_PIE_ROLES,
      opts=_WAFFLE_OPTS, desc="Two columns: a category and a positive value; 100 dots (by default) coloured "
                              "by category share.", size=(7.5, 5), axis_labels=False)
def circle_waffle(fig, df, c, o, st):
    _waffle(fig, df, c, o, st, True)


# ── funnels ──────────────────────────────────────────────────────────────────

_FUNNEL_ROLES = [Role("stage", "Stages (top → bottom)", "cat"), Role("value", "Values", "num")]
_FUNNEL_OPTS = [Opt("pct", "Percentage", "choice", "of first", choices=["of first", "of previous", "none"]),
                Opt("colours", "Colours", "choice", "palette", choices=["palette", "one colour (gradient)"])]
_FUNNEL_DESC = ("Two columns: stage name and value, in funnel order from top to bottom (e.g. reads left after "
                "each processing step); rows of the same stage are summed.")


def _funnel_data(df, c):
    s = _agg(df, c["stage"], c["value"], positive=False).clip(lower=0)
    return s if len(s) and s.max() > 0 else pd.Series(dtype=float)


def _funnel_text(s, i, mode):
    v = s.iloc[i]
    if mode == "of first":
        return f"{_fmt(v)} ({100 * v / s.iloc[0]:.1f}%)" if s.iloc[0] > 0 else _fmt(v)
    if mode == "of previous" and i > 0:
        return f"{_fmt(v)} ({100 * v / s.iloc[i - 1]:.1f}%)" if s.iloc[i - 1] > 0 else _fmt(v)
    return _fmt(v)


def _funnel_colours(st, n, mode):
    if mode == "palette":
        return st.colors(n)
    return [_shade(st.color(0), k) for k in np.linspace(-0.15, 0.55, n)]


def _funnel_axes(fig, st, s, o, half_fn, top=0.5, bottom=None):
    """Bare axes with names to the left of the shape and values centred (or right of narrow stages)."""
    n = len(s)
    fs = st.font * 0.85
    mx = float(s.max())
    texts = [_funnel_text(s, i, o["pct"]) for i in range(n)]
    name_w = max(_tw(x, fs) for x in s.index) + fs
    val_w = max(_tw(t, fs) for t in texts) + fs
    AW = fig.get_size_inches()[0] * 72 * 0.94
    halves = [half_fn(i) for i in range(n)]
    outside = [2 * h * 0 for h in halves]
    xspan = mx * 1.04
    for _ in range(4):
        ppu = AW / xspan
        outside = [_tw(t, fs) > 2 * h * ppu * 0.92 for t, h in zip(texts, halves)]
        extra = val_w if any(outside) else 0.0
        frac = min(0.75, (name_w + extra) / AW)
        xspan = mx * 1.04 / (1 - frac)
        right = mx / 2 + (extra / ppu if any(outside) else 0.02 * mx)
    ax = st.bare(fig.add_subplot())
    ppu = AW / xspan
    for i in range(n):
        y = -i - 0.5
        ax.text(-halves[i] - 0.02 * mx, y, str(s.index[i]), ha="right", va="center", fontsize=fs)
        if outside[i]:
            ax.text(halves[i] + 0.02 * mx, y, texts[i], ha="left", va="center", fontsize=fs, color="#333333")
        else:
            ax.text(0, y, texts[i], ha="center", va="center", fontsize=fs, fontweight="bold",
                    color=_ink(o["_cols"][i]))
    ax.set_xlim(-mx / 2 - name_w / ppu - 0.01 * mx, max(right, mx / 2 + 0.01 * mx))
    ax.set_ylim(-n if bottom is None else bottom, top - 0.5 + 0.02)
    return ax


@plot("rect_funnel_007", "Rectangle funnel", CP, number="007", example=_ex_funnel, roles=_FUNNEL_ROLES,
      opts=_FUNNEL_OPTS + [Opt("connect", "Shaded connectors", "bool", True),
                           Opt("height", "Bar height", "float", 0.72, lo=0.2, hi=1, step=0.04)],
      desc=_FUNNEL_DESC, size=(8, 5), axis_labels=False)
def rect_funnel(fig, df, c, o, st):
    from matplotlib.patches import Polygon, Rectangle
    s = _funnel_data(df, c)
    if s.empty:
        return _nodata(fig, st)
    n, h = len(s), float(o["height"])
    v = s.to_numpy(float)
    o = dict(o, _cols=_funnel_colours(st, n, o["colours"]))
    ax = _funnel_axes(fig, st, s, o, lambda i: v[i] / 2)
    for i in range(n):
        y = -i - 0.5
        ax.add_patch(Rectangle((-v[i] / 2, y - h / 2), v[i], h, facecolor=o["_cols"][i], edgecolor="none"))
        if o["connect"] and i < n - 1 and h < 1:
            ax.add_patch(Polygon([(-v[i] / 2, y - h / 2), (v[i] / 2, y - h / 2), (v[i + 1] / 2, y - 1 + h / 2),
                                  (-v[i + 1] / 2, y - 1 + h / 2)], facecolor="#e6e6e6", edgecolor="none"))
    st.labels(ax)


def _smooth_halves(v, tail):
    from scipy.interpolate import PchipInterpolator
    n = len(v)
    t = np.r_[0.0, np.arange(n) + 0.5, float(n)]
    w = np.r_[v[0], v, v[-1] * tail]
    return PchipInterpolator(t, w / 2)


@plot("funnel_plot_005", "Funnel plot", CP, number="005", example=_ex_funnel, roles=_FUNNEL_ROLES,
      opts=_FUNNEL_OPTS + [Opt("tail", "Bottom width (× last stage)", "float", 0.6, lo=0.05, hi=1, step=0.05),
                           Opt("sep", "White lines between stages", "bool", True)],
      desc=_FUNNEL_DESC + " The outline is a smooth curve through the stage values.", size=(8, 5),
      axis_labels=False)
def funnel_plot(fig, df, c, o, st):
    s = _funnel_data(df, c)
    if s.empty:
        return _nodata(fig, st)
    n = len(s)
    v = s.to_numpy(float)
    f = _smooth_halves(v, o["tail"])
    o = dict(o, _cols=_funnel_colours(st, n, o["colours"]))
    ax = _funnel_axes(fig, st, s, o, lambda i: float(f(i + 0.5)))
    for i in range(n):
        t = np.linspace(i, i + 1, 40)
        hw = np.clip(f(t), 0, None)
        ax.fill_betweenx(-t, -hw, hw, color=o["_cols"][i], linewidth=0)
        if o["sep"] and i:
            ax.plot([-hw[0], hw[0]], [-i, -i], color="white", lw=2)
    st.labels(ax)


@plot("trapezoid_funnel_006", "Trapezoid funnel", CP, number="006", example=_ex_funnel, roles=_FUNNEL_ROLES,
      opts=_FUNNEL_OPTS + [Opt("gap", "Gap between stages", "float", 0.08, lo=0, hi=0.6, step=0.02),
                           Opt("tail", "Bottom width (× last stage)", "float", 0.7, lo=0.05, hi=1, step=0.05)],
      desc=_FUNNEL_DESC + " Each stage is a trapezoid from its own value to the next one.", size=(8, 5),
      axis_labels=False)
def trapezoid_funnel(fig, df, c, o, st):
    from matplotlib.patches import Polygon
    s = _funnel_data(df, c)
    if s.empty:
        return _nodata(fig, st)
    n = len(s)
    v = s.to_numpy(float)
    nxt = np.r_[v[1:], v[-1] * o["tail"]]
    g = float(o["gap"]) / 2
    o = dict(o, _cols=_funnel_colours(st, n, o["colours"]))
    ax = _funnel_axes(fig, st, s, o, lambda i: (v[i] + nxt[i]) / 4)
    for i in range(n):
        top, bot = -i - g, -i - 1 + g
        ax.add_patch(Polygon([(-v[i] / 2, top), (v[i] / 2, top), (nxt[i] / 2, bot), (-nxt[i] / 2, bot)],
                             facecolor=o["_cols"][i], edgecolor="none"))
    st.labels(ax)


# ── area-proportional shapes ─────────────────────────────────────────────────

def _sorted_pos(df, c, o):
    s = _agg(df, c["label"], c["value"])
    if o.get("sort", "none") != "none":
        s = s.sort_values(ascending=o["sort"] == "ascending")
    return s


@plot("circle_area_021", "Circle area", CP, number="021", example=_ex_circle_area, roles=_PIE_ROLES,
      opts=[Opt("sort", "Sort", "choice", "descending", choices=["none", "descending", "ascending"]),
            Opt("align", "Align circles", "choice", "bottom", choices=["bottom", "centre"]),
            Opt("values", "Show values", "bool", True)],
      desc="Two columns: a category and a positive value; each value is a circle whose area is proportional "
           "to it.", size=(8, 4.5), axis_labels=False)
def circle_area(fig, df, c, o, st):
    from matplotlib.patches import Circle
    s = _sorted_pos(df, c, o)
    if s.empty:
        return _nodata(fig, st)
    v = s.to_numpy(float)
    r = np.sqrt(v / v.max())
    fs = st.font * 0.85
    names = [str(x) for x in s.index]
    vt = [_fmt(x) for x in v]
    X, Y = 2 * r.sum() + 0.2 * len(r), 2.6
    for _ in range(4):
        ppu = _ppu(fig, X, Y)
        slot = np.maximum(2 * r, np.array([_tw(nm, fs) for nm in names]) / ppu) + 0.15
        X = slot.sum()
        Y = 2.0 + 2.5 * fs * 1.3 / ppu
    xc = np.cumsum(slot) - slot / 2
    cols = st.colors(len(v))
    ax = st.bare(fig.add_subplot())
    line = fs * 1.45 / ppu          # some margin: the final plot can come out a bit smaller
    for i in range(len(v)):
        yc = r[i] if o["align"] == "bottom" else 1.0
        ax.add_patch(Circle((xc[i], yc), r[i], facecolor=cols[i], edgecolor="white", lw=1.2, alpha=0.95))
        ax.text(xc[i], (0 if o["align"] == "bottom" else 1 - r.max()) - 0.4 * line, names[i], ha="center",
                va="top", fontsize=fs)
        if o["values"]:
            if _tw(vt[i], fs) < 1.7 * r[i] * ppu and fs * 1.3 < 2 * r[i] * ppu:
                ax.text(xc[i], yc, vt[i], ha="center", va="center", fontsize=fs, fontweight="bold",
                        color=_ink(cols[i]))
            else:
                ax.text(xc[i], yc + r[i] + 0.2 * line, vt[i], ha="center", va="bottom", fontsize=fs)
    if o["align"] == "bottom":
        ax.axhline(0, color="#999999", lw=0.8)
    lo = (0 if o["align"] == "bottom" else 1 - r.max()) - 1.6 * line
    _frame(ax, (0, X), (lo, (2 * r.max() if o["align"] == "bottom" else 1 + r.max()) + 1.3 * line))
    st.labels(ax)


@plot("half_circle_022", "Half-circle area", CP, number="022", example=_ex_nested, roles=_PIE_ROLES,
      opts=[Opt("layout", "Layout", "choice", "nested", choices=["nested", "side by side"]),
            Opt("alpha", "Opacity", "float", 0.9, lo=0.2, hi=1, step=0.05)],
      desc="Two columns: a category and a positive value (e.g. nested gene sets: all → expressed → DE); each "
           "value is a half circle on a common baseline with area proportional to it.", size=(8, 5),
      axis_labels=False)
def half_circle(fig, df, c, o, st):
    from matplotlib.patches import Wedge
    s = _agg(df, c["label"], c["value"])
    if s.empty:
        return _nodata(fig, st)
    nested = o["layout"] == "nested"
    if nested:
        s = s.sort_values(ascending=False)
    v = s.to_numpy(float)
    r = np.sqrt(v / v.max())
    cols = st.colors(len(v))
    fs = st.font * 0.85
    labs = [f"{n}: {_fmt(x)}" for n, x in zip(s.index, v)]
    ax = st.bare(fig.add_subplot())
    if nested:
        X, Y = 2.4, 1.5
        for _ in range(4):
            ppu = _ppu(fig, X, Y)
            line = fs * 1.45 / ppu
            right = max(rr + _tw(t, fs) / ppu + 0.02 for rr, t in zip(r, labs))
            X = 1.05 + max(1.05, right)
            Y = 1.08 + (len(v) + 0.6) * line
        for i in range(len(v)):
            ax.add_patch(Wedge((0, 0), r[i], 0, 180, facecolor=cols[i], edgecolor="white", lw=1.5,
                               alpha=o["alpha"]))
        for i in range(len(v)):
            yt = -(i + 0.7) * line
            ax.plot([r[i], r[i]], [0, yt + 0.35 * line], color="#777777", lw=0.7)
            ax.plot([r[i]], [0], "o", ms=3, color="#555555")
            ax.text(r[i] + 0.012, yt, labs[i], ha="left", va="center", fontsize=fs)
        ax.axhline(0, color="#666666", lw=0.9)
        _frame(ax, (-1.05, max(1.05, right)), (-(len(v) + 0.4) * line, 1.08))
    else:
        X = 2 * r.sum() + 0.3 * len(r)
        for _ in range(4):
            ppu = _ppu(fig, X, 1.6)
            line = fs * 1.4 / ppu
            slot = np.maximum(2 * r, np.array([_tw(x, fs) for x in s.index]) / ppu) + 0.15
            X = slot.sum()
        xc = np.cumsum(slot) - slot / 2
        for i in range(len(v)):
            ax.add_patch(Wedge((xc[i], 0), r[i], 0, 180, facecolor=cols[i], edgecolor="white", lw=1.2,
                               alpha=o["alpha"]))
            ax.text(xc[i], -0.4 * line, str(s.index[i]), ha="center", va="top", fontsize=fs)
            ax.text(xc[i], r[i] + 0.2 * line, _fmt(v[i]), ha="center", va="bottom", fontsize=fs)
        ax.axhline(0, color="#666666", lw=0.9)
        _frame(ax, (0, X), (-1.6 * line, 1 + 1.4 * line))
    st.labels(ax)


@plot("square_area_030", "Four-square area", CP, number="030", example=_ex_quadrants, roles=_PIE_ROLES,
      opts=[Opt("layout", "Layout", "choice", "2 × 2 around the centre", choices=["2 × 2 around the centre", "row"]),
            Opt("sort", "Sort", "choice", "none", choices=["none", "descending", "ascending"]),
            Opt("gap", "Gap", "float", 0.02, lo=0, hi=0.3, step=0.01)],
      desc="Two columns: a category and a positive value (best with 4 rows); each value is a square whose "
           "area is proportional to it, the four squares meeting at the centre.", size=(7, 6),
      axis_labels=False)
def square_area(fig, df, c, o, st):
    from matplotlib.patches import Rectangle
    s = _sorted_pos(df, c, o)
    if s.empty:
        return _nodata(fig, st)
    v = s.to_numpy(float)
    a = np.sqrt(v / v.max())
    cols = st.colors(len(v))
    fs = st.font * 0.9
    names = [str(x) for x in s.index]
    txt = [f"{nm}\n{_fmt(x)}" for nm, x in zip(names, v)]
    g = float(o["gap"])
    ax = st.bare(fig.add_subplot())
    quad = o["layout"].startswith("2") and len(v) <= 4
    if quad:
        ow = max(_tw(t, fs) for t in txt)
        xh = 1.1
        for _ in range(3):
            ppu = _ppu(fig, 2 * xh, 2.3)
            xh = 1.05 + g + ow / ppu * 0.6
        sgn = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
        need_x = 1.0 + g
        for i in range(len(v)):
            sx, sy = sgn[i]
            x0 = g if sx > 0 else -g - a[i]
            y0 = g if sy > 0 else -g - a[i]
            ax.add_patch(Rectangle((x0, y0), a[i], a[i], facecolor=cols[i], edgecolor="none"))
            fits = _tw(txt[i], fs) < 0.92 * a[i] * ppu and _th(txt[i], fs) < 0.9 * a[i] * ppu
            if fits:
                ax.text(x0 + a[i] / 2, y0 + a[i] / 2, txt[i], ha="center", va="center", fontsize=fs,
                        color=_ink(cols[i]), fontweight="bold")
            else:
                xt = sx * (g + a[i] + 0.03)
                ax.text(xt, sy * (g + a[i] / 2), txt[i], ha="left" if sx > 0 else "right", va="center",
                        fontsize=fs)
                need_x = max(need_x, g + a[i] + 0.04 + _tw(txt[i], fs) / ppu)
        _frame(ax, (-need_x, need_x), (-1.03 - g, 1.03 + g))
    else:
        X = a.sum() + 0.15 * len(a)
        for _ in range(4):
            ppu = _ppu(fig, X, 1.5)
            slot = np.maximum(a, np.array([_tw(nm, fs) for nm in names]) / ppu) + 0.12
            X = slot.sum()
        xc = np.cumsum(slot) - slot / 2
        line = fs * 1.4 / ppu
        for i in range(len(v)):
            ax.add_patch(Rectangle((xc[i] - a[i] / 2, 0), a[i], a[i], facecolor=cols[i], edgecolor="none"))
            ax.text(xc[i], -0.4 * line, names[i], ha="center", va="top", fontsize=fs)
            ax.text(xc[i], a[i] + 0.2 * line, _fmt(v[i]), ha="center", va="bottom", fontsize=fs)
        _frame(ax, (0, X), (-1.6 * line, 1 + 1.4 * line))
    st.labels(ax)


# ── treemaps & packing ───────────────────────────────────────────────────────

def _worst(row, short):
    s = sum(row)
    return max(max(short * short * r / (s * s), s * s / (short * short * r)) for r in row)


def _squarify(vals, x, y, w, h):
    """Squarified treemap (Bruls et al.): *vals* (any order) scaled to fill the rectangle.

    Returns rectangles (x, y, w, h) in the order of *vals*."""
    vals = np.asarray(vals, float)
    order = np.argsort(-vals, kind="stable")
    sv = list(vals[order] * (w * h) / vals.sum())
    rects = []
    while sv:
        short = min(w, h)
        if short <= 1e-12:
            rects += [(x, y, 0.0, 0.0)] * len(sv)
            break
        row, i = [sv[0]], 1
        while i < len(sv) and _worst(row + [sv[i]], short) <= _worst(row, short):
            row.append(sv[i])
            i += 1
        s = sum(row)
        if w >= h:
            cw = s / h
            yy = y
            for v in row:
                rh = v / cw
                rects.append((x, yy, cw, rh))
                yy += rh
            x, w = x + cw, w - cw
        else:
            rh = s / w
            xx = x
            for v in row:
                rw = v / rh
                rects.append((xx, y, rw, rh))
                xx += rw
            y, h = y + rh, h - rh
        sv = sv[i:]
    out = [None] * len(vals)
    for k, idx in enumerate(order):
        out[idx] = rects[k]
    return out


def _hier(df, c):
    """label / value / optional group → DataFrame(l, v, g) with positive values."""
    d = pd.DataFrame({"l": as_str(df, c["label"]), "v": num(df, c["value"]),
                      "g": as_str(df, c["group"]) if c.get("group") else ""}).dropna()
    d = d[np.isfinite(d["v"]) & (d["v"] > 0)]
    return d.groupby(["g", "l"], sort=False, as_index=False)["v"].sum()


def _wrap(name) -> str:
    """Split a long name into two lines at the space nearest its middle."""
    name = str(name)
    sp = [i for i, ch in enumerate(name) if ch in " -_"]
    if len(name) < 10 or not sp:
        return name
    i = min(sp, key=lambda k: abs(k - len(name) / 2))
    return name[: i + (name[i] == "-")].rstrip() + "\n" + name[i + 1:].lstrip()


def _label_opts(name, value, show_value):
    out = [f"{name}\n{_fmt(value)}", f"{_wrap(name)}\n{_fmt(value)}"] if show_value else []
    return out + [str(name), _wrap(name)]


def _fit_text(lines_opts, w_pt, h_pt, fmax, fmin=5.5):
    """First text in *lines_opts* that fits a w × h box (points), with its font size."""
    for t in lines_opts:
        fs = min(fmax, 0.9 * w_pt / max(_tw(t, 1.0), 1e-9), 0.85 * h_pt / _th(t, 1.0))
        if fs >= fmin:
            return t, fs
    return None, 0


_HIER_ROLES = [Role("label", "Items", "cat"), Role("value", "Values", "num"),
               Role("group", "Groups (optional)", "cat", optional=True)]
_HIER_DESC = ("Columns: item name (e.g. pathway), positive value (e.g. gene count) and an optional group "
              "(e.g. KEGG class); areas are proportional to the values.")


@plot("treemap_063", "Treemap", CP, number="063", example=_ex_pathways, roles=_HIER_ROLES,
      opts=[Opt("values", "Show values", "bool", True),
            Opt("headers", "Group names as headers", "bool", True),
            Opt("pad", "Padding between groups (pt)", "float", 3, lo=0, hi=20, step=1)],
      desc=_HIER_DESC, size=(9, 5.5), axis_labels=False)
def treemap(fig, df, c, o, st):
    from matplotlib.patches import Patch, Rectangle
    d = _hier(df, c)
    if d.empty:
        return _nodata(fig, st)
    fw, fh = fig.get_size_inches()
    W, H = fw * 0.95, fh * (0.86 if (st.opts.get("title")) else 0.93)
    if st.legend_mode == "outside right":
        W *= 0.8
    ppu = 72.0
    ax = st.bare(fig.add_subplot())
    groups = list(dict.fromkeys(d["g"]))
    grouped = bool(c.get("group")) and len(groups) >= 1
    gcol = st.colors(len(groups)) if grouped else None
    icol = st.colors(len(d))
    gs = d.groupby("g", sort=False)["v"].sum()
    grects = _squarify(gs.to_numpy(), 0, 0, W, H) if grouped else [(0, 0, W, H)]
    pad = float(o["pad"]) / ppu if grouped else 0.0
    hfs = st.font * 0.9
    k = 0
    for gi, (g, (gx, gy, gw, gh)) in enumerate(zip(gs.index if grouped else [""], grects)):
        items = d[d["g"] == g] if grouped else d
        ix, iy, iw, ih = gx + pad, gy + pad, gw - 2 * pad, gh - 2 * pad
        hdr = 0.0
        if grouped and o["headers"] and ih * ppu > 3.5 * hfs and _tw(g, hfs * 0.8) < iw * ppu:
            hdr = hfs * 1.5 / ppu
            ax.add_patch(Rectangle((ix, iy), iw, hdr, facecolor=_shade(gcol[gi], -0.15), edgecolor="none"))
            t, f = _fit_text([g], iw * ppu - 6, hdr * ppu, hfs)
            if t:
                ax.text(ix + 3 / ppu, iy + hdr / 2, t, ha="left", va="center", fontsize=f, fontweight="bold",
                        color=_ink(_shade(gcol[gi], -0.15)))
        if iw <= 0 or ih - hdr <= 0:
            continue
        rects = _squarify(items["v"].to_numpy(), ix, iy + hdr, iw, ih - hdr)
        ks = np.linspace(0, 0.45, len(items)) if len(items) > 1 else [0.0]
        rank = np.argsort(np.argsort(-items["v"].to_numpy()))
        for j, ((_, row), (x, y, w, h)) in enumerate(zip(items.iterrows(), rects)):
            col = _shade(gcol[gi], ks[rank[j]]) if grouped else icol[k]
            k += 1
            ax.add_patch(Rectangle((x, y), w, h, facecolor=col, edgecolor="white", lw=1.2))
            opts = _label_opts(row["l"], row["v"], o["values"])
            t, f = _fit_text(opts, w * ppu - 4, h * ppu - 4, st.font * 1.3)
            if t:
                ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=f, color=_ink(col),
                        linespacing=1.15)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    st.labels(ax)
    if grouped and not o["headers"]:
        _side_legend(ax, st, [Patch(facecolor=cc) for cc in gcol], groups, title=c["group"])


# Voronoi treemap: iterative power-diagram (additively weighted Voronoi) adjustment inside a convex region.

_VCACHE: dict = {}


def _area_centroid(P):
    if len(P) < 3:
        return 0.0, (P.mean(0) if len(P) else np.zeros(2))
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2
    if abs(A) < 1e-14:
        return 0.0, P.mean(0)
    return abs(A), np.array([((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)])


def _clip(P, a, b):
    """Convex polygon P ∩ {x : a·x ≤ b}."""
    if len(P) == 0:
        return P
    d = P @ a - b
    ins = d <= 0
    if ins.all():
        return P
    if not ins.any():
        return P[:0]
    m = len(P)
    cross = np.nonzero(ins != np.roll(ins, -1))[0]

    def X(i):
        j = (i + 1) % m
        t = d[i] / (d[i] - d[j])
        return P[i] + t * (P[j] - P[i])
    if len(cross) != 2:
        out = []
        for i in range(m):
            if ins[i]:
                out.append(P[i])
            if ins[i] != ins[(i + 1) % m]:
                out.append(X(i))
        return np.array(out)
    i_out = cross[ins[cross]][0]
    i_in = cross[~ins[cross]][0]
    k = (i_in + 1) % m
    cnt = (i_out - k) % m + 1
    return np.vstack([X(i_in), np.roll(P, -k, axis=0)[:cnt], X(i_out)])


def _power_cells(P, w, region):
    sq = (P ** 2).sum(1)
    cells = []
    for i in range(len(P)):
        C = region
        for j in np.argsort(((P - P[i]) ** 2).sum(1))[1:]:
            C = _clip(C, 2 * (P[j] - P[i]), sq[j] - sq[i] - w[j] + w[i])
            if len(C) == 0:
                break
        cells.append(C)
    return cells


def _inside(P, pts):
    """Points inside the convex counter-clockwise polygon P."""
    e = np.roll(P, -1, 0) - P
    rel = pts[:, None, :] - P[None, :, :]
    return ((e[None, :, 0] * rel[:, :, 1] - e[None, :, 1] * rel[:, :, 0]) >= 0).all(1)


def _voronoi(region, fracs, seed, iters=70):
    n = len(fracs)
    if n == 1:
        return [region]
    A, _ = _area_centroid(region)
    r = rng(seed)
    lo, hi = region.min(0), region.max(0)
    pts = np.zeros((0, 2))
    while len(pts) < n:
        cand = lo + r.random((4 * n, 2)) * (hi - lo)
        pts = np.vstack([pts, cand[_inside(region, cand)]])
    pts = pts[:n]
    # big items start near the middle, small ones towards the edge
    _, cen = _area_centroid(region)
    order = np.argsort(-np.asarray(fracs))
    dist = np.argsort(((pts - cen) ** 2).sum(1))
    pts = pts[dist][np.argsort(order)]
    w = np.zeros(n)
    target = np.asarray(fracs) * A
    best, best_err = None, np.inf
    for it in range(iters):
        cells = _power_cells(pts, w, region)
        ac = [_area_centroid(C) for C in cells]
        areas = np.array([a for a, _ in ac])
        err = np.abs(areas - target).sum() / A
        if err < best_err:
            best, best_err = cells, err
        if err < 0.01:
            break
        for i, (a, cc) in enumerate(ac):
            if a > 0:
                pts[i] = cc
        w = w + 0.9 * (target - areas)
        D2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(D2, np.inf)
        for _ in range(3):                          # keep every site outside the others' power circles
            w = np.minimum(w, (w[None, :] + 0.9 * D2).min(1))
        w -= w.max()
    return best


def _voronoi_layout(d, grouped, seed):
    key = (tuple(d["g"]), tuple(d["l"]), tuple(np.round(d["v"].to_numpy(), 9)), grouped, seed)
    if key in _VCACHE:
        return _VCACHE[key]
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    out = []      # (group index, row index, polygon)
    if grouped:
        gs = d.groupby("g", sort=False)["v"].sum()
        gcells = _voronoi(circle, (gs / gs.sum()).to_numpy(), seed)
        for gi, (g, cell) in enumerate(zip(gs.index, gcells)):
            idx = np.nonzero((d["g"] == g).to_numpy())[0]
            v = d["v"].to_numpy()[idx]
            for j, poly in zip(idx, _voronoi(cell, v / v.sum(), seed + gi + 1)):
                out.append((gi, j, poly))
    else:
        v = d["v"].to_numpy()
        for j, poly in enumerate(_voronoi(circle, v / v.sum(), seed)):
            out.append((0, j, poly))
    if len(_VCACHE) > 32:
        _VCACHE.clear()
    _VCACHE[key] = (out, gcells if grouped else None)
    return _VCACHE[key]


@plot("voronoi_treemap_079", "Voronoi treemap", CP, number="079", example=_ex_pathways, roles=_HIER_ROLES,
      opts=[Opt("values", "Show values", "bool", True),
            Opt("seed", "Layout seed (try another arrangement)", "int", 1, lo=0, hi=999)],
      desc=_HIER_DESC + " Cells are weighted Voronoi (power-diagram) regions inside a circle.", size=(8, 6.5),
      axis_labels=False)
def voronoi_treemap(fig, df, c, o, st):
    from matplotlib.patches import Circle, Patch, Polygon
    d = _hier(df, c).reset_index(drop=True)
    if d.empty:
        return _nodata(fig, st)
    groups = list(dict.fromkeys(d["g"]))
    grouped = bool(c.get("group"))
    cells, gcells = _voronoi_layout(d, grouped, int(o["seed"]))
    gcol = st.colors(len(groups))
    icol = st.colors(len(d))
    ax = st.bare(fig.add_subplot())
    ppu = _ppu(fig, 2.1, 2.1, wfrac=0.7)
    for gi, j, poly in cells:
        if len(poly) < 3:
            continue
        if grouped:
            vv = d.loc[d["g"] == groups[gi], "v"]
            rank = int((vv > d.at[j, "v"]).sum())
            col = _shade(gcol[gi], 0.5 * rank / max(len(vv), 2))
        else:
            col = icol[j]
        ax.add_patch(Polygon(poly, closed=True, facecolor=col, edgecolor="white", lw=1.2, joinstyle="round"))
        a, cc = _area_centroid(poly)
        size = np.sqrt(a) * ppu * 0.85
        opts = _label_opts(d.at[j, "l"], d.at[j, "v"], o["values"])
        t, f = _fit_text(opts, size, size * 0.8, st.font * 1.1)
        if t:
            ax.text(cc[0], cc[1], t, ha="center", va="center", fontsize=f, color=_ink(col), linespacing=1.1)
    if grouped:
        for cell in gcells:
            if len(cell) >= 3:
                ax.add_patch(Polygon(cell, closed=True, facecolor="none", edgecolor="white", lw=3.5))
    ax.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor="#555555", lw=1.2))
    _frame(ax, (-1.03, 1.03), (-1.03, 1.03))
    st.labels(ax)
    if grouped:
        _side_legend(ax, st, [Patch(facecolor=cc) for cc in gcol], groups, title=c["group"])


def _pack(radii):
    """Greedy front packing: each circle tangent to two placed ones, as close to their centroid as possible."""
    r = np.asarray(radii, float)
    n = len(r)
    P = np.zeros((n, 2))
    if n >= 2:
        P[1] = [r[0] + r[1], 0]
    for k in range(2, n):
        I, J = np.triu_indices(k, 1)
        a, b = P[I], P[J]
        ra, rb = r[I] + r[k], r[J] + r[k]
        dv = b - a
        dd = np.hypot(dv[:, 0], dv[:, 1])
        ok = (dd > 1e-12) & (dd <= ra + rb) & (dd >= np.abs(ra - rb))
        a, dv, dd, ra, rb = a[ok], dv[ok], dd[ok], ra[ok], rb[ok]
        x = (ra ** 2 - rb ** 2 + dd ** 2) / (2 * dd)
        h = np.sqrt(np.clip(ra ** 2 - x ** 2, 0, None))
        base = a + dv * (x / dd)[:, None]
        perp = np.column_stack([-dv[:, 1], dv[:, 0]]) / dd[:, None]
        cand = np.vstack([base + perp * h[:, None], base - perp * h[:, None]])
        if len(cand):
            D = np.hypot(cand[:, None, 0] - P[None, :k, 0], cand[:, None, 1] - P[None, :k, 1])
            cand = cand[(D >= r[None, :k] + r[k] - 1e-9).all(1)]
        wts = r[:k] ** 2
        cen = (P[:k] * wts[:, None]).sum(0) / wts.sum()
        if len(cand):
            P[k] = cand[np.argmin(((cand - cen) ** 2).sum(1))]
        else:
            P[k] = [P[:k, 0].max() + r[:k].max() + r[k], 0]
    return P


def _enclose(P, r):
    best = None
    for c0 in (P.mean(0), (P.min(0) + P.max(0)) / 2,
               ((P - r[:, None]).min(0) + (P + r[:, None]).max(0)) / 2):
        R = (np.hypot(*(P - c0).T) + r).max()
        if best is None or R < best[1]:
            best = (c0, R)
    return best


@plot("circle_packing_129", "Circle packing", CP, number="129", example=_ex_taxa, roles=_HIER_ROLES,
      opts=[Opt("values", "Show values", "bool", True),
            Opt("pad", "Spacing between circles", "float", 0.03, lo=0, hi=0.3, step=0.01)],
      desc="Columns: item (e.g. genus), positive value (e.g. abundance) and an optional group (e.g. phylum); "
           "circle areas are proportional to the values, groups are packed inside an outer circle.",
      size=(8, 6.5), axis_labels=False)
def circle_packing(fig, df, c, o, st):
    from matplotlib.patches import Circle, Patch
    d = _hier(df, c).reset_index(drop=True)
    if d.empty:
        return _nodata(fig, st)
    grouped = bool(c.get("group"))
    groups = list(dict.fromkeys(d["g"]))
    gcol = st.colors(len(groups))
    icol = st.colors(len(d))
    r = np.sqrt(d["v"].to_numpy() / d["v"].max())
    pad = float(o["pad"])
    pos = np.zeros((len(d), 2))
    gcirc = []
    for gi, g in enumerate(groups):
        idx = np.nonzero((d["g"] == g).to_numpy())[0]
        idx = idx[np.argsort(-r[idx])]
        P = _pack(r[idx] + pad)
        c0, R = _enclose(P, r[idx] + pad)
        pos[idx] = P - c0
        gcirc.append(R + 2 * pad + 0.04)
    gcirc = np.array(gcirc)
    if grouped and len(groups) > 1:
        go = np.argsort(-gcirc)
        GP = np.zeros((len(groups), 2))
        GP[go] = _pack(gcirc[go] + pad)
        c0, Rall = _enclose(GP, gcirc)
        GP -= c0
    else:
        GP = np.zeros((len(groups), 2))
        Rall = gcirc.max()
    ax = st.bare(fig.add_subplot())
    ppu = _ppu(fig, 2 * Rall, 2 * Rall, wfrac=0.7 if grouped else 0.9)
    if grouped and len(groups) > 1:
        for gi in range(len(groups)):
            ax.add_patch(Circle(GP[gi], gcirc[gi], facecolor=_shade(gcol[gi], 0.85), edgecolor=gcol[gi],
                                lw=1.2))
    for i in range(len(d)):
        gi = groups.index(d.at[i, "g"])
        center = GP[gi] + pos[i]
        col = gcol[gi] if grouped else icol[i]
        ax.add_patch(Circle(center, r[i], facecolor=col, edgecolor="white", lw=1.0, alpha=0.95))
        opts = _label_opts(d.at[i, "l"], d.at[i, "v"], o["values"])
        t, f = _fit_text(opts, 1.55 * r[i] * ppu, 1.3 * r[i] * ppu, st.font * 1.1)
        if t:
            ax.text(*center, t, ha="center", va="center", fontsize=f, color=_ink(col), linespacing=1.1)
    _frame(ax, (-Rall * 1.02, Rall * 1.02), (-Rall * 1.02, Rall * 1.02))
    st.labels(ax)
    if grouped:
        _side_legend(ax, st, [Patch(facecolor=cc) for cc in gcol], groups, title=c["group"])


@plot("wordcloud_118", "Word cloud", CP, number="118", example=_ex_words,
      roles=[Role("word", "Words / terms", "cat"), Role("freq", "Frequency / weight", "num")],
      opts=[Opt("max_words", "Show at most (words)", "int", 80, lo=1, hi=400),
            Opt("scale", "Size scaling", "choice", "sqrt", choices=["linear", "sqrt", "log"]),
            Opt("vertical", "Share of vertical words", "float", 0.2, lo=0, hi=1, step=0.05),
            Opt("ratio", "Width / height of the cloud", "float", 1.6, lo=0.5, hi=4, step=0.1),
            Opt("colour_by", "Colour", "choice", "rank", choices=["rank", "random"]),
            Opt("seed", "Layout seed", "int", 1, lo=0, hi=999)],
      desc="Two columns: a word (or term) and its frequency / weight; size ∝ frequency, no overlaps.",
      size=(8, 5.5), axis_labels=False)
def wordcloud(fig, df, c, o, st):
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D
    s = _agg(df, c["word"], c["freq"]).sort_values(ascending=False).head(int(o["max_words"]))
    if s.empty:
        return _nodata(fig, st)
    v = s.to_numpy(float)
    f = {"linear": v, "sqrt": np.sqrt(v), "log": np.log1p(v)}[o["scale"]]
    span = f.max() - f.min()
    sizes = 10 + 50 * ((f - f.min()) / span if span > 0 else np.ones_like(f))
    r = rng(int(o["seed"]))
    vert = r.random(len(s)) < o["vertical"]
    vert[0] = False
    cols = st.colors(min(len(s), len(st.palette)))
    fp = FontProperties(family="sans-serif", weight="bold")
    ratio = float(o["ratio"])
    boxes = np.zeros((0, 4))           # x0, y0, x1, y1
    patches = []
    t = np.linspace(0, 1, 2500)
    for i, (word, size) in enumerate(zip(s.index, sizes)):
        tp = TextPath((0, 0), str(word), size=size, prop=fp)
        vx = tp.vertices
        (ex0, ey0), (ex1, ey1) = vx.min(0), vx.max(0)        # control-point box: fast, slightly generous
        w, h = ex1 - ex0, ey1 - ey0
        tr = Affine2D().translate(-(ex0 + w / 2), -(ey0 + h / 2))
        if vert[i]:
            tr = tr.rotate_deg(90)
            w, h = h, w
        pad = 0.06 * size
        placed = None
        step = 1.5
        k0 = 0
        while placed is None:
            th = (k0 + t * 2500) * 0.12
            rad = step * th / (2 * np.pi) * 2.2
            phase = r.random() * 2 * np.pi if i == 0 else 0.0
            cx = rad * np.cos(th + phase) * np.sqrt(ratio)
            cy = rad * np.sin(th + phase) / np.sqrt(ratio)
            if len(boxes):
                hit = ((cx[:, None] - w / 2 - pad < boxes[None, :, 2]) & (cx[:, None] + w / 2 + pad > boxes[None, :, 0])
                       & (cy[:, None] - h / 2 - pad < boxes[None, :, 3]) & (cy[:, None] + h / 2 + pad > boxes[None, :, 1]))
                free = np.nonzero(~hit.any(1))[0]
            else:
                free = np.array([0])
            if len(free):
                j = free[0]
                placed = (cx[j], cy[j])
            k0 += 2500
            if k0 > 2500 * 40:
                placed = (0.0, boxes[:, 3].max() + h) if len(boxes) else (0.0, 0.0)
        x, y = placed
        boxes = np.vstack([boxes, [x - w / 2, y - h / 2, x + w / 2, y + h / 2]])
        col = cols[i % len(cols)] if o["colour_by"] == "rank" else cols[int(r.integers(len(cols)))]
        path = (tr + Affine2D().translate(x, y)).transform_path(tp)
        patches.append(PathPatch(path, facecolor=col, edgecolor="none"))
    from matplotlib.collections import PatchCollection
    ax = st.bare(fig.add_subplot())
    ax.add_collection(PatchCollection(patches, match_original=True))
    x0, y0 = boxes[:, :2].min(0)
    x1, y1 = boxes[:, 2:].max(0)
    m = 0.03 * max(x1 - x0, y1 - y0)
    _frame(ax, (x0 - m, x1 + m), (y0 - m, y1 + m))
    st.labels(ax)


@plot("flower_039", "Flower plot", CP, number="039", example=_ex_flower,
      roles=[Role("sample", "Samples (one petal each)", "cat"), Role("unique", "Unique count", "num"),
             Role("core", "Core / shared count (optional)", "num", optional=True)],
      opts=[Opt("core_value", "Core count (if no core column)", "float", 0, lo=0, hi=1e12),
            Opt("core_text", "Centre label", "str", "Core"),
            Opt("alpha", "Petal opacity", "float", 0.6, lo=0.1, hi=1, step=0.05),
            Opt("width", "Petal width", "float", 1.0, lo=0.2, hi=3, step=0.1),
            Opt("start", "First petal at (°)", "float", 90, lo=-360, hi=360, step=5)],
      desc="Columns: sample name, number of elements unique to it (e.g. genes), and optionally the core count "
           "shared by all samples (first value used) — otherwise set it as an option.", size=(7, 7),
      axis_labels=False)
def flower(fig, df, c, o, st):
    from matplotlib.patches import Circle, Ellipse
    s = _agg(df, c["sample"], c["unique"], positive=False)
    if s.empty:
        return _nodata(fig, st)
    core = o["core_value"]
    if c.get("core"):
        cv = num(df, c["core"]).dropna()
        if len(cv):
            core = float(cv.iloc[0])
    n = len(s)
    cols = st.colors(n)
    fs = st.font * 0.85
    L = 1.0                                   # petal length (radius of the petal's far end ≈ 2 L)
    hw = min(0.9, 2 * np.pi * 0.95 / n * 0.9) * o["width"]
    ax = st.bare(fig.add_subplot())
    names = [str(x) for x in s.index]
    wmax = max(_tw(nm, fs) for nm in names)
    lim = 2.3
    for _ in range(3):
        ppu = _ppu(fig, 2 * lim, 2 * lim)
        lim = 2.15 + wmax / ppu
    for i in range(n):
        ang = o["start"] - 360.0 * i / n
        a = np.deg2rad(ang)
        ax.add_patch(Ellipse((L * np.cos(a), L * np.sin(a)), 2 * L, hw, angle=ang, facecolor=cols[i],
                             edgecolor="white", lw=0.8, alpha=o["alpha"]))
        ax.text(1.45 * L * np.cos(a), 1.45 * L * np.sin(a), _fmt(s.iloc[i]), ha="center", va="center",
                fontsize=fs, fontweight="bold", color="#222222")
        ca = np.cos(a)
        rot = ang if ca >= 0 else ang - 180
        rot = ((rot + 180) % 360) - 180
        if n <= 12:
            rot = 0
        ax.text(2.1 * L * np.cos(a), 2.1 * L * np.sin(a), names[i], ha="left" if ca > 0.05 else
                ("right" if ca < -0.05 else "center"), va="center" if abs(ca) > 0.05 or n > 6 else
                ("bottom" if np.sin(a) > 0 else "top"), rotation=rot, rotation_mode="anchor", fontsize=fs)
    ax.add_patch(Circle((0, 0), 0.5 * L, facecolor="white", edgecolor="#555555", lw=1.2, zorder=3))
    ax.text(0, 0, "\n".join(x for x in (o["core_text"], _fmt(core)) if x), ha="center", va="center",
            fontsize=st.font, fontweight="bold", zorder=4, linespacing=1.2)
    _frame(ax, (-lim, lim), (-lim, lim))
    st.labels(ax)
