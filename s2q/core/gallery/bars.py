"""Bar charts: simple, grouped, stacked, error bars, lollipops, dual axes, waterfall, bullet…"""
from __future__ import annotations

import numpy as np
import pandas as pd

from s2q.core.gallery.base import Opt, Role, as_str, cats, num, plot, rng

CAT = "Bar charts"


# ── example tables ───────────────────────────────────────────────────────────

def _ex_groups_values():
    r = rng(1)
    groups = ["Control", "Drug A", "Drug B", "Drug C", "Drug D"]
    return pd.DataFrame({"Group": groups, "Value": np.round(r.uniform(20, 90, len(groups)), 1)})


def _ex_replicates():
    r = rng(2)
    rows = []
    for g, mu in zip(["Control", "Drug A", "Drug B", "Drug C"], [40, 55, 70, 48]):
        for _ in range(6):
            rows.append((g, round(float(r.normal(mu, 6)), 2)))
    return pd.DataFrame(rows, columns=["Group", "Value"])


# ── plots ────────────────────────────────────────────────────────────────────

@plot("vertical_bars", "Vertical bars", CAT, number="073", example=_ex_groups_values,
      roles=[Role("x", "Categories", "cat"), Role("y", "Values", "num")],
      opts=[Opt("labels", "Show values on bars", "bool", True), Opt("sort", "Sort", "choice", "none",
                                                                     choices=["none", "ascending", "descending"]),
            Opt("width", "Bar width", "float", 0.7, lo=0.1, hi=1.0, step=0.05)],
      desc="One bar per category; height = value.")
def vertical_bars(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "y": num(df, c["y"])}).dropna()
    if o["sort"] != "none":
        d = d.sort_values("y", ascending=o["sort"] == "ascending")
    ax = st.ax(fig.add_subplot(), "y")
    bars = ax.bar(d["x"], d["y"], width=o["width"], color=st.colors(len(d)), edgecolor="black", linewidth=0.6)
    if o["labels"]:
        ax.bar_label(bars, fmt="%.4g", padding=2, fontsize=st.font * 0.8)
    ax.tick_params(axis="x", rotation=45 if len(d) > 6 else 0)
    st.labels(ax, c["x"], c["y"])


@plot("errorbar_sd", "Error bars (mean ± SD / SEM)", CAT, number="023", example=_ex_replicates,
      roles=[Role("group", "Groups", "cat"), Role("value", "Values (one row per replicate)", "num")],
      opts=[Opt("err", "Error", "choice", "SD", choices=["SD", "SEM", "95% CI"]),
            Opt("points", "Show the replicates", "bool", True)],
      desc="Bar = mean of the replicates of each group, whisker = SD, SEM or 95 % CI.")
def errorbar_sd(fig, df, c, o, st):
    from scipy import stats
    d = pd.DataFrame({"g": as_str(df, c["group"]), "v": num(df, c["value"])}).dropna()
    groups = cats(d, "g")
    ax = st.ax(fig.add_subplot(), "y")
    cols = st.colors(len(groups))
    for i, g in enumerate(groups):
        v = d.loc[d["g"] == g, "v"].to_numpy()
        m, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
        e = {"SD": sd, "SEM": sd / np.sqrt(len(v)),
             "95% CI": stats.t.ppf(0.975, max(len(v) - 1, 1)) * sd / np.sqrt(len(v))}[o["err"]]
        ax.bar(i, m, color=cols[i], edgecolor="black", linewidth=0.6, width=0.65, alpha=0.9)
        ax.errorbar(i, m, yerr=e, color="black", capsize=6, linewidth=1.2)
        if o["points"]:
            jit = rng(i).uniform(-0.15, 0.15, len(v))
            ax.scatter(i + jit, v, s=16, color="white", edgecolor="black", linewidth=0.6, zorder=3)
    ax.set_xticks(range(len(groups)), groups)
    st.labels(ax, c["group"], f"{c['value']} (mean ± {o['err']})")


# ── shared helpers ───────────────────────────────────────────────────────────

_SORT = ["none", "ascending", "descending"]


def _fmt(v) -> str:
    """Compact number for value labels."""
    if v is None or not np.isfinite(v):
        return ""
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


def _nodata(fig, st, msg="No numeric data in the selected columns"):
    ax = st.bare(fig.add_subplot())
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=st.font, color="#777777")
    return ax


def _xcat(ax, pos, labels, st, axis="x"):
    """Category tick labels; rotated when there are many or they are long."""
    labels = [str(s) for s in labels]
    if axis == "y":
        ax.set_yticks(pos, labels)
        return
    ax.set_xticks(pos, labels)
    longest = max((len(s) for s in labels), default=0)
    if len(labels) > 7 or len(labels) * longest > 55:
        for t in ax.get_xticklabels():
            t.set_rotation(40)
            t.set_ha("right")
            t.set_rotation_mode("anchor")


def _side_legend(ax, st, handles=None, labels=None, title=None, **kw):
    """Legend that, in 'best' mode, sits right of the axes (for plots that fill the whole axes)."""
    if st.legend_mode == "best":
        kw.setdefault("loc", "upper left")
        kw.setdefault("bbox_to_anchor", (1.01, 1))
    return st.legend(ax, handles, labels, title=title, **kw)


def _sorted(d, col, how):
    if how == "ascending":
        return d.sort_values(col, ascending=True, kind="stable")
    if how == "descending":
        return d.sort_values(col, ascending=False, kind="stable")
    return d


def _agg_err(v, how):
    from scipy import stats
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, 0.0
    m = v.mean()
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    if how == "SEM":
        return m, sd / np.sqrt(len(v))
    if how == "95% CI":
        return m, (stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))) if len(v) > 1 else 0.0
    if how == "none":
        return m, 0.0
    return m, sd


def _patch(color, label, **kw):
    from matplotlib.patches import Patch
    kw.setdefault("edgecolor", "black")
    kw.setdefault("linewidth", 0.6)
    return Patch(facecolor=color, label=label, **kw)


def _is_num_col(df, col):
    s = num(df, col)
    return s.notna().sum() > 0 and s.notna().sum() >= 0.8 * df[col].notna().sum()


# ── more example tables ──────────────────────────────────────────────────────

def _ex_updown():
    return pd.DataFrame({"Comparison": ["Day1 vs Ctrl", "Day3 vs Ctrl", "Day7 vs Ctrl", "Day14 vs Ctrl",
                                        "Day28 vs Ctrl"],
                         "Up": [312, 845, 1290, 976, 540], "Down": [205, 610, 1104, 880, 397]})


def _ex_timecourse():
    r = rng(3)
    rows = []
    for g, slope in zip(["Vehicle", "Drug A", "Drug B"], [0.02, 0.09, 0.16]):
        for t in [0, 6, 12, 24, 48]:
            for _ in range(4):
                rows.append((t, round(float(1 + slope * t * (1 - t / 120) + r.normal(0, 0.25)), 3), g))
    return pd.DataFrame(rows, columns=["Time (h)", "Relative expression", "Treatment"])


def _ex_dual():
    days = [f"Day {i}" for i in range(1, 9)]
    cells = [0.5, 0.9, 1.7, 3.1, 5.2, 7.4, 8.6, 8.9]
    via = [98, 97, 96, 95, 93, 90, 84, 77]
    return pd.DataFrame({"Culture day": days, "Cell count (×10⁶)": cells, "Viability (%)": via})


def _ex_composition():
    r = rng(4)
    samples = [f"Ctrl_{i}" for i in range(1, 5)] + [f"HFD_{i}" for i in range(1, 5)]
    phyla = {"Firmicutes": (420, 610), "Bacteroidetes": (380, 210), "Proteobacteria": (60, 110),
             "Actinobacteria": (50, 40), "Other": (40, 30)}
    d = {"Sample": samples}
    for p, (a, b) in phyla.items():
        d[p] = [int(max(1, r.normal(a if i < 4 else b, 0.12 * (a if i < 4 else b)))) for i in range(8)]
    return pd.DataFrame(d)


def _ex_butterfly():
    terms = ["Cell cycle", "DNA replication", "Immune response", "Apoptosis", "Lipid metabolism",
             "Angiogenesis", "Oxidative phosphorylation", "Wnt signalling"]
    return pd.DataFrame({"Pathway": terms, "Up-regulated genes": [48, 35, 62, 27, 18, 22, 12, 30],
                         "Down-regulated genes": [12, 8, 25, 19, 41, 15, 53, 21]})


def _ex_trend():
    return pd.DataFrame({"Week": [f"Wk {i}" for i in range(0, 9)],
                         "Tumour volume (mm³)": [102, 138, 171, 230, 262, 318, 371, 405, 468]})


def _ex_go_counts():
    terms = ["T cell activation", "Response to interferon-gamma", "Leukocyte migration",
             "Cytokine-mediated signalling", "Antigen processing and presentation", "Chemotaxis",
             "Regulation of inflammatory response", "Neutrophil degranulation"]
    return pd.DataFrame({"GO term": terms, "Gene count": [58, 41, 37, 52, 29, 33, 46, 24]})


def _ex_genes_expr():
    genes = ["TP53", "MYC", "EGFR", "KRAS", "PTEN", "CDKN2A", "BRCA1", "VEGFA", "GAPDH", "ACTB"]
    vals = [6.2, 8.9, 5.4, 4.1, 3.3, 2.6, 4.8, 7.1, 11.6, 12.3]
    return pd.DataFrame({"Gene": genes, "log2(TPM + 1)": vals})


def _ex_stem():
    r = rng(5)
    pos = np.arange(1, 51) * 20
    v = np.round(r.normal(0, 0.06, len(pos)) + 0.35 * np.exp(-((pos - 400) / 120) ** 2)
                 - 0.3 * np.exp(-((pos - 820) / 80) ** 2), 3)
    return pd.DataFrame({"CpG position (bp)": pos, "Δ methylation": v})


def _ex_polar():
    tissues = ["Brain", "Liver", "Kidney", "Heart", "Lung", "Muscle", "Testis", "Spleen", "Skin", "Pancreas"]
    return pd.DataFrame({"Tissue": tissues,
                         "Tissue-enriched genes": [1450, 420, 380, 210, 260, 330, 2210, 300, 270, 190]})


def _ex_dotbars():
    r = rng(6)
    rows = []
    mu = {("Vehicle", "WT"): 20, ("Vehicle", "KO"): 22, ("Drug A", "WT"): 34, ("Drug A", "KO"): 25,
          ("Drug B", "WT"): 45, ("Drug B", "KO"): 28}
    for (t, g), m in mu.items():
        for _ in range(6):
            rows.append((t, g, round(float(r.normal(m, 4)), 2)))
    return pd.DataFrame(rows, columns=["Treatment", "Genotype", "IL-6 (pg/ml)"])


def _ex_reads():
    r = rng(7)
    samples = [f"S{i}" for i in range(1, 9)]
    d = {"Sample": samples}
    for k, m in {"Exonic": 18, "Intronic": 7, "Intergenic": 3, "rRNA": 1.2}.items():
        d[k] = np.round(r.normal(m, 0.18 * m, len(samples)).clip(0.1), 1)
    return pd.DataFrame(d)


def _ex_cleveland():
    genes = ["IL6", "TNF", "CXCL8", "CCL2", "IL1B", "SOCS3", "NFKBIA", "PTGS2", "MMP9", "SERPINE1"]
    r = rng(8)
    normal = np.round(r.uniform(2, 6, len(genes)), 2)
    tumour = np.round(normal + r.uniform(-1, 4, len(genes)), 2)
    return pd.DataFrame({"Gene": genes, "Normal": normal, "Tumour": tumour})


def _ex_bullet():
    return pd.DataFrame({"QC metric": ["Read depth (M)", "Mapping rate (%)", "Genes detected (k)",
                                       "Unique reads (%)", "RIN"],
                         "Actual": [27, 91, 15.8, 72, 8.1], "Target": [30, 90, 16, 80, 8],
                         "Poor": [15, 70, 10, 50, 5], "Acceptable": [25, 85, 14, 70, 7],
                         "Good": [40, 100, 20, 100, 10]})


def _ex_matrix():
    r = rng(9)
    genes = ["HIF1A", "VEGFA", "LDHA", "PGK1", "CA9", "BNIP3"]
    d = {"Gene": genes}
    for j, t in enumerate(["0 h", "6 h", "12 h", "24 h"]):
        d[t] = np.round(1 + j * r.uniform(0.3, 1.4, len(genes)), 2)
    return pd.DataFrame(d)


def _ex_pcr():
    genes = ["IL6", "TNF", "CXCL10", "ISG15", "MX1", "IFIT1", "SOCS1", "GAPDH"]
    return pd.DataFrame({"Gene": genes,
                         "qPCR fold change": [6.8, 4.2, 12.5, 9.1, 7.7, 10.3, 2.9, 1.0],
                         "RNA-seq FPKM": [48, 31, 120, 95, 70, 104, 22, 9],
                         "qPCR SD": [0.9, 0.6, 1.8, 1.2, 1.0, 1.5, 0.4, 0.1]})


def _ex_diverging():
    terms = ["Interferon alpha response", "Inflammatory response", "E2F targets", "G2M checkpoint",
             "Hypoxia", "Oxidative phosphorylation", "Fatty acid metabolism", "Adipogenesis",
             "TNF-alpha signalling via NF-kB", "Epithelial–mesenchymal transition", "Myogenesis",
             "Bile acid metabolism"]
    z = [3.4, 2.8, 2.1, 1.7, 1.2, -2.6, -2.1, -1.5, 2.5, 0.9, -0.8, -1.9]
    return pd.DataFrame({"Hallmark pathway": terms, "z-score": z})


def _ex_truncation():
    return pd.DataFrame({"Tissue": ["Liver", "Kidney", "Heart", "Lung", "Spleen", "Brain"],
                         "ALB expression (TPM)": [8650, 42, 18, 35, 27, 9],
                         "SD": [720, 6, 3, 5, 4, 2]})


def _ex_waterfall():
    return pd.DataFrame({"Step": ["Seeded cells", "Proliferation", "Apoptosis", "Differentiation",
                                  "Migration out", "Recruitment"],
                         "Change (cells)": [1000, 620, -240, -180, -95, 130]})


def _ex_dumbbell():
    genes = ["SOX2", "POU5F1", "NANOG", "KLF4", "MYC", "LIN28A", "DNMT3B", "TDGF1"]
    return pd.DataFrame({"Gene": genes, "Day 0": [9.1, 8.7, 8.2, 5.1, 6.3, 7.8, 7.2, 6.9],
                         "Day 7": [4.2, 2.9, 2.1, 6.4, 5.0, 3.3, 2.7, 1.8]})


def _ex_elements():
    r = rng(10)
    kinds = ["Missense", "Synonymous", "Nonsense", "Frameshift", "Splice site", "In-frame indel", "UTR"]
    p = np.array([0.42, 0.24, 0.07, 0.09, 0.05, 0.04, 0.09])
    return pd.DataFrame({"Variant type": r.choice(kinds, 300, p=p / p.sum()),
                         "Sample": r.choice(["T1", "T2", "T3"], 300)})


# ── 014 gene up/down ─────────────────────────────────────────────────────────

@plot("gene_up_down_bar_014", "Gene up/down bars", CAT, number="014", example=_ex_updown,
      roles=[Role("x", "Comparisons", "cat"), Role("up", "Up-regulated count", "num"),
             Role("down", "Down-regulated count", "num")],
      opts=[Opt("labels", "Show counts", "bool", True),
            Opt("up_color", "Up colour", "color", "#D6404E"),
            Opt("down_color", "Down colour", "color", "#3B6FB6"),
            Opt("width", "Bar width", "float", 0.6, lo=0.1, hi=1.0, step=0.05)],
      desc="One row per comparison with the number of up- and of down-regulated genes "
           "(down counts may be written positive or negative).")
def gene_up_down_bar(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "u": num(df, c["up"]).abs(),
                      "d": num(df, c["down"]).abs()})
    d = d[d[["u", "d"]].notna().any(axis=1)].fillna(0)
    if d.empty:
        return _nodata(fig, st)
    from matplotlib.ticker import FuncFormatter
    ax = st.ax(fig.add_subplot(), "y")
    x = np.arange(len(d))
    b1 = ax.bar(x, d["u"], o["width"], color=o["up_color"], edgecolor="black", linewidth=0.6, label="Up")
    b2 = ax.bar(x, -d["d"], o["width"], color=o["down_color"], edgecolor="black", linewidth=0.6,
                label="Down")
    ax.axhline(0, color="#333333", linewidth=0.8)
    if o["labels"]:
        ax.bar_label(b1, labels=[_fmt(v) for v in d["u"]], padding=2, fontsize=st.font * 0.8)
        ax.bar_label(b2, labels=[_fmt(v) for v in d["d"]], padding=2, fontsize=st.font * 0.8)
    m = max(d["u"].max(), d["d"].max(), 1)
    ax.set_ylim(-m * 1.18, m * 1.18)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}"))
    _xcat(ax, x, d["x"], st)
    st.legend(ax)
    st.labels(ax, c["x"], "Number of DEGs")


# ── 025 error bars (points) ──────────────────────────────────────────────────

@plot("errorbar_025", "Error bars (points & lines)", CAT, number="025", example=_ex_timecourse,
      roles=[Role("x", "X (time / dose / category)", "any"),
             Role("value", "Values (one row per replicate)", "num"),
             Role("group", "Series (optional)", "cat", optional=True)],
      opts=[Opt("err", "Error", "choice", "SD", choices=["SD", "SEM", "95% CI"]),
            Opt("line", "Connect the means", "bool", True),
            Opt("points", "Show the replicates", "bool", False),
            Opt("dodge", "Shift series apart", "float", 0.06, lo=0, hi=0.4, step=0.02),
            Opt("capsize", "Cap size", "int", 4, lo=0, hi=15)],
      desc="Long table: one row per replicate with its x (e.g. time point), value and optional series; "
           "points = mean, whiskers = SD / SEM / 95 % CI.")
def errorbar_025(fig, df, c, o, st):
    numeric_x = _is_num_col(df, c["x"])
    d = pd.DataFrame({"x": num(df, c["x"]) if numeric_x else as_str(df, c["x"]), "v": num(df, c["value"]),
                      "g": as_str(df, c["group"]) if c.get("group") else ""}).dropna()
    if d.empty:
        return _nodata(fig, st)
    ax = st.ax(fig.add_subplot(), "y")
    if numeric_x:
        xvals = list(np.sort(d["x"].unique()))
        pos = {v: float(v) for v in xvals}
        step = float(np.min(np.diff(xvals))) if len(xvals) > 1 else 1.0
    else:
        xvals = cats(d, "x")
        pos = {v: i for i, v in enumerate(xvals)}
        step = 1.0
    groups = cats(d, "g")
    cols = st.colors(len(groups))
    k = len(groups)
    for i, g in enumerate(groups):
        sub = d[d["g"] == g]
        off = (i - (k - 1) / 2) * o["dodge"] * step
        px, m, e = [], [], []
        for xv in xvals:
            vv = sub.loc[sub["x"] == xv, "v"]
            if len(vv) == 0:
                continue
            mm, ee = _agg_err(vv, o["err"])
            px.append(pos[xv] + off)
            m.append(mm)
            e.append(ee)
            if o["points"]:
                ax.scatter(np.full(len(vv), pos[xv] + off), vv, s=12, color=cols[i], alpha=0.4,
                           linewidth=0, zorder=2)
        ax.errorbar(px, m, yerr=e, fmt="o-" if o["line"] else "o", color=cols[i], capsize=o["capsize"],
                    markersize=6, linewidth=1.6, markeredgecolor="white", markeredgewidth=0.8,
                    elinewidth=1.1, label=g if c.get("group") else None, zorder=3)
    if not numeric_x:
        _xcat(ax, range(len(xvals)), xvals, st)
    if c.get("group"):
        st.legend(ax, title=c["group"])
    st.labels(ax, c["x"], f"{c['value']} (mean ± {o['err']})")


# ── 037 dual y ───────────────────────────────────────────────────────────────

def _twin(ax, color, st):
    ax2 = ax.twinx()
    for s in ("top", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color(color)
    ax2.tick_params(axis="y", colors=color, labelsize=st.font * 0.85)
    ax2.grid(False)
    return ax2


@plot("dual_y_bar_037", "Dual Y axis: bars + line", CAT, number="037", example=_ex_dual,
      roles=[Role("x", "Categories", "cat"), Role("bar", "Bar values (left axis)", "num"),
             Role("line", "Line values (right axis)", "num")],
      opts=[Opt("labels", "Show values on bars", "bool", False),
            Opt("zero", "Right axis starts at 0", "bool", False),
            Opt("width", "Bar width", "float", 0.65, lo=0.1, hi=1.0, step=0.05)],
      desc="One row per category with two measurements on different scales: "
           "the first as bars (left axis), the second as a line (right axis).")
def dual_y_bar(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "b": num(df, c["bar"]), "l": num(df, c["line"])})
    d = d[d[["b", "l"]].notna().any(axis=1)]
    if d.empty:
        return _nodata(fig, st)
    ax = st.ax(fig.add_subplot(), "y")
    x = np.arange(len(d))
    cb, cl = st.color(0), st.color(1)
    bars = ax.bar(x, d["b"].fillna(0), o["width"], color=cb, edgecolor="black", linewidth=0.6,
                  alpha=0.9, label=c["bar"])
    if o["labels"]:
        ax.bar_label(bars, labels=[_fmt(v) for v in d["b"]], padding=2, fontsize=st.font * 0.75)
    ax2 = _twin(ax, cl, st)
    ln, = ax2.plot(x, d["l"], "-o", color=cl, linewidth=2, markersize=6, markeredgecolor="white",
                   label=c["line"])
    lo_, hi_ = np.nanmin(d["l"]), np.nanmax(d["l"])
    pad = (hi_ - lo_) * 0.15 or abs(hi_) * 0.1 or 1
    ax2.set_ylim(0 if o["zero"] and lo_ >= 0 else lo_ - pad, hi_ + pad * 2.5)
    bmax = np.nanmax(d["b"])
    ax.set_ylim(min(0, np.nanmin(d["b"]) * 1.1), bmax * 1.3 if bmax > 0 else 1)
    ax2.set_ylabel(c["line"], fontsize=st.font, color=cl)
    ax.tick_params(axis="y", colors=cb)
    ax.spines["left"].set_color(cb)
    _xcat(ax, x, d["x"], st)
    st.legend(ax, [bars, ln], [c["bar"], c["line"]], loc="upper center")
    st.labels(ax, c["x"], c["bar"])
    ax.yaxis.label.set_color(cb)


# ── 038 100 % stacked ────────────────────────────────────────────────────────

@plot("filled_stackbar_038", "100 % stacked bars", CAT, number="038", example=_ex_composition,
      roles=[Role("x", "Samples / categories", "cat"), Role("parts", "Components", "num", multi=True)],
      opts=[Opt("horizontal", "Horizontal bars", "bool", False),
            Opt("labels", "Show percentages", "bool", True),
            Opt("min_label", "Hide labels below (%)", "float", 6, lo=0, hi=50, step=1),
            Opt("width", "Bar width", "float", 0.75, lo=0.1, hi=1.0, step=0.05)],
      desc="One row per sample, one numeric column per component (counts or abundances); "
           "each bar is scaled to 100 %.")
def filled_stackbar(fig, df, c, o, st):
    parts = [p for p in (c["parts"] or []) if p]
    if not parts:
        return _nodata(fig, st)
    m = pd.DataFrame({p: num(df, p) for p in parts}).clip(lower=0).fillna(0)
    lab = as_str(df, c["x"])
    tot = m.sum(axis=1)
    keep = tot > 0
    m, lab, tot = m[keep], lab[keep], tot[keep]
    if m.empty:
        return _nodata(fig, st)
    pct = m.div(tot, axis=0) * 100
    horiz = o["horizontal"]
    ax = st.ax(fig.add_subplot(), "x" if horiz else "y")
    pos = np.arange(len(pct))
    cols = st.colors(len(parts))
    base = np.zeros(len(pct))
    handles = []
    for j, p in enumerate(parts):
        v = pct[p].to_numpy()
        kw = dict(color=cols[j], edgecolor="white", linewidth=0.7, label=p)
        b = (ax.barh(pos, v, o["width"], left=base, **kw) if horiz
             else ax.bar(pos, v, o["width"], bottom=base, **kw))
        handles.append(b)
        if o["labels"]:
            ax.bar_label(b, labels=[f"{x:.0f}%" if x >= o["min_label"] else "" for x in v],
                         label_type="center", fontsize=st.font * 0.68, color="white")
        base += v
    if horiz:
        ax.set_xlim(0, 100)
        _xcat(ax, pos, lab, st, axis="y")
        ax.set_ylim(len(pct) - 0.5, -0.5)
        st.labels(ax, "Relative abundance (%)", c["x"])
        _side_legend(ax, st, handles, parts)
    else:
        ax.set_ylim(0, 100)
        _xcat(ax, pos, lab, st)
        st.labels(ax, c["x"], "Relative abundance (%)")
        _side_legend(ax, st, handles[::-1], parts[::-1])


# ── 056 butterfly ────────────────────────────────────────────────────────────

@plot("butterfly_bar_056", "Butterfly bars", CAT, number="056", example=_ex_butterfly,
      roles=[Role("x", "Categories", "cat"), Role("left", "Left values", "num"),
             Role("right", "Right values", "num")],
      opts=[Opt("labels", "Show values", "bool", True),
            Opt("sort", "Sort by", "choice", "none", choices=["none", "left", "right", "total"]),
            Opt("shared", "Same scale on both sides", "bool", True)],
      desc="One row per category with two numeric columns, drawn mirrored to the left and right "
           "of the category names.")
def butterfly_bar(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "l": num(df, c["left"]).abs(),
                      "r": num(df, c["right"]).abs()})
    d = d[d[["l", "r"]].notna().any(axis=1)].fillna(0)
    if d.empty:
        return _nodata(fig, st)
    if o["sort"] != "none":
        key = {"left": d["l"], "right": d["r"]}.get(o["sort"], d["l"] + d["r"])
        d = d.assign(_k=key).sort_values("_k", ascending=False, kind="stable")
    longest = max(len(s) for s in d["x"])
    mid = max(0.3, 0.026 * longest * st.font / 11)
    axL, axM, axR = fig.subplots(1, 3, sharey=True, gridspec_kw={"width_ratios": [1, mid, 1]})
    y = np.arange(len(d))
    cl, cr = st.color(0), st.color(1)
    for a in (axL, axR):
        st.ax(a, "x")
        a.spines["left"].set_visible(False)
        a.tick_params(axis="y", left=False, labelleft=False)
    bl = axL.barh(y, d["l"], 0.7, color=cl, edgecolor="black", linewidth=0.5)
    br = axR.barh(y, d["r"], 0.7, color=cr, edgecolor="black", linewidth=0.5)
    ml, mr = max(d["l"].max(), 1e-9), max(d["r"].max(), 1e-9)
    if o["shared"]:
        ml = mr = max(ml, mr)
    axL.set_xlim(ml * 1.2, 0)
    axR.set_xlim(0, mr * 1.2)
    if o["labels"]:
        axL.bar_label(bl, labels=[_fmt(v) for v in d["l"]], padding=2, fontsize=st.font * 0.75)
        axR.bar_label(br, labels=[_fmt(v) for v in d["r"]], padding=2, fontsize=st.font * 0.75)
    st.bare(axM)
    for yi, s in zip(y, d["x"]):
        axM.text(0.5, yi, s, ha="center", va="center", fontsize=st.font * 0.85, color="#222222",
                 transform=axM.get_yaxis_transform())
    axL.set_ylim(len(d) - 0.5, -0.5)
    axL.set_title(c["left"], fontsize=st.font, color=cl, fontweight="bold")
    axR.set_title(c["right"], fontsize=st.font, color=cr, fontweight="bold")
    if o.get("xlabel"):
        fig.supxlabel(o["xlabel"], fontsize=st.font)
    st.suptitle(fig)


# ── 061 bars + trend ─────────────────────────────────────────────────────────

@plot("vertical_bar_trend_061", "Vertical bars with trend line", CAT, number="061", example=_ex_trend,
      roles=[Role("x", "Categories (in order)", "cat"), Role("y", "Values", "num")],
      opts=[Opt("trend", "Trend line", "choice", "linear fit", choices=["linear fit", "connect tops", "none"]),
            Opt("labels", "Show values on bars", "bool", True),
            Opt("multicolor", "One colour per bar", "bool", False),
            Opt("width", "Bar width", "float", 0.65, lo=0.1, hi=1.0, step=0.05)],
      desc="One row per ordered category (e.g. time point) with a value; a line shows the trend.")
def vertical_bar_trend(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "y": num(df, c["y"])}).dropna()
    if d.empty:
        return _nodata(fig, st)
    ax = st.ax(fig.add_subplot(), "y")
    x = np.arange(len(d))
    col = st.colors(len(d)) if o["multicolor"] else st.color(0)
    bars = ax.bar(x, d["y"], o["width"], color=col, edgecolor="black", linewidth=0.6, alpha=0.85,
                  label=c["y"])
    if o["labels"]:
        ax.bar_label(bars, labels=[_fmt(v) for v in d["y"]], padding=2, fontsize=st.font * 0.75)
    tc = "#333333" if o["multicolor"] else st.color(1)
    if o["trend"] == "connect tops":
        ax.plot(x, d["y"], "-o", color=tc, linewidth=1.8, markersize=5, label="Trend", zorder=4)
    elif o["trend"] == "linear fit" and len(d) > 1:
        k, b = np.polyfit(x, d["y"], 1)
        fit = k * x + b
        ss = ((d["y"] - d["y"].mean()) ** 2).sum()
        r2 = 1 - ((d["y"] - fit) ** 2).sum() / ss if ss > 0 else 1.0
        ax.plot(x, fit, "--", color=tc, linewidth=2, label=f"Linear fit (R² = {r2:.3f})", zorder=4)
    top = d["y"].max()
    ax.set_ylim(min(0, d["y"].min() * 1.1), top * 1.15 if top > 0 else 0)
    _xcat(ax, x, d["x"], st)
    st.legend(ax, loc="upper left")
    st.labels(ax, c["x"], c["y"])


# ── 062 horizontal bars ──────────────────────────────────────────────────────

@plot("horizontal_bar_062", "Horizontal bars", CAT, number="062", example=_ex_go_counts,
      roles=[Role("x", "Categories", "cat"), Role("y", "Values", "num")],
      opts=[Opt("sort", "Sort", "choice", "descending", choices=_SORT),
            Opt("labels", "Show values", "bool", True),
            Opt("colors", "Colours", "choice", "gradient (colour map)",
                choices=["single", "palette", "gradient (colour map)"]),
            Opt("width", "Bar height", "float", 0.7, lo=0.1, hi=1.0, step=0.05)],
      desc="One row per category with a value; long category names stay readable on the y axis.")
def horizontal_bar(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "y": num(df, c["y"])}).dropna()
    if d.empty:
        return _nodata(fig, st)
    d = _sorted(d, "y", o["sort"])
    ax = st.ax(fig.add_subplot(), "x")
    y = np.arange(len(d))
    if o["colors"] == "palette":
        col = st.colors(len(d))
    elif o["colors"].startswith("gradient"):
        from matplotlib import colormaps
        from matplotlib.colors import Normalize
        cm = colormaps[st.cmap]
        lo_, hi_ = d["y"].min(), d["y"].max()
        nv = Normalize(lo_, hi_ if hi_ > lo_ else lo_ + 1)
        col = [cm(0.15 + 0.7 * nv(v)) for v in d["y"]]
    else:
        col = st.color(0)
    bars = ax.barh(y, d["y"], o["width"], color=col, edgecolor="black", linewidth=0.5)
    if o["labels"]:
        ax.bar_label(bars, labels=[_fmt(v) for v in d["y"]], padding=3, fontsize=st.font * 0.75)
    ax.set_yticks(y, d["x"])
    ax.set_ylim(len(d) - 0.5, -0.5)
    if d["y"].max() > 0:
        ax.set_xlim(min(0, d["y"].min() * 1.15), d["y"].max() * 1.12)
    st.labels(ax, c["y"], c["x"])


# ── 065 / 066 lollipops ──────────────────────────────────────────────────────

_LOLLI_OPTS = [Opt("sort", "Sort", "choice", "descending", choices=_SORT),
               Opt("labels", "Show values", "bool", False),
               Opt("colors", "Colours", "choice", "single", choices=["single", "palette", "by sign"]),
               Opt("size", "Dot size", "float", 9, lo=2, hi=25, step=1),
               Opt("baseline", "Baseline", "float", 0.0, lo=-1e9, hi=1e9, step=1)]


def _lollipop(fig, df, c, o, st, horizontal):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "y": num(df, c["y"])}).dropna()
    if d.empty:
        return _nodata(fig, st)
    d = _sorted(d, "y", o["sort"])
    base = float(o["baseline"])
    if o["colors"] == "palette":
        col = st.colors(len(d))
    elif o["colors"] == "by sign":
        col = [st.color(0) if v >= base else st.color(1) for v in d["y"]]
    else:
        col = [st.color(0)] * len(d)
    ax = st.ax(fig.add_subplot(), "x" if horizontal else "y")
    p = np.arange(len(d))
    v = d["y"].to_numpy()
    if horizontal:
        ax.hlines(p, base, v, color=col, linewidth=2, alpha=0.8)
        ax.scatter(v, p, s=o["size"] ** 2, color=col, edgecolor="black", linewidth=0.6, zorder=3)
        ax.axvline(base, color="#555555", linewidth=0.8)
        ax.set_yticks(p, d["x"])
        ax.set_ylim(len(d) - 0.5, -0.5)
    else:
        ax.vlines(p, base, v, color=col, linewidth=2, alpha=0.8)
        ax.scatter(p, v, s=o["size"] ** 2, color=col, edgecolor="black", linewidth=0.6, zorder=3)
        ax.axhline(base, color="#555555", linewidth=0.8)
        _xcat(ax, p, d["x"], st)
        ax.set_xlim(-0.6, len(d) - 0.4)
    lo_, hi_ = min(v.min(), base), max(v.max(), base)
    pad = (hi_ - lo_) * (0.14 if o["labels"] else 0.08) or 1
    (ax.set_xlim if horizontal else ax.set_ylim)(lo_ - (pad if lo_ < base else 0), hi_ + pad)
    if o["labels"]:
        off = 0.5 * o["size"] + 3
        for pi, vi in zip(p, v):
            up = vi >= base
            if horizontal:
                ax.annotate(_fmt(vi), (vi, pi), xytext=(off if up else -off, 0), textcoords="offset points",
                            ha="left" if up else "right", va="center", fontsize=st.font * 0.75)
            else:
                ax.annotate(_fmt(vi), (pi, vi), xytext=(0, off if up else -off), textcoords="offset points",
                            ha="center", va="bottom" if up else "top", fontsize=st.font * 0.75)
    if horizontal:
        st.labels(ax, c["y"], c["x"])
    else:
        st.labels(ax, c["x"], c["y"])


@plot("vertical_lollipop_065", "Vertical lollipop chart", CAT, number="065", example=_ex_genes_expr,
      roles=[Role("x", "Categories", "cat"), Role("y", "Values", "num")], opts=_LOLLI_OPTS,
      desc="One row per category with a value; a stick from the baseline ends in a dot.")
def vertical_lollipop(fig, df, c, o, st):
    _lollipop(fig, df, c, o, st, False)


@plot("horizontal_lollipop_066", "Horizontal lollipop chart", CAT, number="066", example=_ex_genes_expr,
      roles=[Role("x", "Categories", "cat"), Role("y", "Values", "num")], opts=_LOLLI_OPTS,
      desc="One row per category with a value; horizontal sticks, labels on the y axis.")
def horizontal_lollipop(fig, df, c, o, st):
    _lollipop(fig, df, c, o, st, True)


# ── 071 stem ─────────────────────────────────────────────────────────────────

@plot("stem_plot_071", "Stem plot", CAT, number="071", example=_ex_stem,
      roles=[Role("x", "X (position / category)", "any"), Role("y", "Values", "num")],
      opts=[Opt("baseline", "Baseline", "float", 0.0, lo=-1e9, hi=1e9, step=0.1),
            Opt("sign", "Colour by side of baseline", "bool", True),
            Opt("marker", "Marker", "choice", "circle", choices=["circle", "square", "diamond", "none"]),
            Opt("size", "Marker size", "float", 5, lo=1, hi=15, step=0.5)],
      desc="One row per position (numeric x such as a genomic coordinate, or a category) with a value; "
           "each value is a stem from the baseline.")
def stem_plot(fig, df, c, o, st):
    numeric_x = _is_num_col(df, c["x"])
    d = pd.DataFrame({"x": num(df, c["x"]) if numeric_x else as_str(df, c["x"]), "y": num(df, c["y"])}).dropna()
    if d.empty:
        return _nodata(fig, st)
    if numeric_x:
        d = d.sort_values("x")
        px = d["x"].to_numpy(float)
    else:
        px = np.arange(len(d), dtype=float)
    v = d["y"].to_numpy()
    base = float(o["baseline"])
    ax = st.ax(fig.add_subplot(), "y")
    mk = {"circle": "o", "square": "s", "diamond": "D", "none": "None"}[o["marker"]]
    groups = ([(v >= base, st.color(0), f"≥ {base:g}"), (v < base, st.color(1), f"< {base:g}")]
              if o["sign"] else [(np.ones(len(v), bool), st.color(0), None)])
    for msk, col, lab in groups:
        if not msk.any():
            continue
        cont = ax.stem(px[msk], v[msk], bottom=base, basefmt=" ", label=lab)
        cont.markerline.set(color=col, marker=mk, markersize=o["size"], markeredgecolor="white",
                            markeredgewidth=0.5)
        cont.stemlines.set(color=col, linewidth=1.2)
    ax.axhline(base, color="#444444", linewidth=0.9)
    if not numeric_x:
        _xcat(ax, px, d["x"], st)
    if o["sign"]:
        st.legend(ax)
    st.labels(ax, c["x"], c["y"])


# ── 072 polar bars ───────────────────────────────────────────────────────────

@plot("polar_bar_072", "Polar bars (rose chart)", CAT, number="072", example=_ex_polar,
      roles=[Role("x", "Categories", "cat"), Role("y", "Values (≥ 0)", "num")],
      opts=[Opt("sort", "Sort", "choice", "none", choices=_SORT),
            Opt("scale", "Value shown by", "choice", "radius", choices=["radius", "area (Nightingale)"]),
            Opt("labels", "Show values", "bool", True),
            Opt("hole", "Inner hole (fraction)", "float", 0.1, lo=0, hi=0.6, step=0.05),
            Opt("gap", "Gap between bars", "float", 0.08, lo=0, hi=0.8, step=0.02)],
      desc="One row per category with a non-negative value; bars radiate from the centre.",
      size=(6.5, 6.0), axis_labels=False)
def polar_bar(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "y": num(df, c["y"]).clip(lower=0)}).dropna()
    if d.empty or d["y"].max() <= 0:
        return _nodata(fig, st)
    d = _sorted(d, "y", o["sort"])
    n = len(d)
    theta = 2 * np.pi * np.arange(n) / n
    r = d["y"].to_numpy(float)
    if o["scale"] != "radius":
        r = np.sqrt(r)
    hole = o["hole"] * r.max()
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(theta, r, width=2 * np.pi / n * (1 - o["gap"]), bottom=hole, color=st.colors(n),
           edgecolor="white", linewidth=1, alpha=0.92)
    rmax = (hole + r.max()) * 1.15
    ax.set_ylim(0, rmax)
    ax.set_xticks(theta, d["x"], fontsize=st.font * 0.85)
    if o["scale"] == "radius":
        from matplotlib.ticker import MaxNLocator
        ticks = [t + hole for t in MaxNLocator(4).tick_values(0, r.max()) if 0 < t <= r.max()]
        ax.set_yticks(ticks, [_fmt(t - hole) for t in ticks], fontsize=st.font * 0.65, color="#888888")
    else:
        ax.set_yticks([])
    ax.set_rlabel_position(360 / n / 2)
    ax.grid(st.grid, color="#d9d9d9", linestyle="--", linewidth=0.6)
    ax.spines["polar"].set_color("#bbbbbb")
    if o["labels"]:
        for t, ri, v in zip(theta, r, d["y"]):
            ax.text(t, hole + ri + r.max() * 0.06, _fmt(v), ha="center", va="center",
                    fontsize=st.font * 0.7, color="#333333")
    t = o.get("title")
    if t:
        ax.set_title(t, fontsize=st.font * 1.15, fontweight="bold", pad=14)


# ── 074 grouped bars with dots ───────────────────────────────────────────────

@plot("dot_bars_074", "Bars with dots (grouped)", CAT, number="074", example=_ex_dotbars,
      roles=[Role("x", "X groups", "cat"), Role("hue", "Sub-groups (optional)", "cat", optional=True),
             Role("value", "Values (one row per replicate)", "num")],
      opts=[Opt("err", "Error bar", "choice", "SD", choices=["SD", "SEM", "95% CI", "none"]),
            Opt("points", "Show the dots", "bool", True),
            Opt("size", "Dot size", "float", 22, lo=2, hi=80, step=2),
            Opt("width", "Group width", "float", 0.8, lo=0.3, hi=1.0, step=0.05)],
      desc="Long table: one row per replicate with its group, optional sub-group (e.g. genotype) and value; "
           "bars = means, dots = replicates.")
def dot_bars(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "h": as_str(df, c["hue"]) if c.get("hue") else "",
                      "v": num(df, c["value"])}).dropna()
    if d.empty:
        return _nodata(fig, st)
    xs, hs = cats(d, "x"), cats(d, "h")
    k = len(hs)
    w = o["width"] / k
    cols = st.colors(k)
    ax = st.ax(fig.add_subplot(), "y")
    r = rng(0)
    for j, h in enumerate(hs):
        first = True
        for i, xv in enumerate(xs):
            v = d.loc[(d["x"] == xv) & (d["h"] == h), "v"].to_numpy()
            if len(v) == 0:
                continue
            p = i + (j - (k - 1) / 2) * w
            m, e = _agg_err(v, o["err"])
            ax.bar(p, m, w * 0.9, color=cols[j], edgecolor="black", linewidth=0.6, alpha=0.75,
                   label=h if (first and c.get("hue")) else None)
            first = False
            if o["err"] != "none":
                ax.errorbar(p, m, yerr=e, color="black", capsize=4, linewidth=1.1)
            if o["points"]:
                ax.scatter(p + r.uniform(-w * 0.22, w * 0.22, len(v)), v, s=o["size"], color=cols[j],
                           edgecolor="black", linewidth=0.6, zorder=3)
    _xcat(ax, range(len(xs)), xs, st)
    if c.get("hue"):
        st.legend(ax, title=c["hue"], loc="upper left")
    ylab = c["value"] + (f" (mean ± {o['err']})" if o["err"] != "none" else " (mean)")
    st.labels(ax, c["x"], ylab)


# ── 075 / 076 stacked bars ───────────────────────────────────────────────────

_STACK_OPTS = [Opt("totals", "Show totals", "bool", True),
               Opt("values", "Show segment values", "bool", False),
               Opt("sort", "Sort by total", "choice", "none", choices=_SORT),
               Opt("width", "Bar width", "float", 0.7, lo=0.1, hi=1.0, step=0.05)]


def _stack(fig, df, c, o, st, horiz):
    parts = [p for p in (c["parts"] or []) if p]
    if not parts:
        return _nodata(fig, st)
    m = pd.DataFrame({p: num(df, p) for p in parts})
    m = m[m.notna().any(axis=1)].fillna(0)
    if m.empty:
        return _nodata(fig, st)
    m["_x"] = as_str(df, c["x"]).loc[m.index].to_numpy()
    m["_t"] = m[parts].sum(axis=1)
    m = _sorted(m, "_t", o["sort"])
    ax = st.ax(fig.add_subplot(), "x" if horiz else "y")
    pos = np.arange(len(m))
    cols = st.colors(len(parts))
    pos_base, neg_base = np.zeros(len(m)), np.zeros(len(m))
    handles = []
    span = max(float(m["_t"].abs().max()), 1e-12)
    for j, p in enumerate(parts):
        v = m[p].to_numpy(float)
        base = np.where(v >= 0, pos_base, neg_base)
        kw = dict(color=cols[j], edgecolor="white", linewidth=0.7, label=p)
        b = (ax.barh(pos, v, o["width"], left=base, **kw) if horiz
             else ax.bar(pos, v, o["width"], bottom=base, **kw))
        handles.append(b)
        if o["values"]:
            ax.bar_label(b, labels=[_fmt(x) if abs(x) > 0.06 * span else "" for x in v],
                         label_type="center", fontsize=st.font * 0.65, color="white")
        pos_base += np.clip(v, 0, None)
        neg_base += np.clip(v, None, 0)
    if o["totals"]:
        for pi, t, top in zip(pos, m["_t"], pos_base):
            if horiz:
                ax.annotate(_fmt(t), (top, pi), xytext=(3, 0), textcoords="offset points", ha="left",
                            va="center", fontsize=st.font * 0.75)
            else:
                ax.annotate(_fmt(t), (pi, top), xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=st.font * 0.75)
    lim = (neg_base.min() * 1.1, max(pos_base.max(), 1e-12) * 1.12)
    if horiz:
        ax.set_xlim(*lim)
        _xcat(ax, pos, m["_x"], st, axis="y")
        ax.set_ylim(len(m) - 0.5, -0.5)
        st.labels(ax, "Value", c["x"])
        _side_legend(ax, st, handles, parts)
    else:
        ax.set_ylim(*lim)
        _xcat(ax, pos, m["_x"], st)
        st.labels(ax, c["x"], "Value")
        _side_legend(ax, st, handles[::-1], parts[::-1])


@plot("vertical_stack_bars_075", "Vertical stacked bars", CAT, number="075", example=_ex_reads,
      roles=[Role("x", "Categories", "cat"), Role("parts", "Components", "num", multi=True)],
      opts=_STACK_OPTS,
      desc="One row per category, one numeric column per stacked component (absolute values).")
def vertical_stack_bars(fig, df, c, o, st):
    _stack(fig, df, c, o, st, False)


@plot("horizontal_stack_bars_076", "Horizontal stacked bars", CAT, number="076", example=_ex_reads,
      roles=[Role("x", "Categories", "cat"), Role("parts", "Components", "num", multi=True)],
      opts=_STACK_OPTS,
      desc="One row per category, one numeric column per stacked component; bars run left to right.")
def horizontal_stack_bars(fig, df, c, o, st):
    _stack(fig, df, c, o, st, True)


# ── 077 Cleveland dot plot ───────────────────────────────────────────────────

@plot("dot_bar_plot_077", "Dot plot (Cleveland)", CAT, number="077", example=_ex_cleveland,
      roles=[Role("x", "Categories", "cat"), Role("v1", "Values", "num"),
             Role("v2", "Second values (optional)", "num", optional=True)],
      opts=[Opt("sort", "Sort by first values", "choice", "descending", choices=_SORT),
            Opt("size", "Dot size", "float", 9, lo=2, hi=25, step=1),
            Opt("labels", "Show values", "bool", False)],
      desc="One row per category with one (or two) numeric values, drawn as dots on a guide line.")
def dot_bar_plot(fig, df, c, o, st):
    cols_ = [c["v1"]] + ([c["v2"]] if c.get("v2") else [])
    d = pd.DataFrame({"x": as_str(df, c["x"]), **{f"v{i}": num(df, col) for i, col in enumerate(cols_)}})
    d = d[d[[f"v{i}" for i in range(len(cols_))]].notna().any(axis=1)]
    if d.empty:
        return _nodata(fig, st)
    d = _sorted(d, "v0", o["sort"])
    ax = st.ax(fig.add_subplot(), "x")
    ax.spines["left"].set_visible(False)
    y = np.arange(len(d))
    for yi in y:
        ax.axhline(yi, color="#bdbdbd", linewidth=0.8, linestyle=":", zorder=0)
    for i, col in enumerate(cols_):
        v = d[f"v{i}"].to_numpy()
        ax.scatter(v, y, s=o["size"] ** 2, color=st.color(i), edgecolor="black", linewidth=0.6, zorder=3,
                   label=col)
        if o["labels"]:
            for vi, yi in zip(v, y):
                if np.isfinite(vi):
                    ax.annotate(_fmt(vi), (vi, yi), xytext=(0, 0.5 * o["size"] + 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=st.font * 0.7)
    ax.tick_params(axis="y", left=False)
    ax.set_yticks(y, d["x"])
    ax.set_ylim(len(d) - 0.5, -0.6)
    if len(cols_) > 1:
        _side_legend(ax, st)
    st.labels(ax, c["v1"] if len(cols_) == 1 else "Value", c["x"])


# ── 078 bullet chart ─────────────────────────────────────────────────────────

@plot("bullet_chart_078", "Bullet chart", CAT, number="078", example=_ex_bullet,
      roles=[Role("x", "Measures", "cat"), Role("value", "Actual value", "num"),
             Role("target", "Target (optional)", "num", optional=True),
             Role("ranges", "Range limits, e.g. poor / ok / good (optional)", "num", multi=True,
                  optional=True)],
      opts=[Opt("shared", "Same scale for all rows", "bool", False),
            Opt("labels", "Show the actual value", "bool", True),
            Opt("bar_color", "Bar colour", "color", "#2F3B4C"),
            Opt("target_color", "Target colour", "color", "#D6404E")],
      desc="One row per measure: actual value, optional target and up to three upper limits of the "
           "qualitative ranges (poor, acceptable, good).", size=(7.5, 4.5))
def bullet_chart(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "v": num(df, c["value"])})
    d["t"] = num(df, c["target"]) if c.get("target") else np.nan
    rcols = [r for r in (c.get("ranges") or []) if r][:4]
    for i, rc in enumerate(rcols):
        d[f"r{i}"] = num(df, rc)
    d = d.dropna(subset=["v"]).reset_index(drop=True)
    if d.empty:
        return _nodata(fig, st)
    n = len(d)
    axes = np.atleast_1d(fig.subplots(n, 1, sharex=bool(o["shared"])))
    greys = ["#a6a6a6", "#c4c4c4", "#dddddd", "#ececec"]
    rows, gmax = [], 0.0
    for _, row in d.iterrows():
        rs = sorted(float(row[f"r{i}"]) for i in range(len(rcols)) if np.isfinite(row[f"r{i}"]))
        t = row["t"] if np.isfinite(row["t"]) else 0.0
        top = max([row["v"], t] + rs)
        if not rs:
            rs = [top * 0.6 * 1.1, top * 0.85 * 1.1, top * 1.1]
        rows.append(rs)
        gmax = max(gmax, top, max(rs))
    for ax, (_, row), rs in zip(axes, d.iterrows(), rows):
        for ri, sh in sorted(zip(rs, greys[:len(rs)]), key=lambda z: -z[0]):
            ax.barh(0, ri, 0.9, color=sh, linewidth=0)
        ax.barh(0, row["v"], 0.32, color=o["bar_color"])
        t = row["t"] if np.isfinite(row["t"]) else None
        if t is not None:
            ax.plot([t, t], [-0.3, 0.3], color=o["target_color"], linewidth=3, solid_capstyle="butt")
        top = gmax if o["shared"] else max(max(rs), row["v"], t or 0)
        ax.set_xlim(0, top * 1.02 if top > 0 else 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0], [row["x"]], fontsize=st.font * 0.9)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#555555")
        ax.tick_params(axis="y", left=False)
        ax.tick_params(axis="x", labelsize=st.font * 0.75)
        if o["labels"]:
            ax.text(1.01, 0.5, _fmt(row["v"]), transform=ax.transAxes, ha="left", va="center",
                    fontsize=st.font * 0.85, fontweight="bold")
    from matplotlib.lines import Line2D
    handles = [_patch(o["bar_color"], "Actual", linewidth=0)]
    if d["t"].notna().any():
        handles.append(Line2D([], [], color=o["target_color"], linewidth=3, label="Target"))
    names = rcols if rcols else ["Poor", "Acceptable", "Good"]
    for nm, sh in zip(names, greys):
        handles.append(_patch(sh, nm, linewidth=0))
    if st.legend_mode == "best":
        fig.legend(handles=handles, frameon=False, fontsize=st.font * 0.8, loc="outside lower center",
                   ncol=min(6, len(handles)))
    else:
        st.legend(axes[0], handles)
    if o.get("xlabel"):
        axes[-1].set_xlabel(o["xlabel"], fontsize=st.font)
    st.suptitle(fig)


# ── 080 3-D bars ─────────────────────────────────────────────────────────────

@plot("vertical_3d_bars_080", "3-D bars", CAT, number="080", example=_ex_matrix,
      roles=[Role("x", "Categories", "cat"), Role("series", "Series (one column each)", "num", multi=True)],
      opts=[Opt("color_by", "Colour by", "choice", "series", choices=["series", "height (colour map)"]),
            Opt("elev", "View elevation", "int", 25, lo=0, hi=90),
            Opt("azim", "View rotation", "int", -55, lo=-180, hi=180),
            Opt("depth", "Bar size", "float", 0.6, lo=0.1, hi=1.0, step=0.05)],
      desc="A matrix: one row per category (e.g. gene), one numeric column per series (e.g. time point).",
      size=(7.5, 6.0))
def vertical_3d_bars(fig, df, c, o, st):
    series = [s for s in (c["series"] or []) if s]
    if not series:
        return _nodata(fig, st)
    m = pd.DataFrame({s: num(df, s) for s in series})
    keep = m.notna().any(axis=1)
    m = m[keep].fillna(0)
    if m.empty:
        return _nodata(fig, st)
    lab = list(as_str(df, c["x"])[keep])
    n, k = m.shape
    ax = fig.add_subplot(projection="3d")
    w = o["depth"]
    cols = st.colors(k)
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    cm = colormaps[st.cmap]
    vals = m.to_numpy(float)
    nv = Normalize(vals.min(), vals.max() if vals.max() > vals.min() else vals.min() + 1)
    for j in range(k):
        z = vals[:, j]
        col = [cm(nv(v)) for v in z] if o["color_by"] != "series" else cols[j]
        ax.bar3d(np.arange(n) - w / 2, np.full(n, j) - w / 2, np.zeros(n), w, w, z, color=col, shade=True,
                 edgecolor=(0, 0, 0, 0.25), linewidth=0.3)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(lab, fontsize=st.font * 0.75, rotation=0 if n <= 6 else 30,
                       ha="right" if n > 6 else "center", va="center")
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels(series, fontsize=st.font * 0.75, ha="left", va="center")
    ax.tick_params(axis="z", labelsize=st.font * 0.75)
    ax.view_init(o["elev"], o["azim"])
    ax.set_xlabel(o.get("xlabel") or c["x"], fontsize=st.font, labelpad=8)
    ax.set_zlabel(o.get("ylabel") or "Value", fontsize=st.font, labelpad=6)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0.97, 0.97, 0.97, 1))
    ax.grid(st.grid)
    if o["color_by"] == "series":
        st.legend(ax, [_patch(cols[j], s) for j, s in enumerate(series)], series, loc="upper left")
    else:
        from matplotlib.cm import ScalarMappable
        fig.colorbar(ScalarMappable(nv, cm), ax=ax, shrink=0.55, pad=0.08)
    t = o.get("title")
    if t:
        ax.set_title(t, fontsize=st.font * 1.15, fontweight="bold")


# ── 084 qPCR vs RNA-seq ──────────────────────────────────────────────────────

@plot("pcr_rnaseq_dual_y_084", "qPCR vs RNA-seq (dual Y)", CAT, number="084", example=_ex_pcr,
      roles=[Role("x", "Genes", "cat"), Role("qpcr", "qPCR values (left axis)", "num"),
             Role("rnaseq", "RNA-seq values (right axis)", "num"),
             Role("err", "qPCR SD / SEM (optional)", "num", optional=True)],
      opts=[Opt("style", "RNA-seq shown as", "choice", "bars", choices=["bars", "points + line", "points"]),
            Opt("corr", "Show Pearson r", "bool", True)],
      desc="One row per gene: qPCR value (e.g. fold change), RNA-seq value (e.g. FPKM) and optionally "
           "the qPCR error; the two methods are compared on twin y axes.")
def pcr_rnaseq_dual(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "q": num(df, c["qpcr"]), "r": num(df, c["rnaseq"]),
                      "e": num(df, c["err"]).abs() if c.get("err") else np.nan})
    d = d[d[["q", "r"]].notna().any(axis=1)]
    if d.empty:
        return _nodata(fig, st)
    ax = st.ax(fig.add_subplot(), "y")
    cq, cr = st.color(0), st.color(1)
    ax2 = _twin(ax, cr, st)
    x = np.arange(len(d))
    bars = o["style"] == "bars"
    w = 0.38 if bars else 0.6
    has_e = d["e"].notna().any()
    hq = ax.bar(x - (w / 2 if bars else 0), d["q"].fillna(0), w, color=cq, edgecolor="black", linewidth=0.6,
                yerr=d["e"].fillna(0) if has_e else None, capsize=3, error_kw={"elinewidth": 1})
    if bars:
        hr = ax2.bar(x + w / 2, d["r"].fillna(0), w, color=cr, edgecolor="black", linewidth=0.6)
    else:
        hr, = ax2.plot(x, d["r"], "-o" if o["style"] == "points + line" else "o", color=cr, linewidth=2,
                       markersize=7, markeredgecolor="white")
    top_q = float(np.nanmax(d["q"] + d["e"].fillna(0)))
    ax.set_ylim(min(0, float(np.nanmin(d["q"])) * 1.1), top_q * 1.18 if top_q > 0 else 1)
    top_r = float(np.nanmax(d["r"]))
    ax2.set_ylim(min(0, float(np.nanmin(d["r"])) * 1.1), top_r * 1.18 if top_r > 0 else 1)
    ax2.set_ylabel(c["rnaseq"], fontsize=st.font, color=cr)
    ax.tick_params(axis="y", colors=cq)
    ax.spines["left"].set_color(cq)
    _xcat(ax, x, d["x"], st)
    ok = d[["q", "r"]].dropna()
    if o["corr"] and len(ok) > 2 and ok["q"].std() > 0 and ok["r"].std() > 0:
        rr = np.corrcoef(ok["q"], ok["r"])[0, 1]
        ax.text(0.99, 0.99, f"Pearson r = {rr:.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=st.font * 0.8, color="#333333")
    st.legend(ax, [hq, hr], ["qPCR", "RNA-seq"], loc="upper center")
    st.labels(ax, c["x"], c["qpcr"])
    ax.yaxis.label.set_color(cq)


# ── 100 diverging bars ───────────────────────────────────────────────────────

@plot("two_directional_bars_100", "Two-directional bars", CAT, number="100", example=_ex_diverging,
      roles=[Role("x", "Terms", "cat"), Role("y", "Signed values", "num")],
      opts=[Opt("sort", "Sort", "choice", "descending", choices=_SORT),
            Opt("names", "Term names", "choice", "next to the bars", choices=["next to the bars", "y axis"]),
            Opt("labels", "Show values", "bool", False),
            Opt("pos_color", "Positive colour", "color", "#D6404E"),
            Opt("neg_color", "Negative colour", "color", "#3B6FB6")],
      desc="One row per term (e.g. pathway) with a signed value (z-score, log2 FC, NES); "
           "positive bars go right, negative left.", size=(7.5, 5.5))
def two_directional_bars(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "y": num(df, c["y"])}).dropna()
    if d.empty:
        return _nodata(fig, st)
    d = _sorted(d, "y", o["sort"])
    ax = st.ax(fig.add_subplot(), "x")
    y = np.arange(len(d))
    v = d["y"].to_numpy()
    col = [o["pos_color"] if t >= 0 else o["neg_color"] for t in v]
    bars = ax.barh(y, v, 0.72, color=col, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="#333333", linewidth=0.9)
    m = max(np.abs(v).max(), 1e-12)
    ax.set_xlim(-m * 1.15, m * 1.15)
    ax.set_ylim(len(d) - 0.5, -0.5)
    if o["names"] == "y axis":
        ax.set_yticks(y, d["x"])
    else:
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        for yi, t, s in zip(y, v, d["x"]):
            ax.annotate(s, (0, yi), xytext=(-4 if t >= 0 else 4, 0), textcoords="offset points",
                        ha="right" if t >= 0 else "left", va="center", fontsize=st.font * 0.8)
    if o["labels"]:
        ax.bar_label(bars, labels=[_fmt(t) for t in v], padding=3, fontsize=st.font * 0.72)
    st.legend(ax, [_patch(o["pos_color"], "Positive"), _patch(o["neg_color"], "Negative")],
              loc="lower right")
    st.labels(ax, c["y"], "")


# ── 117 broken y axis ────────────────────────────────────────────────────────

@plot("bars_with_truncation_117", "Bars with broken axis", CAT, number="117", example=_ex_truncation,
      roles=[Role("x", "Categories", "cat"), Role("y", "Values", "num"),
             Role("err", "Error (optional)", "num", optional=True)],
      opts=[Opt("auto", "Choose the break automatically", "bool", True),
            Opt("low_top", "Manual: lower part ends at", "float", 100.0, lo=-1e12, hi=1e12, step=10),
            Opt("high_bottom", "Manual: upper part starts at", "float", 1000.0, lo=-1e12, hi=1e12, step=10),
            Opt("labels", "Show values", "bool", True),
            Opt("ratio", "Height of the upper part", "float", 0.35, lo=0.1, hi=0.9, step=0.05)],
      desc="One row per category with a value (and optional error), for data where one or two bars "
           "are far taller than the rest; the y axis is cut to show both.")
def bars_with_truncation(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "y": num(df, c["y"]),
                      "e": num(df, c["err"]).abs() if c.get("err") else 0.0}).dropna(subset=["y"])
    if d.empty:
        return _nodata(fig, st)
    d["e"] = d["e"].fillna(0)
    tops = (d["y"] + d["e"]).to_numpy(float)
    x = np.arange(len(d))
    low_top = high_bottom = None
    if o["auto"]:
        s = np.sort(tops[tops > 0])
        if len(s) >= 2:
            ratios = s[1:] / np.maximum(s[:-1], 1e-12)
            i = int(np.argmax(ratios))
            if ratios[i] >= 3:
                low_top = s[i] * 1.22
                upper = d[(d["y"] + d["e"]) >= s[i + 1]]
                high_bottom = max(low_top * 1.05, float((upper["y"] - upper["e"]).min()) * 0.8)
    else:
        low_top, high_bottom = float(o["low_top"]), float(o["high_bottom"])
        if not (high_bottom > low_top):
            low_top = high_bottom = None
    cols = st.colors(len(d))
    has_e = bool((d["e"] > 0).any())

    def draw(ax):
        b = ax.bar(x, d["y"], 0.65, color=cols, edgecolor="black", linewidth=0.6)
        if has_e:
            ax.errorbar(x, d["y"], yerr=d["e"], fmt="none", ecolor="black", capsize=5, linewidth=1.1)
        if o["labels"]:
            for xi, yi, ei in zip(x, d["y"], d["e"]):
                ax.annotate(_fmt(yi), (xi, yi + ei), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=st.font * 0.75, annotation_clip=True)
        return b

    if low_top is None:
        ax = st.ax(fig.add_subplot(), "y")
        draw(ax)
        top = float(np.max(tops))
        if top > 0:
            ax.set_ylim(min(0, float(d["y"].min()) * 1.1), top * 1.12)
        _xcat(ax, x, d["x"], st)
        st.labels(ax, c["x"], c["y"])
        return
    axT, axB = fig.subplots(2, 1, sharex=True,
                            gridspec_kw={"height_ratios": [o["ratio"], 1 - o["ratio"]]})
    fig.get_layout_engine().set(h_pad=0.03, hspace=0.0)
    for a in (axT, axB):
        st.ax(a, "y")
        draw(a)
    top = float(np.max(tops))
    axT.set_ylim(high_bottom, top + (top - high_bottom) * 0.2)
    axB.set_ylim(min(0, float(d["y"].min()) * 1.1), low_top)
    axT.spines["bottom"].set_visible(False)
    axT.tick_params(axis="x", bottom=False, labelbottom=False)
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=11, linestyle="none", color="black", mec="black",
              mew=1.2, clip_on=False)
    axT.plot([0], [0], transform=axT.transAxes, **kw)
    axB.plot([0], [1], transform=axB.transAxes, **kw)
    _xcat(axB, x, d["x"], st)
    axB.set_xlabel(o.get("xlabel") or c["x"], fontsize=st.font)
    fig.supylabel(o.get("ylabel") or c["y"], fontsize=st.font)
    t = o.get("title")
    if t:
        axT.set_title(t, fontsize=st.font * 1.15, fontweight="bold", pad=10)


# ── 011 waterfall ────────────────────────────────────────────────────────────

@plot("waterfall_chart_011", "Waterfall chart", CAT, number="011", example=_ex_waterfall,
      roles=[Role("x", "Steps", "cat"), Role("y", "Changes", "num")],
      opts=[Opt("start", "First row is the start value", "bool", True),
            Opt("total", "Add a final total bar", "bool", True),
            Opt("total_label", "Total bar label", "str", "Final"),
            Opt("labels", "Show values", "bool", True),
            Opt("connectors", "Connector lines", "bool", True),
            Opt("inc_color", "Increase colour", "color", "#2BA36B"),
            Opt("dec_color", "Decrease colour", "color", "#D6404E"),
            Opt("tot_color", "Start / total colour", "color", "#5B6B82")],
      desc="One row per step with its change (positive or negative); the first row can be the start value.")
def waterfall_chart(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "y": num(df, c["y"])}).dropna()
    if d.empty:
        return _nodata(fig, st)
    labels, bottoms, heights, colors, kinds = [], [], [], [], []
    cum = 0.0
    for i, (lab, v) in enumerate(zip(d["x"], d["y"])):
        if i == 0 and o["start"]:
            labels.append(lab)
            bottoms.append(0.0)
            heights.append(v)
            colors.append(o["tot_color"])
            kinds.append("tot")
            cum = v
            continue
        labels.append(lab)
        bottoms.append(cum)
        heights.append(v)
        colors.append(o["inc_color"] if v >= 0 else o["dec_color"])
        kinds.append("step")
        cum += v
    if o["total"]:
        labels.append(o["total_label"] or "Total")
        bottoms.append(0.0)
        heights.append(cum)
        colors.append(o["tot_color"])
        kinds.append("tot")
    ax = st.ax(fig.add_subplot(), "y")
    x = np.arange(len(labels))
    b = np.array(bottoms, float)
    h = np.array(heights, float)
    ax.bar(x, h, 0.62, bottom=b, color=colors, edgecolor="black", linewidth=0.6)
    ends = b + h
    if o["connectors"]:
        for i in range(len(x) - 1):
            ax.plot([x[i] + 0.31, x[i + 1] - 0.31], [ends[i], ends[i]], color="#777777", linewidth=0.8,
                    linestyle="--")
    if o["labels"]:
        for xi, bi, hi, k in zip(x, b, h, kinds):
            txt = _fmt(hi) if k == "tot" else (("+" if hi >= 0 else "−") + _fmt(abs(hi)))
            ax.annotate(txt, (xi, max(bi, bi + hi)), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=st.font * 0.75)
    lo_ = min(0.0, float(np.min(np.minimum(b, ends))))
    hi_ = float(np.max(np.maximum(b, ends)))
    ax.set_ylim(lo_ * 1.1, hi_ * 1.12 if hi_ > 0 else 1)
    ax.axhline(0, color="#333333", linewidth=0.8)
    _xcat(ax, x, labels, st)
    hs = [_patch(o["inc_color"], "Increase"), _patch(o["dec_color"], "Decrease"),
          _patch(o["tot_color"], "Start / total" if o["start"] else "Total")]
    st.legend(ax, hs)
    st.labels(ax, c["x"], c["y"])


# ── 032 dumbbell ─────────────────────────────────────────────────────────────

@plot("dumbbell_chart_032", "Dumbbell chart", CAT, number="032", example=_ex_dumbbell,
      roles=[Role("x", "Categories", "cat"), Role("a", "First values", "num"), Role("b", "Second values", "num")],
      opts=[Opt("sort", "Sort by", "choice", "difference",
                choices=["none", "difference", "first values", "second values"]),
            Opt("labels", "Show the difference", "bool", True),
            Opt("horizontal", "Horizontal", "bool", True),
            Opt("size", "Dot size", "float", 10, lo=3, hi=25, step=1)],
      desc="One row per category with two numeric values (e.g. before / after); a line joins them.")
def dumbbell_chart(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "a": num(df, c["a"]), "b": num(df, c["b"])}).dropna()
    if d.empty:
        return _nodata(fig, st)
    key = {"difference": d["b"] - d["a"], "first values": d["a"], "second values": d["b"]}.get(o["sort"])
    if key is not None:
        d = d.assign(_k=key).sort_values("_k", ascending=False, kind="stable")
    ca, cb = st.color(0), st.color(1)
    hz = o["horizontal"]
    ax = st.ax(fig.add_subplot(), "x" if hz else "y")
    p = np.arange(len(d))
    a, b = d["a"].to_numpy(), d["b"].to_numpy()
    s = o["size"] ** 2
    if hz:
        ax.hlines(p, a, b, color="#9e9e9e", linewidth=3, zorder=1)
        ax.scatter(a, p, s=s, color=ca, edgecolor="black", linewidth=0.6, zorder=3, label=c["a"])
        ax.scatter(b, p, s=s, color=cb, edgecolor="black", linewidth=0.6, zorder=3, label=c["b"])
        ax.set_yticks(p, d["x"])
        ax.set_ylim(len(d) - 0.5, -0.7)
    else:
        ax.vlines(p, a, b, color="#9e9e9e", linewidth=3, zorder=1)
        ax.scatter(p, a, s=s, color=ca, edgecolor="black", linewidth=0.6, zorder=3, label=c["a"])
        ax.scatter(p, b, s=s, color=cb, edgecolor="black", linewidth=0.6, zorder=3, label=c["b"])
        _xcat(ax, p, d["x"], st)
    lo_, hi_ = min(a.min(), b.min()), max(a.max(), b.max())
    pad = (hi_ - lo_) * 0.08 or 1
    if o["labels"]:
        for pi, ai, bi in zip(p, a, b):
            diff = bi - ai
            txt = ("+" if diff >= 0 else "−") + _fmt(abs(diff))
            if hz:
                ax.annotate(txt, (max(ai, bi), pi), xytext=(0.5 * o["size"] + 4, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=st.font * 0.72, color="#444444")
            else:
                ax.annotate(txt, (pi, max(ai, bi)), xytext=(0, 0.5 * o["size"] + 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=st.font * 0.72, color="#444444")
        hi_ += pad * 1.5
    (ax.set_xlim if hz else ax.set_ylim)(lo_ - pad, hi_ + pad)
    _side_legend(ax, st)
    if hz:
        st.labels(ax, "Value", c["x"])
    else:
        st.labels(ax, c["x"], "Value")


# ── t001 element statistics ──────────────────────────────────────────────────

@plot("element_statistics_t001", "Element statistics (counts)", CAT, number="t001", example=_ex_elements,
      roles=[Role("x", "Column with repeated elements", "any")],
      opts=[Opt("sort", "Order", "choice", "by count", choices=["by count", "as in table", "alphabetical"]),
            Opt("horizontal", "Horizontal bars", "bool", False),
            Opt("pct", "Show percentages", "bool", True),
            Opt("top", "Show only the top N (0 = all)", "int", 0, lo=0, hi=1000),
            Opt("multicolor", "One colour per bar", "bool", True)],
      desc="Any column with repeated values (e.g. the variant type of each mutation); each distinct value "
           "is counted and shown as a bar.")
def element_statistics(fig, df, c, o, st):
    s = df[c["x"]].dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return _nodata(fig, st, "The selected column is empty")
    order = list(dict.fromkeys(s))
    vc = s.value_counts()
    total = int(vc.sum())
    if o["top"] and o["top"] < len(vc):
        vc = vc.iloc[:int(o["top"])]
    if o["sort"] == "as in table":
        vc = vc.reindex([k for k in order if k in vc.index])
    elif o["sort"] == "alphabetical":
        vc = vc.sort_index()
    p = np.arange(len(vc))
    cnt = vc.to_numpy()
    col = st.colors(len(vc)) if o["multicolor"] else st.color(0)
    if o["horizontal"]:
        ax = st.ax(fig.add_subplot(), "x")
        bars = ax.barh(p, cnt, 0.7, color=col, edgecolor="black", linewidth=0.6)
        ax.set_yticks(p, vc.index)
        ax.set_ylim(len(vc) - 0.5, -0.5)
        ax.set_xlim(0, cnt.max() * (1.3 if o["pct"] else 1.12))
        labs = [f"{v} ({100 * v / total:.1f}%)" if o["pct"] else f"{v}" for v in cnt]
        ax.bar_label(bars, labels=labs, padding=3, fontsize=st.font * 0.72)
        st.labels(ax, "Count", c["x"])
    else:
        ax = st.ax(fig.add_subplot(), "y")
        bars = ax.bar(p, cnt, 0.7, color=col, edgecolor="black", linewidth=0.6)
        _xcat(ax, p, vc.index, st)
        labs = [f"{v}\n{100 * v / total:.1f}%" if o["pct"] else f"{v}" for v in cnt]
        ax.set_ylim(0, cnt.max() * (1.22 if o["pct"] else 1.12))
        ax.bar_label(bars, labels=labs, padding=2, fontsize=st.font * 0.7)
        st.labels(ax, c["x"], "Count")
