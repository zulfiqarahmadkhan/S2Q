"""Distributions (box, violin, raincloud…), scatter & correlation, heatmaps, clusters, PCA, clinical & statistics plots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from s2q.core.gallery.base import CATEGORIES, Opt, Role, as_str, cats, num, p_stars, plot, rng

CD = CATEGORIES[3]   # Distribution (box, violin…)
CS = CATEGORIES[4]   # Scatter & correlation
CH = CATEGORIES[5]   # Heatmap, cluster & PCA
CC = CATEGORIES[6]   # Clinical & statistics


# ═════════════════════════════════════════════════════════════════════════════
#  shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def _long(df, c, g="group", v="value"):
    """(group names, list of value arrays) from a long table; an empty group role → one group."""
    gcol = c.get(g) or ""
    d = pd.DataFrame({"v": num(df, c[v])})
    d["g"] = as_str(df, gcol) if gcol else str(c[v])
    d = d[np.isfinite(d["v"])]
    if d.empty:
        raise ValueError(f"No numeric values in column '{c[v]}'.")
    groups = cats(d, "g")
    return groups, [d.loc[d["g"] == k, "v"].to_numpy(float) for k in groups]


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "p = n/a"
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def _kde(v, grid, bw=1.0):
    """Gaussian KDE of *v* on *grid*; a narrow bump when there are too few distinct values."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) >= 2 and np.std(v) > 0:
        from scipy.stats import gaussian_kde
        k = gaussian_kde(v)
        k.set_bandwidth(k.factor * max(bw, 0.05))
        return k(grid)
    m = v.mean() if len(v) else 0.0
    s = (grid.max() - grid.min()) / 30 or 1.0
    return np.exp(-0.5 * ((grid - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def _span(data):
    allv = np.concatenate([np.asarray(d, float) for d in data]) if data else np.array([0.0])
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    return lo, hi, (hi - lo) or (abs(hi) or 1.0)


def _violin(ax, v, pos, width, color, side=0, horiz=False, bw=1.0, alpha=0.75, pad=0.0, lw=0.8,
            ec="black", zorder=2, offset=0.0):
    """Draw a violin (side 0 = both halves, +1 / -1 = one half) at *pos*; returns f(value) → half width."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return lambda y: 0.0
    lo, hi = v.min(), v.max()
    pad = pad or 0.08 * ((hi - lo) or 1.0)
    y = np.linspace(lo - pad, hi + pad, 200)
    d = _kde(v, y, bw)
    d = d / d.max() * width / 2 if d.max() > 0 else d
    base = pos + offset * (side or 1)
    if side == 0:
        a, b = base - d, base + d
    elif side > 0:
        a, b = np.full_like(d, base), base + d
    else:
        a, b = base - d, np.full_like(d, base)
    fill = ax.fill_between if horiz else ax.fill_betweenx
    fill(y, a, b, facecolor=color, alpha=alpha, edgecolor="none", zorder=zorder)
    if horiz:
        ax.plot(y, a, color=ec, lw=lw, zorder=zorder + 0.1)
        ax.plot(y, b, color=ec, lw=lw, zorder=zorder + 0.1)
    else:
        ax.plot(a, y, color=ec, lw=lw, zorder=zorder + 0.1)
        ax.plot(b, y, color=ec, lw=lw, zorder=zorder + 0.1)
    return lambda q: float(np.interp(q, y, d))


def _quartile_lines(ax, v, pos, half, side=0, horiz=False, offset=0.0):
    """Dashed quartile / solid median lines across a violin."""
    for q, ls, lw in ((25, (0, (3, 2)), 0.9), (50, "-", 1.4), (75, (0, (3, 2)), 0.9)):
        y = np.percentile(v, q)
        w = half(y)
        base = pos + offset * (side or 1)
        a, b = (base - w, base + w) if side == 0 else (base, base + w) if side > 0 else (base - w, base)
        if horiz:
            ax.plot([y, y], [a, b], color="black", lw=lw, ls=ls, zorder=4)
        else:
            ax.plot([a, b], [y, y], color="black", lw=lw, ls=ls, zorder=4)


def _minibox(ax, v, pos, w=0.08, horiz=False, color="#222222", zorder=4):
    """A slim box (IQR), whiskers (1.5 IQR) and a white median dot, as drawn inside violins."""
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo = v[v >= q1 - 1.5 * iqr].min()
    hi = v[v <= q3 + 1.5 * iqr].max()
    from matplotlib.patches import Rectangle
    if horiz:
        ax.plot([lo, hi], [pos, pos], color=color, lw=1, zorder=zorder)
        ax.add_patch(Rectangle((q1, pos - w / 2), q3 - q1, w, facecolor=color, edgecolor=color, zorder=zorder))
        ax.scatter([med], [pos], s=22, color="white", edgecolor=color, linewidth=0.6, zorder=zorder + 1)
    else:
        ax.plot([pos, pos], [lo, hi], color=color, lw=1, zorder=zorder)
        ax.add_patch(Rectangle((pos - w / 2, q1), w, q3 - q1, facecolor=color, edgecolor=color, zorder=zorder))
        ax.scatter([pos], [med], s=22, color="white", edgecolor=color, linewidth=0.6, zorder=zorder + 1)


def _box(ax, data, pos, colors, width, horiz=False, fill="light", notch=False, fliers=False, lw=1.0,
         zorder=2):
    from matplotlib.colors import to_rgba
    bp = ax.boxplot(data, positions=list(pos), widths=width, patch_artist=True, notch=notch,
                    showfliers=fliers, orientation="horizontal" if horiz else "vertical", manage_ticks=False,
                    medianprops=dict(color="black", linewidth=lw * 1.6),
                    whiskerprops=dict(color="black", linewidth=lw), capprops=dict(color="black", linewidth=lw),
                    boxprops=dict(linewidth=lw, edgecolor="black"),
                    flierprops=dict(marker="o", markersize=3.5, markerfacecolor="none", markeredgecolor="#555555"),
                    zorder=zorder)
    for i, (patch, col) in enumerate(zip(bp["boxes"], colors)):
        if fill == "none":
            patch.set_facecolor("white")
            patch.set_edgecolor(col)
            patch.set_linewidth(lw * 1.4)
            for part in (bp["whiskers"][2 * i:2 * i + 2] + bp["caps"][2 * i:2 * i + 2] + [bp["medians"][i]]):
                part.set_color(col)
        else:
            patch.set_facecolor(to_rgba(col, 0.4 if fill == "light" else 0.9))
    return bp


def _cat_axis(ax, groups, horiz=False, st=None):
    n = len(groups)
    if horiz:
        ax.set_yticks(range(n), groups)
        ax.set_ylim(n - 0.4, -0.6)
    else:
        long = max((len(g) for g in groups), default=0)
        rot = 30 if (n > 6 or long * n > 60) else 0
        ax.set_xticks(range(n), groups, rotation=rot, ha="right" if rot else "center")
        ax.set_xlim(-0.6, n - 0.4)


def _swarm(y, dx, dy, maxw=None):
    """Beeswarm x offsets: points placed greedily (low → high) at the nearest free horizontal slot."""
    y = np.asarray(y, float)
    n = len(y)
    xs = np.zeros(n)
    if n == 0:
        return xs
    if n > 2000:   # too many for a swarm: fall back to a uniform jitter
        return rng(0).uniform(-(maxw or 0.3), maxw or 0.3, n)
    order = np.argsort(y)
    placed = []
    for i in order:
        yi = y[i]
        near = []
        for j in reversed(placed):
            if yi - y[j] >= dy:
                break
            near.append((xs[j], y[j]))
        k = 0
        while True:
            cand = 0.0 if k == 0 else ((k + 1) // 2) * dx * (1 if k % 2 else -1) * 0.5
            if all(((cand - xj) / dx) ** 2 + ((yi - yj) / dy) ** 2 >= 0.98 for xj, yj in near):
                break
            k += 1
        xs[i] = cand
        placed.append(i)
    if maxw and np.abs(xs).max() > maxw:
        xs *= maxw / np.abs(xs).max()
    return xs


def _swarm_steps(fig, ngroups, vrange, size, horiz=False):
    """Marker diameter in data units along the category axis and the value axis."""
    w, h = fig.get_size_inches()
    d_in = np.sqrt(size) / 72 * 1.05
    cat_len, val_len = (h * 0.72, w * 0.75) if horiz else (w * 0.75, h * 0.72)
    return (ngroups + 0.2) * d_in / cat_len, vrange * 1.25 * d_in / val_len


def _two_sample(a, b, test):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        if test.startswith("Mann"):
            return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        if test.startswith("Student"):
            return float(stats.ttest_ind(a, b).pvalue)
        return float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        return np.nan


def _pairs(n, mode):
    if n < 2 or mode == "none":
        return []
    if mode.startswith("vs first"):
        return [(0, j) for j in range(1, n)]
    if mode == "adjacent":
        return [(i, i + 1) for i in range(n - 1)]
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _adjust(ps, method):
    ps = np.asarray(ps, float)
    m = np.isfinite(ps).sum()
    if method == "Bonferroni":
        return np.minimum(ps * m, 1)
    if method == "Holm" and m:
        out = ps.copy()
        idx = [i for i in np.argsort(ps) if np.isfinite(ps[i])]
        run = 0.0
        for r, i in enumerate(idx):
            run = max(run, min(1.0, ps[i] * (m - r)))
            out[i] = run
        return out
    return ps


def _brackets(ax, st, pairs, texts, top, vrange, horiz=False):
    """Stacked significance brackets between category positions; extends the value axis to fit."""
    if not pairs:
        return
    h, step = vrange * 0.025, vrange * 0.085
    base = top + vrange * 0.05
    levels = []   # per level: list of occupied (lo, hi)
    order = sorted(range(len(pairs)), key=lambda k: (abs(pairs[k][1] - pairs[k][0]), pairs[k][0]))
    ytop = base
    for k in order:
        a, b = sorted(pairs[k])
        lev = 0
        while lev < len(levels) and any(not (b < lo - 0.05 or a > hi + 0.05) for lo, hi in levels[lev]):
            lev += 1
        if lev == len(levels):
            levels.append([])
        levels[lev].append((a, b))
        y = base + lev * step
        ytop = max(ytop, y + h)
        xs, ys = [a, a, b, b], [y, y + h, y + h, y]
        if horiz:
            ax.plot(ys, xs, color="black", lw=0.9, zorder=5)
            ax.text(y + h + vrange * 0.008, (a + b) / 2, texts[k], ha="left", va="center",
                    fontsize=st.font * 0.75, rotation=-90 if len(texts[k]) > 4 else 0, zorder=5)
        else:
            ax.plot(xs, ys, color="black", lw=0.9, zorder=5)
            ax.text((a + b) / 2, y + h + vrange * 0.004, texts[k], ha="center", va="bottom",
                    fontsize=st.font * 0.75, zorder=5)
    lim = ytop + step * 0.75
    if horiz:
        ax.set_xlim(right=max(ax.get_xlim()[1], lim))
    else:
        ax.set_ylim(top=max(ax.get_ylim()[1], lim))


def _suplabels(fig, st, x="", y=""):
    xl, yl = st.opts.get("xlabel") or x, st.opts.get("ylabel") or y
    if xl:
        fig.supxlabel(xl, fontsize=st.font)
    if yl:
        fig.supylabel(yl, fontsize=st.font)


def _err(v, kind):
    from scipy import stats
    n = len(v)
    sd = v.std(ddof=1) if n > 1 else 0.0
    if kind == "SD":
        return sd
    if kind == "SEM":
        return sd / np.sqrt(n)
    if kind == "95% CI":
        return stats.t.ppf(0.975, max(n - 1, 1)) * sd / np.sqrt(n)
    return 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  example tables
# ═════════════════════════════════════════════════════════════════════════════

def _ex_treat():
    r = rng(11)
    rows = []
    for g, mu, sd in [("Control", 10, 1.3), ("Drug A", 12.4, 1.7), ("Drug B", 14.2, 2.3), ("Drug C", 11, 1.1)]:
        for v in r.normal(mu, sd, 20):
            rows.append((g, round(float(v), 2)))
    return pd.DataFrame(rows, columns=["Treatment", "Expression"])


def _ex_violin():
    r = rng(5)
    rows = []
    gen = {"WT": lambda: r.normal(5, 1, 40),
           "Het": lambda: r.normal(6, 1.2, 40),
           "KO": lambda: np.concatenate([r.normal(6.5, 0.7, 22), r.normal(9, 0.8, 18)])}
    for g, f in gen.items():
        for v in f():
            rows.append((g, round(float(v), 2)))
    return pd.DataFrame(rows, columns=["Genotype", "log2 expression"])


def _ex_paired():
    r = rng(7)
    rows = []
    for i in range(12):
        base = r.normal(50, 8)
        for cond, eff in [("Baseline", 0), ("Week 4", 6), ("Week 12", 10)]:
            rows.append((f"P{i + 1:02d}", cond, round(float(base + eff + r.normal(0, 4)), 1)))
    return pd.DataFrame(rows, columns=["Patient", "Visit", "IL-6 (pg/mL)"])


def _ex_tissues():
    r = rng(3)
    rows = []
    for g, mu, sd in [("Liver", 7.5, 1.0), ("Kidney", 6.8, 1.2), ("Heart", 5.2, 0.9), ("Lung", 6.0, 1.4),
                      ("Brain", 3.9, 1.1), ("Muscle", 4.6, 0.8), ("Spleen", 5.8, 1.3), ("Adipose", 4.2, 1.0)]:
        for v in r.normal(mu, sd, 25):
            rows.append((g, round(float(v), 2)))
    return pd.DataFrame(rows, columns=["Tissue", "log2 TPM"])


def _ex_split():
    r = rng(9)
    rows = []
    for ct, mu in [("T cells", 4), ("B cells", 5.5), ("NK cells", 3.5), ("Monocytes", 6.5)]:
        for gt, d in [("WT", 0), ("KO", 1.2)]:
            for v in r.normal(mu + d, 0.9, 40):
                rows.append((ct, gt, round(float(v), 2)))
    return pd.DataFrame(rows, columns=["Cell type", "Genotype", "Expression"])


def _ex_ridges():
    r = rng(4)
    rows = []
    for g, f in [("Day 0", lambda: r.normal(2, 0.6, 150)),
                 ("Day 1", lambda: r.normal(2.8, 0.8, 150)),
                 ("Day 3", lambda: np.concatenate([r.normal(3, 0.6, 80), r.normal(5, 0.6, 70)])),
                 ("Day 5", lambda: np.concatenate([r.normal(3.2, 0.6, 50), r.normal(5.6, 0.7, 100)])),
                 ("Day 7", lambda: r.normal(6, 0.8, 150)),
                 ("Day 14", lambda: r.normal(6.4, 1.0, 150))]:
        for v in f():
            rows.append((g, round(float(v), 3)))
    return pd.DataFrame(rows, columns=["Time point", "log2 CD69 intensity"])


def _ex_kde():
    r = rng(8)
    rows = []
    for g, f in [("Naive", lambda: r.normal(3, 0.8, 200)),
                 ("Activated", lambda: r.normal(5.5, 1.1, 200)),
                 ("Exhausted", lambda: np.concatenate([r.normal(4, 0.6, 120), r.normal(7, 0.7, 80)]))]:
        for v in f():
            rows.append((g, round(float(v), 3)))
    return pd.DataFrame(rows, columns=["T cell state", "PD-1 log2 intensity"])


def _ex_hist():
    r = rng(12)
    rows = [(round(float(v), 2), "Batch 1") for v in r.normal(14.5, 2.2, 300)]
    rows += [(round(float(v), 2), "Batch 2") for v in r.normal(18.0, 2.6, 260)]
    return pd.DataFrame(rows, columns=["Cell diameter (um)", "Batch"])


def _ex_qq():
    r = rng(21)
    v = np.concatenate([r.normal(0, 1, 110), r.standard_t(3, 30)])
    return pd.DataFrame({"Sample": [f"S{i + 1:03d}" for i in range(len(v))],
                         "Model residual": np.round(v, 3)})


def _ex_tidy():
    r = rng(14)
    rows = []
    for g, mu in [("siCtrl", 1.0), ("siGene1", 0.45), ("siGene2", 0.62)]:
        for v in r.normal(mu, 0.12, 6):
            rows.append((g, round(float(v), 3)))
    return pd.DataFrame(rows, columns=["Knock-down", "Relative mRNA"])


# ═════════════════════════════════════════════════════════════════════════════
#  distribution plots
# ═════════════════════════════════════════════════════════════════════════════

_GV = [Role("group", "Groups", "cat"), Role("value", "Values (one row per observation)", "num")]
_DESC_LONG = "Long table: a group column and a numeric value column, one row per observation."


@plot("box_with_jitter_123", "Box plot with jitter", CD, number="123", example=_ex_treat, roles=_GV,
      opts=[Opt("points", "Show points", "bool", True),
            Opt("jitter", "Jitter width", "float", 0.16, lo=0, hi=0.5, step=0.02),
            Opt("width", "Box width", "float", 0.6, lo=0.1, hi=1.0, step=0.05),
            Opt("fill", "Box fill", "choice", "light", choices=["light", "solid", "none"]),
            Opt("notch", "Notched boxes", "bool", False),
            Opt("psize", "Point size", "float", 20, lo=1, hi=200, step=2)],
      desc=_DESC_LONG)
def box_with_jitter_123(fig, df, c, o, st):
    groups, data = _long(df, c)
    ax = st.ax(fig.add_subplot(), "y")
    cols = st.colors(len(groups))
    _box(ax, data, range(len(groups)), cols, o["width"], fill=o["fill"], notch=o["notch"],
         fliers=not o["points"])
    if o["points"]:
        for i, v in enumerate(data):
            x = i + rng(i).uniform(-o["jitter"], o["jitter"], len(v))
            ax.scatter(x, v, s=o["psize"], color=cols[i], edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3)
    _cat_axis(ax, groups)
    st.labels(ax, c["group"], c["value"])


@plot("beeswarm_box_027", "Beeswarm box plot", CD, number="027", example=_ex_treat, roles=_GV,
      opts=[Opt("box", "Show boxes", "bool", True),
            Opt("width", "Box width", "float", 0.6, lo=0.1, hi=1.0, step=0.05),
            Opt("psize", "Point size", "float", 26, lo=2, hi=200, step=2),
            Opt("maxw", "Max swarm half-width", "float", 0.38, lo=0.05, hi=0.5, step=0.02),
            Opt("colour_points", "Colour points by group", "bool", True)],
      desc=_DESC_LONG + " Points are spread so that they do not overlap.")
def beeswarm_box_027(fig, df, c, o, st):
    groups, data = _long(df, c)
    lo, hi, R = _span(data)
    ax = st.ax(fig.add_subplot(), "y")
    cols = st.colors(len(groups))
    if o["box"]:
        _box(ax, data, range(len(groups)), cols, o["width"], fill="none", lw=1.1, zorder=4)
        for art in ax.patches:
            art.set_facecolor("none")
    dx, dy = _swarm_steps(fig, len(groups), R, o["psize"])
    for i, v in enumerate(data):
        xs = _swarm(v, dx, dy, o["maxw"])
        col = cols[i] if o["colour_points"] else "#777777"
        ax.scatter(i + xs, v, s=o["psize"], color=col, edgecolor="white", linewidth=0.5, alpha=0.95, zorder=3)
    _cat_axis(ax, groups)
    st.labels(ax, c["group"], c["value"])


@plot("boxplot_lines_047", "Box plot with paired lines", CD, number="047", example=_ex_paired,
      roles=[Role("subject", "Subject (pairs rows)", "cat"), Role("group", "Condition", "cat"),
             Role("value", "Value", "num")],
      opts=[Opt("colour_lines", "Line colour", "choice", "by direction", choices=["by direction", "grey", "by subject"]),
            Opt("test", "Paired test (adjacent conditions)", "choice", "paired t-test",
                choices=["none", "paired t-test", "Wilcoxon"]),
            Opt("label", "Show", "choice", "stars", choices=["stars", "p-value"]),
            Opt("width", "Box width", "float", 0.45, lo=0.1, hi=1.0, step=0.05)],
      desc="Long table: subject ID, condition and value; rows with the same subject are joined by a line.")
def boxplot_lines_047(fig, df, c, o, st):
    from scipy import stats
    d = pd.DataFrame({"s": as_str(df, c["subject"]), "g": as_str(df, c["group"]), "v": num(df, c["value"])})
    d = d[np.isfinite(d["v"])]
    if d.empty:
        raise ValueError("No numeric values.")
    groups = cats(d, "g")
    wide = d.groupby(["s", "g"], sort=False)["v"].mean().unstack("g").reindex(columns=groups)
    ax = st.ax(fig.add_subplot(), "y")
    cols = st.colors(len(groups))
    data = [d.loc[d["g"] == g, "v"].to_numpy() for g in groups]
    _box(ax, data, range(len(groups)), cols, o["width"], fill="light", zorder=2)
    up, down = st.color(0), st.color(1)
    scols = st.colors(len(wide))
    for k, (_, row) in enumerate(wide.iterrows()):
        vals = row.to_numpy(float)
        ok = np.isfinite(vals)
        if ok.sum() < 1:
            continue
        x = np.arange(len(groups))[ok]
        y = vals[ok]
        if o["colour_lines"] == "grey":
            col = "#8c8c8c"
        elif o["colour_lines"] == "by subject":
            col = scols[k]
        else:
            col = up if y[-1] >= y[0] else down
        ax.plot(x, y, color=col, lw=1.0, alpha=0.75, zorder=3)
        ax.scatter(x, y, s=22, color=col, edgecolor="white", linewidth=0.5, zorder=4)
    if o["colour_lines"] == "by direction":
        from matplotlib.lines import Line2D
        st.legend(ax, [Line2D([], [], color=up, marker="o", lw=1.2), Line2D([], [], color=down, marker="o", lw=1.2)],
                  ["Increase", "Decrease"])
    if o["test"] != "none" and len(groups) > 1:
        lo, hi, R = _span(data)
        pairs, texts = [], []
        for i in range(len(groups) - 1):
            pr = wide.iloc[:, [i, i + 1]].dropna()
            p = np.nan
            if len(pr) >= 2:
                try:
                    a, b = pr.iloc[:, 0].to_numpy(), pr.iloc[:, 1].to_numpy()
                    p = (stats.ttest_rel(a, b) if o["test"] == "paired t-test" else stats.wilcoxon(a, b)).pvalue
                except Exception:
                    p = np.nan
            if np.isfinite(p):
                pairs.append((i, i + 1))
                texts.append(p_stars(p) if o["label"] == "stars" else _fmt_p(p))
        _brackets(ax, st, pairs, texts, hi, R)
    _cat_axis(ax, groups)
    st.labels(ax, c["group"], c["value"])


@plot("horizontal_box_067", "Horizontal box plot", CD, number="067", example=_ex_tissues, roles=_GV,
      opts=[Opt("sort", "Sort by median", "choice", "descending", choices=["none", "ascending", "descending"]),
            Opt("points", "Show points", "bool", True),
            Opt("fill", "Box fill", "choice", "light", choices=["light", "solid", "none"]),
            Opt("width", "Box height", "float", 0.6, lo=0.1, hi=1.0, step=0.05)],
      desc=_DESC_LONG + " Groups are listed top to bottom.", size=(6.5, 5.5))
def horizontal_box_067(fig, df, c, o, st):
    groups, data = _long(df, c)
    if o["sort"] != "none":
        order = np.argsort([np.median(v) for v in data])
        if o["sort"] == "descending":
            order = order[::-1]
        groups, data = [groups[i] for i in order], [data[i] for i in order]
    ax = st.ax(fig.add_subplot(), "x")
    cols = st.colors(len(groups))
    _box(ax, data, range(len(groups)), cols, o["width"], horiz=True, fill=o["fill"], fliers=not o["points"])
    if o["points"]:
        for i, v in enumerate(data):
            y = i + rng(i).uniform(-0.15, 0.15, len(v))
            ax.scatter(v, y, s=12, color=cols[i], edgecolor="black", linewidth=0.3, alpha=0.8, zorder=3)
    _cat_axis(ax, groups, horiz=True)
    st.labels(ax, c["value"], c["group"])


_TESTS = ["Welch t-test", "Student t-test", "Mann-Whitney U"]


@plot("violin_114", "Violin plot", CD, number="114", example=_ex_violin, roles=_GV,
      opts=[Opt("inner", "Inside the violin", "choice", "box", choices=["box", "quartiles", "points", "none"]),
            Opt("compare", "Compare groups", "choice", "all pairs", choices=["none", "all pairs", "vs first group", "adjacent"]),
            Opt("test", "Test", "choice", "Welch t-test", choices=_TESTS),
            Opt("label", "Show", "choice", "stars", choices=["stars", "p-value"]),
            Opt("bw", "Smoothing (bandwidth ×)", "float", 1.0, lo=0.1, hi=5, step=0.1),
            Opt("width", "Violin width", "float", 0.8, lo=0.2, hi=1.0, step=0.05)],
      desc=_DESC_LONG)
def violin_114(fig, df, c, o, st):
    groups, data = _long(df, c)
    lo, hi, R = _span(data)
    ax = st.ax(fig.add_subplot(), "y")
    cols = st.colors(len(groups))
    for i, v in enumerate(data):
        half = _violin(ax, v, i, o["width"], cols[i], bw=o["bw"], alpha=0.8)
        if o["inner"] == "box":
            _minibox(ax, v, i, w=0.07 * o["width"] / 0.8)
        elif o["inner"] == "quartiles":
            _quartile_lines(ax, v, i, half)
        elif o["inner"] == "points":
            ax.scatter(i + rng(i).uniform(-0.06, 0.06, len(v)), v, s=8, color="black", alpha=0.6, zorder=4)
    pairs = _pairs(len(groups), o["compare"])
    ps = [_two_sample(data[a], data[b], o["test"]) for a, b in pairs]
    keep = [k for k, p in enumerate(ps) if np.isfinite(p)]
    _brackets(ax, st, [pairs[k] for k in keep],
              [p_stars(ps[k]) if o["label"] == "stars" else _fmt_p(ps[k]) for k in keep], hi + 0.08 * R, R)
    _cat_axis(ax, groups)
    st.labels(ax, c["group"], c["value"])


@plot("split_violin_068", "Split violin", CD, number="068", example=_ex_split,
      roles=[Role("x", "Categories (x)", "cat"), Role("hue", "Split by (2 levels)", "cat"), Role("value", "Value", "num")],
      opts=[Opt("inner", "Inside", "choice", "quartiles", choices=["quartiles", "none"]),
            Opt("test", "Test between halves", "choice", "none", choices=["none"] + _TESTS),
            Opt("bw", "Smoothing (bandwidth ×)", "float", 1.0, lo=0.1, hi=5, step=0.1),
            Opt("width", "Violin width", "float", 0.85, lo=0.2, hi=1.0, step=0.05)],
      desc="Long table: category, a two-level split column (e.g. WT / KO) and a value; each violin shows "
           "one level on each side.")
def split_violin_068(fig, df, c, o, st):
    d = pd.DataFrame({"x": as_str(df, c["x"]), "h": as_str(df, c["hue"]), "v": num(df, c["value"])})
    d = d[np.isfinite(d["v"])]
    if d.empty:
        raise ValueError("No numeric values.")
    xs, hs = cats(d, "x"), cats(d, "h")[:2]
    lo, hi, R = _span([d["v"].to_numpy()])
    ax = st.ax(fig.add_subplot(), "y")
    cols = st.colors(len(hs))
    from matplotlib.patches import Patch
    for i, x in enumerate(xs):
        for k, h in enumerate(hs):
            v = d.loc[(d["x"] == x) & (d["h"] == h), "v"].to_numpy()
            if len(v) == 0:
                continue
            side = (-1 if k == 0 else 1) if len(hs) == 2 else 0
            half = _violin(ax, v, i, o["width"], cols[k], side=side, bw=o["bw"], alpha=0.85)
            if o["inner"] == "quartiles":
                _quartile_lines(ax, v, i, half, side=side)
        if o["test"] != "none" and len(hs) == 2:
            a = d.loc[(d["x"] == x) & (d["h"] == hs[0]), "v"]
            b = d.loc[(d["x"] == x) & (d["h"] == hs[1]), "v"]
            p = _two_sample(a, b, o["test"])
            if np.isfinite(p):
                top = max(a.max(), b.max()) + 0.1 * R
                ax.text(i, top, p_stars(p), ha="center", va="bottom", fontsize=st.font * 0.8)
                ax.set_ylim(top=max(ax.get_ylim()[1], top + 0.1 * R))
    _cat_axis(ax, xs)
    st.legend(ax, [Patch(facecolor=cols[k], edgecolor="black", alpha=0.85) for k in range(len(hs))], hs,
              title=c["hue"])
    st.labels(ax, c["x"], c["value"])


@plot("violin_stats_113", "Violin plot with statistics", CD, number="113", example=_ex_treat, roles=_GV,
      opts=[Opt("test", "Overall test", "choice", "Kruskal-Wallis", choices=["Kruskal-Wallis", "One-way ANOVA"]),
            Opt("compare", "Pairwise comparisons", "choice", "vs first group",
                choices=["none", "all pairs", "vs first group", "adjacent"]),
            Opt("adjust", "p adjustment", "choice", "Holm", choices=["none", "Holm", "Bonferroni"]),
            Opt("label", "Show", "choice", "p-value", choices=["p-value", "stars"]),
            Opt("points", "Show points", "bool", True)],
      desc=_DESC_LONG + " The overall p (title) and pairwise p values are computed from the data.")
def violin_stats_113(fig, df, c, o, st):
    from scipy import stats
    groups, data = _long(df, c)
    lo, hi, R = _span(data)
    ax = st.ax(fig.add_subplot(), "y")
    cols = st.colors(len(groups))
    for i, v in enumerate(data):
        _violin(ax, v, i, 0.8, cols[i], alpha=0.35, ec=cols[i], lw=1.4)
        _minibox(ax, v, i, w=0.06)
        if o["points"]:
            ax.scatter(i + rng(i).uniform(-0.2, 0.2, len(v)), v, s=10, color=cols[i], alpha=0.8, zorder=3,
                       edgecolor="none")
    ok = [v for v in data if len(v) >= 2]
    title = ""
    if len(ok) >= 2:
        try:
            p = (stats.kruskal(*ok) if o["test"] == "Kruskal-Wallis" else stats.f_oneway(*ok)).pvalue
            title = f"{o['test']}, {_fmt_p(p)}"
        except Exception:
            pass
    pw = "Mann-Whitney U" if o["test"] == "Kruskal-Wallis" else "Welch t-test"
    pairs = _pairs(len(groups), o["compare"])
    ps = _adjust([_two_sample(data[a], data[b], pw) for a, b in pairs], o["adjust"])
    keep = [k for k, p in enumerate(ps) if np.isfinite(p)]
    _brackets(ax, st, [pairs[k] for k in keep],
              [p_stars(ps[k]) if o["label"] == "stars" else _fmt_p(ps[k]) for k in keep], hi + 0.08 * R, R)
    _cat_axis(ax, groups)
    st.labels(ax, c["group"], c["value"], title)


@plot("half_violin_097", "Half violin with box and points", CD, number="097", example=_ex_violin, roles=_GV,
      opts=[Opt("points", "Show points", "bool", True), Opt("box", "Show box", "bool", True),
            Opt("bw", "Smoothing (bandwidth ×)", "float", 1.0, lo=0.1, hi=5, step=0.1)],
      desc=_DESC_LONG)
def half_violin_097(fig, df, c, o, st):
    groups, data = _long(df, c)
    ax = st.ax(fig.add_subplot(), "y")
    cols = st.colors(len(groups))
    for i, v in enumerate(data):
        _violin(ax, v, i, 0.8, cols[i], side=1, offset=0.04, bw=o["bw"], alpha=0.8)
        if o["box"]:
            _box(ax, [v], [i - 0.04], [cols[i]], 0.1, fill="solid", zorder=3, lw=0.9)
        if o["points"]:
            ax.scatter(i - 0.24 + rng(i).uniform(-0.08, 0.08, len(v)), v, s=12, color=cols[i],
                       edgecolor="black", linewidth=0.3, alpha=0.8, zorder=3)
    _cat_axis(ax, groups)
    st.labels(ax, c["group"], c["value"])


@plot("raincloud_093", "Raincloud plot", CD, number="093", example=_ex_violin, roles=_GV,
      opts=[Opt("orient", "Orientation", "choice", "horizontal", choices=["horizontal", "vertical"]),
            Opt("bw", "Smoothing (bandwidth ×)", "float", 1.0, lo=0.1, hi=5, step=0.1),
            Opt("psize", "Point size", "float", 12, lo=1, hi=100, step=1)],
      desc=_DESC_LONG + " Cloud = density, rain = the observations, plus a narrow box.")
def raincloud_093(fig, df, c, o, st):
    groups, data = _long(df, c)
    horiz = o["orient"] == "horizontal"
    s = -1 if horiz else 1
    ax = st.ax(fig.add_subplot(), "x" if horiz else "y")
    cols = st.colors(len(groups))
    for i, v in enumerate(data):
        _violin(ax, v, i, 1.2, cols[i], side=s, offset=0.0, horiz=horiz, bw=o["bw"], alpha=0.8)
        _box(ax, [v], [i - s * 0.1], [cols[i]], 0.1, horiz=horiz, fill="light", zorder=3, lw=0.9)
        p = i - s * 0.28 + rng(i).uniform(-0.07, 0.07, len(v))
        if horiz:
            ax.scatter(v, p, s=o["psize"], color=cols[i], edgecolor="none", alpha=0.7, zorder=3)
        else:
            ax.scatter(p, v, s=o["psize"], color=cols[i], edgecolor="none", alpha=0.7, zorder=3)
    _cat_axis(ax, groups, horiz=horiz)
    if horiz:
        ax.set_ylim(len(groups) - 0.55, -0.75)
        st.labels(ax, c["value"], c["group"])
    else:
        ax.set_xlim(-0.55, len(groups) - 0.25)
        st.labels(ax, c["group"], c["value"])


@plot("ridgeline_045", "Ridgeline plot (joyplot)", CD, number="045", example=_ex_ridges, roles=_GV,
      opts=[Opt("overlap", "Overlap", "float", 1.6, lo=0.3, hi=5, step=0.1),
            Opt("colour", "Colour by", "choice", "palette", choices=["palette", "colour map (median)", "single"]),
            Opt("median", "Mark the median", "bool", True),
            Opt("bw", "Smoothing (bandwidth ×)", "float", 1.0, lo=0.1, hi=5, step=0.1)],
      desc=_DESC_LONG + " One density curve per group, stacked top to bottom.", size=(6.5, 5.5))
def ridgeline_045(fig, df, c, o, st):
    import matplotlib as mpl
    groups, data = _long(df, c)
    lo, hi, R = _span(data)
    x = np.linspace(lo - 0.1 * R, hi + 0.1 * R, 300)
    n = len(groups)
    ax = st.ax(fig.add_subplot(), "x")
    ax.spines["left"].set_visible(False)
    meds = np.array([np.median(v) for v in data])
    if o["colour"] == "colour map (median)":
        cm = mpl.colormaps[st.cmap]
        nm = (meds - meds.min()) / ((meds.max() - meds.min()) or 1)
        cols = [cm(0.15 + 0.7 * t) for t in nm]
    elif o["colour"] == "single":
        cols = [st.color(0)] * n
    else:
        cols = st.colors(n)
    for i, v in enumerate(data):
        base = n - 1 - i
        d = _kde(v, x, o["bw"])
        d = d / d.max() * o["overlap"] if d.max() > 0 else d
        ax.fill_between(x, base, base + d, color=cols[i], alpha=0.85, zorder=2 * i + 1, linewidth=0)
        ax.plot(x, base + d, color="white", lw=1.6, zorder=2 * i + 1.5)
        ax.plot(x, base + d, color="#222222", lw=0.7, zorder=2 * i + 1.6)
        ax.axhline(base, color="#888888", lw=0.5, zorder=2 * i + 1.7)
        if o["median"]:
            m = np.median(v)
            ax.plot([m, m], [base, base + float(np.interp(m, x, d))], color="#222222", lw=1.0, ls="--",
                    zorder=2 * i + 1.8)
    ax.set_yticks(range(n), groups[::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.1, n - 1 + o["overlap"] * 1.05)
    ax.set_xlim(x[0], x[-1])
    st.labels(ax, c["value"], "")


@plot("kernel_density_036", "Kernel density", CD, number="036", example=_ex_kde,
      roles=[Role("value", "Values", "num"), Role("group", "Groups (optional)", "cat", optional=True)],
      opts=[Opt("fill", "Fill under the curve", "bool", True), Opt("rug", "Rug (ticks for every value)", "bool", True),
            Opt("mean", "Mean lines", "bool", False),
            Opt("bw", "Smoothing (bandwidth ×)", "float", 1.0, lo=0.1, hi=5, step=0.1)],
      desc="A numeric value column and (optionally) a group column; one density curve per group.")
def kernel_density_036(fig, df, c, o, st):
    groups, data = _long(df, c)
    lo, hi, R = _span(data)
    x = np.linspace(lo - 0.15 * R, hi + 0.15 * R, 400)
    ax = st.ax(fig.add_subplot(), "y")
    cols = st.colors(len(groups))
    ymax = 0
    for i, v in enumerate(data):
        d = _kde(v, x, o["bw"])
        ymax = max(ymax, d.max())
        ax.plot(x, d, color=cols[i], lw=1.8, label=groups[i] if len(groups) > 1 else None)
        if o["fill"]:
            ax.fill_between(x, 0, d, color=cols[i], alpha=0.25, linewidth=0)
        if o["mean"]:
            ax.axvline(v.mean(), color=cols[i], ls="--", lw=1)
    if o["rug"]:
        for i, v in enumerate(data):
            y0 = -ymax * (0.03 + 0.045 * i)
            ax.plot(np.repeat(v, 3), np.tile([y0, y0 - ymax * 0.035, np.nan], len(v)), color=cols[i], lw=0.6,
                    alpha=0.7)
        ax.set_ylim(-ymax * (0.03 + 0.045 * len(groups)), ymax * 1.08)
    else:
        ax.set_ylim(0, ymax * 1.08)
    ax.set_xlim(x[0], x[-1])
    if len(groups) > 1:
        st.legend(ax, title=c.get("group") or None)
    st.labels(ax, c["value"], "Density")


@plot("histogram_fit_069", "Histogram with fitted curve", CD, number="069", example=_ex_hist,
      roles=[Role("value", "Values", "num"), Role("group", "Groups (optional)", "cat", optional=True)],
      opts=[Opt("bins", "Bins (0 = automatic)", "int", 0, lo=0, hi=200),
            Opt("fit", "Fitted curve", "choice", "normal", choices=["normal", "KDE", "none"]),
            Opt("yaxis", "Y axis", "choice", "count", choices=["count", "density"]),
            Opt("stats", "Show mean / SD", "bool", True)],
      desc="A numeric value column and optionally a group column (overlaid histograms).")
def histogram_fit_069(fig, df, c, o, st):
    from scipy import stats
    groups, data = _long(df, c)
    lo, hi, R = _span(data)
    allv = np.concatenate(data)
    nb = o["bins"] or int(np.clip(np.histogram_bin_edges(allv, "fd").size - 1, 5, 60))
    edges = np.linspace(lo, hi if hi > lo else lo + 1, nb + 1)
    bwid = edges[1] - edges[0]
    x = np.linspace(lo - 0.1 * R, hi + 0.1 * R, 300)
    ax = st.ax(fig.add_subplot(), "y")
    cols = st.colors(len(groups))
    dens = o["yaxis"] == "density"
    multi = len(groups) > 1
    lines = []
    for i, v in enumerate(data):
        lab = groups[i] if multi else None
        ax.hist(v, bins=edges, density=dens, color=cols[i], alpha=0.55 if multi else 0.8, edgecolor="white",
                linewidth=0.6, label=lab)
        scale = 1 if dens else len(v) * bwid
        m, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
        if o["fit"] == "normal" and sd > 0:
            ax.plot(x, stats.norm.pdf(x, m, sd) * scale, color=cols[i] if multi else "#222222", lw=2)
        elif o["fit"] == "KDE":
            ax.plot(x, _kde(v, x) * scale, color=cols[i] if multi else "#222222", lw=2)
        lines.append(f"{groups[i] + ': ' if multi else ''}mean = {m:.3g}, SD = {sd:.3g}, n = {len(v)}")
    if o["stats"]:
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
                fontsize=st.font * 0.75, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9))
        ax.set_ylim(top=ax.get_ylim()[1] * (1.12 + 0.07 * len(groups)))
    if multi:
        st.legend(ax, title=c.get("group") or None, loc="upper right")
    st.labels(ax, c["value"], "Density" if dens else "Count")


@plot("qq_plot_042", "QQ plot", CD, number="042", example=_ex_qq,
      roles=[Role("value", "Values", "num")],
      opts=[Opt("band", "95 % confidence band", "bool", True),
            Opt("line", "Reference line", "choice", "quartiles", choices=["quartiles", "mean / SD", "y = x (standardised)"]),
            Opt("shapiro", "Shapiro-Wilk test", "bool", True)],
      desc="One numeric column; its quantiles are compared with those of a normal distribution.", size=(5.5, 5))
def qq_plot_042(fig, df, c, o, st):
    from scipy import stats
    v = num(df, c["value"]).to_numpy(float)
    v = np.sort(v[np.isfinite(v)])
    n = len(v)
    if n < 3:
        raise ValueError("A QQ plot needs at least 3 values.")
    pp = (np.arange(1, n + 1) - 0.5) / n
    th = stats.norm.ppf(pp)
    if o["line"].startswith("y = x"):
        v = (v - v.mean()) / (v.std(ddof=1) or 1)
        a, b = 0.0, 1.0
    elif o["line"] == "mean / SD":
        a, b = v.mean(), v.std(ddof=1)
    else:
        q1, q3 = np.percentile(v, [25, 75])
        b = (q3 - q1) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
        a = np.median(v)
    ax = st.ax(fig.add_subplot())
    if o["band"]:
        i = np.arange(1, n + 1)
        lo = a + b * stats.norm.ppf(stats.beta.ppf(0.025, i, n - i + 1))
        hi = a + b * stats.norm.ppf(stats.beta.ppf(0.975, i, n - i + 1))
        ax.fill_between(th, lo, hi, color=st.color(1), alpha=0.18, linewidth=0, label="95 % band")
        out = (v < lo) | (v > hi)
    else:
        out = np.zeros(n, bool)
    ax.plot(th[[0, -1]], a + b * th[[0, -1]], color=st.color(1), lw=1.6, label="Reference line")
    ax.scatter(th[~out], v[~out], s=18, color=st.color(0), edgecolor="white", linewidth=0.4, zorder=3,
               label="Observations")
    if out.any():
        ax.scatter(th[out], v[out], s=22, color="#d62728", edgecolor="black", linewidth=0.4, zorder=4,
                   label="Outside band")
    if o["shapiro"] and n <= 5000:
        p = stats.shapiro(v).pvalue
        ax.text(0.98, 0.04, f"Shapiro-Wilk {_fmt_p(p)}\nn = {n}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=st.font * 0.8)
    st.legend(ax, loc="upper left")
    st.labels(ax, "Theoretical quantiles (normal)", f"Sample quantiles ({c['value']})")


@plot("tidy_plot", "Tidy plot (mean ± SEM with points)", CD, number="tidy", example=_ex_tidy, roles=_GV,
      opts=[Opt("center", "Mean shown as", "choice", "bar", choices=["bar", "dash", "dot"]),
            Opt("err", "Error", "choice", "SEM", choices=["SEM", "SD", "95% CI", "none"]),
            Opt("points", "Show points", "bool", True)],
      desc=_DESC_LONG + " A compact, minimal plot in the style of tidyplots.", size=(3.6, 3.8))
def tidy_plot(fig, df, c, o, st):
    groups, data = _long(df, c)
    ax = st.ax(fig.add_subplot(), "")
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)
    cols = st.colors(len(groups))
    for i, v in enumerate(data):
        m, e = v.mean(), _err(v, o["err"])
        if o["center"] == "bar":
            ax.bar(i, m, width=0.6, color=cols[i], alpha=0.35, edgecolor="none", zorder=1)
        elif o["center"] == "dash":
            ax.plot([i - 0.25, i + 0.25], [m, m], color=cols[i], lw=2, zorder=3)
        else:
            ax.scatter([i], [m], s=60, color=cols[i], edgecolor="black", linewidth=0.5, zorder=4)
        if o["err"] != "none" and e > 0:
            ax.errorbar(i, m, yerr=e, color=cols[i], capsize=4, lw=1.2, capthick=1.2, zorder=3)
        if o["points"]:
            ax.scatter(i + rng(i).uniform(-0.12, 0.12, len(v)), v, s=14, color=cols[i], edgecolor="white",
                       linewidth=0.4, zorder=5)
    _cat_axis(ax, groups)
    if o["center"] == "bar":
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    st.labels(ax, "", c["value"])


# ═════════════════════════════════════════════════════════════════════════════
#  scatter & correlation
# ═════════════════════════════════════════════════════════════════════════════

def _xy(df, c, *extra):
    """Numeric x / y (plus optional extra roles) with incomplete rows dropped."""
    d = pd.DataFrame({"x": num(df, c["x"]), "y": num(df, c["y"])})
    for k in extra:
        col = c.get(k) or ""
        if col:
            d[k] = num(df, col) if k in ("size", "z") else as_str(df, col)
    d = d.dropna(subset=["x", "y"] + [k for k in extra if k in d and k in ("size", "z")])
    if d.empty:
        raise ValueError("No rows with numeric x and y.")
    return d


def _polyfit_ci(x, y, xs, deg=1, level=0.95):
    """OLS polynomial fit: coefficients, fitted curve on xs, CI half-width (or None), R², slope p."""
    from scipy import stats
    X, Xs = np.vander(x, deg + 1), np.vander(xs, deg + 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (res ** 2).sum() / sst if sst > 0 else np.nan
    dof = len(x) - deg - 1
    if dof <= 0:
        return beta, Xs @ beta, None, r2, np.nan
    cov = (res ** 2).sum() / dof * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Xs, cov, Xs), 0))
    t = stats.t.ppf(0.5 + level / 2, dof)
    sb = np.sqrt(max(cov[-2, -2], 0))
    p = 2 * stats.t.sf(abs(beta[-2] / sb), dof) if sb > 0 else np.nan
    return beta, Xs @ beta, t * se, r2, p


def _eq(beta):
    terms = []
    deg = len(beta) - 1
    for k, b in enumerate(beta):
        pw = deg - k
        s = f"{abs(b):.3g}" + ("x" if pw == 1 else f"x$^{pw}$" if pw > 1 else "")
        terms.append(("−" if b < 0 else "+") + " " + s if terms else ("−" if b < 0 else "") + s)
    return "y = " + " ".join(terms)


def _place_labels(ax, st, xs, ys, texts, fs=None, color="black"):
    """Annotate points with labels pushed apart so that they do not overlap each other (greedy)."""
    if not len(texts):
        return
    fs = fs or st.font * 0.7
    pt = ax.figure.dpi / 72
    P = ax.transData.transform(np.column_stack([xs, ys]))
    boxes = []
    for (px, py), (x, y), t in zip(P, zip(xs, ys), texts):
        w, h = len(str(t)) * fs * 0.56 * pt, fs * 1.2 * pt
        choice = None
        for r in (5, 12, 20, 30, 42, 56):
            for ang in (45, 135, -45, -135, 0, 180, 90, -90):
                dx, dy = r * np.cos(np.radians(ang)), r * np.sin(np.radians(ang))
                x0 = px + dx * pt - (w if dx < -0.5 else w / 2 if abs(dx) <= 0.5 else 0)
                y0 = py + dy * pt - (h if dy < -0.5 else h / 2 if abs(dy) <= 0.5 else 0)
                b = (x0, y0, x0 + w, y0 + h)
                hit = any(not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3]) for o in boxes)
                hit = hit or any(b[0] - 2 < qx < b[2] + 2 and b[1] - 2 < qy < b[3] + 2 for qx, qy in P)
                if not hit:
                    choice = (dx, dy, b, r)
                    break
            if choice:
                break
        if choice is None:
            dx, dy = 5 * 0.7, 5 * 0.7
            choice = (dx, dy, (px, py, px + w, py + h), 5)
        dx, dy, b, r = choice
        boxes.append(b)
        ax.annotate(str(t), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fs, zorder=6, color=color,
                    ha="left" if dx > 0.5 else "right" if dx < -0.5 else "center",
                    va="bottom" if dy > 0.5 else "top" if dy < -0.5 else "center",
                    arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5, shrinkA=0, shrinkB=2) if r > 5 else None)


def _group_colors(d, st, key="group"):
    if key in d:
        gs = cats(d, key)
        return gs, dict(zip(gs, st.colors(len(gs))))
    return [], {}


def _size_scale(s, lo=15, hi=220):
    s = np.asarray(s, float)
    a, b = np.nanmin(s), np.nanmax(s)
    return lo + (s - a) / ((b - a) or 1) * (hi - lo), (a, b)


def _size_handles(vals_range, lo=15, hi=220, n=3):
    from matplotlib.lines import Line2D
    a, b = vals_range
    vals = np.unique(np.round(np.linspace(a, b, n), 3)) if b > a else [a]
    hs = []
    for v in vals:
        s = lo + (v - a) / ((b - a) or 1) * (hi - lo)
        hs.append(Line2D([], [], marker="o", ls="", color="#999999", markeredgecolor="black",
                         markeredgewidth=0.4, markersize=np.sqrt(s)))
    return hs, [f"{v:.3g}" for v in vals]


def _ex_scatter():
    r = rng(31)
    rows = []
    for comp, n, sl in [("Cytoplasm", 45, 0.8), ("Nucleus", 35, 0.6), ("Mitochondria", 25, 1.0)]:
        x = r.uniform(2, 12, n)
        y = 4 + sl * x + r.normal(0, 1.2, n)
        pep = np.clip(np.round(r.gamma(2, 6, n)), 1, None)
        rows += [(round(a, 2), round(b, 2), comp, int(p)) for a, b, p in zip(x, y, pep)]
    return pd.DataFrame(rows, columns=["RNA (log2 TPM)", "Protein (log2 LFQ)", "Compartment", "Peptides"])


def _ex_dose():
    r = rng(32)
    x = np.round(np.sort(r.uniform(0, 50, 40)), 1)
    y = 12 + 1.35 * x + r.normal(0, 7, 40)
    return pd.DataFrame({"Dose (mg/kg)": x, "Plasma level (ng/mL)": np.round(y, 1)})


def _ex_corr2():
    r = rng(33)
    x = r.lognormal(1.5, 0.6, 50)
    y = 2 * np.log(x) + r.normal(0, 0.5, 50)
    return pd.DataFrame({"Patient": [f"Pt{i + 1:02d}" for i in range(50)],
                         "Gene A (TPM)": np.round(x, 2), "Gene B (log2 TPM)": np.round(y, 2)})


def _ex_marginal():
    r = rng(34)
    rows = []
    for g, (mx, my), cov in [("Responder", (5, 7), [[1, 0.6], [0.6, 1.2]]),
                             ("Non-responder", (7, 4.5), [[1.2, -0.3], [-0.3, 0.8]])]:
        for a, b in r.multivariate_normal([mx, my], cov, 70):
            rows.append((round(float(a), 2), round(float(b), 2), g))
    return pd.DataFrame(rows, columns=["CD8 score", "IFN-γ score", "Response"])


def _ex_3d():
    r = rng(35)
    rows = []
    for g, m in [("CD4 T", (6, 1.5, 1)), ("CD8 T", (1.5, 6, 1)), ("B cell", (1, 1.2, 6)), ("NK", (1.5, 2.5, 1.5))]:
        for v in r.normal(m, 0.7, (40, 3)):
            rows.append((*np.round(v, 2), g))
    return pd.DataFrame(rows, columns=["CD4", "CD8", "CD19", "Population"])


def _ex_quadrant():
    r = rng(36)
    n = 500
    x = r.normal(0, 1.3, n)
    y = 0.6 * x + r.normal(0, 1.0, n)
    return pd.DataFrame({"Gene": [f"Gene{i + 1}" for i in range(n)],
                         "RNA log2FC": np.round(x, 3), "Protein log2FC": np.round(y, 3)})


def _ex_fc():
    r = rng(37)
    n = 600
    a = r.lognormal(5, 1.6, n)
    fc = np.exp(r.normal(0, 0.45, n))
    k = r.choice(n, 70, replace=False)
    fc[k[:40]] *= r.uniform(2.5, 8, 40)
    fc[k[40:]] /= r.uniform(2.5, 8, 30)
    b = a * fc * np.exp(r.normal(0, 0.1, n))
    return pd.DataFrame({"Gene": [f"Gene{i + 1}" for i in range(n)],
                         "Control (counts)": np.round(a, 1), "Treated (counts)": np.round(b, 1)})


def _ex_cormat():
    r = rng(38)
    n = 30
    f1, f2 = r.normal(0, 1, n), r.normal(0, 1, n)
    genes = {"MKI67": f1, "TOP2A": f1, "CCNB1": f1, "CD3E": f2, "CD8A": f2, "GZMB": f2 * 0.6 - f1 * 0.3,
             "ACTB": np.zeros(n)}
    d = {"Sample": [f"S{i + 1:02d}" for i in range(n)]}
    for g, f in genes.items():
        d[g] = np.round(8 + f + r.normal(0, 0.6, n), 2)
    return pd.DataFrame(d)


_TIMES = ["0 h", "2 h", "6 h", "12 h", "24 h", "48 h"]


def _ex_lines():
    r = rng(39)
    pats = {"Early": [0, 3, 2.5, 1.5, 1, 0.5], "Late": [0, 0.3, 1, 2, 3, 3.5],
            "Repressed": [0, -0.5, -1.5, -2.5, -3, -3]}
    rows = []
    k = 0
    for cl, p in pats.items():
        for _ in range(4):
            k += 1
            v = 6 + np.array(p) * r.uniform(0.7, 1.2) + r.normal(0, 0.25, len(p))
            rows.append([f"Gene{k:02d}", cl] + list(np.round(v, 2)))
    return pd.DataFrame(rows, columns=["Gene", "Cluster"] + _TIMES)


def _ex_trend():
    r = rng(40)
    pats = [[0, 2, 3, 2.5, 1.5, 1], [0, 0.2, 0.8, 1.8, 2.8, 3.2], [3, 2.6, 1.6, 0.7, 0.2, 0],
            [1, 0, 0, 0.3, 1.2, 2.4]]
    rows = []
    k = 0
    for p in pats:
        for _ in range(40):
            k += 1
            v = 5 + np.array(p) * r.uniform(0.6, 1.4) + r.normal(0, 0.35, len(p))
            rows.append([f"G{k:03d}"] + list(np.round(v, 2)))
    rows = [rows[i] for i in r.permutation(len(rows))]
    return pd.DataFrame(rows, columns=["Gene"] + _TIMES)


@plot("scatter_053", "Scatter plot", CS, number="053", example=_ex_scatter,
      roles=[Role("x", "X", "num"), Role("y", "Y", "num"), Role("group", "Colour by (optional)", "cat", optional=True),
             Role("size", "Point size by (optional)", "num", optional=True)],
      opts=[Opt("psize", "Point size", "float", 40, lo=2, hi=300, step=2),
            Opt("alpha", "Opacity", "float", 0.8, lo=0.1, hi=1, step=0.05),
            Opt("edge", "Black outline", "bool", True)],
      desc="Two numeric columns (x, y) and optionally a group column (colour) and a numeric column (point size).")
def scatter_053(fig, df, c, o, st):
    from matplotlib.lines import Line2D
    d = _xy(df, c, "group", "size")
    ax = st.ax(fig.add_subplot())
    s = _size_scale(d["size"])[0] if "size" in d else np.full(len(d), o["psize"])
    kw = dict(alpha=o["alpha"], edgecolor="black" if o["edge"] else "none", linewidth=0.4)
    groups, cmap = _group_colors(d, st)
    hs, ls = [], []
    if groups:
        for g in groups:
            m = (d["group"] == g).to_numpy()
            ax.scatter(d["x"][m], d["y"][m], s=s[m], color=cmap[g], **kw)
            hs.append(Line2D([], [], marker="o", ls="", color=cmap[g], markeredgecolor="black", markeredgewidth=0.4,
                             markersize=7))
            ls.append(g)
    else:
        ax.scatter(d["x"], d["y"], s=s, color=st.color(0), **kw)
    if "size" in d:
        sh, sl = _size_handles(_size_scale(d["size"])[1])
        hs.append(Line2D([], [], ls=""))
        ls.append(f"{c['size']}:")
        hs += sh
        ls += sl
    if hs:
        st.legend(ax, hs, ls, title=c.get("group") or None, loc="upper left")
    st.labels(ax, c["x"], c["y"])


@plot("regression_scatter_041", "Regression scatter", CS, number="041", example=_ex_dose,
      roles=[Role("x", "X", "num"), Role("y", "Y", "num")],
      opts=[Opt("deg", "Fit", "choice", "linear", choices=["linear", "quadratic", "cubic"]),
            Opt("ci", "95 % confidence band", "bool", True),
            Opt("pi", "95 % prediction band", "bool", False),
            Opt("eq", "Show equation and R²", "bool", True)],
      desc="Two numeric columns; a least-squares line (or polynomial) with its 95 % confidence band.")
def regression_scatter_041(fig, df, c, o, st):
    from scipy import stats
    d = _xy(df, c)
    x, y = d["x"].to_numpy(), d["y"].to_numpy()
    deg = {"linear": 1, "quadratic": 2, "cubic": 3}[o["deg"]]
    deg = min(deg, max(len(np.unique(x)) - 1, 0))
    ax = st.ax(fig.add_subplot())
    ax.scatter(x, y, s=34, color=st.color(0), edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3,
               label="Observations")
    if deg >= 1:
        xs = np.linspace(x.min(), x.max(), 200)
        beta, fit, ci, r2, p = _polyfit_ci(x, y, xs, deg)
        ax.plot(xs, fit, color=st.color(1), lw=2, zorder=4, label="Fit")
        if o["ci"] and ci is not None:
            ax.fill_between(xs, fit - ci, fit + ci, color=st.color(1), alpha=0.2, linewidth=0, label="95 % CI")
        if o["pi"] and ci is not None:
            X = np.vander(x, deg + 1)
            res = y - X @ beta
            dof = max(len(x) - deg - 1, 1)
            s = np.sqrt((res ** 2).sum() / dof)
            t = stats.t.ppf(0.975, dof)
            pi = np.sqrt((ci / t) ** 2 + s ** 2) * t
            ax.plot(xs, fit - pi, color=st.color(1), ls="--", lw=1, label="95 % PI")
            ax.plot(xs, fit + pi, color=st.color(1), ls="--", lw=1)
        if o["eq"]:
            txt = f"{_eq(beta)}\n$R^2$ = {r2:.3f}" + (f", {_fmt_p(p)}" if deg == 1 else "") + f"\nn = {len(x)}"
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=st.font * 0.85,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb", alpha=0.9))
    st.legend(ax, loc="lower right")
    st.labels(ax, c["x"], c["y"])


@plot("pearson_spearman_049", "Pearson / Spearman correlation scatter", CS, number="049", example=_ex_corr2,
      roles=[Role("x", "X", "num"), Role("y", "Y", "num")],
      opts=[Opt("method", "Coefficients", "choice", "both", choices=["both", "Pearson", "Spearman"]),
            Opt("fit", "Regression line", "bool", True), Opt("ci", "95 % confidence band", "bool", True),
            Opt("logx", "Log scale x", "bool", True)],
      desc="Two numeric columns; Pearson r and Spearman ρ with their p values are shown on the plot.")
def pearson_spearman_049(fig, df, c, o, st):
    from scipy import stats
    d = _xy(df, c)
    if o["logx"]:
        d = d[d["x"] > 0]
    x, y = d["x"].to_numpy(), d["y"].to_numpy()
    ax = st.ax(fig.add_subplot())
    ax.scatter(x, y, s=34, color=st.color(0), edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3)
    fx = np.log10(x) if o["logx"] else x
    if o["fit"] and len(x) >= 3 and np.ptp(fx) > 0:
        xs = np.linspace(fx.min(), fx.max(), 200)
        beta, fit, ci, r2, p = _polyfit_ci(fx, y, xs, 1)
        xp = 10 ** xs if o["logx"] else xs
        ax.plot(xp, fit, color=st.color(1), lw=2, zorder=4)
        if o["ci"] and ci is not None:
            ax.fill_between(xp, fit - ci, fit + ci, color=st.color(1), alpha=0.18, linewidth=0)
    if o["logx"]:
        ax.set_xscale("log")
    lines = []
    if len(x) >= 3 and np.ptp(fx) > 0 and np.ptp(y) > 0:
        if o["method"] in ("both", "Pearson"):
            r, p = stats.pearsonr(fx, y)
            lines.append(f"Pearson r = {r:.2f}, {_fmt_p(p)}" + (" (log x)" if o["logx"] else ""))
        if o["method"] in ("both", "Spearman"):
            r, p = stats.spearmanr(x, y)
            lines.append(f"Spearman ρ = {r:.2f}, {_fmt_p(p)}")
    lines.append(f"n = {len(x)}")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=st.font * 0.85,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb", alpha=0.9))
    st.labels(ax, c["x"], c["y"])


@plot("scatter_marginal_104", "Scatter with marginal histograms", CS, number="104", example=_ex_marginal,
      roles=[Role("x", "X", "num"), Role("y", "Y", "num"), Role("group", "Colour by (optional)", "cat", optional=True)],
      opts=[Opt("marg", "Margins", "choice", "histogram", choices=["histogram", "density", "box"]),
            Opt("bins", "Bins", "int", 20, lo=3, hi=100),
            Opt("psize", "Point size", "float", 26, lo=2, hi=200, step=2)],
      desc="Two numeric columns and optionally a group column; distributions of x and y are shown on the margins.",
      size=(6.5, 6))
def scatter_marginal_104(fig, df, c, o, st):
    d = _xy(df, c, "group")
    groups, cmap = _group_colors(d, st)
    sets = [(g, d[d["group"] == g], cmap[g]) for g in groups] if groups else [(None, d, st.color(0))]
    eng = fig.get_layout_engine()
    if eng is not None and hasattr(eng, "set"):
        eng.set(wspace=0.0, hspace=0.0, w_pad=0.02, h_pad=0.02)
    gs = fig.add_gridspec(2, 2, width_ratios=[4.5, 1], height_ratios=[1, 4.5])
    ax = st.ax(fig.add_subplot(gs[1, 0]))
    tx = fig.add_subplot(gs[0, 0], sharex=ax)
    ty = fig.add_subplot(gs[1, 1], sharey=ax)
    for a in (tx, ty):
        for s in ("top", "right", "left", "bottom"):
            a.spines[s].set_visible(False)
        a.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    for g, sub, col in sets:
        ax.scatter(sub["x"], sub["y"], s=o["psize"], color=col, edgecolor="white", linewidth=0.4, alpha=0.85,
                   label=g)
    ex = np.histogram_bin_edges(d["x"], o["bins"])
    ey = np.histogram_bin_edges(d["y"], o["bins"])
    gx = np.linspace(ex[0], ex[-1], 200)
    gy = np.linspace(ey[0], ey[-1], 200)
    for k, (g, sub, col) in enumerate(sets):
        if o["marg"] == "histogram":
            tx.hist(sub["x"], bins=ex, color=col, alpha=0.55, edgecolor="white", linewidth=0.4)
            ty.hist(sub["y"], bins=ey, color=col, alpha=0.55, edgecolor="white", linewidth=0.4,
                    orientation="horizontal")
        elif o["marg"] == "density":
            fx, fy = _kde(sub["x"], gx), _kde(sub["y"], gy)
            tx.fill_between(gx, 0, fx, color=col, alpha=0.35)
            tx.plot(gx, fx, color=col, lw=1.2)
            ty.fill_betweenx(gy, 0, fy, color=col, alpha=0.35)
            ty.plot(fy, gy, color=col, lw=1.2)
        else:
            _box(tx, [sub["x"].to_numpy()], [k], [col], 0.6, horiz=True, fill="light")
            _box(ty, [sub["y"].to_numpy()], [k], [col], 0.6, fill="light")
    if o["marg"] == "box":
        tx.set_ylim(len(sets) - 0.4, -0.6)
        ty.set_xlim(-0.6, len(sets) - 0.4)
    if groups:
        st.legend(ax, title=c["group"])
    st.labels(ax, c["x"], c["y"])
    if st.opts.get("title"):
        ax.set_title("")
        st.suptitle(fig)


@plot("scatter_3d_091", "3D scatter", CS, number="091", example=_ex_3d,
      roles=[Role("x", "X", "num"), Role("y", "Y", "num"), Role("z", "Z", "num"),
             Role("group", "Colour by (optional)", "cat", optional=True)],
      opts=[Opt("elev", "Elevation (°)", "int", 22, lo=-90, hi=90),
            Opt("azim", "Azimuth (°)", "int", -55, lo=-180, hi=180),
            Opt("psize", "Point size", "float", 28, lo=2, hi=200, step=2),
            Opt("drop", "Shadow on the floor", "bool", False)],
      desc="Three numeric columns (x, y, z) and optionally a group column for colours.", size=(6.5, 5.8))
def scatter_3d_091(fig, df, c, o, st):
    d = _xy(df, c, "z", "group")
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=o["elev"], azim=o["azim"])
    for p in (ax.xaxis, ax.yaxis, ax.zaxis):
        p.set_pane_color((0.97, 0.97, 0.97, 1))
        p._axinfo["grid"].update(color="#dddddd", linewidth=0.6)
    if not st.grid:
        ax.grid(False)
    groups, cmap = _group_colors(d, st)
    sets = [(g, d[d["group"] == g], cmap[g]) for g in groups] if groups else [(None, d, st.color(0))]
    z0 = d["z"].min() - 0.05 * (np.ptp(d["z"]) or 1)
    for g, sub, col in sets:
        ax.scatter(sub["x"], sub["y"], sub["z"], s=o["psize"], color=col, edgecolor="black", linewidth=0.3,
                   alpha=0.85, label=g, depthshade=True)
        if o["drop"]:
            ax.scatter(sub["x"], sub["y"], np.full(len(sub), z0), s=o["psize"] * 0.4, color="#bbbbbb", alpha=0.4)
    ax.tick_params(labelsize=st.font * 0.75, pad=1)
    ax.set_box_aspect(None, zoom=0.86)
    st.labels(ax, c["x"], c["y"])
    ax.set_zlabel(c["z"], fontsize=st.font, labelpad=2)
    if groups:
        st.legend(ax, title=c["group"], loc="upper left")


@plot("quadrant_scatter_010", "Four-quadrant scatter", CS, number="010", example=_ex_quadrant,
      roles=[Role("x", "X (e.g. RNA log2FC)", "num"), Role("y", "Y (e.g. protein log2FC)", "num"),
             Role("label", "Labels (optional)", "cat", optional=True)],
      opts=[Opt("tx", "X threshold (±)", "float", 1.0, lo=0, hi=100, step=0.1),
            Opt("ty", "Y threshold (±)", "float", 1.0, lo=0, hi=100, step=0.1),
            Opt("counts", "Show counts per quadrant", "bool", True),
            Opt("ntop", "Label the most extreme points", "int", 5, lo=0, hi=50)],
      desc="Two numeric columns (e.g. RNA and protein log2 fold changes) and optional labels; points beyond both "
           "thresholds are coloured by quadrant.")
def quadrant_scatter_010(fig, df, c, o, st):
    d = _xy(df, c, "label")
    x, y = d["x"].to_numpy(), d["y"].to_numpy()
    tx, ty = o["tx"], o["ty"]
    ax = st.ax(fig.add_subplot())
    cols = st.colors(4)
    sx, sy = np.abs(x) > tx, np.abs(y) > ty
    names = {(1, 1): "Both up", (-1, -1): "Both down", (1, -1): "X up, Y down", (-1, 1): "X down, Y up"}
    ax.scatter(x[~sx & ~sy], y[~sx & ~sy], s=10, color="#d0d0d0", edgecolor="none", label="Unchanged")
    one = sx ^ sy
    ax.scatter(x[one], y[one], s=12, color="#909090", edgecolor="none", label="One changed")
    both = sx & sy
    for k, (qx, qy) in enumerate([(1, 1), (-1, -1), (1, -1), (-1, 1)]):
        m = both & (np.sign(x) == qx) & (np.sign(y) == qy)
        ax.scatter(x[m], y[m], s=24, color=cols[k], edgecolor="black", linewidth=0.3, label=names[(qx, qy)])
        if o["counts"]:
            ax.text(0.985 if qx > 0 else 0.015, 0.985 if qy > 0 else 0.015, f"n = {int(m.sum())}",
                    transform=ax.transAxes, ha="right" if qx > 0 else "left", va="top" if qy > 0 else "bottom",
                    fontsize=st.font * 0.95, color=cols[k], fontweight="bold")
    for v in (tx, -tx):
        ax.axvline(v, color="#666666", ls="--", lw=0.8)
    for v in (ty, -ty):
        ax.axhline(v, color="#666666", ls="--", lw=0.8)
    ax.axhline(0, color="#999999", lw=0.6)
    ax.axvline(0, color="#999999", lw=0.6)
    # leave room for the corner counts
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(xl[0] - 0.04 * np.ptp(xl), xl[1] + 0.04 * np.ptp(xl))
    ax.set_ylim(yl[0] - 0.08 * np.ptp(yl), yl[1] + 0.08 * np.ptp(yl))
    if o["ntop"] and "label" in d:
        score = np.where(both, np.hypot(x, y), -1)
        top = [i for i in np.argsort(-score)[:o["ntop"]] if score[i] >= 0]
        _place_labels(ax, st, x[top], y[top], d["label"].iloc[top].tolist())
    st.legend(ax, loc="center left", bbox_to_anchor=(1.01, 0.5)) if st.legend_mode == "best" else st.legend(ax)
    st.labels(ax, c["x"], c["y"])


@plot("fold_change_scatter_044", "Fold-change scatter", CS, number="044", example=_ex_fc,
      roles=[Role("x", "Condition A (x)", "num"), Role("y", "Condition B (y)", "num"),
             Role("label", "Labels (optional)", "cat", optional=True)],
      opts=[Opt("fc", "Fold-change threshold", "float", 2.0, lo=1, hi=100, step=0.5),
            Opt("log", "Log axes", "bool", True),
            Opt("ntop", "Label the largest changes", "int", 6, lo=0, hi=50)],
      desc="Expression of each feature (row) in condition A and B, e.g. mean counts; points above / below the "
           "± fold-change lines are coloured up / down.", size=(6, 5.6))
def fold_change_scatter_044(fig, df, c, o, st):
    d = _xy(df, c, "label")
    if o["log"]:
        d = d[(d["x"] > 0) & (d["y"] > 0)]
    if d.empty:
        raise ValueError("No positive values for log axes.")
    x, y = d["x"].to_numpy(), d["y"].to_numpy()
    fc = max(o["fc"], 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = y / x
    up, down = ratio >= fc, ratio <= 1 / fc
    ax = st.ax(fig.add_subplot())
    ax.scatter(x[~up & ~down], y[~up & ~down], s=10, color="#c8c8c8", edgecolor="none",
               label=f"Unchanged ({int((~up & ~down).sum())})")
    ax.scatter(x[up], y[up], s=18, color=st.color(0), edgecolor="black", linewidth=0.25, label=f"Up ({int(up.sum())})")
    ax.scatter(x[down], y[down], s=18, color=st.color(1), edgecolor="black", linewidth=0.25,
               label=f"Down ({int(down.sum())})")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    if o["log"]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        g = np.geomspace(lo / 1.3, hi * 1.3, 50)
    else:
        lo = min(lo, 0)
        g = np.linspace(lo, hi * 1.05, 50)
    ax.plot(g, g, color="#444444", lw=1)
    ax.plot(g, g * fc, color="#444444", lw=0.8, ls="--")
    ax.plot(g, g / fc, color="#444444", lw=0.8, ls="--")
    ax.set_xlim(g[0], g[-1])
    ax.set_ylim(g[0], g[-1])
    if o["ntop"] and "label" in d:
        lr = np.abs(np.log(np.where(up | down, ratio, 1.0)))
        top = [i for i in np.argsort(-lr)[:o["ntop"]] if lr[i] > 0]
        _place_labels(ax, st, x[top], y[top], d["label"].iloc[top].tolist())
    st.legend(ax, loc="upper left")
    st.labels(ax, c["x"], c["y"])


def _corr(df, cols, method):
    from scipy import stats
    X = pd.DataFrame({k: num(df, k) for k in cols})
    X = X.loc[:, X.notna().sum() >= 3]
    if X.shape[1] < 2:
        raise ValueError("Select at least two numeric columns with 3 or more values.")
    n = X.shape[1]
    r, p = np.eye(n), np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pr = X.iloc[:, [i, j]].dropna()
            a, b = pr.iloc[:, 0].to_numpy(), pr.iloc[:, 1].to_numpy()
            if len(pr) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
                rr, pp = np.nan, np.nan
            else:
                rr, pp = (stats.pearsonr(a, b) if method == "Pearson" else stats.spearmanr(a, b))
            r[i, j] = r[j, i] = rr
            p[i, j] = p[j, i] = pp
    return list(X.columns), r, p


def _corr_order(r, on):
    if not on or len(r) < 3:
        return np.arange(len(r))
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    dist = 1 - np.nan_to_num(r, nan=0.0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    np.fill_diagonal(dist, 0)
    return leaves_list(linkage(squareform(dist, checks=False), "average"))


def _colorbar(fig, mappable, ax, st, label):
    cb = fig.colorbar(mappable, ax=ax, shrink=0.7, aspect=22)
    cb.set_label(label, fontsize=st.font * 0.9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=st.font * 0.8)
    return cb


_CORR_ROLES = [Role("vars", "Numeric columns to correlate", "num", multi=True)]


@plot("correlation_heatmap_082", "Correlation matrix", CS, number="082", example=_ex_cormat, roles=_CORR_ROLES,
      opts=[Opt("method", "Method", "choice", "Pearson", choices=["Pearson", "Spearman"]),
            Opt("tri", "Show", "choice", "lower triangle", choices=["lower triangle", "upper triangle", "full"]),
            Opt("annot", "Write coefficients", "bool", True),
            Opt("cluster", "Order by clustering", "bool", False)],
      desc="Several numeric columns (e.g. genes) measured on the same rows (samples); every pair is correlated.",
      size=(6.2, 5.4), axis_labels=False)
def correlation_heatmap_082(fig, df, c, o, st):
    import matplotlib as mpl
    names, r, p = _corr(df, c["vars"], o["method"])
    idx = _corr_order(r, o["cluster"])
    names, r = [names[i] for i in idx], r[np.ix_(idx, idx)]
    n = len(names)
    mask = np.zeros_like(r, bool)
    if o["tri"] == "lower triangle":
        mask = np.triu(np.ones_like(r, bool), 1)
    elif o["tri"] == "upper triangle":
        mask = np.tril(np.ones_like(r, bool), -1)
    ax = fig.add_subplot()
    cm = mpl.colormaps[st.cmap]
    im = ax.imshow(np.ma.array(r, mask=mask | ~np.isfinite(r)), cmap=cm, vmin=-1, vmax=1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(n), names, rotation=45, ha="right", fontsize=st.font * 0.85)
    ax.set_yticks(range(n), names, fontsize=st.font * 0.85)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    if o["annot"]:
        fs = st.font * (0.8 if n <= 8 else 0.6 if n <= 14 else 0.45)
        for i in range(n):
            for j in range(n):
                if not mask[i, j] and np.isfinite(r[i, j]):
                    rgb = cm((r[i, j] + 1) / 2)[:3]
                    dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.5
                    ax.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", fontsize=fs,
                            color="white" if dark else "black")
    _colorbar(fig, im, ax, st, f"{o['method']} r")
    st.labels(ax, "", "")


@plot("correlation_dot_054", "Correlation dot plot", CS, number="054", example=_ex_cormat, roles=_CORR_ROLES,
      opts=[Opt("method", "Method", "choice", "Spearman", choices=["Pearson", "Spearman"]),
            Opt("tri", "Show", "choice", "full", choices=["full", "lower triangle", "upper triangle"]),
            Opt("stars", "Significance stars", "bool", True),
            Opt("cluster", "Order by clustering", "bool", True)],
      desc="Several numeric columns measured on the same rows; circle size and colour show the correlation.",
      size=(6.2, 5.4), axis_labels=False)
def correlation_dot_054(fig, df, c, o, st):
    import matplotlib as mpl
    names, r, p = _corr(df, c["vars"], o["method"])
    idx = _corr_order(r, o["cluster"])
    names, r, p = [names[i] for i in idx], r[np.ix_(idx, idx)], p[np.ix_(idx, idx)]
    n = len(names)
    keep = np.ones_like(r, bool)
    if o["tri"] == "lower triangle":
        keep = np.tril(keep)
    elif o["tri"] == "upper triangle":
        keep = np.triu(keep)
    ii, jj = np.where(keep & np.isfinite(r))
    ax = fig.add_subplot()
    w, h = fig.get_size_inches()
    cell_pt = min(w * 0.62, h * 0.72) / n * 72
    cm = mpl.colormaps[st.cmap]
    sc = ax.scatter(jj, ii, s=(np.sqrt(np.abs(r[ii, jj])) * cell_pt * 0.9) ** 2, c=r[ii, jj], cmap=cm, vmin=-1,
                    vmax=1, edgecolor="#555555", linewidth=0.3, zorder=3)
    if o["stars"]:
        for i, j in zip(ii, jj):
            if i != j and np.isfinite(p[i, j]) and p[i, j] < 0.05:
                ax.text(j, i, p_stars(p[i, j]), ha="center", va="center_baseline", fontsize=st.font * 0.6,
                        zorder=4, color="white" if abs(r[i, j]) > 0.6 else "black")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n), names, rotation=45, ha="right", fontsize=st.font * 0.85)
    ax.set_yticks(range(n), names, fontsize=st.font * 0.85)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="#dddddd", linewidth=0.8)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_color("#dddddd")
    cb = _colorbar(fig, sc, ax, st, f"{o['method']} r")
    if o["stars"]:
        cb.ax.set_xlabel("* p<0.05\n** p<0.01\n*** p<0.001", fontsize=st.font * 0.65, labelpad=8)
    st.labels(ax, "", "")


def _matrix(df, c, rows="gene", cols="conds"):
    X = pd.DataFrame({k: num(df, k) for k in c[cols]})
    lab = as_str(df, c[rows]) if c.get(rows) else pd.Series([f"Row {i + 1}" for i in range(len(df))],
                                                           index=df.index)
    X.index = lab.to_numpy()
    X = X.dropna(how="all").dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("No numeric values in the selected columns.")
    return X


def _row_attr(df, c, key, labels, keycol="gene"):
    """Values of the optional role *key* aligned to the matrix row *labels* (or None)."""
    col = c.get(key) or ""
    if not col:
        return None
    idx = as_str(df, c[keycol]).to_numpy() if c.get(keycol) else np.array([f"Row {i + 1}" for i in range(len(df))])
    g = pd.Series(as_str(df, col).to_numpy(), index=idx)
    return g[~g.index.duplicated()].reindex(labels).fillna("(blank)").to_numpy()


def _zrows(M):
    M = np.asarray(M, float)
    sd = np.nanstd(M, axis=1, ddof=1, keepdims=True) if M.shape[1] > 1 else np.ones((len(M), 1))
    sd[~np.isfinite(sd) | (sd == 0)] = 1
    return (M - np.nanmean(M, axis=1, keepdims=True)) / sd


@plot("expression_lines_029", "Expression connection lines", CS, number="029", example=_ex_lines,
      roles=[Role("gene", "Gene / feature", "cat"), Role("conds", "Conditions (ordered columns)", "num", multi=True),
             Role("group", "Colour by cluster (optional)", "cat", optional=True)],
      opts=[Opt("scale", "Values", "choice", "as is",
                choices=["as is", "z-score per gene", "log2(x+1)", "relative to first"]),
            Opt("mean", "Bold mean line per group", "bool", True),
            Opt("labels", "Label genes at the line ends", "bool", False)],
      desc="Wide table: one row per gene, one numeric column per condition / time point (in order), optionally a "
           "cluster column for colours.")
def expression_lines_029(fig, df, c, o, st):
    X = _matrix(df, c)
    M = X.to_numpy(float)
    if o["scale"] == "z-score per gene":
        M = _zrows(M)
    elif o["scale"] == "log2(x+1)":
        M = np.log2(np.clip(M, 0, None) + 1)
    elif o["scale"] == "relative to first":
        M = M - M[:, [0]]
    grp = _row_attr(df, c, "group", X.index)
    xs = np.arange(M.shape[1])
    ax = st.ax(fig.add_subplot(), "y")
    if grp is not None:
        gs = list(dict.fromkeys(grp))
        cmap = dict(zip(gs, st.colors(len(gs))))
        for k, row in enumerate(M):
            ax.plot(xs, row, color=cmap[grp[k]], lw=1, alpha=0.55, marker="o", ms=3.5)
        for gname in gs:
            if o["mean"]:
                ax.plot(xs, np.nanmean(M[grp == gname], axis=0), color=cmap[gname], lw=3, marker="o", ms=7,
                        markeredgecolor="black", markeredgewidth=0.6, label=gname, zorder=5)
            else:
                ax.plot([], [], color=cmap[gname], lw=2, marker="o", label=gname)
        st.legend(ax, title=c["group"])
    else:
        cols = st.colors(len(M))
        for k, row in enumerate(M):
            ax.plot(xs, row, color=cols[k], lw=1.4, marker="o", ms=5, label=X.index[k] if len(M) <= 12 else None)
        if len(M) <= 12 and not o["labels"]:
            st.legend(ax)
    if o["labels"]:
        for k, row in enumerate(M):
            ax.text(xs[-1] + 0.08, row[-1], X.index[k], fontsize=st.font * 0.65, va="center")
        ax.set_xlim(-0.2, xs[-1] + 0.6)
    ax.set_xticks(xs, list(X.columns))
    st.labels(ax, "", {"as is": "Expression", "z-score per gene": "z-score", "log2(x+1)": "log2(x+1)",
                       "relative to first": f"Change vs {X.columns[0]}"}[o["scale"]])


@plot("expression_trend_101", "Expression trend clusters", CS, number="101", example=_ex_trend,
      roles=[Role("gene", "Gene / feature", "cat"), Role("conds", "Conditions (ordered columns)", "num", multi=True),
             Role("cluster", "Given clusters (optional)", "cat", optional=True)],
      opts=[Opt("k", "Number of clusters (k-means)", "int", 4, lo=1, hi=20),
            Opt("z", "z-score each gene", "bool", True),
            Opt("ncol", "Panels per row (0 = auto)", "int", 0, lo=0, hi=10)],
      desc="Wide table: one row per gene, one numeric column per condition / time point; genes are grouped by "
           "k-means on their z-scored profiles (or by a given cluster column).", size=(8, 5.5))
def expression_trend_101(fig, df, c, o, st):
    X = _matrix(df, c).dropna()
    if X.empty:
        raise ValueError("No complete rows.")
    M = X.to_numpy(float)
    if o["z"]:
        M = _zrows(M)
    lab = _row_attr(df, c, "cluster", X.index)
    if lab is not None:
        names = list(dict.fromkeys(lab))
    else:
        from sklearn.cluster import KMeans
        k = int(max(1, min(o["k"], len(np.unique(M, axis=0)))))
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(M)
        cen = km.cluster_centers_
        order = sorted(range(k), key=lambda q: (int(np.argmax(cen[q])), -cen[q][-1]))
        remap = {old: f"Cluster {i + 1}" for i, old in enumerate(order)}
        lab = np.array([remap[v] for v in km.labels_])
        names = [remap[q] for q in order]
    n = len(names)
    ncol = o["ncol"] or int(np.ceil(np.sqrt(n)))
    ncol = max(1, min(ncol, n))
    nrow = int(np.ceil(n / ncol))
    axs = fig.subplots(nrow, ncol, sharex=True, sharey=True, squeeze=False).ravel()
    cols = st.colors(n)
    xs = np.arange(M.shape[1])
    for k, name in enumerate(names):
        ax = st.ax(axs[k], "y")
        sub = M[lab == name]
        ax.plot(xs, sub.T, color="#b5b5b5", lw=0.6, alpha=0.6)
        ax.plot(xs, sub.mean(axis=0), color=cols[k], lw=2.6, marker="o", ms=5)
        ax.set_title(f"{name} (n = {len(sub)})", fontsize=st.font * 0.9)
        ax.set_xticks(xs, list(X.columns), rotation=45 if len(xs) > 5 else 0, fontsize=st.font * 0.75)
    for ax in axs[n:]:
        ax.set_visible(False)
    st.suptitle(fig)
    _suplabels(fig, st, "", "z-score" if o["z"] else "Expression")


# ═════════════════════════════════════════════════════════════════════════════
#  heatmaps, clustering & PCA
# ═════════════════════════════════════════════════════════════════════════════

_METHODS = ["average", "complete", "ward", "single"]
_METRICS = ["euclidean", "correlation", "cityblock", "cosine"]


def _filled(M):
    M = np.array(M, float)
    if np.isnan(M).any():
        cm = np.nanmean(M, axis=0)
        cm[~np.isfinite(cm)] = 0
        r, k = np.where(np.isnan(M))
        M[r, k] = cm[k]
    return M


def _linkage(M, method="average", metric="euclidean"):
    """Hierarchical clustering of the rows of M (None when there are fewer than 2 rows)."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    M = _filled(M)
    if len(M) < 2:
        return None
    if method == "ward":
        metric = "euclidean"
    with np.errstate(all="ignore"):
        D = pdist(M, metric)
    D = np.nan_to_num(D, nan=np.nanmax(D) if np.isfinite(D).any() else 1.0)
    return linkage(np.clip(D, 0, None), method)


def _members(Z, lab):
    n = len(Z) + 1
    mem = {i: {lab[i]} for i in range(n)}
    for j, (a, b, _, _) in enumerate(Z):
        mem[n + j] = mem[int(a)] | mem[int(b)]
    return mem


def _tree(Z, k=0, colors=None, default="#444444"):
    """dendrogram coordinates (leaf k at position k), link colours (clusters cut into k), leaf order, labels."""
    from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list
    n = len(Z) + 1
    lab = fcluster(Z, k, "maxclust") if k and k > 1 else np.ones(n, int)
    order = leaves_list(Z)
    seen = list(dict.fromkeys(lab[order]))
    cmap = {cl: (colors[i % len(colors)] if colors else default) for i, cl in enumerate(seen)}
    mem = _members(Z, lab)
    single = k and k > 1
    fn = (lambda nid: cmap[next(iter(mem[nid]))] if single and len(mem[nid]) == 1 else default)
    dd = dendrogram(Z, no_plot=True, link_color_func=fn)
    ic = (np.array(dd["icoord"]) - 5) / 10
    dc = np.array(dd["dcoord"])
    return ic, dc, dd["color_list"], np.array(dd["leaves"]), lab, cmap


def _cut_height(Z, k):
    h = np.sort(Z[:, 2])
    if not k or k < 2 or k > len(h):
        return None
    return (h[-k] + h[-k + 1]) / 2 if k >= 2 else None


def _draw_tree(ax, ic, dc, colors, orient="top", lw=1.0):
    for xs, ys, col in zip(ic, dc, colors):
        if orient in ("left", "right"):
            ax.plot(ys, xs, color=col, lw=lw, solid_capstyle="butt")
        else:
            ax.plot(xs, ys, color=col, lw=lw, solid_capstyle="butt")


def _draw_polar_tree(ax, ic, dc, colors, theta, radius, lw=1.0):
    """Tree in polar axes: theta(pos) gives the angle of a leaf position, radius(h) the radius of a height."""
    for xs, ys, col in zip(ic, dc, colors):
        t1, t2 = theta(xs[0]), theta(xs[3])
        arc = np.linspace(t1, t2, 40)
        tt = np.concatenate([[t1, t1], arc, [t2, t2]])
        rr = np.concatenate([[radius(ys[0]), radius(ys[1])], np.full(40, radius(ys[1])), [radius(ys[2]), radius(ys[3])]])
        ax.plot(tt, rr, color=col, lw=lw, solid_capstyle="butt")


def _heat_norm(M, centre):
    fin = M[np.isfinite(M)]
    if not len(fin):
        return 0, 1
    if centre:
        v = np.max(np.abs(fin)) or 1
        return -v, v
    lo, hi = fin.min(), fin.max()
    return lo, hi if hi > lo else lo + 1


def _scale(M, how):
    if how.startswith("row"):
        return _zrows(M), True
    if how.startswith("col"):
        return _zrows(M.T).T, True
    return M, False


def _ticklabels_fs(st, n):
    return st.font * (0.85 if n <= 20 else 0.7 if n <= 40 else 0.55 if n <= 70 else 0.4)


def _annotate(ax, M, im, st, decimals=1):
    n, m = M.shape
    fs = st.font * (0.75 if max(n, m) <= 10 else 0.6 if max(n, m) <= 18 else 0.45)
    for i in range(n):
        for j in range(m):
            if np.isfinite(M[i, j]):
                rgb = im.cmap(im.norm(M[i, j]))[:3]
                dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.5
                ax.text(j, i, f"{M[i, j]:.{decimals}f}", ha="center", va="center", fontsize=fs,
                        color="white" if dark else "black")


def _ex_heat():
    r = rng(41)
    genes = ["IL6", "TNF", "CXCL8", "IL1B", "CCL2", "NFKBIA", "SOCS3", "MYC", "CCND1", "PCNA", "MKI67", "GAPDH",
             "ACTB", "TP53"]
    eff = np.array([3, 2.5, 2.8, 2.2, 1.8, 1.5, 1.2, -1.5, -1.2, -1.8, -2.2, 0, 0, -0.4])
    base = r.uniform(5, 10, len(genes))
    cols = {}
    for s in ["Ctrl_1", "Ctrl_2", "Ctrl_3", "LPS_1", "LPS_2", "LPS_3"]:
        e = eff if s.startswith("LPS") else 0
        cols[s] = np.round(base + e + r.normal(0, 0.3, len(genes)), 2)
    return pd.DataFrame({"Gene": genes, **cols})


def _ex_cluster():
    r = rng(42)
    samples = ["WT_1", "WT_2", "WT_3", "WT_4", "KO_1", "KO_2", "KO_3", "KO_4"]
    mods = [(10, [1, 1, 1, 1, -1, -1, -1, -1]), (10, [-1, -1, -1, -1, 1, 1, 1, 1]),
            (8, [1, 1, -1, -1, 1, 1, -1, -1]), (8, [0, 0, 0, 0, 0, 0, 0, 0])]
    rows, k = [], 0
    for n, pat in mods:
        for _ in range(n):
            k += 1
            rows.append([f"Gene{k:02d}"] + list(np.round(7 + 1.2 * np.array(pat) * r.uniform(0.5, 1.5)
                                                          + r.normal(0, 0.5, 8), 2)))
    rows = [rows[i] for i in r.permutation(len(rows))]
    return pd.DataFrame(rows, columns=["Gene"] + samples)


def _ex_circ():
    r = rng(43)
    genes = [f"G{k:02d}" for k in range(1, 37)]
    samples = ["Day0", "Day1", "Day3", "Day7", "Day14", "Day28"]
    t = np.arange(6)
    rows = []
    for k, g in enumerate(genes):
        kind = k % 3
        prof = [t * 0.6, -t * 0.5, np.sin(t / 1.6) * 2][kind]
        rows.append([g] + list(np.round(6 + prof * r.uniform(0.6, 1.3) + r.normal(0, 0.4, 6), 2)))
    return pd.DataFrame(rows, columns=["Gene"] + samples)


def _samples_table(first="Sample", seed=44):
    r = rng(seed)
    groups = [("Control", 0), ("IFN-γ", 1), ("IL-4", 2)]
    genes = ["CXCL9", "CXCL10", "IDO1", "GBP1", "STAT1", "ARG1", "MRC1", "CCL17", "RETNLA", "CHIL3", "ACTB", "RPL13"]
    load = np.zeros((3, len(genes)))
    load[1, :5] = r.uniform(2, 3.5, 5)
    load[2, 5:10] = r.uniform(2, 3.5, 5)
    rows = []
    for g, gi in groups:
        for k in range(6):
            v = 6 + load[gi] + r.normal(0, 0.7, len(genes))
            rows.append((f"{g[:4].rstrip('-')}_{k + 1}", g, *np.round(v, 2)))
    d = pd.DataFrame(rows, columns=["Sample", "Group"] + genes)
    return d if first == "Sample" else d[["Group", "Sample"] + genes]


def _ex_dendro():
    return _samples_table("Sample")


def _ex_pca():
    return _samples_table("Group")


def _ex_balloon():
    r = rng(45)
    paths = ["Interferon response", "Antigen presentation", "Cell cycle", "Oxidative phosphorylation",
             "Glycolysis", "TNF signalling", "Apoptosis", "Fatty acid metabolism"]
    cells = ["CD4 T", "CD8 T", "B cell", "NK", "Monocyte"]
    rows = []
    for p in paths:
        for cl in cells:
            rows.append((p, cl, round(float(r.uniform(0.02, 0.3)), 3), round(float(r.gamma(2, 2.2)), 2)))
    return pd.DataFrame(rows, columns=["Pathway", "Cell type", "Gene ratio", "-log10 padj"])


@plot("matrix_heatmap_064", "Matrix heatmap", CH, number="064", example=_ex_heat,
      roles=[Role("gene", "Row labels", "cat"), Role("conds", "Numeric columns", "num", multi=True)],
      opts=[Opt("scale", "Scale", "choice", "none", choices=["none", "row z-score", "column z-score"]),
            Opt("annot", "Write values", "bool", True),
            Opt("decimals", "Decimals", "int", 1, lo=0, hi=4),
            Opt("square", "Square cells", "bool", False)],
      desc="Wide table: one row per feature (label column) and one numeric column per sample.",
      size=(6.2, 6), axis_labels=False)
def matrix_heatmap_064(fig, df, c, o, st):
    X = _matrix(df, c)
    M, centre = _scale(X.to_numpy(float), o["scale"])
    ax = fig.add_subplot()
    lo, hi = _heat_norm(M, centre)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=st.cmap, vmin=lo, vmax=hi, aspect="equal" if o["square"] else "auto")
    n, m = M.shape
    ax.set_xticks(range(m), list(X.columns), rotation=45 if m > 4 else 0, ha="right" if m > 4 else "center",
                  fontsize=_ticklabels_fs(st, m))
    ax.set_yticks(range(n), list(X.index), fontsize=_ticklabels_fs(st, n))
    ax.set_xticks(np.arange(-0.5, m), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0 if max(n, m) < 40 else 0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if o["annot"]:
        _annotate(ax, M, im, st, o["decimals"])
    _colorbar(fig, im, ax, st, {"none": "Value"}.get(o["scale"], "z-score"))
    st.labels(ax, "", "")


@plot("cluster_heatmap_024", "Clustered heatmap", CH, number="024", example=_ex_cluster,
      roles=[Role("gene", "Row labels", "cat"), Role("conds", "Numeric columns (samples)", "num", multi=True)],
      opts=[Opt("scale", "Scale", "choice", "row z-score", choices=["row z-score", "column z-score", "none"]),
            Opt("cluster", "Cluster", "choice", "rows and columns", choices=["rows and columns", "rows", "columns", "none"]),
            Opt("method", "Linkage", "choice", "average", choices=_METHODS),
            Opt("metric", "Distance", "choice", "euclidean", choices=_METRICS),
            Opt("k", "Colour row clusters (k, 0 = off)", "int", 0, lo=0, hi=20),
            Opt("rowlabels", "Row labels", "choice", "auto", choices=["auto", "show", "hide"])],
      desc="Wide table: one row per feature (label column) and one numeric column per sample; rows / columns are "
           "reordered by hierarchical clustering.", size=(6.5, 7), axis_labels=False)
def cluster_heatmap_024(fig, df, c, o, st):
    X = _matrix(df, c)
    M, centre = _scale(X.to_numpy(float), o["scale"])
    n, m = M.shape
    rows_on = o["cluster"] in ("rows and columns", "rows") and n >= 2
    cols_on = o["cluster"] in ("rows and columns", "columns") and m >= 2
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 6], height_ratios=[1.1, 6])
    ax = fig.add_subplot(gs[1, 1])
    ro, co = np.arange(n), np.arange(m)
    corner = fig.add_subplot(gs[0, 0])
    corner.set_axis_off()
    axr = fig.add_subplot(gs[1, 0], sharey=ax)
    axc = fig.add_subplot(gs[0, 1], sharex=ax)
    for a in (axr, axc):
        a.set_axis_off()
    if rows_on:
        Z = _linkage(M, o["method"], o["metric"])
        ic, dc, cl, ro, lab, cmap = _tree(Z, o["k"], st.colors(max(o["k"], 1)))
        _draw_tree(axr, ic, dc, cl, "left", lw=0.9)
        axr.set_xlim(dc.max() * 1.03 or 1, 0)
    if cols_on:
        Z = _linkage(M.T, o["method"], o["metric"])
        ic, dc, cl, co, _, _ = _tree(Z)
        _draw_tree(axc, ic, dc, cl, "top", lw=0.9)
        axc.set_ylim(0, dc.max() * 1.03 or 1)
    P = M[np.ix_(ro, co)]
    lo, hi = _heat_norm(P, centre)
    im = ax.imshow(np.ma.masked_invalid(P), cmap=st.cmap, vmin=lo, vmax=hi, aspect="auto")
    ax.set_xticks(range(m), [X.columns[k] for k in co], rotation=90, fontsize=_ticklabels_fs(st, m))
    show = o["rowlabels"] == "show" or (o["rowlabels"] == "auto" and n <= 60)
    ax.yaxis.tick_right()
    ax.set_yticks(range(n), [X.index[k] for k in ro] if show else [""] * n, fontsize=_ticklabels_fs(st, n))
    if rows_on and o["k"] > 1:
        for t, k in zip(ax.get_yticklabels(), ro):
            t.set_color(cmap[lab[k]])
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = corner.inset_axes([0.15, 0.12, 0.12, 0.8])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=st.font * 0.7)
    cb.ax.yaxis.set_ticks_position("left")
    cb.set_label("z-score" if centre else "Value", fontsize=st.font * 0.8)
    cb.ax.yaxis.set_label_position("left")
    st.suptitle(fig)


@plot("circular_heatmap_070", "Circular cluster heatmap", CH, number="070", example=_ex_circ,
      roles=[Role("gene", "Row labels (sectors)", "cat"), Role("conds", "Numeric columns (rings)", "num", multi=True)],
      opts=[Opt("scale", "Scale", "choice", "row z-score", choices=["row z-score", "none"]),
            Opt("cluster", "Cluster rows (tree in the centre)", "bool", True),
            Opt("method", "Linkage", "choice", "average", choices=_METHODS),
            Opt("gap", "Gap for sample names (°)", "int", 30, lo=0, hi=120)],
      desc="Wide table: one row per feature (drawn as sectors) and one numeric column per sample (drawn as rings).",
      size=(7, 6.5), axis_labels=False)
def circular_heatmap_070(fig, df, c, o, st):
    import matplotlib as mpl
    X = _matrix(df, c)
    M, centre = _scale(X.to_numpy(float), o["scale"])
    n, m = M.shape
    ro = np.arange(n)
    Z = _linkage(M, o["method"]) if o["cluster"] and n >= 2 else None
    if Z is not None:
        ic, dc, cl, ro, _, _ = _tree(Z)
    P = M[ro]
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    gap = np.radians(o["gap"])
    w = (2 * np.pi - gap) / n
    theta = lambda pos: gap / 2 + (np.asarray(pos) + 0.5) * w   # noqa: E731
    r0, rh = 1.0, 0.32
    lo, hi = _heat_norm(P, centre)
    norm = mpl.colors.Normalize(lo, hi)
    cm = mpl.colormaps[st.cmap]
    for j in range(m):
        vals = P[:, j]
        cols = [cm(norm(v)) if np.isfinite(v) else (0.9, 0.9, 0.9, 1) for v in vals]
        ax.bar(theta(np.arange(n)), rh, width=w, bottom=r0 + j * rh, color=cols, edgecolor="white", linewidth=0.4,
               align="center")
    outer = r0 + m * rh
    fs = _ticklabels_fs(st, n)
    for k in range(n):
        t = float(theta(k))
        ang = (90 - np.degrees(t)) % 360
        flip = 90 < ang < 270
        ax.text(t, outer + 0.06, X.index[ro[k]], rotation=ang + 180 if flip else ang, rotation_mode="anchor",
                ha="right" if flip else "left", va="center", fontsize=fs)
    if gap > 0:
        for j in range(m):
            ax.text(0, r0 + (j + 0.5) * rh, X.columns[j], ha="center", va="center", fontsize=st.font * 0.62)
    if Z is not None:
        hmax = dc.max() or 1
        _draw_polar_tree(ax, ic, dc, cl, theta, lambda h: (r0 - 0.04) * (1 - np.asarray(h) / hmax * 0.95), lw=1.1)
    ax.set_ylim(0, outer + 0.05)
    ax.set_axis_off()
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    cb = fig.colorbar(sm, ax=ax, shrink=0.45, aspect=18, pad=0.12)
    cb.set_label("z-score" if centre else "Value", fontsize=st.font * 0.85)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=st.font * 0.75)
    st.suptitle(fig)


def _features(df, c, label="label"):
    X = _matrix(df, c, label, "features")
    return X


@plot("dendrogram_018", "Dendrogram", CH, number="018", example=_ex_dendro,
      roles=[Role("label", "Sample labels", "cat"), Role("features", "Numeric features", "num", multi=True),
             Role("group", "Colour labels by (optional)", "cat", optional=True)],
      opts=[Opt("method", "Linkage", "choice", "ward", choices=_METHODS),
            Opt("metric", "Distance", "choice", "euclidean", choices=_METRICS),
            Opt("k", "Clusters to colour (0 = off)", "int", 3, lo=0, hi=20),
            Opt("standardise", "Standardise features", "bool", True),
            Opt("cutline", "Show the cut line", "bool", True)],
      desc="One row per sample: a label column and numeric feature columns (e.g. genes); samples are clustered.",
      size=(7.5, 5))
def dendrogram_018(fig, df, c, o, st):
    X = _features(df, c)
    M = _filled(X.to_numpy(float))
    if o["standardise"]:
        sd = M.std(axis=0, ddof=1) if len(M) > 1 else np.ones(M.shape[1])
        sd[~np.isfinite(sd) | (sd == 0)] = 1
        M = (M - M.mean(axis=0)) / sd
    Z = _linkage(M, o["method"], o["metric"])
    if Z is None:
        raise ValueError("A dendrogram needs at least 2 rows.")
    k = min(o["k"], len(M))
    ic, dc, cl, order, lab, cmap = _tree(Z, k, st.colors(max(k, 1)))
    ax = st.ax(fig.add_subplot(), "y")
    ax.spines["bottom"].set_visible(False)
    _draw_tree(ax, ic, dc, cl, "top", lw=1.3)
    n = len(order)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, dc.max() * 1.05 or 1)
    ax.set_xticks(range(n), [X.index[i] for i in order], rotation=90, fontsize=_ticklabels_fs(st, n))
    ax.tick_params(axis="x", length=0)
    grp = _row_attr(df, c, "group", X.index, "label")
    if grp is not None:
        gs = list(dict.fromkeys(grp))
        off = k if k > 1 else 0   # group colours differ from the cluster colours
        gcol = dict(zip(gs, st.colors(off + len(gs))[off:]))
        for t, i in zip(ax.get_xticklabels(), order):
            t.set_color(gcol[grp[i]])
            t.set_fontweight("bold")
        from matplotlib.patches import Patch
        st.legend(ax, [Patch(color=gcol[g]) for g in gs], gs, title=c["group"], loc="upper right")
    h = _cut_height(Z, k)
    if o["cutline"] and h is not None:
        ax.axhline(h, color="#888888", ls="--", lw=0.9)
    st.labels(ax, "", f"Height ({o['method']} linkage)")


@plot("upgma_tree_102", "UPGMA cluster tree", CH, number="102", example=_ex_dendro,
      roles=[Role("label", "Leaf labels", "cat"), Role("features", "Numeric features", "num", multi=True),
             Role("group", "Colour leaves by (optional)", "cat", optional=True)],
      opts=[Opt("layout", "Layout", "choice", "rectangular", choices=["rectangular", "circular"]),
            Opt("metric", "Distance", "choice", "euclidean", choices=_METRICS),
            Opt("standardise", "Standardise features", "bool", True),
            Opt("heights", "Write node heights", "bool", False)],
      desc="One row per sample / strain / species: a label column and numeric features; the tree is built with "
           "UPGMA (average linkage).", size=(6.5, 6))
def upgma_tree_102(fig, df, c, o, st):
    X = _features(df, c)
    M = _filled(X.to_numpy(float))
    if o["standardise"]:
        sd = M.std(axis=0, ddof=1) if len(M) > 1 else np.ones(M.shape[1])
        sd[~np.isfinite(sd) | (sd == 0)] = 1
        M = (M - M.mean(axis=0)) / sd
    Z = _linkage(M, "average", o["metric"])
    if Z is None:
        raise ValueError("A tree needs at least 2 rows.")
    ic, dc, cl, order, _, _ = _tree(Z, default="#333333")
    n = len(order)
    hmax = dc.max() or 1
    grp = _row_attr(df, c, "group", X.index, "label")
    gcol = {}
    if grp is not None:
        gs = list(dict.fromkeys(grp))
        gcol = dict(zip(gs, st.colors(len(gs))))
    leafc = [gcol[grp[i]] if grp is not None else "#333333" for i in order]
    from matplotlib.lines import Line2D
    fs = _ticklabels_fs(st, n)
    if o["layout"] == "circular":
        ax = fig.add_subplot(projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        theta = lambda p: 2 * np.pi * (np.asarray(p) + 0.5) / n   # noqa: E731
        rad = lambda h: 1 - np.asarray(h) / hmax                   # noqa: E731
        _draw_polar_tree(ax, ic, dc, cl, theta, rad, lw=1.1)
        ax.scatter(theta(np.arange(n)), np.ones(n), s=28, color=leafc, zorder=3, edgecolor="white", linewidth=0.5)
        for k, i in enumerate(order):
            t = float(theta(k))
            ang = (90 - np.degrees(t)) % 360
            flip = 90 < ang < 270
            ax.text(t, 1.06, X.index[i], rotation=ang + 180 if flip else ang, rotation_mode="anchor",
                    ha="right" if flip else "left", va="center", fontsize=fs, color=leafc[k])
        ax.set_ylim(0, 1.02)
        ax.set_axis_off()
        if gcol:
            st.legend(ax, [Line2D([], [], marker="o", ls="", color=v) for v in gcol.values()], list(gcol),
                      title=c["group"], loc="upper left", bbox_to_anchor=(1.05, 1.0))
        st.suptitle(fig)
        return
    ax = st.ax(fig.add_subplot(), "x")
    ax.spines["left"].set_visible(False)
    _draw_tree(ax, ic, dc, cl, "left", lw=1.2)
    ax.scatter(np.zeros(n), np.arange(n), s=30, color=leafc, zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_xlim(hmax * 1.04, -hmax * 0.01)
    ax.set_ylim(-0.7, n - 0.3)
    ax.yaxis.tick_right()
    ax.set_yticks(range(n), [X.index[i] for i in order], fontsize=fs)
    for t, col in zip(ax.get_yticklabels(), leafc):
        t.set_color(col)
    ax.tick_params(axis="y", length=0, pad=6)
    if o["heights"]:
        for xs, ys in zip(ic, dc):
            ax.text(ys[1], (xs[1] + xs[2]) / 2, f"{ys[1]:.2f}", ha="right", va="bottom", fontsize=st.font * 0.6,
                    color="#555555")
    if gcol:
        st.legend(ax, [Line2D([], [], marker="o", ls="", color=v) for v in gcol.values()], list(gcol),
                  title=c["group"], loc="upper left")
    st.labels(ax, f"Distance ({o['metric']}, UPGMA)", "")


def _pca(df, c, scale):
    """PCA of the rows: (row labels, feature names, fitted PCA, scores, group labels or None)."""
    from sklearn.decomposition import PCA
    F = pd.DataFrame({k: num(df, k) for k in c["features"]})
    keep = F.notna().any(axis=1).to_numpy()
    F = F[keep].dropna(axis=1, how="all")
    M = _filled(F.to_numpy(float))
    ok = M.std(axis=0) > 0
    M, feats = M[:, ok], [f for f, k in zip(F.columns, ok) if k]
    if M.shape[0] < 3 or M.shape[1] < 2:
        raise ValueError("PCA needs at least 3 rows and 2 varying numeric columns.")
    M = M - M.mean(axis=0)
    if scale:
        M = M / M.std(axis=0, ddof=1)
    p = PCA(n_components=min(M.shape[0], M.shape[1], 10)).fit(M)
    labels = as_str(df, c["label"]).to_numpy()[keep] if c.get("label") else np.array(
        [str(i + 1) for i in np.where(keep)[0]])
    grp = as_str(df, c["group"]).to_numpy()[keep] if c.get("group") else None
    return labels, feats, p, p.transform(M), grp


def _ellipse(ax, x, y, color, level=0.95):
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    k = np.sqrt(chi2.ppf(level, 2))
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for fc, ec in ((color, "none"), ("none", color)):
        ax.add_patch(Ellipse((np.mean(x), np.mean(y)), 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0]), angle=ang,
                             facecolor=fc, edgecolor=ec, alpha=0.14 if fc != "none" else 0.9, lw=1.2, zorder=1))


_PCS = ["PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3"]


@plot("pca_112", "PCA", CH, number="112", example=_ex_pca,
      roles=[Role("group", "Groups (colour)", "cat", optional=True),
             Role("features", "Numeric features", "num", multi=True),
             Role("label", "Sample labels (optional)", "cat", optional=True)],
      opts=[Opt("pcs", "Components", "choice", "PC1 vs PC2", choices=_PCS),
            Opt("scale", "Standardise features", "bool", True),
            Opt("ellipse", "95 % ellipses", "bool", True),
            Opt("labels", "Write sample labels", "bool", False),
            Opt("psize", "Point size", "float", 60, lo=5, hi=300, step=5)],
      desc="One row per sample: a group column, numeric features (e.g. genes) and optionally a sample-label column.")
def pca_112(fig, df, c, o, st):
    labels, feats, p, S, grp = _pca(df, c, o["scale"])
    a, b = [int(t[-1]) - 1 for t in o["pcs"].split(" vs ")]
    if b >= S.shape[1]:
        a, b = 0, 1
    ev = p.explained_variance_ratio_ * 100
    ax = st.ax(fig.add_subplot())
    ax.axhline(0, color="#aaaaaa", lw=0.6, zorder=0)
    ax.axvline(0, color="#aaaaaa", lw=0.6, zorder=0)
    if grp is not None:
        gs = list(dict.fromkeys(grp))
        cols = dict(zip(gs, st.colors(len(gs))))
        for g in gs:
            m = grp == g
            ax.scatter(S[m, a], S[m, b], s=o["psize"], color=cols[g], edgecolor="black", linewidth=0.5, label=g,
                       zorder=3)
            if o["ellipse"]:
                _ellipse(ax, S[m, a], S[m, b], cols[g])
        st.legend(ax, title=c["group"])
    else:
        ax.scatter(S[:, a], S[:, b], s=o["psize"], color=st.color(0), edgecolor="black", linewidth=0.5, zorder=3)
        if o["ellipse"]:
            _ellipse(ax, S[:, a], S[:, b], st.color(0))
    if o["labels"]:
        _place_labels(ax, st, S[:, a], S[:, b], list(labels))
    st.labels(ax, f"PC{a + 1} ({ev[a]:.1f} %)", f"PC{b + 1} ({ev[b]:.1f} %)")


@plot("pca_biplot_034", "PCA biplot", CH, number="034", example=_ex_pca,
      roles=[Role("group", "Groups (colour, optional)", "cat", optional=True),
             Role("features", "Numeric features", "num", multi=True),
             Role("label", "Sample labels (optional)", "cat", optional=True)],
      opts=[Opt("pcs", "Components", "choice", "PC1 vs PC2", choices=_PCS),
            Opt("scale", "Standardise features", "bool", True),
            Opt("ntop", "Loading arrows (top n features)", "int", 8, lo=1, hi=100),
            Opt("scree", "Scree and loadings panels", "bool", True)],
      desc="One row per sample: optional group column and numeric features; arrows show how features load on "
           "the components.", size=(8.5, 5.5))
def pca_biplot_034(fig, df, c, o, st):
    labels, feats, p, S, grp = _pca(df, c, o["scale"])
    a, b = [int(t[-1]) - 1 for t in o["pcs"].split(" vs ")]
    if b >= S.shape[1]:
        a, b = 0, 1
    ev = p.explained_variance_ratio_ * 100
    if o["scree"]:
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1.15])
        ax = st.ax(fig.add_subplot(gs[:, 0]))
        a1, a2 = st.ax(fig.add_subplot(gs[0, 1]), "y"), st.ax(fig.add_subplot(gs[1, 1]), "x")
    else:
        ax = st.ax(fig.add_subplot())
    ax.axhline(0, color="#aaaaaa", lw=0.6, zorder=0)
    ax.axvline(0, color="#aaaaaa", lw=0.6, zorder=0)
    if grp is not None:
        gs_ = list(dict.fromkeys(grp))
        cols = dict(zip(gs_, st.colors(len(gs_))))
        for g in gs_:
            m = grp == g
            ax.scatter(S[m, a], S[m, b], s=40, color=cols[g], edgecolor="white", linewidth=0.5, label=g, alpha=0.9)
        st.legend(ax, title=c["group"], loc="best")
    else:
        ax.scatter(S[:, a], S[:, b], s=40, color="#9e9e9e", edgecolor="white", linewidth=0.5)
    L = p.components_[[a, b]].T * np.sqrt(p.explained_variance_[[a, b]])
    mag = np.hypot(L[:, 0], L[:, 1])
    top = np.argsort(-mag)[:o["ntop"]]
    f = 0.85 * np.abs(S[:, [a, b]]).max() / (mag[top].max() or 1)
    for i in top:
        x, y = L[i] * f
        ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.2,
                                                        mutation_scale=10), zorder=4)
    lim = np.abs(np.concatenate([S[:, [a, b]].ravel(), (L[top] * f * 1.25).ravel()])).max() * 1.08
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    tips = L[top] * f
    _place_labels(ax, st, tips[:, 0], tips[:, 1], [feats[i] for i in top], color="#c0392b")
    st.labels(ax, f"PC{a + 1} ({ev[a]:.1f} %)", f"PC{b + 1} ({ev[b]:.1f} %)")
    if o["scree"]:
        k = len(ev)
        a1.bar(range(1, k + 1), ev, color=st.color(0), edgecolor="black", linewidth=0.4)
        a1.plot(range(1, k + 1), np.cumsum(ev), color="#444444", marker="o", ms=3, lw=1)
        a1.set_xticks(range(1, k + 1), [str(i) for i in range(1, k + 1)], fontsize=st.font * 0.65)
        a1.set_title("Scree (% variance)", fontsize=st.font * 0.85)
        a1.set_xlabel("Component", fontsize=st.font * 0.8)
        a1.set_ylim(0, 105)
        idx = np.argsort(-np.abs(p.components_[a]))[:min(8, len(feats))][::-1]
        a2.barh(range(len(idx)), p.components_[a][idx],
                color=[st.color(0) if v >= 0 else st.color(1) for v in p.components_[a][idx]])
        a2.set_yticks(range(len(idx)), [feats[i] for i in idx], fontsize=st.font * 0.65)
        a2.axvline(0, color="#444444", lw=0.6)
        a2.set_title(f"PC{a + 1} loadings", fontsize=st.font * 0.85)
        a2.tick_params(axis="x", labelsize=st.font * 0.65)


@plot("balloon_plot_048", "Balloon plot", CH, number="048", example=_ex_balloon,
      roles=[Role("row", "Rows (categories)", "cat"), Role("col", "Columns (categories)", "cat"),
             Role("size", "Circle size", "num"), Role("color", "Circle colour (optional)", "num", optional=True)],
      opts=[Opt("maxsize", "Largest circle (pt²)", "float", 420, lo=20, hi=3000, step=20),
            Opt("sort", "Order rows by", "choice", "as in table", choices=["as in table", "mean size", "name"]),
            Opt("values", "Write values", "bool", False)],
      desc="Long table: row category, column category, a value for the circle size and optionally another value "
           "for its colour (e.g. pathway × cell type, gene ratio, −log10 p).", size=(7.5, 5.5), axis_labels=False)
def balloon_plot_048(fig, df, c, o, st):
    d = pd.DataFrame({"r": as_str(df, c["row"]), "c": as_str(df, c["col"]), "s": num(df, c["size"])})
    d["k"] = num(df, c["color"]) if c.get("color") else d["s"]
    d = d.dropna(subset=["s"]).groupby(["r", "c"], sort=False, as_index=False).mean(numeric_only=True)
    if d.empty:
        raise ValueError("No numeric sizes.")
    rows, cols = cats(d, "r"), cats(d, "c")
    if o["sort"] == "mean size":
        rows = list(d.groupby("r")["s"].mean().sort_values(ascending=False).index)
    elif o["sort"] == "name":
        rows = sorted(rows)
    ri = {r: i for i, r in enumerate(rows)}
    ci = {k: i for i, k in enumerate(cols)}
    smax = np.nanmax(np.abs(d["s"])) or 1
    area = np.abs(d["s"]) / smax * o["maxsize"]
    ax = fig.add_subplot()
    ax.set_axisbelow(True)
    if st.grid:
        ax.grid(True, color="#e3e3e3", lw=0.7)
    sc = ax.scatter(d["c"].map(ci), d["r"].map(ri), s=area, c=d["k"], cmap=st.cmap, edgecolor="#444444",
                    linewidth=0.5, zorder=3)
    if o["values"]:
        for _, row in d.iterrows():
            ax.text(ci[row["c"]], ri[row["r"]], f"{row['s']:.2g}", ha="center", va="center",
                    fontsize=st.font * 0.55, zorder=4)
    ax.set_xticks(range(len(cols)), cols, rotation=45 if len(cols) > 4 else 0, ha="right" if len(cols) > 4 else "center",
                  fontsize=st.font * 0.85)
    ax.set_yticks(range(len(rows)), rows, fontsize=_ticklabels_fs(st, len(rows)))
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_color("#bbbbbb")
    from matplotlib.lines import Line2D
    vals = np.unique(np.round(np.quantile(np.abs(d["s"]), [0.2, 0.6, 1.0]), 3))
    hs = [Line2D([], [], marker="o", ls="", markerfacecolor="#bbbbbb", markeredgecolor="#444444",
                 markersize=np.sqrt(v / smax * o["maxsize"])) for v in vals]
    kw = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), labelspacing=1.2, borderaxespad=0) \
        if st.legend_mode == "best" else {}
    st.legend(ax, hs, [f"{v:.3g}" for v in vals], title=c["size"], **kw)
    cb = fig.colorbar(sc, ax=ax, shrink=0.45, aspect=14, anchor=(0.0, 0.0), panchor=(1.0, 0.0))
    cb.set_label(c["color"] or c["size"], fontsize=st.font * 0.85)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=st.font * 0.75)
    st.labels(ax, "", "")


# ═════════════════════════════════════════════════════════════════════════════
#  clinical & statistics
# ═════════════════════════════════════════════════════════════════════════════

_EVENT_WORDS = {"1", "dead", "death", "died", "deceased", "event", "yes", "true", "relapse", "progression", "y"}


def _events(df, col):
    e = num(df, col)
    if e.notna().sum() >= max(1, 0.5 * df[col].notna().sum()):
        return (e > 0).astype(float).where(e.notna())
    s = df[col].astype(str).str.strip().str.lower()
    return s.isin(_EVENT_WORDS).astype(float).where(df[col].notna())


def _km(t, e):
    """Kaplan–Meier: event times, survival, Greenwood SE (at event times)."""
    times = np.unique(t[e == 1])
    S, var, s, acc = [], [], 1.0, 0.0
    for u in times:
        n = (t >= u).sum()
        d = ((t == u) & (e == 1)).sum()
        s *= 1 - d / n
        acc += d / (n * (n - d)) if n > d else 0.0
        S.append(s)
        var.append(s * s * acc)
    return times, np.array(S), np.sqrt(np.array(var))


def _surv_at(times, S, x):
    i = np.searchsorted(times, x, side="right") - 1
    return np.where(i >= 0, S[np.clip(i, 0, None)] if len(S) else 1.0, 1.0)


def _logrank(t, e, g):
    from scipy import stats
    groups = list(dict.fromkeys(g))
    k = len(groups)
    if k < 2:
        return np.nan
    O, E = np.zeros(k), np.zeros(k)
    V = np.zeros((k, k))
    for u in np.unique(t[e == 1]):
        at = t >= u
        n = at.sum()
        d = ((t == u) & (e == 1)).sum()
        ng = np.array([(at & (g == gg)).sum() for gg in groups], float)
        dg = np.array([((t == u) & (e == 1) & (g == gg)).sum() for gg in groups], float)
        O += dg
        E += d * ng / n
        if n > 1:
            V += d * (n - d) / (n - 1) * (np.diag(ng / n) - np.outer(ng, ng) / n ** 2)
    x = (O - E)[:-1]
    try:
        chi = float(x @ np.linalg.pinv(V[:-1, :-1]) @ x)
    except Exception:
        return np.nan
    return float(stats.chi2.sf(chi, k - 1))


def _ex_km():
    r = rng(51)
    rows = []
    for g, scale, n in [("Low expression", 48, 60), ("High expression", 22, 60)]:
        t = r.exponential(scale, n)
        cens = r.uniform(10, 72, n)
        for a, b in zip(t, cens):
            rows.append((round(float(min(a, b)), 1), int(a <= b), g))
    return pd.DataFrame(rows, columns=["Time (months)", "Status (1 = death)", "Risk group"])


def _ex_roc():
    r = rng(52)
    n1, n0 = 60, 70
    y = ["Tumor"] * n1 + ["Normal"] * n0
    a = np.concatenate([r.normal(6, 2, n1), r.normal(3.5, 1.8, n0)])
    b = np.concatenate([r.normal(40, 15, n1), r.normal(30, 14, n0)])
    c = 0.6 * (a - 4.5) / 2 + 0.4 * (b - 35) / 15 + r.normal(0, 0.4, n1 + n0)
    return pd.DataFrame({"Diagnosis": y, "CEA (ng/mL)": np.round(np.clip(a, 0.1, None), 2),
                         "CA19-9 (U/mL)": np.round(np.clip(b, 1, None), 1), "Panel score": np.round(c, 3)})


def _ex_forest():
    rows = [("Age ≥ 65 y", 1.42, 1.08, 1.87, 0.012, 412), ("Male sex", 1.10, 0.86, 1.41, 0.45, 530),
            ("Smoker", 1.65, 1.21, 2.25, 0.0016, 298), ("BMI ≥ 30", 1.18, 0.88, 1.58, 0.27, 241),
            ("Diabetes", 1.37, 0.98, 1.92, 0.065, 187), ("Stage III–IV", 2.31, 1.72, 3.10, 0.00001, 344),
            ("High TMB", 0.62, 0.45, 0.86, 0.004, 205), ("PD-L1 positive", 0.74, 0.55, 0.99, 0.043, 260),
            ("Prior therapy", 1.05, 0.79, 1.40, 0.73, 310)]
    return pd.DataFrame(rows, columns=["Variable", "HR", "Lower 95% CI", "Upper 95% CI", "P value", "N"])


def _ex_species():
    r = rng(53)
    rows = []
    nsp = 30
    for site, rich, lam in [("Forest", 26, 3.0), ("Grassland", 18, 4.0), ("Wetland", 12, 6.0)]:
        pool = r.choice(nsp, rich, replace=False)
        w = r.dirichlet(np.full(rich, 0.6))
        for k in range(10):
            counts = np.zeros(nsp, int)
            counts[pool] = r.poisson(w * lam * rich)
            rows.append([site, f"{site[:2].upper()}{k + 1:02d}"] + list(counts))
    return pd.DataFrame(rows, columns=["Site", "Sample"] + [f"Sp{j + 1:02d}" for j in range(nsp)])


@plot("km_survival_040", "Kaplan-Meier survival curve", CC, number="040", example=_ex_km,
      roles=[Role("time", "Time", "num"), Role("event", "Event (1 = event, 0 = censored)", "num"),
             Role("group", "Groups (optional)", "cat", optional=True)],
      opts=[Opt("risk", "Number-at-risk table", "bool", True), Opt("ci", "95 % confidence bands", "bool", True),
            Opt("censor", "Censor ticks", "bool", True), Opt("median", "Median survival lines", "bool", False),
            Opt("pct", "Survival in %", "bool", False)],
      desc="One row per patient: follow-up time, event status (1 / 0, or dead / alive) and optionally a group.",
      size=(7, 5.8))
def km_survival_040(fig, df, c, o, st):
    d = pd.DataFrame({"t": num(df, c["time"]), "e": _events(df, c["event"])})
    d["g"] = as_str(df, c["group"]) if c.get("group") else "All"
    d = d.dropna()
    d = d[d["t"] >= 0]
    if d.empty:
        raise ValueError("No rows with a time and an event status.")
    groups = cats(d, "g")
    cols = st.colors(len(groups))
    k = 100.0 if o["pct"] else 1.0
    if o["risk"]:
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 0.5 + 0.28 * len(groups)])
        ax = st.ax(fig.add_subplot(gs[0]))
        axr = fig.add_subplot(gs[1], sharex=ax)
    else:
        ax = st.ax(fig.add_subplot())
    tmax = d["t"].max()
    for i, g in enumerate(groups):
        sub = d[d["g"] == g]
        t, e = sub["t"].to_numpy(), sub["e"].to_numpy()
        times, S, se = _km(t, e)
        xs = np.concatenate([[0], times, [t.max()]])
        ys = np.concatenate([[1], S, [S[-1] if len(S) else 1]])
        ax.step(xs, ys * k, where="post", color=cols[i], lw=1.8, label=f"{g} (n = {len(sub)})")
        if o["ci"] and len(S):
            lo = np.clip(S - 1.96 * se, 0, 1)
            hi = np.clip(S + 1.96 * se, 0, 1)
            ax.fill_between(xs, np.concatenate([[1], lo, [lo[-1]]]) * k, np.concatenate([[1], hi, [hi[-1]]]) * k,
                            step="post", color=cols[i], alpha=0.13, linewidth=0)
        if o["censor"]:
            tc = t[e == 0]
            ax.plot(tc, _surv_at(times, S, tc) * k, ls="", marker="|", ms=8, mew=1.2, color=cols[i])
        if o["median"] and len(S) and S.min() <= 0.5:
            m = times[np.argmax(S <= 0.5)]
            ax.plot([0, m, m], [0.5 * k, 0.5 * k, 0], color=cols[i], ls=":", lw=1)
    p = _logrank(d["t"].to_numpy(), d["e"].to_numpy(), d["g"].to_numpy())
    if np.isfinite(p):
        ax.text(0.03, 0.05, f"Log-rank {_fmt_p(p)}", transform=ax.transAxes, fontsize=st.font * 0.9)
    ax.set_xlim(-0.03 * tmax, tmax * 1.03)
    ax.set_ylim(-0.02 * k, 1.04 * k)
    st.legend(ax, loc="upper right")
    st.labels(ax, "" if o["risk"] else c["time"], "Survival (%)" if o["pct"] else "Survival probability")
    if o["risk"]:
        ticks = [x for x in ax.get_xticks() if 0 <= x <= tmax * 1.03]
        for i, g in enumerate(groups):
            t = d.loc[d["g"] == g, "t"].to_numpy()
            for x in ticks:
                axr.text(x, len(groups) - 1 - i, str(int((t >= x).sum())), ha="center", va="center",
                         fontsize=st.font * 0.8, color=cols[i])
        axr.set_ylim(-0.6, len(groups) - 0.4)
        axr.set_yticks(range(len(groups)), groups[::-1], fontsize=st.font * 0.8)
        for t_, col in zip(axr.get_yticklabels(), cols[::-1]):
            t_.set_color(col)
        for s in ("top", "right", "left"):
            axr.spines[s].set_visible(False)
        axr.spines["bottom"].set_color("#555555")
        axr.tick_params(axis="y", length=0, pad=14)
        axr.tick_params(axis="x", labelsize=st.font * 0.85)
        axr.set_title("Number at risk", fontsize=st.font * 0.85, loc="left")
        axr.set_xlabel(st.opts.get("xlabel") or c["time"], fontsize=st.font)
        ax.tick_params(labelbottom=False)


@plot("roc_curve_106", "ROC curve", CC, number="106", example=_ex_roc,
      roles=[Role("truth", "True class (two levels)", "any"), Role("scores", "Scores / markers", "num", multi=True)],
      opts=[Opt("positive", "Positive class (blank = auto)", "str", ""),
            Opt("flip", "Flip markers with AUC < 0.5", "bool", True),
            Opt("ci", "95 % CI of AUC (bootstrap)", "bool", False),
            Opt("youden", "Mark best cut-off (Youden)", "bool", True)],
      desc="One row per subject: the true class (e.g. Tumor / Normal or 1 / 0) and one or more numeric scores.",
      size=(5.8, 5.4))
def roc_curve_106(fig, df, c, o, st):
    from sklearn.metrics import auc, roc_curve
    truth = as_str(df, c["truth"])
    tv = num(df, c["truth"])
    levels = [v for v in cats(df.assign(_t=truth), "_t") if v != "(blank)"]
    if len(levels) < 2:
        raise ValueError("The true-class column needs two levels.")
    pos = o["positive"].strip()
    if pos not in levels:
        if tv.notna().all():
            pos = truth[tv.idxmax()]
        else:
            low = [v.lower() for v in levels]
            pref = [lv for lv, lw in zip(levels, low) if lw in
                    ("1", "yes", "true", "case", "tumor", "tumour", "disease", "positive", "pos", "cancer", "dead")]
            pos = pref[0] if pref else sorted(levels)[-1]
    y = (truth == pos).astype(int).where(df[c["truth"]].notna())
    ax = st.ax(fig.add_subplot())
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1)
    cols = st.colors(len(c["scores"]))
    r = rng(0)
    for i, sc in enumerate(c["scores"]):
        s = num(df, sc)
        m = s.notna() & y.notna()
        yy, ss = y[m].to_numpy(int), s[m].to_numpy(float)
        if len(np.unique(yy)) < 2:
            continue
        fpr, tpr, thr = roc_curve(yy, ss)
        A = auc(fpr, tpr)
        flipped = False
        if o["flip"] and A < 0.5:
            ss, flipped = -ss, True
            fpr, tpr, thr = roc_curve(yy, ss)
            A = auc(fpr, tpr)
        lab = f"{sc}{' (−)' if flipped else ''}: AUC = {A:.3f}"
        if o["ci"]:
            boots = []
            for _ in range(300):
                k = r.integers(0, len(yy), len(yy))
                if len(np.unique(yy[k])) == 2:
                    f2, t2, _ = roc_curve(yy[k], ss[k])
                    boots.append(auc(f2, t2))
            if boots:
                lo, hi = np.percentile(boots, [2.5, 97.5])
                lab += f" ({lo:.2f}–{hi:.2f})"
        ax.plot(fpr, tpr, color=cols[i], lw=2, label=lab)
        if o["youden"]:
            j = np.argmax(tpr - fpr)
            ax.scatter([fpr[j]], [tpr[j]], s=40, color=cols[i], edgecolor="black", linewidth=0.6, zorder=4)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    st.legend(ax, loc="lower right", title=f"Positive: {pos}")
    st.labels(ax, "1 − Specificity (false-positive rate)", "Sensitivity (true-positive rate)")


@plot("forest_plot_099", "Forest plot", CC, number="099", example=_ex_forest,
      roles=[Role("label", "Row labels", "cat"), Role("est", "Estimate (HR / OR / effect)", "num"),
             Role("lo", "Lower CI", "num"), Role("hi", "Upper CI", "num"),
             Role("p", "p value (optional)", "num", optional=True),
             Role("weight", "Weight / N (optional, box size)", "num", optional=True)],
      opts=[Opt("log", "Log scale (ratios)", "bool", True),
            Opt("ref", "Reference line at", "float", 1.0, lo=-1e9, hi=1e9, step=0.5),
            Opt("text", "Estimate (CI) column", "bool", True),
            Opt("sig", "Colour rows whose CI excludes the reference", "bool", True),
            Opt("measure", "Name of the estimate", "str", "HR")],
      desc="One row per variable / study: label, estimate, lower and upper confidence limits, and optionally a "
           "p value and a weight (e.g. N) for the box size.", size=(8.5, 5))
def forest_plot_099(fig, df, c, o, st):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    d = pd.DataFrame({"lab": as_str(df, c["label"]), "e": num(df, c["est"]), "lo": num(df, c["lo"]),
                      "hi": num(df, c["hi"])})
    d["p"] = num(df, c["p"]) if c.get("p") else np.nan
    d["w"] = num(df, c["weight"]) if c.get("weight") else np.nan
    d = d.dropna(subset=["e"])
    if d.empty:
        raise ValueError("No numeric estimates.")
    log = o["log"] and (d[["e", "lo", "hi"]].min().min() > 0)
    ref = o["ref"] if (o["ref"] > 0 or not log) else 1.0
    n = len(d)
    y = np.arange(n)[::-1].astype(float)
    ncol = 1 + (1 if o["text"] else 0) + (1 if c.get("p") else 0)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 2.6, 0.75 * ncol])
    axl = fig.add_subplot(gs[0])
    ax = st.ax(fig.add_subplot(gs[1], sharey=axl), "x")
    axt = fig.add_subplot(gs[2], sharey=axl)
    for a in (axl, axt):
        a.set_axis_off()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylim(-0.8, n + 0.2)
    w = d["w"].to_numpy(float)
    if np.isfinite(w).any():
        wm = np.nanmax(w) or 1
        ms = 5 + 9 * np.sqrt(np.clip(np.nan_to_num(w, nan=np.nanmin(w)), 0, None) / wm)
    else:
        ms = np.full(n, 8.0)
    for k, (_, r) in enumerate(d.iterrows()):
        sig = np.isfinite(r["lo"]) and np.isfinite(r["hi"]) and (r["lo"] > ref or r["hi"] < ref)
        col = (st.color(0) if r["e"] > ref else st.color(1)) if (o["sig"] and sig) else "#555555"
        if np.isfinite(r["lo"]) and np.isfinite(r["hi"]):
            ax.plot([r["lo"], r["hi"]], [y[k], y[k]], color=col, lw=1.4, solid_capstyle="butt")
        ax.plot([r["e"]], [y[k]], marker="s", ms=ms[k], color=col, markeredgecolor="black", markeredgewidth=0.4)
    ax.axvline(ref, color="#333333", lw=0.9, ls="--")
    if log:
        ax.set_xscale("log")
        lo, hi = np.nanmin(d[["lo", "e"]].to_numpy()), np.nanmax(d[["hi", "e"]].to_numpy())
        lo, hi = min(lo, ref) / 1.15, max(hi, ref) * 1.15
        ax.set_xlim(lo, hi)
        nice = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        t = nice[(nice >= lo) & (nice <= hi)]
        if len(t) < 4:
            fine = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10])
            t = np.union1d(t, fine[(fine >= lo) & (fine <= hi)])
        ax.xaxis.set_major_locator(FixedLocator(t))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    fs = st.font * 0.85
    head = n - 0.1
    axl.text(0, head, c["label"], fontweight="bold", fontsize=fs, va="bottom", transform=axl.get_yaxis_transform())
    for k, lab in enumerate(d["lab"]):
        axl.text(0, y[k], lab, fontsize=fs, va="center", transform=axl.get_yaxis_transform())
    x = 0.0
    m = o["measure"] or c["est"]
    if o["text"]:
        axt.text(x, head, f"{m} (95% CI)", fontweight="bold", fontsize=fs, va="bottom",
                 transform=axt.get_yaxis_transform())
        for k, (_, r) in enumerate(d.iterrows()):
            axt.text(x, y[k], f"{r['e']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})", fontsize=fs, va="center",
                     transform=axt.get_yaxis_transform())
        x += 0.66 if c.get("p") else 0
    if c.get("p"):
        axt.text(x, head, "p", fontweight="bold", fontsize=fs, va="bottom", transform=axt.get_yaxis_transform())
        for k, pv in enumerate(d["p"]):
            if np.isfinite(pv):
                axt.text(x, y[k], "<0.001" if pv < 0.001 else f"{pv:.3f}", fontsize=fs, va="center",
                         transform=axt.get_yaxis_transform(), fontweight="bold" if pv < 0.05 else "normal")
    for a in (axl, ax, axt):
        a.axhline(n - 0.45, color="#888888", lw=0.6)
    st.labels(ax, f"{m} (95% CI)", "")


@plot("species_accumulation_127", "Species accumulation curves", CC, number="127", example=_ex_species,
      roles=[Role("group", "Groups / sites (optional)", "cat", optional=True),
             Role("species", "Species abundance columns", "num", multi=True)],
      opts=[Opt("method", "Method", "choice", "random (samples)", choices=["random (samples)", "rarefaction (individuals)"]),
            Opt("perm", "Permutations", "int", 100, lo=10, hi=2000),
            Opt("band", "± SD band", "bool", True)],
      desc="Samples × species table: one row per sample, one numeric column per species (counts or presence), "
           "optionally a group / site column.")
def species_accumulation_127(fig, df, c, o, st):
    from scipy.special import gammaln
    A = pd.DataFrame({k: num(df, k) for k in c["species"]}).fillna(0).clip(lower=0)
    g = as_str(df, c["group"]) if c.get("group") else pd.Series("All samples", index=df.index)
    groups = list(dict.fromkeys(g))
    cols = st.colors(len(groups))
    ax = st.ax(fig.add_subplot())
    r = rng(0)
    rare = o["method"].startswith("rarefaction")
    for i, gg in enumerate(groups):
        P = A[g == gg].to_numpy(float)
        if len(P) == 0:
            continue
        if rare:
            Ni = np.round(P.sum(axis=0))
            Ni = Ni[Ni > 0]
            N = int(Ni.sum())
            if N == 0:
                continue
            ns = np.unique(np.linspace(1, N, min(N, 80)).astype(int))
            ES = []
            for nn in ns:
                with np.errstate(all="ignore"):
                    lp = gammaln(N - Ni + 1) - gammaln(N - Ni - nn + 1) - gammaln(N + 1) + gammaln(N - nn + 1)
                    q = np.where(N - Ni >= nn, np.exp(lp), 0.0)
                ES.append(float((1 - q).sum()))
            ax.plot(ns, ES, color=cols[i], lw=2, label=f"{gg} (S = {len(Ni)})")
            ax.plot(ns[-1], ES[-1], "o", color=cols[i], ms=5)
            continue
        B = P > 0
        curves = np.empty((o["perm"], len(B)))
        for k in range(o["perm"]):
            curves[k] = np.maximum.accumulate(B[r.permutation(len(B))], axis=0).sum(axis=1)
        m, sd = curves.mean(axis=0), curves.std(axis=0)
        x = np.arange(1, len(B) + 1)
        ax.plot(x, m, color=cols[i], lw=2, marker="o", ms=3.5, label=f"{gg} (S = {int(B.any(axis=0).sum())})")
        if o["band"]:
            ax.fill_between(x, m - sd, m + sd, color=cols[i], alpha=0.18, linewidth=0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    st.legend(ax, loc="lower right", title=c.get("group") or None)
    st.labels(ax, "Number of individuals" if rare else "Number of samples", "Species richness")
