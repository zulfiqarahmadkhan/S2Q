"""Transcriptome & enrichment, sets & flows (Venn, UpSet, Sankey, chord) and genome & sequence plots."""
from __future__ import annotations

import textwrap
from collections import Counter

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Patch, PathPatch,
                                Rectangle, Wedge)
from matplotlib.path import Path

from s2q.core.gallery.base import Opt, Role, as_str, cats, num, plot, rng

T = "Transcriptome & enrichment"
F = "Sets & flows (Venn, Sankey, chord)"
G = "Genome & sequence"


# ═══════════════════════════════════════════════════════════════════════════════
# shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cmap(st, reverse=False):
    try:
        cm = mpl.colormaps[st.cmap]
    except KeyError:
        cm = mpl.colormaps["RdBu_r"]
    return cm.reversed() if reverse else cm


def _plog_norm(p):
    """Colour scale for p-values: logarithmic when they span more than a decade."""
    p = np.asarray(p, float)
    p = p[np.isfinite(p) & (p > 0)]
    if not len(p):
        return Normalize(0, 1)
    lo, hi = p.min(), p.max()
    if hi <= lo:
        return LogNorm(lo / 2, hi * 2)
    return LogNorm(lo, hi) if hi / lo > 10 else Normalize(lo, hi)


def _wrap(s, w=40):
    s = str(s)
    return "\n".join(textwrap.wrap(s, w)) if len(s) > w else s


def _cbar(fig, st, cmap, norm, label, ax=None, cax=None, **kw):
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    if cax is not None:
        cb = fig.colorbar(sm, cax=cax, **kw)
    else:
        kw.setdefault("shrink", 0.6)
        kw.setdefault("aspect", 18)
        kw.setdefault("pad", 0.02)
        cb = fig.colorbar(sm, ax=ax, **kw)
    cb.set_label(label, fontsize=st.font * 0.85)
    cb.ax.tick_params(labelsize=st.font * 0.75)
    cb.outline.set_linewidth(0.5)
    return cb


def _side(fig, ncols_main=1, ratios=None):
    """Main axes on the left and a narrow right column split into colour-bar (top) / legend (bottom) slots."""
    ratios = ratios or [24] * ncols_main
    gs = fig.add_gridspec(2, ncols_main + 1, width_ratios=list(ratios) + [1], height_ratios=[1, 1.3])
    lax = fig.add_subplot(gs[1, -1])
    lax.set_axis_off()
    return gs, fig.add_subplot(gs[0, -1]), lax


def _sizes(v, smin=25.0, smax=320.0, lo=None, hi=None):
    v = np.asarray(v, float)
    lo = np.nanmin(v) if lo is None else lo
    hi = np.nanmax(v) if hi is None else hi
    if not np.isfinite(lo) or hi <= lo:
        return np.full(len(v), (smin + smax) / 2)
    return smin + (v - lo) / (hi - lo) * (smax - smin)


def _size_handles(v, smin=25.0, smax=320.0, n=4, integer=True, fmt="{:g}", color="#8c8c8c"):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return []
    lo, hi = v.min(), v.max()
    ticks = np.linspace(lo, hi, n)
    if integer:
        ticks = np.unique(np.round(ticks).astype(int))
    else:
        ticks = np.unique(np.round(ticks, 3))
    s = _sizes(ticks, smin, smax, lo, hi)
    return [Line2D([], [], ls="", marker="o", ms=np.sqrt(max(a, 1)), mfc=color, mec="none", label=fmt.format(t))
            for t, a in zip(ticks, s)]


def _legend_sections(st, ax, sections, **kw):
    """One legend made of titled sections [(title, handles), …] (bold section titles)."""
    hs, ls, heads = [], [], []
    secs = [(t, h) for t, h in sections if h]
    for t, h in secs:
        if t:
            heads.append(len(hs))
            hs.append(Line2D([], [], ls="", marker=""))
            ls.append(t)
        hs += h
        ls += [x.get_label() for x in h]
    leg = st.legend(ax, hs, ls, **kw)
    if leg is not None:
        for i in heads:
            leg.get_texts()[i].set_fontweight("bold")
    return leg


def _ratio(df, col):
    """Gene ratio column: numbers or 'k/N' text."""
    v = num(df, col)
    if v.notna().mean() >= 0.8:
        return v
    s = df[col].astype(str).str.split("/", n=1, expand=True)
    if s.shape[1] == 2:
        a, b = pd.to_numeric(s[0], errors="coerce"), pd.to_numeric(s[1], errors="coerce")
        return (a / b).where(b > 0)
    return v


def _repel(ax, xs, ys, texts, st, fs=None, color="#222222"):
    """Label points with non-overlapping text (simple box repulsion) and thin leader lines."""
    texts = [str(t) for t in texts]
    n = len(texts)
    if not n:
        return
    fig = ax.figure
    fs = fs or st.font * 0.72
    W, H = fig.get_size_inches() * 72.0
    pos = ax.get_position()
    aw, ah = W * pos.width, H * pos.height
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    px = (xs - x0) / (x1 - x0) * aw
    py = (ys - y0) / (y1 - y0) * ah
    w = np.array([len(t) * fs * 0.62 + 4 for t in texts])
    h = fs * 1.35
    lx = px + np.where(px < aw / 2, -1, 1) * (w / 2 + 5)
    ly = py + h * 0.8
    for _ in range(250):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx, dy = lx[j] - lx[i], ly[j] - ly[i]
                ox, oy = (w[i] + w[j]) / 2 - abs(dx), h - abs(dy)
                if ox > 0 and oy > 0:
                    s = 1.0 if dy >= 0 else -1.0
                    ly[j] += s * (oy / 2 + 0.5)
                    ly[i] -= s * (oy / 2 + 0.5)
                    moved = True
            for k in range(n):
                dx, dy = px[k] - lx[i], py[k] - ly[i]
                if abs(dx) < w[i] / 2 + 2 and abs(dy) < h / 2 + 2:
                    ly[i] += (1.0 if dy <= 0 else -1.0) * (h / 2 + 2.5 - abs(dy))
                    moved = True
        lx = np.clip(lx, w / 2, aw - w / 2)
        ly = np.clip(ly, h / 2, ah - h / 2)
        if not moved:
            break
    tx = x0 + lx / aw * (x1 - x0)
    ty = y0 + ly / ah * (y1 - y0)
    for i in range(n):
        ax.annotate(texts[i], xy=(xs[i], ys[i]), xytext=(tx[i], ty[i]), textcoords="data", fontsize=fs,
                    ha="center", va="center", color=color, zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6, shrinkA=1, shrinkB=2))


def _leg_out(st, where="right"):
    """kwargs putting an axes legend outside the axes (only for the 'best' legend mode)."""
    if st.legend_mode != "best":
        return {}
    if where == "below":
        return dict(outside="below")          # under the figure; the layout makes room for it
    return dict(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)


def _ribbon(x0, y0a, y0b, x1, y1a, y1b):
    """Sankey ribbon (two cubic Béziers) from the span [y0a, y0b] at x0 to [y1a, y1b] at x1."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4,
             Path.CURVE4, Path.CLOSEPOLY]
    return Path(verts, codes)


def _arcpts(t1, t2, r, n=24):
    t = np.linspace(t1, t2, n)
    return np.c_[r * np.cos(t), r * np.sin(t)]


def _chord_path(a1, a2, b1, b2, r):
    """Chord ribbon between the arcs [a1, a2] and [b1, b2] (radians) of a circle of radius r."""
    A, B = _arcpts(a1, a2, r), _arcpts(b1, b2, r)
    verts = list(A) + [(0, 0), tuple(B[0])] + list(B[1:]) + [(0, 0), tuple(A[0]), tuple(A[0])]
    codes = ([Path.MOVETO] + [Path.LINETO] * (len(A) - 1) + [Path.CURVE3, Path.CURVE3] +
             [Path.LINETO] * (len(B) - 1) + [Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY])
    return Path(verts, codes)


def _radial_text(ax, ang, r, s, fs, color="#222222", **kw):
    """Text pointing away from the centre at angle *ang* (radians), readable on both sides."""
    deg = np.degrees(ang) % 360
    right = deg <= 90 or deg >= 270
    ax.text(r * np.cos(ang), r * np.sin(ang), s, rotation=deg if right else deg - 180, rotation_mode="anchor",
            ha="left" if right else "right", va="center", fontsize=fs, color=color, **kw)


def _thin(centers, labels, total_pts, span, fs):
    """Drop tick labels that would collide (keeps the first, then every label with enough room)."""
    keep, last = [], -1e9
    for cx, lab in zip(centers, labels):
        p = cx / span * total_pts
        need = (len(str(lab)) * fs * 0.62 + 4)
        if p - last >= need:
            keep.append(True)
            last = p
        else:
            keep.append(False)
    return keep


# ── genome helpers ───────────────────────────────────────────────────────────

_HG38 = {  # chromosome: (length Mb, centromere Mb)
    "1": (248.96, 123.4), "2": (242.19, 93.9), "3": (198.30, 90.9), "4": (190.21, 50.0), "5": (181.54, 48.8),
    "6": (170.81, 59.8), "7": (159.35, 60.1), "8": (145.14, 45.2), "9": (138.39, 43.0), "10": (133.80, 39.8),
    "11": (135.09, 53.4), "12": (133.28, 35.5), "13": (114.36, 17.7), "14": (107.04, 17.2),
    "15": (101.99, 19.0), "16": (90.34, 36.8), "17": (83.26, 25.1), "18": (80.37, 18.5), "19": (58.62, 26.2),
    "20": (64.44, 28.1), "21": (46.71, 12.0), "22": (50.82, 15.0), "X": (156.04, 60.6), "Y": (57.23, 10.4)}
_EX_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

_GENOME_OPT = Opt("genome", "Chromosome lengths", "choice", "auto",
                  choices=["auto", "human hg38", "from data (max position)"])


def _nc(name):
    s = str(name).strip()
    s = s[3:] if s.lower().startswith("chr") else s
    return s.upper() if s.lower() in ("x", "y", "m", "mt") else s


def _short(name):
    return _nc(name)


def _chrom_sort(names):
    def key(n):
        s = _nc(n)
        if s.isdigit():
            return (0, int(s), "")
        return (1, {"X": 0, "Y": 1, "M": 2, "MT": 2}.get(s, 9), s)
    return sorted(dict.fromkeys(str(n) for n in names), key=key)


def _genome(chrom, pos, o, end=None):
    """Ordered chromosome names, lengths (bp) and centromeres (bp or None) for a chromosome/position table."""
    chrom = chrom.astype(str)
    names = _chrom_sort(chrom.unique())
    top = pos if end is None else np.fmax(pos, end)
    mx = top.groupby(chrom).max()
    mode = o.get("genome", "auto")
    known = all(_nc(n) in _HG38 for n in names)
    fits = known and all(mx.get(n, 0) <= _HG38[_nc(n)][0] * 1.0e6 * 1.01 for n in names)
    use = known and (mode == "human hg38" or (mode == "auto" and fits))
    if use:
        L = {n: _HG38[_nc(n)][0] * 1e6 for n in names}
        cen = {n: _HG38[_nc(n)][1] * 1e6 for n in names}
    else:
        L = {n: float(mx.get(n, 1.0)) * 1.02 if np.isfinite(mx.get(n, np.nan)) else 1.0 for n in names}
        cen = None
    return names, L, cen


def _ex_peaks(seed, n, chroms=None, value=False):
    """Random positions spread over the human genome proportionally to chromosome length."""
    r = rng(seed)
    chroms = chroms or _EX_CHROMS
    lens = np.array([_HG38[_nc(c)][0] * 1e6 for c in chroms])
    ch = r.choice(len(chroms), n, p=lens / lens.sum())
    pos = (r.uniform(0.01, 0.99, n) * lens[ch]).astype(int)
    order = np.lexsort((pos, ch))
    return pd.DataFrame({"Chrom": np.array(chroms)[ch][order], "Position": pos[order]})


# ═══════════════════════════════════════════════════════════════════════════════
# Transcriptome & enrichment
# ═══════════════════════════════════════════════════════════════════════════════

_SYMS = ("TP53 MYC EGFR KRAS CDKN1A MDM2 VEGFA IL6 TNF CXCL8 CCL2 FOS JUN EGR1 ATF3 HMOX1 SOX2 SERPINE1 MMP9 "
         "COL1A1 FN1 SPP1 IL1B PTGS2 ICAM1 CXCL10 ISG15 IFIT1 MX1 STAT1 IRF7 SOCS3 KLF4 NR4A1 DUSP1 GADD45B "
         "CYR61 CTGF THBS1 HSPA1A DDIT3 TXNIP PDK4 ANGPTL4 LCN2 S100A8 S100A9 SAA1 CRP APOE").split()


def _ex_deg():
    from scipy import stats
    r = rng(86)
    n = 2500
    fc = r.normal(0, 0.9, n)
    k = 160
    fc[:k] = r.choice([-1, 1], k, p=[0.4, 0.6]) * r.uniform(1.2, 5.0, k)
    z = np.abs(fc) * r.uniform(0.8, 3.2, n) + np.abs(r.normal(0, 0.7, n))
    p = np.clip(2 * stats.norm.sf(z), 1e-300, 1)
    names = np.array([f"GENE{i:04d}" for i in range(n)], dtype=object)
    names[np.argsort(p)[:len(_SYMS)]] = _SYMS
    return pd.DataFrame({"Gene": names, "log2FoldChange": np.round(fc, 3), "pvalue": p})


_DEG_ROLES = [Role("gene", "Gene names (labels)", "cat", optional=True), Role("fc", "log2 fold change", "num"),
              Role("p", "p-value (or adjusted p)", "num")]


def _deg_frame(df, c, o):
    g = as_str(df, c["gene"]) if c.get("gene") else pd.Series([f"#{i}" for i in range(len(df))], index=df.index)
    d = pd.DataFrame({"g": g, "fc": num(df, c["fc"]), "p": num(df, c["p"])}).dropna(subset=["fc", "p"])
    if o.get("plog") == "−log10 p":
        d["y"] = d["p"]
        d["p"] = 10 ** (-d["y"])
    else:
        d = d[d["p"] > 0] if (d["p"] > 0).any() else d
        pmin = d.loc[d["p"] > 0, "p"].min() if (d["p"] > 0).any() else 1e-300
        d["y"] = -np.log10(d["p"].clip(lower=pmin * 0.1))
    return d


@plot("volcano_086", "Volcano plot (3 colours)", T, number="086", example=_ex_deg, roles=_DEG_ROLES,
      opts=[Opt("fc_thr", "|log2FC| threshold", "float", 1.0, lo=0, hi=20, step=0.1),
            Opt("p_thr", "p-value threshold", "float", 0.05, lo=1e-300, hi=1, step=0.01),
            Opt("plog", "p column is", "choice", "p-value", choices=["p-value", "−log10 p"]),
            Opt("top", "Label top N genes", "int", 10, lo=0, hi=100),
            Opt("highlight", "Always label (comma separated)", "str", ""),
            Opt("up", "Up colour", "color", "#d6404e"), Opt("down", "Down colour", "color", "#3a78c2"),
            Opt("psize", "Point size", "float", 12, lo=1, hi=200)],
      desc="One row per gene: gene name, log2 fold change and p-value (e.g. a DESeq2 / limma result table).",
      size=(7, 6))
def volcano_086(fig, df, c, o, st):
    d = _deg_frame(df, c, o)
    thr, pthr = float(o["fc_thr"]), float(o["p_thr"])
    sig = d["p"] < pthr
    up, dn = sig & (d["fc"] >= thr), sig & (d["fc"] <= -thr)
    ns = ~(up | dn)
    ax = st.ax(fig.add_subplot())
    s = float(o["psize"])
    ax.scatter(d.loc[ns, "fc"], d.loc[ns, "y"], s=s * 0.8, c="#bdbdbd", lw=0, alpha=0.7, rasterized=True,
               label=f"NS ({int(ns.sum())})")
    ax.scatter(d.loc[dn, "fc"], d.loc[dn, "y"], s=s, c=o["down"], lw=0, alpha=0.85, rasterized=True,
               label=f"Down ({int(dn.sum())})")
    ax.scatter(d.loc[up, "fc"], d.loc[up, "y"], s=s, c=o["up"], lw=0, alpha=0.85, rasterized=True,
               label=f"Up ({int(up.sum())})")
    m = max(float(d["fc"].abs().max() if len(d) else 1), thr) * 1.08
    ax.set_xlim(-m, m)
    ax.set_ylim(0, max(float(d["y"].max() if len(d) else 1), -np.log10(pthr)) * 1.08)
    for v in (-thr, thr):
        ax.axvline(v, color="#666666", ls="--", lw=0.8)
    ax.axhline(-np.log10(pthr), color="#666666", ls="--", lw=0.8)
    lab = d[up | dn].sort_values("p").head(int(o["top"]))
    hl = [h.strip() for h in str(o["highlight"]).split(",") if h.strip()]
    if hl:
        lab = pd.concat([lab, d[d["g"].isin(hl)]]).drop_duplicates()
    _repel(ax, lab["fc"], lab["y"], lab["g"], st)
    st.labels(ax, "log2 fold change", "−log10 p-value" if o["plog"] == "p-value" else "−log10 p")
    st.legend(ax, loc="upper center", markerscale=1.6)


@plot("cerebro_115", "Cerebro colour-map volcano", T, number="115", example=_ex_deg, roles=_DEG_ROLES,
      opts=[Opt("plog", "p column is", "choice", "p-value", choices=["p-value", "−log10 p"]),
            Opt("top", "Label top N genes", "int", 12, lo=0, hi=100),
            Opt("fc_thr", "|log2FC| guide line (0 = none)", "float", 1.0, lo=0, hi=20, step=0.1),
            Opt("p_thr", "p guide line (0 = none)", "float", 0.05, lo=0, hi=1, step=0.01),
            Opt("smax", "Largest point size", "float", 90, lo=5, hi=400)],
      desc="One row per gene: gene name, log2 fold change and p-value; colour = signed −log10 p × |log2FC| "
           "gradient, size = significance.", size=(7.5, 6))
def cerebro_115(fig, df, c, o, st):
    d = _deg_frame(df, c, o)
    d["score"] = np.sign(d["fc"]) * d["y"] * d["fc"].abs()
    d = d.iloc[np.argsort(d["score"].abs().to_numpy())]
    lim = float(np.nanpercentile(d["score"].abs(), 99)) if len(d) else 1.0
    lim = lim if lim > 0 else 1.0
    norm = Normalize(-lim, lim)
    cm = _cmap(st)
    ax = st.ax(fig.add_subplot())
    sz = _sizes(d["y"], 4, float(o["smax"]), 0, max(d["y"].max(), 1e-9))
    ax.scatter(d["fc"], d["y"], s=sz, c=d["score"].clip(-lim, lim), cmap=cm, norm=norm, lw=0.2,
               edgecolor="white", alpha=0.9, rasterized=True)
    m = float(d["fc"].abs().max() if len(d) else 1) * 1.08
    ax.set_xlim(-m, m)
    ax.set_ylim(0, float(d["y"].max() if len(d) else 1) * 1.08)
    if o["fc_thr"] > 0:
        for v in (-o["fc_thr"], o["fc_thr"]):
            ax.axvline(v, color="#888888", ls=":", lw=0.8)
    if 0 < o["p_thr"] < 1:
        ax.axhline(-np.log10(o["p_thr"]), color="#888888", ls=":", lw=0.8)
    lab = d.reindex(d["score"].abs().sort_values(ascending=False).index).head(int(o["top"]))
    _repel(ax, lab["fc"], lab["y"], lab["g"], st)
    _cbar(fig, st, cm, norm, "sign(FC) · −log10 p · |log2FC|", ax=ax)
    ymax = max(float(d["y"].max()), 1e-9)
    hs = _size_handles([ymax * 0.25, ymax * 0.5, ymax], 4, float(o["smax"]), n=3, integer=False,
                       fmt="{:.0f}")
    for h, v in zip(hs, [ymax * 0.25, ymax * 0.5, ymax]):
        h.set_markersize(np.sqrt(_sizes([v], 4, float(o["smax"]), 0, ymax)[0]))
    st.legend(ax, hs, title="−log10 p", loc="upper center")
    st.labels(ax, "log2 fold change", "−log10 p-value")


# ── enrichment examples ──────────────────────────────────────────────────────

_GO = {
    "BP": ["inflammatory response", "response to lipopolysaccharide", "cytokine-mediated signaling pathway",
           "regulation of cell migration", "positive regulation of cell death", "angiogenesis",
           "extracellular matrix organization", "leukocyte chemotaxis", "response to hypoxia",
           "regulation of apoptotic process", "T cell activation", "wound healing",
           "positive regulation of ERK1 and ERK2 cascade", "neutrophil degranulation", "cell-cell adhesion"],
    "CC": ["extracellular space", "cell surface", "focal adhesion", "collagen-containing extracellular matrix",
           "membrane raft", "endoplasmic reticulum lumen", "secretory granule lumen", "basement membrane"],
    "MF": ["cytokine activity", "growth factor binding", "integrin binding", "chemokine receptor binding",
           "extracellular matrix structural constituent", "receptor ligand activity", "heparin binding",
           "protease binding"],
}
_KEGG = {
    "Metabolism": ["Glycolysis / Gluconeogenesis", "Oxidative phosphorylation", "Fatty acid degradation",
                   "Steroid biosynthesis"],
    "Genetic Information Processing": ["Ribosome", "Spliceosome", "Proteasome", "DNA replication"],
    "Environmental Information Processing": ["MAPK signaling pathway", "PI3K-Akt signaling pathway",
                                             "Cytokine-cytokine receptor interaction", "ECM-receptor interaction"],
    "Cellular Processes": ["Focal adhesion", "Cell cycle", "Apoptosis", "p53 signaling pathway"],
    "Organismal Systems": ["Toll-like receptor signaling pathway", "Chemokine signaling pathway",
                           "Complement and coagulation cascades"],
    "Human Diseases": ["Pathways in cancer", "Rheumatoid arthritis", "Tuberculosis", "Hepatitis B"],
}


def _enrich_rows(terms, seed, N=180):
    r = rng(seed)
    cnt = r.integers(6, 42, len(terms))
    p = 10 ** -(r.uniform(1.5, 12, len(terms)))
    p = np.sort(p)
    cnt = np.sort(cnt)[::-1] + r.integers(-4, 5, len(terms))
    cnt = np.clip(cnt, 3, None)
    return cnt, np.round(cnt / N, 4), p


def _ex_enrich():
    terms = _GO["BP"]
    cnt, ratio, p = _enrich_rows(terms, 81)
    return pd.DataFrame({"Term": terms, "GeneRatio": ratio, "Count": cnt, "p.adjust": p})


def _ex_multi_groups():
    r = rng(16)
    groups = ["Day1 vs Ctrl", "Day3 vs Ctrl", "Day7 vs Ctrl", "Day14 vs Ctrl"]
    terms = _GO["BP"][:12]
    rows = []
    for gi, g in enumerate(groups):
        for ti, t in enumerate(terms):
            if r.random() < 0.25:
                continue
            strength = 1 + (gi if ti < 6 else 3 - gi)
            rows.append((t, g, int(r.integers(4, 12) * strength / 1.5 + 3), 10 ** -(r.uniform(1.4, 3) * strength)))
    return pd.DataFrame(rows, columns=["Term", "Group", "Count", "p.adjust"])


def _ex_enrich_bar():
    terms = _KEGG["Environmental Information Processing"] + _KEGG["Cellular Processes"] + \
        _KEGG["Organismal Systems"] + _KEGG["Human Diseases"][:3]
    cnt, ratio, p = _enrich_rows(terms, 105)
    return pd.DataFrame({"Term": terms, "pvalue": p, "Count": cnt})


def _ex_go_kegg():
    rows = []
    r = rng(111)
    for db, terms in (("GO", _GO["BP"][:8]), ("KEGG", [t for v in _KEGG.values() for t in v][8:16])):
        cnt, ratio, p = _enrich_rows(terms, int(r.integers(1000)))
        rows += [(t, db, int(k), float(pp)) for t, k, pp in zip(terms, cnt, p)]
    return pd.DataFrame(rows, columns=["Term", "Database", "Count", "pvalue"])


def _ex_bp_cc_mf():
    rows = []
    for i, (ont, terms) in enumerate(_GO.items()):
        tt = terms[:8]
        cnt, ratio, p = _enrich_rows(tt, 46 + i)
        rows += [(t, ont, int(k), float(pp)) for t, k, pp in zip(tt, cnt, p)]
    return pd.DataFrame(rows, columns=["Term", "Ontology", "Count", "pvalue"])


def _ex_pathway_summary():
    r = rng(124)
    rows = [(t, cls, int(r.integers(4, 60))) for cls, ts in _KEGG.items() for t in ts]
    return pd.DataFrame(rows, columns=["Pathway", "Class", "Count"])


def _ex_go_result():
    rows = []
    for i, (cat, terms) in enumerate(list(_GO.items()) + [("KEGG", _KEGG["Environmental Information Processing"]
                                                             + _KEGG["Cellular Processes"][:2])]):
        tt = terms[:6]
        cnt, ratio, p = _enrich_rows(tt, 122 + i)
        rows += [(t, cat, float(rr), int(k), float(pp)) for t, k, rr, pp in zip(tt, cnt, ratio, p)]
    return pd.DataFrame(rows, columns=["Term", "Category", "GeneRatio", "Count", "p.adjust"])


# ── enrichment plots ─────────────────────────────────────────────────────────

def _sorted_terms(d, by, top):
    d = d.sort_values("p").head(int(top))
    return d.sort_values(by, ascending=True) if by in d else d


def _pnorm(p):
    lp = -np.log10(np.clip(np.asarray(p, float), 1e-300, 1))
    return lp


@plot("enrich_bubble_081", "Enrichment bubble plot", T, number="081", example=_ex_enrich,
      roles=[Role("term", "Terms", "cat"), Role("ratio", "Gene ratio (number or k/N)", "any"),
             Role("count", "Gene count", "num"), Role("padj", "Adjusted p-value", "num")],
      opts=[Opt("top", "Show top N terms (by p)", "int", 15, lo=1, hi=100),
            Opt("sort", "Order terms by", "choice", "gene ratio", choices=["gene ratio", "p-value", "count"]),
            Opt("wrap", "Wrap term names at (characters)", "int", 45, lo=10, hi=200),
            Opt("smax", "Largest bubble size", "float", 320, lo=20, hi=1500)],
      desc="One row per enriched term: term, gene ratio, gene count and adjusted p-value (clusterProfiler-like).",
      size=(8, 6))
def enrich_bubble_081(fig, df, c, o, st):
    d = pd.DataFrame({"t": as_str(df, c["term"]), "x": _ratio(df, c["ratio"]), "n": num(df, c["count"]),
                      "p": num(df, c["padj"])}).dropna()
    by = {"gene ratio": "x", "p-value": "lp", "count": "n"}[o["sort"]]
    d["lp"] = _pnorm(d["p"])
    d = _sorted_terms(d, by, o["top"])
    gs, cax, lax = _side(fig)
    ax = st.ax(fig.add_subplot(gs[:, 0]))
    cm = _cmap(st, reverse=True)
    norm = _plog_norm(d["p"])
    y = np.arange(len(d))
    ax.scatter(d["x"], y, s=_sizes(d["n"], 25, o["smax"]), c=d["p"], cmap=cm, norm=norm, edgecolor="#333333",
               lw=0.5, zorder=3)
    ax.set_yticks(y, [_wrap(t, o["wrap"]) for t in d["t"]], fontsize=st.font * 0.8)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.margins(x=0.12)
    _cbar(fig, st, cm, norm, "p.adjust", cax=cax)
    st.legend(lax, _size_handles(d["n"], 25, o["smax"]), title="Count", loc="center left", labelspacing=1.1)
    st.labels(ax, "Gene ratio", "")


@plot("multi_groups_bubble_016", "Multi-group enrichment bubble plot", T, number="016",
      example=_ex_multi_groups,
      roles=[Role("term", "Terms", "cat"), Role("group", "Groups / comparisons", "cat"),
             Role("count", "Gene count", "num"), Role("padj", "Adjusted p-value", "num")],
      opts=[Opt("top", "Max terms", "int", 25, lo=1, hi=200),
            Opt("order", "Order terms", "choice", "cluster by pattern",
                choices=["cluster by pattern", "as in table", "by best p"]),
            Opt("wrap", "Wrap term names at (characters)", "int", 45, lo=10, hi=200),
            Opt("smax", "Largest bubble size", "float", 300, lo=20, hi=1500)],
      desc="Long table, one row per term × group: term, group (comparison), gene count, adjusted p-value.",
      size=(8, 6))
def multi_groups_bubble_016(fig, df, c, o, st):
    d = pd.DataFrame({"t": as_str(df, c["term"]), "g": as_str(df, c["group"]), "n": num(df, c["count"]),
                      "p": num(df, c["padj"])}).dropna()
    groups = cats(d, "g")
    best = d.groupby("t")["p"].min().sort_values()
    keep = list(best.index[:int(o["top"])])
    d = d[d["t"].isin(keep)]
    if o["order"] == "as in table":
        terms = [t for t in cats(d, "t")][::-1]
    elif o["order"] == "by best p":
        terms = list(best[best.index.isin(keep)].sort_values(ascending=False).index)
    else:
        M = d.pivot_table(index="t", columns="g", values="p", aggfunc="min").reindex(columns=groups)
        L = _pnorm(M.fillna(1).to_numpy())
        w = (L * np.arange(1, len(groups) + 1)).sum(1) / np.maximum(L.sum(1), 1e-9)
        terms = list(M.index[np.lexsort((-L.max(1), w))])[::-1]
    yi = {t: i for i, t in enumerate(terms)}
    xi = {g: i for i, g in enumerate(groups)}
    gs, cax, lax = _side(fig)
    ax = st.ax(fig.add_subplot(gs[:, 0]))
    cm = _cmap(st, reverse=True)
    norm = _plog_norm(d["p"])
    ax.scatter(d["g"].map(xi), d["t"].map(yi), s=_sizes(d["n"], 20, o["smax"]), c=d["p"], cmap=cm, norm=norm,
               edgecolor="#333333", lw=0.5, zorder=3)
    ax.set_xticks(range(len(groups)), groups, rotation=30 if len(groups) > 3 else 0,
                  ha="right" if len(groups) > 3 else "center")
    ax.set_yticks(range(len(terms)), [_wrap(t, o["wrap"]) for t in terms], fontsize=st.font * 0.8)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(-0.7, len(terms) - 0.3)
    _cbar(fig, st, cm, norm, "p.adjust", cax=cax)
    st.legend(lax, _size_handles(d["n"], 20, o["smax"]), title="Count", loc="center left", labelspacing=1.1)
    st.labels(ax, "", "")


@plot("enrich_bar_color_105", "Enrichment bar plot with colour", T, number="105", example=_ex_enrich_bar,
      roles=[Role("term", "Terms", "cat"), Role("p", "p-value", "num"),
             Role("value", "Colour by (count / gene ratio)", "num")],
      opts=[Opt("top", "Show top N terms", "int", 20, lo=1, hi=100),
            Opt("plog", "p column is", "choice", "p-value", choices=["p-value", "−log10 p"]),
            Opt("wrap", "Wrap term names at (characters)", "int", 45, lo=10, hi=200),
            Opt("labels", "Show −log10 p on bars", "bool", False)],
      desc="One row per term: term, p-value and a value for the colour (gene count or gene ratio).",
      size=(8, 6))
def enrich_bar_color_105(fig, df, c, o, st):
    d = pd.DataFrame({"t": as_str(df, c["term"]), "p": num(df, c["p"]), "v": num(df, c["value"])}).dropna()
    d["lp"] = d["p"] if o["plog"] == "−log10 p" else _pnorm(d["p"])
    d = d.sort_values("lp", ascending=False).head(int(o["top"])).iloc[::-1]
    ax = st.ax(fig.add_subplot(), "x")
    cm = _cmap(st)
    norm = Normalize(d["v"].min(), d["v"].max()) if len(d) and d["v"].max() > d["v"].min() else \
        Normalize(d["v"].min() - 1, d["v"].max() + 1)
    y = np.arange(len(d))
    bars = ax.barh(y, d["lp"], color=cm(norm(d["v"])), edgecolor="#333333", lw=0.5, height=0.72)
    if o["labels"]:
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=st.font * 0.7)
    ax.set_yticks(y, [_wrap(t, o["wrap"]) for t in d["t"]], fontsize=st.font * 0.8)
    ax.set_ylim(-0.6, len(d) - 0.4)
    _cbar(fig, st, cm, norm, c["value"], ax=ax)
    st.labels(ax, "−log10 p-value", "")


@plot("go_kegg_bar_dot_111", "GO / KEGG bar + dot plot", T, number="111", example=_ex_go_kegg,
      roles=[Role("term", "Terms", "cat"), Role("db", "Database (GO / KEGG …)", "cat"),
             Role("count", "Gene count", "num"), Role("p", "p-value", "num")],
      opts=[Opt("top", "Top N terms per database", "int", 10, lo=1, hi=100),
            Opt("wrap", "Wrap term names at (characters)", "int", 42, lo=10, hi=200),
            Opt("dsize", "Dot size", "float", 70, lo=5, hi=500)],
      desc="One row per term: term, database (GO, KEGG, Reactome…), gene count and p-value.", size=(9, 6))
def go_kegg_bar_dot_111(fig, df, c, o, st):
    d = pd.DataFrame({"t": as_str(df, c["term"]), "db": as_str(df, c["db"]), "n": num(df, c["count"]),
                      "p": num(df, c["p"])}).dropna()
    d["lp"] = _pnorm(d["p"])
    dbs = cats(d, "db")
    d = pd.concat([d[d["db"] == b].sort_values("lp", ascending=False).head(int(o["top"])) for b in dbs])
    d = d.iloc[::-1].reset_index(drop=True)
    col = dict(zip(dbs, st.colors(len(dbs))))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    a1 = st.ax(fig.add_subplot(gs[0]), "x")
    a2 = st.ax(fig.add_subplot(gs[1], sharey=a1), "x")
    y = np.arange(len(d))
    a1.barh(y, d["n"], color=d["db"].map(col), edgecolor="#333333", lw=0.5, height=0.7)
    a1.set_yticks(y, [_wrap(t, o["wrap"]) for t in d["t"]], fontsize=st.font * 0.8)
    a1.set_ylim(-0.6, len(d) - 0.4)
    a1.set_xlabel("Gene count", fontsize=st.font)
    a2.hlines(y, 0, d["lp"], color="#cccccc", lw=1, zorder=1)
    a2.scatter(d["lp"], y, s=o["dsize"], c=d["db"].map(col), edgecolor="#333333", lw=0.5, zorder=3)
    a2.tick_params(labelleft=False)
    a2.set_xlim(0, d["lp"].max() * 1.12 if len(d) else 1)
    a2.set_xlabel("−log10 p-value", fontsize=st.font)
    st.legend(a2, [Patch(fc=col[b], ec="#333333", lw=0.5) for b in dbs], dbs, title=c["db"], loc="lower right")
    st.suptitle(fig)
    if st.opts.get("xlabel"):
        a1.set_xlabel(st.opts["xlabel"], fontsize=st.font)
    if st.opts.get("ylabel"):
        a1.set_ylabel(st.opts["ylabel"], fontsize=st.font)


def _grouped_hbars(fig, st, d, xlabel, wrapw=45, gwrap=20, sort=True, cols=None, show_vals=False):
    """Horizontal bars grouped into coloured blocks with a labelled colour strip on the right.

    *d* has columns t (term), g (group), v (value).
    """
    groups = cats(d, "g")
    cols = cols or dict(zip(groups, st.colors(len(groups))))
    gs = fig.add_gridspec(1, 2, width_ratios=[30, 1])
    ax = st.ax(fig.add_subplot(gs[0]), "x")
    sax = fig.add_subplot(gs[1], sharey=ax)
    sax.set_axis_off()
    y, ticks, labels, blocks = 0.0, [], [], []
    rows = []
    for g in groups:
        sub = d[d["g"] == g]
        if sort:
            sub = sub.sort_values("v", ascending=False)
        top = y
        for _, r in sub.iterrows():
            rows.append((y, r["v"], cols[g]))
            ticks.append(y)
            labels.append(_wrap(r["t"], wrapw))
            y += 1
        blocks.append((g, top - 0.45, y - 0.55))
        y += 0.6
    ys = np.array([r[0] for r in rows])
    bars = ax.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], edgecolor="#333333", lw=0.5,
                   height=0.72)
    if show_vals:
        ax.bar_label(bars, fmt="%.4g", padding=2, fontsize=st.font * 0.7)
    ax.set_yticks(ticks, labels, fontsize=st.font * 0.78)
    ax.set_ylim(y - 0.4, -0.8)
    for g, a, b in blocks:
        sax.add_patch(Rectangle((0, a), 1, b - a, fc=cols[g], ec="none", clip_on=False))
        sax.text(1.6, (a + b) / 2, _wrap(g, gwrap), ha="left", va="center", fontsize=st.font * 0.8,
                 color=cols[g], fontweight="bold", clip_on=False)
    sax.set_xlim(0, 1)
    ax.margins(x=0.08 if show_vals else 0.02)
    st.labels(ax, xlabel, "")
    return ax


@plot("bp_cc_mf_046", "GO BP / CC / MF three-in-one bars", T, number="046", example=_ex_bp_cc_mf,
      roles=[Role("term", "GO terms", "cat"), Role("ont", "Ontology (BP / CC / MF)", "cat"),
             Role("value", "Value (count or p-value)", "num")],
      opts=[Opt("value", "Value column is", "choice", "count", choices=["count", "p-value (plot −log10)"]),
            Opt("top", "Top N terms per ontology", "int", 10, lo=1, hi=100),
            Opt("wrap", "Wrap term names at (characters)", "int", 45, lo=10, hi=200),
            Opt("vals", "Show values", "bool", True)],
      desc="One row per GO term: term, ontology (BP, CC or MF) and gene count (or p-value).", size=(8, 7.5))
def bp_cc_mf_046(fig, df, c, o, st):
    d = pd.DataFrame({"t": as_str(df, c["term"]), "g": as_str(df, c["ont"]), "v": num(df, c["value"])}).dropna()
    if o["value"].startswith("p"):
        d["v"] = _pnorm(d["v"])
    names = {"BP": "Biological process", "CC": "Cellular component", "MF": "Molecular function"}
    d["g"] = d["g"].map(lambda s: names.get(s.strip().upper(), s))
    d = d.groupby("g", sort=False, group_keys=False).apply(lambda x: x.nlargest(int(o["top"]), "v"))
    _grouped_hbars(fig, st, d, "−log10 p-value" if o["value"].startswith("p") else "Gene count",
                   o["wrap"], 12, show_vals=o["vals"])


@plot("pathway_summary_124", "KEGG pathway classification summary", T, number="124",
      example=_ex_pathway_summary,
      roles=[Role("term", "Pathways", "cat"), Role("cls", "Level-1 class", "cat"), Role("value", "Gene count", "num")],
      opts=[Opt("sort", "Sort pathways within class", "bool", True), Opt("vals", "Show counts", "bool", True),
            Opt("wrap", "Wrap pathway names at (characters)", "int", 45, lo=10, hi=200)],
      desc="One row per pathway: pathway name, its KEGG level-1 class (Metabolism, Human Diseases…) and gene count.",
      size=(9, 7.5))
def pathway_summary_124(fig, df, c, o, st):
    d = pd.DataFrame({"t": as_str(df, c["term"]), "g": as_str(df, c["cls"]), "v": num(df, c["value"])}).dropna()
    _grouped_hbars(fig, st, d, "Number of genes", o["wrap"], 20, sort=o["sort"], show_vals=o["vals"])


@plot("go_pathway_enrichment_122", "GO / pathway enrichment result (dot + bar)", T, number="122",
      example=_ex_go_result,
      roles=[Role("term", "Terms", "cat"), Role("cat", "Category (BP / CC / MF / KEGG)", "cat"),
             Role("ratio", "Gene ratio (number or k/N)", "any"), Role("count", "Gene count", "num"),
             Role("padj", "Adjusted p-value", "num")],
      opts=[Opt("top", "Top N terms per category", "int", 6, lo=1, hi=50),
            Opt("wrap", "Wrap term names at (characters)", "int", 42, lo=10, hi=200),
            Opt("smax", "Largest bubble size", "float", 240, lo=20, hi=1500)],
      desc="Plots an enrichment RESULT table (term, category, gene ratio, count, p.adjust) as a dot plot + "
           "−log10 p bars; the enrichment test itself is not run.", size=(10, 7.5))
def go_pathway_enrichment_122(fig, df, c, o, st):
    d = pd.DataFrame({"t": as_str(df, c["term"]), "g": as_str(df, c["cat"]), "x": _ratio(df, c["ratio"]),
                      "n": num(df, c["count"]), "p": num(df, c["padj"])}).dropna()
    d["lp"] = _pnorm(d["p"])
    groups = cats(d, "g")
    col = dict(zip(groups, st.colors(len(groups))))
    parts = [d[d["g"] == g].nlargest(int(o["top"]), "lp").sort_values("x") for g in groups]
    yy, y = [], 0.0
    for p_ in parts:
        yy += list(np.arange(len(p_)) + y)
        y += len(p_) + 0.6
    d = pd.concat(parts)
    d["y"] = yy
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 0.07], height_ratios=[1, 1.3])
    a1 = st.ax(fig.add_subplot(gs[:, 0]))
    a2 = st.ax(fig.add_subplot(gs[:, 1], sharey=a1), "x")
    cax = fig.add_subplot(gs[0, 2])
    lax = fig.add_subplot(gs[1, 2])
    lax.set_axis_off()
    cm = _cmap(st, reverse=True)
    norm = _plog_norm(d["p"])
    a1.scatter(d["x"], d["y"], s=_sizes(d["n"], 20, o["smax"]), c=d["p"], cmap=cm, norm=norm,
               edgecolor="#333333", lw=0.5, zorder=3)
    a1.set_yticks(d["y"], [_wrap(t, o["wrap"]) for t in d["t"]], fontsize=st.font * 0.75)
    for lab, tick in zip(a1.get_yticklabels(), d["g"]):
        lab.set_color(col[tick])
    a1.set_ylim(y - 0.2, -0.8)
    a1.margins(x=0.15)
    a1.set_xlabel(st.opts.get("xlabel") or "Gene ratio", fontsize=st.font)
    a2.barh(d["y"], d["lp"], color=d["g"].map(col), edgecolor="#333333", lw=0.5, height=0.72)
    a2.tick_params(labelleft=False)
    a2.set_xlabel("−log10 p.adjust", fontsize=st.font)
    _cbar(fig, st, cm, norm, "p.adjust", cax=cax)
    _legend_sections(st, lax, [("Category", [Patch(fc=col[g], ec="#333333", lw=0.5, label=g) for g in groups]),
                               ("Count", _size_handles(d["n"], 20, o["smax"], n=3))], loc="center left")
    st.suptitle(fig)


def _ex_sankey_dot():
    r = rng(121)
    paths = ["MAPK signaling pathway", "PI3K-Akt signaling pathway", "Apoptosis", "p53 signaling pathway",
             "Cytokine-cytokine receptor interaction", "Focal adhesion"]
    genes = ["EGFR", "KRAS", "MAPK1", "FOS", "JUN", "PIK3CA", "AKT1", "BCL2", "BAX", "CASP3", "TP53", "MDM2",
             "CDKN1A", "IL6", "TNF", "CXCL8", "ITGB1", "VEGFA"]
    links = {0: [0, 1, 2, 3, 4, 13, 14], 1: [0, 1, 5, 6, 7, 13, 16, 17], 2: [6, 7, 8, 9, 10, 14],
             3: [7, 8, 10, 11, 12], 4: [13, 14, 15, 17], 5: [0, 5, 6, 7, 16, 17]}
    rows = []
    for pi, gl in links.items():
        cnt = len(gl) + int(r.integers(5, 25))
        ratio, p = round(cnt / 160, 4), float(10 ** -r.uniform(2, 9))
        rows += [(genes[g], paths[pi], ratio, cnt, p) for g in gl]
    return pd.DataFrame(rows, columns=["Gene", "Pathway", "GeneRatio", "Count", "pvalue"])


@plot("sankey_dot_121", "Sankey + dot plot (genes → pathways)", T, number="121", example=_ex_sankey_dot,
      roles=[Role("gene", "Genes", "cat"), Role("path", "Pathways / terms", "cat"),
             Role("ratio", "Gene ratio (number or k/N)", "any"), Role("count", "Gene count", "num"),
             Role("p", "p-value", "num")],
      opts=[Opt("wrap", "Wrap pathway names at (characters)", "int", 32, lo=10, hi=200),
            Opt("alpha", "Ribbon opacity", "float", 0.45, lo=0.05, hi=1, step=0.05),
            Opt("smax", "Largest dot size", "float", 260, lo=20, hi=1500)],
      desc="Long table, one row per gene–pathway link: gene, pathway, and the pathway's gene ratio, count and p.",
      size=(11, 7))
def sankey_dot_121(fig, df, c, o, st):
    d = pd.DataFrame({"g": as_str(df, c["gene"]), "t": as_str(df, c["path"]), "x": _ratio(df, c["ratio"]),
                      "n": num(df, c["count"]), "p": num(df, c["p"])}).dropna(subset=["x", "p"])
    d = d.drop_duplicates(["g", "t"])
    paths = cats(d, "t")
    genes = cats(d, "g")
    pi = {t: i for i, t in enumerate(paths)}
    gmean = d.assign(k=d["t"].map(pi)).groupby("g")["k"].mean()
    genes = sorted(genes, key=lambda g: gmean[g])
    gi = {g: i for i, g in enumerate(genes)}
    E = len(d)
    gap = 0.012
    unit = min((1 - gap * (len(genes) - 1)) / E, (1 - 0.04 * (len(paths) - 1)) / E)
    deg_g = d["g"].value_counts()
    deg_t = d["t"].value_counts()

    def stack(order, deg, g_):
        tot = sum(deg[k] * unit for k in order) + g_ * (len(order) - 1)
        y, out = (1 + tot) / 2, {}
        for k in order:
            out[k] = (y - deg[k] * unit, y)
            y -= deg[k] * unit + g_
        return out
    G = stack(genes, deg_g, gap)
    P = stack(paths, deg_t, 0.04)
    cols = dict(zip(paths, st.colors(len(paths))))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1, 0.07], height_ratios=[1, 1.3])
    a1 = fig.add_subplot(gs[:, 0])
    a2 = st.ax(fig.add_subplot(gs[:, 1], sharey=a1))
    cax = fig.add_subplot(gs[0, 2])
    lax = fig.add_subplot(gs[1, 2])
    lax.set_axis_off()
    a1.set_axis_off()
    gcur = {g: G[g][1] for g in genes}
    tcur = {t: P[t][1] for t in paths}
    # order segments: per gene by pathway index, per pathway by gene index (fewest crossings)
    links = sorted(zip(d["g"], d["t"]), key=lambda z: (gi[z[0]], pi[z[1]]))
    gseg = {}
    for g, t in links:
        gseg[(g, t)] = (gcur[g] - unit, gcur[g])
        gcur[g] -= unit
    for g, t in sorted(zip(d["g"], d["t"]), key=lambda z: (pi[z[1]], gi[z[0]])):
        a, b = gseg[(g, t)]
        ta, tb = tcur[t] - unit, tcur[t]
        tcur[t] -= unit
        a1.add_patch(PathPatch(_ribbon(0.06, b, a, 0.94, tb, ta), fc=cols[t], ec="none", alpha=o["alpha"]))
    for g in genes:
        a, b = G[g]
        a1.add_patch(Rectangle((0.0, a), 0.06, b - a, fc="#555555", ec="white", lw=0.4))
        a1.text(-0.01, (a + b) / 2, g, ha="right", va="center", fontsize=st.font * 0.72)
    for t in paths:
        a, b = P[t]
        a1.add_patch(Rectangle((0.94, a), 0.06, b - a, fc=cols[t], ec="white", lw=0.4))
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    dd = d.drop_duplicates("t").set_index("t").loc[paths]
    yc = [(P[t][0] + P[t][1]) / 2 for t in paths]
    cm = _cmap(st, reverse=True)
    norm = _plog_norm(dd["p"])
    nn = dd["n"].fillna(dd["n"].median() if dd["n"].notna().any() else 1)
    a2.scatter(dd["x"], yc, s=_sizes(nn, 30, o["smax"]), c=dd["p"], cmap=cm, norm=norm, edgecolor="#333333",
               lw=0.5, zorder=3)
    a2.set_yticks(yc, [_wrap(t, o["wrap"]) for t in paths], fontsize=st.font * 0.78)
    a2.margins(x=0.2)
    a2.set_xlabel(st.opts.get("xlabel") or "Gene ratio", fontsize=st.font)
    _cbar(fig, st, cm, norm, "p-value", cax=cax)
    st.legend(lax, _size_handles(nn, 30, o["smax"], n=3), title="Count", loc="center left", labelspacing=1.1)
    st.suptitle(fig)


def _ex_go_chord():
    r = rng(85)
    terms = ["inflammatory response", "angiogenesis", "response to hypoxia", "regulation of apoptotic process",
             "cell migration"]
    genes = ["IL6", "TNF", "CXCL8", "CCL2", "VEGFA", "HIF1A", "EPAS1", "ANGPT2", "BCL2", "BAX", "CASP3",
             "MMP9", "ITGB1", "SERPINE1", "PTGS2", "HMOX1"]
    fc = dict(zip(genes, np.round(r.normal(0, 2.0, len(genes)), 2)))
    links = {0: [0, 1, 2, 3, 14, 15], 1: [4, 5, 6, 7, 11, 13], 2: [4, 5, 6, 15, 13], 3: [8, 9, 10, 1, 15],
             4: [2, 3, 11, 12, 13, 7]}
    rows = [(genes[g], terms[t], fc[genes[g]]) for t, gl in links.items() for g in gl]
    return pd.DataFrame(rows, columns=["Gene", "GO term", "log2FC"])


@plot("go_chord_085", "GO chord diagram", T, number="085", example=_ex_go_chord,
      roles=[Role("gene", "Genes", "cat"), Role("term", "GO terms", "cat"),
             Role("fc", "Gene log2 fold change", "num", optional=True)],
      opts=[Opt("gap", "Gap between items (degrees)", "float", 1.2, lo=0, hi=10, step=0.2),
            Opt("alpha", "Ribbon opacity", "float", 0.6, lo=0.05, hi=1, step=0.05),
            Opt("wrap", "Wrap term names at (characters)", "int", 34, lo=10, hi=200)],
      desc="Long table, one row per gene–term link: gene, GO term and the gene's log2 fold change.",
      size=(9, 7.5), axis_labels=False)
def go_chord_085(fig, df, c, o, st):
    fc = num(df, c["fc"]) if c.get("fc") else pd.Series(np.nan, index=df.index)
    d = pd.DataFrame({"g": as_str(df, c["gene"]), "t": as_str(df, c["term"]), "fc": fc})
    d = d.drop_duplicates(["g", "t"])
    terms = cats(d, "t")
    ti = {t: i for i, t in enumerate(terms)}
    genes = sorted(cats(d, "g"), key=lambda g: d.loc[d["g"] == g, "t"].map(ti).min())
    gi = {g: i for i, g in enumerate(genes)}
    E = len(d)
    big, gap = 14.0, float(o["gap"])
    avail = 360 - 2 * big - gap * (len(genes) - 1 + len(terms) - 1)
    unit = avail / (2 * E)
    ang = 90 + big / 2      # start top, go clockwise through genes, then terms
    spans = {}
    for g in genes:
        w = (d["g"] == g).sum() * unit
        spans[("g", g)] = (ang - w, ang)
        ang -= w + gap
    ang -= big - gap
    for t in terms:
        w = (d["t"] == t).sum() * unit
        spans[("t", t)] = (ang - w, ang)
        ang -= w + gap
    cols = dict(zip(terms, st.colors(len(terms))))
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    st.bare(ax)
    R = 1.0
    fcv = d.drop_duplicates("g").set_index("g")["fc"]
    lim = float(np.nanmax(np.abs(fcv))) if fcv.notna().any() else 1.0
    cm, norm = _cmap(st), Normalize(-lim, lim)
    # gene names as large as their slice of the circle allows (they must not touch at small sizes)
    fw, fh = fig.get_size_inches()
    # the circle spans ~2.9 data units of an axes that gets ~55 % of the width (legend, colour bar)
    inch_per_unit = min(fw * 0.55, fh * 0.85) / 2.9
    min_arc = min((spans[("g", g)][1] - spans[("g", g)][0] + gap) for g in genes) if genes else 360
    arc_pt = np.radians(min_arc) * (R + 0.21) * inch_per_unit * 72
    gene_fs = max(4.0, min(st.font * 0.78, arc_pt * 0.6))
    for g in genes:
        a, b = spans[("g", g)]
        ax.add_patch(Wedge((0, 0), R + 0.07, a, b, width=0.07, fc="#666666", ec="white", lw=0.5))
        v = fcv.get(g, np.nan)
        ax.add_patch(Wedge((0, 0), R + 0.17, a, b, width=0.08, fc=cm(norm(v)) if np.isfinite(v) else "#eeeeee",
                           ec="white", lw=0.5))
        _radial_text(ax, np.radians((a + b) / 2), R + 0.21, g, gene_fs)
    for t in terms:
        a, b = spans[("t", t)]
        ax.add_patch(Wedge((0, 0), R + 0.07, a, b, width=0.07, fc=cols[t], ec="white", lw=0.5))
    gcur = {g: spans[("g", g)][1] for g in genes}
    tcur = {t: spans[("t", t)][0] for t in terms}
    for g, t in sorted(zip(d["g"], d["t"]), key=lambda z: (gi[z[0]], -ti[z[1]])):
        a2 = gcur[g]
        a1 = a2 - unit
        gcur[g] = a1
        b1 = tcur[t]
        b2 = b1 + unit
        tcur[t] = b2
        ax.add_patch(PathPatch(_chord_path(np.radians(a1), np.radians(a2), np.radians(b1), np.radians(b2), R),
                               fc=cols[t], ec="none", alpha=o["alpha"]))
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    if fcv.notna().any():
        _cbar(fig, st, cm, norm, "log2FC", ax=ax, shrink=0.35, location="left")
    st.legend(ax, [Patch(fc=cols[t], label=_wrap(t, o["wrap"])) for t in terms], title=c["term"],
              **_leg_out(st))
    st.labels(ax, "", "", "")


def _ex_m6a():
    r = rng(88)
    n = 1200
    x = r.normal(0, 1.1, n)
    y = 0.55 * x + r.normal(0, 1.0, n)
    names = np.array([f"GENE{i:04d}" for i in range(n)], dtype=object)
    names[np.argsort(-np.abs(x * y))[:len(_SYMS)]] = _SYMS
    return pd.DataFrame({"Gene": names, "m6A_log2FC": np.round(x, 3), "Expression_log2FC": np.round(y, 3)})


@plot("m6a_expression_088", "m6A vs expression log2FC (four quadrants)", T, number="088", example=_ex_m6a,
      roles=[Role("gene", "Genes (labels)", "cat", optional=True), Role("x", "m6A log2FC", "num"),
             Role("y", "Expression log2FC", "num")],
      opts=[Opt("tx", "m6A |log2FC| threshold", "float", 0.58, lo=0, hi=20, step=0.1),
            Opt("ty", "Expression |log2FC| threshold", "float", 0.58, lo=0, hi=20, step=0.1),
            Opt("top", "Label top N genes", "int", 8, lo=0, hi=100),
            Opt("psize", "Point size", "float", 12, lo=1, hi=200)],
      desc="One row per gene: gene, m6A (methylation) log2 fold change and expression log2 fold change.",
      size=(7, 6.5))
def m6a_expression_088(fig, df, c, o, st):
    g = as_str(df, c["gene"]) if c.get("gene") else pd.Series([""] * len(df), index=df.index)
    d = pd.DataFrame({"g": g, "x": num(df, c["x"]), "y": num(df, c["y"])}).dropna(subset=["x", "y"])
    tx, ty = float(o["tx"]), float(o["ty"])
    q = {"Hyper-up": (d.x > tx) & (d.y > ty), "Hyper-down": (d.x > tx) & (d.y < -ty),
         "Hypo-up": (d.x < -tx) & (d.y > ty), "Hypo-down": (d.x < -tx) & (d.y < -ty)}
    other = ~(q["Hyper-up"] | q["Hyper-down"] | q["Hypo-up"] | q["Hypo-down"])
    cols = st.colors(4)
    ax = st.ax(fig.add_subplot())
    ax.scatter(d.loc[other, "x"], d.loc[other, "y"], s=o["psize"] * 0.8, c="#c7c7c7", lw=0, alpha=0.7,
               rasterized=True, label=f"Other ({int(other.sum())})")
    for (k, m), col in zip(q.items(), cols):
        ax.scatter(d.loc[m, "x"], d.loc[m, "y"], s=o["psize"], c=[col], lw=0, alpha=0.85, rasterized=True,
                   label=f"{k} ({int(m.sum())})")
    mx = float(d["x"].abs().max() or 1) * 1.1
    my = float(d["y"].abs().max() or 1) * 1.1
    ax.set_xlim(-mx, mx)
    ax.set_ylim(-my, my)
    for v in (-tx, tx):
        ax.axvline(v, color="#777777", ls="--", lw=0.8)
    for v in (-ty, ty):
        ax.axhline(v, color="#777777", ls="--", lw=0.8)
    for (k, m), col, (hx, hy) in zip(q.items(), cols, [(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        ax.text(hx * mx * 0.97, hy * my * 0.97, f"{k}\n{int(m.sum())}", ha="right" if hx > 0 else "left",
                va="top" if hy > 0 else "bottom", color=col, fontsize=st.font * 0.85, fontweight="bold")
    lab = d[~other].assign(s=lambda z: (z.x * z.y).abs()).nlargest(int(o["top"]), "s")
    if c.get("gene"):
        _repel(ax, lab["x"], lab["y"], lab["g"], st)
    st.labels(ax, "m6A log2 fold change", "Expression log2 fold change")
    st.legend(ax, markerscale=1.5, ncol=3, **_leg_out(st, "below"))


def _ex_metaplot():
    r = rng(13)
    rows = []
    for grp, shift in (("Control", 0.0), ("METTL3-KO", 1.0)):
        n = 900
        reg = r.choice(["5'UTR", "CDS", "3'UTR"], n, p=[0.10, 0.45, 0.45])
        pos = np.empty(n)
        for i, rg in enumerate(reg):
            if rg == "5'UTR":
                pos[i] = r.beta(2, 2)
            elif rg == "CDS":
                pos[i] = r.beta(3.5 - shift * 1.3, 1.3) if r.random() < 0.7 else r.uniform()
            else:
                pos[i] = r.beta(1.2, 3.2 + shift * 1.5) if r.random() < 0.75 else r.uniform()
        if shift:
            keep = r.random(n) > 0.25
            reg, pos = reg[keep], pos[keep]
        rows += [(grp, a, round(float(b), 4)) for a, b in zip(reg, pos)]
    return pd.DataFrame(rows, columns=["Group", "Region", "Relative position"])


@plot("m6a_metaplot_013", "m6A metagene plot (5'UTR | CDS | 3'UTR)", T, number="013", example=_ex_metaplot,
      roles=[Role("group", "Group / sample", "cat"), Role("region", "Region (5'UTR / CDS / 3'UTR)", "cat"),
             Role("pos", "Relative position within the region (0–1)", "num")],
      opts=[Opt("w5", "5'UTR width", "float", 1.0, lo=0.1, hi=10, step=0.1),
            Opt("wc", "CDS width", "float", 2.0, lo=0.1, hi=10, step=0.1),
            Opt("w3", "3'UTR width", "float", 2.0, lo=0.1, hi=10, step=0.1),
            Opt("bw", "Smoothing (KDE bandwidth)", "float", 0.03, lo=0.005, hi=1, step=0.005),
            Opt("fill", "Fill under curves", "bool", True)],
      desc="One row per peak: group/sample, transcript region (5'UTR, CDS, 3'UTR) and the relative position "
           "of the peak inside that region (0 = start, 1 = end).", size=(8, 5))
def m6a_metaplot_013(fig, df, c, o, st):
    from scipy.stats import gaussian_kde
    d = pd.DataFrame({"g": as_str(df, c["group"]), "r": as_str(df, c["region"]).str.upper(),
                      "p": num(df, c["pos"])}).dropna()
    d["p"] = d["p"].clip(0, 1)
    w = [float(o["w5"]), float(o["wc"]), float(o["w3"])]
    off = [0, w[0], w[0] + w[1]]

    def seg(s):
        s = s.replace(" ", "")
        if "5" in s:
            return 0
        if "3" in s:
            return 2
        return 1
    k = d["r"].map(seg)
    d["m"] = [off[i] + p * w[i] for i, p in zip(k, d["p"])]
    total = sum(w)
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 1.1])
    ax = st.ax(fig.add_subplot(gs[0]), "y")
    bx = fig.add_subplot(gs[1], sharex=ax)
    grid = np.linspace(0, total, 400)
    groups = cats(d, "g")
    for i, g in enumerate(groups):
        v = d.loc[d["g"] == g, "m"].to_numpy()
        col = st.color(i)
        if len(v) < 2 or np.ptp(v) == 0:
            continue
        kde = gaussian_kde(v, bw_method=float(o["bw"]) * total / max(np.std(v), 1e-9))
        y = kde(grid)
        ax.plot(grid, y, color=col, lw=2, label=f"{g} (n={len(v)})")
        if o["fill"]:
            ax.fill_between(grid, 0, y, color=col, alpha=0.15)
    for b in off[1:]:
        ax.axvline(b, color="#888888", ls="--", lw=0.8)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, total)
    ax.tick_params(labelbottom=False)
    bx.set_axis_off()
    names = ["5'UTR", "CDS", "3'UTR"]
    for i in range(3):
        h = 0.8 if i == 1 else 0.4
        bx.add_patch(Rectangle((off[i], 0.5 - h / 2), w[i], h, fc="#4d4d4d" if i == 1 else "#a6a6a6", ec="white"))
        bx.text(off[i] + w[i] / 2, 0.5, names[i], ha="center", va="center", color="white",
                fontsize=st.font * 0.8, fontweight="bold")
    bx.set_ylim(0, 1)
    st.labels(ax, "", "Peak density")
    if st.opts.get("xlabel"):
        bx.set_title(st.opts["xlabel"], y=-0.9, fontsize=st.font)
    st.legend(ax, loc="upper left")


def _ex_pathway_map():
    edges = [("EGF", "EGFR"), ("EGFR", "GRB2"), ("GRB2", "SOS1"), ("SOS1", "HRAS"), ("HRAS", "RAF1"),
             ("RAF1", "MAP2K1"), ("MAP2K1", "MAPK1"), ("MAPK1", "ELK1"), ("ELK1", "FOS"), ("MAPK1", "MYC"),
             ("EGFR", "PIK3CA"), ("PIK3CA", "AKT1"), ("AKT1", "MTOR"), ("AKT1", "BAD"), ("MTOR", "RPS6KB1"),
             ("EGFR", "PLCG1"), ("PLCG1", "PRKCA"), ("PRKCA", "RAF1"), ("AKT1", "FOXO3"), ("HRAS", "PIK3CA")]
    genes = sorted({g for e in edges for g in e})
    r = rng(92)
    fc = np.round(r.normal(0, 1.4, len(genes)), 2)
    fc[genes.index("BAD")] = np.nan
    n = max(len(edges), len(genes))
    pad = lambda v: list(v) + [np.nan] * (n - len(v))  # noqa: E731
    return pd.DataFrame({"From": pad([e[0] for e in edges]), "To": pad([e[1] for e in edges]),
                         "Gene": pad(genes), "log2FC": pad(fc)})


@plot("pathway_map_color_092", "Pathway map coloured by log2FC", T, number="092", example=_ex_pathway_map,
      roles=[Role("src", "Edge: from", "cat"), Role("dst", "Edge: to", "cat"),
             Role("gene", "Gene (for the colour)", "cat", optional=True),
             Role("fc", "Gene log2FC", "num", optional=True)],
      opts=[Opt("layout", "Layout", "choice", "layered (left → right)",
                choices=["layered (left → right)", "layered (top → bottom)", "spring"]),
            Opt("box_w", "Box width", "float", 0.78, lo=0.2, hi=2, step=0.05),
            Opt("fs", "Gene label size (× font)", "float", 0.75, lo=0.3, hi=2, step=0.05)],
      desc="Draws the user's own pathway: edge columns (from → to) plus a gene / log2FC list for the box "
           "colours; KEGG maps themselves are not downloaded.", size=(10, 6), axis_labels=False)
def pathway_map_color_092(fig, df, c, o, st):
    import networkx as nx
    e = pd.DataFrame({"a": df[c["src"]], "b": df[c["dst"]]}).dropna().astype(str)
    Gr = nx.DiGraph()
    Gr.add_edges_from(zip(e["a"].str.strip(), e["b"].str.strip()))
    fc = {}
    if c.get("gene") and c.get("fc"):
        gg = pd.DataFrame({"g": df[c["gene"]], "v": num(df, c["fc"])}).dropna()
        fc = dict(zip(gg["g"].astype(str).str.strip(), gg["v"]))
    ax = fig.add_subplot()
    st.bare(ax)
    if not len(Gr):
        st.labels(ax, "", "", "")
        return
    vertical = "top" in o["layout"]
    if o["layout"].startswith("layered"):
        try:
            gens = [list(g) for g in nx.topological_generations(Gr)]
        except nx.NetworkXUnfeasible:
            src = [n for n in Gr if Gr.in_degree(n) == 0] or [next(iter(Gr))]
            dist = {}
            for s in src:
                for n_, dd in nx.single_source_shortest_path_length(Gr, s).items():
                    dist[n_] = min(dist.get(n_, 1e9), dd)
            for n_ in Gr:
                dist.setdefault(n_, 0)
            gens = [[n_ for n_ in Gr if dist[n_] == k] for k in range(max(dist.values()) + 1)]
        pos, prev = {}, {}
        for li, layer in enumerate(gens):
            if li:
                layer.sort(key=lambda n_: np.mean([prev[p] for p in Gr.predecessors(n_) if p in prev] or [0]))
            m = len(layer)
            for k, n_ in enumerate(layer):
                yy = (m - 1) / 2 - k
                prev[n_] = -yy
                pos[n_] = (li, yy) if not vertical else (yy * 1.25, -li)
        bw, bh = float(o["box_w"]), 0.42
        if vertical:
            bw, bh = bw * 1.2, 0.36
    else:
        pos = {k: tuple(v * 4) for k, v in nx.spring_layout(Gr, seed=3, k=1.2).items()}
        bw, bh = float(o["box_w"]), 0.35
    lim = max([abs(v) for v in fc.values() if np.isfinite(v)] or [1.0])
    cm, norm = _cmap(st), Normalize(-lim, lim)
    patches = {}
    for n_, (x, y) in pos.items():
        v = fc.get(n_, np.nan)
        fcol = cm(norm(v)) if np.isfinite(v) else "#ffffff"
        p = FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.06",
                           fc=fcol, ec="#333333", lw=0.9, zorder=3)
        ax.add_patch(p)
        patches[n_] = p
        lum = np.dot(to_rgb(fcol), [0.299, 0.587, 0.114])
        ax.text(x, y, n_, ha="center", va="center", fontsize=st.font * o["fs"], zorder=4,
                color="white" if lum < 0.5 else "black", fontweight="bold")
    for a, b in Gr.edges():
        (xa, ya), (xb, yb) = pos[a], pos[b]
        span = abs(xb - xa) if not vertical else abs(yb - ya)
        rad = 0.25 if (o["layout"].startswith("layered") and span > 1.5) else 0.0
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], patchA=patches[a], patchB=patches[b], arrowstyle="-|>",
                                     mutation_scale=12, color="#444444", lw=1.0,
                                     connectionstyle=f"arc3,rad={rad}", zorder=2))
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ym, yr = (min(ys) + max(ys)) / 2, max((max(ys) - min(ys)) / 2 + 0.6, 1.6)
    ax.set_ylim(ym - yr, ym + yr)
    xm, xr = (min(xs) + max(xs)) / 2, max((max(xs) - min(xs)) / 2 + bw, 2.5)
    ax.set_xlim(xm - xr, xm + xr)
    _cbar(fig, st, cm, norm, "log2FC", ax=ax, shrink=0.5)
    if any(not np.isfinite(fc.get(n_, np.nan)) for n_ in Gr):
        st.legend(ax, [Patch(fc="white", ec="#333333", label="not measured")], loc="lower right")
    st.labels(ax, "", "", "")


def _ex_miranda():
    r = rng(120)
    mirs = ["hsa-miR-21-5p", "hsa-miR-155-5p", "hsa-let-7a-5p", "hsa-miR-34a-5p", "hsa-miR-145-5p",
            "hsa-miR-200c-3p", "hsa-miR-17-5p", "hsa-miR-126-3p"]
    targets = ["PTEN", "PDCD4", "KRAS", "HMGA2", "SIRT1", "BCL2", "E2F3", "MYC", "ZEB1", "ZEB2", "SOCS1",
               "CDKN1A", "VEGFA", "SPRED1", "TP53INP1"]
    rows = set()
    while len(rows) < 42:
        rows.add((mirs[int(r.integers(len(mirs)))], targets[int(r.integers(len(targets)))]))
    rows = sorted(rows)
    sc = np.round(r.uniform(140, 192, len(rows)), 1)
    en = np.round(-12 - (sc - 140) * 0.35 - r.uniform(0, 8, len(rows)), 2)
    return pd.DataFrame({"miRNA": [a for a, _ in rows], "Target": [b for _, b in rows], "Score": sc,
                         "Energy": en})


@plot("miranda_targets_120", "miRanda miRNA–target predictions", T, number="120", example=_ex_miranda,
      roles=[Role("mir", "miRNA", "cat"), Role("target", "Target gene", "cat"), Role("score", "Score", "num"),
             Role("energy", "Energy (kcal/mol)", "num")],
      opts=[Opt("view", "View", "choice", "bubble matrix", choices=["bubble matrix", "network"]),
            Opt("top", "Keep top N pairs (by score)", "int", 60, lo=1, hi=2000),
            Opt("smax", "Largest bubble size", "float", 260, lo=20, hi=1500)],
      desc="Plots a miRanda RESULT table (miRNA, target, score, energy), one row per predicted pair; the "
           "prediction itself is not run.", size=(8.5, 7))
def miranda_targets_120(fig, df, c, o, st):
    d = pd.DataFrame({"m": as_str(df, c["mir"]), "t": as_str(df, c["target"]), "s": num(df, c["score"]),
                      "e": num(df, c["energy"])}).dropna()
    d = d.sort_values("s", ascending=False).drop_duplicates(["m", "t"]).head(int(o["top"]))
    cm = _cmap(st, reverse=True)
    norm = Normalize(d["e"].min(), d["e"].max()) if d["e"].max() > d["e"].min() else Normalize(-1, 1)
    gs, cax, lax = _side(fig)
    if o["view"] == "network":
        import networkx as nx
        Gr = nx.Graph()
        Gr.add_edges_from(zip("m:" + d["m"], "t:" + d["t"]))
        pos = nx.kamada_kawai_layout(Gr) if len(Gr) < 300 else nx.spring_layout(Gr, seed=1)
        ax = fig.add_subplot(gs[:, 0])
        st.bare(ax)
        w = _sizes(d["s"], 0.6, 3.5)
        for (_, r), lw in zip(d.iterrows(), w):
            (x0, y0), (x1, y1) = pos["m:" + r["m"]], pos["t:" + r["t"]]
            ax.plot([x0, x1], [y0, y1], color=cm(norm(r["e"])), lw=lw, alpha=0.85, zorder=1)
        mi = [n for n in Gr if n.startswith("m:")]
        ti = [n for n in Gr if n.startswith("t:")]
        ax.scatter([pos[n][0] for n in mi], [pos[n][1] for n in mi], s=260, marker="D", c=st.color(0),
                   edgecolor="white", zorder=3, label="miRNA")
        ax.scatter([pos[n][0] for n in ti], [pos[n][1] for n in ti], s=130, c=st.color(1), edgecolor="white",
                   zorder=3, label="target")
        for n in Gr:
            x, y = pos[n]
            ax.text(x, y - 0.055, n[2:].replace("hsa-", ""), ha="center", va="top", fontsize=st.font * 0.68,
                    zorder=4)
        ax.margins(0.08)
        hs = [Line2D([], [], color="#888888", lw=w_, label=f"{v:.0f}")
              for v, w_ in zip(*(lambda vv: (vv, _sizes(vv, 0.6, 3.5, d["s"].min(), d["s"].max())))(
                  np.linspace(d["s"].min(), d["s"].max(), 3)))]
        hh, ll = ax.get_legend_handles_labels()
        _legend_sections(st, lax, [("Node", hh), ("Score", hs)], loc="center left")
        st.labels(ax, "", "", "")
    else:
        mirs = cats(d, "m")
        tg = d.groupby("t")["s"].max().sort_values().index.tolist()
        ax = st.ax(fig.add_subplot(gs[:, 0]))
        ax.scatter(d["m"].map({m: i for i, m in enumerate(mirs)}), d["t"].map({t: i for i, t in enumerate(tg)}),
                   s=_sizes(d["s"], 20, o["smax"]), c=d["e"], cmap=cm, norm=norm, edgecolor="#333333", lw=0.5,
                   zorder=3)
        ax.set_xticks(range(len(mirs)), mirs, rotation=40, ha="right", fontsize=st.font * 0.8)
        ax.set_yticks(range(len(tg)), tg, fontsize=st.font * 0.8)
        ax.set_xlim(-0.6, len(mirs) - 0.4)
        ax.set_ylim(-0.6, len(tg) - 0.4)
        st.legend(lax, _size_handles(d["s"], 20, o["smax"], n=4), title="Score", loc="center left",
                  labelspacing=1.1)
        st.labels(ax, "", "")
    _cbar(fig, st, cm, norm, "Energy (kcal/mol)", cax=cax)


# ═══════════════════════════════════════════════════════════════════════════════
# Sets & flows
# ═══════════════════════════════════════════════════════════════════════════════

_SETFMT = Opt("fmt", "Input layout", "choice", "one column per set",
              choices=["one column per set", "long (item, set)"])


def _read_sets(df, c, o, key="sets", maxn=None):
    """{set name: set of items} from one-column-per-set (blanks allowed) or long (item, set) tables."""
    cols = c.get(key) or []
    cols = [x for x in (cols if isinstance(cols, list) else [cols]) if x]
    out = {}
    if str(o.get("fmt", "")).startswith("long") and len(cols) >= 2:
        d = df[[cols[0], cols[1]]].dropna()
        for it, s in zip(d[cols[0]].astype(str).str.strip(), as_str(d, cols[1])):
            if it:
                out.setdefault(s, set()).add(it)
    else:
        for col in cols:
            v = df[col].dropna().astype(str).str.strip()
            out[str(col)] = set(v[v != ""])
    out = {k: v for k, v in out.items() if v}
    if maxn:
        out = dict(list(out.items())[:maxn])
    return out


def _exclusive_counts(sets):
    """{bitmask: number of items in exactly those sets} (bit i = i-th set)."""
    names = list(sets)
    member = {}
    for i, n in enumerate(names):
        for it in sets[n]:
            member[it] = member.get(it, 0) | (1 << i)
    return Counter(member.values())


def _pad_cols(d: dict):
    n = max(len(v) for v in d.values())
    return pd.DataFrame({k: list(v) + [np.nan] * (n - len(v)) for k, v in d.items()})


def _ex_sets(k, seed, sizes=None, names=None):
    r = rng(seed)
    universe = np.array(_SYMS + [f"GENE{i:04d}" for i in range(1500)], dtype=object)
    w = r.gamma(0.6, 1.0, len(universe))
    w /= w.sum()
    names = names or [f"Set {chr(65 + i)}" for i in range(k)]
    sizes = sizes or list(r.integers(150, 420, k))
    return _pad_cols({n: list(r.choice(universe, int(s), replace=False, p=w)) for n, s in zip(names, sizes)})


def _ex_venn():
    return _ex_sets(4, 103, [320, 260, 380, 210], ["DrugA_up", "DrugB_up", "DrugC_up", "DrugD_up"])


# shapes as (cx, cy, width, height, angle°) in a common frame
def _venn_shapes(n):
    if n == 1:
        return [(0, 0, 2, 2, 0)]
    if n == 2:
        return [(-0.55, 0, 2, 2, 0), (0.55, 0, 2, 2, 0)]
    if n == 3:
        return [(0.62 * np.cos(np.radians(a)), 0.62 * np.sin(np.radians(a)), 2, 2, 0) for a in (150, 30, 270)]
    if n == 4:
        raw = [(0.350, 0.400, 140.0), (0.450, 0.500, 140.0), (0.544, 0.500, 40.0), (0.644, 0.400, 40.0)]
        return [((x - 0.497) * 4, (y - 0.45) * 4, 0.72 * 4, 0.45 * 4, a) for x, y, a in raw]
    raw = [(0.428, 0.449, 155.0), (0.469, 0.543, 82.0), (0.558, 0.523, 10.0), (0.578, 0.432, 118.0),
           (0.489, 0.383, 46.0)]
    return [((x - 0.504) * 4, (y - 0.466) * 4, 0.87 * 4, 0.50 * 4, a) for x, y, a in raw]


def _inside(shape, X, Y):
    cx, cy, w, h, a = shape
    t = np.radians(a)
    dx, dy = X - cx, Y - cy
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    return (u / (w / 2)) ** 2 + (v / (h / 2)) ** 2 <= 1


def _region_points(shapes, res=320):
    """Label point (deepest inside) of every region bitmask of the overlapping shapes."""
    from scipy.ndimage import distance_transform_edt
    ext = max(max(abs(s[0]), abs(s[1])) + max(s[2], s[3]) / 2 for s in shapes) * 1.05
    xs = np.linspace(-ext, ext, res)
    X, Y = np.meshgrid(xs, xs)
    code = np.zeros(X.shape, int)
    for i, s in enumerate(shapes):
        code |= _inside(s, X, Y).astype(int) << i
    out = {}
    for m in np.unique(code):
        if m == 0:
            continue
        dist = distance_transform_edt(code == m)
        k = np.unravel_index(np.argmax(dist), dist.shape)
        out[int(m)] = (X[k], Y[k], dist[k] * (xs[1] - xs[0]))
    return out, (X, Y, code)


def _set_label_pos(shapes, i, pts):
    """A point just outside the union, in the direction of set i's exclusive region."""
    p = pts.get(1 << i)
    cx, cy = (p[0], p[1]) if p else shapes[i][:2]
    d = np.array([cx, cy], float)
    nrm = np.hypot(*d)
    d = d / nrm if nrm > 1e-9 else np.array([0.0, 1.0])
    pt = np.array([cx, cy], float)
    for _ in range(400):
        if not any(_inside(s, pt[0], pt[1]) for s in shapes):
            break
        pt = pt + d * 0.02
    return pt + d * 0.1, d


def _draw_venn(ax, st, names, counts, shapes, sizes, pct=False, alpha=0.3, fs_scale=1.0):
    n = len(names)
    cols = st.colors(n)
    total = sum(counts.values()) or 1
    for s, col in zip(shapes, cols):
        ax.add_patch(Ellipse(s[:2], s[2], s[3], angle=s[4], fc=col, alpha=alpha, ec="none"))
        ax.add_patch(Ellipse(s[:2], s[2], s[3], angle=s[4], fc="none", ec=col, lw=1.6))
    pts, _ = _region_points(shapes)
    fs = st.font * fs_scale * {1: 1.0, 2: 1.0, 3: 0.95, 4: 0.85, 5: 0.72}.get(n, 0.7)
    for m, (x, y, _) in pts.items():
        k = counts.get(m, 0)
        s = f"{k}" + (f"\n{100 * k / total:.1f}%" if pct else "")
        ax.text(x, y, s, ha="center", va="center", fontsize=fs, color="#222222", linespacing=0.95)
    ext = 0
    for i, (nm, col) in enumerate(zip(names, cols)):
        (x, y), d = _set_label_pos(shapes, i, pts)
        ha = "left" if d[0] > 0.3 else "right" if d[0] < -0.3 else "center"
        va = "bottom" if d[1] > 0.3 else "top" if d[1] < -0.3 else "center"
        ax.text(x, y, f"{nm}\n(n={sizes[i]})", ha=ha, va=va, fontsize=st.font * 0.9, color=col,
                fontweight="bold", linespacing=1.0)
        ext = max(ext, abs(x), abs(y))
    lim = max(ext, max(max(abs(s[0]), abs(s[1])) + max(s[2], s[3]) / 2 for s in shapes)) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    st.bare(ax)


@plot("venn_103", "Venn diagram (2–5 sets)", F, number="103", example=_ex_venn,
      roles=[Role("sets", "Sets (one column per set, or item + set columns)", "any", multi=True)],
      opts=[_SETFMT, Opt("pct", "Show percentages", "bool", False),
            Opt("alpha", "Fill opacity", "float", 0.3, lo=0, hi=1, step=0.05),
            Opt("legend_sets", "Also list the sets in a legend", "bool", False)],
      desc="One column per set listing its items (gene names…; columns may have different lengths), or a long "
           "table of item + set; 2 to 5 sets.", size=(7, 6.5), axis_labels=False)
def venn_103(fig, df, c, o, st):
    sets = _read_sets(df, c, o, maxn=5)
    ax = fig.add_subplot()
    if not sets:
        st.bare(ax)
        return
    names = list(sets)
    _draw_venn(ax, st, names, _exclusive_counts(sets), _venn_shapes(len(names)),
               [len(sets[k]) for k in names], o["pct"], o["alpha"])
    if o["legend_sets"]:
        st.legend(ax, [Patch(fc=col, alpha=0.6, label=n) for n, col in zip(names, st.colors(len(names)))])
    st.labels(ax, "", "", "")


def _circle_overlap(r1, r2, d):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a = r1 ** 2 * np.arccos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    b = r2 ** 2 * np.arccos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    c_ = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a + b - c_


def _solve_d(r1, r2, area):
    lo, hi = abs(r1 - r2), r1 + r2
    if area <= 0:
        return hi * 1.04
    if area >= np.pi * min(r1, r2) ** 2:
        return lo * 0.98
    for _ in range(60):
        m = (lo + hi) / 2
        if _circle_overlap(r1, r2, m) > area:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


@plot("proportional_venn_028", "Area-proportional Venn (2–3 sets)", F, number="028",
      example=lambda: _ex_sets(3, 28, [520, 300, 140], ["RNA-seq DEGs", "Proteomics", "ChIP-seq targets"]),
      roles=[Role("sets", "Sets (one column per set, or item + set columns)", "any", multi=True)],
      opts=[_SETFMT, Opt("pct", "Show percentages", "bool", False),
            Opt("alpha", "Fill opacity", "float", 0.35, lo=0, hi=1, step=0.05)],
      desc="One column per set listing its items (or item + set columns); circle areas and pairwise overlaps "
           "are proportional to the counts (2–3 sets).", size=(7, 6.5), axis_labels=False)
def proportional_venn_028(fig, df, c, o, st):
    sets = _read_sets(df, c, o, maxn=3)
    ax = fig.add_subplot()
    if len(sets) < 2:
        st.bare(ax)
        return
    names = list(sets)
    S = [sets[k] for k in names]
    tot = max(len(s) for s in S)
    r = [np.sqrt(len(s) / tot) for s in S]
    unit = np.pi / tot     # area per item

    def dist(i, j):
        return _solve_d(r[i], r[j], len(S[i] & S[j]) * unit)
    if len(S) == 2:
        dd = dist(0, 1)
        cen = [(-dd / 2, 0), (dd / 2, 0)]
    else:
        dab, dac, dbc = dist(0, 1), dist(0, 2), dist(1, 2)
        cosA = np.clip((dab ** 2 + dac ** 2 - dbc ** 2) / (2 * dab * dac + 1e-12), -1, 1)
        A, B = np.array([0.0, 0.0]), np.array([dab, 0.0])
        C = np.array([dac * cosA, -dac * np.sqrt(1 - cosA ** 2)])
        cent = (A * r[0] ** 2 + B * r[1] ** 2 + C * r[2] ** 2) / sum(x ** 2 for x in r)
        cen = [tuple(p - cent) for p in (A, B, C)]
    shapes = [(x, y, 2 * rr, 2 * rr, 0) for (x, y), rr in zip(cen, r)]
    _draw_venn(ax, st, names, _exclusive_counts(sets), shapes, [len(s) for s in S], o["pct"], o["alpha"])
    st.labels(ax, "", "", "")


def _ex_peak_sets():
    r = rng(26)
    chroms = _EX_CHROMS[:8]
    base = _ex_peaks(26, 700, chroms)
    rows = []
    for si, s in enumerate(["H3K27ac", "H3K4me1", "BRD4"]):
        keep = r.random(len(base)) < [0.62, 0.5, 0.35][si]
        b = base[keep]
        st_ = b["Position"].to_numpy() + r.integers(-300, 300, keep.sum())
        ln = r.integers(400, 2500, keep.sum())
        rows += [(s, ch, int(a), int(a + w)) for ch, a, w in zip(b["Chrom"], st_, ln)]
        extra = _ex_peaks(100 + si, 120, chroms)
        rows += [(s, ch, int(p), int(p + r.integers(400, 2000))) for ch, p in zip(extra["Chrom"], extra["Position"])]
    return pd.DataFrame(rows, columns=["Sample", "Chrom", "Start", "End"])


@plot("genomic_peak_venn_026", "Genomic peak overlap Venn", F, number="026", example=_ex_peak_sets,
      roles=[Role("set", "Peak set / sample", "cat"), Role("chrom", "Chromosome", "any"),
             Role("start", "Start", "num"), Role("end", "End", "num")],
      opts=[Opt("gap", "Merge peaks closer than (bp)", "int", 0, lo=0, hi=1_000_000),
            Opt("pct", "Show percentages", "bool", False),
            Opt("alpha", "Fill opacity", "float", 0.3, lo=0, hi=1, step=0.05)],
      desc="Long table of genomic regions: set / sample, chromosome, start, end. Overlapping regions of all sets "
           "are merged; the Venn counts merged regions by which sets contribute to them (2–5 sets).",
      size=(7, 6.5), axis_labels=False)
def genomic_peak_venn_026(fig, df, c, o, st):
    d = pd.DataFrame({"s": as_str(df, c["set"]), "c": as_str(df, c["chrom"]), "a": num(df, c["start"]),
                      "b": num(df, c["end"])}).dropna()
    d["a"], d["b"] = np.fmin(d["a"], d["b"]), np.fmax(d["a"], d["b"])
    names = cats(d, "s")[:5]
    d = d[d["s"].isin(names)]
    bit = {n: 1 << i for i, n in enumerate(names)}
    counts = Counter()
    gap = float(o["gap"])
    for _, sub in d.groupby("c"):
        sub = sub.sort_values("a")
        end, mask = -np.inf, 0
        for a, b, s in zip(sub["a"], sub["b"], sub["s"]):
            if a <= end + gap:
                end = max(end, b)
                mask |= bit[s]
            else:
                if mask:
                    counts[mask] += 1
                end, mask = b, bit[s]
        if mask:
            counts[mask] += 1
    ax = fig.add_subplot()
    if not names:
        st.bare(ax)
        return
    sizes = [int((d["s"] == n).sum()) for n in names]
    _draw_venn(ax, st, names, counts, _venn_shapes(len(names)), sizes, o["pct"], o["alpha"])
    ax.text(0.5, 0.0, f"{sum(counts.values())} merged regions; set labels show the number of input peaks",
            transform=ax.transAxes, ha="center", va="top", fontsize=st.font * 0.75, color="#555555")
    st.labels(ax, "", "", "")


@plot("upset_009", "UpSet plot", F, number="009",
      example=lambda: _ex_sets(5, 9, [420, 310, 260, 380, 190], ["Liver", "Kidney", "Heart", "Brain", "Lung"]),
      roles=[Role("sets", "Sets (one column per set, or item + set columns)", "any", multi=True)],
      opts=[_SETFMT, Opt("top", "Max intersections", "int", 20, lo=1, hi=200),
            Opt("min", "Min intersection size", "int", 1, lo=0, hi=1e9),
            Opt("sort", "Sort intersections by", "choice", "size", choices=["size", "degree", "sets"]),
            Opt("vals", "Show counts", "bool", True), Opt("empty", "Show empty intersections", "bool", False)],
      desc="One column per set listing its items (or item + set columns); bars = items in exactly that "
           "combination of sets.", size=(9, 5.8), axis_labels=False)
def upset_009(fig, df, c, o, st):
    sets = _read_sets(df, c, o)
    names = list(sets)
    k = len(names)
    cnt = _exclusive_counts(sets)
    if o["empty"] and k <= 10:
        for m in range(1, 2 ** k):
            cnt.setdefault(m, 0)
    items = [(m, v) for m, v in cnt.items() if v >= o["min"] or o["empty"]]
    if o["sort"] == "size":
        items.sort(key=lambda z: -z[1])
    elif o["sort"] == "degree":
        items.sort(key=lambda z: (bin(z[0]).count("1"), -z[1]))
    else:
        items.sort(key=lambda z: (bin(z[0]).count("1"), [-(z[0] >> i & 1) for i in range(k)]))
    items = items[:int(o["top"])]
    order = sorted(range(k), key=lambda i: len(sets[names[i]]))   # largest set at the top
    row = {i: j for j, i in enumerate(order)}
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 3.2], height_ratios=[1.6, 1])
    at = st.ax(fig.add_subplot(gs[0, 1]), "y")
    am = fig.add_subplot(gs[1, 1], sharex=at)
    al = st.ax(fig.add_subplot(gs[1, 0], sharey=am), "x")
    x = np.arange(len(items))
    col = st.color(0)
    bars = at.bar(x, [v for _, v in items], color="#3d3d3d", width=0.62)
    if o["vals"]:
        at.bar_label(bars, padding=2, fontsize=st.font * 0.7)
    at.set_ylabel(st.opts.get("ylabel") or "Intersection size", fontsize=st.font)
    at.tick_params(labelbottom=False)
    at.spines["bottom"].set_visible(False)
    at.tick_params(axis="x", length=0)
    for j in range(k):
        if j % 2 == 0:
            am.axhspan(j - 0.5, j + 0.5, color="#f2f2f2", zorder=0)
    for xi, (m, _) in enumerate(items):
        on = [row[i] for i in range(k) if m >> i & 1]
        am.scatter([xi] * k, range(k), s=70, c="#d9d9d9", zorder=2, lw=0)
        am.scatter([xi] * len(on), on, s=70, c=col, zorder=3, lw=0)
        if len(on) > 1:
            am.plot([xi, xi], [min(on), max(on)], color=col, lw=2.2, zorder=2)
    am.set_yticks(range(k), [names[i] for i in order], fontsize=st.font * 0.85)
    am.tick_params(axis="y", length=0)
    am.set_xlim(-0.6, len(items) - 0.4)
    am.set_ylim(-0.5, k - 0.5)
    for s in am.spines.values():
        s.set_visible(False)
    am.tick_params(axis="x", length=0, labelbottom=False)
    sz = [len(sets[names[i]]) for i in order]
    sb = al.barh(range(k), sz, color=[st.color(i) for i in order], height=0.6)
    if o["vals"]:
        al.bar_label(sb, padding=2, fontsize=st.font * 0.7)
    al.invert_xaxis()
    al.tick_params(labelleft=False, left=False)
    al.spines["left"].set_visible(False)
    al.set_xlabel(st.opts.get("xlabel") or "Set size", fontsize=st.font)
    al.margins(x=0.25)
    st.suptitle(fig)


def _ex_mosaic():
    r = rng(8)
    n = 360
    sub = r.choice(["Luminal A", "Luminal B", "HER2+", "Basal"], n, p=[0.4, 0.25, 0.15, 0.2])
    stage, resp = [], []
    for s in sub:
        pst = {"Luminal A": [0.45, 0.35, 0.15, 0.05], "Luminal B": [0.3, 0.4, 0.2, 0.1],
               "HER2+": [0.2, 0.35, 0.3, 0.15], "Basal": [0.15, 0.3, 0.35, 0.2]}[s]
        stage.append(r.choice(["I", "II", "III", "IV"], p=pst))
        pr = {"Luminal A": 0.7, "Luminal B": 0.55, "HER2+": 0.6, "Basal": 0.35}[s]
        resp.append("Responder" if r.random() < pr else "Non-responder")
    return pd.DataFrame({"Subtype": sub, "Stage": stage, "Response": resp})


@plot("mosaic_two_three_008", "Two / three category mosaic", F, number="008", example=_ex_mosaic,
      roles=[Role("a", "Category 1 (columns)", "cat"), Role("b", "Category 2 (stacked, coloured)", "cat"),
             Role("c", "Category 3 (panels, optional)", "cat", optional=True)],
      opts=[Opt("vals", "Show counts in tiles", "choice", "count", choices=["count", "% of column", "none"]),
            Opt("chi", "Chi-square test in title", "bool", True),
            Opt("gap", "Gap between tiles", "float", 0.012, lo=0, hi=0.1, step=0.002)],
      desc="One row per subject with 2 or 3 categorical columns; column widths = share of category 1, tile "
           "heights = share of category 2 within it, one panel per level of category 3.", size=(9, 5.5))
def mosaic_two_three_008(fig, df, c, o, st):
    from scipy.stats import chi2_contingency
    d = pd.DataFrame({"a": as_str(df, c["a"]), "b": as_str(df, c["b"])})
    d["c"] = as_str(df, c["c"]) if c.get("c") else "all"
    A, B, C = cats(d, "a"), cats(d, "b"), cats(d, "c")
    A = sorted(A) if all(len(x) < 4 for x in A) else A
    B = sorted(B) if all(len(x) < 4 for x in B) else B
    cols = dict(zip(B, st.colors(len(B))))
    axes = [st.ax(fig.add_subplot(1, len(C), i + 1), "") for i in range(len(C))]
    gap = float(o["gap"])
    for ax, lev in zip(axes, C):
        sub = d[d["c"] == lev]
        tab = pd.crosstab(sub["a"], sub["b"]).reindex(index=A, columns=B, fill_value=0)
        n = tab.to_numpy().sum()
        x = 0.0
        ticks = []
        widths = tab.sum(1) / max(n, 1)
        usable = 1 - gap * (len(A) - 1)
        for a in A:
            w = widths[a] * usable
            colsum = tab.loc[a].sum()
            y = 0.0
            usable_y = 1 - gap * (len(B) - 1)
            for b in B:
                h = (tab.loc[a, b] / colsum if colsum else 0) * usable_y
                if h > 0:
                    ax.add_patch(Rectangle((x, y), w, h, fc=cols[b], ec="white", lw=0.8))
                    lum = np.dot(to_rgb(cols[b]), [0.299, 0.587, 0.114])
                    if o["vals"] != "none" and w > 0.06 and h > 0.07:
                        s = str(tab.loc[a, b]) if o["vals"] == "count" else f"{100 * tab.loc[a, b] / colsum:.0f}%"
                        ax.text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=st.font * 0.75,
                                color="white" if lum < 0.55 else "black")
                y += h + gap
            if colsum:
                ticks.append((x + w / 2, f"{a}\n(n={colsum})"))
            x += w + (gap if colsum else 0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([t for t, _ in ticks], [s for _, s in ticks], fontsize=st.font * 0.75)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        ttl = lev if len(C) > 1 else ""
        if o["chi"]:
            t2 = tab.loc[tab.sum(1) > 0, tab.sum(0) > 0]
            if t2.shape[0] > 1 and t2.shape[1] > 1:
                p = chi2_contingency(t2)[1]
                ttl = (ttl + "\n" if ttl else "") + f"χ² p = {p:.2g}"
        if ttl:
            ax.set_title(ttl, fontsize=st.font * 0.9)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    st.legend(axes[-1], [Patch(fc=cols[b], label=b) for b in B], title=c["b"], **_leg_out(st))
    if len(C) > 1:
        st.suptitle(fig, "")
        fig.supxlabel(st.opts.get("xlabel") or c["a"], fontsize=st.font)
        axes[0].set_ylabel(st.opts.get("ylabel") or f"Share of {c['b']}", fontsize=st.font)
    else:
        st.labels(axes[0], c["a"], f"Share of {c['b']}")


def _ex_cerna():
    r = rng(17)
    circ = ["circHIPK3", "circPVT1", "circITCH", "CDR1as", "circSMARCA5"]
    mir = ["miR-124-3p", "miR-7-5p", "miR-145-5p", "miR-21-5p", "miR-17-5p", "miR-338-3p"]
    mrna = ["SP1", "EGFR", "KLF4", "PTEN", "E2F1", "STAT3", "SOX2", "ZEB1", "MYC", "IGF1R"]
    rows = set()
    for cn in circ:
        for m in r.choice(len(mir), 2, replace=False):
            for t in r.choice(len(mrna), int(r.integers(2, 4)), replace=False):
                rows.add((cn, mir[m], mrna[t]))
    return pd.DataFrame(sorted(rows), columns=["circRNA", "miRNA", "mRNA"])


@plot("alluvial_cerna_017", "Alluvial plot (ceRNA network)", F, number="017", example=_ex_cerna,
      roles=[Role("axes", "Axis columns (left → right)", "cat", multi=True),
             Role("w", "Weight (optional)", "num", optional=True)],
      opts=[Opt("alpha", "Ribbon opacity", "float", 0.5, lo=0.05, hi=1, step=0.05),
            Opt("gap", "Gap between boxes (fraction)", "float", 0.02, lo=0, hi=0.2, step=0.005),
            Opt("colour_by", "Colour ribbons by", "choice", "first axis", choices=["first axis", "last axis"])],
      desc="One row per path (e.g. circRNA, miRNA, mRNA); 2 or more categorical columns become the axes, an "
           "optional numeric column the ribbon width.", size=(9, 6.5), axis_labels=False)
def alluvial_cerna_017(fig, df, c, o, st):
    axcols = [a for a in (c["axes"] if isinstance(c["axes"], list) else [c["axes"]]) if a]
    ax = fig.add_subplot()
    st.bare(ax)
    if len(axcols) < 2:
        st.labels(ax, "", "", "")
        return
    d = pd.DataFrame({k: as_str(df, k) for k in axcols})
    d["_w"] = num(df, c["w"]).fillna(0) if c.get("w") else 1.0
    d = d[d["_w"] > 0]
    nax = len(axcols)
    key = axcols[0] if o["colour_by"] == "first axis" else axcols[-1]
    kcats = cats(d, key)
    cols = dict(zip(kcats, st.colors(len(kcats))))
    tot = d["_w"].sum()
    gap = float(o["gap"])
    bw = 0.09
    xs = np.linspace(0, 1, nax)
    nodes = []
    for j, a in enumerate(axcols):
        if j == 0:
            order = kcats if key == a else cats(d, a)
        else:
            ctr = {k: (v[0] + v[1]) / 2 for k, v in nodes[j - 1].items()}
            m = d.assign(_y=d[axcols[j - 1]].map(ctr) * d["_w"]).groupby(a)
            order = list((m["_y"].sum() / m["_w"].sum()).sort_values(ascending=False).index)
        wsum = d.groupby(a)["_w"].sum()
        scale = (1 - gap * (len(order) - 1)) / tot
        y, pos = 1.0, {}
        for k in order:
            h = wsum[k] * scale
            pos[k] = (y - h, y)
            y -= h + gap
        nodes.append(pos)
    scale = [(1 - gap * (len(n) - 1)) / tot for n in nodes]
    ok = {k: i for i, k in enumerate(kcats)}
    for j in range(nax - 1):
        a, b = axcols[j], axcols[j + 1]
        flows = d.groupby(list(dict.fromkeys([a, b, key])), sort=False)["_w"].sum().reset_index()
        oa = {k: i for i, k in enumerate(nodes[j])}
        ob = {k: i for i, k in enumerate(nodes[j + 1])}
        flows["_oa"], flows["_ob"], flows["_k"] = flows[a].map(oa), flows[b].map(ob), flows[key].map(ok)
        curA = {k: v[1] for k, v in nodes[j].items()}
        curB = {k: v[1] for k, v in nodes[j + 1].items()}
        segA = {}
        for _, r in flows.sort_values(["_oa", "_ob", "_k"]).iterrows():
            h = r["_w"] * scale[j]
            segA[(r[a], r[b], r[key])] = (curA[r[a]] - h, curA[r[a]])
            curA[r[a]] -= h
        for _, r in flows.sort_values(["_ob", "_oa", "_k"]).iterrows():
            h = r["_w"] * scale[j + 1]
            y0a, y0b = segA[(r[a], r[b], r[key])]
            ax.add_patch(PathPatch(_ribbon(xs[j] + bw / 2, y0b, y0a, xs[j + 1] - bw / 2, curB[r[b]],
                                           curB[r[b]] - h), fc=cols[r[key]], ec="none", alpha=o["alpha"]))
            curB[r[b]] -= h
    for j, a in enumerate(axcols):
        for k, (y0, y1) in nodes[j].items():
            fc = cols[k] if a == key else "#5a5a5a"
            ax.add_patch(Rectangle((xs[j] - bw / 2, y0), bw, y1 - y0, fc=fc, ec="white", lw=0.8))
            mid = j not in (0, nax - 1)
            txtx = xs[j] - bw / 2 - 0.008 if j == 0 else xs[j] + bw / 2 + 0.008 if j == nax - 1 else xs[j]
            ha = "right" if j == 0 else "left" if j == nax - 1 else "center"
            ax.text(txtx, (y0 + y1) / 2, k, ha=ha, va="center", fontsize=st.font * (0.7 if mid else 0.8),
                    color="white" if mid else "#222222", fontweight="bold" if mid else "normal", zorder=5)
        ax.text(xs[j], 1.03, a, ha="center", va="bottom", fontsize=st.font, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.01, 1.08)
    st.labels(ax, "", "", "")


def _ex_ring():
    r = rng(83)
    circ = ["circHIPK3", "circPVT1", "circITCH", "CDR1as", "circSMARCA5", "circFOXO3", "circCCDC66"]
    mir = ["miR-124-3p", "miR-7-5p", "miR-145-5p", "miR-21-5p", "miR-17-5p", "miR-338-3p", "miR-29b-3p",
           "miR-155-5p", "miR-200a-3p", "miR-34a-5p", "miR-26a-5p", "let-7b-5p"]
    rows = set()
    for cn in circ:
        for m in r.choice(len(mir), int(r.integers(2, 6)), replace=False):
            rows.add((cn, mir[m]))
    rows = sorted(rows)
    return pd.DataFrame({"circRNA": [a for a, _ in rows], "miRNA": [b for _, b in rows],
                         "Score": np.round(r.uniform(140, 190, len(rows)), 1)})


@plot("circrna_mirna_ring_083", "circRNA–miRNA ring network", F, number="083", example=_ex_ring,
      roles=[Role("a", "circRNA (source)", "cat"), Role("b", "miRNA (target)", "cat"),
             Role("w", "Edge weight / score (optional)", "num", optional=True)],
      opts=[Opt("node", "Node size", "float", 160, lo=10, hi=1000),
            Opt("curv", "Edge curvature (0 = straight)", "float", 0.75, lo=0, hi=1, step=0.05),
            Opt("alpha", "Edge opacity", "float", 0.6, lo=0.05, hi=1, step=0.05)],
      desc="Edge table, one row per interaction: circRNA, miRNA and an optional score; both node types are "
           "placed on a ring and joined by curved edges.", size=(7.5, 7), axis_labels=False)
def circrna_mirna_ring_083(fig, df, c, o, st):
    d = pd.DataFrame({"a": as_str(df, c["a"]), "b": as_str(df, c["b"])})
    d["w"] = num(df, c["w"]) if c.get("w") else 1.0
    d = d.drop_duplicates(["a", "b"])
    A = cats(d, "a")
    deg = d["b"].value_counts()
    B = sorted(cats(d, "b"), key=lambda m: (min(A.index(x) for x in d.loc[d["b"] == m, "a"]), -deg[m]))
    gapf = 0.06
    angA = np.pi / 2 + np.pi * (gapf + (1 - 2 * gapf) * (np.arange(len(A)) + 0.5) / max(len(A), 1))
    angB = np.pi / 2 - np.pi * (gapf + (1 - 2 * gapf) * (np.arange(len(B)) + 0.5) / max(len(B), 1))
    pos = {("a", k): a for k, a in zip(A, angA)}
    pos.update({("b", k): a for k, a in zip(B, angB)})
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    st.bare(ax)
    cols = dict(zip(A, st.colors(len(A))))
    w = d["w"].fillna(d["w"].median() if d["w"].notna().any() else 1)
    lw = _sizes(w, 0.8, 3.2) if w.nunique() > 1 else np.full(len(d), 1.6)
    for (_, r), l_ in zip(d.iterrows(), lw):
        t0, t1 = pos[("a", r["a"])], pos[("b", r["b"])]
        p0 = np.array([np.cos(t0), np.sin(t0)])
        p1 = np.array([np.cos(t1), np.sin(t1)])
        ctrl = (p0 + p1) / 2 * (1 - o["curv"])
        ax.add_patch(PathPatch(Path([p0, ctrl, p1], [Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none",
                               ec=cols[r["a"]], lw=l_, alpha=o["alpha"]))
    dA = d["a"].value_counts()
    sA = _sizes([dA[k] for k in A], o["node"] * 0.6, o["node"] * 1.6) if len(A) > 1 else [o["node"]]
    sB = _sizes([deg[k] for k in B], o["node"] * 0.4, o["node"] * 1.2) if len(B) > 1 else [o["node"]]
    ax.scatter(np.cos(angA), np.sin(angA), s=sA, c=[cols[k] for k in A], edgecolor="white", lw=1.2, zorder=3)
    ax.scatter(np.cos(angB), np.sin(angB), s=sB, c="#4d4d4d", edgecolor="white", lw=1.2, zorder=3, marker="D")
    for k, a in zip(A, angA):
        _radial_text(ax, a, 1.09, k, st.font * 0.8, color=cols[k], fontweight="bold")
    for k, a in zip(B, angB):
        _radial_text(ax, a, 1.09, k, st.font * 0.75)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.6, 1.6)
    st.legend(ax, [Line2D([], [], ls="", marker="o", ms=9, mfc=st.color(0), mec="white", label=c["a"]),
                   Line2D([], [], ls="", marker="D", ms=8, mfc="#4d4d4d", mec="white", label=c["b"])],
              loc="lower right")
    st.labels(ax, "", "", "")


# ═══════════════════════════════════════════════════════════════════════════════
# Genome & sequence
# ═══════════════════════════════════════════════════════════════════════════════

_DNA_COL = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839", "U": "#D62839"}
_AA_CLASS = {**{a: "hydrophobic" for a in "AVLIMFWP"}, **{a: "polar" for a in "GSTYC"},
             **{a: "neutral (amide)" for a in "NQ"}, **{a: "basic" for a in "KRH"},
             **{a: "acidic" for a in "DE"}}
_AA_CLASS_COL = {"hydrophobic": "#222222", "polar": "#1b9e3e", "neutral (amide)": "#8e44ad", "basic": "#2c5fd1",
                 "acidic": "#d62728"}
_LETTER_PATHS = {}


def _letter(ax, ch, x, y, w, h, color):
    """Draw *ch* as a filled glyph exactly filling the box (x, y, w, h)."""
    if h <= 1e-4:
        return
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D
    tp = _LETTER_PATHS.get(ch)
    if tp is None:
        tp = TextPath((0, 0), ch, size=1, prop=FontProperties(family="DejaVu Sans", weight="bold"))
        _LETTER_PATHS[ch] = tp
    bb = tp.get_extents()
    tr = Affine2D().translate(-bb.x0, -bb.y0).scale(w / max(bb.width, 1e-9), h / max(bb.height, 1e-9)) \
        .translate(x, y)
    ax.add_patch(PathPatch(tr.transform_path(tp), fc=color, ec="none", lw=0))


def _ex_motif():
    r = rng(43)
    cons = "GCCACCAGGGGGCGC"     # CTCF-like core
    alt = {"G": "AC", "C": "TA", "A": "GT", "T": "CA"}
    strength = [0.55, 0.8, 0.9, 0.5, 0.95, 0.9, 0.85, 0.95, 0.98, 0.9, 0.95, 0.6, 0.85, 0.5, 0.45]
    seqs = []
    for _ in range(60):
        s = "".join(ch if r.random() < p else (r.choice(list(alt[ch])) if r.random() < 0.7 else r.choice(list("ACGT")))
                    for ch, p in zip(cons, strength))
        seqs.append(s)
    return pd.DataFrame({"Sequence": seqs})


@plot("motif_logo_043", "Sequence motif logo", G, number="043", example=_ex_motif,
      roles=[Role("seq", "Aligned sequences (one per row)", "any")],
      opts=[Opt("unit", "Letter height", "choice", "information (bits)", choices=["information (bits)", "probability"]),
            Opt("alphabet", "Alphabet", "choice", "auto", choices=["auto", "DNA / RNA", "protein"]),
            Opt("correct", "Small-sample correction", "bool", True),
            Opt("start", "First position number", "int", 1, lo=-1e6, hi=1e6)],
      desc="One aligned sequence per row (DNA, RNA or protein, same length; '-' = gap); letter heights = "
           "information content.", size=(9, 3.8))
def motif_logo_043(fig, df, c, o, st):
    seqs = [s.strip().upper() for s in df[c["seq"]].dropna().astype(str) if s.strip()]
    L = max((len(s) for s in seqs), default=0)
    seqs = [s.ljust(L, "-") for s in seqs]
    letters = set("".join(seqs)) - set("-. ")
    nuc = letters <= set("ACGTUN") if o["alphabet"] == "auto" else o["alphabet"].startswith("DNA")
    alph = sorted((set("ACGU") if "U" in letters and "T" not in letters else set("ACGT")) if nuc
                  else set("ACDEFGHIKLMNPQRSTVWY") | (letters - set("BJOXZN*")))
    N = len(alph)
    ax = st.ax(fig.add_subplot(), "y")
    bits = o["unit"].startswith("info")
    for j in range(L):
        col = [s[j] for s in seqs if s[j] in alph]
        n = len(col)
        if not n:
            continue
        cnt = Counter(col)
        f = {a: cnt.get(a, 0) / n for a in alph}
        H = -sum(p * np.log2(p) for p in f.values() if p > 0)
        en = (N - 1) / (2 * np.log(2) * n) if o["correct"] else 0
        ic = max(np.log2(N) - (H + en), 0) if bits else 1.0
        y = 0.0
        for a, p in sorted(f.items(), key=lambda z: z[1]):
            h = p * ic
            colr = _DNA_COL.get(a, "#555555") if nuc else _AA_CLASS_COL.get(_AA_CLASS.get(a, ""), "#777777")
            _letter(ax, a, j + 0.55, y, 0.9, h, colr)
            y += h
    ax.set_xlim(0.4, L + 0.6)
    ax.set_ylim(0, np.log2(N) if bits else 1)
    step = 1 if L <= 30 else 5
    ax.set_xticks(np.arange(1, L + 1, step), np.arange(1, L + 1, step) + int(o["start"]) - 1)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    if nuc:
        hs = [Patch(fc=_DNA_COL[a], label=a) for a in alph]
    else:
        hs = [Patch(fc=v, label=k) for k, v in _AA_CLASS_COL.items()]
    st.legend(ax, hs, **_leg_out(st))
    st.labels(ax, "Position", "Information (bits)" if bits else "Probability",
              f"{len(seqs)} sequences, {L} positions")


def _ex_snps():
    r = rng(107)
    d = _ex_peaks(107, 9000)
    hot = r.random(len(d)) < 0.18
    d.loc[hot, "Position"] = (d.loc[hot, "Chrom"].map(lambda c_: _HG38[_nc(c_)][0] * 1e6 * 0.3) *
                              r.normal(1, 0.04, hot.sum())).astype(int)
    d["SNP"] = [f"rs{1000000 + i * 37}" for i in range(len(d))]
    return d[["Chrom", "Position", "SNP"]]


@plot("snp_density_107", "SNP density along chromosomes", G, number="107", example=_ex_snps,
      roles=[Role("chrom", "Chromosome", "any"), Role("pos", "Position (bp)", "num")],
      opts=[Opt("bin", "Window size (Mb)", "float", 1.0, lo=0.001, hi=100, step=0.5), _GENOME_OPT,
            Opt("zero", "Colour of empty windows", "color", "#eeeeee")],
      desc="One row per SNP / variant: chromosome and position (bp); counted in windows along each chromosome.",
      size=(9, 7), defaults={"cmap": "YlOrRd"})
def snp_density_107(fig, df, c, o, st):
    from matplotlib.collections import PatchCollection
    ch = as_str(df, c["chrom"]) if c["chrom"] else pd.Series("chr", index=df.index)
    d = pd.DataFrame({"c": ch, "p": num(df, c["pos"])}).dropna()
    names, L, _ = _genome(d["c"], d["p"], o)
    b = float(o["bin"]) * 1e6
    ax = st.ax(fig.add_subplot(), "x")
    rects, vals = [], []
    for i, n in enumerate(names):
        nb = max(int(np.ceil(L[n] / b)), 1)
        h, _ = np.histogram(d.loc[d["c"] == n, "p"], bins=nb, range=(0, nb * b))
        for k, v in enumerate(h):
            rects.append(Rectangle((k * b / 1e6, i - 0.35), min(b, L[n] - k * b) / 1e6, 0.7))
            vals.append(v)
        ax.add_patch(FancyBboxPatch((0, i - 0.35), L[n] / 1e6, 0.7, boxstyle="round,pad=0,rounding_size=0.3",
                                    mutation_aspect=1 / max(L.values()) * 1e6 * 25, fc="none", ec="#666666",
                                    lw=0.6, zorder=3))
    vals = np.array(vals, float)
    cm = _cmap(st).copy()
    cm.set_under(o["zero"])
    norm = Normalize(0.5, max(float(np.percentile(vals, 99.5)) if len(vals) else 1.0, 1.0))
    pc = PatchCollection(rects, cmap=cm, norm=norm, lw=0)
    pc.set_array(np.where(vals > 0, vals, 0.0))
    ax.add_collection(pc)
    ax.set_yticks(range(len(names)), [_short(n) for n in names], fontsize=st.font * 0.8)
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_xlim(0, max(L.values()) / 1e6 * 1.01)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    cb = _cbar(fig, st, cm, norm, f"SNPs per {o['bin']:g} Mb", ax=ax, shrink=0.5, extend="both")
    cb.ax.set_ylim(0.5, None)
    st.labels(ax, "Position (Mb)", "Chromosome")


def _ex_peak_chroms():
    d = _ex_peaks(87, 4200)
    d["Start"] = d.pop("Position")
    d["End"] = d["Start"] + rng(1).integers(200, 3000, len(d))
    return d


@plot("peak_chrom_dist_087", "Peaks per chromosome", G, number="087", example=_ex_peak_chroms,
      roles=[Role("chrom", "Chromosome", "any"), Role("pos", "Peak start / position (bp)", "num"),
             Role("group", "Sample / group (optional)", "cat", optional=True)],
      opts=[Opt("norm", "Normalise", "choice", "none", choices=["none", "per Mb", "% of peaks"]), _GENOME_OPT,
            Opt("vals", "Show values", "bool", False), Opt("horiz", "Horizontal bars", "bool", False)],
      desc="One row per peak: chromosome and start position (plus an optional sample column for grouped bars).",
      size=(9, 5))
def peak_chrom_dist_087(fig, df, c, o, st):
    d = pd.DataFrame({"c": as_str(df, c["chrom"]), "p": num(df, c["pos"])})
    d["g"] = as_str(df, c["group"]) if c.get("group") else "Peaks"
    d = d.dropna()
    names, L, _ = _genome(d["c"], d["p"], o)
    groups = cats(d, "g")
    tab = pd.crosstab(d["c"], d["g"]).reindex(index=names, columns=groups, fill_value=0).astype(float)
    lab = "Number of peaks"
    if o["norm"] == "per Mb":
        tab = tab.div(pd.Series(L) / 1e6, axis=0)
        lab = "Peaks per Mb"
    elif o["norm"] == "% of peaks":
        tab = tab / tab.sum(0) * 100
        lab = "% of peaks"
    ax = st.ax(fig.add_subplot(), "x" if o["horiz"] else "y")
    k = len(groups)
    w = 0.8 / k
    x = np.arange(len(names))
    for j, g in enumerate(groups):
        off = (j - (k - 1) / 2) * w
        if o["horiz"]:
            b = ax.barh(x + off, tab[g], height=w, color=st.color(j), edgecolor="#333333", lw=0.4, label=g)
        else:
            b = ax.bar(x + off, tab[g], width=w, color=st.color(j), edgecolor="#333333", lw=0.4, label=g)
        if o["vals"]:
            ax.bar_label(b, fmt="%.3g", padding=1, fontsize=st.font * 0.6, rotation=0 if o["horiz"] else 90)
    ticks = [_short(n) for n in names]
    if o["horiz"]:
        ax.set_yticks(x, ticks, fontsize=st.font * 0.8)
        ax.set_ylim(len(names) - 0.4, -0.6)
        st.labels(ax, lab, "Chromosome")
    else:
        ax.set_xticks(x, ticks, fontsize=st.font * 0.8)
        ax.set_xlim(-0.6, len(names) - 0.4)
        st.labels(ax, "Chromosome", lab)
    if k > 1:
        st.legend(ax, title=c["group"])


def _ex_peak_values():
    r = rng(108)
    d = _ex_peaks(108, 650)
    d["Score"] = np.round(r.gamma(2.0, 2.5, len(d)), 2)
    return d


def _rbox(ax, x, y0, y1, w, asp, **kw):
    """Rounded vertical bar from y0 to y1 centred on x (asp = data-y per data-x for round corners)."""
    rs = max(min(w / 2, abs(y1 - y0) / max(asp, 1e-12) / 2), 1e-6)
    p = FancyBboxPatch((x - w / 2, y0), w, y1 - y0, boxstyle=f"round,pad=0,rounding_size={rs}",
                       mutation_aspect=asp, **kw)
    ax.add_patch(p)
    return p


def _ideogram_axes(fig, st, names, L, wpad=1.0):
    ax = fig.add_subplot()
    st.ax(ax, "")
    top = max(L.values()) / 1e6
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.set_ylim(top * 1.03, -top * 0.03)
    fw, fh = fig.get_size_inches()
    asp = (top * 1.06 / (fh * 0.8)) / (len(names) / (fw * 0.85))
    ax.set_xticks(range(len(names)), [_short(n) for n in names], fontsize=st.font * 0.8)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    return ax, asp


@plot("peak_ideogram_108", "Peak positions on chromosomes (ideogram)", G, number="108", example=_ex_peak_values,
      roles=[Role("chrom", "Chromosome", "any"), Role("pos", "Position (bp)", "num"),
             Role("value", "Value (colour, optional)", "num", optional=True)],
      opts=[_GENOME_OPT, Opt("width", "Chromosome width", "float", 0.5, lo=0.1, hi=0.9, step=0.05),
            Opt("mark", "Marks", "choice", "ticks across", choices=["ticks across", "dots beside"]),
            Opt("lw", "Tick thickness", "float", 1.0, lo=0.1, hi=5, step=0.1)],
      desc="One row per peak / site: chromosome, position (bp) and an optional value used for the colour.",
      size=(10, 6), defaults={"cmap": "YlOrRd"})
def peak_ideogram_108(fig, df, c, o, st):
    v = num(df, c["value"]) if c.get("value") else pd.Series(1.0, index=df.index)
    d = pd.DataFrame({"c": as_str(df, c["chrom"]), "p": num(df, c["pos"]), "v": v}).dropna(subset=["p"])
    names, L, cen = _genome(d["c"], d["p"], o)
    ax, asp = _ideogram_axes(fig, st, names, L)
    w = float(o["width"])
    has_v = bool(c.get("value")) and d["v"].notna().any() and d["v"].max() > d["v"].min()
    cm = _cmap(st)
    norm = Normalize(d["v"].min(), d["v"].max()) if has_v else None
    for i, n in enumerate(names):
        _rbox(ax, i, 0, L[n] / 1e6, w, asp, fc="#f4f4f4", ec="#555555", lw=0.8, zorder=1)
        sub = d[d["c"] == n]
        colr = cm(norm(sub["v"])) if has_v else [st.color(0)] * len(sub)
        if o["mark"] == "ticks across":
            ax.hlines(sub["p"] / 1e6, i - w / 2 * 0.92, i + w / 2 * 0.92, colors=colr, lw=o["lw"], zorder=2)
        else:
            ax.scatter(np.full(len(sub), i + w / 2 + 0.12), sub["p"] / 1e6, s=10, c=colr, lw=0, zorder=2)
    if has_v:
        _cbar(fig, st, cm, norm, c["value"], ax=ax, shrink=0.5)
    st.labels(ax, "Chromosome", "Position (Mb)")


def _ex_peak_updown():
    r = rng(109)
    d = _ex_peaks(109, 520)
    d["Regulation"] = np.where(r.random(len(d)) < 0.55, "Up", "Down")
    return d


@plot("peak_centromere_109", "Up / down peaks on chromosomes with centromeres", G, number="109",
      example=_ex_peak_updown,
      roles=[Role("chrom", "Chromosome", "any"), Role("pos", "Position (bp)", "num"),
             Role("dir", "Direction (Up/Down text or signed value)", "any"),
             Role("cen", "Centromere position (bp, optional)", "num", optional=True)],
      opts=[_GENOME_OPT, Opt("cen_frac", "Centromere at (fraction of length) when unknown", "float", 0.4,
                             lo=0.05, hi=0.95, step=0.05),
            Opt("width", "Chromosome width", "float", 0.36, lo=0.1, hi=0.8, step=0.02),
            Opt("tick", "Tick length", "float", 0.22, lo=0.05, hi=0.5, step=0.01),
            Opt("up", "Up colour", "color", "#d6404e"), Opt("down", "Down colour", "color", "#3a78c2")],
      desc="One row per peak: chromosome, position (bp) and direction (Up/Down or a signed log2FC); human hg38 "
           "centromeres are used when names match, else an optional centromere column or a fraction of length.",
      size=(10, 6))
def peak_centromere_109(fig, df, c, o, st):
    raw = df[c["dir"]]
    nv = num(df, c["dir"])
    if nv.notna().mean() >= 0.8:
        up = nv > 0
    else:
        s = raw.astype(str).str.strip().str.lower()
        up = s.str.startswith(("up", "+", "pos", "hyper", "gain", "increase"))
    d = pd.DataFrame({"c": as_str(df, c["chrom"]), "p": num(df, c["pos"]), "up": up})
    d["cen"] = num(df, c["cen"]) if c.get("cen") else np.nan
    d = d.dropna(subset=["p"])
    names, L, cen = _genome(d["c"], d["p"], o)
    ax, asp = _ideogram_axes(fig, st, names, L)
    w, tl = float(o["width"]), float(o["tick"])
    for i, n in enumerate(names):
        sub = d[d["c"] == n]
        cpos = sub["cen"].dropna()
        cp = float(cpos.iloc[0]) if len(cpos) else (cen[n] if cen else L[n] * o["cen_frac"])
        if len(cpos) and 0 < cp <= 1:      # given as a fraction of the chromosome length
            cp *= L[n]
        cp = min(max(cp, L[n] * 0.02), L[n] * 0.98) / 1e6
        _rbox(ax, i, 0, cp, w, asp, fc="#f0f0f0", ec="#555555", lw=0.8, zorder=1)
        _rbox(ax, i, cp, L[n] / 1e6, w, asp, fc="#f0f0f0", ec="#555555", lw=0.8, zorder=1)
        ax.plot([i - w / 2 - 0.02, i + w / 2 + 0.02], [cp, cp], color="white", lw=2.5, zorder=1.5,
                solid_capstyle="butt")
        u = sub[sub["up"]]["p"] / 1e6
        dn = sub[~sub["up"]]["p"] / 1e6
        ax.hlines(u, i - w / 2 - tl, i - w / 2, colors=o["up"], lw=0.9, zorder=2)
        ax.hlines(dn, i + w / 2, i + w / 2 + tl, colors=o["down"], lw=0.9, zorder=2)
    ax.set_xlim(-0.5 - tl, len(names) - 0.5 + tl)
    st.legend(ax, [Line2D([], [], color=o["up"], lw=2, label=f"Up ({int(d['up'].sum())})"),
                   Line2D([], [], color=o["down"], lw=2, label=f"Down ({int((~d['up']).sum())})")],
              loc="best")
    st.labels(ax, "Chromosome", "Position (Mb)")


def _ex_gwas():
    r = rng(110)
    rows = []
    hits = {"chr2": 0.42, "chr6": 0.18, "chr11": 0.6, "chr16": 0.31, "chr19": 0.5}
    k = 0
    for ch in _EX_CHROMS:
        Lc = _HG38[_nc(ch)][0] * 1e6
        n = int(Lc / 1e6 * 5.5)
        pos = np.sort(r.uniform(0, Lc, n)).astype(int)
        p = r.uniform(0, 1, n) ** 1.05
        if ch in hits:
            ctr = hits[ch] * Lc
            dist = np.abs(pos - ctr) / 1.5e6
            strength = 10 ** -(r.uniform(9, 16) * np.exp(-dist ** 2))
            p = np.minimum(p, strength * r.uniform(0.2, 5, n))
        for a, b in zip(pos, p):
            rows.append((f"rs{2000000 + k * 13}", ch, a, float(b)))
            k += 1
    return pd.DataFrame(rows, columns=["SNP", "Chrom", "Position", "P"])


@plot("manhattan_110", "Manhattan plot", G, number="110", example=_ex_gwas,
      roles=[Role("snp", "SNP / marker names (labels)", "cat", optional=True), Role("chrom", "Chromosome", "any"),
             Role("pos", "Position (bp)", "num"), Role("p", "p-value", "num")],
      opts=[Opt("sig", "Genome-wide threshold", "float", 5e-8, lo=0, hi=1, step=1e-8),
            Opt("sugg", "Suggestive threshold (0 = none)", "float", 1e-5, lo=0, hi=1, step=1e-6),
            Opt("top", "Label top hits (one per locus)", "int", 5, lo=0, hi=50),
            Opt("psize", "Point size", "float", 6, lo=0.5, hi=100),
            Opt("colours", "Colours", "choice", "two alternating", choices=["two alternating", "palette"]),
            Opt("hl", "Highlight hits", "color", "#2bb673")],
      desc="One row per SNP: SNP id, chromosome, position (bp) and p-value (e.g. a GWAS summary table).",
      size=(11, 5))
def manhattan_110(fig, df, c, o, st):
    snp = as_str(df, c["snp"]) if c.get("snp") else pd.Series("", index=df.index)
    d = pd.DataFrame({"s": snp, "c": as_str(df, c["chrom"]), "p": num(df, c["pos"]), "pv": num(df, c["p"])})
    d = d.dropna(subset=["p", "pv"])
    d = d[d["pv"] > 0]
    names, L, _ = _genome(d["c"], d["p"], o)
    gap = sum(L.values()) * 0.004
    off, x0 = {}, 0.0
    for n in names:
        off[n] = x0
        x0 += L[n] + gap
    d["x"] = d["p"] + d["c"].map(off)
    d["y"] = -np.log10(d["pv"])
    ax = st.ax(fig.add_subplot(), "y")
    cols = st.colors(2) if o["colours"] == "two alternating" else st.colors(len(names))
    for i, n in enumerate(names):
        sub = d[d["c"] == n]
        ax.scatter(sub["x"], sub["y"], s=o["psize"], c=[cols[i % len(cols)]], lw=0, rasterized=True)
    sig = d[d["pv"] < o["sig"]] if o["sig"] > 0 else d.iloc[:0]
    ax.scatter(sig["x"], sig["y"], s=o["psize"] * 1.6, c=o["hl"], lw=0, zorder=3)
    if o["sig"] > 0:
        ax.axhline(-np.log10(o["sig"]), color="#d6404e", ls="--", lw=0.9)
    if o["sugg"] > 0:
        ax.axhline(-np.log10(o["sugg"]), color="#3a78c2", ls=":", lw=0.9)
    ax.set_xlim(-gap, x0)
    ymax = max(float(d["y"].max() if len(d) else 1), -np.log10(o["sig"]) if o["sig"] > 0 else 0)
    ax.set_ylim(0, ymax * 1.12)
    centers = [off[n] + L[n] / 2 for n in names]
    labs = [_short(n) for n in names]
    W = fig.get_size_inches()[0] * 72 * 0.85
    keep = _thin(centers, labs, W, x0, st.font * 0.8)
    ax.set_xticks([cc for cc, k in zip(centers, keep) if k], [lb for lb, k in zip(labs, keep) if k],
                  fontsize=st.font * 0.8)
    if o["top"] > 0 and len(sig) and c.get("snp"):
        loci = []
        for _, r in sig.sort_values("pv").iterrows():
            if all(r["c"] != q["c"] or abs(r["p"] - q["p"]) > 1e6 for q in loci):
                loci.append(r)
            if len(loci) >= o["top"]:
                break
        lab = pd.DataFrame(loci)
        _repel(ax, lab["x"], lab["y"], lab["s"], st)
    st.labels(ax, "Chromosome", "−log10 p")


def _ex_ecc():
    r = rng(95)
    d = _ex_peaks(95, 1400)
    d["Regulation"] = np.where(r.random(len(d)) < 0.58, "Up", "Down")
    return d[["Chrom", "Regulation"]]


@plot("ecc_chrom_dist_095", "Up / down counts per chromosome (mirrored bars)", G, number="095", example=_ex_ecc,
      roles=[Role("chrom", "Chromosome", "any"), Role("dir", "Direction (Up/Down text or signed value)", "any")],
      opts=[Opt("norm", "Normalise per Mb (needs human chromosome names)", "bool", False), _GENOME_OPT,
            Opt("vals", "Show counts", "bool", True),
            Opt("up", "Up colour", "color", "#d6404e"), Opt("down", "Down colour", "color", "#3a78c2")],
      desc="One row per gene / eccDNA: chromosome and direction (Up/Down or a signed log2FC); up counts are "
           "drawn above, down counts below the axis.", size=(10, 5))
def ecc_chrom_dist_095(fig, df, c, o, st):
    nv = num(df, c["dir"])
    if nv.notna().mean() >= 0.8:
        up = nv > 0
    else:
        up = df[c["dir"]].astype(str).str.strip().str.lower().str.startswith(("up", "+", "pos", "hyper", "gain"))
    d = pd.DataFrame({"c": as_str(df, c["chrom"]), "up": up})
    names = _chrom_sort(d["c"].unique())
    u = d[d["up"]]["c"].value_counts().reindex(names, fill_value=0).astype(float)
    dn = d[~d["up"]]["c"].value_counts().reindex(names, fill_value=0).astype(float)
    lab = "Count"
    if o["norm"] and all(_nc(n) in _HG38 for n in names):
        Lmb = pd.Series({n: _HG38[_nc(n)][0] for n in names})
        u, dn = u / Lmb, dn / Lmb
        lab = "Count per Mb"
    ax = st.ax(fig.add_subplot(), "y")
    x = np.arange(len(names))
    b1 = ax.bar(x, u, color=o["up"], edgecolor="#333333", lw=0.4, width=0.72, label=f"Up ({int((d['up']).sum())})")
    b2 = ax.bar(x, -dn, color=o["down"], edgecolor="#333333", lw=0.4, width=0.72,
                label=f"Down ({int((~d['up']).sum())})")
    ax.axhline(0, color="#333333", lw=0.8)
    if o["vals"]:
        fmt = "%.2g" if lab != "Count" else "%d"
        ax.bar_label(b1, labels=[fmt % v for v in u], padding=1, fontsize=st.font * 0.6)
        ax.bar_label(b2, labels=[fmt % v for v in dn], padding=1, fontsize=st.font * 0.6)
    m = max(u.max(), dn.max(), 1e-9) * 1.15
    ax.set_ylim(-m, m)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{abs(v):g}"))
    ax.set_xticks(x, [_short(n) for n in names], fontsize=st.font * 0.8)
    ax.set_xlim(-0.6, len(names) - 0.4)
    st.labels(ax, "Chromosome", lab)
    st.legend(ax, loc="upper right")


def _ex_circos():
    r = rng(31)
    rows = []
    for ch in _EX_CHROMS:
        Lc = _HG38[_nc(ch)][0]
        for s in np.arange(0, Lc, 5.0):
            base = 2 + np.sin(s / 17 + len(ch)) * 0.6
            rows.append((ch, int(s * 1e6), round(float(base + r.gamma(2, 0.5)), 2),
                         round(float(base * 0.8 + r.gamma(2, 0.45) + (1.8 if ch in ("chr8", "chr17") else 0)), 2)))
    return pd.DataFrame(rows, columns=["Chrom", "Start", "Tumour", "Normal"])


@plot("circos_two_tracks_031", "Circos plot with two tracks", G, number="031", example=_ex_circos,
      roles=[Role("chrom", "Chromosome", "any"), Role("pos", "Position / bin start (bp)", "num"),
             Role("tracks", "Track values (1–4 columns)", "num", multi=True)],
      opts=[Opt("bin", "Bin size (Mb)", "float", 5.0, lo=0.01, hi=100, step=1), _GENOME_OPT,
            Opt("style", "Track style", "choice", "histogram", choices=["histogram", "points", "line"]),
            Opt("gap", "Gap between chromosomes (degrees)", "float", 1.5, lo=0, hi=10, step=0.5)],
      desc="One row per genomic bin or feature: chromosome, position (bp) and one numeric column per track "
           "(values are averaged in bins).", size=(7.5, 7.5), axis_labels=False)
def circos_two_tracks_031(fig, df, c, o, st):
    tracks = [t for t in (c["tracks"] if isinstance(c["tracks"], list) else [c["tracks"]]) if t][:4]
    d = pd.DataFrame({"c": as_str(df, c["chrom"]), "p": num(df, c["pos"])})
    for t in tracks:
        d[t] = num(df, t)
    d = d.dropna(subset=["p"])
    names, L, _ = _genome(d["c"], d["p"], o)
    total = sum(L.values())
    gap = np.radians(float(o["gap"]))
    scale = (2 * np.pi - gap * len(names)) / total
    start, a = {}, 0.0
    for n in names:
        start[n] = a
        a += L[n] * scale + gap
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_axis_off()
    ax.set_ylim(0, 1.2)
    chc = st.colors(max(len(names), 1)) if len(names) <= 10 else None
    for i, n in enumerate(names):
        a0, w = start[n], L[n] * scale
        col = chc[i] if chc else ("#8c8c8c" if i % 2 else "#4d4d4d")
        ax.bar(a0 + w / 2, 0.05, width=w, bottom=0.96, color=col, edgecolor="none")
        mid = a0 + w / 2
        deg = np.degrees(mid)
        ax.text(mid, 1.1, _short(n), ha="center", va="center", fontsize=st.font * 0.72,
                rotation=-deg if deg <= 90 or deg >= 270 else 180 - deg, rotation_mode="anchor")
    b = float(o["bin"]) * 1e6
    d["bin"] = (d["p"] // b).astype(int)
    agg = d.groupby(["c", "bin"])[tracks].mean().reset_index()
    nt = max(len(tracks), 1)
    band = 0.7 / nt
    tcols = st.colors(nt)
    for k, t in enumerate(tracks):
        r1 = 0.92 - k * band - 0.03
        r0 = r1 - band + 0.06
        v = agg[t]
        lo, hi = v.min(), v.max()
        lo = lo - 0.08 * (hi - lo) if hi > lo else lo - 1
        hi = hi if hi > lo else lo + 1
        ring = np.linspace(0, 2 * np.pi, 361)
        ax.fill_between(ring, r0, r1, color="#f5f5f5", lw=0, zorder=0)
        for n in names:
            sub = agg[agg["c"] == n]
            th = start[n] + (sub["bin"] * b + b / 2).clip(upper=L[n]) * scale
            h = (sub[t] - lo) / (hi - lo) * (r1 - r0)
            if o["style"] == "histogram":
                nb = max(int(np.ceil(L[n] / b)), 1)
                hv = np.zeros(nb)
                ok = (sub["bin"] >= 0) & (sub["bin"] < nb)
                hv[sub.loc[ok, "bin"].to_numpy()] = h[ok].fillna(0).to_numpy()
                edges = start[n] + np.minimum(np.arange(nb + 1) * b, L[n]) * scale
                tx = np.repeat(edges, 2)[1:-1]
                ax.fill_between(tx, r0, r0 + np.repeat(hv, 2), color=tcols[k], lw=0, zorder=2)
            elif o["style"] == "points":
                ax.scatter(th, r0 + h, s=4, color=tcols[k], lw=0, zorder=2)
            else:
                ax.plot(th, r0 + h, color=tcols[k], lw=1, zorder=2)
        ax.plot(np.linspace(0, 2 * np.pi, 361), np.full(361, r0), color="#bbbbbb", lw=0.5)
    st.legend(ax, [Patch(fc=tcols[k], label=t) for k, t in enumerate(tracks)],
              **({"loc": "upper left", "bbox_to_anchor": (-0.08, 1.08)} if st.legend_mode == "best" else {}))
    t = st.opts.get("title")
    if t:
        ax.set_title(t, fontsize=st.font * 1.15, fontweight="bold", pad=18)


def _ex_lod():
    r = rng(94)
    pos = np.round(np.arange(0, 80.01, 0.4), 2)
    lod = 0.4 + 6.8 * np.exp(-((pos - 41.5) / 4.2) ** 2) + 2.2 * np.exp(-((pos - 15) / 3) ** 2)
    lod = np.clip(lod + r.normal(0, 0.25, len(pos)), 0, None)
    return pd.DataFrame({"Marker": [f"M{i:03d}" for i in range(len(pos))], "Position_Mb": pos,
                         "LOD": np.round(lod, 3)})


@plot("snp_likelihood_094", "SNP / QTL likelihood (LOD) profile", G, number="094", example=_ex_lod,
      roles=[Role("marker", "Marker names (optional)", "cat", optional=True), Role("pos", "Position", "num"),
             Role("lod", "LOD / likelihood", "num")],
      opts=[Opt("thr", "Significance threshold", "float", 3.0, lo=0, hi=1e6, step=0.5),
            Opt("drop", "Peak interval: LOD drop", "float", 1.5, lo=0.1, hi=10, step=0.1),
            Opt("points", "Show points", "bool", True),
            Opt("colour", "Line colour", "color", "#3C5488")],
      desc="One row per marker along a chromosome region: marker name, position and LOD (or log-likelihood); "
           "the peak interval is where LOD stays within the chosen drop of the maximum.", size=(9, 4.8))
def snp_likelihood_094(fig, df, c, o, st):
    mk = as_str(df, c["marker"]) if c.get("marker") else pd.Series("", index=df.index)
    d = pd.DataFrame({"m": mk, "x": num(df, c["pos"]), "y": num(df, c["lod"])}).dropna(subset=["x", "y"])
    d = d.sort_values("x").reset_index(drop=True)
    ax = st.ax(fig.add_subplot())
    ax.plot(d["x"], d["y"], color=o["colour"], lw=1.6, zorder=2)
    if o["points"]:
        ax.scatter(d["x"], d["y"], s=10, color=o["colour"], zorder=3, lw=0)
    ax.axhline(o["thr"], color="#d6404e", ls="--", lw=1, label=f"Threshold ({o['thr']:g})")
    if len(d):
        k = int(d["y"].idxmax())
        top = d.loc[k]
        lim = top["y"] - o["drop"]
        lo = k
        while lo > 0 and d.loc[lo - 1, "y"] >= lim:
            lo -= 1
        hi = k
        while hi < len(d) - 1 and d.loc[hi + 1, "y"] >= lim:
            hi += 1
        a, b = d.loc[lo, "x"], d.loc[hi, "x"]
        ax.axvspan(a, b, color="#f0a030", alpha=0.22, lw=0, label=f"{o['drop']:g}-LOD interval ({a:g}–{b:g})")
        ax.scatter([top["x"]], [top["y"]], s=60, color="#d6404e", zorder=4, edgecolor="white")
        ax.annotate(f"{top['m'] + ': ' if top['m'] else ''}{top['x']:g}, LOD {top['y']:.2f}",
                    (top["x"], top["y"]), xytext=(8, 6), textcoords="offset points", fontsize=st.font * 0.8)
        ax.set_ylim(0, max(d["y"].max(), o["thr"]) * 1.15)
        ax.set_xlim(d["x"].min(), d["x"].max())
    st.labels(ax, c["pos"], c["lod"])
    st.legend(ax, loc="upper left")


def _ex_triplex():
    return pd.DataFrame({
        "TFO": ["TTCTTTCTTCTCTTCT", "GAGGGAAGGAGGGAGG", "TCTCTTTTCCCTCTT"],
        "Target (purine strand)": ["AAGAAAGAAGAGAAGA", "GGAGGGAGGAAGGGAG", "AGAGAAACGGGAGAA"],
        "Name": ["Pyrimidine motif TFO", "Purine motif TFO", "Mixed (1 mismatch)"]})


_COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "U": "A", "N": "N"}


@plot("triplex_dna_096", "Triplex DNA (TFO · duplex)", G, number="096", example=_ex_triplex,
      roles=[Role("tfo", "TFO sequence (5'→3')", "any"), Role("target", "Duplex purine strand (5'→3')", "any"),
             Role("name", "Name (optional)", "cat", optional=True)],
      opts=[Opt("row", "Row to draw", "int", 1, lo=1, hi=1e6),
            Opt("motif", "Binding motif", "choice", "auto",
                choices=["auto", "pyrimidine (parallel, Hoogsteen)", "purine (antiparallel, reverse Hoogsteen)"]),
            Opt("max", "Max bases shown", "int", 40, lo=4, hi=200)],
      desc="One row per triplex: TFO sequence and the purine-rich strand of the target duplex (5'→3'); the "
           "complementary strand is derived and Hoogsteen pairs are checked.", size=(10, 4), axis_labels=False)
def triplex_dna_096(fig, df, c, o, st):
    rows = df[[c["tfo"], c["target"]]].dropna()
    ax = fig.add_subplot()
    st.bare(ax)
    if not len(rows):
        return
    i = int(min(max(o["row"], 1), len(rows))) - 1
    tfo = str(rows.iloc[i, 0]).strip().upper().replace("U", "T")
    tgt = str(rows.iloc[i, 1]).strip().upper().replace("U", "T")
    name = str(df.loc[rows.index[i], c["name"]]) if c.get("name") else ""
    motif = o["motif"]
    if motif == "auto":
        motif = "pyr" if sum(ch in "CT" for ch in tfo) >= len(tfo) / 2 else "pur"
    else:
        motif = "pyr" if motif.startswith("pyr") else "pur"
    rules = {"pyr": {("T", "A"), ("C", "G")}, "pur": {("G", "G"), ("A", "A"), ("T", "A")}}[motif]
    t = tfo if motif == "pyr" else tfo[::-1]      # antiparallel TFO is written 3'→5' above the target
    best, off = -1, 0
    for k in range(-len(t) + 1, len(tgt)):
        sc = sum((t[j], tgt[j + k]) in rules for j in range(len(t)) if 0 <= j + k < len(tgt))
        if sc > best:
            best, off = sc, k
    n = min(len(tgt), int(o["max"]))
    comp = "".join(_COMP.get(ch, "N") for ch in tgt)
    yT, yP, yC = 2.0, 1.0, 0.0
    bc = {k: st.color(i) for i, k in enumerate("AGCT")}

    def base(x, y, ch):
        ax.add_patch(FancyBboxPatch((x - 0.36, y - 0.22), 0.72, 0.44, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc=bc.get(ch, "#999999"), ec="none", zorder=3))
        ax.text(x, y, ch, ha="center", va="center", color="white", fontsize=st.font * 0.85, fontweight="bold",
                zorder=4)
    nmatch = 0
    npair = 0
    for j in range(n):
        base(j, yP, tgt[j])
        base(j, yC, comp[j])
        nb = 3 if tgt[j] in "GC" else 2
        for q in range(nb):
            dx = (q - (nb - 1) / 2) * 0.12
            ax.plot([j + dx, j + dx], [yC + 0.24, yP - 0.24], color="#777777", lw=1, zorder=1)
        k = j - off
        if 0 <= k < len(t):
            base(j, yT, t[k])
            npair += 1
            ok = (t[k], tgt[j]) in rules
            nmatch += ok
            if ok:
                for dx in (-0.08, 0.08):
                    ax.plot([j + dx, j + dx], [yP + 0.24, yT - 0.24], color="#333333", lw=1.1, ls=(0, (2, 1.6)),
                            zorder=1)
            else:
                ax.text(j, (yP + yT) / 2, "×", ha="center", va="center", color="#d62728",
                        fontsize=st.font * 1.1, fontweight="bold")
    tl = [j for j in range(n) if 0 <= j - off < len(t)]
    lab = st.font * 0.85
    ax.text(-0.8, yP, "Purine strand", ha="right", va="center", fontsize=lab)
    ax.text(-0.8, yC, "Pyrimidine strand", ha="right", va="center", fontsize=lab)
    if tl:
        ax.text(tl[0] - 0.8, yT, "TFO", ha="right", va="center", fontsize=lab, fontweight="bold")
    small = st.font * 0.75
    for y, l5 in ((yP, True), (yC, False)):
        ax.text(-0.5, y + 0.3, "5'" if l5 else "3'", fontsize=small, color="#555555", ha="center")
        ax.text(n - 0.5, y + 0.3, "3'" if l5 else "5'", fontsize=small, color="#555555", ha="center")
    if tl:
        par = motif == "pyr"
        ax.text(tl[0] - 0.5, yT + 0.3, "5'" if par else "3'", fontsize=small, color="#555555", ha="center")
        ax.text(tl[-1] + 0.5, yT + 0.3, "3'" if par else "5'", fontsize=small, color="#555555", ha="center")
    ax.set_xlim(-1, n)
    ax.set_ylim(-0.6, 2.75)
    mot = "pyrimidine motif, parallel (T·A-T, C⁺·G-C)" if motif == "pyr" else \
        "purine motif, antiparallel (G·G-C, A·A-T, T·A-T)"
    ax.text(n / 2 - 0.5, 2.62, f"{mot} — {nmatch}/{npair} Hoogsteen pairs", ha="center", va="center",
            fontsize=st.font * 0.8, color="#444444")
    st.legend(ax, [Line2D([], [], color="#333333", ls=(0, (2, 1.6)), lw=1.2, label="Hoogsteen bond"),
                   Line2D([], [], color="#777777", lw=1, label="Watson–Crick bond"),
                   Line2D([], [], ls="", marker="x", mec="#d62728", mew=2, label="mismatch")],
              **_leg_out(st, "below"), ncol=3)
    st.labels(ax, "", "", name)


_EISEN = {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85, "E": -0.74, "G": 0.48,
          "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18,
          "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08}
_HCLASS = [("hydrophobic", "AILMFVW"), ("polar", "STNQYC"), ("positive", "KRH"), ("negative", "DE"),
           ("special (G, P)", "GP")]


def _ex_peptides():
    return pd.DataFrame({"Sequence": ["GIGKFLHSAKKFGKAFVGEIMNS", "GIGAVLKVLTTGLPALISWIKRKRQQ",
                                      "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"],
                         "Peptide": ["Magainin 2", "Melittin", "LL-37"]})


@plot("wenxiang_119", "Helical wheel / wenxiang diagram", G, number="119", example=_ex_peptides,
      roles=[Role("seq", "Peptide sequence (one-letter code)", "any"), Role("name", "Name (optional)", "cat",
                                                                            optional=True)],
      opts=[Opt("mode", "Diagram", "choice", "both", choices=["both", "helical wheel", "wenxiang (spiral)"]),
            Opt("row", "Row to draw", "int", 1, lo=1, hi=1e6),
            Opt("n", "Residues in the wheel", "int", 18, lo=3, hi=200),
            Opt("moment", "Show hydrophobic moment", "bool", True)],
      desc="One peptide per row (one-letter amino-acid code, optional name column); residues are placed every "
           "100° and coloured by hydrophobicity class.", size=(11, 5.8), axis_labels=False)
def wenxiang_119(fig, df, c, o, st):
    # choose the sequence column even if auto-assignment picked the name column
    def looks_like_seq(col):
        v = df[col].dropna().astype(str).str.strip()
        return len(v) > 0 and v.str.fullmatch(r"[A-Za-z]{5,}").mean() > 0.8
    seqcol, namecol = c["seq"], c.get("name")
    if namecol and not looks_like_seq(seqcol) and looks_like_seq(namecol):
        seqcol, namecol = namecol, seqcol
    rows = df[df[seqcol].notna()]
    i = int(min(max(o["row"], 1), max(len(rows), 1))) - 1
    seq = "".join(ch for ch in str(rows[seqcol].iloc[i]).upper() if ch.isalpha()) if len(rows) else ""
    name = str(rows[namecol].iloc[i]) if namecol and len(rows) else ""
    cls = {a: k for k, (_, s) in enumerate(_HCLASS) for a in s}
    cols = st.colors(len(_HCLASS))
    modes = ["helical wheel", "wenxiang (spiral)"] if o["mode"] == "both" else [o["mode"]]
    axes = [fig.add_subplot(1, len(modes), k + 1) for k in range(len(modes))]
    for ax, mode in zip(axes, modes):
        ax.set_aspect("equal")
        st.bare(ax)
        if mode == "helical wheel":
            s = seq[:max(int(o["n"]), 1)]
            ang = np.radians(90 - 100 * np.arange(len(s)))
            r = 1 + 0.32 * (np.arange(len(s)) // 18)
            xs, ys = r * np.cos(ang), r * np.sin(ang)
            for k in range(len(s) - 1):
                ax.plot(xs[k:k + 2], ys[k:k + 2], color="#555555", lw=2.2 * (1 - 0.8 * k / max(len(s), 1)) + 0.3,
                        alpha=0.8, zorder=1)
            rad = 0.13
            title = "Helical wheel"
        else:
            s = seq
            k_ = np.arange(len(s))
            ang = np.radians(90 - 100 * k_)
            r = 0.35 + 0.085 * k_
            xs, ys = r * np.cos(ang), r * np.sin(ang)
            tt = np.linspace(0, max(len(s) - 1, 1), 600)
            ax.plot((0.35 + 0.085 * tt) * np.cos(np.radians(90 - 100 * tt)),
                    (0.35 + 0.085 * tt) * np.sin(np.radians(90 - 100 * tt)), color="#999999", lw=1, zorder=1)
            rad = 0.12
            title = "Wenxiang diagram"
        for k, a in enumerate(s):
            col = cols[cls.get(a, 1)]
            ax.add_patch(Circle((xs[k], ys[k]), rad, fc=col, ec="#333333", lw=0.6, zorder=3))
            lum = np.dot(to_rgb(col), [0.299, 0.587, 0.114])
            ax.text(xs[k], ys[k], a, ha="center", va="center", fontsize=st.font * 0.8, fontweight="bold",
                    color="white" if lum < 0.55 else "black", zorder=4)
            ax.text(xs[k] * (1 + (rad + 0.08) / max(np.hypot(xs[k], ys[k]), 1e-9)),
                    ys[k] * (1 + (rad + 0.08) / max(np.hypot(xs[k], ys[k]), 1e-9)), str(k + 1),
                    ha="center", va="center", fontsize=st.font * 0.55, color="#666666", zorder=4)
        if o["moment"] and len(s):
            h = np.array([_EISEN.get(a, 0) for a in s])
            ang_all = np.radians(90 - 100 * np.arange(len(s)))
            mx, my = (h * np.cos(ang_all)).sum(), (h * np.sin(ang_all)).sum()
            mu = np.hypot(mx, my) / len(s)
            if mode == "helical wheel":
                sc = 0.8 * min(1.0, 0.35 + mu / 0.6) / max(np.hypot(mx, my), 1e-9)
                ax.add_patch(FancyArrowPatch((0, 0), (mx * sc, my * sc), arrowstyle="-|>", mutation_scale=14,
                                             color="#d62728", lw=1.6, zorder=2))
                ax.text(0, -0.15 if my > 0 else 0.15, f"μH = {mu:.2f}", ha="center", va="center",
                        fontsize=st.font * 0.7, color="#d62728", zorder=2)
            else:
                ax.text(0, -(0.35 + 0.085 * len(s)) - 0.3, f"hydrophobic moment μH = {mu:.2f}", ha="center",
                        va="top", fontsize=st.font * 0.7, color="#d62728")
        R = (np.max(np.hypot(xs, ys)) if len(xs) else 1) + 0.35
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_title(title, fontsize=st.font)
    st.legend(axes[-1], [Patch(fc=cols[k], ec="#333333", lw=0.5, label=nm) for k, (nm, _) in enumerate(_HCLASS)],
              **_leg_out(st))
    st.suptitle(fig, f"{name + ': ' if name else ''}{seq}" if len(seq) <= 60 else name)


def _ex_pairs():
    return pd.DataFrame({
        "Sequence 1": ["ATGGCGTACGATCGTAGCTAGCTAGGCTAACGTTAGCCGATCGATCGGCTAGCTAAGCTTGCAGTCGATGCATGCTAGG",
                       "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"],
        "Sequence 2": ["ATGGCGTACGTTCGTAGCTAGCAGGCTAACGTTAGCCGTTCGATCGGCTAGGCTAAGCTTGCAGTCGATCATGCTAGG",
                       "MHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESF"],
        "Name": ["Gene fragment", "Haemoglobin α vs β (N-term)"]})


def _nw(a, b, match, mismatch, gap):
    n, m = len(a), len(b)
    S = np.zeros((n + 1, m + 1))
    S[:, 0] = np.arange(n + 1) * gap
    S[0, :] = np.arange(m + 1) * gap
    P = np.zeros((n + 1, m + 1), np.int8)    # 0 diag, 1 up (gap in b), 2 left (gap in a)
    P[1:, 0] = 1
    P[0, 1:] = 2
    bb = np.array(list(b))
    for i in range(1, n + 1):
        sub = np.where(bb == a[i - 1], match, mismatch)
        diag = S[i - 1, :-1] + sub
        up = S[i - 1, 1:] + gap
        best = np.maximum(diag, up)
        ptr = np.where(diag >= up, 0, 1)
        row = S[i]
        for j in range(1, m + 1):      # left moves depend on the same row
            lv = row[j - 1] + gap
            if lv > best[j - 1]:
                row[j] = lv
                P[i, j] = 2
            else:
                row[j] = best[j - 1]
                P[i, j] = ptr[j - 1]
    i, j, A, B = n, m, [], []
    while i > 0 or j > 0:
        p = P[i, j]
        if i > 0 and j > 0 and p == 0:
            A.append(a[i - 1]); B.append(b[j - 1]); i -= 1; j -= 1   # noqa: E702
        elif i > 0 and (p == 1 or j == 0):
            A.append(a[i - 1]); B.append("-"); i -= 1   # noqa: E702
        else:
            A.append("-"); B.append(b[j - 1]); j -= 1   # noqa: E702
    return "".join(A[::-1]), "".join(B[::-1]), S[n, m]


@plot("pairwise_alignment_spiral_125", "Pairwise sequence alignment (spiral)", G, number="125", example=_ex_pairs,
      roles=[Role("a", "Sequence 1", "any"), Role("b", "Sequence 2", "any"),
             Role("name", "Name (optional)", "cat", optional=True)],
      opts=[Opt("row", "Row to align", "int", 1, lo=1, hi=1e6),
            Opt("match", "Match score", "float", 2, lo=-100, hi=100), Opt("mis", "Mismatch score", "float", -1,
                                                                          lo=-100, hi=100),
            Opt("gap", "Gap score", "float", -2, lo=-100, hi=0), Opt("max", "Max length aligned", "int", 400,
                                                                     lo=10, hi=3000),
            Opt("layout", "Layout", "choice", "spiral", choices=["spiral", "rows"])],
      desc="One pair of sequences (DNA or protein) per row; they are globally aligned (Needleman–Wunsch) and "
           "drawn along a spiral with matches, mismatches and gaps coloured.", size=(8, 8), axis_labels=False)
def pairwise_alignment_spiral_125(fig, df, c, o, st):
    rows = df[[c["a"], c["b"]]].dropna()
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    st.bare(ax)
    if not len(rows):
        return
    i = int(min(max(o["row"], 1), len(rows))) - 1
    clean = lambda s: "".join(ch for ch in str(s).upper() if ch.isalpha())[:int(o["max"])]  # noqa: E731
    a, b = clean(rows.iloc[i, 0]), clean(rows.iloc[i, 1])
    name = str(df.loc[rows.index[i], c["name"]]) if c.get("name") else ""
    A, B, score = _nw(a, b, o["match"], o["mis"], o["gap"])
    n = len(A)
    stat = ["match" if x == y else "gap" if "-" in (x, y) else "mismatch" for x, y in zip(A, B)]
    cnt = Counter(stat)
    ident = 100 * cnt["match"] / max(n, 1)
    scol = {"match": st.color(2), "mismatch": st.color(0), "gap": "#bdbdbd"}
    if o["layout"] == "spiral":
        lane, sp, pitch, r0 = 0.48, 0.95, 2.6, 3.2
        th, r, pts = 0.0, r0, []
        for k in range(n):
            pts.append((th, r))
            th += sp / r
            r = r0 + pitch * th / (2 * np.pi)
        th_ = np.array([p[0] for p in pts])
        rr = np.array([p[1] for p in pts])
        ang = np.pi / 2 - th_
        ux, uy = np.cos(ang), np.sin(ang)
        tt = np.linspace(0, th_[-1] if n else 1, 800)
        rs = r0 + pitch * tt / (2 * np.pi)
        ax.plot(rs * np.cos(np.pi / 2 - tt), rs * np.sin(np.pi / 2 - tt), color="#e0e0e0", lw=1, zorder=0)
        P1 = np.c_[(rr + lane) * ux, (rr + lane) * uy]
        P2 = np.c_[(rr - lane) * ux, (rr - lane) * uy]
        R = rr.max() + 1.6 if n else 5
        rot = np.degrees(ang) - 90
    else:
        per = 30
        P1 = np.array([(k % per, -(k // per) * 3.2) for k in range(n)], float)
        P2 = P1 - [0, 1.0]
        P1 = P1 + [0, 0.0]
        rot = np.zeros(n)
        R = None
    fw = fig.get_size_inches()[0] * 72 * 0.8
    span = 2 * R if R else max(per if n else 1, 1) + 2
    ppu = fw / span
    fs = float(np.clip(ppu * 0.55, 3, st.font * 1.1))
    rad = 0.36
    for k in range(n):
        for (x, y), ch in ((P1[k], A[k]), (P2[k], B[k])):
            ax.add_patch(Circle((x, y), rad, fc=scol[stat[k]], ec="none", alpha=0.9 if ch != "-" else 0.5,
                                zorder=2))
            ax.text(x, y, ch, ha="center", va="center", fontsize=fs, color="white" if ch != "-" else "#555555",
                    fontweight="bold", rotation=rot[k], rotation_mode="anchor", zorder=3)
        if stat[k] == "match":
            ax.plot([P1[k][0], P2[k][0]], [P1[k][1], P2[k][1]], color=scol["match"], lw=1, zorder=1)
    if R:
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.text(0, 0.4, f"{ident:.1f}%", ha="center", va="center", fontsize=st.font * 1.6, fontweight="bold")
        ax.text(0, -0.7, "identity", ha="center", va="center", fontsize=st.font * 0.85, color="#555555")
        ax.annotate("start", P1[0] + [0, 0.9], ha="center", fontsize=st.font * 0.7, color="#555555")
        ax.text(-R + 0.3, R - 0.3, "outer: sequence 1\ninner: sequence 2", ha="left", va="top",
                fontsize=st.font * 0.7, color="#555555")
    else:
        ax.set_xlim(-1, per)
        ax.set_ylim(P2[-1][1] - 1 if n else -2, 1)
    st.legend(ax, [Patch(fc=scol[k], label=f"{k} ({cnt.get(k, 0)})") for k in ("match", "mismatch", "gap")],
              loc="upper right" if R else "lower right")
    st.labels(ax, "", "", f"{name + ' — ' if name else ''}identity {ident:.1f}% "
              f"({cnt['match']}/{n}), score {score:g}")


def _ex_maf():
    r = rng(134)
    genes = {"TP53": 0.52, "PIK3CA": 0.34, "TTN": 0.30, "MUC16": 0.22, "KRAS": 0.20, "PTEN": 0.16, "CDH1": 0.14,
             "GATA3": 0.13, "ARID1A": 0.12, "KMT2C": 0.12, "APC": 0.10, "BRCA2": 0.09, "NF1": 0.09, "RB1": 0.08,
             "FAT4": 0.08, "SYNE1": 0.07, "LRP1B": 0.07, "EGFR": 0.06, "CSMD3": 0.06, "FLG": 0.05}
    vc = ["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site",
          "In_Frame_Del"]
    pw = [0.62, 0.11, 0.09, 0.06, 0.08, 0.04]
    samples = [f"TCGA-{chr(65 + i // 10)}{i % 10}-{1000 + i * 7}" for i in range(40)]
    burden = r.gamma(2, 1, len(samples))
    burden /= burden.mean()
    rows = []
    for si, s in enumerate(samples):
        for g, f in genes.items():
            if r.random() < min(f * (0.6 + 0.4 * burden[si]), 0.95):
                for _ in range(2 if r.random() < 0.08 else 1):
                    v = vc[int(r.choice(len(vc), p=pw))]
                    rows.append((g, v, s))
        for _ in range(int(r.poisson(12 * burden[si]))):
            v = vc[int(r.choice(len(vc), p=pw))]
            rows.append((f"GENE{int(r.integers(1, 6000)):04d}", v, s))
    d = pd.DataFrame(rows, columns=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"])
    vt = {"Frame_Shift_Del": "DEL", "In_Frame_Del": "DEL", "Frame_Shift_Ins": "INS"}
    d["Variant_Type"] = [vt.get(v, "DNP" if r.random() < 0.03 else "SNP") for v in d["Variant_Classification"]]
    return d


_MAF_ROLES = [Role("gene", "Gene (Hugo_Symbol)", "cat"), Role("vc", "Variant_Classification", "cat"),
              Role("sample", "Sample (Tumor_Sample_Barcode)", "cat")]


def _maf_frame(df, c):
    d = pd.DataFrame({"g": as_str(df, c["gene"]), "v": as_str(df, c["vc"]), "s": as_str(df, c["sample"])})
    return d[(d["g"] != "(blank)") & (d["s"] != "(blank)")]


@plot("maf_summary_134", "MAF summary", G, number="134", example=_ex_maf,
      roles=_MAF_ROLES + [Role("vt", "Variant_Type (optional)", "cat", optional=True)],
      opts=[Opt("top", "Top mutated genes", "int", 10, lo=1, hi=100)],
      desc="A MAF-like table, one row per mutation: Hugo_Symbol, Variant_Classification, Tumor_Sample_Barcode "
           "and (optional) Variant_Type.", size=(13, 7.5))
def maf_summary_134(fig, df, c, o, st):
    d = _maf_frame(df, c)
    d["t"] = as_str(df, c["vt"]) if c.get("vt") else "n/a"
    vcs = list(d["v"].value_counts().index)
    col = dict(zip(vcs, st.colors(len(vcs))))
    gs = fig.add_gridspec(2, 3)
    fs = st.font * 0.8

    def title(ax, t):
        ax.set_title(t, fontsize=st.font, fontweight="bold", loc="left")
    a = st.ax(fig.add_subplot(gs[0, 0]), "x")
    vcnt = d["v"].value_counts().iloc[::-1]
    a.barh(range(len(vcnt)), vcnt.values, color=[col[v] for v in vcnt.index], height=0.7)
    a.set_yticks(range(len(vcnt)), [v.replace("_", " ") for v in vcnt.index], fontsize=fs)
    title(a, "Variant classification")
    a = st.ax(fig.add_subplot(gs[0, 1]), "x")
    if not c.get("vt"):
        a.set_axis_off()
        a.text(0.5, 0.5, "no Variant_Type column", ha="center", va="center", color="#888888",
               fontsize=fs, transform=a.transAxes)
    tcnt = d["t"].value_counts().iloc[::-1] if c.get("vt") else pd.Series(dtype=int)
    a.barh(range(len(tcnt)), tcnt.values, color=st.colors(len(tcnt))[::-1], height=0.6)
    a.set_yticks(range(len(tcnt)), tcnt.index, fontsize=fs)
    title(a, "Variant type")
    a = st.ax(fig.add_subplot(gs[0, 2]), "y")
    per = pd.crosstab(d["s"], d["v"]).reindex(columns=vcs, fill_value=0)
    per = per.loc[per.sum(1).sort_values(ascending=False).index]
    bottom = np.zeros(len(per))
    for v in vcs:
        a.bar(range(len(per)), per[v], bottom=bottom, color=col[v], width=0.85)
        bottom += per[v].to_numpy()
    med = float(np.median(per.sum(1))) if len(per) else 0
    a.axhline(med, color="#333333", ls="--", lw=0.9)
    a.text(len(per) - 0.5, med, f"median {med:g}", ha="right", va="bottom", fontsize=fs * 0.9)
    a.set_xticks([])
    a.set_xlim(-0.6, len(per) - 0.4)
    title(a, "Variants per sample")
    a.set_xlabel(f"{len(per)} samples", fontsize=fs)
    a = st.ax(fig.add_subplot(gs[1, 0]), "y")
    bp = a.boxplot([per[v].to_numpy() for v in vcs], patch_artist=True, widths=0.6, showfliers=True,
                   flierprops=dict(marker="o", ms=2.5, mfc="#999999", mec="none"),
                   medianprops=dict(color="black"))
    for p_, v in zip(bp["boxes"], vcs):
        p_.set_facecolor(col[v])
        p_.set_edgecolor("#333333")
    a.set_xticks(range(1, len(vcs) + 1), [v.replace("_", " ") for v in vcs], rotation=35, ha="right", fontsize=fs)
    title(a, "Variant classification summary")
    a.set_ylabel("Variants per sample", fontsize=fs)
    a = st.ax(fig.add_subplot(gs[1, 1]), "x")
    ns = d["s"].nunique()
    top = d.groupby("g")["s"].nunique().sort_values(ascending=False).head(int(o["top"]))
    gv = pd.crosstab(d["g"], d["v"]).reindex(index=top.index, columns=vcs, fill_value=0).iloc[::-1]
    left = np.zeros(len(gv))
    for v in vcs:
        a.barh(range(len(gv)), gv[v], left=left, color=col[v], height=0.7)
        left += gv[v].to_numpy()
    for k, g in enumerate(gv.index):
        a.text(left[k], k, f" {100 * top[g] / ns:.0f}%", va="center", fontsize=fs * 0.9)
    a.set_yticks(range(len(gv)), gv.index, fontsize=fs, fontstyle="italic")
    a.margins(x=0.15)
    title(a, f"Top {len(gv)} mutated genes")
    a.set_xlabel("Variants (label: % samples)", fontsize=fs)
    lax = fig.add_subplot(gs[1, 2])
    lax.set_axis_off()
    st.legend(lax, [Patch(fc=col[v], label=v.replace("_", " ")) for v in vcs], title="Variant classification",
              loc="center")
    st.suptitle(fig, f"MAF summary — {len(d)} variants in {ns} samples")


@plot("maf_oncoplot_135", "Oncoplot (MAF waterfall)", G, number="135", example=_ex_maf, roles=_MAF_ROLES,
      opts=[Opt("top", "Top N genes", "int", 15, lo=1, hi=100),
            Opt("names", "Show sample names", "bool", False),
            Opt("bg", "Background (no mutation)", "color", "#ececec")],
      desc="A MAF-like table, one row per mutation: Hugo_Symbol, Variant_Classification, Tumor_Sample_Barcode; "
           "samples with several different mutations in a gene are shown as Multi_Hit.", size=(12, 7))
def maf_oncoplot_135(fig, df, c, o, st):
    d = _maf_frame(df, c)
    vcs = list(d["v"].value_counts().index)
    col = dict(zip(vcs, st.colors(len(vcs))))
    col["Multi_Hit"] = "#1a1a1a"
    samples = cats(d, "s")
    ns = len(samples)
    freq = d.groupby("g")["s"].nunique().sort_values(ascending=False)
    genes = list(freq.index[:int(o["top"])])
    cell = d[d["g"].isin(genes)].groupby(["g", "s"])["v"].agg(lambda x: x.iloc[0] if x.nunique() == 1
                                                              else "Multi_Hit")
    M = np.zeros((len(genes), ns), bool)
    si = {s: k for k, s in enumerate(samples)}
    gi = {g: k for k, g in enumerate(genes)}
    for (g, s), v in cell.items():
        M[gi[g], si[s]] = True
    keys = [M[k] for k in range(len(genes) - 1, -1, -1)]
    order = np.lexsort(keys)[::-1] if len(genes) else np.arange(ns)
    order = list(order)
    sorder = [samples[k] for k in order]
    xs = {s: k for k, s in enumerate(sorder)}
    gs = fig.add_gridspec(2, 2, width_ratios=[6, 1], height_ratios=[1, 4])
    at = st.ax(fig.add_subplot(gs[0, 0]), "y")
    am = fig.add_subplot(gs[1, 0], sharex=at)
    ar = st.ax(fig.add_subplot(gs[1, 1], sharey=am), "x")
    per = pd.crosstab(d["s"], d["v"]).reindex(index=sorder, columns=vcs, fill_value=0)
    bottom = np.zeros(ns)
    for v in vcs:
        at.bar(range(ns), per[v], bottom=bottom, color=col[v], width=0.85)
        bottom += per[v].to_numpy()
    at.set_ylabel("TMB\n(variants)", fontsize=st.font * 0.8)
    at.tick_params(labelbottom=False, bottom=False)
    at.spines["bottom"].set_visible(False)
    for g in genes:
        for s in sorder:
            v = cell.get((g, s))
            am.add_patch(Rectangle((xs[s] - 0.45, gi[g] - 0.42), 0.9, 0.84,
                                   fc=o["bg"] if v is None else col.get(v, "#999999"), ec="none"))
    am.set_xlim(-0.6, ns - 0.4)
    am.set_ylim(len(genes) - 0.5, -0.5)
    am.set_yticks(range(len(genes)), genes, fontsize=st.font * 0.8, fontstyle="italic")
    for s_ in am.spines.values():
        s_.set_visible(False)
    am.tick_params(axis="y", length=0)
    if o["names"]:
        am.set_xticks(range(ns), sorder, rotation=90, fontsize=st.font * 0.55)
    else:
        am.set_xticks([])
    gv = pd.crosstab(d["g"], d["v"]).reindex(index=genes, columns=vcs, fill_value=0)
    pct = freq.reindex(genes) / ns * 100
    left = np.zeros(len(genes))
    for v in vcs:
        frac = gv[v] / gv.sum(1).replace(0, 1) * pct
        ar.barh(range(len(genes)), frac, left=left, color=col[v], height=0.8)
        left += frac.to_numpy()
    for k, p_ in enumerate(pct):
        ar.text(p_, k, f" {p_:.0f}%", va="center", fontsize=st.font * 0.7)
    ar.tick_params(labelleft=False, left=False)
    ar.set_xlabel("% samples", fontsize=st.font * 0.8)
    ar.margins(x=0.3)
    used = [v for v in vcs + ["Multi_Hit"] if v in set(cell.values)]
    hs = [Patch(fc=col[v], label=v.replace("_", " ")) for v in used]
    st.legend(am, hs, ncol=min(4, len(hs)), **_leg_out(st, "below"))
    st.suptitle(fig, f"Altered in {int(M.any(0).sum())} of {ns} samples ({100 * M.any(0).mean():.1f}%)")
    if st.opts.get("xlabel"):
        am.set_xlabel(st.opts["xlabel"], fontsize=st.font)
