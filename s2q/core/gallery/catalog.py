"""Order, short card names and menu filters of the gallery start screen.

``ORDER`` lists the plot numbers in the order of the original 7 pages (18 per page), so the card
grid shows the plots in the same order. ``CARD_NAMES`` are the short lower-case captions under the
cards. ``MENU`` is the left menu: section headers with keyword entries; an entry shows every plot
whose name, card caption or description contains one of its keywords.
"""
from __future__ import annotations

ORDER = [
    # page 1
    "002", "014", "024", "046", "083", "081", "017", "028", "009", "tidy", "011", "045", "114", "001",
    "020", "007", "018", "t001",
    # page 2
    "123", "021", "022", "023", "033", "025", "026", "027", "112", "029", "030", "031", "032", "088",
    "036", "037", "038", "039",
    # page 3
    "040", "041", "042", "043", "044", "091", "070", "047", "048", "049", "050", "005", "052", "053",
    "104", "055", "056", "057",
    # page 4
    "058", "059", "060", "061", "062", "063", "064", "065", "066", "067", "068", "069", "054", "071",
    "072", "073", "074", "075",
    # page 5
    "076", "077", "078", "092", "080", "103", "082", "051", "084", "085", "086", "087", "003", "089",
    "004", "006", "093", "094",
    # page 6
    "105", "096", "097", "098", "099", "100", "101", "102", "013", "016", "113", "106", "107", "108",
    "109", "110", "111", "079",
    # page 7
    "115", "117", "118", "119", "120", "121", "122", "095", "124", "125", "127", "128", "129", "134",
    "135", "008", "034", "010",
]

CARD_NAMES = {
    "002": "2D pie", "014": "gene up down bar", "024": "cluster heatmap", "046": "BP CC MF 3 in 1",
    "083": "circRNA-miRNA ring", "081": "enrichment bubble", "017": "alluvial plot, ceRNA",
    "028": "proportional venn", "009": "upsetR", "tidy": "tidy plot", "011": "waterfall chart",
    "045": "ridgeline plot, joyplot", "114": "violin", "001": "smooth line", "020": "3D pie",
    "007": "rectangle funnel", "018": "dendrogram", "t001": "element statistics",
    "123": "box with jitter", "021": "circle square area", "022": "half circle area", "023": "errorbar",
    "033": "pie matrix", "025": "errorbar", "026": "genomic peak venn", "027": "beeswarm box",
    "112": "PCA", "029": "expression connection lines", "030": "x4 square area", "031": "two tracks circos",
    "032": "dumbbell chart", "088": "m6A and expression log2fc", "036": "kernel density",
    "037": "dual Y bar", "038": "filled stackbar", "039": "flower plot",
    "040": "KM survival curve", "041": "regression scatter", "042": "QQplot", "043": "motif logo",
    "044": "scatter plot fold change", "091": "3d scatter", "070": "circular cluster heatmap",
    "047": "boxplot with lines", "048": "ballon plot", "049": "pearson spearman scatter",
    "050": "waffle chart", "005": "funnel plot", "052": "radar", "053": "scatter",
    "104": "scatter with marginal hist", "055": "line with marker", "056": "butterfly bar", "057": "area plot",
    "058": "3d area plot", "059": "area stack", "060": "line stack", "061": "vertical bar with trend",
    "062": "horizontal bar", "063": "treemap", "064": "matrix heatmap", "065": "vertical lollipop chart",
    "066": "horizontal lollipop chart", "067": "horizontal box plot", "068": "violin",
    "069": "histogram with fit", "054": "correlation dot", "071": "stem plot", "072": "polar bar",
    "073": "vertical bars", "074": "dot bars", "075": "vertical stack bars",
    "076": "horizontal stack bars", "077": "dot bar plot", "078": "bullet chart",
    "092": "pathway map add color", "080": "vertical 3d bars", "103": "venn diagram", "082": "correlation",
    "051": "donut chart", "084": "PCR, RNAseq dual Y", "085": "GO chord", "086": "volcano plot 3 colors",
    "087": "peak chromosome distribution", "003": "two layer pie", "089": "circle waffle",
    "004": "two layer donut", "006": "trapezoid funnel", "093": "raincloud", "094": "SNP likelihood",
    "105": "enrichment bar with color", "096": "triplex DNA", "097": "half violin", "098": "calendar heatmap",
    "099": "forest plot", "100": "two directional bars", "101": "expression trend", "102": "UPGMA cluster",
    "013": "m6A metaplot", "016": "multi groups bubble", "113": "violin with statistics", "106": "ROC curve",
    "107": "SNP density", "108": "peak chromosome distribution", "109": "peak distribution with centromere",
    "110": "Manhattan plot", "111": "GO KEGG bar dot", "079": "voronoi treemaps",
    "115": "cerebro colormap", "117": "bars with truncation", "118": "wordcloud", "119": "wenxiang diagram",
    "120": "miRanda miRNA targets prediction", "121": "Sankey and dot", "122": "GO pathway enrichment analysis",
    "095": "eccDNA chromosome distribution", "124": "pathway summary",
    "125": "paired sequence alignment sparials", "127": "species accumulation curves",
    "128": "multi types pie", "129": "circle packing", "134": "MAF summary plot", "135": "MAF oncoplot",
    "008": "two three category", "034": "Principal components analysis", "010": "quadrant scatter",
}

# left menu: (header, [(entry, [keywords])]); a header click shows every plot of its entries
MENU = [
    ("Basic plot", [("pie plot", ["pie", "donut", "waffle"]),
                    ("bar plot", ["bar", "lollipop", "stem", "waterfall", "bullet", "dumbbell"]),
                    ("line plot", ["line", "smooth", "area plot", "area stack", "3d area", "stacked area",
                                   "stacked line", "trend", "radar"]),
                    ("scatter plot", ["scatter", "quadrant"])]),
    ("Genome", [("SNP density", ["snp"]),
                ("Chrom distribution", ["chromosome", "chrom", "ideogram", "manhattan"]),
                ("Peak venn", ["peak venn", "genomic peak"]),
                ("Circos plot", ["circos"])]),
    ("Transcriptome", [("Heatmap", ["heatmap"]),
                       ("Volcano Plot", ["volcano", "cerebro"]),
                       ("Violin Plot", ["violin", "raincloud"]),
                       ("Bubble plot", ["bubble", "ballon", "balloon"]),
                       ("Chord plot", ["chord", "ring"])]),
    ("Epigenome", [("Metagene plot", ["metaplot", "metagene", "m6a"]),
                   ("Motif plot", ["motif", "logo"])]),
    ("Clinical plot", [("forest plot", ["forest"]),
                       ("KM plot", ["km ", "survival", "kaplan"]),
                       ("ROC curve", ["roc"])]),
    ("Miscellaneous", [("PCA", ["pca", "principal component"]),
                       ("Venn / UpSet", ["venn", "upset"]),
                       ("Sankey / alluvial", ["sankey", "alluvial"]),
                       ("Cluster / tree", ["cluster", "dendrogram", "upgma", "tree"]),
                       ("Box / distribution", ["box", "density", "histogram", "ridgeline", "qq"]),
                       ("Sequence / MAF", ["sequence", "maf", "oncoplot", "triplex", "wenxiang", "helical"])]),
]

PER_PAGE = 18


def ordered(registry) -> list:
    """All plot specs in the order of the original pages (any extra ones at the end)."""
    by_num = {s.number: s for s in registry.values()}
    out = [by_num[n] for n in ORDER if n in by_num]
    out += [s for s in registry.values() if s not in out]
    return out


def card_name(spec) -> str:
    return CARD_NAMES.get(spec.number, spec.name)


def matches(spec, keywords, in_desc: bool = False) -> bool:
    """Keyword at the start of a word of the plot's name / caption (and description if *in_desc*)."""
    import re
    text = f" {spec.name} {card_name(spec)} {spec.id.replace('_', ' ')} "
    if in_desc:
        text += f" {spec.desc} "
    text = text.lower()
    return any(re.search(r"(?<![a-z0-9])" + re.escape(k.lower().strip()), text) for k in keywords)


def search(specs, text: str) -> list:
    """Search box: every word must be found in the name, caption, number or description."""
    words = text.lower().split()
    return [s for s in specs if all(matches(s, [w], in_desc=True) or w == s.number.lower() for w in words)]
