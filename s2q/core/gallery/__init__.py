"""S2Q plot gallery: 126 plot types that work on any table (CSV / Excel).

Importing this package registers every plot type in ``REGISTRY``. Each module holds one family:
bars, shapes (lines / areas / pies / proportions), stats (distributions, scatter, clusters,
clinical), bio (transcriptome, sets & flows, genome & sequence).
"""
from s2q.core.gallery.base import (  # noqa: F401
    CATEGORIES, COMMON_OPTS, PALETTES, REGISTRY, PlotSpec, auto_columns, default_opts, render,
)
from s2q.core.gallery import bars, shapes, stats, bio  # noqa: F401,E402  (registration)


def by_category() -> dict[str, list[PlotSpec]]:
    out = {c: [] for c in CATEGORIES}
    for spec in REGISTRY.values():
        out[spec.category].append(spec)
    for c in out:
        out[c].sort(key=lambda s: s.name.lower())
    return {c: v for c, v in out.items() if v}
