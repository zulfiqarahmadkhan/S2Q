"""Plotting page: compare groups of cells from a results folder, and SRPlots (126 plot types)."""
import io
import os
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from s2q.core.results import NUMERIC_METRICS
from s2q.ui.theme import COLORS, PALETTES
from s2q.ui.widgets import (
    browse_row, card, field_row, page_shell, remember_text, scrollable, section_header,
    show_error, variant,
)

_SERIES = ["#7c6aef", "#e5609a", "#3b9ee8", "#f0a030"]

# Saved figures always use this white "paper" style, whatever the app theme.
PRINT_COLORS = dict(PALETTES["light"], bg_card="#ffffff", bg_input="#ffffff", text="#1a1a1a",
                    text_secondary="#333333", plot_grid="#d0d0d0")


def _short(name: str, n: int = 28) -> str:
    return name if len(name) <= n else "…" + name[-(n - 1):]


class _LoadWorker(QThread):
    """Reads every per-image sheet of a results folder (can take a few seconds)."""
    done = pyqtSignal(object)
    error = pyqtSignal(str, str)

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder

    def run(self):
        import traceback
        from s2q.core.results import load_results_folder
        try:
            self.done.emit(load_results_folder(self.folder))
        except Exception as e:
            self.error.emit(str(e), traceback.format_exc())


def build_group_figure(df, metric: str, by: str, kind: str = "box", points: bool = True, unit: str = "cells",
                       C=None):
    """Box / violin plot of *metric* per group, styled for the current theme (or palette *C*)."""
    from matplotlib.figure import Figure
    C = C or COLORS
    d = df.dropna(subset=[metric])
    if unit == "images":
        d = d.groupby(list(dict.fromkeys([by, "image"])))[metric].mean().reset_index()
    names = sorted(d[by].astype(str).unique(), key=lambda s: (len(s), s))
    data = [d.loc[d[by].astype(str) == n, metric].to_numpy() for n in names]
    sub, grid = C["text_secondary"], C["plot_grid"]
    fig = Figure(figsize=(max(6.0, 1.1 * len(names) + 2), 5.2), facecolor=C["bg_card"], dpi=110,
                 layout="constrained")
    ax = fig.add_subplot(111)
    ax.set_facecolor(C["bg_input"])
    pos = np.arange(1, len(names) + 1)
    if kind == "violin" and all(len(v) > 1 for v in data):
        parts = ax.violinplot(data, positions=pos, showmedians=True, widths=0.8)
        for i, b in enumerate(parts["bodies"]):
            b.set_facecolor(_SERIES[i % len(_SERIES)])
            b.set_alpha(0.55)
        for k in ("cmedians", "cmins", "cmaxes", "cbars"):
            if k in parts:
                parts[k].set_color(C["text"])
    else:
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=not points)
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor(_SERIES[i % len(_SERIES)])
            b.set_alpha(0.55)
        for k in ("medians", "whiskers", "caps"):
            for line in bp[k]:
                line.set_color(C["text"])
    if points:
        rng = np.random.default_rng(0)
        for i, v in enumerate(data):
            show = v if len(v) <= 3000 else rng.choice(v, 3000, replace=False)
            ax.scatter(pos[i] + rng.uniform(-0.18, 0.18, len(show)), show, s=5 if unit == "cells" else 22,
                       color=C["text"], alpha=0.25 if unit == "cells" else 0.8, edgecolors="none")
    ax.set_xticks(pos)
    labels = [f"{_short(n)}\nn={len(v)}" for n, v in zip(names, data)]
    if max(len(n) for n in names) > 10:     # long names would overlap side by side
        ax.set_xticklabels(labels, fontsize=8, rotation=35, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(metric + (" (image mean)" if unit == "images" else ""), color=sub)
    ax.set_title(f"{metric} by {by}", color=C["text"], fontsize=12, pad=10)
    ax.tick_params(colors=sub, labelsize=8)
    ax.grid(True, axis="y", color=grid, linestyle=":", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color(grid)
    return fig


def _fig_to_pixmap(fig, max_w: int, max_h: int) -> QPixmap:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    pix = QPixmap.fromImage(QImage.fromData(buf.getvalue(), "PNG"))
    return pix.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


class PlottingPage(QWidget):
    open_in_viewer = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._group_fig = None
        self._loader: _LoadWorker | None = None
        self._df = None            # per-cell table of the loaded results folder
        page, col = page_shell("Plotting", "Compare the measurements of your cells, or make any of 126 plot types.")

        self.tabs = QTabWidget()
        # SRPlots first; it is built the first time the Plotting page is shown (keeps start-up fast)
        self._gallery_holder = QWidget()
        QVBoxLayout(self._gallery_holder).setContentsMargins(0, 0, 0, 0)
        self._gallery = None
        self._gallery_index = self.tabs.addTab(self._gallery_holder, "SRPlots")
        self.tabs.addTab(self._build_compare_tab(), "Compare groups")
        self.tabs.currentChanged.connect(self._ensure_gallery)
        col.addWidget(self.tabs)
        col.addStretch()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(page))

    def showEvent(self, ev):
        super().showEvent(ev)
        self._ensure_gallery(self.tabs.currentIndex())

    def _ensure_gallery(self, index: int):
        if index != self._gallery_index or self._gallery is not None:
            return
        from s2q.ui.pages.plot_gallery import PlotGalleryTab
        try:
            self._gallery = PlotGalleryTab(self)
        except Exception as e:
            show_error(self, "SRPlots", "SRPlots could not be opened.", str(e))
            return
        self._gallery_holder.layout().addWidget(self._gallery)
        self.tabs.setTabText(self._gallery_index, "SRPlots")

    def _save_fig(self, make_fig, ref_path: str, suffix: str):
        """*make_fig* builds the figure in the white print style (saved plots never use the dark theme)."""
        if make_fig is None:
            return
        ref = Path(ref_path) if ref_path else Path.home() / "plot"
        suggested = str(ref.with_name(ref.stem + suffix + ".png")) if ref.suffix else str(ref / f"plot{suffix}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", suggested,
                                              "PNG image (*.png);;SVG vector (*.svg);;PDF (*.pdf)")
        if not path:
            return
        try:
            make_fig().savefig(path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
            QMessageBox.information(self, "Saved", f"Plot saved to:\n{path}")
        except Exception as e:
            show_error(self, "Save failed", "The plot could not be saved.", str(e))

    # ══ results folder ═════════════════════════════════════════════════════

    def _results_card(self, tab_key: str):
        lay = QVBoxLayout()
        lay.addLayout(section_header(
            "Results folder", "An output folder of a segmentation run (all <image>_cp_masks.xlsx sheets)."))
        edit = QLineEdit()
        edit.setPlaceholderText("Output folder with the Excel sheets…")
        row = browse_row(edit, "Browse", lambda: self._pick_results(edit))
        load = variant(QPushButton("Load"), "primary")
        load.clicked.connect(lambda: self._load_results(edit.text().strip()))
        row.addWidget(load)
        lay.addLayout(field_row("Folder", row))
        info = QLabel("")
        info.setObjectName("Hint")
        info.setWordWrap(True)
        lay.addWidget(info)
        setattr(self, f"_res_edit_{tab_key}", edit)
        setattr(self, f"_res_info_{tab_key}", info)
        remember_text(edit, "plot/results")
        return card(lay)

    def _pick_results(self, edit):
        d = QFileDialog.getExistingDirectory(self, "Select results folder", edit.text())
        if d:
            edit.setText(d)
            self._load_results(d)

    def _load_results(self, folder: str):
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Choose an existing results folder.")
            return
        for k in ("compare",):
            getattr(self, f"_res_edit_{k}").setText(folder)
            getattr(self, f"_res_info_{k}").setText("Loading…")
        old = self._loader
        if old is not None and old.isRunning():
            self._old_loaders = [t for t in getattr(self, "_old_loaders", []) if t.isRunning()] + [old]
        ld = self._loader = _LoadWorker(folder)
        # only the newest load may change the data (an older, slower one would overwrite it)
        ld.done.connect(lambda df: self._on_loaded(folder, df) if ld is self._loader else None)
        ld.error.connect(lambda m, d: self._on_load_error(m, d) if ld is self._loader else None)
        ld.start()

    def _on_load_error(self, msg: str, details: str):
        for k in ("compare",):
            getattr(self, f"_res_info_{k}").setText("Loading failed.")
        show_error(self, "Cannot load", msg, details)

    def _on_loaded(self, folder: str, df):
        self._df = df
        if df is None or df.empty:
            msg = "No *_cp_masks.xlsx sheets in this folder — run the ROI analysis first."
        else:
            wells = sorted({w for w in df["well"] if w})
            msg = (f"{len(df):,} cells from {df['image'].nunique()} images"
                   + (f"; wells: {', '.join(wells)}" if wells else "") + ".")
        for k in ("compare",):
            getattr(self, f"_res_info_{k}").setText(msg)
        if df is not None and not df.empty:
            metrics = [m for m in NUMERIC_METRICS if m in df.columns]
            combo = self.metric_combo
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(metrics)
            idx = combo.findText(keep)
            combo.setCurrentIndex(max(idx, 0))
            combo.blockSignals(False)

    def _df_with_conditions(self):
        from s2q.core.results import add_condition, parse_condition_map
        return add_condition(self._df, parse_condition_map(self.cond_edit.text()))

    # ══ compare groups ════════════════════════════════════════════════════

    def _build_compare_tab(self):
        from s2q.core.results import GROUP_KEYS
        tab = QWidget()
        col = QVBoxLayout(tab)
        col.setContentsMargins(0, 14, 0, 0)
        col.setSpacing(16)
        col.addWidget(self._results_card("compare"))

        opt = QVBoxLayout()
        opt.addLayout(section_header("Comparison"))
        r1 = QHBoxLayout()
        r1.setSpacing(12)
        self.metric_combo = QComboBox()
        self.group_combo = QComboBox()
        for g in GROUP_KEYS:
            self.group_combo.addItem(g)
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(["Box plot", "Violin plot"])
        self.unit_combo = QComboBox()
        self.unit_combo.addItem("Cells", "cells")
        self.unit_combo.addItem("Image means", "images")
        r1.addLayout(field_row("Measurement", self.metric_combo), 2)
        r1.addLayout(field_row("Group by", self.group_combo), 1)
        r1.addLayout(field_row("Plot", self.kind_combo), 1)
        r1.addLayout(field_row("Statistical unit", self.unit_combo), 1)
        opt.addLayout(r1)
        self.cond_edit = QLineEdit()
        self.cond_edit.setPlaceholderText("e.g.  A6,A7=WT; A8-A10=mutant")
        opt.addLayout(field_row(
            "Conditions (for “Group by: condition”)", self.cond_edit,
            "Name groups of wells; wells not listed become “(unassigned)”."))
        remember_text(self.cond_edit, "plot/conditions")
        self.points_chk = QCheckBox("Show individual points")
        self.points_chk.setChecked(True)
        opt.addWidget(self.points_chk)
        btns = QHBoxLayout()
        go = variant(QPushButton("Compare"), "primary")
        go.clicked.connect(self._compare)
        self.group_save_btn = QPushButton("Save plot…")
        self.group_save_btn.setEnabled(False)
        self.group_save_btn.clicked.connect(
            lambda: self._save_fig(self._group_fig, self._res_edit_compare.text(), "_compare"))
        self.csv_btn = QPushButton("Export table (CSV)…")
        self.csv_btn.setEnabled(False)
        self.csv_btn.clicked.connect(self._export_csv)
        for b in (go, self.group_save_btn, self.csv_btn):
            btns.addWidget(b)
        btns.addStretch()
        opt.addLayout(btns)
        col.addWidget(card(opt))

        res = QVBoxLayout()
        res.addLayout(section_header("Result"))
        self.stats_lbl = QLabel("Load a results folder and press Compare.")
        self.stats_lbl.setObjectName("RunSummary")
        self.stats_lbl.setWordWrap(True)
        self.stats_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        res.addWidget(self.stats_lbl)
        self.group_table = QTableWidget(0, 0)
        self.group_table.setMinimumHeight(160)
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        res.addWidget(self.group_table)
        self.group_preview = QLabel("")
        self.group_preview.setObjectName("ImagePanel")
        self.group_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.group_preview.setMinimumHeight(420)
        res.addWidget(self.group_preview)
        col.addWidget(card(res))
        col.addStretch()   # keep cards at their natural height
        return tab

    def _compare(self):
        from s2q.core.results import compare, group_table
        if self._df is None or self._df.empty:
            QMessageBox.information(self, "No data", "Load a results folder first.")
            return
        df = self._df_with_conditions()
        metric, by = self.metric_combo.currentText(), self.group_combo.currentText()
        unit = self.unit_combo.currentData()
        if df[by].astype(str).nunique() < 2:
            QMessageBox.information(self, "One group only",
                                    f"Every cell has the same “{by}”. Choose another grouping"
                                    + (" or fill in Conditions." if by == "condition" else "."))
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._table = group_table(df, metric, by)
            lines = compare(df, metric, by, unit)
            args = (df, metric, by, "violin" if self.kind_combo.currentIndex() else "box",
                    self.points_chk.isChecked(), unit)
            shown = build_group_figure(*args)
            self._group_fig = lambda: build_group_figure(*args, C=PRINT_COLORS)
        finally:
            QApplication.restoreOverrideCursor()
        self.stats_lbl.setText("\n".join(lines))
        t = self._table
        self.group_table.setColumnCount(len(t.columns))
        self.group_table.setRowCount(len(t))
        self.group_table.setHorizontalHeaderLabels([str(c) for c in t.columns])
        for r in range(len(t)):
            for c, v in enumerate(t.iloc[r]):
                txt = f"{v:.4g}" if isinstance(v, float) else str(v)
                self.group_table.setItem(r, c, QTableWidgetItem(txt))
        self.group_table.resizeColumnsToContents()
        self.group_preview.setPixmap(_fig_to_pixmap(shown,max(500, self.group_preview.width() - 20), 560))
        self.group_save_btn.setEnabled(True)
        self.csv_btn.setEnabled(True)

    def _export_csv(self):
        folder = self._res_edit_compare.text().strip() or str(Path.home())
        name = f"{self.metric_combo.currentText()}_by_{self.group_combo.currentText()}.csv".replace(" ", "_")
        path, _ = QFileDialog.getSaveFileName(self, "Export table", str(Path(folder) / name), "CSV (*.csv)")
        if path:
            self._table.to_csv(path, index=False)
            QMessageBox.information(self, "Exported", f"Table saved to:\n{path}")
