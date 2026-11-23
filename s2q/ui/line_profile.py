"""Line-profile window of the correction editor: intensity of every channel along a drawn line."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

# channel colours: named RGB channels get their own colour, others follow this list
_PALETTE = ["#e5484d", "#30a46c", "#3b82f6", "#d946ef", "#06b6d4", "#f59e0b", "#64748b", "#111111"]
_NAMED = {"red": "#e5484d", "r": "#e5484d", "green": "#30a46c", "g": "#30a46c", "blue": "#3b82f6", "b": "#3b82f6",
          "intensity": "#111111"}


def themed_window(w: QWidget):
    """Top-level windows don't get the page background; give them the theme's background."""
    from s2q.ui.theme import COLORS
    w.setObjectName("ToolWindow")
    w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    w.setStyleSheet(f"QWidget#ToolWindow {{ background-color: {COLORS['bg_base']}; }} "
                    f"QWidget#ToolWindow QLabel {{ color: {COLORS['text']}; background: transparent; }}")


def channel_colour(name: str, i: int) -> str:
    return _NAMED.get(name.strip().lower().split()[-1], _PALETTE[i % len(_PALETTE)])


class LineProfileWindow(QWidget):
    """Plot + table of a line profile. Emits ``width_changed`` when the line width is changed."""
    width_changed = pyqtSignal(int)
    clear_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Line profile")
        self.resize(980, 860)
        themed_window(self)
        self._result = None
        self._source = ""

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)
        self.width_spin = self._spin(ctrl, "Line width", 1, 500, 1, " px",
                                     "Thickness of the line: pixels across it are averaged (plot) and summed "
                                     "(integrated intensity)")
        self.width_spin.valueChanged.connect(self.width_changed.emit)
        self.plot_w = self._spin(ctrl, "Plot width", 300, 4000, 820, " px", "Width of the plot (also when saving)")
        self.plot_h = self._spin(ctrl, "Plot height", 200, 3000, 380, " px", "Height of the plot (also when saving)")
        self.plot_w.valueChanged.connect(self._resize_plot)
        self.plot_h.valueChanged.connect(self._resize_plot)
        self.norm_chk = QCheckBox("Normalise each channel (0–1)")
        self.norm_chk.setToolTip("Compare the shape of channels with very different brightness")
        self.norm_chk.toggled.connect(lambda _: self._draw())
        ctrl.addWidget(self.norm_chk)
        self.cells_chk = QCheckBox("Shade cells")
        self.cells_chk.setChecked(True)
        self.cells_chk.setToolTip("Grey bands where the line crosses a cell of the mask")
        self.cells_chk.toggled.connect(lambda _: self._draw())
        ctrl.addWidget(self.cells_chk)
        ctrl.addStretch()
        root.addLayout(ctrl)

        # second row: legend placement + which channels to draw
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        row2.addWidget(QLabel("Legend"))
        self.legend_combo = QComboBox()
        for text, key in [("Inside (best place)", "best"), ("Outside, right", "outside"),
                          ("Below the plot", "below"), ("Off", "off")]:
            self.legend_combo.addItem(text, key)
        self.legend_combo.setToolTip("Move the legend off the curves, or hide it")
        self.legend_combo.currentIndexChanged.connect(lambda _: self._draw())
        row2.addWidget(self.legend_combo)
        row2.addSpacing(16)
        row2.addWidget(QLabel("Channels"))
        self._chan_box = QHBoxLayout()
        self._chan_box.setSpacing(8)
        row2.addLayout(self._chan_box)
        row2.addStretch()
        root.addLayout(row2)
        self._chan_checks: dict[str, QCheckBox] = {}

        self.fig = Figure(facecolor="white", layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        holder = QScrollArea()
        holder.setWidget(self.canvas)
        holder.setWidgetResizable(False)
        holder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        holder.setStyleSheet("QScrollArea { background: white; }")
        self._holder = holder
        root.addWidget(holder, 1)
        self._resize_plot()

        self.info = QLabel("Drag a line on the image with the Line profile tool (I).")
        self.info.setWordWrap(True)
        root.addWidget(self.info)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Channel", "Mean", "Min", "Max", "Integrated intensity", "Points"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(150)
        root.addWidget(self.table)

        btns = QHBoxLayout()
        save = QPushButton("Save plot…")
        save.clicked.connect(self._save_plot)
        csv_btn = QPushButton("Export CSV…")
        csv_btn.clicked.connect(self._export_csv)
        clear = QPushButton("Clear line")
        clear.clicked.connect(self.clear_requested.emit)
        for b in (save, csv_btn, clear):
            btns.addWidget(b)
        btns.addStretch()
        root.addLayout(btns)

    @staticmethod
    def _spin(layout, label, lo, hi, val, suffix, tip):
        lbl = QLabel(label)
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSuffix(suffix)
        s.setToolTip(tip)
        lbl.setToolTip(tip)
        layout.addWidget(lbl)
        layout.addWidget(s)
        return s

    # ── data ──────────────────────────────────────────────────────────────

    def set_result(self, result: dict, source: str = ""):
        self._result, self._source = result, source
        r = result
        n = len(r["distance"])
        self.info.setText(f"{source}\nLine length {r['length']:.1f} {r['unit']} · width {r['width']} px · "
                          f"{n} points · {len(r['channels'])} channel(s). Integrated intensity = sum of all "
                          f"pixel values inside the line.")
        self.table.setRowCount(len(r["channels"]))
        for row, ch in enumerate(r["channels"]):
            vals = [ch["name"], f"{ch['mean']:.4g}", f"{ch['min']:.4g}", f"{ch['max']:.4g}",
                    f"{ch['integrated']:.6g}", str(n)]
            for col, v in enumerate(vals):
                self.table.setItem(row, col, QTableWidgetItem(v))
        self._sync_channel_checks([c["name"] for c in r["channels"]])
        self._draw()

    def _sync_channel_checks(self, names: list[str]):
        """One tick box per channel; ticks are kept when the line or the time point changes."""
        if list(self._chan_checks) == names:
            return
        old = {k: cb.isChecked() for k, cb in self._chan_checks.items()}
        for cb in self._chan_checks.values():
            self._chan_box.removeWidget(cb)
            cb.deleteLater()
        self._chan_checks = {}
        for i, name in enumerate(names):
            cb = QCheckBox(name)
            cb.setChecked(old.get(name, True))
            cb.setStyleSheet(f"QCheckBox {{ color: {channel_colour(name, i)}; font-weight: 600; }}")
            cb.toggled.connect(lambda _: self._draw())
            self._chan_box.addWidget(cb)
            self._chan_checks[name] = cb

    def _resize_plot(self, *_):
        w, h = self.plot_w.value(), self.plot_h.value()
        self.canvas.setFixedSize(w, h)
        # show the whole plot (legend below it included) without scrolling, up to a sensible size
        self._holder.setMinimumHeight(min(h + 6, 900))
        if self.isVisible() and self.height() < self.sizeHint().height():
            self.resize(self.width(), min(self.sizeHint().height(), 1200))
        dpi = self.fig.get_dpi()
        self.fig.set_size_inches(w / dpi, h / dpi)
        self._draw()

    def _draw(self, fig=None):
        fig = fig or self.fig
        fig.clear()
        ax = fig.add_subplot(111)
        r = self._result
        if r is None:
            ax.text(0.5, 0.5, "Draw a line on the image (Line profile tool, I)", ha="center", va="center",
                    transform=ax.transAxes, color="#666666")
            ax.set_axis_off()
        else:
            d = r["distance"]
            if self.cells_chk.isChecked() and r.get("labels") is not None and len(d) > 1:
                inside = r["labels"] > 0
                edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
                for a, b in zip(edges[::2], edges[1::2]):
                    ax.axvspan(d[a], d[min(b, len(d) - 1)], color="#000000", alpha=0.07, lw=0)
            for i, ch in enumerate(r["channels"]):
                cb = self._chan_checks.get(ch["name"])
                if cb is not None and not cb.isChecked():
                    continue
                y = ch["profile"]
                if self.norm_chk.isChecked():
                    lo, hi = float(np.min(y)), float(np.max(y))
                    y = (y - lo) / (hi - lo) if hi > lo else y * 0
                ax.plot(d, y, color=channel_colour(ch["name"], i), lw=1.8, label=ch["name"])
            ax.set_xlabel(f"Distance along the line ({r['unit']})")
            ax.set_ylabel("Normalised intensity" if self.norm_chk.isChecked() else "Intensity (mean across the line width)")
            ax.set_xlim(d[0], d[-1] if len(d) > 1 else 1)
            ax.grid(True, color="#e5e5e5", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            where = self.legend_combo.currentData()
            if len(ax.lines) > 1 and where != "off":
                # figure legends with "outside …" are fitted by the layout, so they are never cut off
                if where == "outside":
                    fig.legend(frameon=False, fontsize=9, loc="outside right upper")
                elif where == "below":
                    fig.legend(frameon=False, fontsize=9, loc="outside lower center", ncol=min(len(ax.lines), 6))
                else:
                    ax.legend(frameon=True, framealpha=0.85, edgecolor="none", fontsize=9, loc="best")
        if fig is self.fig:
            self.canvas.draw_idle()

    # ── export ────────────────────────────────────────────────────────────

    def _default_path(self, ext: str) -> str:
        base = Path(self._source.split("\n")[0].replace("Image: ", "")).stem if self._source else "line"
        return str(Path.home() / f"{base}_line_profile.{ext}")

    def _save_plot(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", self._default_path("png"),
                                              "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;JPEG (*.jpg)")
        if not path:
            return
        from matplotlib.figure import Figure
        dpi = 300
        fig = Figure(figsize=(self.plot_w.value() / 100, self.plot_h.value() / 100), dpi=dpi,
                     facecolor="white", layout="constrained")
        self._draw(fig)
        fig.savefig(path, dpi=dpi, facecolor="white")
        QMessageBox.information(self, "Saved", f"Plot saved to:\n{path}")

    def _export_csv(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export line profile", self._default_path("csv"), "CSV (*.csv)")
        if not path:
            return
        r = self._result
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            names = [c["name"] for c in r["channels"]]
            w.writerow([f"distance_{r['unit']}"] + [f"{n} (mean across width)" for n in names]
                       + (["cell_label"] if r.get("labels") is not None else []))
            for i, dist in enumerate(r["distance"]):
                row = [f"{dist:.4f}"] + [f"{c['profile'][i]:.6g}" for c in r["channels"]]
                if r.get("labels") is not None:
                    row.append(int(r["labels"][i]))
                w.writerow(row)
            w.writerow([])
            w.writerow(["channel", "mean", "min", "max", "integrated_intensity", "line_length", "line_width_px"])
            for c in r["channels"]:
                w.writerow([c["name"], f"{c['mean']:.6g}", f"{c['min']:.6g}", f"{c['max']:.6g}",
                            f"{c['integrated']:.6g}", f"{r['length']:.3f} {r['unit']}", r["width"]])
        QMessageBox.information(self, "Exported", f"Line profile saved to:\n{path}")
