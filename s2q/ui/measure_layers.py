"""Window: apply one mask to other images (layers) and measure every cell's intensity there.

Typical use: segment the cells in one channel / image, then read the integrated intensity of the
same cells in the other channels, time points or stainings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                             QListWidget, QListWidgetItem, QMessageBox, QPushButton, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from s2q.ui.line_profile import themed_window


def short_name(name: str, keep: int = 24) -> str:
    """'3 · long_file_name_part20_r3c4.tif' → 'L3 …name_part20_r3c4' (table / Excel column names)."""
    num, _, rest = name.partition(" · ")
    if not rest:
        num, rest = "", name
    stem = rest
    for ext in (".ome.tiff", ".ome.tif", ".ome.zarr", ".tiff", ".tif", ".png", ".jpg", ".jpeg", ".nd2", ".lif",
                ".czi", ".lsm", ".bmp"):
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    if len(stem) > keep:
        stem = "…" + stem[-(keep - 1):]
    return f"L{num} {stem}" if num.isdigit() else stem


class MeasureOnImagesWindow(QWidget):
    """*masks*: [(name, callable → label array)]; *images*: [(name, callable → [(channel, array)])]."""

    def __init__(self, parent, masks: list, images: list, default_path: Path, info: dict | None = None,
                 pixel_size: float | None = None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Measure a mask on other images")
        self.resize(1000, 720)
        themed_window(self)
        self._masks, self._images = masks, images
        self._default_path, self._info = Path(default_path), dict(info or {})
        self._cells = self._summary = None
        self._pixel_size = pixel_size       # µm per pixel → area in µm² and centre in µm, or None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        intro = QLabel("The cells of the chosen mask are laid on each ticked image; for every cell and channel "
                       "you get the mean and the integrated intensity (sum of the original pixel values in the cell).")
        intro.setWordWrap(True)
        root.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Mask (cells):"))
        self.mask_combo = QComboBox()
        for name, _ in masks:
            self.mask_combo.addItem(name)
        row.addWidget(self.mask_combo, 1)
        root.addLayout(row)

        root.addWidget(QLabel("Measure on these images (all their channels):"))
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(140)
        for name, _ in images:
            it = QListWidgetItem(name)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            self.image_list.addItem(it)
        root.addWidget(self.image_list)

        opt = QHBoxLayout()
        self.bg_chk = QCheckBox("Subtract background (median of the pixels outside all cells)")
        self.bg_chk.setChecked(True)
        opt.addWidget(self.bg_chk, 1)
        self.measure_btn = QPushButton("📊 Measure")
        self.measure_btn.setProperty("variant", "primary")
        self.measure_btn.clicked.connect(self.measure)
        opt.addWidget(self.measure_btn)
        root.addLayout(opt)

        self.tabs = QTabWidget()
        self.cell_table, self.sum_table = QTableWidget(), QTableWidget()
        for t in (self.cell_table, self.sum_table):
            t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tabs.addTab(self.sum_table, "Summary per image / channel")
        self.tabs.addTab(self.cell_table, "Per cell")
        root.addWidget(self.tabs, 1)

        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        root.addWidget(self.result_lbl)
        save = QHBoxLayout()
        save.addStretch(1)
        self.xlsx_btn = QPushButton("Save Excel…")
        self.xlsx_btn.clicked.connect(lambda: self.save("xlsx"))
        self.csv_btn = QPushButton("Save CSV (per cell)…")
        self.csv_btn.clicked.connect(lambda: self.save("csv"))
        for b in (self.xlsx_btn, self.csv_btn):
            b.setEnabled(False)
            save.addWidget(b)
        root.addLayout(save)
        # the table (and what is saved) always follows the current choices
        self.mask_combo.currentIndexChanged.connect(self._remeasure)
        self.image_list.itemChanged.connect(self._remeasure)
        self.bg_chk.toggled.connect(self._remeasure)

    def _remeasure(self, *_):
        if self._cells is not None:
            self.measure()

    def _chosen(self) -> list:
        return [self._images[i] for i in range(self.image_list.count())
                if self.image_list.item(i).checkState() == Qt.CheckState.Checked]

    # ── measuring ─────────────────────────────────────────────────────────
    def measure(self):
        from s2q.core.intensity import measure_on_images
        chosen = self._chosen()
        if not chosen:
            self.result_lbl.setText("Tick at least one image.")
            for t in (self.cell_table, self.sum_table):
                t.clear()
                t.setRowCount(0)
            for b in (self.xlsx_btn, self.csv_btn):
                b.setEnabled(False)
            return False
        m_name, m_get = self._masks[self.mask_combo.currentIndex()]
        try:
            mask = m_get()
            images = [(short_name(name), get()) for name, get in chosen]
            self._cells, self._summary = measure_on_images(mask, images, self.bg_chk.isChecked(),
                                                             pixel_size=self._pixel_size)
        except Exception as e:
            QMessageBox.critical(self, "Measuring failed", str(e))
            return False
        self._info = {k: v for k, v in self._info.items() if not k.startswith("Image L")}
        self._info.update({"Mask": m_name, "Background subtracted": "yes" if self.bg_chk.isChecked() else "no"})
        for name, _ in chosen:
            self._info[f"Image {short_name(name)}"] = name
        self._fill(self.cell_table, self._cells)
        self._fill(self.sum_table, self._summary)
        self.result_lbl.setText(f"{len(self._cells)} cells of “{m_name}” measured on {len(chosen)} image(s).")
        for b in (self.xlsx_btn, self.csv_btn):
            b.setEnabled(True)
        return True

    @staticmethod
    def _fill(table: QTableWidget, df):
        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r, row in enumerate(df.itertuples(index=False)):
            for c, v in enumerate(row):
                if isinstance(v, (float, np.floating)):
                    text = "" if np.isnan(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}")
                else:
                    text = str(v)
                it = QTableWidgetItem(text)
                if not isinstance(v, str):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(r, c, it)

    def save(self, kind: str):
        from s2q.core.intensity import save_measurements
        if not self.measure():               # always save what is ticked now, never older results
            return
        chosen = self._chosen()
        stem = self._default_path.name
        if len(chosen) == 1:                 # one image: its name in the file name, so files differ
            stem = short_name(chosen[0][0], keep=1000).split(" ", 1)[-1] + "_intensities"
        default = self._default_path.with_name(stem + "." + kind)
        flt = "Excel (*.xlsx)" if kind == "xlsx" else "CSV (*.csv)"
        path, _ = QFileDialog.getSaveFileName(self, "Save measurements", str(default), flt)
        if not path:
            return
        try:
            save_measurements(path, self._cells, self._summary, self._info)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        self.result_lbl.setText(f"Saved: {path}")
