"""Plot gallery tab: 126 plot types for any table (CSV / Excel), with example data for each."""
from __future__ import annotations

import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSpinBox,
    QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from s2q.ui.widgets import card, remember_text, section_header, settings, show_error, variant

SAVE_FILTER = ("PNG image (*.png);;TIFF image (*.tif);;SVG vector (*.svg);;PDF vector (*.pdf);;"
               "JPEG image (*.jpg)")
_NONE = "(none)"


def _combo() -> QComboBox:
    """A drop-down that does not grow to its longest entry (long column names would widen the page)."""
    w = QComboBox()
    w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    w.setMinimumContentsLength(10)
    w.view().setMinimumWidth(240)       # the opened list still shows long entries in full
    return w


def read_table(path: str, sheet: str | None = None):
    """CSV / TSV / TXT (separator detected) or an Excel sheet → DataFrame."""
    import pandas as pd
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(p, sheet_name=sheet or 0)
    if suf in (".tsv", ".tab"):
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p, sep=None, engine="python")


class PlotEditor(QWidget):
    """One plot type: data, columns, options, preview and saving. ``back`` returns to the cards."""
    back = pyqtSignal()

    def __init__(self, gallery, parent=None):
        super().__init__(parent)
        self.G = gallery
        self._spec = None
        self._df = None                 # the user's table (None = use the plot's example data)
        self._example = None
        self._role_widgets: dict = {}
        self._opt_widgets: dict = {}
        self._canvas = None
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(350)
        self._redraw_timer.timeout.connect(self.draw)
        self._build()

    # ── layout ────────────────────────────────────────────────────────────
    def _build(self):
        rl = QVBoxLayout(self)
        rl.setContentsMargins(0, 14, 0, 0)
        rl.setSpacing(12)

        head_row = QHBoxLayout()
        back_btn = QPushButton("←  All plots")
        back_btn.setToolTip("Back to the plot cards")
        back_btn.clicked.connect(self.back.emit)
        head_row.addWidget(back_btn, 0, Qt.AlignmentFlag.AlignTop)
        head = QVBoxLayout()
        self.title_lbl = QLabel("")
        self.title_lbl.setObjectName("SectionTitle")
        self.title_lbl.setStyleSheet("font-size: 17px; font-weight: 700;")
        self.title_lbl.setWordWrap(True)
        self.desc_lbl = QLabel("")
        self.desc_lbl.setObjectName("Hint")
        self.desc_lbl.setWordWrap(True)
        head.addWidget(self.title_lbl)
        head.addWidget(self.desc_lbl)
        head_row.addLayout(head, 1)
        rl.addLayout(head_row)

        # data
        dl = QVBoxLayout()
        dl.addLayout(section_header("Data", "Your table (CSV, TSV, TXT or Excel) — or the example data of this "
                                            "plot, which also shows the layout the plot expects."))
        row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Table file…  (empty = example data)")
        self.file_edit.editingFinished.connect(lambda: self._load(self.file_edit.text().strip()))
        row.addWidget(self.file_edit, 1)
        b = QPushButton("Browse")
        b.clicked.connect(self._browse)
        row.addWidget(b)
        self.sheet_combo = QComboBox()
        self.sheet_combo.setMinimumWidth(120)
        self.sheet_combo.setToolTip("Excel sheet")
        self.sheet_combo.hide()
        self.sheet_combo.currentTextChanged.connect(self._on_sheet)
        row.addWidget(self.sheet_combo)
        dl.addLayout(row)
        row2 = QHBoxLayout()
        self.example_btn = QPushButton("Use example data")
        self.example_btn.clicked.connect(self._use_example)
        row2.addWidget(self.example_btn)
        tmpl = QPushButton("Save example as template…")
        tmpl.setToolTip("Save this plot's example table (CSV / Excel) to fill in with your own values")
        tmpl.clicked.connect(self._save_template)
        row2.addWidget(tmpl)
        self.data_lbl = QLabel("")
        self.data_lbl.setObjectName("Hint")
        row2.addWidget(self.data_lbl, 1)
        dl.addLayout(row2)
        self.table = QTableWidget()
        self.table.setMaximumHeight(150)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        dl.addWidget(self.table)
        data_card = card(dl)

        # columns + options side by side
        co = QHBoxLayout()
        co.setSpacing(12)
        cl = QVBoxLayout()
        cl.addLayout(section_header("Columns", "Which column of the table plays which part."))
        self.roles_form = QFormLayout()
        self.roles_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        cl.addLayout(self.roles_form)
        cl.addStretch(1)
        co.addWidget(card(cl), 1)
        ol = QVBoxLayout()
        ol.addLayout(section_header("Options"))
        self.opts_grid = QGridLayout()
        self.opts_grid.setHorizontalSpacing(10)
        ol.addLayout(self.opts_grid)
        ol.addStretch(1)
        co.addWidget(card(ol), 2)

        # preview + save
        pl = QVBoxLayout()
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Size (inches)"))
        self.w_spin, self.h_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for sp in (self.w_spin, self.h_spin):
            sp.setRange(2, 30)
            sp.setSingleStep(0.5)
            sp.setDecimals(1)
            sp.valueChanged.connect(self._schedule)
            bar.addWidget(sp)
        bar.addWidget(QLabel("Save DPI"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)
        bar.addWidget(self.dpi_spin)
        self.auto_chk = QCheckBox("Redraw automatically")
        self.auto_chk.setChecked(True)
        bar.addWidget(self.auto_chk)
        bar.addStretch(1)
        draw_btn = variant(QPushButton("Draw"), "primary")
        draw_btn.clicked.connect(self.draw)
        bar.addWidget(draw_btn)
        save_btn = QPushButton("Save figure…")
        save_btn.clicked.connect(self._save_figure)
        bar.addWidget(save_btn)
        pl.addLayout(bar)
        self.msg_lbl = QLabel("")
        self.msg_lbl.setWordWrap(True)
        pl.addWidget(self.msg_lbl)
        self.canvas_holder = QVBoxLayout()
        holder = QWidget()
        holder.setLayout(self.canvas_holder)
        holder.setMinimumHeight(420)
        holder.setStyleSheet("background: white; border-radius: 8px;")
        # the panel takes the width it is given; the plot is fitted into it (never the other way round)
        from PyQt6.QtWidgets import QSizePolicy
        holder.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._holder = holder
        self._holder_w = 0
        holder.installEventFilter(self)          # redraw at the new size when the panel width changes
        pl.addWidget(holder, 1)
        # order on screen: the plot first, then its data, columns and options underneath
        rl.addWidget(card(pl), 1)
        rl.addWidget(data_card)
        rl.addLayout(co)
        remember_text(self.file_edit, "gallery/file")
        if self.file_edit.text().strip():
            QTimer.singleShot(0, lambda: self._load(self.file_edit.text().strip(), quiet=True))

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if obj is getattr(self, "_holder", None) and event.type() == QEvent.Type.Resize:
            w = obj.width()
            if self._spec is not None and abs(w - self._holder_w) > 40:
                self._holder_w = w
                self._redraw_timer.start()
        return super().eventFilter(obj, event)

    # ── choosing a plot ───────────────────────────────────────────────────
    def select(self, pid: str):
        self._select(pid)

    def _select(self, pid: str):
        spec = self.G.REGISTRY[pid]
        self._spec = spec
        self._example = spec.example()
        self.title_lbl.setText(f"{spec.name}   ·   {spec.category}")
        self.desc_lbl.setText(spec.desc or "")
        for sp, v in ((self.w_spin, spec.size[0]), (self.h_spin, spec.size[1])):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)
        self._build_opts()
        self._refresh_data()

    # ── data ──────────────────────────────────────────────────────────────
    def data(self):
        return self._df if self._df is not None else self._example

    def _browse(self):
        start = self.file_edit.text().strip() or settings().value("gallery/dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(self, "Table to plot", start,
                                              "Tables (*.csv *.tsv *.txt *.xlsx *.xls *.xlsm);;All (*)")
        if path:
            settings().setValue("gallery/dir", str(Path(path).parent))
            self.file_edit.setText(path)
            self._load(path)

    def _load(self, path: str, quiet: bool = False):
        if not path:
            self._use_example()
            return
        try:
            sheets = []
            if Path(path).suffix.lower() in (".xlsx", ".xlsm", ".xls"):
                import pandas as pd
                sheets = pd.ExcelFile(path).sheet_names
            self.sheet_combo.blockSignals(True)
            self.sheet_combo.clear()
            self.sheet_combo.addItems(sheets)
            self.sheet_combo.setVisible(len(sheets) > 1)
            self.sheet_combo.blockSignals(False)
            self._df = read_table(path, sheets[0] if sheets else None)
        except Exception as e:
            self._df = None
            if not quiet:
                show_error(self, "Cannot read the table", f"{Path(path).name} could not be read.", str(e))
        self._refresh_data()

    def _on_sheet(self, name: str):
        if name and self.file_edit.text().strip():
            try:
                self._df = read_table(self.file_edit.text().strip(), name)
            except Exception as e:
                show_error(self, "Cannot read the sheet", name, str(e))
                return
            self._refresh_data()

    def _use_example(self):
        self._df = None
        self.file_edit.clear()
        self.sheet_combo.hide()
        self._refresh_data()

    def _save_template(self):
        if self._spec is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save example table", f"{self._spec.id}_example.csv",
                                              "CSV (*.csv);;Excel (*.xlsx)")
        if not path:
            return
        try:
            if path.lower().endswith(".xlsx"):
                self._example.to_excel(path, index=False)
            else:
                self._example.to_csv(path, index=False)
            self.data_lbl.setText(f"Template saved: {path}")
        except Exception as e:
            show_error(self, "Save failed", "The template could not be saved.", str(e))

    def _refresh_data(self):
        df = self.data()
        if df is None:
            return
        src = "example data" if self._df is None else Path(self.file_edit.text()).name
        self.data_lbl.setText(f"{src}: {len(df):,} rows × {len(df.columns)} columns")
        show = df.head(100)
        self.table.clear()
        self.table.setRowCount(len(show))
        self.table.setColumnCount(len(show.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in show.columns])
        for r, row in enumerate(show.itertuples(index=False)):
            for k, v in enumerate(row):
                self.table.setItem(r, k, QTableWidgetItem("" if v is None else str(v)))
        self._build_roles()
        self.draw()

    # ── column pickers + options ──────────────────────────────────────────
    @staticmethod
    def _clear_form(form):
        """Remove every widget of a layout at once (hidden immediately, deleted later)."""
        while form.count():
            it = form.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.deleteLater()

    def _build_roles(self):
        spec, df = self._spec, self.data()
        if spec is None or df is None:
            return
        self._clear_form(self.roles_form)
        self._role_widgets = {}
        guess = self.G.auto_columns(spec, df, fill_optional=self._df is None)
        columns = [str(c) for c in df.columns]
        for role in spec.roles:
            label = role.label + ("" if not role.optional else "  (optional)")
            if role.multi:
                w = QListWidget()
                w.setMaximumHeight(110)
                chosen = set(map(str, guess.get(role.key) or []))
                for c in columns:
                    it = QListWidgetItem(c)
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    it.setCheckState(Qt.CheckState.Checked if c in chosen else Qt.CheckState.Unchecked)
                    w.addItem(it)
                w.itemChanged.connect(self._schedule)
            else:
                w = _combo()
                if role.optional:
                    w.addItem(_NONE)
                w.addItems(columns)
                g = guess.get(role.key)
                if g and str(g) in columns:
                    w.setCurrentText(str(g))
                elif role.optional:
                    w.setCurrentText(_NONE)
                w.currentTextChanged.connect(self._schedule)
            w.setToolTip(role.hint or role.label)
            name_lbl = QLabel(label)
            name_lbl.setWordWrap(True)
            name_lbl.setMaximumWidth(200)
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            w.setMinimumWidth(120)
            self.roles_form.addRow(name_lbl, w)
            self._role_widgets[role.key] = (role, w)

    def columns(self) -> dict:
        df = self.data()
        name = {str(c): c for c in df.columns}
        out = {}
        for key, (role, w) in self._role_widgets.items():
            if role.multi:
                out[key] = [name[w.item(i).text()] for i in range(w.count())
                            if w.item(i).checkState() == Qt.CheckState.Checked]
            else:
                t = w.currentText()
                out[key] = "" if t in ("", _NONE) else name.get(t, t)
        return out

    def _build_opts(self):
        spec = self._spec
        self._clear_form(self.opts_grid)
        self._opt_widgets = {}
        common = [o for o in self.G.COMMON_OPTS if spec.axis_labels or o.key not in ("xlabel", "ylabel")]
        from dataclasses import replace
        for i, o in enumerate(spec.opts + common):
            if o.key in spec.defaults:                  # this plot starts the common option elsewhere
                o = replace(o, default=spec.defaults[o.key])
            w = self._opt_widget(o)
            # a normal field width: number boxes would otherwise size themselves for their largest value
            w.setMinimumWidth(90)
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.setMaximumWidth(220)
            lbl = QLabel(o.label)
            lbl.setWordWrap(True)
            lbl.setMinimumWidth(60)
            r, c = divmod(i, 2)
            self.opts_grid.addWidget(lbl, r, c * 2)
            self.opts_grid.addWidget(w, r, c * 2 + 1)
            self._opt_widgets[o.key] = (o, w)

    def _opt_widget(self, o):
        if o.type == "bool":
            w = QCheckBox()
            w.setChecked(bool(o.default))
            w.toggled.connect(self._schedule)
        elif o.type in ("choice", "cmap"):
            w = _combo()
            w.addItems([str(c) for c in (o.choices or [])])
            w.setCurrentText(str(o.default))
            w.currentTextChanged.connect(self._schedule)
        elif o.type == "int":
            w = QSpinBox()
            w.setRange(int(o.lo), int(min(o.hi, 2_000_000_000)))
            w.setSingleStep(int(max(1, o.step)))
            w.setValue(int(o.default))
            w.valueChanged.connect(self._schedule)
        elif o.type == "float":
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setRange(float(o.lo), float(min(o.hi, 1e12)))
            w.setSingleStep(float(o.step))
            w.setValue(float(o.default))
            w.valueChanged.connect(self._schedule)
        else:                                    # str / color
            w = QLineEdit(str(o.default))
            w.editingFinished.connect(self._schedule)
        return w

    def options(self) -> dict:
        out = {}
        for key, (o, w) in self._opt_widgets.items():
            if o.type == "bool":
                out[key] = w.isChecked()
            elif o.type in ("choice", "cmap"):
                out[key] = w.currentText()
            elif o.type in ("int", "float"):
                out[key] = w.value()
            else:
                out[key] = w.text()
        return out

    # ── drawing / saving ──────────────────────────────────────────────────
    def _schedule(self, *_):
        if self.auto_chk.isChecked():
            self._redraw_timer.start()

    def _figure(self, dpi: int = 100):
        return self.G.render(self._spec, self.data(), self.columns(), self.options(),
                             size=(self.w_spin.value(), self.h_spin.value()), dpi=dpi)

    def draw(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        if self._spec is None or self.data() is None:
            return
        missing = [r.label for k, (r, w) in self._role_widgets.items()
                   if not r.optional and not self.columns().get(k)]
        if missing:
            self._message("Choose a column for: " + ", ".join(missing), error=True)
            return
        # preview with the figure's true proportions (as it will be saved), scaled to fit the panel
        w_in, h_in = self.w_spin.value(), self.h_spin.value()
        avail_w = max(300, self._holder.width() - 30)
        dpi = max(30.0, min(100.0, avail_w / w_in, 720.0 / h_in))
        try:
            fig = self._figure(dpi=dpi)
        except Exception as e:
            self._message(f"This table does not fit this plot: {e}\n"
                          "Check the columns above, or press “Use example data” to see the expected layout.",
                          error=True, details=traceback.format_exc())
            return
        if self._canvas is not None:
            self.canvas_holder.removeWidget(self._canvas)
            self._canvas.hide()
            self._canvas.deleteLater()
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setFixedSize(int(w_in * dpi), int(h_in * dpi))
        self.canvas_holder.addWidget(self._canvas, 0, Qt.AlignmentFlag.AlignCenter)
        self._canvas.draw_idle()
        self._message("")

    def _message(self, text: str, error: bool = False, details: str = ""):
        self.msg_lbl.setText(text)
        self.msg_lbl.setStyleSheet("color: #d9534f; font-weight: 600;" if error else "")
        self.msg_lbl.setToolTip(details[-3000:] if details else "")

    def _save_figure(self):
        if self._spec is None:
            return
        start = str(Path(settings().value("gallery/save_dir", "", type=str) or Path.home()) / f"{self._spec.id}.png")
        path, flt = QFileDialog.getSaveFileName(self, "Save figure", start, SAVE_FILTER)
        if not path:
            return
        if not Path(path).suffix:
            path += "." + flt.split("*.")[-1].rstrip(")")
        try:
            dpi = self.dpi_spin.value()
            fig = self._figure(dpi=dpi)
            fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.1)
            settings().setValue("gallery/save_dir", str(Path(path).parent))
            self._message(f"Saved: {path}")
        except Exception as e:
            show_error(self, "Save failed", "The figure could not be saved.", str(e))


# ══ start screen: picture cards, left menu, search, pages ════════════════════

_LINK = "#1b7fc4"          # card captions / menu headers (blue links on white)
_PANEL_CSS = f"""
QWidget#GalPanel {{ background: #ffffff; border-radius: 10px; }}
QWidget#GalPanel QLabel {{ background: transparent; }}
QFrame#GalCard {{ background: #ffffff; border: 1px solid transparent; border-radius: 8px; }}
QFrame#GalCard:hover {{ border: 1px solid #b9d7ee; background: #f6fbff; }}
QLabel#GalCaption {{ color: {_LINK}; font-size: 15px; }}
QLabel#GalCount {{ color: #666666; font-size: 13px; }}
QWidget#GalMenu {{ background: #ffffff; border-right: 1px solid #e3e3e3; }}
QLabel#GalMenuHead {{ color: #0b5c94; font-weight: 700; font-size: 15px; padding: 3px 6px 1px 4px;
                      border-radius: 4px; }}
QLabel#GalMenuItem {{ color: #555555; font-size: 13px; padding: 1px 6px 1px 22px; border-radius: 4px; }}
QLabel#GalMenuHead:hover, QLabel#GalMenuItem:hover {{ background: #eef6fc; }}
QLabel#GalMenuHead[selected="true"], QLabel#GalMenuItem[selected="true"] {{ background: #dcecf8;
                                                                             color: #0b5c94; }}
QLineEdit#GalSearch {{ background: #ffffff; color: #222222; border: 1px solid #cfd6dc; border-radius: 4px;
                       padding: 7px 10px; font-size: 14px; }}
QPushButton#GalSearchBtn {{ background: #17a2b8; color: white; border: none; border-radius: 4px;
                            padding: 8px 18px; font-size: 14px; }}
QPushButton#GalSearchBtn:hover {{ background: #138496; }}
QPushButton#GalPage {{ background: #ffffff; color: {_LINK}; border: 1px solid #dee2e6; border-radius: 0px;
                       padding: 6px 12px; min-width: 14px; font-size: 13px; }}
QPushButton#GalPage:hover {{ background: #e9ecef; }}
QPushButton#GalPage:checked {{ background: #0d6efd; color: white; border-color: #0d6efd; }}
QPushButton#GalPage:disabled {{ color: #9aa0a6; }}
"""


class _ThumbWorker(QThread):
    """Draws the missing card pictures in the background (first use only)."""
    ready = pyqtSignal(str, str)          # plot id, picture path

    def __init__(self, specs):
        super().__init__()
        self.specs = list(specs)

    def run(self):
        from s2q.core.gallery.thumbs import make_missing
        for s in self.specs:
            try:
                self.ready.emit(s.id, str(make_missing(s)))
            except Exception:
                pass


class _Card(QFrame):
    """A picture of the plot with its name underneath; a click opens the plot."""
    clicked = pyqtSignal(str)
    W, H = 262, 192                       # largest picture size on screen

    def __init__(self, spec, caption: str, width: int = 262, parent=None):
        super().__init__(parent)
        self.W, self.H = int(width), int(width * 0.733)
        self.setObjectName("GalCard")
        self.spec_id = spec.id
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setToolTip(f"{spec.name}  ({spec.number})\n{spec.desc}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.pic = QLabel()
        self.pic.setFixedSize(self.W, self.H)
        self.pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pic.setStyleSheet("color: #aaaaaa;")
        self.pic.setText("drawing…")
        lay.addWidget(self.pic, 0, Qt.AlignmentFlag.AlignHCenter)
        cap = QLabel(caption)
        cap.setObjectName("GalCaption")
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap.setWordWrap(True)
        cap.setFixedWidth(self.W)
        lay.addWidget(cap)

    def set_picture(self, path: str):
        pm = QPixmap(path)
        if pm.isNull():
            return
        ratio = self.devicePixelRatioF() or 1.0
        pm = pm.scaled(int(self.W * ratio), int(self.H * ratio), Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        pm.setDevicePixelRatio(ratio)
        self.pic.setPixmap(pm)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit(self.spec_id)
        super().mouseReleaseEvent(e)


class GalleryBrowser(QWidget):
    """Search bar, left menu (ALL modules + keyword groups), a page of cards and page buttons."""
    open_plot = pyqtSignal(str)

    def __init__(self, gallery, parent=None):
        super().__init__(parent)
        from s2q.core.gallery import catalog
        self.G, self.K = gallery, catalog
        self._all = catalog.ordered(gallery.REGISTRY)
        self._shown = list(self._all)
        self._page = 0
        self._cols = 6
        self._cards: list = []
        self._worker = None
        self._build()
        self._show_list(self._all)

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 14, 0, 0)
        panel = QWidget()
        panel.setObjectName("GalPanel")
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setStyleSheet(_PANEL_CSS)
        outer.addWidget(panel)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(14, 14, 14, 14)
        pl.setSpacing(10)

        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setObjectName("GalSearch")
        self.search.setPlaceholderText("input keyword to search")
        self.search.setMaximumWidth(520)
        self.search.returnPressed.connect(self._do_search)
        top.addWidget(self.search, 1)
        go = QPushButton("Search")
        go.setObjectName("GalSearchBtn")
        go.clicked.connect(self._do_search)
        top.addWidget(go)
        top.addStretch(1)
        self.count_lbl = QLabel("")
        self.count_lbl.setObjectName("GalCount")
        top.addWidget(self.count_lbl)
        pl.addLayout(top)

        body = QHBoxLayout()
        body.setSpacing(12)
        self.menu = QWidget()
        self.menu.setObjectName("GalMenu")
        self.menu.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.menu.setFixedWidth(190)
        self._menu_lay = QVBoxLayout(self.menu)
        self._menu_lay.setContentsMargins(2, 0, 8, 0)
        self._menu_lay.setSpacing(1)
        self._menu_items: list = []          # (label, keywords)
        self._add_menu_item("ALL modules", None, header=True)
        for head, entries in self.K.MENU:
            kws = [k for _, ks in entries for k in ks]
            self._add_menu_item(head, kws, header=True)
            for name, ks in entries:
                self._add_menu_item(name, ks)
        self._menu_lay.addStretch(1)
        self._select_menu(0)
        body.addWidget(self.menu)

        # the card area takes the width it is given (fewer columns when narrow) instead of
        # forcing the page wider than the window
        from PyQt6.QtWidgets import QSizePolicy
        cards_area = QWidget()
        cards_area.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        right = QVBoxLayout(cards_area)
        right.setContentsMargins(0, 0, 0, 0)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(14)
        right.addLayout(self.grid)
        right.addSpacing(10)
        self.pager = QHBoxLayout()          # page buttons right under the cards
        self.pager.setSpacing(0)
        right.addLayout(self.pager)
        right.addStretch(1)
        body.addWidget(cards_area, 1)
        pl.addLayout(body, 1)

    def _add_menu_item(self, text: str, keywords, header: bool = False):
        lbl = QLabel(text)
        lbl.setObjectName("GalMenuHead" if header else "GalMenuItem")
        lbl.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        idx = len(self._menu_items)
        lbl.mouseReleaseEvent = lambda e, i=idx: self.choose_menu(i)
        self._menu_lay.addWidget(lbl)
        self._menu_items.append((lbl, keywords))

    def _select_menu(self, idx: int):
        """Highlight menu entry *idx* (-1 = none, e.g. while a search is shown)."""
        for i, (lbl, _) in enumerate(self._menu_items):
            lbl.setProperty("selected", "true" if i == idx else "false")
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    # ── filtering ─────────────────────────────────────────────────────────
    def choose_menu(self, idx: int):
        """Show the plots of menu entry *idx* (0 = ALL modules)."""
        kws = self._menu_items[idx][1]
        self._select_menu(idx)
        self.search.clear()
        self._show_list(self._all if not kws else [s for s in self._all if self.K.matches(s, kws)])

    def _do_search(self):
        text = self.search.text().strip()
        self._select_menu(0 if not text else -1)
        self._show_list(self.K.search(self._all, text) if text else self._all)

    def _show_list(self, specs):
        self._shown = list(specs)
        self._page = 0
        self._render_page()

    # ── cards + pages ─────────────────────────────────────────────────────
    def _layout_for_width(self) -> tuple[int, int]:
        """(columns, card picture width): 6 columns like the original pages while cards stay ≥ 170 px."""
        avail = self.width() - 190 - 70
        if avail <= 0:
            return 6, _Card.W
        per_col = avail // 6 - 22                  # card frame margins + grid spacing
        if per_col >= 170:
            return 6, min(_Card.W, per_col)
        cols = max(2, avail // (170 + 22))
        return cols, 170

    def resizeEvent(self, e):
        # never rebuild inside a resize (that resizes again); do it once the size has settled
        super().resizeEvent(e)
        if not hasattr(self, "_resize_timer"):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.setInterval(120)
            self._resize_timer.timeout.connect(self._on_resized)
        self._resize_timer.start()

    def showEvent(self, e):
        # the window may have changed size while a plot editor was shown
        super().showEvent(e)
        QTimer.singleShot(0, self._on_resized)

    def _on_resized(self):
        cols, w = self._layout_for_width()
        if cols != self._cols or abs(w - getattr(self, "_card_w", w)) > 12:
            self._render_page()

    def _render_page(self):
        from s2q.core.gallery.thumbs import thumb_path
        for c in self._cards:                   # hidden now, deleted by Qt once (still owned by us)
            self.grid.removeWidget(c)
            c.hide()
            c.deleteLater()
        self._cards = []
        per = self.K.PER_PAGE
        n_pages = max(1, -(-len(self._shown) // per))
        self._page = min(self._page, n_pages - 1)
        chunk = self._shown[self._page * per:(self._page + 1) * per]
        self._cols, self._card_w = self._layout_for_width()
        missing = []
        for i, spec in enumerate(chunk):
            card_w = _Card(spec, self.K.card_name(spec), self._card_w)
            card_w.clicked.connect(self.open_plot.emit)
            p = thumb_path(spec)
            if p is not None:
                card_w.set_picture(str(p))
            else:
                missing.append(spec)
            self.grid.addWidget(card_w, i // self._cols, i % self._cols, Qt.AlignmentFlag.AlignTop)
            self._cards.append(card_w)
        if not chunk:
            empty = QLabel("No plot matches this search.")
            empty.setStyleSheet("color: #666666; font-size: 15px; padding: 30px;")
            self.grid.addWidget(empty, 0, 0)
            self._cards.append(empty)
        total = len(self._shown)
        self.count_lbl.setText(f"{total} plot type{'s' if total != 1 else ''}"
                               + (f"  ·  page {self._page + 1} of {n_pages}" if n_pages > 1 else ""))
        self._build_pager(n_pages)
        if missing:
            self._draw_missing(missing)

    def _draw_missing(self, specs):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
        self._worker = _ThumbWorker(specs)
        self._worker.ready.connect(self._on_thumb)
        self._worker.start()

    def _on_thumb(self, pid: str, path: str):
        for c in self._cards:
            if isinstance(c, _Card) and c.spec_id == pid:
                c.set_picture(path)

    def _build_pager(self, n_pages: int):
        while self.pager.count():
            it = self.pager.takeAt(0)
            if it.widget() is not None:
                it.widget().hide()
                it.widget().deleteLater()
        if n_pages <= 1:
            return
        self.pager.addStretch(1)

        def btn(text, page, checked=False, enabled=True):
            b = QPushButton(text)
            b.setObjectName("GalPage")
            b.setCheckable(True)
            b.setChecked(checked)
            b.setEnabled(enabled)
            b.clicked.connect(lambda _, p=page: self.go_to(p))
            self.pager.addWidget(b)

        btn("Previous", self._page - 1, enabled=self._page > 0)
        for p in range(n_pages):
            btn(str(p + 1), p, checked=p == self._page)
        btn("Next", self._page + 1, enabled=self._page < n_pages - 1)
        self.pager.addStretch(1)

    def go_to(self, page: int):
        self._page = page
        self._render_page()


class PlotGalleryTab(QWidget):
    """The gallery: picture cards first; a click opens that plot's editor (← All plots goes back)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from s2q.core import gallery
        self.G = gallery
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack)
        self.browser = GalleryBrowser(gallery)
        self.editor = PlotEditor(gallery)
        self.stack.addWidget(self.browser)
        self.stack.addWidget(self.editor)
        self.browser.open_plot.connect(self.open_plot)
        self.editor.back.connect(lambda: self.stack.setCurrentWidget(self.browser))

    def open_plot(self, pid: str):
        # show the editor first so the preview knows its width, then fill it
        self.stack.setCurrentWidget(self.editor)
        self.editor.select(pid)
