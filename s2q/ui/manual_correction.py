"""Manual Segmentation Correction editor.

Review a segmentation mask over the original image and fix it: pencil, partial
eraser, outline-and-fill, magic wand, AI add-cell (Cellpose-SAM on a box),
delete (click or rectangle) and fill holes, with undo/redo, clean-up filters,
cell info on hover, cell numbers, a second-channel overlay, and Previous / Next
navigation through a folder with autosave and automatic re-measurement.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygonF, QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QGroupBox,
    QGridLayout, QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from s2q.core.mask_edit import (
    cell_centroids, delete_cell, delete_in_rect, diff_region, fill_holes_all, fill_holes_cell,
    fill_polygon, insert_object, magic_wand, next_label, paint_stroke,
    remove_border, remove_small, grow_cells, shrink_cells,
)
from s2q.ui.theme import COLORS

logger = logging.getLogger(__name__)

# ── colour palette for overlay ────────────────────────────────────────────────
_BRUSH_COLOURS = [         # cyclic colours per cell label
    (0, 230, 118), (255, 171, 0), (0, 176, 255), (255, 64, 129),
    (213, 0, 249), (0, 230, 64), (255, 214, 0), (100, 255, 218),
]
_COLOUR_LUT = np.array(_BRUSH_COLOURS, dtype=np.uint8)

_MODES = ("rgb", "r", "g", "b")


# ── image helpers (also used by the Correction page thumbnails) ───────────────

def read_raw(path, plane=None) -> np.ndarray:
    """An image with its original values: ordinary files, or one plane of a microscopy file
    (ND2, LIF, CZI, LSM, OME-TIFF, OME-Zarr; *plane* picks series / time / channel / Z)."""
    from s2q.core.microscopy import is_microscopy, read_plane
    if is_microscopy(path):
        return read_plane(path, plane)
    from skimage.io import imread
    from s2q.core.image_files import channels_last
    return channels_last(imread(str(path)))


def image_channels(path, plane=None) -> list:
    """[(channel name, 2-D array of original values), ...] of an image or microscopy plane."""
    from s2q.core.microscopy import is_microscopy
    if is_microscopy(path):
        from s2q.core.microscopy import read_channels
        return read_channels(path, plane)
    raw = read_raw(path)
    if raw.ndim == 3:
        n = min(raw.shape[2], 3)
        names = ["Red", "Green", "Blue"] if n == 3 else [f"Channel {i + 1}" for i in range(n)]
        if n == 3 and np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 0], raw[..., 2]):
            return [("Intensity", raw[..., 0])]      # grey image stored as RGB
        return [(names[i], raw[..., i]) for i in range(n)]
    return [("Intensity", raw)]


def load_image_normalised(path, max_side: int | None = None, plane=None) -> np.ndarray:
    """Read an image as float32 (H, W, C) in [0, 1], C = 1 or 3.

    Each channel is stretched between its 0.5 and 99.5 percentiles so 8-bit,
    16-bit and mostly-dark fluorescence images are all visible.
    """
    return normalise_image(read_raw(path, plane), max_side)


def normalise_image(img: np.ndarray, max_side: int | None = None) -> np.ndarray:
    import cv2
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.shape[2] == 4:
        img = img[:, :, :3]                  # drop alpha
    elif img.shape[2] == 2:
        img = img[:, :, :1]
    if max_side and max(img.shape[:2]) > max_side:
        f = max_side / max(img.shape[:2])
        size = (max(1, int(img.shape[1] * f)), max(1, int(img.shape[0] * f)))
        # float32: OpenCV cannot resize every pixel type (e.g. signed 8-bit microscopy data)
        img = cv2.resize(img.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
    out = np.zeros(img.shape, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float32)
        lo, hi = np.percentile(ch, (0.5, 99.5))
        if hi > lo:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def load_mask(path, shape=None) -> np.ndarray:
    """Label image; an empty mask of *shape* if the file does not exist yet (a new mask)."""
    from skimage.io import imread
    if shape is not None and not Path(path).exists():
        return np.zeros(shape[:2], dtype=np.int32)
    mask = imread(str(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int32)


def render_view(raw: np.ndarray, mode: str = "rgb", intensity: float = 1.0,
                brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    """float (H, W, C) image → uint8 RGB for display."""
    h, w, n = raw.shape
    if n == 1:
        rgb = np.repeat(raw, 3, axis=2)
    else:
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[:, :, :n] = raw[:, :, :3]
        if mode in ("r", "g", "b"):
            keep = "rgb".index(mode)
            for c in range(3):
                if c != keep:
                    rgb[:, :, c] = 0
    rgb = (rgb * intensity - 0.5) * contrast + 0.5 + brightness
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def cell_outlines(mask: np.ndarray) -> np.ndarray:
    """Boolean map of pixels on the edge of a cell (label differs from a 4-neighbour)."""
    edge = np.zeros(mask.shape, dtype=bool)
    edge[:-1, :] |= mask[:-1, :] != mask[1:, :]
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    return edge & (mask > 0)


def overlay_mask(display: np.ndarray, mask: np.ndarray, opacity: float,
                 outline: bool = False, outline_px: int = 1) -> np.ndarray:
    """Blend coloured cell labels onto a uint8 RGB image (only touches cell pixels).

    With *outline* only cell edges are drawn (``outline_px`` thick), so dense
    images stay readable.
    """
    out = display.copy()
    if opacity <= 0:
        return out
    if outline:
        fg = cell_outlines(mask)
        if outline_px > 1:
            import cv2
            fg = cv2.dilate(fg.astype(np.uint8), np.ones((outline_px, outline_px), np.uint8)) > 0
            fg &= mask > 0
        opacity = min(1.0, opacity * 2)
    else:
        fg = mask > 0
    colours = _COLOUR_LUT[(mask[fg] - 1) % len(_COLOUR_LUT)].astype(np.float32)
    out[fg] = (display[fg].astype(np.float32) * (1 - opacity) + colours * opacity).astype(np.uint8)
    return out


VIEW_MODES = [("overlay", "Image + mask"), ("cells", "Cells only (background removed)"),
              ("background", "Background only (cells removed)"), ("image", "Image only"), ("mask", "Mask only")]


def compose_view(display: np.ndarray, mask: np.ndarray, mode: str, opacity: float,
                 outline: bool = False, outline_px: int = 1) -> np.ndarray:
    """uint8 RGB for one view mode: the overlay, only the segmented area, only the background, …"""
    if mode == "image":
        return display
    if mode == "cells":
        out = display.copy()
        out[mask == 0] = 0
        return out
    if mode == "background":
        out = display.copy()
        out[mask > 0] = 0
        return out
    if mode == "mask":
        return overlay_mask(np.zeros_like(display), mask, 1.0, outline, outline_px)
    return overlay_mask(display, mask, opacity, outline, outline_px)


def render_pair_thumbnail(image_path, mask_path, max_side: int = 220) -> np.ndarray:
    """Small uint8 RGB preview of an image with its mask overlaid."""
    import cv2
    from s2q.core.microscopy import is_microscopy, plane_from_mask_name
    plane = plane_from_mask_name(image_path, mask_path) if is_microscopy(image_path) else None
    raw = load_image_normalised(image_path, max_side=max_side, plane=plane)
    mask = load_mask(mask_path, raw.shape)
    mask = cv2.resize(mask.astype(np.float32), (raw.shape[1], raw.shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return overlay_mask(render_view(raw), mask, 0.45)


def ndarray_to_qimage(arr: np.ndarray) -> QImage:
    """uint8 (H, W, 3) → QImage that owns its own copy of the data."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


# ── AI (Cellpose-SAM) helper ──────────────────────────────────────────────────

class _AISegmentWorker(QThread):
    """Runs Cellpose-SAM on the box (inference only; the model is loaded beforehand)."""
    done = pyqtSignal(object, str)   # boolean object mask (full image) or None, method / reason
    failed = pyqtSignal(str)

    def __init__(self, model, image, free, box, from_click, region=None, one_cell_box=False):
        super().__init__()
        self.model, self.image, self.free, self.box, self.from_click = model, image, free, box, from_click
        self.region = region            # selected area (bool, full size) or None
        self.one_cell_box = one_cell_box  # box about one cell big: may fall back to the one-cell search

    def run(self):
        from s2q.core.ai_assist import segment_all, segment_box
        try:
            if not self.from_click:     # a box / an area: every missed cell in it
                labels = segment_all(self.model, self.image, self.free, self.box, self.region)
                if labels.max() > 0 or not self.one_cell_box:
                    self.done.emit(labels, "Cellpose-SAM")
                    return
            obj, how = segment_box(self.model, self.image, self.free, self.box, self.from_click)
            if not self.from_click and how != "Cellpose-SAM":
                obj = None              # a box with no missed cells: no brightness guesswork
            self.done.emit(obj, how)
        except Exception as e:
            self.failed.emit(str(e))


class _Toast(QLabel):
    """Short message floating over the image (bottom centre), hides after a few seconds."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text: str, kind: str = "Pending", ms: int = 3500):
        colour = {"Success": COLORS["success"], "Warning": COLORS["warning"],
                  "Error": COLORS["error"]}.get(kind, COLORS["accent"])
        self.setStyleSheet(
            f"background: rgba(20, 20, 26, 225); color: #ffffff; border: 2px solid {colour}; "
            f"border-radius: 10px; padding: 10px 16px; font-size: 13px; font-weight: 600;")
        self.setText(text)
        par = self.parentWidget()
        self.setFixedWidth(min(560, max(260, par.width() - 40)))
        self.adjustSize()
        # bottom centre: the top is where an opened / focused cell often sits
        self.move((par.width() - self.width()) // 2, par.height() - self.height() - 18)
        self.raise_()
        self.show()
        if ms > 0:
            self._timer.start(ms)
        else:
            self._timer.stop()


class _MeasureWorker(QThread):
    """Re-runs the ROI measurements (Excel sheet + overlay) for one corrected image."""
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, mask_path: Path, image_path: Path, image=None, pixel_size=None):
        super().__init__()
        self.mask_path, self.image_path = mask_path, image_path
        self.pixel_size = pixel_size        # µm per pixel → extra µm columns, or None
        self.image = image          # the plane of a microscopy file (measured instead of reading the file)

    def run(self):
        try:
            from s2q.core.image_files import list_masks
            from s2q.core.roi_converter import ROIVisualizer
            from s2q.core.summary import generate_summary_sheet
            folder = self.mask_path.parent
            stem = self.mask_path.stem
            xlsx = folder / f"{stem}.xlsx"
            ROIVisualizer(str(self.mask_path), str(self.image_path), str(xlsx),
                          str(folder / f"{stem}_ROI.png"), show_labels=True,
                          image=self.image, pixel_size=self.pixel_size).save_rois_to_excel()
            summary = folder / "SummarySheet" / "Summary.xlsx"
            if summary.exists():
                sheets = [folder / f"{m.stem}.xlsx" for m in list_masks(folder)]
                generate_summary_sheet(str(folder), str(summary), files=[s for s in sheets if s.exists()])
            self.done.emit(str(xlsx))
        except Exception as e:
            self.failed.emit(str(e))


# ── image canvas widget ───────────────────────────────────────────────────────

TOOLS = [
    # key, label, shortcut, tooltip
    ("pencil",  "✏️ Pencil",      "P", "Paint a cell. Starting inside a cell extends it; on background it adds a "
                                         "new cell. Hold Ctrl to paint over neighbours."),
    ("eraser",  "🩹 Eraser",      "E", "Erase only the pixels you paint over (trim a cell)."),
    ("outline", "➰ Outline",     "L", "Draw around a missed cell; it is filled when you release. Ctrl = over neighbours."),
    ("wand",    "🪄 Wand",        "W", "Click: grow a cell over pixels of similar brightness (tolerance below). "
                                         "Best for evenly bright cells; for textured cells use AI add cell."),
    ("ai",      "🤖 AI add cells", "A", "Drag a box: Cellpose-SAM adds every missed cell in it. "
                                         "Click on one missed cell to add just that cell."),
    ("delete",  "🗑 Delete",      "D", "Click a cell to delete it, or drag a rectangle to delete every cell it touches."),
    ("fill",    "🪣 Fill holes",  "F", "Click a cell to fill the holes inside it."),
    ("area",    "⬚ Area",        "R", "Drag a rectangle (or draw freehand, see “Area of interest”) to choose a "
                                         "region: only the cells in it are shown."),
    ("line",    "📈 Line profile", "I", "Drag a line: the intensity of every channel along it is plotted in a "
                                          "separate window (line width, plot size, integrated intensity, CSV). "
                                          "Drag the line to move it, drag an end to adjust it, click on the line "
                                          "to delete it."),
]
_DRAG_TOOLS = {"outline", "ai", "delete", "area", "line"}


class _Canvas(QLabel):
    """Displays the image + mask overlay and turns mouse input into edits."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

        # assigned by the dialog
        self.mask: np.ndarray | None = None        # H×W integer labels
        self.display: np.ndarray | None = None     # H×W×3 uint8 display image
        self.opacity = 0.4
        self.outline = False
        self.show_numbers = False
        self.centroids: dict[int, tuple[float, float]] = {}
        self._scale = 1.0
        self._tool = "pencil"
        self._brush_size = 10
        self._drawing = False
        self._last_pt: tuple[int, int] | None = None
        self._next_label = 1
        self._overwrite = False
        self._drag_start: QPointF | None = None    # widget coords for rect tools
        self._drag_now: QPointF | None = None
        self._path: list[tuple[int, int]] = []     # image coords for the outline tool
        self._pan_from: QPointF | None = None
        self.pending_box: tuple[int, int, int, int] | None = None   # AI box being processed
        self.space_down = False
        self.hover_label = 0
        self._hover_poly: list[QPolygonF] = []
        self.composer = None          # (outline_px) → uint8 RGB; set by the dialog (views, layers, area)
        self.area_poly: list[tuple[int, int]] | None = None   # area of interest, image coords
        self.area_freehand = False
        self.line_seg: tuple | None = None        # ((x0, y0), (x1, y1)) of the line profile
        self.line_width = 1
        self._line_drag = None                     # (part "p0"/"p1"/"body", press point, line at press)
        self._line_moved = False

        # callbacks set by the dialog
        self.on_mask_changed = None   # () → after a brush stroke
        self.on_stroke_start = None   # () → before a brush stroke edits the mask
        self.on_zoom_wheel = None     # (steps) → Ctrl + mouse wheel
        self.on_action = None         # (tool, payload, overwrite) → dialog-side edits
        self.on_hover = None          # (label, x, y)
        self.on_pan = None            # (dx, dy) in widget pixels

    # ── rendering ─────────────────────────────────────────────────────────

    def refresh(self):
        if self.display is None or self.mask is None:
            return
        # thicker edges when zoomed out, otherwise 1-px outlines vanish on screen
        px = 1 if self._scale >= 0.6 else min(4, int(round(1 / self._scale)))
        composed = (self.composer(px) if self.composer
                    else overlay_mask(self.display, self.mask, self.opacity, self.outline, px))
        pix = QPixmap.fromImage(ndarray_to_qimage(composed))
        if self._scale != 1.0:
            # fast scaling while drawing keeps strokes responsive on large images
            mode = (Qt.TransformationMode.FastTransformation if self._drawing
                    else Qt.TransformationMode.SmoothTransformation)
            pix = pix.scaled(max(1, int(pix.width() * self._scale)),
                             max(1, int(pix.height() * self._scale)),
                             Qt.AspectRatioMode.KeepAspectRatio, mode)
        self.setPixmap(pix)
        self.setFixedSize(pix.size())
        self._update_hover_poly()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self._scale
        if self.show_numbers and self.centroids:
            font = QFont()
            font.setPointSizeF(max(6.0, min(11.0, 9 * s ** 0.5)))
            font.setBold(True)
            p.setFont(font)
            visible = self.visibleRegion().boundingRect()
            shown = [(lab, cx * s, cy * s) for lab, (cy, cx) in self.centroids.items()
                     if visible.contains(int(cx * s), int(cy * s))]
            if len(shown) <= 400:   # more than this is unreadable — zoom in to see numbers
                for lab, x, y in shown:
                    p.setPen(QColor(0, 0, 0, 200))
                    p.drawText(QPointF(x - 4, y + 5), str(lab))
                    p.setPen(QColor(255, 255, 255))
                    p.drawText(QPointF(x - 5, y + 4), str(lab))
        if self._hover_poly:
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for poly in self._hover_poly:
                p.drawPolygon(poly)
        if self.area_poly:
            p.setPen(QPen(QColor(255, 214, 0), 2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolygon(QPolygonF([QPointF(x * s, y * s) for x, y in self.area_poly]))
        if self.line_seg is not None:
            (lx0, ly0), (lx1, ly1) = self.line_seg
            a, b = QPointF((lx0 + 0.5) * s, (ly0 + 0.5) * s), QPointF((lx1 + 0.5) * s, (ly1 + 0.5) * s)
            if self.line_width > 1:      # the band that is averaged
                p.setPen(QPen(QColor(0, 229, 255, 70), max(1.0, self.line_width * s),
                              Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
                p.drawLine(a, b)
            p.setPen(QPen(QColor(0, 229, 255), 2))
            p.drawLine(a, b)
            p.setBrush(QColor(0, 229, 255))
            p.drawEllipse(a, 4, 4)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(b, 5, 5)
        if self._drag_start is not None and self._drag_now is not None and self._tool == "line":
            p.setPen(QPen(QColor(0, 229, 255), 2, Qt.PenStyle.DashLine))
            p.drawLine(self._drag_start, self._drag_now)
        if self._drag_start is not None and self._drag_now is not None and self._tool in ("ai", "delete", "area"):
            colour = {"ai": QColor(COLORS["accent"]), "delete": QColor(COLORS["error"]),
                      "area": QColor(255, 214, 0)}[self._tool]
            p.setPen(QPen(colour, 2, Qt.PenStyle.DashLine))
            p.setBrush(QColor(colour.red(), colour.green(), colour.blue(), 40))
            p.drawRect(QRectF(self._drag_start, self._drag_now).normalized())
        if self.pending_box is not None:
            x0, y0, x1, y1 = self.pending_box
            r = QRectF(x0 * s, y0 * s, (x1 - x0) * s, (y1 - y0) * s)
            p.setPen(QPen(QColor(COLORS["accent"]), 2, Qt.PenStyle.DashLine))
            p.setBrush(QColor(124, 106, 239, 50))
            p.drawRect(r)
            p.setPen(QColor(255, 255, 255))
            p.drawText(r.adjusted(4, 2, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "AI…")
        if self._path:
            p.setPen(QPen(QColor(COLORS["accent"]), 2))
            pts = [QPointF(x * s, y * s) for x, y in self._path]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        p.end()

    def _update_hover_poly(self):
        """Outline of the cell under the mouse, in widget coordinates."""
        import cv2
        self._hover_poly = []
        lab = self.hover_label
        if not lab or self.mask is None:
            return
        ys, xs = np.nonzero(self.mask == lab)
        if not len(ys):
            return
        y0, x0 = ys.min(), xs.min()
        sub = (self.mask[y0:ys.max() + 1, x0:xs.max() + 1] == lab).astype(np.uint8)
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        s = self._scale
        for c in contours:
            self._hover_poly.append(QPolygonF([QPointF((pt[0][0] + x0 + 0.5) * s, (pt[0][1] + y0 + 0.5) * s)
                                               for pt in c]))

    # ── coordinate helpers ────────────────────────────────────────────────

    def _line_hit(self, pos) -> str | None:
        """Which part of the profile line is under *pos* (widget coords): "p0", "p1", "body" or None."""
        if self.line_seg is None:
            return None
        s = self._scale
        (x0, y0), (x1, y1) = [((x + 0.5) * s, (y + 0.5) * s) for x, y in self.line_seg]
        px, py = pos.x(), pos.y()
        for name, (ex, ey) in (("p0", (x0, y0)), ("p1", (x1, y1))):
            if (px - ex) ** 2 + (py - ey) ** 2 <= 9 ** 2:
                return name
        vx, vy = x1 - x0, y1 - y0
        L2 = vx * vx + vy * vy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x0) * vx + (py - y0) * vy) / L2))
        d = ((px - (x0 + t * vx)) ** 2 + (py - (y0 + t * vy)) ** 2) ** 0.5
        return "body" if d <= max(6.0, self.line_width * s / 2 + 3) else None

    def _canvas_to_img(self, pt) -> tuple[int, int]:
        x = int(pt.x() / self._scale)
        y = int(pt.y() / self._scale)
        if self.mask is not None:
            h, w = self.mask.shape[:2]
            x = min(max(x, 0), w - 1)
            y = min(max(y, 0), h - 1)
        return x, y

    # ── mouse events ──────────────────────────────────────────────────────

    def mousePressEvent(self, ev):
        pos = ev.position()
        if ev.button() == Qt.MouseButton.MiddleButton or (
                ev.button() == Qt.MouseButton.LeftButton and self.space_down):
            self._pan_from = ev.globalPosition()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if ev.button() != Qt.MouseButton.LeftButton or self.mask is None:
            return
        self._overwrite = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
        x, y = self._canvas_to_img(pos)
        tool = self._tool
        if tool in ("pencil", "eraser"):
            if self.on_stroke_start:
                self.on_stroke_start()
            self._drawing = True
            clicked = int(self.mask[y, x])
            # starting inside a cell extends it; starting on background makes a new cell
            self._next_label = 0 if tool == "eraser" else (clicked or int(self.mask.max()) + 1)
            self._last_pt = None
            self._apply_stroke(pos)
        elif tool == "outline" or (tool == "area" and self.area_freehand):
            # 0 = a new cell; its number is chosen on release (the mask may change meanwhile)
            self._next_label = int(self.mask[y, x])
            self._path = [(x, y)]
        elif tool == "line" and self._line_hit(pos):
            # grab an end point (adjust) or the line itself (move; a click without moving deletes it)
            self._line_drag = (self._line_hit(pos), pos, self.line_seg)
            self._line_moved = False
        elif tool in ("ai", "delete", "area", "line"):
            self._drag_start = self._drag_now = pos
        elif self.on_action:
            self.on_action(tool, (x, y), self._overwrite)

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._pan_from is not None:
            d = ev.globalPosition() - self._pan_from
            self._pan_from = ev.globalPosition()
            if self.on_pan:
                self.on_pan(-d.x(), -d.y())
            return
        if self.mask is None:
            return
        if self._line_drag is not None:
            part, start, (p0, p1) = self._line_drag
            dx, dy = (pos.x() - start.x()) / self._scale, (pos.y() - start.y()) / self._scale
            if abs(dx) + abs(dy) > 0.5:
                self._line_moved = True
            h, w = self.mask.shape[:2]
            clamp = lambda x, y: (int(min(max(round(x), 0), w - 1)), int(min(max(round(y), 0), h - 1)))
            if part == "p0":
                seg = (clamp(p0[0] + dx, p0[1] + dy), p1)
            elif part == "p1":
                seg = (p0, clamp(p1[0] + dx, p1[1] + dy))
            else:   # move the whole line, keeping it inside the image
                dx = min(max(dx, -min(p0[0], p1[0])), w - 1 - max(p0[0], p1[0]))
                dy = min(max(dy, -min(p0[1], p1[1])), h - 1 - max(p0[1], p1[1]))
                seg = (clamp(p0[0] + dx, p0[1] + dy), clamp(p1[0] + dx, p1[1] + dy))
            if self._line_moved and seg != self.line_seg:
                self.line_seg = seg
                self.update()
                if self.on_action:
                    self.on_action("line_move", seg, False)
            return
        if self._drawing:
            self._apply_stroke(pos)
            return
        if self._path:
            x, y = self._canvas_to_img(pos)
            if (x, y) != self._path[-1]:
                self._path.append((x, y))
                self.update()
            return
        if self._drag_start is not None:
            self._drag_now = pos
            self.update()
            return
        if self._tool == "line":
            part = self._line_hit(pos)
            self.setCursor(Qt.CursorShape.SizeAllCursor if part == "body" else
                           Qt.CursorShape.PointingHandCursor if part else Qt.CursorShape.CrossCursor)
        # hover: highlight the cell under the mouse and report its details
        x, y = self._canvas_to_img(pos)
        lab = int(self.mask[y, x])
        if lab != self.hover_label:
            self.hover_label = lab
            self._update_hover_poly()
            self.update()
        if self.on_hover:
            self.on_hover(lab, x, y)

    def mouseReleaseEvent(self, ev):
        if self._pan_from is not None:
            self._pan_from = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.space_down else Qt.CursorShape.CrossCursor)
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if self._line_drag is not None:
            part = self._line_drag[0]
            self._line_drag = None
            if self.on_action:
                if self._line_moved:
                    self.on_action("line_set", self.line_seg, False)
                elif part == "body":
                    self.on_action("line_delete", None, False)      # a click on the line deletes it
            return
        if self._drawing:
            self._drawing = False
            self._last_pt = None
            self.refresh()   # redraw with smooth scaling
            if self.on_mask_changed:
                self.on_mask_changed()
        elif self._path:
            path, self._path = self._path, []
            self.update()
            if self.on_action:
                if self._tool == "area":
                    self.on_action("area", ("poly", path), False)
                else:
                    self.on_action("outline", (path, self._next_label), self._overwrite)
        elif self._drag_start is not None:
            a = self._canvas_to_img(self._drag_start)
            b = self._canvas_to_img(ev.position())
            self._drag_start = self._drag_now = None
            self.update()
            if self.on_action:
                self.on_action(self._tool, ("rect", (a, b)) if self._tool == "area" else (a, b), self._overwrite)

    def leaveEvent(self, ev):
        if self.hover_label:
            self.hover_label = 0
            self._hover_poly = []
            self.update()
        super().leaveEvent(ev)

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier and self.on_zoom_wheel:
            self.on_zoom_wheel(1 if ev.angleDelta().y() > 0 else -1)
            ev.accept()
        else:
            super().wheelEvent(ev)

    def _apply_stroke(self, pt):
        x, y = self._canvas_to_img(pt)
        r = max(1, self._brush_size // 2)
        paint_stroke(self.mask, self._last_pt or (x, y), (x, y), r, self._next_label, self._overwrite)
        self._last_pt = (x, y)
        self.refresh()


# ── layers ────────────────────────────────────────────────────────────────────

_LAYER_COLOURS = [("Magenta", (1.0, 0.0, 1.0)), ("Cyan", (0.0, 1.0, 1.0)), ("Yellow", (1.0, 1.0, 0.0)),
                  ("Red", (1.0, 0.0, 0.0)), ("Green", (0.0, 1.0, 0.0)), ("Blue", (0.0, 0.4, 1.0)),
                  ("Orange", (1.0, 0.55, 0.0)), ("White", (1.0, 1.0, 1.0))]


class Layer:
    """An extra image or mask shown on top of the image being corrected (like a Photoshop layer).

    Image layers are added in a colour (e.g. another channel or time point); mask layers are
    drawn as coloured outlines (e.g. the cells of another time point, or another mask version).
    """

    def __init__(self, kind: str, name: str, data: np.ndarray, colour=(1.0, 0.0, 1.0),
                 opacity: float = 0.8, visible: bool = True, path: str | None = None,
                 image_path: str | None = None, plane=None):
        self.kind, self.name, self.data = kind, name, data
        self.data0 = data                   # unshifted
        self.offset = (0, 0)                # (dx, dy) in pixels, set with the arrow buttons
        self.colour, self.opacity, self.visible = colour, opacity, visible
        self.path, self.image_path, self.plane = path, image_path, plane
        self.edges = cell_outlines(data) if kind == "mask" else None

    def move_to(self, dx: int, dy: int):
        """Show the layer shifted by (dx, dy) pixels (right / down positive); uncovered pixels are empty."""
        self.offset = (int(dx), int(dy))
        self.data = shift_array(self.data0, *self.offset)
        if self.kind == "mask":
            self.edges = cell_outlines(self.data)


def shift_array(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """*a* moved dx pixels right and dy pixels down (negative = left / up), filled with 0."""
    if dx == 0 and dy == 0:
        return a
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    if abs(dx) >= w or abs(dy) >= h:
        return out
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


def apply_layers(display: np.ndarray, layers: list, kind: str) -> np.ndarray:
    """Draw the visible *kind* layers ('image' or 'mask') onto a uint8 RGB image, in list order."""
    out = None
    for lay in layers:
        if not lay.visible or lay.kind != kind or lay.data.shape[:2] != display.shape[:2] or lay.opacity <= 0:
            continue
        if out is None:
            out = display.astype(np.float32)
        tint = np.array(lay.colour, np.float32) * 255
        if kind == "image":
            out += lay.data[:, :, None] * tint * lay.opacity
        else:
            e = lay.edges
            out[e] = out[e] * (1 - lay.opacity) + tint * lay.opacity
    return display if out is None else np.clip(out, 0, 255).astype(np.uint8)


def _crop_layer(lay: "Layer", y0, y1, x0, x1) -> "Layer":
    from copy import copy
    c = copy(lay)
    c.data = lay.data[y0:y1, x0:x1]
    if lay.edges is not None:
        c.edges = lay.edges[y0:y1, x0:x1]
    return c


class _CompareWindow(QWidget):
    """Separate window: the same part of the image as the editor shows, in four views side by side."""

    _VIEWS = [("overlay", "Image + mask"), ("image", "Image only"),
              ("cells", "Cells only (background removed)"), ("background", "Background only (cells removed)")]

    def __init__(self, parent):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Compare views")
        self.resize(1100, 820)
        from s2q.ui.line_profile import themed_window
        themed_window(self)
        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)
        self._labels = {}
        for i, (key, title) in enumerate(self._VIEWS):
            box = QVBoxLayout()
            t = QLabel(title)
            t.setObjectName("SectionTitle")
            box.addWidget(t)
            img = QLabel()
            img.setObjectName("ImagePanel")
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img.setMinimumSize(200, 160)
            img.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            box.addWidget(img, 1)
            grid.addLayout(box, i // 2, i % 2)
            self._labels[key] = img
        hint = QLabel("Shows the part of the image visible in the editor; it follows zoom, scrolling and edits.")
        hint.setStyleSheet(_muted_style())
        grid.addWidget(hint, 2, 0, 1, 2)

    def show_views(self, views: dict):
        for key, lbl in self._labels.items():
            rgb = views.get(key)
            if rgb is None or not rgb.size:
                continue
            pix = QPixmap.fromImage(ndarray_to_qimage(rgb))
            lbl.setPixmap(pix.scaled(lbl.width() - 4, lbl.height() - 4, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))


# ── main dialog ───────────────────────────────────────────────────────────────

class ManualCorrectionDialog(QDialog):
    """Interactive segmentation correction editor.

    Parameters
    ----------
    image_path, mask_path : str
        The pair to open first.
    pairs : list of (mask_path, image_path), optional
        All pairs in the folder, for Previous / Next navigation.
    index : int
        Position of the first pair in *pairs*.
    """

    _MAX_UNDO = 40
    _ZOOM_STEPS = (0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

    def __init__(self, image_path: str, mask_path: str, parent=None,
                 pairs: list[tuple[str, str]] | None = None, index: int = 0, new_mask_folder: str | None = None):
        super().__init__(parent)
        self._new_mask_folder = new_mask_folder      # where masks of images without one are created
        self.setMinimumSize(800, 560)
        self.resize(1440, 920)
        self.setModal(True)
        # a dialog has no maximise button by default: give it the normal window buttons
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowTitleHint |
                            Qt.WindowType.WindowSystemMenuHint | Qt.WindowType.WindowMinimizeButtonHint |
                            Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint)

        self._pairs = [(str(m), str(i)) for m, i in pairs] if pairs else [(str(mask_path), str(image_path))]
        self._index = index if pairs else 0
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._before: np.ndarray | None = None
        self._dirty = False
        self._saved_path: Path | None = None
        self._ai_worker: _AISegmentWorker | None = None
        self._measure_worker: _MeasureWorker | None = None
        self._layers: list[Layer] = []
        self._mask_offset = (0, 0)          # how far the edited mask was moved with the arrows
        self._base_visible = True           # layer 1: the opened image
        self._mask_visible = True           # layer 2: the mask being edited
        self._plane = None                 # microscopy files: series / time / channel / Z shown
        self._minfo = None
        self._view_mode = "overlay"
        self._area: np.ndarray | None = None        # area of interest (bool), None = whole image
        self._area_poly: list | None = None
        self._line = None                   # ((x0, y0), (x1, y1)) of the line profile
        self._profile_win = None
        self._prof_cache = (None, None)     # (key, channels) of the image being profiled
        self._compare: _CompareWindow | None = None
        self._measure_layers_win = None
        self._px_file = None                # µm per pixel recorded in the image file, or None
        self._mode = "rgb"
        self._intensity = 1.0
        self._brightness = 0.0
        self._contrast = 1.0
        self._last_opacity = 40

        self._load_pair(mask_path, image_path)     # raises on unreadable / mismatched pair
        self._build()
        self._apply_pair_to_ui()
        QTimer.singleShot(0, self._fit_zoom)
        QApplication.instance().installEventFilter(self)

    # ── loading ───────────────────────────────────────────────────────────

    def _load_pair(self, mask_path, image_path, plane=None):
        from s2q.core import microscopy as M
        self._pending_ai = None            # an AI result for the previous image must not land here
        minfo = None
        if M.is_microscopy(image_path):
            plane = plane or M.plane_from_mask_name(image_path, mask_path)
            minfo = M.info(image_path, plane.scene)
            plane.scene = min(plane.scene, len(minfo.scenes) - 1)
            plane.t = min(plane.t, minfo.T - 1)
            plane.z = min(plane.z, minfo.Z - 1)
            plane.c = min(plane.c, minfo.C - 1)
        orig = read_raw(image_path, plane)
        raw = normalise_image(orig)
        new_mask = not Path(mask_path).exists()
        mask = load_mask(mask_path, raw.shape)
        if mask.shape != raw.shape[:2]:
            raise ValueError(
                f"Mask size {mask.shape[1]}×{mask.shape[0]} does not match "
                f"image size {raw.shape[1]}×{raw.shape[0]}.\n"
                "Check that you picked the image this mask was made from."
            )
        self._raw_orig = orig if minfo is not None else None     # plane values, for measuring
        if orig.ndim == 3:
            orig = orig[..., :3].mean(axis=-1)
        self._plane, self._minfo, self._new_mask = plane, minfo, new_mask
        self._img_path, self._mask_path = Path(image_path), Path(mask_path)
        self._raw_img = raw                              # float32 (H, W, C) for display
        self._intensity_map = raw.mean(axis=2)           # [0, 1] for the magic wand
        self._raw_gray = orig.astype(np.float32)         # original values for the cell info
        self._mask = mask
        self._mask_backup = mask.copy()                  # for "revert all"
        self._n_channels = raw.shape[2]
        self._is_gray = self._n_channels == 1
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._dirty = False
        self._mask_offset = (0, 0)
        # layers and the area of interest stay when the next image has the same size
        # (e.g. the next time point of the same position)
        self._layers = [l for l in self._layers if l.data.shape[:2] == mask.shape]
        if self._area is not None and self._area.shape != mask.shape:
            self._area = None

    def _apply_pair_to_ui(self):
        self.setWindowTitle(f"Manual Correction — {self._mask_path.name}")
        plane_txt = ""
        if self._minfo is not None:
            p = self._plane
            plane_txt = (f"\nSeries {p.scene + 1}/{len(self._minfo.scenes)} · time {p.t + 1}/{self._minfo.T} · "
                         f"{'max projection' if p.z < 0 and self._minfo.Z > 1 else f'Z {max(p.z, 0) + 1}/{self._minfo.Z}'}")
        self._names_lbl.setText(f"Image: {self._img_path.name}{plane_txt}\nMask: {self._mask_path.name}"
                                + ("  (new — not saved yet)" if self._new_mask else ""))
        self._sync_plane_ui()
        self._sync_scale_from_file()
        self._refresh_layer_list()
        self._canvas.area_poly = self._area_poly if self._area is not None else None
        if self._line is not None:
            h, w = self._mask.shape
            if any(not (0 <= x < w and 0 <= y < h) for x, y in self._line):
                self._clear_line()          # the new image is too small for this line
        self._canvas.mask = self._mask
        self._mode_btns["rgb"].setText("Gray" if self._is_gray else "RGB")
        for m in ("r", "g", "b"):
            self._mode_btns[m].setEnabled(not self._is_gray)
        if self._is_gray and self._mode != "rgb":
            self._mode = "rgb"
            self._mode_btns["rgb"].setChecked(True)
        n = len(self._pairs)
        self._nav_lbl.setText(f"{self._index + 1} / {n}")
        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < n - 1)
        self._canvas.hover_label = 0
        self._info_lbl.setText("Move the mouse over a cell.")
        self._refresh_display()
        self._mask_changed(dirty=False)

    # ── build UI ──────────────────────────────────────────────────────────

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._canvas = _Canvas()
        self._canvas.on_mask_changed = self._end_edit
        self._canvas.on_stroke_start = self._begin_edit
        self._canvas.on_zoom_wheel = self._zoom_step
        self._canvas.on_action = self._on_canvas_action
        self._canvas.on_hover = self._on_hover
        self._canvas.on_pan = self._pan_by

        self._scroll = QScrollArea()
        self._scroll.setWidget(self._canvas)
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setObjectName("ImageCanvas")
        self._toast = _Toast(self._scroll.viewport())

        panel = QWidget()
        panel.setObjectName("SideBar")
        p_lay = QVBoxLayout(panel)
        p_lay.setContentsMargins(16, 16, 16, 16)
        p_lay.setSpacing(12)

        title = QLabel("Manual Correction")
        title.setObjectName("SectionTitle")
        p_lay.addWidget(title)
        # ── window size + panel width ──
        win_row = QHBoxLayout()
        win_row.setSpacing(4)
        self._full_btn = QPushButton("⛶ Full screen")
        self._full_btn.setProperty("variant", "small")
        self._full_btn.setToolTip("Fill the whole screen (F11). F11 or Esc goes back")
        self._full_btn.clicked.connect(self._toggle_full_screen)
        win_row.addWidget(self._full_btn)
        max_btn = QPushButton("□ Maximise")
        max_btn.setProperty("variant", "small")
        max_btn.setToolTip("Maximise / restore the window (keeps the title bar and taskbar)")
        max_btn.clicked.connect(self._toggle_maximised)
        win_row.addWidget(max_btn)
        win_row.addStretch(1)
        p_lay.addLayout(win_row)
        win_row = QHBoxLayout()
        win_row.setSpacing(4)
        pl_lbl = QLabel("Side panel size")
        pl_lbl.setStyleSheet(_muted_style())
        win_row.addWidget(pl_lbl)
        for text, width, tip in [("S", 340, "Small side panel — more room for the image"),
                                 ("M", 380, "Medium side panel"),
                                 ("L", 520, "Large side panel")]:
            b = QPushButton(text)
            b.setProperty("variant", "small")
            b.setFixedWidth(30)
            b.setToolTip(tip + " (or drag the divider left of the panel)")
            b.clicked.connect(lambda _, w=width: self._set_panel_width(w))
            win_row.addWidget(b)
        hide_btn = QPushButton("⇥")
        hide_btn.setProperty("variant", "small")
        hide_btn.setFixedWidth(30)
        hide_btn.setToolTip("Hide the side panel (Ctrl+B shows / hides it)")
        hide_btn.clicked.connect(lambda: self._set_panel_width(0))
        win_row.addWidget(hide_btn)
        win_row.addStretch(1)
        p_lay.addLayout(win_row)
        self._names_lbl = QLabel("")
        self._names_lbl.setWordWrap(True)
        self._names_lbl.setStyleSheet(_muted_style())
        p_lay.addWidget(self._names_lbl)
        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setObjectName("StatusPending")
        p_lay.addWidget(self._status_lbl)

        # ── image navigation ──────────────────────────────────────────────
        nav_grp, nl = _group("Images in this folder", QVBoxLayout)
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Previous")
        self._prev_btn.clicked.connect(lambda: self._go(-1))
        self._nav_lbl = QLabel("1 / 1")
        self._nav_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._next_btn = QPushButton("Next ▶")
        self._next_btn.clicked.connect(lambda: self._go(1))
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._nav_lbl)
        nav_row.addWidget(self._next_btn)
        nl.addLayout(nav_row)
        self._autosave_chk = QCheckBox("Autosave when moving to another image")
        self._autosave_chk.setChecked(True)
        self._measure_chk = QCheckBox("Update Excel measurements after saving")
        self._measure_chk.setChecked(True)
        nl.addWidget(self._autosave_chk)
        nl.addWidget(self._measure_chk)
        p_lay.addWidget(nav_grp)

        # ── pixel → micrometre ───────────────────────────────────────────
        from s2q.ui.widgets import settings
        st = settings()
        scale_grp, sl = _group("Pixel size (pixel → µm)", QVBoxLayout)
        self._um_chk = QCheckBox("Show sizes in micrometres (µm)")
        self._um_chk.setToolTip("Adds µm / µm² next to the pixel values: cell info, the Excel sheet "
                                "(extra µm columns), “Measure mask on images” and the line profile")
        self._um_chk.setChecked(st.value("correction/use_um", False, type=bool))
        sl.addWidget(self._um_chk)
        px_row = QHBoxLayout()
        px_row.addWidget(QLabel("1 pixel ="))
        self._px_spin = QDoubleSpinBox()
        self._px_spin.setDecimals(5)
        self._px_spin.setRange(0.00001, 10000.0)
        self._px_spin.setSingleStep(0.01)
        self._px_spin.setSuffix(" µm")
        self._px_spin.setValue(st.value("correction/pixel_size_um", 0.1, type=float))
        self._px_spin.setToolTip("Micrometres per pixel (from the microscope / objective, or the file)")
        px_row.addWidget(self._px_spin, 1)
        sl.addLayout(px_row)
        px_row = QHBoxLayout()
        self._px_line_btn = QPushButton("From line…")
        self._px_line_btn.setProperty("variant", "small")
        self._px_line_btn.setToolTip("Draw a line over something of known length (e.g. a scale bar) with the "
                                     "Line tool, then type its length in µm")
        self._px_line_btn.clicked.connect(self._pixel_size_from_line)
        px_row.addWidget(self._px_line_btn)
        self._px_file_btn = QPushButton("From file")
        self._px_file_btn.setProperty("variant", "small")
        self._px_file_btn.setToolTip("Use the pixel size recorded in the image file")
        self._px_file_btn.clicked.connect(self._pixel_size_from_file)
        px_row.addWidget(self._px_file_btn)
        sl.addLayout(px_row)
        self._px_hint = QLabel("")
        self._px_hint.setWordWrap(True)
        self._px_hint.setStyleSheet(_muted_style())
        sl.addWidget(self._px_hint)
        self._um_chk.toggled.connect(self._on_scale_changed)
        self._px_spin.valueChanged.connect(self._on_scale_changed)
        p_lay.addWidget(scale_grp)

        # ── plane of a microscopy file (series / time / channel / Z) ─────
        self._plane_grp, pl = _group("Plane  ([ / ] = previous / next time point)", QGridLayout)
        self._scene_combo = QComboBox()
        self._t_spin = QSpinBox()
        self._z_spin = QSpinBox()
        self._z_spin.setSpecialValueText("Max projection")
        self._c_combo = QComboBox()
        for r, (name, w) in enumerate([("Series", self._scene_combo), ("Time point", self._t_spin),
                                       ("Z-plane", self._z_spin), ("Channel", self._c_combo)]):
            lbl = QLabel(name)
            lbl.setStyleSheet(_muted_style())
            pl.addWidget(lbl, r, 0)
            pl.addWidget(w, r, 1)
        self._plane_hint = QLabel("Each series / time point / Z-plane has its own mask file.")
        self._plane_hint.setWordWrap(True)
        self._plane_hint.setStyleSheet(_muted_style())
        pl.addWidget(self._plane_hint, 4, 0, 1, 2)
        self._scene_combo.currentIndexChanged.connect(lambda _: self._on_plane_ui())
        self._t_spin.valueChanged.connect(lambda _: self._on_plane_ui())
        self._z_spin.valueChanged.connect(lambda _: self._on_plane_ui())
        self._c_combo.currentIndexChanged.connect(lambda _: self._on_plane_ui())
        self._plane_grp.setVisible(False)
        p_lay.addWidget(self._plane_grp)

        # ── tools ─────────────────────────────────────────────────────────
        tool_grp, tl = _group("Tools  (Ctrl = paint over neighbours)", QVBoxLayout)
        grid = QGridLayout()
        grid.setSpacing(6)
        self._tool_btns = {}
        for i, (key, label, sc, tip) in enumerate(TOOLS):
            b = QPushButton(f"{label}  ({sc})")
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setProperty("variant", "small")
            b.clicked.connect(lambda _, k=key: self._set_tool(k))
            grid.addWidget(b, i // 2, i % 2)
            self._tool_btns[key] = b
        tl.addLayout(grid)
        self._tool_hint = QLabel("")
        self._tool_hint.setWordWrap(True)
        self._tool_hint.setStyleSheet(_muted_style())
        tl.addWidget(self._tool_hint)
        self._brush_slider, _ = self._slider_row(tl, "Brush size", 2, 80, 10, lambda v: f"{v} px",
                                                 self._on_brush_changed)
        self._wand_slider, _ = self._slider_row(tl, "Wand tolerance", 1, 50, 15, lambda v: f"{v / 100:.2f}",
                                                lambda v: None)
        grow_row = QHBoxLayout()
        grow_row.addWidget(QLabel("Grow/shrink"))
        self._grow_px = QSpinBox()
        self._grow_px.setRange(1, 100)
        self._grow_px.setValue(2)
        self._grow_px.setSuffix(" px")
        self._grow_px.setToolTip("How many pixels to add to / remove from the edge of every cell")
        grow_row.addWidget(self._grow_px)
        for text, sign, tip in [("⊕ Grow", 1, "Make every cell bigger (into the background only — "
                                             "neighbouring cells never merge). Ctrl+Z to undo."),
                                ("⊖ Shrink", -1, "Make every cell smaller from its edge. Ctrl+Z to undo.")]:
            b = QPushButton(text)
            b.setProperty("variant", "small")
            b.setToolTip(tip + "  With an area selected, only the cells in the area change.")
            b.clicked.connect(lambda _, s=sign: self._grow_shrink(s))
            grow_row.addWidget(b)
        tl.addLayout(grow_row)
        p_lay.addWidget(tool_grp)

        # ── cell info ─────────────────────────────────────────────────────
        info_grp, il = _group("Cell under the mouse", QVBoxLayout)
        self._info_lbl = QLabel("Move the mouse over a cell.")
        self._info_lbl.setWordWrap(True)
        self._info_lbl.setStyleSheet(_muted_style())
        il.addWidget(self._info_lbl)
        p_lay.addWidget(info_grp)

        # ── view ──────────────────────────────────────────────────────────
        zoom_grp, zl = _group("Zoom  (+ / −, Ctrl + wheel; pan: Space + drag)", QHBoxLayout)
        for lbl, val in [("Fit", None), ("50%", 0.5), ("100%", 1.0), ("200%", 2.0)]:
            b = QPushButton(lbl)
            b.setProperty("variant", "small")
            b.clicked.connect(lambda _, v=val: self._fit_zoom() if v is None else self._set_zoom(v))
            zl.addWidget(b)
        p_lay.addWidget(zoom_grp)

        col_grp, cl = _group("Colour", QVBoxLayout)
        mode_row = QGridLayout()
        mode_row.setSpacing(6)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_btns = {}
        for i, m in enumerate(_MODES):
            b = QPushButton({"rgb": "RGB", "r": "R", "g": "G", "b": "B"}[m])
            b.setCheckable(True)
            b.setProperty("channel", m)
            b.clicked.connect(lambda _, mm=m: self._set_mode(mm))
            self._mode_group.addButton(b)
            self._mode_btns[m] = b
            mode_row.addWidget(b, 0, i)
        self._mode_btns["rgb"].setChecked(True)
        cl.addLayout(mode_row)
        self._intensity_slider, _ = self._slider_row(cl, "Intensity", 0, 300, 100, lambda v: f"{v}%",
                                                     self._on_intensity)
        self._brightness_slider, _ = self._slider_row(cl, "Brightness", -100, 100, 0, lambda v: f"{v:+d}",
                                                      self._on_brightness)
        self._contrast_slider, _ = self._slider_row(cl, "Contrast", 20, 300, 100, lambda v: f"{v}%",
                                                    self._on_contrast)
        reset_btn = QPushButton("Reset colour")
        reset_btn.setProperty("variant", "small")
        reset_btn.clicked.connect(self._reset_colour)
        cl.addWidget(reset_btn)
        p_lay.addWidget(col_grp)

        # ── layers (other channels, time points, masks) ──────────────────
        lay_grp, lg = _group("Layers  (on top of the image)", QVBoxLayout)
        self._layer_list = QListWidget()
        self._layer_list.setMinimumHeight(110)
        self._layer_list.setMaximumHeight(170)
        self._layer_list.itemChanged.connect(self._on_layer_checked)
        self._layer_list.currentRowChanged.connect(self._on_layer_selected)
        lg.addWidget(self._layer_list)
        add_row = QHBoxLayout()
        for text, slot, tip in [("+ Image…", self._add_image_layer, "Another channel / image of the same size"),
                                ("+ Mask…", self._add_mask_layer, "Another mask, drawn as coloured outlines"),
                                ("+ Time points", self._add_time_layers,
                                 "The masks (and images) of the other time points of this position")]:
            b = QPushButton(text)
            b.setProperty("variant", "small")
            b.setToolTip(tip)
            b.clicked.connect(slot)
            add_row.addWidget(b)
            if text == "+ Time points":
                self._time_layers_btn = b
        lg.addLayout(add_row)
        edit_row = QHBoxLayout()
        for text, slot, tip in [("▲", lambda: self._move_layer(-1), "Move up (drawn later, on top)"),
                                ("▼", lambda: self._move_layer(1), "Move down"),
                                ("Remove", self._remove_layer, "Remove the selected layer"),
                                ("Edit this mask", self._edit_layer_mask,
                                 "Make the selected mask layer the one you edit (the current mask becomes a layer)")]:
            b = QPushButton(text)
            b.setProperty("variant", "small")
            b.setToolTip(tip)
            b.clicked.connect(slot)
            edit_row.addWidget(b)
        lg.addLayout(edit_row)
        # ── move (align) the selected layer ──
        move_row = QHBoxLayout()
        move_row.addWidget(QLabel("Move layer"))
        self._move_step = QSpinBox()
        self._move_step.setRange(1, 500)
        self._move_step.setValue(1)
        self._move_step.setSuffix(" px")
        self._move_step.setToolTip("How far one arrow click moves the selected layer")
        move_row.addWidget(self._move_step)
        for text, dx, dy, tip in [("←", -1, 0, "Move left"), ("↑", 0, -1, "Move up"),
                                  ("↓", 0, 1, "Move down"), ("→", 1, 0, "Move right")]:
            b = QPushButton(text)
            b.setProperty("variant", "small")
            b.setFixedWidth(34)
            b.setToolTip(tip + " — aligns the selected layer with the image (layer 1 stays fixed). "
                               "Layer 2 moves the mask you edit (Ctrl+Z to undo).")
            b.setAutoRepeat(True)          # hold the button to keep moving
            b.setAutoRepeatDelay(350)
            b.setAutoRepeatInterval(60)
            b.clicked.connect(lambda _, x=dx, y=dy: self._nudge_layer(x, y))
            move_row.addWidget(b)
        reset_btn = QPushButton("Reset")
        reset_btn.setProperty("variant", "small")
        reset_btn.setToolTip("Put the selected layer back where it was")
        reset_btn.clicked.connect(self._reset_layer_offset)
        move_row.addWidget(reset_btn)
        lg.addLayout(move_row)
        self._offset_lbl = QLabel("")
        self._offset_lbl.setStyleSheet(_muted_style())
        lg.addWidget(self._offset_lbl)
        measure_btn = QPushButton("📊 Measure mask on images…")
        measure_btn.setToolTip("Lay a mask on the image layers and get each cell's mean and integrated "
                               "intensity in every image and channel (Excel / CSV)")
        measure_btn.clicked.connect(self._open_measure_layers)
        lg.addWidget(measure_btn)
        prop_row = QHBoxLayout()
        self._layer_colour = QComboBox()
        for name, rgb in _LAYER_COLOURS:
            self._layer_colour.addItem(name, rgb)
        self._layer_colour.currentIndexChanged.connect(self._on_layer_props)
        prop_row.addWidget(self._layer_colour)
        self._layer_opacity = QSlider(Qt.Orientation.Horizontal)
        self._layer_opacity.setRange(0, 200)
        self._layer_opacity.setToolTip("Layer strength / opacity")
        self._layer_opacity.valueChanged.connect(self._on_layer_props)
        prop_row.addWidget(self._layer_opacity, 1)
        lg.addLayout(prop_row)
        self._layer_hint = QLabel("Layer 1 is the opened image, layer 2 the mask you edit; added layers come "
                                  "after them. Tick a layer to show it — with all unticked the screen is empty. "
                                  "Added masks are drawn as outlines in their colour.")
        self._layer_hint.setWordWrap(True)
        self._layer_hint.setStyleSheet(_muted_style())
        lg.addWidget(self._layer_hint)
        p_lay.addWidget(lay_grp)

        # ── mask overlay ──────────────────────────────────────────────────
        mask_grp, ml = _group("Mask  (M = show / hide, O = outline, N = numbers)", QVBoxLayout)
        style_row = QHBoxLayout()
        self._fill_btn = QPushButton("Filled")
        self._outline_btn = QPushButton("Outline")
        for b, on in ((self._fill_btn, False), (self._outline_btn, True)):
            b.setCheckable(True)
            b.setProperty("variant", "small")
            b.clicked.connect(lambda _, o=on: self._set_outline(o))
            style_row.addWidget(b)
        self._fill_btn.setChecked(True)
        ml.addLayout(style_row)
        self._numbers_chk = QCheckBox("Show cell numbers (zoom in on dense images)")
        self._numbers_chk.toggled.connect(self._set_numbers)
        ml.addWidget(self._numbers_chk)
        self._opacity_slider, _ = self._slider_row(
            ml, "Opacity", 0, 100, self._last_opacity, lambda v: f"{v}%", self._on_opacity)
        view_row = QHBoxLayout()
        self._view_combo = QComboBox()
        for key, text in VIEW_MODES:
            self._view_combo.addItem(text, key)
        self._view_combo.setToolTip("V = next view")
        self._view_combo.currentIndexChanged.connect(self._on_view_mode)
        view_row.addWidget(self._view_combo, 1)
        cmp_btn = QPushButton("🗗 Compare")
        cmp_btn.setProperty("variant", "small")
        cmp_btn.setToolTip("Open a second window showing image + mask, image, cells only and background only "
                           "side by side")
        cmp_btn.clicked.connect(self._open_compare)
        view_row.addWidget(cmp_btn)
        ml.addLayout(view_row)
        p_lay.addWidget(mask_grp)

        # ── area of interest ──────────────────────────────────────────────
        area_grp, ag = _group("Area of interest  (R = Area tool)", QVBoxLayout)
        self._area_lbl = QLabel("Whole image. Use the Area tool (R) to select a region.")
        self._area_lbl.setWordWrap(True)
        self._area_lbl.setStyleSheet(_muted_style())
        ag.addWidget(self._area_lbl)
        self._freehand_chk = QCheckBox("Freehand (draw around the region)")
        self._freehand_chk.toggled.connect(lambda on: setattr(self._canvas, "area_freehand", on))
        ag.addWidget(self._freehand_chk)
        self._area_only_chk = QCheckBox("Show only the cells in the area")
        self._area_only_chk.setChecked(True)
        self._area_only_chk.toggled.connect(lambda _: self._canvas.refresh())
        ag.addWidget(self._area_only_chk)
        a_row = QHBoxLayout()
        for text, slot in [("Zoom to area", self._zoom_to_area), ("Clear", self._clear_area),
                           ("Save area cells…", self._save_area_mask)]:
            b = QPushButton(text)
            b.setProperty("variant", "small")
            b.clicked.connect(slot)
            a_row.addWidget(b)
        ag.addLayout(a_row)
        ai_area_btn = QPushButton("🤖 AI: add all missed cells in the area")
        ai_area_btn.setToolTip("Cellpose-SAM segments the selected area; every cell not in the mask yet is added")
        ai_area_btn.clicked.connect(self._ai_area)
        ag.addWidget(ai_area_btn)
        p_lay.addWidget(area_grp)

        # ── clean-up ──────────────────────────────────────────────────────
        clean_grp, cg = _group("Clean-up (whole image)", QVBoxLayout)
        small_row = QHBoxLayout()
        self._min_size = QSpinBox()
        self._min_size.setRange(1, 100000)
        self._min_size.setValue(50)
        self._min_size.setSuffix(" px")
        small_btn = QPushButton("Remove smaller than")
        small_btn.setProperty("variant", "small")
        small_btn.clicked.connect(self._remove_small)
        small_row.addWidget(small_btn)
        small_row.addWidget(self._min_size)
        cg.addLayout(small_row)
        clean_row = QHBoxLayout()
        border_btn = QPushButton("Remove border cells")
        border_btn.setProperty("variant", "small")
        border_btn.clicked.connect(self._remove_border)
        holes_btn = QPushButton("Fill all holes")
        holes_btn.setProperty("variant", "small")
        holes_btn.clicked.connect(self._fill_all_holes)
        clean_row.addWidget(border_btn)
        clean_row.addWidget(holes_btn)
        cg.addLayout(clean_row)
        p_lay.addWidget(clean_grp)

        self._stats_lbl = QLabel("")
        self._stats_lbl.setWordWrap(True)
        self._stats_lbl.setStyleSheet(_muted_style())
        p_lay.addWidget(self._stats_lbl)
        p_lay.addStretch()

        # ── action buttons ────────────────────────────────────────────────
        ur = QHBoxLayout()
        undo_btn = QPushButton("↩ Undo (Ctrl+Z)")
        undo_btn.clicked.connect(self._undo)
        redo_btn = QPushButton("↪ Redo (Ctrl+Y)")
        redo_btn.clicked.connect(self._redo)
        ur.addWidget(undo_btn)
        ur.addWidget(redo_btn)
        p_lay.addLayout(ur)
        revert_btn = QPushButton("⟲ Revert all changes to this image")
        revert_btn.clicked.connect(self._revert_all)
        p_lay.addWidget(revert_btn)
        sr = QHBoxLayout()
        save_btn = QPushButton("💾 Save (Ctrl+S)")
        save_btn.setProperty("variant", "solid-success")
        save_btn.setToolTip("Overwrite the mask file")
        save_btn.clicked.connect(lambda: self._save())
        save_as_btn = QPushButton("Save as…")
        save_as_btn.clicked.connect(self._save_as)
        sr.addWidget(save_btn, 2)
        sr.addWidget(save_as_btn, 1)
        p_lay.addLayout(sr)
        cr = QHBoxLayout()
        help_btn = QPushButton("⌨ Shortcuts (F1)")
        help_btn.clicked.connect(self._show_help)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._close_clicked)
        cr.addWidget(help_btn)
        cr.addWidget(close_btn)
        p_lay.addLayout(cr)

        panel_scroll = QScrollArea()
        panel_scroll.setWidget(panel)
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel_scroll.setMinimumWidth(0)
        panel.setMinimumWidth(300)          # the contents squeeze to the panel width (S … L)
        self._panel_scroll = panel_scroll
        panel_scroll.setStyleSheet(
            f"QScrollArea {{ background: {COLORS['bg_base']}; border: none; "
            f"border-left: 1px solid {COLORS['border']}; }}")

        # keyboard shortcuts
        for key, _, sc, _ in TOOLS:
            QShortcut(QKeySequence(sc), self, lambda k=key: self._set_tool(k))
        QShortcut(QKeySequence("M"), self, self._toggle_mask)
        QShortcut(QKeySequence("O"), self, lambda: self._set_outline(not self._canvas.outline))
        QShortcut(QKeySequence("N"), self, lambda: self._numbers_chk.setChecked(not self._numbers_chk.isChecked()))
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._redo)
        QShortcut(QKeySequence("Ctrl+S"), self, lambda: self._save())
        for k in ("+", "="):
            QShortcut(QKeySequence(k), self, lambda: self._zoom_step(1))
        QShortcut(QKeySequence("-"), self, lambda: self._zoom_step(-1))
        QShortcut(QKeySequence("F1"), self, self._show_help)
        QShortcut(QKeySequence("F11"), self, self._toggle_full_screen)
        QShortcut(QKeySequence("Ctrl+B"), self, self._toggle_panel)
        QShortcut(QKeySequence("?"), self, self._show_help)
        QShortcut(QKeySequence("PgDown"), self, lambda: self._go(1))
        QShortcut(QKeySequence("PgUp"), self, lambda: self._go(-1))
        QShortcut(QKeySequence("V"), self, lambda: self._view_combo.setCurrentIndex(
            (self._view_combo.currentIndex() + 1) % self._view_combo.count()))
        QShortcut(QKeySequence("]"), self, lambda: self._t_spin.stepBy(1) if self._plane_grp.isVisible() else None)
        QShortcut(QKeySequence("["), self, lambda: self._t_spin.stepBy(-1) if self._plane_grp.isVisible() else None)

        self._canvas.composer = self._compose
        self._compare_timer = QTimer(self)
        self._compare_timer.setSingleShot(True)
        self._compare_timer.setInterval(150)
        self._compare_timer.timeout.connect(self._update_compare)
        self._scroll.horizontalScrollBar().valueChanged.connect(lambda _: self._schedule_compare())
        self._scroll.verticalScrollBar().valueChanged.connect(lambda _: self._schedule_compare())

        # image | side panel, with a draggable divider (panel small … large)
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(True)
        self._splitter.setHandleWidth(6)
        self._splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {COLORS['border']}; }}"
            f"QSplitter::handle:hover {{ background: {COLORS.get('accent', '#2a9d8f')}; }}")
        self._splitter.addWidget(self._scroll)
        self._splitter.addWidget(panel_scroll)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setCollapsible(0, False)
        self._splitter.splitterMoved.connect(lambda *_: self._remember_panel_width())
        root.addWidget(self._splitter, 1)
        from s2q.ui.widgets import settings
        self._panel_width = max(0, settings().value("correction/panel_width", 380, type=int))
        self._last_panel_width = self._panel_width or 380
        QTimer.singleShot(0, lambda: self._set_panel_width(self._panel_width, remember=False))
        self._set_tool("pencil")

    # ── window size / side panel ──────────────────────────────────────────

    def _set_panel_width(self, width: int, remember: bool = True):
        total = max(self._splitter.width(), self.width() - 4, 400)
        width = max(0, min(int(width), total - 200))
        self._splitter.setSizes([total - width, width])
        if width:
            self._last_panel_width = width
        if remember:
            self._remember_panel_width()

    def _remember_panel_width(self):
        sizes = self._splitter.sizes()
        if len(sizes) == 2:
            self._panel_width = sizes[1]
            if sizes[1]:
                self._last_panel_width = sizes[1]
            from s2q.ui.widgets import settings
            settings().setValue("correction/panel_width", int(sizes[1]))

    def _toggle_panel(self):
        sizes = self._splitter.sizes()
        self._set_panel_width(0 if len(sizes) == 2 and sizes[1] > 0 else self._last_panel_width)

    def _toggle_full_screen(self):
        if self.isFullScreen():
            self.showMaximized() if getattr(self, "_was_max", False) else self.showNormal()
            self._full_btn.setText("⛶ Full screen")
        else:
            self._was_max = self.isMaximized()
            self.showFullScreen()
            self._full_btn.setText("⛶ Exit full screen")

    def _close_clicked(self):
        self._closing_by_button = True
        try:
            self.reject()
        finally:
            self._closing_by_button = False

    def _toggle_maximised(self):
        if self.isFullScreen():
            self._toggle_full_screen()
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def _slider_row(self, layout, name, lo, hi, value, fmt, slot):
        """Label + value readout above a horizontal slider; returns (slider, value label)."""
        head = QHBoxLayout()
        lbl = QLabel(name)
        lbl.setStyleSheet(_muted_style())
        val = QLabel(fmt(value))
        val.setStyleSheet(_muted_style())
        val.setAlignment(Qt.AlignmentFlag.AlignRight)
        head.addWidget(lbl)
        head.addWidget(val)
        layout.addLayout(head)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(value)
        s.valueChanged.connect(lambda v: (val.setText(fmt(v)), slot(v)))
        layout.addWidget(s)
        return s, val

    # ── keyboard: Space pans while held ───────────────────────────────────

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and self.isActiveWindow():
            if ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat():
                down = ev.type() == QEvent.Type.KeyPress
                self._canvas.space_down = down
                self._canvas.setCursor(Qt.CursorShape.OpenHandCursor if down else Qt.CursorShape.CrossCursor)
                return True
            if ev.key() == Qt.Key.Key_Space:
                return True
        return super().eventFilter(obj, ev)

    def _pan_by(self, dx, dy):
        h, v = self._scroll.horizontalScrollBar(), self._scroll.verticalScrollBar()
        h.setValue(int(h.value() + dx))
        v.setValue(int(v.value() + dy))

    # ── tools ─────────────────────────────────────────────────────────────

    def _set_tool(self, tool_id: str):
        if tool_id == "ai" and hasattr(self, "_canvas") and self._canvas._tool != "ai":
            QTimer.singleShot(50, self._preload_ai)
        self._canvas._tool = tool_id
        for k, b in self._tool_btns.items():
            b.setChecked(k == tool_id)
        self._tool_hint.setText(next(t[3] for t in TOOLS if t[0] == tool_id))
        self._canvas.setCursor(Qt.CursorShape.PointingHandCursor if tool_id in ("delete", "fill", "wand")
                               else Qt.CursorShape.CrossCursor)

    def _preload_ai(self):
        try:
            self._ensure_ai_model()
        except Exception as e:
            self._status(f"Could not load the AI model: {e}", "Error")

    def _on_brush_changed(self, v: int):
        self._canvas._brush_size = v

    def _on_canvas_action(self, tool, payload, overwrite):
        """Edits that need more than the brush: outline, wand, AI, delete, fill."""
        if tool == "ai":
            self._start_ai(payload)
            return
        if tool == "area":
            self._set_area(payload)
            return
        if tool == "line_move":              # dragging the line or one of its ends: follow live
            self._line = payload
            self._update_profile()
            return
        if tool == "line_set":               # move / adjust finished: show the graph again if it was closed
            self._line = payload
            self._update_profile(open_window=True)
            return
        if tool == "line_delete":            # a click on the line
            self._clear_line()
            self._status("Line deleted. Drag to draw a new one.", "Pending")
            return
        if tool == "line":
            a, b = payload
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 3:
                self._status("Drag to draw a line. Click on the line to delete it; drag the line or its "
                             "ends to move / adjust it.", "Pending")
                return
            self._line = (a, b)
            self._canvas.line_seg = self._line
            self._canvas.update()
            self._update_profile(open_window=True)
            return
        self._begin_edit()
        msg = ""
        if tool == "outline":
            path, label = payload
            n = fill_polygon(self._mask, path, label or next_label(self._mask), overwrite)
            msg = f"Outline filled ({n:,} px)." if n else "Outline too small — draw around the whole cell."
        elif tool == "wand":
            x, y = payload
            label = next_label(self._mask)
            n = magic_wand(self._mask, self._intensity_map, x, y, self._wand_slider.value() / 100,
                           label, overwrite)
            msg = f"Wand added a cell of {n:,} px." if n else "Nothing grown — click on the cell, or raise the tolerance."
        elif tool == "delete":
            (x0, y0), (x1, y1) = payload
            if abs(x1 - x0) < 3 and abs(y1 - y0) < 3:
                lab = delete_cell(self._mask, x0, y0)
                msg = f"Deleted cell #{lab}." if lab else "Click on a cell to delete it."
            else:
                n = delete_in_rect(self._mask, x0, y0, x1, y1)
                msg = f"Deleted {n} cell{'s' if n != 1 else ''}."
        elif tool == "fill":
            x, y = payload
            lab = int(self._mask[y, x])
            n = fill_holes_cell(self._mask, lab) if lab else 0
            msg = (f"Filled {n:,} px of holes in cell #{lab}." if n else
                   ("Cell #%d has no holes." % lab if lab else "Click on a cell to fill its holes."))
        self._end_edit()
        self._status(msg, "Pending")

    # ── AI add cell ───────────────────────────────────────────────────────

    def _ensure_ai_model(self):
        """Load Cellpose-SAM on the main thread (once), showing that it is happening."""
        from s2q.core.ai_assist import get_model, model_loaded
        if not model_loaded():
            self._status("Loading the AI model (first time only, a few seconds)…", "Pending", toast_ms=0)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            try:
                get_model()
            finally:
                QApplication.restoreOverrideCursor()
            self._status("AI model ready — drag a box to add all missed cells in it, or click on one cell.",
                         "Success")
        return get_model()

    def _ai_area(self):
        """🤖 button of the Area group: every missed cell inside the selected area."""
        if self._area is None:
            self._status("Select an area first (Area tool, R), then press this button.", "Warning")
            return
        ys, xs = np.nonzero(self._area)
        self._start_ai(((int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max()))), region=self._area)

    def _start_ai(self, payload, region=None):
        if self._ai_worker is not None and self._ai_worker.isRunning():
            self._status("AI is still working on the previous box…", "Warning")
            return
        (x0, y0), (x1, y1) = payload
        h, w = self._mask.shape
        from_click = region is None and (abs(x1 - x0) < 6 or abs(y1 - y0) < 6)
        sizes = np.bincount(self._mask.ravel())[1:]
        diam = int(2 * np.sqrt(np.median(sizes[sizes > 0]) / np.pi)) if (sizes > 0).any() else 40
        # a box around just one cell may fall back to the one-cell search if nothing else is found
        one_cell_box = region is None and max(abs(x1 - x0), abs(y1 - y0)) <= 3 * max(diam, 15)
        if from_click:
            # a click: use a box about twice the typical cell size around it
            half = max(24, diam)
            x0, x1, y0, y1 = x0 - half, x0 + half, y0 - half, y0 + half
        x0, x1 = sorted((max(0, x0), min(w - 1, x1)))
        y0, y1 = sorted((max(0, y0), min(h - 1, y1)))
        try:
            model = self._ensure_ai_model()
        except Exception as e:
            self._status(f"Could not load the AI model: {e}", "Error")
            return
        self._canvas.pending_box = (x0, y0, x1, y1)
        self._canvas.update()
        self._status("AI is finding the cell…" if from_click else
                     "AI is finding all missed cells in the " + ("area…" if region is not None else "box…"),
                     "Pending", toast_ms=0)
        self._tool_btns["ai"].setEnabled(False)
        self._ai_worker = _AISegmentWorker(model, self._raw_img, self._mask == 0, (x0, y0, x1, y1), from_click,
                                           region, one_cell_box)
        self._ai_for = self._mask_path          # results for another image are dropped
        self._ai_worker.done.connect(self._on_ai_done)
        self._ai_worker.failed.connect(self._on_ai_failed)
        self._ai_worker.start()

    def _on_ai_done(self, obj, how):
        self._tool_btns["ai"].setEnabled(True)
        self._canvas.pending_box = None
        self._canvas.update()
        if obj is None or (obj.dtype != bool and obj.max() == 0):
            self._status("No missed cell found there. Try a slightly larger box around the cells, "
                         "or draw them with Outline (L).", "Warning")
            return
        if getattr(self, "_ai_for", None) != self._mask_path or obj.shape != self._mask.shape:
            self._status("AI result discarded: the image was changed while it was working.", "Warning")
            return
        if self._canvas._drawing or self._canvas._path:
            self._pending_ai = (obj, how)        # applied when the current stroke / outline ends
            return
        self._apply_ai(obj, how)

    def _apply_ai(self, obj, how):
        if obj.dtype != bool:                # several cells (label image): one undo step for all
            self._begin_edit()
            start, added, px = next_label(self._mask), 0, 0
            for i in range(1, int(obj.max()) + 1):
                k = insert_object(self._mask, obj == i, 0, 0, start + added, False)
                if k:
                    added += 1
                    px += k
            self._end_edit()
            self._status(f"Added {added} cell{'s' if added != 1 else ''} ({px:,} px) using {how}. "
                         "Ctrl+Z undoes them all." if added else "Those cells already exist here.",
                         "Success" if added else "Warning")
            return
        self._begin_edit()
        n = insert_object(self._mask, obj, 0, 0, next_label(self._mask), False)
        self._end_edit()
        if n:
            self._status(f"Added a cell of {n:,} px using {how}. Ctrl+Z to undo.", "Success")
        else:
            self._status("That cell already exists here.", "Warning")

    def _on_ai_failed(self, msg):
        self._tool_btns["ai"].setEnabled(True)
        self._canvas.pending_box = None
        self._canvas.update()
        self._status(f"AI failed: {msg}", "Error")

    # ── undo / redo ───────────────────────────────────────────────────────

    def _begin_edit(self):
        self._before = self._mask.copy()

    def _end_edit(self):
        if self._before is None:
            return
        d = diff_region(self._before, self._mask)
        self._before = None
        pending = getattr(self, "_pending_ai", None)
        if pending is not None:
            # an AI result arrived during this stroke: record the stroke, then add the cell
            self._pending_ai = None
            QTimer.singleShot(0, lambda p=pending: self._apply_ai(*p) if p[0].shape == self._mask.shape else None)
        if d is None:
            self._canvas.refresh()
            return
        self._undo_stack.append(d)
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._mask_changed()

    def _undo(self):
        if not self._undo_stack:
            return
        sl, old, new = self._undo_stack.pop()
        self._mask[sl] = old
        self._redo_stack.append((sl, old, new))
        self._mask_changed()

    def _redo(self):
        if not self._redo_stack:
            return
        sl, old, new = self._redo_stack.pop()
        self._mask[sl] = new
        self._undo_stack.append((sl, old, new))
        self._mask_changed()

    def _revert_all(self):
        self._begin_edit()
        self._mask[:] = self._mask_backup
        self._end_edit()

    def _mask_changed(self, dirty: bool = True):
        if dirty:
            self._dirty = True
        if self._canvas.show_numbers:
            self._canvas.centroids = cell_centroids(self._mask)
        self._canvas.refresh()
        self._update_stats()
        if self._area is not None:
            self._update_area_label()
        self._schedule_compare()
        if self._line is not None:
            self._update_profile()           # cell bands follow the edits

    # ── clean-up ──────────────────────────────────────────────────────────

    def _remove_small(self):
        self._begin_edit()
        n = remove_small(self._mask, self._min_size.value())
        self._end_edit()
        self._status(f"Removed {n} cell{'s' if n != 1 else ''} smaller than {self._min_size.value()} px.", "Pending")

    def _grow_shrink(self, sign: int):
        """⊕ / ⊖ in Tools: every cell (or the cells in the selected area) by N pixels."""
        px = self._grow_px.value()
        labels, where = None, ""
        if self._area is not None:
            labs = np.unique(self._mask[self._area])
            labels = [int(l) for l in labs if l > 0]
            where = " in the area"
            if not labels:
                self._status("No cells in the selected area. Clear the area to change every cell.", "Warning")
                return
        self._begin_edit()
        if sign > 0:
            n = grow_cells(self._mask, px, labels)
            msg = f"Grew the cells{where} by {px} px ({n:,} px added)."
        else:
            n, gone = shrink_cells(self._mask, px, labels)
            msg = f"Shrank the cells{where} by {px} px ({n:,} px removed" + \
                  (f"; {gone} cell{'s' if gone != 1 else ''} disappeared)." if gone else ").")
        self._end_edit()
        if self._canvas.show_numbers:
            self._canvas.centroids = cell_centroids(self._mask)
        self._status(msg + " Ctrl+Z to undo.", "Pending")

    def _remove_border(self):
        self._begin_edit()
        n = remove_border(self._mask)
        self._end_edit()
        self._status(f"Removed {n} cell{'s' if n != 1 else ''} touching the border.", "Pending")

    def _fill_all_holes(self):
        self._begin_edit()
        n = fill_holes_all(self._mask)
        self._end_edit()
        self._status(f"Filled {n:,} px of holes.", "Pending")

    # ── hover info ────────────────────────────────────────────────────────

    def _on_hover(self, lab, x, y):
        if not lab:
            self._info_lbl.setText(f"Background  ·  x {x}, y {y}{self._pos_um(x, y)}  ·  "
                                   f"value {self._raw_gray[y, x]:.1f}")
            return
        from s2q.core.scale import fmt_um
        cell = self._mask == lab
        vals = self._raw_gray[cell]
        self._info_lbl.setText(
            f"Cell #{lab}\nArea: {fmt_um(int(cell.sum()), self._pixel_size(), 2)}\n"
            f"Mean intensity: {vals.mean():.1f}  (min {vals.min():.0f}, max {vals.max():.0f})\n"
            f"Mouse at x {x}, y {y}{self._pos_um(x, y)}")

    def _pos_um(self, x, y) -> str:
        k = self._pixel_size()
        return f" ({x * k:,.2f}, {y * k:,.2f} µm)" if k else ""

    # ── pixel → micrometre ────────────────────────────────────────────────

    def _pixel_size(self) -> float | None:
        """µm per pixel when “Show sizes in micrometres” is ticked, else None."""
        if not hasattr(self, "_um_chk") or not self._um_chk.isChecked():
            return None
        return float(self._px_spin.value())

    def _sync_scale_from_file(self):
        """Use the pixel size the image file records, when it records one."""
        from s2q.core.scale import read_pixel_size
        px = self._minfo.pixel_size_um if self._minfo is not None else None
        if not px:
            px = read_pixel_size(self._img_path)
        self._px_file = float(px) if px else None
        self._px_file_btn.setEnabled(self._px_file is not None)
        if self._px_file is not None and abs(self._px_spin.value() - self._px_file) > 1e-9:
            self._px_spin.blockSignals(True)
            self._px_spin.setValue(self._px_file)
            self._px_spin.blockSignals(False)
            self._save_scale()
        self._update_scale_hint()

    def _pixel_size_from_file(self):
        if self._px_file is not None:
            self._px_spin.setValue(self._px_file)
            self._um_chk.setChecked(True)

    def _pixel_size_from_line(self):
        if self._line is None:
            self._status("Draw a line first with the Line tool (along a scale bar or anything of known "
                         "length), then press “From line…” again.", "Warning", 6000)
            return
        (x0, y0), (x1, y1) = self._line
        n = float(np.hypot(x1 - x0, y1 - y0))
        if n < 1:
            self._status("The line is too short.", "Warning")
            return
        um, ok = QInputDialog.getDouble(self, "Pixel size from a line",
                                        f"The line is {n:,.1f} px long.\nHow long is it in µm?",
                                        n * self._px_spin.value(), 0.00001, 1e7, 4)
        if ok and um > 0:
            self._px_spin.setValue(um / n)
            self._um_chk.setChecked(True)
            self._status(f"Pixel size set to {um / n:.5f} µm ({n:,.1f} px = {um:g} µm).", "Success")

    def _save_scale(self):
        from s2q.ui.widgets import settings
        st = settings()
        st.setValue("correction/use_um", self._um_chk.isChecked())
        st.setValue("correction/pixel_size_um", float(self._px_spin.value()))

    def _update_scale_hint(self):
        k = self._px_spin.value()
        h, w = self._mask.shape
        if self._px_file is None:
            src = "Not recorded in the image file — type it, or use “From line…”."
        elif abs(k - self._px_file) < 1e-9:
            src = f"Read from the image file ({self._px_file:.5g} µm)."
        else:
            src = f"Changed by you (the file says {self._px_file:.5g} µm)."
        on = "" if self._um_chk.isChecked() else "  Tick the box to use it."
        self._px_hint.setText(f"{src}  Image = {w * k:,.1f} × {h * k:,.1f} µm.{on}")

    def _on_scale_changed(self, *_):
        self._save_scale()
        self._update_scale_hint()
        self._update_area_label()
        if self._line is not None:
            self._update_profile()

    # ── zoom ──────────────────────────────────────────────────────────────

    def _show_help(self):
        rows = "".join(f"<tr><td><b>{k}</b></td><td>&nbsp;&nbsp;{v}</td></tr>" for k, v in [
            ("P / E / L / W", "Pencil · Eraser (partial) · Outline + fill · Magic wand"),
            ("A", "AI add cell — drag a box around a missed cell, or click on it"),
            ("D / F", "Delete (click a cell, or drag a rectangle) · Fill holes"),
            ("R", "Area tool — select a region; only its cells are shown"),
            ("I", "Line profile — drag a line: intensity of every channel along it, integrated intensity, CSV"),
            ("V", "Next view: image + mask · cells only · background only · image · mask"),
            ("[ / ]", "Previous / next time point (microscopy files)"),
            ("Ctrl + drag", "Paint / outline / wand over neighbouring cells"),
            ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
            ("Ctrl+S", "Save (overwrites the mask; Excel is updated if ticked)"),
            ("PgUp / PgDn", "Previous / next image in the folder (autosaves)"),
            ("+ / − , Ctrl + wheel", "Zoom in / out"),
            ("Space + drag, middle mouse", "Pan"),
            ("M / O / N", "Show-hide mask · Outline view · Cell numbers"),
            ("F1 or ?", "This help"),
        ])
        QMessageBox.information(self, "Keyboard shortcuts", f"<table>{rows}</table>")

    def focus_cell(self, label: int):
        """Zoom in on cell *label*, centre it and outline it (used by Plotting → Explore)."""
        ys, xs = np.nonzero(self._mask == label)
        if not len(ys):
            self._status(f"Cell #{label} is not in this mask (it may have been edited).", "Warning")
            return
        if self._canvas._scale < 1.5:
            self._set_zoom(1.5)
        s = self._canvas._scale
        vp = self._scroll.viewport()
        self._scroll.horizontalScrollBar().setValue(int(xs.mean() * s - vp.width() / 2))
        self._scroll.verticalScrollBar().setValue(int(ys.mean() * s - vp.height() / 2))
        self._canvas.hover_label = label
        self._canvas._update_hover_poly()
        self._canvas.update()
        self._on_hover(label, int(xs.mean()), int(ys.mean()))
        self._status(f"Cell #{label} (outlined in white).", "Success")

    def _set_zoom(self, scale: float):
        self._canvas._scale = max(0.05, min(scale, 8.0))
        self._canvas.refresh()
        self._schedule_compare()

    def _fit_zoom(self):
        vp = self._scroll.viewport().size()
        h, w = self._mask.shape[:2]
        if vp.width() > 10 and vp.height() > 10:
            self._set_zoom(min(vp.width() / w, vp.height() / h, 1.0) * 0.98)

    def _zoom_step(self, direction: int):
        cur = self._canvas._scale
        if direction > 0:
            nxt = next((z for z in self._ZOOM_STEPS if z > cur + 1e-6), self._ZOOM_STEPS[-1])
        else:
            nxt = next((z for z in reversed(self._ZOOM_STEPS) if z < cur - 1e-6), self._ZOOM_STEPS[0])
        self._set_zoom(nxt)

    # ── colour / overlay ──────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        self._mode = mode
        self._mode_btns[mode].setChecked(True)
        self._redraw()

    def _on_intensity(self, v: int):
        self._intensity = v / 100.0
        self._redraw()

    def _on_brightness(self, v: int):
        self._brightness = v / 100.0
        self._redraw()

    def _on_contrast(self, v: int):
        self._contrast = v / 100.0
        self._redraw()

    def _reset_colour(self):
        self._intensity_slider.setValue(100)
        self._brightness_slider.setValue(0)
        self._contrast_slider.setValue(100)
        self._set_mode("rgb")

    def set_overlay(self, path: str):
        """Add *path* as an image layer (kept for callers of the former overlay feature)."""
        self._add_layer_from_file(path, "image")

    def _redraw(self):
        self._refresh_display()
        self._canvas.refresh()
        self._schedule_compare()

    def _refresh_display(self):
        """Rebuild the uint8 RGB display image from the raw channels (+ image layers)."""
        disp = render_view(self._raw_img, self._mode, self._intensity, self._brightness, self._contrast)
        if not self._base_visible:          # layer 1 unticked: start from black
            disp = np.zeros_like(disp)
        self._canvas.display = apply_layers(disp, self._layers, "image")

    # ── composing the view: view mode, area of interest, mask layers ──────

    def _area_mask(self) -> np.ndarray:
        """The mask as shown: with an area selected, only the cells that touch it."""
        if self._area is None or not self._area_only_chk.isChecked():
            return self._mask
        labs = np.unique(self._mask[self._area])
        labs = labs[labs > 0]
        return np.where(np.isin(self._mask, labs), self._mask, 0)

    def _compose_region(self, display, mask, area, mode, px, layers=True):
        if area is not None and self._area_only_chk.isChecked():
            display = display.copy()
            display[~area] = (display[~area] * 0.35).astype(np.uint8)     # dim outside the area
        if not self._mask_visible and mode in ("overlay", "mask"):     # layer 2 unticked
            out = display if mode == "overlay" else np.zeros_like(display)
        else:
            out = compose_view(display, mask, mode, self._canvas.opacity, self._canvas.outline, px)
        if layers and mode in ("overlay", "image", "mask"):
            out = apply_layers(out, self._layers, "mask")
        return out

    def _compose(self, px: int = 1) -> np.ndarray:
        return self._compose_region(self._canvas.display, self._area_mask(), self._area, self._view_mode, px)

    def _on_view_mode(self, _=None):
        self._view_mode = self._view_combo.currentData()
        self._canvas.refresh()
        self._schedule_compare()

    # ── compare window ────────────────────────────────────────────────────

    def _open_compare(self):
        if self._compare is None:
            self._compare = _CompareWindow(self)
        self._compare.show()
        self._compare.raise_()
        self._update_compare()

    def _schedule_compare(self):
        if self._compare is not None and self._compare.isVisible():
            self._compare_timer.start()

    def _update_compare(self):
        if self._compare is None or not self._compare.isVisible() or self._canvas.display is None:
            return
        s = self._canvas._scale
        vp = self._scroll.viewport()
        h, w = self._mask.shape
        x0 = max(0, int(self._scroll.horizontalScrollBar().value() / s))
        y0 = max(0, int(self._scroll.verticalScrollBar().value() / s))
        x1 = min(w, x0 + int(vp.width() / s) + 1)
        y1 = min(h, y0 + int(vp.height() / s) + 1)
        disp = self._canvas.display[y0:y1, x0:x1]
        mask = self._area_mask()[y0:y1, x0:x1]
        area = self._area[y0:y1, x0:x1] if self._area is not None else None
        crop_layers = self._layers
        self._layers = [_crop_layer(l, y0, y1, x0, x1) for l in crop_layers]
        try:
            views = {k: self._compose_region(disp, mask, area, k, 1) for k, _ in _CompareWindow._VIEWS}
        finally:
            self._layers = crop_layers
        self._compare.show_views(views)

    # ── area of interest ──────────────────────────────────────────────────

    def _set_area(self, payload):
        import cv2
        kind, data = payload
        h, w = self._mask.shape
        area = np.zeros((h, w), np.uint8)
        if kind == "rect":
            (xa, ya), (xb, yb) = data
            x0, x1 = sorted((xa, xb))
            y0, y1 = sorted((ya, yb))
            if x1 - x0 < 3 or y1 - y0 < 3:
                self._status("Drag a larger rectangle to select an area.", "Warning")
                return
            poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        else:
            poly = data
            if len(poly) < 3:
                self._status("Draw around the region to select it.", "Warning")
                return
        cv2.fillPoly(area, [np.array(poly, np.int32)], 1)
        if not area.any():
            return
        self._area, self._area_poly = area.astype(bool), poly
        self._canvas.area_poly = poly
        self._update_area_label()
        self._canvas.refresh()
        self._schedule_compare()

    def _update_area_label(self):
        if self._area is None:
            self._area_lbl.setText("Whole image. Use the Area tool (R) to select a region.")
            return
        labs = np.unique(self._mask[self._area])
        n = int((labs > 0).sum())
        from s2q.core.scale import fmt_um
        self._area_lbl.setText(f"Area: {fmt_um(int(self._area.sum()), self._pixel_size(), 2)} · {n:,} cell{'s' if n != 1 else ''} in it "
                               f"(cells touching the area are counted).")

    def _clear_area(self):
        self._area = self._area_poly = None
        self._canvas.area_poly = None
        self._update_area_label()
        self._canvas.refresh()
        self._schedule_compare()

    def _zoom_to_area(self):
        if self._area is None:
            self._status("No area selected — use the Area tool (R).", "Warning")
            return
        ys, xs = np.nonzero(self._area)
        vp = self._scroll.viewport()
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        self._set_zoom(min(vp.width() / bw, vp.height() / bh, 8.0) * 0.95)
        s = self._canvas._scale
        self._scroll.horizontalScrollBar().setValue(int((xs.min() + bw / 2) * s - vp.width() / 2))
        self._scroll.verticalScrollBar().setValue(int((ys.min() + bh / 2) * s - vp.height() / 2))

    def _save_area_mask(self):
        from skimage.io import imsave
        from skimage.segmentation import relabel_sequential
        if self._area is None:
            self._status("No area selected — use the Area tool (R).", "Warning")
            return
        default = self._mask_path.with_name(self._mask_path.name.replace("_cp_masks", "_area_cp_masks"))
        dest, _ = QFileDialog.getSaveFileName(self, "Save the cells of the area as a mask", str(default),
                                              "TIFF (*.tif *.tiff)")
        if not dest:
            return
        out = relabel_sequential(self._area_mask())[0]
        imsave(dest, out.astype(np.uint32 if out.max() > 65535 else np.uint16), check_contrast=False)
        self._status(f"✓ Saved {int(out.max())} cells of the area to {Path(dest).name}", "Success")

    # ── line profile ──────────────────────────────────────────────────────

    def _profile_channels(self) -> list:
        """Every channel of the image (or microscopy plane) shown, with original values."""
        p = self._plane
        key = (str(self._img_path), None if p is None else (p.scene, p.t, p.z))
        if self._prof_cache[0] == key:
            return self._prof_cache[1]
        chans = image_channels(self._img_path, p if self._minfo is not None else None)
        self._prof_cache = (key, chans)
        return chans

    def _update_profile(self, open_window: bool = False):
        from s2q.core.profile import line_profile
        from s2q.ui.line_profile import LineProfileWindow
        if self._profile_win is None:
            if not open_window:
                return
            self._profile_win = LineProfileWindow(self)
            self._profile_win.width_changed.connect(self._on_profile_width)
            self._profile_win.clear_requested.connect(self._clear_line)
            self._profile_win.width_spin.setValue(self._canvas.line_width)
        if open_window:
            self._profile_win.show()
            self._profile_win.raise_()
        if self._line is None or not self._profile_win.isVisible():
            return
        try:
            chans = self._profile_channels()
            px = self._pixel_size()
            res = line_profile(chans, *self._line, width=self._canvas.line_width, labels=self._mask, pixel_size=px)
        except Exception as e:
            self._status(f"Line profile failed: {e}", "Error")
            return
        plane = ""
        if self._minfo is not None:
            plane = f" · time {self._plane.t + 1}" + (f" · Z {self._plane.z + 1}" if self._plane.z >= 0 else
                                                       (" · max projection" if self._minfo.Z > 1 else ""))
        (x0, y0), (x1, y1) = self._line
        self._profile_win.set_result(res, f"Image: {self._img_path.name}{plane} · from ({x0}, {y0}) to ({x1}, {y1})")

    def _on_profile_width(self, w: int):
        self._canvas.line_width = w
        self._canvas.update()
        self._update_profile()

    def _clear_line(self):
        self._line = None
        self._canvas.line_seg = None
        self._canvas.update()
        if self._profile_win is not None:
            self._profile_win.hide()

    # ── layers ────────────────────────────────────────────────────────────

    _BASE_ROWS = 2       # rows 1–2 of the list: the opened image and the mask being edited

    def _refresh_layer_list(self):
        self._layer_list.blockSignals(True)
        cur = self._layer_list.currentRow()
        self._layer_list.clear()
        for n, (text, on) in enumerate([(f"1 · 🖼 Image: {self._img_path.name}", self._base_visible),
                                        (f"2 · ⬡ Mask (editing): {self._mask_path.name}", self._mask_visible)]):
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            item.setToolTip("Untick to hide. With every layer unticked the screen is empty.")
            self._layer_list.addItem(item)
        for n, lay in enumerate(self._layers, start=self._BASE_ROWS + 1):
            item = QListWidgetItem(f"{n} · " + ("🖼 " if lay.kind == "image" else "⬡ ") + lay.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if lay.visible else Qt.CheckState.Unchecked)
            col = QColor(*[int(c * 255) for c in lay.colour])
            item.setForeground(col)
            self._layer_list.addItem(item)
        self._layer_list.blockSignals(False)
        self._layer_list.setCurrentRow(min(max(cur, 0), self._layer_list.count() - 1))
        if hasattr(self, "_time_layers_btn"):
            self._time_layers_btn.setEnabled(self._minfo is not None and self._minfo.T > 1)
        self._on_layer_selected(self._layer_list.currentRow())

    def _layer_index(self, row: int | None = None) -> int:
        """Index into self._layers of a list row (−1 for the two fixed rows / nothing)."""
        r = (self._layer_list.currentRow() if row is None else row) - self._BASE_ROWS
        return r if 0 <= r < len(self._layers) else -1

    def _selected_layer(self) -> Layer | None:
        i = self._layer_index()
        return self._layers[i] if i >= 0 else None

    def _on_layer_checked(self, item):
        row = self._layer_list.row(item)
        on = item.checkState() == Qt.CheckState.Checked
        if row == 0:
            self._base_visible = on
        elif row == 1:
            self._mask_visible = on
        elif self._layer_index(row) >= 0:
            self._layers[self._layer_index(row)].visible = on
        else:
            return
        self._redraw()

    def _on_layer_selected(self, _row):
        self._show_offset()
        lay = self._selected_layer()
        for w in (self._layer_colour, self._layer_opacity):
            w.setEnabled(lay is not None)
            w.blockSignals(True)
        if lay is not None:
            i = next((k for k, (_, rgb) in enumerate(_LAYER_COLOURS) if rgb == tuple(lay.colour)), 0)
            self._layer_colour.setCurrentIndex(i)
            self._layer_opacity.setValue(int(lay.opacity * 100))
        for w in (self._layer_colour, self._layer_opacity):
            w.blockSignals(False)

    def _on_layer_props(self, _=None):
        lay = self._selected_layer()
        if lay is None:
            return
        lay.colour = tuple(self._layer_colour.currentData())
        lay.opacity = self._layer_opacity.value() / 100.0
        item = self._layer_list.currentItem()
        if item is not None:
            item.setForeground(QColor(*[int(c * 255) for c in lay.colour]))
        self._redraw()

    def _next_colour(self):
        return _LAYER_COLOURS[len(self._layers) % len(_LAYER_COLOURS)][1]

    def _image_filter(self) -> str:
        from s2q.core.microscopy import FILE_FILTER
        return f"Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;{FILE_FILTER};;All (*)"

    def _add_image_layer(self):
        path, _ = QFileDialog.getOpenFileName(self, "Image to add as a layer (same size, e.g. another channel)",
                                              str(self._img_path.parent), self._image_filter())
        if path:
            self._add_layer_from_file(path, "image")

    def _add_mask_layer(self):
        path, _ = QFileDialog.getOpenFileName(self, "Mask to add as a layer (same size)",
                                              str(self._mask_path.parent), "TIFF (*.tif *.tiff);;All (*)")
        if path:
            self._add_layer_from_file(path, "mask")

    def _add_layer_from_file(self, path: str, kind: str, plane=None, name: str | None = None,
                             visible: bool = True, image_path: str | None = None) -> bool:
        try:
            data = (load_image_normalised(path, plane=plane).mean(axis=2) if kind == "image"
                    else load_mask(path))
        except Exception as e:
            self._status(f"Cannot read {Path(path).name}: {e}", "Error")
            return False
        if data.shape[:2] != self._mask.shape:
            self._status(f"{Path(path).name} is {data.shape[1]}×{data.shape[0]}, but this image is "
                         f"{self._mask.shape[1]}×{self._mask.shape[0]}.", "Error")
            return False
        self._layers.append(Layer(kind, name or Path(path).name, data, self._next_colour(),
                                  1.0 if kind == "image" else 0.85, visible,
                                  path=path, image_path=image_path, plane=plane))
        self._refresh_layer_list()
        self._layer_list.setCurrentRow(self._BASE_ROWS + len(self._layers) - 1)
        self._redraw()
        return True

    def _add_time_layers(self):
        """Masks (shown) and images (hidden, tick to show) of every other time point of this position."""
        from copy import copy
        from s2q.core import microscopy as M
        if self._minfo is None or self._minfo.T < 2:
            self._status("This image has no other time points.", "Warning")
            return
        existing = {(l.kind, l.path, getattr(l.plane, "t", None)) for l in self._layers}
        n_masks = n_imgs = 0
        for t in range(self._minfo.T):
            if t == self._plane.t:
                continue
            pl = copy(self._plane)
            pl.t = t
            mpath = self._mask_path.parent / M.mask_name(self._img_path, self._minfo, pl)
            if mpath.exists() and ("mask", str(mpath), t) not in existing:
                n_masks += self._add_layer_from_file(str(mpath), "mask", plane=pl, name=f"t{t + 1} mask",
                                                     image_path=str(self._img_path))
            if ("image", str(self._img_path), t) not in existing:
                n_imgs += self._add_layer_from_file(str(self._img_path), "image", plane=pl,
                                                    name=f"t{t + 1} image", visible=False)
        self._status(f"Added {n_masks} mask layer(s) and {n_imgs} image layer(s) of other time points "
                     f"(tick an image layer to show it).", "Success")

    def _move_layer(self, step: int):
        r = self._layer_index()
        j = r + step
        if r >= 0 and 0 <= j < len(self._layers):
            self._layers[r], self._layers[j] = self._layers[j], self._layers[r]
            self._refresh_layer_list()
            self._layer_list.setCurrentRow(self._BASE_ROWS + j)
            self._redraw()
        elif r < 0:
            self._status("Layers 1 and 2 (image and mask being edited) stay at the bottom.", "Warning")

    def _remove_layer(self):
        r = self._layer_index()
        if r >= 0:
            self._layers.pop(r)
            self._refresh_layer_list()
            self._redraw()
        elif self._layer_list.currentRow() in (0, 1):
            self._status("Layers 1 and 2 cannot be removed — untick them to hide them.", "Warning")

    def _nudge_layer(self, sx: int, sy: int):
        """Arrow buttons: move the selected layer by one step (aligning it with the image)."""
        step = self._move_step.value()
        dx, dy = sx * step, sy * step
        row = self._layer_list.currentRow()
        if row == 0 or row < 0:
            self._status("Layer 1 (the image) is the reference and stays in place — select another layer "
                         "to move it.", "Warning")
            return
        if row == 1:                                   # the mask being edited: an undoable edit
            self._begin_edit()
            self._mask[:] = shift_array(self._mask, dx, dy)
            self._end_edit()
            ox, oy = self._mask_offset
            self._mask_offset = (ox + dx, oy + dy)
            if self._canvas.show_numbers:
                self._canvas.centroids = cell_centroids(self._mask)
        else:
            lay = self._selected_layer()
            if lay is None:
                return
            lay.move_to(lay.offset[0] + dx, lay.offset[1] + dy)
            self._redraw()
        self._show_offset()

    def _reset_layer_offset(self):
        row = self._layer_list.currentRow()
        if row == 1:
            ox, oy = self._mask_offset
            if (ox, oy) != (0, 0):
                self._begin_edit()
                self._mask[:] = shift_array(self._mask, -ox, -oy)
                self._end_edit()
                self._mask_offset = (0, 0)
                self._status("Mask moved back. Cells pushed past the image edge are not restored — "
                             "use Ctrl+Z for those.", "Pending")
        else:
            lay = self._selected_layer()
            if lay is not None and lay.offset != (0, 0):
                lay.move_to(0, 0)
                self._redraw()
        self._show_offset()

    def _show_offset(self):
        if not hasattr(self, "_offset_lbl"):
            return
        row = self._layer_list.currentRow()
        if row == 1:
            off = self._mask_offset
        else:
            lay = self._selected_layer()
            off = lay.offset if lay is not None else None
        if off is None:
            self._offset_lbl.setText("Select a layer (3, 4, … or 2 = the mask) to move it with the arrows.")
        else:
            self._offset_lbl.setText(f"Selected layer moved by x = {off[0]:+d} px, y = {off[1]:+d} px"
                                     + ("" if off != (0, 0) else "  (original position)"))

    def _open_measure_layers(self):
        """Window: the cells of one mask measured on layer 1 and the image layers."""
        from s2q.ui.measure_layers import MeasureOnImagesWindow
        masks = [(f"2 · Mask (editing): {self._mask_path.name}", lambda: self._mask.copy())]
        images = [(f"1 · {self._img_path.name}", self._profile_channels)]
        for n, lay in enumerate(self._layers, start=self._BASE_ROWS + 1):
            if lay.kind == "mask":
                masks.append((f"{n} · {lay.name}", lambda l=lay: l.data))
            elif lay.path:     # moved (aligned) layers are measured where they are shown
                images.append((f"{n} · {lay.name}" + (f" (moved {lay.offset[0]:+d}, {lay.offset[1]:+d})"
                                                       if lay.offset != (0, 0) else ""),
                               lambda l=lay: [(c, shift_array(a, *l.offset))
                                              for c, a in image_channels(l.path, l.plane)]))
        if len(images) == 1 and len(masks) == 1:
            self._status("Tip: add other images / channels with “+ Image…” (or “+ Time points”) to measure "
                         "these cells on them too.", "Pending")
        info = {"Image (layer 1)": str(self._img_path), "Edited mask": str(self._mask_path)}
        if self._pixel_size():
            info["Pixel size (µm/px)"] = f"{self._pixel_size():g}"
        if self._plane is not None:
            p = self._plane
            info["Plane"] = f"series {p.scene + 1}, time {p.t + 1}, Z {'max' if p.z < 0 else p.z + 1}"
        if self._measure_layers_win is not None:
            self._measure_layers_win.close()
        self._measure_layers_win = MeasureOnImagesWindow(
            self, masks, images, self._mask_path.parent / f"{self._mask_path.stem}_intensities", info,
            pixel_size=self._pixel_size())
        self._measure_layers_win.show()
        self._measure_layers_win.raise_()

    def _edit_layer_mask(self):
        """Swap: the selected mask layer becomes the edited mask; the current mask becomes a layer."""
        lay = self._selected_layer()
        if lay is None or lay.kind != "mask" or not lay.path:
            self._status("Select a mask layer (⬡) first.", "Warning")
            return
        if self._dirty:
            if self._autosave_chk.isChecked():
                if not self._save():
                    return
            elif not self._confirm_discard():
                return
        old = (str(self._mask_path), str(self._img_path), self._plane)
        image = lay.image_path or str(self._img_path)
        self._layers.remove(lay)
        try:
            self._load_pair(lay.path, image, lay.plane)
        except Exception as e:
            self._status(f"Cannot open {Path(lay.path).name}: {e}", "Error")
            return
        if Path(old[0]).exists():
            name = (f"t{old[2].t + 1} mask" if old[2] is not None and self._minfo and self._minfo.T > 1
                    else Path(old[0]).name)
            self._layers.append(Layer("mask", name, load_mask(old[0]), self._next_colour(), 0.85,
                                      path=old[0], image_path=old[1], plane=old[2]))
        self._apply_pair_to_ui()
        self._status(f"Now editing {self._mask_path.name}.", "Success")

    # ── plane of a microscopy file ────────────────────────────────────────

    def _sync_plane_ui(self):
        inf, p = self._minfo, self._plane
        self._plane_grp.setVisible(inf is not None and inf.multi)
        if inf is None:
            return
        for w in (self._scene_combo, self._t_spin, self._z_spin, self._c_combo):
            w.blockSignals(True)
        self._scene_combo.clear()
        self._scene_combo.addItems([f"{i + 1}: {s}" for i, s in enumerate(inf.scenes)])
        self._scene_combo.setCurrentIndex(p.scene)
        self._scene_combo.setEnabled(len(inf.scenes) > 1)
        self._t_spin.setRange(1, inf.T)
        self._t_spin.setSuffix(f" / {inf.T}")
        self._t_spin.setValue(p.t + 1)
        self._t_spin.setEnabled(inf.T > 1)
        self._z_spin.setRange(0, inf.Z)          # 0 = max projection, then 1..Z
        self._z_spin.setSuffix(f" / {inf.Z}")
        self._z_spin.setValue(0 if p.z < 0 else p.z + 1)
        self._z_spin.setEnabled(inf.Z > 1)
        self._c_combo.clear()
        names = inf.channel_names or [f"Channel {i + 1}" for i in range(inf.C)]
        self._c_combo.addItem("All channels (colour)" if inf.C > 1 else names[0], -1)
        if inf.C > 1:
            for i, n in enumerate(names[:inf.C]):
                self._c_combo.addItem(f"{i + 1}: {n}", i)
        self._c_combo.setCurrentIndex(max(0, self._c_combo.findData(p.c)))
        self._c_combo.setEnabled(inf.C > 1)
        for w in (self._scene_combo, self._t_spin, self._z_spin, self._c_combo):
            w.blockSignals(False)

    def _on_plane_ui(self):
        from s2q.core import microscopy as M
        if self._minfo is None:
            return
        new = M.Plane(scene=max(0, self._scene_combo.currentIndex()), t=self._t_spin.value() - 1,
                      z=self._z_spin.value() - 1, c=self._c_combo.currentData() if self._c_combo.count() else -1)
        new_mask = self._mask_path.parent / M.mask_name(self._img_path, self._minfo, new)
        if new_mask == self._mask_path:
            # only the channel (or nothing that changes the mask) changed: reload the picture, keep the mask
            orig = read_raw(self._img_path, new)
            self._plane, self._raw_orig = new, orig
            self._raw_img = normalise_image(orig)
            self._intensity_map = self._raw_img.mean(axis=2)
            self._raw_gray = (orig[..., :3].mean(axis=-1) if orig.ndim == 3 else orig).astype(np.float32)
            self._n_channels = self._raw_img.shape[2]
            self._is_gray = self._n_channels == 1
            self._apply_pair_to_ui()
            return
        if self._dirty:
            if self._autosave_chk.isChecked():
                if not self._save():
                    self._sync_plane_ui()
                    return
            elif not self._confirm_discard():
                self._sync_plane_ui()
                return
        try:
            self._load_pair(str(new_mask), str(self._img_path), new)
        except Exception as e:
            self._status(f"Cannot open that plane: {e}", "Error")
            self._sync_plane_ui()
            return
        self._apply_pair_to_ui()
        self._status(f"Time point {new.t + 1}: {new_mask.name}" + ("  (new mask)" if self._new_mask else ""),
                     "Pending")

    def _on_opacity(self, v: int):
        self._canvas.opacity = v / 100.0
        if v:
            self._last_opacity = v
        self._canvas.refresh()

    def _toggle_mask(self):
        self._opacity_slider.setValue(0 if self._opacity_slider.value() else self._last_opacity)

    def _set_outline(self, on: bool):
        self._canvas.outline = on
        self._fill_btn.setChecked(not on)
        self._outline_btn.setChecked(on)
        self._canvas.refresh()

    def _set_numbers(self, on: bool):
        self._canvas.show_numbers = on
        self._canvas.centroids = cell_centroids(self._mask) if on else {}
        self._canvas.update()

    # ── stats / status ────────────────────────────────────────────────────

    def _update_stats(self):
        n_cells = int(np.count_nonzero(np.bincount(self._mask.ravel())[1:]))
        self._stats_lbl.setText(
            f"Cells: {n_cells:,}{'   • unsaved changes' if self._dirty else ''}\n"
            f"Image: {self._raw_img.shape[1]}×{self._raw_img.shape[0]} px, "
            f"{'1 channel' if self._is_gray else f'{self._n_channels} channels'}")

    def _status(self, text: str, kind: str = "Pending", toast_ms: int = 3500):
        if text and hasattr(self, "_toast"):
            self._toast.show_message(text, kind, toast_ms)
        self._status_lbl.setText(text)
        self._status_lbl.setObjectName(f"Status{kind}")
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)

    # ── navigation ────────────────────────────────────────────────────────

    def _go(self, step: int):
        target = self._index + step
        if not 0 <= target < len(self._pairs):
            return
        if self._dirty:
            if self._autosave_chk.isChecked():
                if not self._save():
                    return
            elif not self._confirm_discard():
                return
        mask_path, image_path = self._pairs[target]
        try:
            if not mask_path:          # an image without a mask yet: start a new one
                from s2q.core import microscopy as M
                folder = Path(self._new_mask_folder or Path(str(image_path).rstrip("/\\")).parent)
                name = (M.mask_name(image_path, M.info(image_path), M.Plane()) if M.is_microscopy(image_path)
                        else f"{Path(image_path).stem}_cp_masks.tif")
                mask_path = str(folder / name)
                self._pairs[target] = (mask_path, image_path)
            self._load_pair(mask_path, image_path)
        except Exception as e:
            self._status(f"Cannot open {Path(mask_path).name}: {e}", "Error")
            self._pairs.pop(target)
            if target < self._index:
                self._index -= 1
            self._apply_pair_to_ui()
            return
        self._index = target
        self._apply_pair_to_ui()
        self._fit_zoom()
        self._status(f"Opened {Path(mask_path).name}", "Pending")

    def _confirm_discard(self) -> bool:
        r = QMessageBox.question(
            self, "Unsaved changes", f"Save your changes to {self._mask_path.name}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if r == QMessageBox.StandardButton.Save:
            return self._save()
        return r == QMessageBox.StandardButton.Discard

    # ── save ──────────────────────────────────────────────────────────────

    def _save(self, dest: str | None = None) -> bool:
        """Save the mask (default: overwrite the mask file) and refresh its measurements."""
        from skimage.io import imsave
        from skimage.segmentation import relabel_sequential
        dest = Path(dest) if dest else self._mask_path
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # erased cells leave gaps in the numbering; keep labels 1..N
            out = relabel_sequential(self._mask)[0]
            # uint16 holds up to 65,535 cells; more would wrap around and merge cells
            imsave(str(dest), out.astype(np.uint32 if out.max() > 65535 else np.uint16), check_contrast=False)
        except Exception as e:
            self._status(f"Save failed: {e}", "Error")
            QMessageBox.critical(self, "Save failed", str(e))
            return False
        self._mask[:] = out
        self._mask_backup = self._mask.copy()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._dirty = False
        self._saved_path = dest
        if dest == self._mask_path and self._new_mask:
            self._new_mask = False
            self._names_lbl.setText(self._names_lbl.text().replace("  (new — not saved yet)", ""))
        self._mask_changed(dirty=False)
        self._status(f"✓ Saved {dest.name}", "Success")
        if self._measure_chk.isChecked():
            self._update_measurements(dest)      # Excel + ROI image go next to the saved mask
        return True

    def _save_as(self):
        dest, _ = QFileDialog.getSaveFileName(self, "Save corrected mask as", str(self._mask_path),
                                              "TIFF (*.tif *.tiff);;PNG (*.png);;All (*)")
        if dest:
            self._save(dest)

    def _measured_image(self):
        """(name, path, pixel values or None) of the image the Excel sheet measures: the one shown —
        layer 1 when it is ticked, otherwise the top-most ticked image layer."""
        if not self._base_visible:
            for lay in reversed(self._layers):          # later in the list = drawn on top
                if lay.kind == "image" and lay.visible and lay.path:
                    values = read_raw(lay.path, lay.plane) if (lay.plane is not None or lay.offset != (0, 0)) \
                        else None
                    if values is not None and lay.offset != (0, 0):
                        values = shift_array(values, *lay.offset)     # measured where it is shown
                    return lay.name, Path(lay.path), values
        return self._img_path.name, self._img_path, self._raw_orig

    def _update_measurements(self, mask_path: Path | None = None):
        if self._measure_worker is not None and self._measure_worker.isRunning():
            self._measure_worker.wait()
        try:
            name, img_path, values = self._measured_image()
        except Exception as e:
            self._status(f"Saved, but the shown image could not be read for measuring: {e}", "Warning")
            return
        self._measure_worker = _MeasureWorker(Path(mask_path or self._mask_path), img_path, values,
                                              pixel_size=self._pixel_size())
        self._measure_worker.done.connect(
            lambda p, n=name: self._status(f"✓ Saved. Excel sheet measured on {n}: {p}", "Success"))
        self._measure_worker.failed.connect(lambda m: self._status(f"Saved, but measuring failed: {m}", "Warning"))
        self._measure_worker.start()
        self._status(f"✓ Saved — measuring the cells on {name} for the Excel sheet…", "Success")

    def saved_path(self) -> Path | None:
        return self._saved_path

    # ── closing ───────────────────────────────────────────────────────────

    def _can_close(self) -> bool:
        if self._dirty and not self._confirm_discard():
            return False
        for w in (self._ai_worker, self._measure_worker):
            if w is not None and w.isRunning():
                w.wait()
        QApplication.instance().removeEventFilter(self)
        return True

    def reject(self):
        """Close button, Esc and the window X all end here."""
        if self.isFullScreen() and QApplication.keyboardModifiers() == Qt.KeyboardModifier.NoModifier                 and not getattr(self, "_closing_by_button", False):
            self._toggle_full_screen()          # Esc in full screen: back to the normal window
            return
        if self._can_close():
            super().reject()

    def closeEvent(self, ev):
        ev.ignore()
        self.reject()


# ── stylesheet helpers ────────────────────────────────────────────────────────

def _group(title: str, layout_cls):
    grp = QGroupBox(title)
    lay = layout_cls(grp)
    lay.setSpacing(8)
    return grp, lay


def _muted_style() -> str:
    return f"color: {COLORS['text_secondary']}; font-size: 11px; background: transparent;"
