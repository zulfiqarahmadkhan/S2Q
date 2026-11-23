"""Correction page: browse masks with thumbnails and open the manual correction editor."""
import logging
import os
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from s2q.core.image_files import MASK_SUFFIX, is_derived, list_images, list_masks, original_stem
from s2q.ui.theme import COLORS
from s2q.ui.widgets import (
    browse_row, card, field_row, page_shell, remember_text, scrollable, section_header,
    show_error, variant,
)

logger = logging.getLogger(__name__)


class _PairThumbLoader(QThread):
    """Renders image + mask thumbnails off the UI thread (QImage only; no QPixmap here)."""
    thumb_ready = pyqtSignal(int, QImage)

    def __init__(self, pairs: list[tuple[str, str | None]], size: int):
        super().__init__()
        self._pairs = pairs
        self._size = size
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        from s2q.ui.manual_correction import (
            load_mask, ndarray_to_qimage, overlay_mask, render_pair_thumbnail,
        )
        import cv2
        import numpy as np
        for idx, (mask_path, image_path) in enumerate(self._pairs):
            if self._cancel:
                return
            try:
                if image_path and not mask_path:
                    # an image without a mask yet: the image alone
                    from s2q.ui.manual_correction import load_image_normalised, render_view
                    rgb = render_view(load_image_normalised(image_path, max_side=self._size))
                elif image_path:
                    rgb = render_pair_thumbnail(image_path, mask_path, self._size)
                else:
                    # no image known yet: show the mask alone on black
                    m = load_mask(mask_path)
                    f = self._size / max(m.shape)
                    m = cv2.resize(m.astype(np.float32), (max(1, int(m.shape[1] * f)), max(1, int(m.shape[0] * f))),
                                   interpolation=cv2.INTER_NEAREST).astype(np.int32)
                    rgb = overlay_mask(np.zeros((*m.shape, 3), np.uint8), m, 1.0)
                self.thumb_ready.emit(idx, ndarray_to_qimage(rgb))
            except Exception as e:
                logger.warning(f"Thumbnail failed for {mask_path}: {e}")


class CorrectionPage(QWidget):
    """Pick any mask + original image pair and open the correction editor."""

    _THUMB = 150
    _PREVIEW = 420

    def __init__(self):
        super().__init__()
        self._known_originals: dict[str, str] = {}   # stem -> path, from the last run
        self._chosen: dict[str, str] = {}            # mask path -> image picked by the user
        self._pairs: list[tuple[str, str | None]] = []
        self._loader: _PairThumbLoader | None = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        page, col = page_shell("Manual Correction",
                               "Fix segmentation masks by hand: add missed cells, erase wrong ones.")

        # ── card: folders ─────────────────────────────────────────────────
        src_lay = QVBoxLayout()
        src_lay.addLayout(section_header(
            "Folders", f"Where the masks (*{MASK_SUFFIX}.tif) are and where the images are. A folder of images without masks works too (TIFF, PNG, JPEG, BMP, GIF, WebP, JPEG 2000, ND2, LIF, CZI, LSM, OME-TIFF)."))
        src_lay.addSpacing(4)
        self.mask_dir_edit = QLineEdit()
        self.mask_dir_edit.setPlaceholderText("Folder with masks — or with images (new masks are saved here)…")
        self.mask_dir_edit.editingFinished.connect(self._refresh)
        src_lay.addLayout(field_row(
            "Mask folder", browse_row(self.mask_dir_edit, "Browse", self._pick_mask_dir)))

        self.orig_dir_edit = QLineEdit()
        self.orig_dir_edit.setPlaceholderText("Optional — defaults to the mask folder")
        self.orig_dir_edit.editingFinished.connect(self._refresh)
        src_lay.addLayout(field_row(
            "Original images folder", browse_row(self.orig_dir_edit, "Browse", self._pick_orig_dir),
            hint="Images are matched to masks by file name. If a match is missing or wrong, "
                 "select the mask below and use “Choose image…”.",
        ))
        col.addWidget(card(src_lay))

        # ── card: thumbnails + preview ────────────────────────────────────
        list_lay = QVBoxLayout()
        list_lay.addLayout(section_header(
            "Images and masks", "Click to preview; double-click to open the editor. Images marked “no mask yet” open with an empty mask to draw on."))

        body = QHBoxLayout()
        body.setSpacing(16)
        self.mask_list = QListWidget()
        self.mask_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.mask_list.setIconSize(QSize(self._THUMB, self._THUMB))
        self.mask_list.setGridSize(QSize(self._THUMB + 24, self._THUMB + 44))
        self.mask_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.mask_list.setMovement(QListWidget.Movement.Static)
        self.mask_list.setWordWrap(True)
        self.mask_list.setMinimumHeight(420)
        self.mask_list.currentRowChanged.connect(self._show_preview)
        self.mask_list.itemDoubleClicked.connect(lambda _: self._open_selected())
        body.addWidget(self.mask_list, 3)

        prev_col = QVBoxLayout()
        prev_col.setSpacing(8)
        self._preview = QLabel("Select a mask")
        self._preview.setObjectName("ImagePanel")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setFixedSize(self._PREVIEW, self._PREVIEW)
        prev_col.addWidget(self._preview)
        self._pair_lbl = QLabel("")
        self._pair_lbl.setObjectName("Hint")
        self._pair_lbl.setWordWrap(True)
        self._pair_lbl.setMaximumWidth(self._PREVIEW)
        prev_col.addWidget(self._pair_lbl)
        btns = QHBoxLayout()
        choose_btn = QPushButton("Choose image…")
        choose_btn.setToolTip("Pick the original image for the selected mask")
        choose_btn.clicked.connect(self._choose_image_for_selected)
        open_btn = variant(QPushButton("Open editor"), "primary")
        open_btn.clicked.connect(self._open_selected)
        btns.addWidget(choose_btn)
        btns.addWidget(open_btn)
        prev_col.addLayout(btns)
        refresh_btn = QPushButton("Refresh list")
        refresh_btn.clicked.connect(self._refresh)
        prev_col.addWidget(refresh_btn)
        prev_col.addStretch()
        body.addLayout(prev_col)
        list_lay.addLayout(body)
        col.addWidget(card(list_lay))

        # ── card: explicit pair ───────────────────────────────────────────
        pair_lay = QVBoxLayout()
        pair_lay.addLayout(section_header(
            "Open a specific pair — or a whole folder",
            "Pick one image (also ND2, LIF, CZI, LSM, OME-TIFF, OME-Zarr) and its mask, or pick a folder "
            "(“Folder…”) to open all its images: ◀ Previous / Next ▶ then walks through them. "
            "Images without a mask start with an empty one."))
        pair_lay.addSpacing(4)
        self.pair_image_edit = QLineEdit()
        self.pair_image_edit.setPlaceholderText("Image file — or a folder of images (or an .ome.zarr folder)…")
        img_row = browse_row(self.pair_image_edit, "File…", self._pick_pair_image)
        folder_btn = QPushButton("Folder…")
        folder_btn.setToolTip("A folder: every image in it is opened (Previous / Next to move between them)")
        folder_btn.clicked.connect(self._pick_pair_image_dir)
        img_row.addWidget(folder_btn)
        zarr_btn = QPushButton("Zarr folder…")
        zarr_btn.setToolTip("OME-Zarr images are folders (name.ome.zarr)")
        zarr_btn.clicked.connect(self._pick_pair_zarr)
        img_row.addWidget(zarr_btn)
        pair_lay.addLayout(field_row("Image file or folder", img_row))
        self.pair_mask_edit = QLineEdit()
        self.pair_mask_edit.setPlaceholderText(
            f"Mask file (*{MASK_SUFFIX}.tif) or a folder of masks — empty = new masks, saved in the mask "
            "folder above (or next to the images)")
        mask_row = browse_row(self.pair_mask_edit, "File…", self._pick_pair_mask)
        mfolder_btn = QPushButton("Folder…")
        mfolder_btn.setToolTip("Folder with the masks of the images (matched by file name); new masks go here too")
        mfolder_btn.clicked.connect(self._pick_pair_mask_dir)
        mask_row.addWidget(mfolder_btn)
        pair_lay.addLayout(field_row("Mask file or folder (optional)", mask_row))
        pair_open = variant(QPushButton("Open editor"), "primary")
        pair_open.clicked.connect(self._open_pair)
        pair_lay.addWidget(pair_open)
        col.addWidget(card(pair_lay))

        col.addStretch()
        root.addWidget(scrollable(page))
        remember_text(self.mask_dir_edit, "corr/masks")
        remember_text(self.orig_dir_edit, "corr/images")
        if self.mask_dir_edit.text():
            self._refresh()

    # ── public ────────────────────────────────────────────────────────────

    def set_source(self, mask_dir: str, originals: list[str]):
        """Called after a segmentation run with its mask folder and input images."""
        self._known_originals = {Path(p).stem: p for p in originals}
        self.mask_dir_edit.setText(mask_dir)
        self.orig_dir_edit.clear()
        self._refresh()

    # ── folder / file pickers ─────────────────────────────────────────────

    def _pick_mask_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select mask folder")
        if d:
            self.mask_dir_edit.setText(d)
            self._refresh()

    def _pick_orig_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select original images folder")
        if d:
            self.orig_dir_edit.setText(d)
            self._refresh()

    def _ask_image(self, mask_path: str, start_dir: str = "") -> str | None:
        from s2q.core.microscopy import FILE_FILTER
        from s2q.core.image_files import IMAGE_PATTERNS
        start = start_dir or self.orig_dir_edit.text().strip() or str(Path(mask_path).parent)
        image_path, _ = QFileDialog.getOpenFileName(
            self, f"Select the original image for {Path(mask_path).name}", start,
            f"All images ({IMAGE_PATTERNS} *.nd2 *.lif *.czi *.lsm);;"
            f"Images ({IMAGE_PATTERNS});;{FILE_FILTER};;All (*)")
        if not image_path:
            return None
        if is_derived(image_path):
            QMessageBox.warning(
                self, "That looks like a mask",
                f"{Path(image_path).name} looks like a mask or overlay file, not an original image.\n"
                "Pick the raw image the mask was made from.")
            return None
        return image_path

    def _pick_pair_mask(self):
        from s2q.core.image_files import IMAGE_PATTERNS
        p, _ = QFileDialog.getOpenFileName(
            self, "Select mask file", self.mask_dir_edit.text().strip(),
            f"Masks (*.tif *.tiff *.png);;All images ({IMAGE_PATTERNS});;All (*)")
        if p:
            self.pair_mask_edit.setText(p)
            guess = self._match(self._originals(), p) if Path(p).stem.endswith(MASK_SUFFIX) else None
            if guess and not self.pair_image_edit.text().strip():
                self.pair_image_edit.setText(guess)

    def _pick_pair_image(self):
        mask = self.pair_mask_edit.text().strip()
        p = self._ask_image(mask or "mask", str(Path(mask).parent) if mask else "")
        if p:
            self.pair_image_edit.setText(p)

    # ── list ──────────────────────────────────────────────────────────────

    def _originals(self) -> dict[str, str]:
        from s2q.core.image_files import list_microscopy
        from s2q.core.microscopy import base_name
        found = dict(self._known_originals)
        for folder in (self.orig_dir_edit.text().strip(), self.mask_dir_edit.text().strip()):
            if folder and os.path.isdir(folder):
                for p in list_images(folder):
                    found.setdefault(Path(p).stem, p)
                for p in list_microscopy(folder):          # ND2, LIF, CZI, LSM, OME-TIFF, OME-Zarr
                    found.setdefault(base_name(p), p)
        return found

    @staticmethod
    def _match(originals: dict, mask_path) -> str | None:
        """The image of a mask; per-plane masks of microscopy files (x_t3_cp_masks) match x.nd2."""
        from s2q.core.microscopy import strip_plane_suffix
        stem = original_stem(mask_path)
        return originals.get(stem) or originals.get(strip_plane_suffix(stem))

    def _refresh(self):
        from s2q.core.microscopy import base_name
        if self._loader is not None:
            self._loader.cancel()
            self._loader.wait()
        self.mask_list.clear()
        self._pairs = []
        self._preview.setPixmap(QPixmap())
        self._preview.setText("Select an image or mask")
        self._pair_lbl.clear()
        mask_dir = self.mask_dir_edit.text().strip()
        img_dir = self.orig_dir_edit.text().strip()
        if not (mask_dir and os.path.isdir(mask_dir)) and not (img_dir and os.path.isdir(img_dir)):
            return
        originals = self._originals()
        placeholder = QPixmap(self._THUMB, self._THUMB)
        placeholder.fill(QColor(COLORS["bg_input"]))
        with_mask = set()
        if mask_dir and os.path.isdir(mask_dir):
            for mp in list_masks(mask_dir):
                orig = self._chosen.get(str(mp)) or self._match(originals, mp)
                if orig:
                    with_mask.add(os.path.normcase(orig))
                self._pairs.append((str(mp), orig))
                item = QListWidgetItem(QIcon(placeholder), ("" if orig else "? ") + original_stem(mp))
                item.setToolTip(f"{mp.name}\nImage: {orig or 'not found — use Choose image…'}")
                self.mask_list.addItem(item)
        # images (also ND2 / LIF / CZI / LSM / OME files) that have no mask yet: open → start a new mask
        for img in sorted(set(originals.values()), key=lambda p: Path(p).name.lower()):
            if os.path.normcase(img) in with_mask or not os.path.exists(img):
                continue
            self._pairs.append(("", img))
            item = QListWidgetItem(QIcon(placeholder), f"{base_name(img)}\n(no mask yet)")
            item.setToolTip(f"{Path(img).name}\nNo mask yet — open it to draw one (saved in the mask folder, "
                            f"or next to the image if no mask folder is set).")
            self.mask_list.addItem(item)
        if not self._pairs:
            self._preview.setText(f"No images or *{MASK_SUFFIX}.tif masks in this folder.")
            return
        self._loader = _PairThumbLoader(self._pairs, self._THUMB)
        self._loader.thumb_ready.connect(self._on_thumb)
        self._loader.start()
        self.mask_list.setCurrentRow(0)

    def _on_thumb(self, idx: int, qimg: QImage):
        item = self.mask_list.item(idx)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(qimg)))

    def _show_preview(self, row: int):
        if not 0 <= row < len(self._pairs):
            return
        mask_path, image_path = self._pairs[row]
        self._pair_lbl.setText(
            f"Mask: {Path(mask_path).name if mask_path else 'none yet — opening starts a new mask'}\n"
            f"Image: {Path(image_path).name if image_path else 'not found — use Choose image…'}")
        if not mask_path:
            try:
                from s2q.ui.manual_correction import load_image_normalised, ndarray_to_qimage, render_view
                rgb = render_view(load_image_normalised(image_path, max_side=self._PREVIEW))
                self._preview.setPixmap(QPixmap.fromImage(ndarray_to_qimage(rgb)))
            except Exception as e:
                self._preview.setPixmap(QPixmap())
                self._preview.setText(f"Cannot read this image:\n{e}")
            return
        if not image_path:
            self._preview.setPixmap(QPixmap())
            self._preview.setText("No original image matched.\nUse “Choose image…”.")
            return
        try:
            from s2q.ui.manual_correction import ndarray_to_qimage, render_pair_thumbnail
            rgb = render_pair_thumbnail(image_path, mask_path, self._PREVIEW)
            self._preview.setPixmap(QPixmap.fromImage(ndarray_to_qimage(rgb)))
        except Exception as e:
            self._preview.setPixmap(QPixmap())
            self._preview.setText(f"Cannot preview this pair:\n{e}")

    def _choose_image_for_selected(self):
        row = self.mask_list.currentRow()
        if not 0 <= row < len(self._pairs):
            QMessageBox.information(self, "No mask", "Select a mask first.")
            return
        mask_path, _ = self._pairs[row]
        if not mask_path:
            QMessageBox.information(self, "No mask yet", "This is an image without a mask — just open it.")
            return
        image_path = self._ask_image(mask_path)
        if image_path:
            self._chosen[mask_path] = image_path
            self._refresh()
            self.mask_list.setCurrentRow(row)

    # ── opening the editor ────────────────────────────────────────────────

    def _open_selected(self):
        row = self.mask_list.currentRow()
        if not 0 <= row < len(self._pairs):
            QMessageBox.information(self, "No mask", "Select a mask file in the list first.")
            return
        mask_path, image_path = self._pairs[row]
        if not image_path:
            image_path = self._ask_image(mask_path)
            if not image_path:
                return
            self._chosen[mask_path] = image_path
        try:
            target = mask_path or self._new_mask_path(image_path)
        except Exception as e:
            show_error(self, "Cannot open", "Could not read this image.", str(e))
            return
        # every pair with a known image, so Previous / Next can walk the folder; images without a
        # mask keep "" and get their new mask name when the editor reaches them
        pairs, index = [], 0
        for k, (m, i) in enumerate(self._pairs):
            if k == row:
                index = len(pairs)
                pairs.append((target, image_path))
            elif i:
                pairs.append((m, i))
        self._open(target, image_path, pairs, index)
        self._refresh()
        self.mask_list.setCurrentRow(row)

    def _pick_pair_image_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select a folder of images", self.pair_image_edit.text().strip()
                                             or self.orig_dir_edit.text().strip())
        if d:
            self.pair_image_edit.setText(d)

    def _pick_pair_mask_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select the folder with the masks", self.pair_mask_edit.text().strip()
                                             or self.mask_dir_edit.text().strip())
        if d:
            self.pair_mask_edit.setText(d)

    @staticmethod
    def _is_image_folder(path: str) -> bool:
        return os.path.isdir(path) and not path.rstrip("/\\").lower().endswith(".zarr")

    def folder_pairs(self, image_dir: str, mask_dir: str) -> list[tuple[str, str]]:
        """(mask path or "" for none yet, image path) for every image in *image_dir*, sorted by name."""
        from s2q.core import microscopy as M
        from s2q.core.image_files import list_microscopy
        images = {os.path.normcase(str(p)): str(p) for p in list_images(image_dir)}
        for p in list_microscopy(image_dir):
            images.setdefault(os.path.normcase(str(p)), str(p))
        masks = {original_stem(m): str(m) for m in list_masks(mask_dir)} if os.path.isdir(mask_dir) else {}
        pairs = []
        for img in sorted(images.values(), key=lambda p: Path(p).name.lower()):
            if M.is_microscopy(img):
                try:
                    cand = str(Path(mask_dir) / M.mask_name(img, M.info(img), M.Plane()))
                except Exception:
                    cand = ""
                mask = cand if cand and os.path.isfile(cand) else masks.get(M.base_name(img), "")
            else:
                mask = masks.get(Path(img).stem, "")
            pairs.append((mask, img))
        return pairs

    def _open_folder(self, image_dir: str, mask_entry: str):
        if mask_entry and os.path.isfile(mask_entry):
            mask_entry = str(Path(mask_entry).parent)
        mask_dir = mask_entry or self.mask_dir_edit.text().strip() or image_dir
        if not os.path.isdir(mask_dir):
            QMessageBox.warning(self, "Missing mask folder", "That mask folder does not exist.")
            return
        pairs = self.folder_pairs(image_dir, mask_dir)
        if not pairs:
            QMessageBox.warning(self, "No images", f"No images found in\n{image_dir}")
            return
        mask, image = pairs[0]
        old_mask_dir = self.mask_dir_edit.text()
        self.mask_dir_edit.setText(mask_dir)          # new masks are saved in the mask folder
        try:
            target = mask or self._new_mask_path(image)
        except Exception as e:
            self.mask_dir_edit.setText(old_mask_dir)
            show_error(self, "Cannot open", "Could not read the first image.", str(e))
            return
        pairs[0] = (target, image)
        self._open(target, image, pairs, 0)
        self._refresh()

    def _pick_pair_zarr(self):
        d = QFileDialog.getExistingDirectory(self, "Select the OME-Zarr image folder (name.ome.zarr)",
                                             self.orig_dir_edit.text().strip())
        if d:
            self.pair_image_edit.setText(d)

    def _open_pair(self):
        from s2q.core import microscopy as M
        mask_path = self.pair_mask_edit.text().strip()
        image_path = self.pair_image_edit.text().strip()
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Missing image", "Choose an existing image file or folder.")
            return
        if self._is_image_folder(image_path):
            self._open_folder(image_path, mask_path)
            return
        if mask_path and os.path.isdir(mask_path):     # one image + a mask folder: its mask is matched by name
            match = next((m for m, i in self.folder_pairs(str(Path(image_path).parent), mask_path)
                          if os.path.normcase(i) == os.path.normcase(image_path)), "")
            if not match:
                old = self.mask_dir_edit.text()
                self.mask_dir_edit.setText(mask_path)
                try:
                    match = self._new_mask_path(image_path)
                finally:
                    self.mask_dir_edit.setText(old)
            mask_path = match
        if mask_path and not os.path.isfile(mask_path) and not os.path.isdir(os.path.dirname(mask_path)):
            QMessageBox.warning(self, "Missing mask", "That mask file does not exist. Leave the field empty "
                                                      "to start a new mask.")
            return
        if not mask_path:
            try:
                mask_path = self._new_mask_path(image_path)
            except Exception as e:
                show_error(self, "Cannot open", "Could not read this image.", str(e))
                return
        self._open(mask_path, image_path)
        self._refresh()

    def _new_mask_path(self, image_path: str) -> str:
        """Where a new mask for *image_path* is saved: the mask folder, or next to the image."""
        from s2q.core import microscopy as M
        folder = self.mask_dir_edit.text().strip()
        folder = Path(folder) if folder and os.path.isdir(folder) else Path(image_path.rstrip("/\\")).parent
        if M.is_microscopy(image_path):
            name = M.mask_name(image_path, M.info(image_path), M.Plane())
        else:
            name = f"{Path(image_path).stem}{MASK_SUFFIX}.tif"
        return str(folder / name)

    def _open(self, mask_path: str, image_path: str, pairs=None, index: int = 0):
        from s2q.ui.manual_correction import ManualCorrectionDialog
        folder = self.mask_dir_edit.text().strip()
        try:
            dlg = ManualCorrectionDialog(image_path, mask_path, self, pairs=pairs, index=index,
                                         new_mask_folder=folder if folder and os.path.isdir(folder) else None)
        except Exception as e:
            show_error(self, "Cannot open", "Could not open this mask + image pair.", str(e))
            return
        dlg.exec()
