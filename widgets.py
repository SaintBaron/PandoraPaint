"""Pandora Paint - UI widgets, styling and texture/imaging helpers.

Split out of the original monolithic app.py. Holds the reusable Qt widgets (colour rows,
pattern/asset panels, the Na'vi sections, the Save/Load bar), the app stylesheet (QSS), the
per-slot label tables, and the lazy-Pillow texture loader. MainWindow (main_window.py) and the
entry point (app.py) build on these; nothing here depends on MainWindow."""

from __future__ import annotations
import os
import json
import logging
import re
import threading
from dataclasses import dataclass
import numpy as np

log = logging.getLogger("pandorapaint.widgets")

from PyQt6.QtCore import (
    Qt,
    QObject,
    QRunnable,
    QThreadPool,
    pyqtSignal,
    QUrl,
    QTimer,
    QEventLoop,
    QPointF,
    QRectF,
    QSize,
)
from PyQt6.QtGui import QColor, QDesktopServices, QPalette, QFont, QPainter, QPen, QPixmap, QIcon, QPolygonF
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QFileDialog,
    QInputDialog,
    QColorDialog,
    QMessageBox,
    QFrame,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QProgressDialog,
    QSlider,
    QSpinBox,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QSizePolicy,
    QScrollArea,
)

import assets
import theme
from patterns import ColorPattern, PatternControl
from recolor_core import palette_from_pattern, recolor


SLOT_LABELS = {
    "body": [
        "Body Accent",
        "Forewing edges",
        "Wing-root streaks",
        "Tail Secondary",
        "Tail Primary",
        "Speckle flecks",
        "Body veins",
        "Fine capillaries",
        "Body Secondary",
        "Body base",
    ],
    "head": [
        "Upper neck / nape",
        "Brow / neck ridge",
        "Lip line / gums",
        "Snout-tip accents",
        "Chin tip",
        "Lower jaw / throat",
        "Jaw / cheek veins",
        "Muzzle / cheek",
        "Head base",
        "Main head / neck",
    ],
}

# Default banshee palettes (the RNF01 / rainforest banshee's own colours, from
# wildlife_banshee_rnf01_{head,body}_color_pattern.mcolorpattern, myColor1..10 ARGB -> RGB).
# These are the no-op baseline: applied to the preview/export while each box is left blank.
BANSHEE_DEFAULT_PALETTE = {
    "head": [
        "78C0DC",
        "20189B",
        "602D4C",
        "44A5C8",
        "C33E0A",
        "B25415",
        "FFFFFF",
        "FFFFFF",
        "E47C48",
        "040D2D",
    ],
    "body": [
        "78C0DC",
        "293D8D",
        "602D4C",
        "000000",
        "FFFFFF",
        "FF8400",
        "A07238",
        "982806",
        "2D337C",
        "040D2D",
    ],
}

QSS = """
* { font-family:'Inter','Inter Variable','Segoe UI Variable Text','Segoe UI','Noto Sans','DejaVu Sans',sans-serif; font-size:12px; color:#E6EAF1; }
QMainWindow, QWidget { background:#0B0D11; }
QLabel { background:transparent; }
QMenuBar { background:#13161C; color:#cfd8e3; padding:2px; }
QMenuBar::item { padding:4px 10px; background:transparent; }
QMenuBar::item:selected { background:#1C212A; }
QMenu { background:#13161C; border:none; padding:4px; }
QMenu::item { padding:5px 22px; }
QMenu::item:selected { background:#1C212A; color:#22D3EE; }
QGroupBox { background:#13161C; border:none; border-radius:0;
            margin-top:20px; padding:14px 12px 12px 12px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; subcontrol-position:top center;
            padding:5px 20px; margin-top:1px; color:#22D3EE; background:#1C212A;
            font-size:11px; font-weight:800; letter-spacing:2px; border-radius:0; }
QLineEdit { background:#0E1116; border:none; border-radius:0;
            padding:5px 8px; selection-background-color:#155E6B; }
QLineEdit:focus { background:#171D26; }
QLineEdit:disabled { background:#121519; color:#566072; }
QPushButton { background:#1C212A; border:none; border-radius:0;
            padding:7px 13px; color:#E6EAF1; }
QPushButton:hover { background:#252B36; }
QPushButton:pressed { background:#0E2A30; color:#22D3EE; }
QPushButton:disabled { background:#13161C; color:#566072; }
QPushButton#arrow { font-size:18px; font-weight:700; padding:0;
            color:#A9B6C4; background:#1C212A; }
QPushButton#arrow:hover { background:#0E2A30; color:#22D3EE; }
QPushButton#accent { background:#22D3EE; border:none; color:#04181E; font-weight:700;
            padding:9px 6px; border-radius:0; font-size:11px; letter-spacing:0px; }
QPushButton#accent:hover { background:#67E8F9; }
QPushButton#accent:disabled { background:#1A222A; color:#566072; }
QPushButton#action { background:#22D3EE; color:#04181E; font-weight:700; border:none;
            border-radius:0; padding:7px 13px; }
QPushButton#action:hover { background:#67E8F9; }
QPushButton#action:pressed { background:#1BA8BE; }
QPushButton#action:disabled { background:#1A222A; color:#566072; }
QPushButton#sectiontoggle { text-align:left; padding:9px 4px 9px 2px; font-weight:400;
            letter-spacing:2px; color:#22D3EE; background:transparent; border:none;
            border-radius:0; font-size:18px; }
QPushButton#sectiontoggle:hover { color:#67E8F9; }
QPushButton#sectiontoggle:checked { color:#22D3EE; }
QProgressBar { border:none; border-radius:3px; background:#0E1116;
            text-align:center; color:#04181E; font-size:16px; }
QProgressBar::chunk { background:#22D3EE; border-radius:2px; }
QCheckBox { color:#C7D0DC; spacing:8px; background:transparent; }
QCheckBox:disabled { color:#566072; }
QCheckBox::indicator { width:16px; height:16px; border-radius:0;
            border:2px solid #556376; background:#2A313C; }
QCheckBox::indicator:checked { background:#22D3EE; border:2px solid #22D3EE; }
QCheckBox::indicator:disabled { background:#13161C; border:1px solid #2A313C; }
QPushButton#swatch { border:2px solid #2A2F3A; border-radius:5px; padding:0; }
QPushButton#swatch:hover { border:2px solid #22D3EE; }
QLabel#subtitle { color:#5C6675; font-size:12px; font-weight:700; padding-left:2px; background:transparent;
            font-family:'Inter','Inter Variable','Segoe UI Variable Text','Segoe UI','Noto Sans','DejaVu Sans',sans-serif; }
QLabel#assetheading { color:#22D3EE; font-size:15px; font-weight:800; letter-spacing:2px;
            padding:10px 2px 3px 2px; background:transparent; }
QLabel#sectiontitle { color:#22D3EE; font-size:12px; font-weight:700;
            padding:0 0 1px 0px; background:transparent;
            font-family:'Inter','Inter Variable','Segoe UI Variable Text','Segoe UI','Noto Sans','DejaVu Sans',sans-serif; }
QToolTip { background:#0E1116; color:#d7e0ea; border:none;
            padding:5px 8px; border-radius:0; }
QLabel#legend { color:#8A93A3; font-size:11px; letter-spacing:1px;
            background:#13161C; border:none; border-radius:0; padding:7px; }
QLabel#gamepath { color:#5C6675; font-size:12px; padding:0 0 4px 2px; background:transparent; }
QFrame#hdivider { background:#2A313C; border:none; }
QComboBox { background:#1C212A; border:none; border-radius:0;
            padding:5px 10px; color:#E6EAF1; }
QComboBox:hover { background:#252B36; }
QComboBox::drop-down { border:none; width:20px; }
QComboBox QAbstractItemView { background:#13161C; border:none;
            selection-background-color:#173A42; outline:none; }
/* Spinboxes are left native: Qt stylesheets don't render the CSS border-triangle arrow
   trick, and styling the spinbox suppresses the native up/down arrows (showing empty boxes
   instead). Native rendering gives proper arrows and matches the banshee panel's spinboxes. */
QSlider::groove:horizontal { height:4px; background:#0E1116; border:none; border-radius:0; }
QSlider::sub-page:horizontal { background:#22D3EE; }
QSlider::handle:horizontal { width:12px; margin:-6px 0; background:#A9B6C4; border:none;
            border-radius:0; }
QSlider::handle:horizontal:hover { background:#22D3EE; }
QSlider:disabled { }
QSlider::sub-page:horizontal:disabled { background:#2A313C; }
QSlider::handle:horizontal:disabled { background:#3F4754; }
QStatusBar { background:#13161C; color:#8A93A3; }
QSplitter::handle { background:#1A1E25; }
QSplitter::handle:hover { background:#22D3EE; }
QScrollArea { background:transparent; border:none; }
QScrollBar:vertical { background:transparent; width:12px; margin:2px 2px 2px 0; }
QScrollBar::handle:vertical { background:#2D3540; min-height:36px; border-radius:0; }
QScrollBar::handle:vertical:hover { background:#22D3EE; }
QScrollBar::handle:vertical:pressed { background:#67E8F9; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:transparent; }
QScrollBar:horizontal { background:transparent; height:12px; margin:0 2px 2px 2px; }
QScrollBar::handle:horizontal { background:#2D3540; min-width:36px; border-radius:0; }
QScrollBar::handle:horizontal:hover { background:#22D3EE; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width:0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background:transparent; }
QTabWidget::pane { border:none; border-radius:0; top:0; background:#0B0D11; }
QTabBar::tab { background:#0F1117; color:#8A93A3; padding:9px 28px; margin-right:6px;
            font-size:21px; border:none; border-bottom:4px solid #234E57; border-radius:0; }
QTabBar::tab:hover { color:#C7D0DC; }
QTabBar::tab:selected { background:#1C212A; color:#22D3EE; border-bottom:4px solid #22D3EE; }
QTabBar::tab:disabled { background:transparent; color:#3F4754; border-bottom:4px solid #13282E; }
/* nested sub-tabs (Edit Ikran|Edit Gear, Edit Na'vi|Edit Gear): a line under the tab row marks the
   nesting; no outer box (kept borderless on request) */
QFrame#subtabbox { border:none; background:transparent; }
QTabWidget#subtabs::pane { border:none; border-top:1px solid #2A313C; background:#0B0D11; }
"""

Image = None
_pil_tried = False
_pil_lock = threading.Lock()


def _pil():
    """Import Pillow lazily and thread-safely, so its ~45 ms import stays off the cold-launch path
    (texture decoding runs on worker threads; the main thread only needs Pillow when exporting or
    loading a non-.dds image).

    The lock, plus setting _pil_tried only AFTER Image is assigned, means a concurrent decode thread
    (or the GUI thread) never observes the mid-import state - previously the first caller set
    _pil_tried=True before the ~45 ms import finished, so others saw tried=True / Image=None and
    spuriously reported Pillow as missing while it was in fact loading fine."""
    global Image, _pil_tried
    if not _pil_tried:
        with _pil_lock:
            if not _pil_tried:  # re-check under the lock; only one thread imports
                try:
                    from PIL import Image as _Im

                    Image = _Im
                except Exception:
                    Image = None
                _pil_tried = True  # set last, so no one sees tried=True while Image is still None
    return Image


def load_rgba(path):
    """RGBA uint8 loader. .dds -> AFoP STF reader, everything else -> Pillow."""
    if os.path.splitext(path)[1].lower() == ".dds":
        import stf_dds

        return stf_dds.load_dds(path)
    if _pil() is None:
        raise RuntimeError("Pillow is required to load textures")
    return np.asarray(Image.open(path).convert("RGBA"))


class _TexDecodeSignals(QObject):
    """Carries a finished texture decode back to the GUI thread."""

    done = pyqtSignal(str, str, int, object)  # key, role, generation, rgba|None


class _TexDecodeTask(QRunnable):
    """Decode one texture off the GUI thread (CPU/Pillow only - no OpenGL)."""

    def __init__(self, key, role, path, gen, signals):
        super().__init__()
        self._key, self._role, self._path = key, role, path
        self._gen, self._signals = gen, signals

    def run(self):
        try:
            arr = np.ascontiguousarray(load_rgba(self._path))
        except Exception:
            arr = None
        self._signals.done.emit(self._key, self._role, self._gen, arr)


def _dim_placeholder(field):
    """Slightly dim a hex field's placeholder (the greyed default colour shown when the box is
    blank) so it reads clearly as a placeholder rather than a typed-in value."""
    pal = field.palette()
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor("#525A66"))
    field.setPalette(pal)


def _swatch_css(rrggbb):
    return (
        f"QPushButton#swatch{{background:#{rrggbb};border:2px solid #2A2F3A;"
        f"border-radius:5px;}} QPushButton#swatch:hover{{border:2px solid {theme.accent_active()};}}"
    )


def _danger_css():
    """Red styling for destructive buttons - same flat shape as the normal buttons (border:none,
    square corners, matching padding), just in red so the intent still reads at a glance."""
    return (
        "QPushButton{background:#7F1216;color:#FFE2E2;font-weight:700;"
        "border:none;border-radius:0;padding:7px 13px;}"
        "QPushButton:hover{background:#A21A1F;}"
        "QPushButton:pressed{background:#6B0F13;}"
        "QPushButton:disabled{background:#33181A;color:#7A5658;}"
    )


def _warning_css():
    """Amber styling for the 'Clear manual picks' buttons - same flat shape as the normal buttons,
    in the same gold as the '(manual pick)' markers so the button reads as the manual-pick action."""
    return (
        "QPushButton{background:#7A5A12;color:#FFF3D6;font-weight:700;"
        "border:none;border-radius:0;padding:7px 13px;}"
        "QPushButton:hover{background:#9A7418;}"
        "QPushButton:pressed{background:#5E4710;}"
        "QPushButton:disabled{background:#2E2814;color:#7A6E50;}"
    )


def _open_in_file_manager(path):
    """Open a folder in the OS file manager (no-op if it isn't a real directory)."""
    if path and os.path.isdir(path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def scan_with_progress(parent, scan_fn, folder, title):
    """Run scan_fn(folder, progress=cb) on a background thread behind a non-blocking, indeterminate
    progress dialog that reports the running file count. Returns whatever scan_fn returns; re-raises
    any error so the caller's existing guard handles it. The UI stays responsive because the heavy
    os.walk runs off-thread while the main thread pumps the event loop (same approach as the
    texture-decode progress)."""
    dlg = QProgressDialog(
        "%s\nScanning export\u2026" % title, None, 0, 0, parent
    )  # 0,0 -> busy
    dlg.setWindowTitle("Scanning")
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.setCancelButton(None)  # a half scan is worse than waiting
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.show()

    holder = {"result": None, "error": None, "done": False}

    class _Sig(QObject):
        progress = pyqtSignal(int)
        finished = pyqtSignal(object, object)  # result, error|None

    sig = _Sig()

    def _on_progress(n):
        dlg.setLabelText("%s\nScanning export\u2026 %s files" % (title, format(n, ",")))

    def _on_finished(result, error):
        holder["result"], holder["error"], holder["done"] = result, error, True

    sig.progress.connect(_on_progress)
    sig.finished.connect(_on_finished)

    class _ScanTask(QRunnable):
        def run(self):
            last = [0]

            def cb(n):
                if n - last[0] >= 1000:  # throttle cross-thread signals
                    last[0] = n
                    sig.progress.emit(n)

            try:
                sig.finished.emit(scan_fn(folder, progress=cb), None)
            except Exception as exc:  # noqa: BLE001
                sig.finished.emit(None, exc)

    QThreadPool.globalInstance().start(_ScanTask())
    try:
        while not holder["done"]:
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.WaitForMoreEvents, 50
            )
            QApplication.processEvents()
    finally:
        dlg.close()
    if holder["error"] is not None:
        raise holder["error"]
    return holder["result"]


def choose_export_folder(panel, title):
    """Shared 'set asset folder' for the asset panels: pick a directory, reflect it in the panel's
    folder field, then resync. Relies on panel.folder / panel.export_edit / panel._resync()."""
    d = QFileDialog.getExistingDirectory(panel, title, panel.folder or "")
    if not d:
        return
    panel.folder = d
    panel.export_edit.setText(d)
    panel.export_edit.setToolTip(d)
    panel._resync()


def scan_export_folder(panel, scan_fn, asset_label):
    """Shared resync scaffolding for the asset panels: validate panel.folder, run scan_fn behind the
    progress dialog, and surface any scan error. Returns the scan result, or None if it bailed (no
    folder set, or a scan error). The caller stores the result and calls panel._commit()."""
    if not (panel.folder and os.path.isdir(panel.folder)):
        QMessageBox.information(
            panel, "No asset folder", "Set an asset folder first, then Resync."
        )
        return None
    try:
        return scan_with_progress(panel, scan_fn, panel.folder, asset_label)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.warning(
            panel, "Scan failed", "Could not scan that folder:\n%s" % exc
        )
        return None


class ColorRow(QWidget):
    """A banshee palette swatch. The DEFAULT colour is active (swatch shows it, the preview/export
    use it) while the hex box is left BLANK - the default hex appears only as greyed placeholder.
    Typing a valid 6-digit hex (or picking one) overrides the default; clearing the box reverts to
    the default. So 'no text in the box' == 'use the default'."""

    def __init__(self, index, on_change, label="", default_hex="000000"):
        super().__init__()
        self.index = index
        self.on_change = on_change
        self._default_hex = (default_hex or "000000").strip().lstrip("#").upper()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(8)
        self.swatch = QPushButton()
        self.swatch.setObjectName("swatch")
        self.swatch.setFixedSize(30, 22)
        self.swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.swatch.setToolTip("Click to pick this colour")
        self.swatch.setStyleSheet(_swatch_css(self._default_hex))
        self.swatch.clicked.connect(self._pick)
        text = f"{index + 1}. {label}" if label else f"Color {index + 1}"
        self.name = QLabel(text)
        self.name.setFixedWidth(138)
        self.name.setToolTip(
            "Coat 1 - pattern gradient stop"
            if index < 5
            else "Coat 2 - base gradient stop"
        )
        self.field = QLineEdit()
        self.field.setMaxLength(6)
        self.field.setFixedWidth(70)
        self.field.setPlaceholderText(self._default_hex)
        _dim_placeholder(self.field)
        self.field.editingFinished.connect(self._typed)
        self.field.textChanged.connect(self._on_text)
        for w in (self.swatch, self.name, self.field):
            lay.addWidget(w)
        lay.addStretch(1)

    def effective(self):
        """The active colour: the typed override if the box holds a valid 6-digit hex, else the
        default (which is also what a blank box means)."""
        t = self.field.text().strip().lstrip("#").upper()
        if len(t) == 6:
            try:
                int(t, 16)
                return t
            except ValueError:
                pass
        return self._default_hex

    def hex(self):
        return self.effective()

    def is_valid(self):
        # a BLANK box is valid (it means 'use the default'); otherwise require a 6-digit hex
        t = self.field.text().strip().lstrip("#")
        if t == "":
            return True
        if len(t) != 6:
            return False
        try:
            int(t, 16)
        except ValueError:
            return False
        return True

    def _on_text(self, *_):
        self.swatch.setStyleSheet(_swatch_css(self.effective()))
        self._mark_validity()

    def _mark_validity(self, *_):
        self.field.setStyleSheet(
            "" if self.is_valid() else "background:#2A1518; color:#F87171;"
        )

    def set_default(self, rrggbb):
        """Set the default colour applied when the box is blank (leaves any typed override alone)."""
        self._default_hex = (rrggbb or "000000").strip().lstrip("#").upper()
        self.field.setPlaceholderText(self._default_hex)
        self.swatch.setStyleSheet(_swatch_css(self.effective()))

    def set_hex(self, rrggbb, notify=False):
        """Populate the box with an explicit override (used by Load / the colour picker)."""
        rrggbb = rrggbb.strip().lstrip("#").upper()
        self.field.blockSignals(True)
        self.field.setText(rrggbb)
        self.field.blockSignals(False)
        self.swatch.setStyleSheet(_swatch_css(self.effective()))
        self._mark_validity()
        if notify:
            self.on_change(self.index, self.effective())

    def _typed(self):
        t = self.field.text().strip().lstrip("#").upper()
        if t == "":  # cleared -> revert to default
            self.swatch.setStyleSheet(_swatch_css(self._default_hex))
            self._mark_validity()
            self.on_change(self.index, self._default_hex)
            return
        if len(t) == 6:
            try:
                int(t, 16)
                self.field.blockSignals(True)
                self.field.setText(t)
                self.field.blockSignals(False)
                self.swatch.setStyleSheet(_swatch_css(t))
                self._mark_validity()
                self.on_change(self.index, t)
            except ValueError:
                pass

    def _pick(self):
        c = QColorDialog.getColor(QColor("#" + self.effective()), self, "Pick colour")
        if c.isValid():
            self.set_hex(f"{c.red():02X}{c.green():02X}{c.blue():02X}", notify=True)


class PatternPanel(QGroupBox):
    def __init__(self, title, key, on_palette_change, on_load_texture=None):
        super().__init__(title)
        self.key = key
        self.on_palette_change = on_palette_change
        self.on_load_texture = on_load_texture
        self.cp = None
        self.ctrl = None  # PatternControl: loaded, edited, or None (neutral)
        self.control_path = None  # path of a loaded .mpatterncontrol (None if none)
        self.path = None
        self.setMaximumWidth(300)
        root = QVBoxLayout(self)
        root.setSpacing(5)
        root.setContentsMargins(10, 8, 10, 8)

        # load row: button + pattern path
        bar = QHBoxLayout()
        bar.setSpacing(6)
        load = QPushButton("Load")
        load.setFixedWidth(86)
        load.setToolTip(
            "Load a .mcolorpattern colour pattern into this panel.\n"
            "Files for the %s are usually named like:\n"
            "e.g. banshee character t1 %s color pattern.mcolorpattern" % (key, key)
        )
        load.clicked.connect(self.load_dialog)
        self.pattern_edit = QLineEdit()
        self.pattern_edit.setPlaceholderText(".mcolorpattern path")
        self.pattern_edit.setToolTip(
            "Path to this panel's colour pattern - type a path and press Enter to load"
        )
        self.pattern_edit.returnPressed.connect(self._load_entered)
        bar.addWidget(load)
        bar.addWidget(self.pattern_edit, 1)
        root.addLayout(bar)

        _hrule(root)
        root.addWidget(self._section_title("Textures"))
        self.tex_edits = {}
        for role, name, suf in (
            ("color", "Base", "_d"),
            ("material", "Mat", "_m"),
            ("pattern", "Coat", "_pc"),
        ):
            row = QHBoxLayout()
            row.setSpacing(6)
            btn = QPushButton(f"{name} ({suf})")
            btn.setFixedWidth(86)
            btn.setToolTip(f"Load this mesh's {name} texture  ({suf})")
            btn.clicked.connect(lambda _=False, r=role: self._browse_texture(r))
            edit = QLineEdit()
            edit.setPlaceholderText(f"{suf} texture path")
            edit.setToolTip(
                f"Path to the {name} texture ({suf}) - "
                "type a path and press Enter to load"
            )
            edit.returnPressed.connect(lambda r=role: self._texture_entered(r))
            self.tex_edits[role] = edit
            row.addWidget(btn)
            row.addWidget(edit, 1)
            root.addLayout(row)

        _hrule(root)
        root.addWidget(self._section_title("Pattern Control"))
        # load a .mpatterncontrol to fill the fields; the four fields are editable and
        # update the preview live.
        cbar = QHBoxLayout()
        cbar.setSpacing(6)
        cload = QPushButton("Load")
        cload.setFixedWidth(86)
        cload.setToolTip(
            "Load this panel's .mpatterncontrol; its values fill the fields below"
        )
        cload.clicked.connect(self.load_control_dialog)
        self.control_edit = QLineEdit()
        self.control_edit.setPlaceholderText(".mpatterncontrol path")
        self.control_edit.setToolTip(
            "Path to a loaded pattern control - type a path and press Enter to load"
        )
        self.control_edit.returnPressed.connect(self._load_control_entered)
        cbar.addWidget(cload)
        cbar.addWidget(self.control_edit, 1)
        root.addLayout(cbar)

        self._ctrl_guard = False
        self._ctrl_name = ""
        self._ctrl_uid = ""
        self.ctrl_spins = {}
        root.addSpacing(8)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(16)
        grid.setContentsMargins(0, 4, 0, 4)
        specs = [
            (
                "level1",
                "Lvl 1",
                "Level 1",
                0.0,
                8.0,
                0.1,
                "Coverage threshold for Coat 1 (colours 1-5). Slides the cutoff on the "
                "pattern coat's red channel - sets how much of the surface Coat 1 covers.",
            ),
            (
                "level2",
                "Lvl 2",
                "Level 2",
                0.0,
                8.0,
                0.1,
                "Coverage threshold for Coat 2 (colours 6-10). Slides the cutoff on the "
                "pattern coat's green channel - sets how much of the surface Coat 2 covers.",
            ),
            (
                "invert1",
                "Inv. 1",
                "Invert 1",
                -1.0,
                1.0,
                0.1,
                "Placement of Coat 1: +1 normal, 0 hides it, -1 inverts it (Coat 1 shows "
                "where it previously didn't). In-between values scale the effect.",
            ),
            (
                "invert2",
                "Inv. 2",
                "Invert 2",
                -1.0,
                1.0,
                0.1,
                "Placement of Coat 2: +1 normal, 0 hides it, -1 inverts it. In-between "
                "values scale the effect.",
            ),
        ]
        for i, (attr, lbl, full, lo, hi, step, tip) in enumerate(specs):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(2)
            sb.setFixedHeight(28)
            sb.setMinimumWidth(90)
            sb.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            tooltip = (
                f"<qt><b>{full}</b><br>{tip}"
                f"<br><i>Edits update the preview live.</i></qt>"
            )
            sb.setToolTip(tooltip)
            sb.valueChanged.connect(self._control_edited)
            self.ctrl_spins[attr] = sb
            lab = QLabel(lbl)
            lab.setToolTip(tooltip)  # hovering the label shows the same hint
            lab.setFixedHeight(28)  # match the spin box so vertical centring lines up
            lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            r, c = i // 2, (i % 2) * 2
            grid.addWidget(lab, r, c)
            grid.addWidget(sb, r, c + 1)
        # labels hug a fixed-width column; the two spin columns share the rest equally
        grid.setColumnMinimumWidth(0, 36)
        grid.setColumnMinimumWidth(2, 36)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        root.addLayout(grid)
        self._populate_control_fields(None)  # neutral defaults

        _hrule(root)
        root.addWidget(self._section_title("Colours"))
        labels = SLOT_LABELS.get(key, [""] * 10)
        defaults = BANSHEE_DEFAULT_PALETTE.get(key, ["000000"] * 10)
        self.rows = [
            ColorRow(i, self._row_changed, labels[i], default_hex=defaults[i])
            for i in range(10)
        ]
        for r in self.rows:
            root.addWidget(r)

        root.addSpacing(10)
        self.overwrite = QCheckBox("Overwrite file")
        self.overwrite.setToolTip(
            "Save over this panel's loaded .mcolorpattern instead "
            "of picking a new file (needs a pattern loaded)"
        )
        owrow = QHBoxLayout()
        owrow.setContentsMargins(0, 0, 0, 0)
        owrow.addWidget(self.overwrite)
        owrow.addStretch(1)
        reset = QPushButton("Reset Colour")
        reset.setToolTip("Clear every colour in this panel back to its default")
        reset.setStyleSheet(_danger_css())
        reset.clicked.connect(self.reset_colours)
        owrow.addWidget(reset)
        root.addLayout(owrow)
        root.addSpacing(12)
        exprow = QHBoxLayout()
        exprow.setSpacing(6)
        ep = QPushButton("Export Pattern")
        ep.setObjectName("accent")
        ep.setToolTip("Save this panel's colours as a .mcolorpattern file")
        ep.clicked.connect(self._export_pattern)
        et = QPushButton("Export as Texture")
        et.setObjectName("accent")
        et.setToolTip(
            "Bake the painted colours onto this mesh's texture and save a PNG"
        )
        et.clicked.connect(lambda: self._export_texture())
        exprow.addWidget(ep)
        exprow.addWidget(et)
        root.addLayout(exprow)
        root.addStretch(1)

        # export buttons are disabled unless every colour is a valid hex code
        self.export_btns = (ep, et)
        self.export_pattern_btn = ep
        self.export_tex_btn = et
        self._export_tips = {ep: ep.toolTip(), et: et.toolTip()}
        self.on_validity_change = None  # set by MainWindow (for Export All)
        for r in self.rows:
            r.field.textChanged.connect(self._refresh_export_enabled)
        self._refresh_export_enabled()

        # Seed the pattern with the default palette so the preview renders the default banshee
        # colours straight away while every box stays blank (the boxes only show a typed override).
        # Deferred to the event loop so the viewer is ready to receive the palette.
        self.cp = ColorPattern(name=f"{self.key} pattern", uid="0" * 32)
        for i, hexv in enumerate(BANSHEE_DEFAULT_PALETTE.get(key, ["000000"] * 10)):
            self.cp.set_rgb(i, hexv)
        QTimer.singleShot(0, self._emit)

    def all_valid(self):
        return all(r.is_valid() for r in self.rows)

    def reset_colours(self):
        """Clear every box (blank = default) and restore the default palette as the active pattern,
        reproducing the startup state for this panel."""
        for r in self.rows:
            r.set_hex("")  # blank box -> the row falls back to its default
        self.cp = ColorPattern(name=f"{self.key} pattern", uid="0" * 32)
        for i, hexv in enumerate(
            BANSHEE_DEFAULT_PALETTE.get(self.key, ["000000"] * 10)
        ):
            self.cp.set_rgb(i, hexv)
        self._refresh_export_enabled()
        self._emit()

    def export_preset(self):
        """Capture this panel's paths, control values, colours and overwrite flag for a Save/Load
        preset. Colours are stored resolved (effective hex) so the preset is self-contained and
        survives a missing .mcolorpattern."""
        return {
            "mcolorpattern": (self.path or ""),
            "mpatterncontrol": self.control_edit.text().strip(),
            "control_name": self._ctrl_name,
            "control_uid": self._ctrl_uid,
            "textures": {
                r: self.tex_edits[r].text().strip()
                for r in self.tex_edits  # live keys, so added/removed texture roles auto-sync
            },
            "controls": {
                a: float(self.ctrl_spins[a].value())
                for a in self.ctrl_spins  # live keys, so added/removed controls auto-sync
            },
            "colours": [self.rows[i].effective() for i in range(len(self.rows))],
            "overwrite": bool(self.overwrite.isChecked()),
        }

    def apply_preset(self, d, missing):
        """Restore a preset produced by export_preset(). Any file path that no longer exists is
        appended to `missing` and the panel falls back to its default (no texture / blank path /
        default colours) in its place."""
        d = d or {}
        # --- textures: load the ones that exist, flag and skip the ones that don't ---
        for r in self.tex_edits:  # live keys: a removed role is skipped, a new one restored/blanked
            p = ((d.get("textures") or {}).get(r) or "").strip()
            if p and os.path.isfile(p):
                self.set_texture_path(r, p)
                if self.on_load_texture is not None:
                    self.on_load_texture(self.key, r, p)
            else:
                if p:
                    missing.append(p)
                self.set_texture_path(r, "")
        # --- pattern control: restore the saved VALUES directly (keeps any live edits); keep the
        #     path only for reference/re-export, flag it if gone ---
        cpath = (d.get("mpatterncontrol") or "").strip()
        if cpath and not os.path.isfile(cpath):
            missing.append(cpath)
            cpath = ""
        ctrls = d.get("controls") or {}
        self._ctrl_guard = True
        for a, sb in self.ctrl_spins.items():
            if a in ctrls:
                sb.setValue(float(ctrls[a]))
        self._ctrl_name = d.get("control_name", "") or ""
        self._ctrl_uid = d.get("control_uid", "") or ""
        self._ctrl_guard = False
        self.control_edit.setText(cpath)
        self.ctrl = self._control_from_fields()
        # --- mcolorpattern path: the overwrite target; colours come from the preset, not the file ---
        ppath = (d.get("mcolorpattern") or "").strip()
        if ppath and not os.path.isfile(ppath):
            missing.append(ppath)
            ppath = ""
        self.path = ppath or None
        self.pattern_edit.setText(ppath)
        # --- colours: rebuild the active pattern from the saved (resolved) hexes ---
        cols = d.get("colours") or []
        self.cp = ColorPattern(name=f"{self.key} pattern", uid="0" * 32)
        for i, row in enumerate(self.rows):
            hexv = (cols[i] if i < len(cols) else "").strip()
            row.set_hex(hexv)  # '' -> blank box (falls back to its default)
            self.cp.set_rgb(i, row.effective())
        self.overwrite.setChecked(bool(d.get("overwrite", False)))
        self._refresh_export_enabled()
        self._emit()

    def _refresh_export_enabled(self, *_):
        valid = self.all_valid()
        loaded = bool(self.path)
        ep, et = self.export_btns  # Export Pattern, Export as Texture
        bad_hex = "Every colour must be a valid 6-digit hex code"
        # Export Pattern writes a .mcolorpattern, so it needs one loaded first; Export as Texture
        # bakes from the loaded textures and only needs valid colours.
        ep_ok = valid and loaded
        ep.setEnabled(ep_ok)
        ep.setToolTip(
            self._export_tips[ep] if ep_ok
            else ("Load a .mcolorpattern into this panel first" if not loaded else bad_hex)
        )
        et.setEnabled(valid)
        et.setToolTip(self._export_tips[et] if valid else bad_hex)
        if self.on_validity_change:
            self.on_validity_change()

    @staticmethod
    def _section_title(text):
        lbl = QLabel(text)
        lbl.setObjectName("sectiontitle")
        return lbl

    def _browse_texture(self, role):
        if self.on_load_texture is None:
            return
        p = self.on_load_texture(self.key, role, None)
        if p:
            self.tex_edits[role].setText(p)

    def _texture_entered(self, role):
        if self.on_load_texture is None:
            return
        p = self.tex_edits[role].text().strip()
        if p:
            self.on_load_texture(self.key, role, p)

    def set_texture_path(self, role, path):
        e = self.tex_edits.get(role)
        if e is not None:
            e.setText(path)

    def set_pattern(self, cp, path=None, as_default=False):
        """Apply a loaded ColorPattern. as_default=True (startup autoload) installs it as the
        blank-box default - the colours are active but every box stays blank. as_default=False
        (the explicit Load button) populates the boxes with the loaded colours as an override."""
        self.cp = cp
        self.path = path
        if path:
            self.pattern_edit.setText(path)
            log.info("Ikran %s pattern loaded: %s", self.key, path)
        for i in range(10):
            if as_default:
                self.rows[i].set_default(cp.rgb_hex(i))
            else:
                self.rows[i].set_hex(cp.rgb_hex(i))
        self._emit()
        self._refresh_export_enabled()  # the loaded path may have changed

    def _row_changed(self, index, rrggbb):
        if self.cp is None:
            self.cp = ColorPattern(name=f"{self.key} pattern", uid="0" * 32)
        self.cp.set_rgb(index, rrggbb)
        self._emit()

    def _emit(self):
        if self.cp is None:
            return
        pal = palette_from_pattern(self.cp)
        # ctrl carries the resolved/edited level/invert constants; None -> viewer neutral
        params = self.ctrl.params() if self.ctrl is not None else None
        self.on_palette_change(self.key, pal, params)

    # ---- pattern control (load / live edit) ----
    def _populate_control_fields(self, ctrl):
        """Fill the four spin boxes from a PatternControl (neutral if None) without emitting; stash its name/uid for re-export."""
        self._ctrl_guard = True
        vals = (
            ctrl.params()
            if ctrl is not None
            else dict(level1=1.0, level2=1.0, invert1=1.0, invert2=0.0)
        )
        for attr, sb in self.ctrl_spins.items():
            sb.setValue(float(vals[attr]))
        self._ctrl_name = ctrl.name if ctrl is not None else ""
        self._ctrl_uid = ctrl.uid if ctrl is not None else ""
        self._ctrl_guard = False

    def _control_from_fields(self):
        return PatternControl(
            name=self._ctrl_name,
            uid=self._ctrl_uid,
            level1=self.ctrl_spins["level1"].value(),
            level2=self.ctrl_spins["level2"].value(),
            invert1=self.ctrl_spins["invert1"].value(),
            invert2=self.ctrl_spins["invert2"].value(),
        )

    def _control_edited(self, *_):
        if self._ctrl_guard:
            return
        self.ctrl = self._control_from_fields()
        self._emit()

    def current_control(self):
        """The control as currently shown in the fields (always a PatternControl)."""
        return self._control_from_fields()

    def set_control(self, ctrl, path=None):
        """Apply a PatternControl to the fields, keep self.ctrl in sync, and emit. None resets to neutral."""
        self.ctrl = ctrl
        self.control_path = path if (ctrl is not None and path) else None
        self._populate_control_fields(ctrl)
        self.control_edit.setText(path or "")
        self._emit()
        if self.on_validity_change:  # control-loaded state gates an export option
            self.on_validity_change()

    def load_control_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load .mpatterncontrol", "", "Pattern control (*.mpatterncontrol)"
        )
        if path:
            self._load_control(path)

    def _load_control_entered(self):
        p = self.control_edit.text().strip()
        if p:
            self._load_control(p)

    def _load_control(self, path):
        try:
            self.set_control(PatternControl.load(path), path)
        except Exception as e:
            QMessageBox.warning(self, "Pattern control load failed", str(e))

    def load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load .mcolorpattern", "", "Color pattern (*.mcolorpattern)"
        )
        if path:
            self.set_pattern(ColorPattern.load(path), path)

    def _load_entered(self):
        path = self.pattern_edit.text().strip()
        if not path or not os.path.isfile(path):
            return
        try:
            self.set_pattern(ColorPattern.load(path), path)
        except Exception:  # noqa: BLE001
            log.warning("failed to load colour pattern from %s", path, exc_info=True)

    def _export_pattern(self):
        if self.cp is None:
            QMessageBox.warning(
                self, "No pattern", "Load or edit some colours before exporting."
            )
            return
        ow = self.overwrite.isChecked() and bool(self.path)
        base = (
            os.path.basename(self.path) if self.path
            else f"{self.cp.name or self.key + '_pattern'}.mcolorpattern"
        )
        # Banshee colour patterns canonically live in blue/gameplay/vanity/juice/.
        blue_rel = export_rel(BLUE_DIR_IKRAN, base)
        out, replicated = resolve_export_path(
            self, blue_rel, ow, self.path,
            default_ext=".mcolorpattern",
            file_filter="Color pattern (*.mcolorpattern)",
            title=f"Export {self.key} pattern",
        )
        if out is None:
            return
        if replicated:
            d = os.path.dirname(out)
            if d:
                os.makedirs(d, exist_ok=True)
        try:
            self.cp.save(out)
            self.path = out
            self.pattern_edit.setText(out)
        except Exception as e:  # noqa: BLE001
            log.exception("Ikran %s pattern export failed", self.key)
            QMessageBox.warning(self, "Save failed", str(e))
            return
        log.info("Ikran %s pattern export -> %s", self.key, out)
        if replicated:
            QMessageBox.information(
                self, "Export successful",
                f"Wrote:\n  {os.path.basename(out)}\n\nInto:\n  {os.path.dirname(out)}")

    def _export_texture(self, fmt=None, out_dir=None):
        """Bake the painted colours onto this panel's textures and save. When ``out_dir`` is given
        the file is written straight there (no dialogs/popups) and the written path is returned;
        otherwise a format + save dialog is shown. Returns the written path or None."""
        batch = out_dir is not None
        if _pil() is None:
            if not batch:
                QMessageBox.warning(self, "Missing dependency", "Pillow is required.")
            return None
        if self.cp is None:
            if not batch:
                QMessageBox.warning(
                    self, "No pattern", "Load or edit some colours before baking a texture."
                )
            return None
        paths = {
            r: self.tex_edits[r].text().strip()
            for r in ("color", "material", "pattern")
        }
        need = [
            ("Base (_d)", "color"),
            ("Material (_m)", "material"),
            ("Pattern Coat (_pc)", "pattern"),
        ]
        missing = [
            name for name, r in need if not (paths[r] and os.path.isfile(paths[r]))
        ]
        if missing:
            if not batch:
                QMessageBox.warning(
                    self,
                    "Missing textures",
                    "These textures must be loaded to bake the result:\n- "
                    + "\n- ".join(missing),
                )
            return None
        try:
            _pil()  # Pillow is needed for the resize/save below
            col = load_rgba(paths["color"])
            h, w = col.shape[:2]

            def load_to(p):
                arr = load_rgba(p)
                if arr.shape[0] != h or arr.shape[1] != w:
                    arr = np.asarray(
                        Image.fromarray(arr, "RGBA").resize((w, h), Image.BILINEAR)
                    )
                return arr.astype(np.float32) / 255.0

            colf = col.astype(np.float32) / 255.0
            matf = load_to(paths["material"])
            patf = load_to(paths["pattern"])
            pal = palette_from_pattern(self.cp)
            if self.ctrl is not None:
                out = recolor(
                    colf,
                    matf,
                    patf,
                    pal,
                    invert1=self.ctrl.invert1,
                    invert2=self.ctrl.invert2,
                    level1=self.ctrl.level1,
                    level2=self.ctrl.level2,
                )
            else:
                out = recolor(colf, matf, patf, pal)
            img8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        except Exception as e:
            if not batch:
                QMessageBox.warning(self, "Bake failed", str(e))
            # batch mode: skip this panel silently
            return None
        if fmt is None:
            fmt = ask_export_format(self)
            if fmt is None:
                return None
        base = os.path.splitext(os.path.basename(paths["color"]))[0]
        ext = ".png" if fmt == "png" else ".dds"
        if batch:  # write straight into the chosen folder, no dialog/popup
            out_base = os.path.join(out_dir, f"{self.key}_{base}_recoloured")
            try:
                return save_texture(Image.fromarray(img8, "RGB"), out_base, fmt)
            except Exception:  # noqa: BLE001 - skip this panel silently in batch
                return None
        filt = {
            "png": "PNG image (*.png)",
            "dds": "DDS texture (*.dds)",
            "stf": "STF DDS texture (*.dds)",
        }[fmt]
        default = os.path.join(
            os.path.dirname(paths["color"]), base + "_recoloured" + ext
        )
        out_path, _ = QFileDialog.getSaveFileName(
            self, f"Export {self.key} texture", default, filt
        )
        if not out_path:
            return None
        out_base = out_path
        for e in (".png", ".dds"):
            if out_base.lower().endswith(e):
                out_base = out_base[:-4]
        try:
            written = save_texture(Image.fromarray(img8, "RGB"), out_base, fmt)
            QMessageBox.information(
                self, "Exported", f"Saved recoloured texture:\n{written}"
            )
            return written
        except NotImplementedError as e:
            QMessageBox.information(self, "Export as Texture", str(e))
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
        return None


class SlotRow(QWidget):
    """One expected-asset row: status mark, label, Select button, resolved path, in-game hint."""

    def __init__(self, slot, label, hint, tier, on_pick, game=None):
        super().__init__()
        self.slot, self.hint, self.tier = slot, hint, tier
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 4, 0, 6)
        v.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.mark = QLabel("\u2013")
        self.mark.setFixedWidth(16)
        name = QLabel(label + ("" if tier == "required" else f"  ({tier})"))
        top.addWidget(self.mark)
        top.addWidget(name, 1)
        v.addLayout(top)
        # path + Select button on one row: the path takes the available width, the button sits at
        # the far right. This row spans the section's content width and the category body uses equal
        # left/right margins, so the button's gap to the section edge matches the path's on the left.
        self.pick = QPushButton("Select file\u2026")
        self.pick.setFixedHeight(26)
        self.pick.clicked.connect(lambda: on_pick(self.slot))
        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.setSpacing(12)  # padding between the path and the button
        self.path_lbl = QLabel()
        self.path_lbl.setObjectName("legend")
        self.path_lbl.setWordWrap(True)
        self.path_lbl.setMinimumHeight(26)  # same height as the Select button beside it
        self.path_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.path_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        path_row.addWidget(self.path_lbl, 1)
        path_row.addWidget(self.pick, 0, Qt.AlignmentFlag.AlignVCenter)
        v.addLayout(path_row)
        # in-game / expected path hint (below the path). `game` overrides the Banshee lookup.
        self.game_lbl = QLabel()
        self.game_lbl.setObjectName("gamepath")
        self.game_lbl.setWordWrap(True)
        self.game_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        v.addWidget(self.game_lbl)
        self.set_game(game if game is not None else assets.GAME_PATH.get(slot))

    def set_game(self, game):
        """Set/refresh the expected in-game path line (used when the gender changes)."""
        if game:
            self.game_lbl.setText("in game:  \u2026/" + game)
            self.game_lbl.show()
        else:
            self.game_lbl.clear()
            self.game_lbl.hide()

    def update_state(self, path, override=False):
        if path and os.path.isfile(path):
            self.mark.setText("\u2713")
            # tick is green when the asset resolves; a manual pick (override) stays amber
            self.mark.setStyleSheet(
                "color:%s" % ("#F0B429" if override else "#22C55E")
            )
            col = (
                "#F0B429" if override else "#FFFFFF"
            )  # path text: amber on manual pick, else white
            self.path_lbl.setText(path + ("   (manual pick)" if override else ""))
            self.path_lbl.setStyleSheet("color:%s" % col)
        elif path:  # set but missing on disk
            self.mark.setText("\u2717")
            self.mark.setStyleSheet("color:#F87171")
            self.path_lbl.setText(path + "   (file not found)")
            self.path_lbl.setStyleSheet("color:#F87171")
        else:
            self.mark.setText("\u2717" if self.tier == "required" else "\u2013")
            self.mark.setStyleSheet(
                "color:#F87171" if self.tier == "required" else "color:#8A93A3"
            )
            self.path_lbl.setText("not selected")
            self.path_lbl.setStyleSheet("color:#5C6675")


class CollapsibleSection(QWidget):
    """A titled section whose body can be collapsed/expanded by clicking its header."""

    def __init__(self, title, body, expanded=False, parent=None):
        super().__init__(parent)
        self._title = title
        self.body = body
        self.toggle = QPushButton()
        self.toggle.setObjectName(
            "sectiontoggle"
        )  # lets the runtime ExtraBold rule reach it
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.clicked.connect(self._sync)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.toggle)
        lay.addWidget(self.body)
        self._sync()

    def restyle_arrow(self):
        """Re-paint the expand/collapse arrow in the current accent. The arrow is a QPainter-drawn
        icon (not QSS), so it doesn't follow a stylesheet swap; the theme re-apply calls this so the
        arrows track accent changes and snap back when 'Reset colours' is pressed."""
        self.toggle.setIcon(_section_arrow_icon(self.toggle.isChecked()))

    def _sync(self):
        on = self.toggle.isChecked()
        self.body.setVisible(on)
        self.toggle.setText(self._title)
        self.toggle.setIcon(_section_arrow_icon(on))


def _section_arrow_icon(down):
    """A small triangle (down when open, right when closed) drawn as a fixed-size icon, so the
    collapsible-section dropdowns sit horizontally centred on the same line as their arrow rather
    than the text glyph dropping below the title's baseline."""
    px = QPixmap(16, 16)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(theme.accent_active()))
    if down:
        tri = QPolygonF([QPointF(4, 5.5), QPointF(12, 5.5), QPointF(8, 11)])
    else:
        tri = QPolygonF([QPointF(5.5, 4), QPointF(5.5, 12), QPointF(11, 8)])
    p.drawPolygon(tri)
    p.end()
    return QIcon(px)


class _AssetFolderPanel(QWidget):
    """Shared base for the asset panels (Ikran / Na'vi / Gear Camo). They all open with the same
    'Asset Folder' picker row - a read-only path field plus 'Set folder' and 'Open folder' buttons.
    A subclass sets ``self.folder`` from its own config first, defines ``_set_export_folder`` (all
    three route through the shared ``choose_export_folder``), then calls ``_add_asset_folder_row``
    with the layout to add the row to and the Set-folder button's tooltip."""

    def _add_asset_folder_row(self, layout, set_tooltip):
        layout.addWidget(_navi_subtitle("Asset Folder"))
        efr = QHBoxLayout()
        efr.setSpacing(6)
        self.export_edit = QLineEdit(self.folder)
        self.export_edit.setReadOnly(True)
        self.export_edit.setPlaceholderText("(no asset folder set)")
        self.export_edit.setToolTip(self.folder or "No asset folder set")
        set_btn = QPushButton("Set folder")
        set_btn.setObjectName("action")
        set_btn.setToolTip(set_tooltip)
        set_btn.clicked.connect(self._set_export_folder)
        ef_open = QPushButton("Open folder")
        ef_open.setToolTip("Open the asset folder in your file manager.")
        ef_open.clicked.connect(lambda: _open_in_file_manager(self.folder))
        efr.addWidget(self.export_edit, 1)
        efr.addWidget(set_btn)
        efr.addWidget(ef_open)
        layout.addLayout(efr)
        return efr


class AssetsPanel(_AssetFolderPanel):
    """Inline Banshee asset manager for embedding in Settings.
    Paths are referenced, never copied; every change is written to the config immediately
    and reported via the on_changed callback."""

    def __init__(self, on_changed=None, on_reset=None, parent=None):
        super().__init__(parent)
        self.on_changed = on_changed
        self.on_reset = on_reset
        self.cfg = assets.load_config(mutable=True)
        self.paths = dict(self.cfg.get("paths", {}))
        self.models = list(self.cfg.get("models", []))
        self.manual = set(
            self.cfg.get("manual_slots", [])
        )  # slots picked individually, not scanned
        self.folder = self.cfg.get("export_folder", "") or ""
        self.setAcceptDrops(True)
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 6, 2, 2)
        v.setSpacing(8)

        # ---- asset folder picker ----
        self._add_asset_folder_row(
            v,
            "Choose your extracted mod / asset folder. The assets are synced from it automatically.",
        )

        # ---- resync + manage row ----
        fr = QHBoxLayout()
        resync_btn = QPushButton("Resync Assets")
        resync_btn.setObjectName("action")
        resync_btn.setToolTip(
            "Re-scan the asset folder above and refresh every auto-detected banshee asset "
            "(prefers .dds, falls back to .png)."
        )
        resync_btn.clicked.connect(self._resync)
        fr.addWidget(resync_btn)
        clear_btn = QPushButton("Clear manual picks")
        clear_btn.setStyleSheet(_warning_css())
        clear_btn.setToolTip(
            "Drop the files you picked individually; folder-detected ones stay. "
            "Your files are not deleted."
        )
        clear_btn.clicked.connect(self._clear_picks)
        fr.addWidget(clear_btn)
        if self.on_reset:
            reset_btn = QPushButton("Reset Ikran assets\u2026")
            reset_btn.setToolTip(
                "Forget every remembered Ikran asset path. Your files are not deleted."
            )
            reset_btn.setStyleSheet(_danger_css())
            reset_btn.setFixedHeight(clear_btn.sizeHint().height())
            reset_btn.clicked.connect(self.on_reset)
            fr.addWidget(reset_btn)
        fr.addStretch(1)
        v.addLayout(fr)

        # ---- description (below the buttons) ----
        intro = QLabel(
            "Point this at your own extracted Avatar: Frontiers of Pandora assets - nothing is "
            "copied, the tool just remembers where the files are. Set an asset folder to "
            "auto-detect every file, or set a slot's file individually. You can also drag files "
            "or folders here."
        )
        intro.setObjectName("legend")
        intro.setWordWrap(True)
        v.addWidget(intro)

        self.rows = {}
        # group the slots into collapsible categories (Head / Body / Shared); the model lives in Body
        cats = [("Head", []), ("Body", []), ("Shared / Detail", [])]
        cat_map = dict(cats)
        for slot, label, hint, tier in assets.SLOTS:
            row = SlotRow(slot, label, hint, tier, self._pick_one)
            self.rows[slot] = row
            if slot.startswith("head"):
                cat_map["Head"].append(row)
            elif slot == "model" or slot.startswith("body"):
                cat_map["Body"].append(row)
            else:
                cat_map["Shared / Detail"].append(row)
        self.sections = []
        for title, rows in cats:
            if not rows:
                continue
            body = QWidget()
            bl = QVBoxLayout(body)
            bl.setContentsMargins(10, 0, 10, 4)
            bl.setSpacing(2)
            for row in rows:
                bl.addWidget(row)
            sec = CollapsibleSection(title, body, expanded=False)
            self.sections.append(sec)
            v.addWidget(sec)
        self._refresh()

    def _commit(self):
        def _upd(cfg):
            cfg["paths"] = {k: val for k, val in self.paths.items() if val}
            cfg["models"] = self.models
            cfg["manual_slots"] = sorted(s for s in self.manual if self.paths.get(s))
            cfg["export_folder"] = self.folder

        self.cfg = assets.update_config(
            _upd
        )  # fresh load - never clobbers the Na'vi section
        self._refresh()
        if self.on_changed:
            self.on_changed()

    def _refresh(self):
        for slot, row in self.rows.items():
            p = self.paths.get(slot)
            # only individually-picked slots are "manual"; folder-scanned ones are auto-detected
            row.update_state(p, override=(bool(p) and slot in self.manual))

    def reload(self):
        """Re-read the config from disk (e.g. after an external reset)."""
        self.cfg = assets.load_config(mutable=True)
        self.paths = dict(self.cfg.get("paths", {}))
        self.models = list(self.cfg.get("models", []))
        self.manual = set(self.cfg.get("manual_slots", []))
        self.folder = self.cfg.get("export_folder", "") or ""
        if hasattr(self, "export_edit"):
            self.export_edit.setText(self.folder)
            self.export_edit.setToolTip(self.folder or "No asset folder set")
        self._refresh()

    def _clear_picks(self):
        """Drop only the slots you picked individually; folder-detected paths stay put.
        (The red 'Reset Ikran assets' button is the full wipe.)"""
        if not self.manual:
            return
        for slot in list(self.manual):
            p = self.paths.pop(slot, None)
            if slot == "model" and p:
                self.models = [m for m in self.models if m != p]
        self.manual = set()
        self._commit()

    def _pick_one(self, slot):
        start = ""
        if assets.get_setting("smart_dialog_start", True):
            start = assets.dialog_start_for(self.paths.get(slot), "")
            if not start and slot != "model":
                # a texture slot with no folder yet: open where the chosen model lives
                start = assets.dialog_start_for(self.paths.get("model"), "")
        if not start:
            start = self.folder if (self.folder and os.path.isdir(self.folder)) else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select " + assets.SLOT_HINT.get(slot, "file"),
            start,
            assets.slot_filter(slot),
        )
        if path:
            self.paths[slot] = path
            self.manual.add(slot)  # an explicit Select file -> manual pick
            if slot == "model" and path not in self.models:
                self.models.append(path)
            self._commit()

    def _set_export_folder(self):
        choose_export_folder(self, "Choose your extracted mod / asset folder")

    def _resync(self):
        res = scan_export_folder(self, assets.scan_folder, "Ikran assets")
        if res is None:
            return
        slots, models = res
        for slot, p in slots.items():
            self.paths[slot] = p
            self.manual.discard(slot)  # auto-detected, not a manual pick
        if models:
            self.models = models
        if not slots:
            QMessageBox.information(
                self,
                "Nothing found",
                "No recognised banshee assets were found in that folder.",
            )
        self._commit()

    def _drop_paths(self, paths):
        changed = False
        for p in paths:
            if os.path.isdir(p):
                slots, models = assets.scan_folder(p)
                for slot, sp in slots.items():
                    self.paths[slot] = sp
                    self.manual.discard(slot)  # dropped folder = auto-detect
                if models:
                    self.models = models
                changed = changed or bool(slots)
            elif os.path.isfile(p):
                slot = assets.classify(os.path.basename(p))
                if slot:
                    self.paths[slot] = p
                    self.manual.add(slot)  # dropped single file = manual pick
                    if slot == "model" and p not in self.models:
                        self.models.append(p)
                    changed = True
        if changed:
            self._commit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self._drop_paths(paths)


class NaviAssetsPanel(_AssetFolderPanel):
    """Inline Na'vi asset manager, mirroring AssetsPanel. Per-gender mesh slots (a part is
    satisfied if either the male OR female mesh is present); textures resolve for the active
    gender. An asset folder bulk auto-detects everything; a per-slot Select file overrides one
    slot. Paths are referenced, never copied; every change writes the config + fires on_changed."""

    def __init__(self, get_gender, on_changed=None, on_reset=None, parent=None):
        super().__init__(parent)
        self.get_gender = get_gender
        self.on_changed = on_changed
        self.on_reset = on_reset
        self.cfg = assets.load_config(mutable=True)
        navi = self.cfg.get("navi", {})
        self.folder = navi.get("folder", "")
        self.overrides = dict(navi.get("paths", {}))
        self.cache = navi.get("cache", {}) or {}
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 6, 2, 2)
        v.setSpacing(8)

        # ---- asset folder picker ----
        self._add_asset_folder_row(
            v,
            "Choose your extracted Na'vi asset folder. The assets are synced from it automatically.",
        )

        # ---- resync + manage row ----
        fr = QHBoxLayout()
        resync_btn = QPushButton("Resync Assets")
        resync_btn.setObjectName("action")
        resync_btn.setToolTip(
            "Re-scan the asset folder above and re-cache every Na'vi mesh + texture (prefers .dds)."
        )
        resync_btn.clicked.connect(self._resync)
        fr.addWidget(resync_btn)
        clear_btn = QPushButton("Clear manual picks")
        clear_btn.setStyleSheet(_warning_css())
        clear_btn.setToolTip(
            "Drop per-slot file overrides and fall back to folder auto-detect."
        )
        clear_btn.clicked.connect(self._clear_overrides)
        fr.addWidget(clear_btn)
        if self.on_reset:
            navi_reset_btn = QPushButton("Reset Na'vi assets\u2026")
            navi_reset_btn.setToolTip(
                "Forget the Na'vi asset folder and every per-slot file override. Your files "
                "are not deleted."
            )
            navi_reset_btn.setStyleSheet(_danger_css())
            navi_reset_btn.setFixedHeight(clear_btn.sizeHint().height())
            navi_reset_btn.clicked.connect(self.on_reset)
            fr.addWidget(navi_reset_btn)
        fr.addStretch(1)
        v.addLayout(fr)

        # ---- description (below the buttons) ----
        intro = QLabel(
            "Point this at your own extracted Avatar: Frontiers of Pandora assets - nothing is "
            "copied, the tool just remembers where the files are. Set an asset folder to "
            "auto-detect every mesh and texture, or set a slot's file individually. Both body "
            "types are listed; a part is satisfied if either the male or female mesh is present. "
            "Colour presets are chosen in the Na'vi tab, not here."
        )
        intro.setObjectName("legend")
        intro.setWordWrap(True)
        v.addWidget(intro)
        self.rows = {}
            # group the slots into collapsible sections by part instead of one long mixed list. A few
            # slots resolve under one bucket but read better under another: the hair cap is a 'head'
            # texture (textures["head"]["haircap"]) and nav_head_cap / Hair Decor (_d) resolve under
            # 'head'/'accessory', but all sit at the bottom of the Hair section in the UI.
        UI_SECTION = {"nav_head_cap": "hair", "nav_hair_acc": "hair"}
        by_bucket = {}
        for slot, label, tier, _kind, a, _b, _g in assets.NAVI_SLOTS:
            if slot in assets.NAVI_EDIT_ONLY_SLOTS:
                continue  # opt-in extras: pickable only in Edit Na'vi, not listed here
            row = SlotRow(
                slot,
                label,
                "",
                tier,
                self._pick_one,
                game=assets.navi_game_path(slot, self.get_gender()),
            )
            self.rows[slot] = row
            by_bucket.setdefault(UI_SECTION.get(slot, a), []).append(row)
        self.sections = []
        section_order = [
            ("head", "Head"),
            ("body", "Body"),
            ("eye", "Eyes"),
            ("hair", "Hair"),
            ("kuru", "Kuru"),
        ]
        seen_buckets = set()
        for bucket, title in section_order:
            rows = by_bucket.get(bucket)
            if not rows:
                continue
            seen_buckets.add(bucket)
            body = QWidget()
            bl = QVBoxLayout(body)
            bl.setContentsMargins(10, 0, 10, 4)
            bl.setSpacing(2)
            for row in rows:
                bl.addWidget(row)
            sec = CollapsibleSection(title, body, expanded=False)
            self.sections.append(sec)
            v.addWidget(sec)
        # any future bucket not in the explicit order still gets its own section
        for bucket, rows in by_bucket.items():
            if bucket in seen_buckets:
                continue
            body = QWidget()
            bl = QVBoxLayout(body)
            bl.setContentsMargins(10, 0, 10, 4)
            bl.setSpacing(2)
            for row in rows:
                bl.addWidget(row)
            sec = CollapsibleSection(bucket.capitalize(), body, expanded=False)
            self.sections.append(sec)
            v.addWidget(sec)
        v.addStretch(1)
        self._refresh()

    def _commit(self):
        def _upd(cfg):
            navi = cfg.setdefault("navi", {})
            navi["folder"] = self.folder
            navi["paths"] = {k: val for k, val in self.overrides.items() if val}
            navi["cache"] = self.cache

        self.cfg = assets.update_config(
            _upd
        )  # fresh load - never clobbers the banshee section
        self._refresh()
        if self.on_changed:
            self.on_changed()

    def _refresh(self):
        g = self.get_gender()
        sp = assets.navi_resolve(
            self.cache, self.overrides, g
        )  # cache lookup, no disk walk
        cache_slots = (self.cache or {}).get("slots", {}) or {}
        for slot, row in self.rows.items():
            row.set_game(assets.navi_game_path(slot, g))
            ov = self.overrides.get(slot)
            # only a genuine deviation is a "manual pick" - an override that matches what the folder
            # scan already provides (e.g. a hair cap auto-detected from the same export) is not.
            manual = bool(ov) and ov != cache_slots.get(slot)
            row.update_state(sp.get(slot), override=manual)
        if hasattr(self, "export_edit"):
            self.export_edit.setText(self.folder or "")
            self.export_edit.setToolTip(self.folder or "No asset folder set")

    def current_path(self, slot):
        """The path currently resolved for `slot` (override first, then cache); None if unset."""
        sp = assets.navi_resolve(self.cache, self.overrides, self.get_gender())
        return sp.get(slot)

    def export_overrides(self):
        """The explicit per-slot path overrides (and asset folder) for a Save/Load preset."""
        return {
            "folder": self.folder or "",
            "paths": {k: v for k, v in self.overrides.items() if v},
        }

    def apply_overrides(self, d, missing):
        """Restore saved per-slot path overrides. Any override file that no longer exists is flagged
        and dropped, so that slot falls back to the scanned default in its place. Overrides for slots
        that no longer exist at all (renamed/removed) are purged so they don't linger in the config."""
        d = d or {}
        valid = {row[0] for row in assets.NAVI_SLOTS}  # current slot names; drop anything else
        kept = {}
        for slot, p in (d.get("paths") or {}).items():
            if slot not in valid:
                continue  # stale slot from an older preset -> purge
            if p and os.path.isfile(p):
                kept[slot] = p
            elif p:
                missing.append(p)
        self.overrides = kept
        folder = (d.get("folder") or "").strip()
        if folder and os.path.isdir(folder):
            self.folder = folder
        self._commit()

    def reload(self):
        """Re-read the config from disk (e.g. after an external reset) and refresh the rows."""
        self.cfg = assets.load_config(mutable=True)
        navi = self.cfg.get("navi", {})
        self.folder = navi.get("folder", "")
        self.overrides = dict(navi.get("paths", {}))
        self.cache = navi.get("cache", {}) or {}
        self._refresh()

    def set_override(self, slot, path, persist=True):
        """Set a per-slot override. persist=True writes it to config (the default - used by the
        Settings asset panel). persist=False applies it for THIS SESSION only (no config write);
        the live viewer still picks it up because mesh resolution reads these in-memory overrides,
        and it's forgotten next launch when overrides reload from config."""
        self.overrides[slot] = path
        if persist:
            self._commit()  # writes config, refreshes rows, fires on_changed
        else:
            self._refresh()
            if self.on_changed:
                self.on_changed()  # reload the viewer from the in-memory override

    def _pick_one(self, slot, persist=True):
        start = self._dialog_start_for_slot(slot)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select " + slot.replace("nav_", "").replace("_", " "),
            start,
            assets.navi_slot_filter(slot),
        )
        if path:
            self.set_override(slot, path, persist=persist)

    # Texture buckets whose mesh lives under a different part (the accessory atlas sits beside the
    # hair mesh, so its picker should open the hair folder).
    _TEX_MESH_PART = {"accessory": "hair"}

    def _dialog_start_for_slot(self, slot):
        """Where this slot's file dialog opens, inferring a folder from already-set related files:
        1. the slot's OWN current file (its folder);
        2. for a model (.mmb) slot, the OTHER model in its hair/kuru pair - so picking Kuru opens
           where Hair was set, and vice-versa;
        3. for a texture slot, its part's mesh folder;
        then the asset folder, else empty. With the setting off, always open at the asset folder."""
        asset_dir = self.folder if (self.folder and os.path.isdir(self.folder)) else ""
        if not assets.get_setting("smart_dialog_start", True):
            return asset_dir

        def folder_of(s):
            try:
                p = self.current_path(s)
            except Exception:
                p = None
            return assets.dialog_start_for(p, "")

        # 1. the slot's own file
        start = folder_of(slot)
        if start:
            return start
        meta = {s[0]: s for s in assets.NAVI_SLOTS}
        me = meta.get(slot)
        if me:
            kind, a, gender = me[3], me[4], me[6]
            if kind == "mesh":
                # 2. the OTHER model in the hair<->kuru pair
                other = {"hair": "kuru", "kuru": "hair"}.get(a)
                if other:
                    for s in assets.NAVI_SLOTS:
                        if s[3] == "mesh" and s[4] == other:
                            start = folder_of(s[0])
                            if start:
                                return start
            elif kind == "tex":
                # 3. this texture's part mesh folder
                mesh_part = self._TEX_MESH_PART.get(a, a)
                for s in assets.NAVI_SLOTS:
                    if s[3] == "mesh" and s[4] == mesh_part and s[6] in (gender, None, ""):
                        start = folder_of(s[0])
                        if start:
                            return start
        # 4. fall back to the asset folder
        return asset_dir

    def _set_export_folder(self):
        choose_export_folder(self, "Choose your extracted Na'vi asset folder")

    def _resync(self):
        res = scan_export_folder(self, assets.scan_navi_folder, "Na'vi assets")
        if res is None:
            return
        self.cache = res
        if not res.get("slots"):
            QMessageBox.information(
                self,
                "Nothing found",
                "No recognised Na'vi assets were found in that folder.",
            )
        self._commit()

    def _clear_overrides(self):
        if not self.overrides:
            return
        self.overrides = {}
        self._commit()


class _NaviSwatchRow(QWidget):
    """A labelled colour swatch + hex field. The DEFAULT colour is active (swatch shows it, the
    preview/export use it) while the hex box is left BLANK - the default hex shows only as greyed
    placeholder. Typing a valid 6-digit hex (or picking one) overrides the default; clearing the
    box reverts to it. So 'no text in the box' == 'use the default'."""

    def __init__(self, label, on_change, default_hex="808080"):
        super().__init__()
        self.on_change = on_change
        self._default_hex = default_hex.strip().lstrip("#").upper()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(8)
        self.swatch = QPushButton()
        self.swatch.setObjectName("swatch")
        self.swatch.setFixedSize(30, 22)
        self.swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.swatch.setStyleSheet(_swatch_css(self._default_hex))
        self.swatch.clicked.connect(self._pick)
        self.name = QLabel(label)
        self.name.setFixedWidth(96)
        self.field = QLineEdit()
        self.field.setMaxLength(6)
        self.field.setFixedWidth(70)
        self.field.setPlaceholderText(self._default_hex)
        _dim_placeholder(self.field)
        self.field.editingFinished.connect(self._typed)
        self.field.textChanged.connect(self._on_text)
        for w in (self.swatch, self.name, self.field):
            lay.addWidget(w)
        lay.addStretch(1)

    def effective(self):
        """The active colour: the typed override if the box holds a valid 6-digit hex, else the
        default (which is also what a blank box means)."""
        t = self.field.text().strip().lstrip("#").upper()
        if len(t) == 6:
            try:
                int(t, 16)
                return t
            except ValueError:
                pass
        return self._default_hex

    def hex(self):
        return self.effective()

    def is_valid(self):
        # a BLANK box is valid (it means 'use the default'); otherwise require a 6-digit hex
        t = self.field.text().strip().lstrip("#")
        if t == "":
            return True
        if len(t) != 6:
            return False
        try:
            int(t, 16)
            return True
        except ValueError:
            return False

    def _on_text(self, *_):
        self.swatch.setStyleSheet(_swatch_css(self.effective()))
        self._mark_validity()

    def _mark_validity(self, *_):
        self.field.setStyleSheet(
            "" if self.is_valid() else "background:#2A1518; color:#F87171;"
        )

    def is_neutral(self):
        """True when the box is blank (no override) - i.e. the swatch is at its default colour, so
        there is nothing changed to export. Reads the field directly so the export gate updates as
        soon as a digit changes."""
        return self.field.text().strip() == ""

    def set_hex(self, rrggbb, notify=False):
        """Populate the box with an explicit override (used by Load / the colour picker)."""
        rrggbb = rrggbb.strip().lstrip("#").upper()
        if len(rrggbb) != 6:
            return
        try:
            int(rrggbb, 16)
        except ValueError:
            return
        self.field.blockSignals(True)
        self.field.setText(rrggbb)
        self.field.blockSignals(False)
        self.swatch.setStyleSheet(_swatch_css(rrggbb))
        self._mark_validity()
        if notify:
            self.on_change()

    def set_default(self, rrggbb):
        """Set the default colour applied when the box is blank (leaves any typed override alone)."""
        self._default_hex = (rrggbb or "808080").strip().lstrip("#").upper()
        self.field.setPlaceholderText(self._default_hex)
        self.swatch.setStyleSheet(_swatch_css(self.effective()))

    def _reset(self):
        self.field.blockSignals(True)
        self.field.clear()
        self.field.blockSignals(False)
        self.swatch.setStyleSheet(_swatch_css(self._default_hex))
        self._mark_validity()
        self.on_change()

    def _typed(self):
        t = self.field.text().strip().lstrip("#").upper()
        if t == "":  # cleared -> revert to default
            self.swatch.setStyleSheet(_swatch_css(self._default_hex))
            self._mark_validity()
            self.on_change()
            return
        if len(t) == 6:
            try:
                int(t, 16)
                self.field.blockSignals(True)
                self.field.setText(t)
                self.field.blockSignals(False)
                self.swatch.setStyleSheet(_swatch_css(t))
                self._mark_validity()
                self.on_change()
            except ValueError:
                pass

    def _pick(self):
        c = QColorDialog.getColor(QColor("#" + self.effective()), self, "Pick colour")
        if c.isValid():
            self.set_hex(c.name()[1:], notify=True)


def _navi_subtitle(text):
    """A sub-section heading styled like the Banshee tab's section titles (Textures/Colours/...)."""
    lbl = QLabel(text)
    lbl.setObjectName("sectiontitle")
    return lbl


def _hrule(layout, gap=8):
    """Insert a faint 1px horizontal rule (the established #hdivider slate) into a vertical layout,
    centred in a gap of `gap` px either side, to separate subsections. Matches the divider already
    used elsewhere in the app so the panels read as one consistent style."""
    layout.addSpacing(gap)
    line = QFrame()
    line.setObjectName("hdivider")
    line.setFixedHeight(1)
    layout.addWidget(line)
    layout.addSpacing(gap)


class _NaviAssetRow(QWidget):
    """File picker for one Na'vi asset slot, styled like the Banshee tab's texture rows: a title
    button (opens the file dialog) on the left and a path field on the right (type a path + Enter to
    set it). Picking/typing delegates to the shared override logic, so a change re-binds the slot
    and reloads the viewer."""

    def __init__(self, label, slot, on_pick, get_path, on_set_path=None):
        super().__init__()
        self.slot = slot
        self._on_pick = on_pick
        self._get_path = get_path
        self._on_set_path = on_set_path
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(6)
        btn = QPushButton(label)
        btn.setFixedWidth(
            99
        )  # Banshee-button size + 15% (uniform across the Na'vi tab)
        btn.setToolTip(
            "Choose the %s file (overrides the auto-detected one)" % label.lower()
        )
        btn.clicked.connect(lambda: self._on_pick(self.slot) if self._on_pick else None)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("(default)")
        if on_set_path is not None:
            self.edit.setToolTip(
                "Path to this file - type a path and press Enter to set it"
            )
            self.edit.returnPressed.connect(self._typed)
        else:
            self.edit.setReadOnly(True)
        lay.addWidget(btn)
        lay.addWidget(self.edit, 1)
        self.refresh()

    def _typed(self):
        t = self.edit.text().strip()
        if t and self._on_set_path is not None:
            self._on_set_path(self.slot, t)

    def refresh(self):
        p = None
        if self._get_path is not None:
            try:
                p = self._get_path(self.slot)
            except Exception:  # never let a refresh abort Qt
                p = None
        self.edit.setText(p or "")
        self.edit.setToolTip(
            p or "Path to this file - type a path and press Enter to set it"
        )


class _NaviSection(QGroupBox):
    """One customization category (skin/eye/hair/warpaint): load a single .blueitemtype
    (Load button + path field, exactly like the Banshee pattern panel - no bulk-scanned
    dropdown to scroll through), edit the swatches, then export the colours back into a
    file (overwrite the loaded file, or Save As)."""

    # Warpaint preview-texture slots: Head + four Body layers.
    TEX_SLOTS = [
        ("head", "Head (_m)"),
        ("body1", "Body 1 (_m)"),
        ("body2", "Body 2 (_m)"),
        ("body3", "Body 3 (_m)"),
        ("body4", "Body 4 (_m)"),
    ]

    def __init__(
        self,
        title,
        role_labels,
        on_change,
        with_bio=False,
        asset_groups=None,
        on_pick_asset=None,
        get_asset_path=None,
        on_set_asset_path=None,
        key=None,
        on_export_texture=None,
        preset_example=None,
        defaults=None,
        note=None,
        role_notes=None,
        strength_spec=None,
        texture_picker=False,
        color_index=False,
        roughness=False,
    ):
        super().__init__(title)
        self._on_change = on_change
        self.with_bio = with_bio
        self._busy = False
        self._title = title
        self._key = key or title.lower()
        self._on_export_texture = on_export_texture
        self._load_handler = None  # set by NaviControls -> MainWindow file loader
        self.path = None  # loaded .blueitemtype path (None until Load succeeds)
        self._color_indices = []  # the file's own myColorN indices, positional with self.rows
        self.on_validity_change = None  # set by NaviControls (drives Export All)

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)

        # ---- colour preset loader (.blueitemtype): a titled Load row ----
        bar = QHBoxLayout()
        bar.setSpacing(6)
        load = QPushButton("Load")
        load.setFixedWidth(99)  # match the Na'vi asset-row buttons (15% wider)
        load.setToolTip(
            (
                "Load a single %s colour item (.blueitemtype) from your AFoP export and apply its "
                "colours.\nFiles for this section are usually named like:\ne.g. %s"
                % (title.lower(), preset_example)
            )
            if preset_example
            else (
                "Load a single %s colour item (.blueitemtype) from your AFoP export and apply it"
                % title.lower()
            )
        )
        load.clicked.connect(self._on_load_clicked)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(".blueitemtype path")
        self.path_edit.setToolTip(
            "Path to this section's loaded colour item - type a path and press Enter to load"
        )
        self.path_edit.returnPressed.connect(self._load_entered)
        bar.addWidget(load)
        bar.addWidget(self.path_edit, 1)
        v.addLayout(bar)

        # ---- files (meshes / textures), grouped under sub-titles, ABOVE the colours ----
        self.hide_toggle = None
        if texture_picker:
            self.hide_toggle = QCheckBox("Hide warpaint")
            self.hide_toggle.setToolTip(
                "Hide the warpaint in the preview without clearing your colours or texture."
            )
            self.hide_toggle.toggled.connect(lambda *_: self._row_changed())
            v.addWidget(self.hide_toggle)

        self.asset_rows = []
        for sub_title, slots in asset_groups or []:
            _hrule(v, 6)
            v.addWidget(_navi_subtitle(sub_title))
            for lbl, slot in slots:
                ar = _NaviAssetRow(
                    lbl, slot, on_pick_asset, get_asset_path, on_set_asset_path
                )
                self.asset_rows.append(ar)
                v.addWidget(ar)

        # ---- warpaint texture pickers: self-contained, NOT part of the asset config (there is no
        # default warpaint). Head + four Body layers, each an optional paint _m map. ----
        self._tex_paths = {}  # slot -> path
        self.tex_slot_edits = {}
        if texture_picker:
            _hrule(v, 6)
            v.addWidget(_navi_subtitle("Textures"))
            _ttip = {
                "head": "Head paint (_m) - painted onto the face/head.",
                "body1": "Body paint (_m) - the main pattern across the third-person body.",
                "body2": "A second body paint (_m), layered over Body 1.",
                "body3": "A third body paint (_m), layered over Body 2.",
                "body4": "A fourth body paint (_m), layered over Body 3.",
            }
            for slot, label in self.TEX_SLOTS:
                trow = QHBoxLayout()
                trow.setSpacing(6)
                btn = QPushButton(label)
                btn.setFixedWidth(99)
                btn.setToolTip(_ttip.get(slot, ""))
                btn.clicked.connect(lambda _=False, s=slot: self._pick_texture(s))
                edit = QLineEdit()
                edit.setPlaceholderText("(optional)")
                edit.setToolTip(_ttip.get(slot, ""))
                edit.editingFinished.connect(lambda s=slot: self._texture_entered(s))
                self.tex_slot_edits[slot] = edit
                trow.addWidget(btn)
                trow.addWidget(edit, 1)
                v.addLayout(trow)
            tnote = QLabel(
                "All optional paint _m maps. Body 1-4 layer over each other. "
                "Leave any you don't need empty."
            )
            tnote.setObjectName("subtitle")
            tnote.setWordWrap(True)
            tnote.setContentsMargins(2, 3, 0, 0)
            v.addWidget(tnote)

        # ---- colours ----
        _hrule(v, 6)
        v.addWidget(_navi_subtitle("Colours"))
        self.rows = []
        role_notes = role_notes or {}
        for i, lbl in enumerate(role_labels):
            dh = defaults[i] if (defaults and i < len(defaults)) else "808080"
            r = _NaviSwatchRow(lbl, self._row_changed, default_hex=dh)
            self.rows.append(r)
            v.addWidget(r)
            if lbl in role_notes:
                rn = QLabel(role_notes[lbl])
                rn.setObjectName("subtitle")
                rn.setWordWrap(True)
                rn.setContentsMargins(2, 0, 0, 4)
                v.addWidget(rn)

        # ---- optional strength slider (e.g. hair-cap blend) ----
        self.strength_slider = None
        self._strength_key = None
        if strength_spec:
            slbl, skey, sdefault = strength_spec
            self._strength_key = skey
            srow = QHBoxLayout()
            srow.setSpacing(8)
            cap = QLabel(slbl)
            cap.setObjectName("subtitle")
            self.strength_slider = QSlider(Qt.Orientation.Horizontal)
            self.strength_slider.setRange(0, 100)
            self.strength_slider.setValue(int(round(sdefault * 100)))
            self.strength_slider.setToolTip(
                "How strongly the hair-cap colour tints the scalp patch (0 = off, 1 = full)."
            )
            self.strength_slider.valueChanged.connect(lambda *_: self._row_changed())
            self._strength_val = QLabel("%.2f" % sdefault)
            self._strength_val.setObjectName("subtitle")
            self._strength_val.setFixedWidth(34)
            self.strength_slider.valueChanged.connect(
                lambda v_: self._strength_val.setText("%.2f" % (v_ / 100.0))
            )
            srow.addWidget(cap)
            srow.addWidget(self.strength_slider, 1)
            srow.addWidget(self._strength_val)
            v.addLayout(srow)

        # ---- warpaint authoring extras: optional myMainColorIndex + per-colour roughness.
        # Both default OFF; a divider separates Colours / Main colour index / Roughness. ----
        self.mci_check = self.mci_spin = self.mci_label = None
        self.rough_check = None
        self.rough_spins = []
        self.rough_labels = []
        if color_index:
            _hrule(v, 6)
            self.mci_check = QCheckBox("Set main colour index")
            self.mci_check.setToolTip(
                "Write myMainColorIndex - which colour the menu treats as this paint's primary "
                "swatch. It doesn't change the 3D look. Off = leave it out of the file."
            )
            v.addWidget(self.mci_check)
            mrow = QHBoxLayout()
            mrow.setSpacing(8)
            mlab = QLabel("Main colour")
            mlab.setFixedWidth(96)  # match the colour-picker label (same font/width)
            mlab.setFixedHeight(28)
            self.mci_label = mlab
            self.mci_spin = QSpinBox()
            self.mci_spin.setRange(1, 4)
            self.mci_spin.setFixedHeight(28)
            self.mci_spin.setMinimumWidth(90)
            self.mci_spin.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.mci_spin.valueChanged.connect(lambda *_: self._row_changed())
            self.mci_check.toggled.connect(self._set_mci_active)
            self.mci_check.toggled.connect(lambda *_: self._row_changed())
            self._set_mci_active(
                self.mci_check.isChecked()
            )  # start greyed + disabled (default off)
            mrow.addWidget(mlab)
            mrow.addStretch(1)
            mrow.addWidget(self.mci_spin)
            v.addLayout(mrow)
        if roughness:
            _hrule(v, 6)
            self.rough_check = QCheckBox("Set per-colour roughness")
            self.rough_check.setToolTip(
                "Write myColorRoughness - a 0..1 roughness per paint colour (matte vs glossy). "
                "Off = leave it out of the file."
            )
            v.addWidget(self.rough_check)
            for i in range(len(role_labels)):
                rr = QHBoxLayout()
                rr.setSpacing(8)
                lab = QLabel(
                    role_labels[i] if i < len(role_labels) else ("Colour %d" % (i + 1))
                )
                lab.setFixedWidth(96)  # match the colour-picker label (same font/width)
                lab.setFixedHeight(28)
                sb = QDoubleSpinBox()
                sb.setRange(0.0, 1.0)
                sb.setSingleStep(0.01)
                sb.setDecimals(3)
                sb.setFixedHeight(28)
                sb.setMinimumWidth(90)
                sb.setValue(0.5)
                sb.setAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                sb.valueChanged.connect(lambda *_: self._row_changed())
                self.rough_spins.append(sb)
                self.rough_labels.append(lab)
                rr.addWidget(lab)
                rr.addStretch(1)
                rr.addWidget(sb)
                v.addLayout(rr)
            self.rough_check.toggled.connect(self._set_rough_active)
            self.rough_check.toggled.connect(lambda *_: self._row_changed())
            self._set_rough_active(
                self.rough_check.isChecked()
            )  # start greyed + disabled (default off)

        v.addSpacing(12)
        self.overwrite = QCheckBox("Overwrite file")
        self.overwrite.setToolTip(
            "Save over this section's loaded .blueitemtype instead of picking a new file "
            "(needs a file loaded)"
        )
        owrow = QHBoxLayout()
        owrow.setContentsMargins(0, 0, 0, 0)
        owrow.addWidget(self.overwrite)
        owrow.addStretch(1)
        reset = QPushButton("Reset Colour")
        reset.setToolTip("Clear every colour in this section back to its default")
        reset.setStyleSheet(_danger_css())
        reset.clicked.connect(self.reset_colours)
        owrow.addWidget(reset)
        v.addLayout(owrow)
        v.addSpacing(12)

        if note:
            notelbl = QLabel(note)
            notelbl.setObjectName("subtitle")
            notelbl.setWordWrap(True)
            v.addWidget(notelbl)
            v.addSpacing(6)

        # ---- export row: Export Pattern (.blueitemtype) + Export as Texture (.png) ----
        exprow = QHBoxLayout()
        exprow.setSpacing(6)
        ep = QPushButton("Export Pattern")
        ep.setObjectName("accent")
        ep.setToolTip(
            "Write the colours above back into the loaded .blueitemtype (or Save As a copy)"
        )
        ep.clicked.connect(self._export)
        exprow.addWidget(ep)
        self.export_btn = ep
        self._export_tip = ep.toolTip()
        # Hair has no "Export as Texture": its recolour isn't baked to a PNG here, and Export All as
        # Texture no longer depends on it - both the gating and the bake loop skip any section that
        # has no export_tex_btn, so leaving it unset for hair drops it out cleanly.
        if self._key != "hair":
            et = QPushButton("Export as Texture")
            et.setObjectName("accent")
            et.setToolTip("Bake this section's recoloured texture(s) and save PNG(s)")
            et.clicked.connect(self._export_texture)
            exprow.addWidget(et)
            self.export_tex_btn = et
            self._export_tex_tip = et.toolTip()
        v.addLayout(exprow)

        for r in self.rows:
            r.field.textChanged.connect(self._refresh_export_enabled)
        self._refresh_export_enabled()

    # ---- single-file load (.blueitemtype) --------------------------------
    def refresh_assets(self):
        """Re-read the current file for each asset picker row (after a pick or a reload)."""
        for ar in getattr(self, "asset_rows", []):
            ar.refresh()

    def set_load_handler(self, cb):
        """cb(explicit_path=None) is invoked when the user clicks Load or presses Enter in the
        path field (explicit_path set in the latter case)."""
        self._load_handler = cb

    def _on_load_clicked(self):
        self._invoke_load(None)

    def _load_entered(self):
        path = self.path_edit.text().strip()
        if path:
            self._invoke_load(path)

    def _invoke_load(self, explicit_path):
        if self._load_handler is None:
            return
        try:
            self._load_handler(explicit_path)
        except Exception:  # never let a slot exception abort Qt (SIGABRT)
            log.exception("slot raised (swallowed to keep Qt alive)")

    def apply_loaded(self, path, colors, color_indices):
        """Apply a parsed .blueitemtype's colours to the swatches and remember enough to export
        back into it later. `colors` is ['#rrggbb', ...] positional with `color_indices`
        (the file's own myColorN indices) and with self.rows."""
        if not colors:
            return
        self.path = path
        self._color_indices = list(color_indices)
        self.path_edit.setText(path)
        log.info("Na'vi %s blueitemtype loaded: %d colours from %s",
                 self._key, len(colors), path)
        self._busy = True
        for i, row in enumerate(self.rows):
            if i < len(colors):
                row.set_hex(colors[i], notify=False)
        self._busy = False
        self._row_changed()
        self._refresh_export_enabled()

    # ---- export ------------------------------------------------------------
    def all_valid(self):
        return all(r.is_valid() for r in self.rows)

    def _refresh_export_enabled(self, *_):
        loaded = bool(self.path) and bool(self._color_indices)
        valid = self.all_valid()
        all_default = valid and all(r.is_neutral() for r in self.rows)
        edited = valid and not all_default  # valid colours with at least one change from default
        if not valid:
            why = "Every colour must be a valid 6-digit hex code"
        elif all_default:
            why = "Change at least one colour from its default to export"
        else:
            why = ""
        # Export Pattern patches the loaded .blueitemtype, so it needs one loaded.
        pattern_ok = loaded and edited
        self.export_btn.setEnabled(pattern_ok)
        self.export_btn.setToolTip(
            self._export_tip if pattern_ok
            else ("Load a .blueitemtype into this section first" if not loaded else why)
        )
        # Export as Texture bakes the painted colours onto the textures - no .blueitemtype needed.
        if hasattr(self, "export_tex_btn"):
            # Skin and Eyes may bake at their DEFAULT colours too (handy for exporting the stock
            # recoloured texture); other sections (Warpaint) still need at least one colour change.
            tex_ok = valid if self._key in ("skin", "eye") else edited
            self.export_tex_btn.setEnabled(tex_ok)
            self.export_tex_btn.setToolTip(self._export_tex_tip if tex_ok else why)
        self.overwrite.setEnabled(bool(self.path))
        if not self.path and self.overwrite.isChecked():
            self.overwrite.setChecked(False)
        if self.on_validity_change:
            self.on_validity_change()

    def _export(self):
        import recolor_core

        if not self.path or not self._color_indices:
            QMessageBox.warning(
                self,
                "No file loaded",
                "Load a .blueitemtype into this section first - Export patches the colours "
                "back into the structure of a real loaded file.",
            )
            return
        if self.overwrite.isChecked():
            out = self.path
            replicated = False
        else:
            out, replicated = resolve_export_path(
                self, export_rel(BLUE_DIR_BLUEITEM, os.path.basename(self.path)),
                False, self.path,
                default_ext=".blueitemtype",
                file_filter="Blue item type (*.blueitemtype)",
                title="Export %s colours" % self._title,
            )
            if out is None:
                return
            if replicated:
                d = os.path.dirname(out)
                if d:
                    os.makedirs(d, exist_ok=True)
        hex_by_index = dict(zip(self._color_indices, self.colors()))
        kwargs = {}
        if self.mci_check is not None and self.mci_check.isChecked():
            kwargs["main_color_index"] = int(self.mci_spin.value())
        if self.rough_check is not None and self.rough_check.isChecked():
            kwargs["roughness"] = [round(s.value(), 4) for s in self.rough_spins]
        try:
            dest, n = recolor_core.update_blueitemtype_colors(
                self.path, hex_by_index, out_path=out, **kwargs
            )
        except Exception as e:  # noqa: BLE001
            log.exception("Na'vi %s blueitemtype export failed", self._key)
            QMessageBox.warning(self, "Export failed", str(e))
            return
        self.path = dest
        self.path_edit.setText(dest)
        self._refresh_export_enabled()
        log.info("Na'vi %s export -> %s (%d colours)", self._key, dest, n)
        if replicated:
            QMessageBox.information(
                self, "Export successful",
                f"Wrote:\n  {os.path.basename(dest)}\n\nInto:\n  {os.path.dirname(dest)}")
        return dest, n

    def _export_texture(self):
        """Bake this section's recoloured texture(s) to PNG. The actual bake lives in the main
        window (it has every resolved source texture + the recolour maths); we just hand it this
        section's key, colours and bio flag."""
        if self._on_export_texture is None:
            QMessageBox.information(
                self,
                "Export as Texture",
                "Texture baking isn't available in this build.",
            )
            return
        if not self.all_valid():
            QMessageBox.warning(
                self,
                "Invalid colours",
                "Every colour must be a valid 6-digit hex code first.",
            )
            return
        fmt = ask_export_format(self)
        if fmt is None:
            return
        try:
            self._on_export_texture(self._key, self.colors(), self.state(), fmt)
        except Exception:  # never let a slot exception abort Qt (SIGABRT)
            log.exception("slot raised (swallowed to keep Qt alive)")

    # ---- outward state ----------------------------------------------------
    def _row_changed(self):
        if not self._busy:
            self._on_change()
            self._refresh_export_enabled()  # picks go through set_hex (signals blocked), so the
            # textChanged-wired gate never fires for them; refresh here

    def _set_mci_active(self, on):
        """Enable + ungrey (or disable + grey) the main-colour-index spinner and its label,
        in lock-step with the tickbox, so an unticked control reads as inactive. The spinner's
        own number text is greyed too (the global '*' colour rule otherwise keeps it bright even
        when disabled); this is a colour-only override, so the native arrows are unaffected."""
        css = "" if on else "color:#566072;"
        if self.mci_spin is not None:
            self.mci_spin.setEnabled(on)
            self.mci_spin.setStyleSheet(css)
        if self.mci_label is not None:
            self.mci_label.setStyleSheet(css)

    def _set_rough_active(self, on):
        """Enable + ungrey (or disable + grey) every per-colour roughness spinner (including its
        number text) and its label, in lock-step with the tickbox."""
        css = "" if on else "color:#566072;"
        for sb in self.rough_spins:
            sb.setEnabled(on)
            sb.setStyleSheet(css)
        for lab in self.rough_labels:
            lab.setStyleSheet(css)

    def reset_colours(self):
        """Clear every swatch in this section (a blank box reads back as the default colour) and
        re-emit just once. The _busy guard swallows each row's own change signal so the viewer is
        updated a single time with the restored defaults."""
        self._busy = True
        for r in self.rows:
            r._reset()  # blank the box -> the row falls back to its default
        if (
            self.rough_check is not None
        ):  # roughness goes back to default (off, 0.5 each)
            self.rough_check.setChecked(False)
            for s in self.rough_spins:
                s.blockSignals(True)
                s.setValue(0.5)
                s.blockSignals(False)
        self._busy = False
        self._row_changed()

    def colors(self):
        return [r.hex() for r in self.rows]

    # ---- warpaint texture pickers (self-contained, per slot) ----
    def _pick_texture(self, slot):
        start = ""
        opt = assets.get_setting("smart_dialog_start", True)
        if opt:
            e = self.tex_slot_edits.get(slot)
            shown = (e.text().strip() if e else "") or self._tex_paths.get(slot) or ""
            start = assets.dialog_start_for(shown, "")  # file / its folder / a dir path
        if not start:
            start = self._tex_paths.get(slot) or ""  # else this slot's manual override,
            if not start:  # then any other populated slot's folder,
                for (
                    p
                ) in self._tex_paths.values():  # then the loaded .blueitemtype's folder
                    if p:
                        start = os.path.dirname(p)
                        break
                if not start and self.path:
                    start = os.path.dirname(self.path)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select %s texture" % slot,
            start,
            "Textures (*.dds *.png *.tga *.jpg *.jpeg);;All files (*)",
        )
        if path:
            self._set_texture(slot, path)

    def _texture_entered(self, slot):
        e = self.tex_slot_edits.get(slot)
        self._set_texture(slot, (e.text().strip() if e else "") or None)

    def _set_texture(self, slot, path):
        if path:
            self._tex_paths[slot] = path
        else:
            self._tex_paths.pop(slot, None)
        e = self.tex_slot_edits.get(slot)
        if e is not None:
            e.blockSignals(True)
            e.setText(path or "")
            e.blockSignals(False)
            e.setToolTip(path or "(optional)")
        self._row_changed()

    def export_preset(self):
        """Capture this section's .blueitemtype path, resolved colours, overwrite and bio flags."""
        d = {
            "blueitemtype": (self.path or ""),
            "color_indices": list(getattr(self, "_color_indices", []) or []),
            "colours": [r.effective() for r in self.rows],
            "overwrite": bool(self.overwrite.isChecked()),
        }
        if self.hide_toggle is not None:
            d["hide_warpaint"] = bool(self.hide_toggle.isChecked())
        if self._tex_paths:
            d["textures"] = {s: p for s, p in self._tex_paths.items() if p}
        if self.mci_check is not None:
            d["main_color_index"] = {
                "on": bool(self.mci_check.isChecked()),
                "value": int(self.mci_spin.value()),
            }
        if self.rough_check is not None:
            d["roughness"] = {
                "on": bool(self.rough_check.isChecked()),
                "values": [round(s.value(), 4) for s in self.rough_spins],
            }
        if self.strength_slider is not None and self._strength_key:
            d["strength"] = {
                "key": self._strength_key,
                "value": round(self.strength_slider.value() / 100.0, 4),
            }
        return d

    def apply_preset(self, d, missing):
        """Restore a section preset. A missing .blueitemtype is flagged and the overwrite target is
        dropped, but the saved colours are still applied (so the look is restored from the preset)."""
        d = d or {}
        bpath = (d.get("blueitemtype") or "").strip()
        if bpath and not os.path.isfile(bpath):
            missing.append(bpath)
            bpath = ""
        self.path = bpath or None
        self.path_edit.setText(bpath)
        if d.get("color_indices"):
            self._color_indices = list(d["color_indices"])
        cols = d.get("colours") or []
        self._busy = True
        for i, row in enumerate(self.rows):
            hexv = (cols[i] if i < len(cols) else "").strip()
            if hexv:
                row.set_hex(hexv, notify=False)
            else:
                row._reset()  # blank -> default (guarded by _busy, no emit)
        if self.hide_toggle is not None:
            self.hide_toggle.blockSignals(True)
            self.hide_toggle.setChecked(bool(d.get("hide_warpaint", False)))
            self.hide_toggle.blockSignals(False)
        # textures: new per-slot dict, or migrate the old single "texture" key -> body1
        tex_in = d.get("textures")
        if not isinstance(tex_in, dict):
            old = (d.get("texture") or "").strip()
            tex_in = {"body1": old} if old else {}
        self._tex_paths = {}
        for slot, _label in self.TEX_SLOTS:
            p = (tex_in.get(slot) or "").strip()
            if p and not os.path.isfile(p):
                missing.append(p)
                p = ""
            if p:
                self._tex_paths[slot] = p
            e = self.tex_slot_edits.get(slot)
            if e is not None:
                e.blockSignals(True)
                e.setText(p)
                e.blockSignals(False)
        mci = d.get("main_color_index")
        if self.mci_check is not None and isinstance(mci, dict):
            self.mci_spin.blockSignals(True)
            self.mci_spin.setValue(int(mci.get("value", 1) or 1))
            self.mci_spin.blockSignals(False)
            self.mci_check.blockSignals(True)
            self.mci_check.setChecked(bool(mci.get("on", False)))
            self.mci_check.blockSignals(False)
            self._set_mci_active(bool(mci.get("on", False)))
        rough = d.get("roughness")
        if self.rough_check is not None and isinstance(rough, dict):
            vals = rough.get("values") or []
            for i, s in enumerate(self.rough_spins):
                if i < len(vals):
                    s.blockSignals(True)
                    s.setValue(float(vals[i]))
                    s.blockSignals(False)
            on = bool(rough.get("on", False))
            self.rough_check.blockSignals(True)
            self.rough_check.setChecked(on)
            self.rough_check.blockSignals(False)
            self._set_rough_active(on)
        strength = d.get("strength")
        if self.strength_slider is not None and isinstance(strength, dict):
            self.strength_slider.blockSignals(True)
            self.strength_slider.setValue(
                int(round(float(strength.get("value", 1.0)) * 100))
            )
            self.strength_slider.blockSignals(False)
            if hasattr(self, "_strength_val"):
                self._strength_val.setText(
                    "%.2f" % (self.strength_slider.value() / 100.0)
                )
        self.overwrite.setChecked(bool(d.get("overwrite", False)))
        self._busy = False
        self._row_changed()
        self._refresh_export_enabled()

    def state(self):
        st = {"colors": self.colors()}
        if self.strength_slider is not None and self._strength_key:
            st[self._strength_key] = self.strength_slider.value() / 100.0
        if self.hide_toggle is not None:
            st["hide_warpaint"] = bool(self.hide_toggle.isChecked())
        if self._tex_paths:
            st["textures"] = dict(self._tex_paths)
        return st


class NaviControls(QWidget):
    """Na'vi recolour editor: skin, eyes, hair and warpaint sections, each loading a single
    .blueitemtype (Load button + path, like Banshee's pattern panels) and exporting the edited
    colours back into a file. Emits on_change(state) so the viewer re-tints live."""

    SECTION_SPECS = (
        ("skin", "Skin", ["Base skin", "Skin pattern"], False),
        (
            "eye",
            "Eyes",
            ["Right outer", "Right inner", "Left outer", "Left inner"],
            False,
        ),
        ("hair", "Hair", ["Root", "Mid", "Tip", "Hair cap"], False),
        ("warpaint", "Warpaint", ["Paint 1", "Paint 2", "Paint 3", "Paint 4"], False),
    )

    # Sections covered by "Export All". Warpaint is deliberately NOT included - it exports only
    # through its own Export Pattern / Export as Texture buttons, so the body-paint files never
    # ride along with a skin/eye/hair export by accident.
    EXPORT_ALL_KEYS = ("skin", "eye", "hair")

    # example expected .blueitemtype filename per section (shown as a hint under the Load row)
    PRESET_EXAMPLE = {
        "skin": "item_customization_player_skin_color_001.blueitemtype",
        "eye": "item_customization_player_eye_color_001.blueitemtype",
        "hair": "item_customization_player_hair_color_001.blueitemtype",
        "warpaint": "item_customization_player_warpaint_body_001.blueitemtype",
    }

    # per-section file pickers, grouped under sub-titles. Each section is a list of
    # (sub_title, [(button_label, asset_slot), ...]); labels follow Banshee's "Name (_suffix)" form.
    ASSET_SLOTS = {
        "skin": [
            ("Model(s)", [("Head (m)", "nav_head_m"), ("Head (f)", "nav_head_f")]),
            (
                "Textures",
                [("Head (_pat)", "nav_head_pat"), ("Body (_pat)", "nav_body_pat")],
            ),
        ],
        "eye": [
            (
                "Textures",
                [
                    ("Eye (_d)", "nav_eye_d"),
                    ("Eye (_n)", "nav_eye_n"),
                    ("Eye (_h)", "nav_eye_h"),
                    ("Eyelash (_d)", "nav_lash"),
                ],
            ),
        ],
        "hair": [
            ("Model(s)", [("Hair (.mmb)", "nav_hair"), ("Kuru (.mmb)", "nav_kuru")]),
            (
                "Hair Textures",
                [
                    ("Hair (_m)", "nav_hair_m"),
                    ("Hair (_ao)", "nav_hair_ao"),
                    ("Hair (_dir)", "nav_hair_dir"),
                    ("Decor (_d)", "nav_hair_acc"),
                    ("Decor (_m)", "nav_hair_acc_m"),
                    ("Decor (_n)", "nav_hair_acc_n"),
                ],
            ),
            (
                "Kuru Textures",
                [
                    ("Kuru (_m)", "nav_kuru_m"),
                    ("Kuru (_ao)", "nav_kuru_ao"),
                    ("Kuru (_dir)", "nav_kuru_dir"),
                    ("Decor (_d)", "nav_kuru_acc"),
                    ("Decor (_m)", "nav_kuru_acc_m"),
                    ("Decor (_n)", "nav_kuru_acc_n"),
                ],
            ),
        ],
    }

    def __init__(
        self,
        on_change=None,
        on_load=None,
        on_pick_asset=None,
        get_asset_path=None,
        on_set_asset_path=None,
        on_export_texture=None,
        fill_width=None,
        parent=None,
    ):
        super().__init__(parent)
        self.on_change = on_change or (lambda *_: None)
        self.on_load = on_load or (lambda *_: None)
        self.on_pick_asset = on_pick_asset
        self.get_asset_path = get_asset_path
        self.on_set_asset_path = on_set_asset_path
        self._on_export_texture = on_export_texture
        self._ov_guard = False
        # When the Na'vi tab passes a fill_width (the Ikran side-column width), the panel widens to
        # match the Ikran side and the sections stretch to fill it - so the Edit Na'vi sections line
        # up with the View Gear sections. Without it, fall back to the bare two-280-section span.
        self._fill_width = int(fill_width) if fill_width else 0
        self.setMaximumWidth(self._fill_width or (2 * 280 + 8 + 2 * 8))  # 584 fallback
        col = QVBoxLayout(self)
        col.setContentsMargins(8, 8, 8, 8)
        col.setSpacing(8)
        intro = QLabel(
            "Na'vi recolouring. Load a single .blueitemtype per feature, fine-tune any "
            "colour, then export the result - overwriting the loaded file or saving a copy."
        )
        intro.setObjectName("legend")
        intro.setWordWrap(True)
        col.addWidget(intro)
        col.addSpacing(12)  # match the Banshee pdbox -> panels gap

        # Top row (Skin | Eyes) is laid out so both stretch to the SAME height - that keeps their
        # top sub-titles aligned AND makes the bottom row start level. Bottom row (Hair | Warpaint)
        # is top-aligned only: both start at the same Y but keep their own natural heights.
        self.sections = {}
            # Startup default colours per section (real default-Na'vi presets):
            #   skin = skin_color_001 (#ADCBD7 base / #848B9A pattern)
            #   hair = hair_color_001 (#6B6B6B root / #1F1F1F mid / #323232 tip)
            #   eye  = eye_color_001  (#CEA86E outer / #D9E683 inner, both eyes)
            # The export gate treats these as each section's no-op baseline (_NaviSwatchRow.is_neutral /
            # reset), so export enables only once a colour changes.
        _DEFAULTS = {
            "skin": ["ADCBD7", "848B9A"],
            "hair": ["6B6B6B", "1F1F1F", "323232", "000000"],
            "eye": ["CEA86E", "D9E683", "CEA86E", "D9E683"],
        }
        for key, title, roles, with_bio in self.SECTION_SPECS:
            sec = _NaviSection(
                title,
                roles,
                self._changed,
                with_bio=with_bio,
                asset_groups=self.ASSET_SLOTS.get(key),
                on_pick_asset=on_pick_asset,
                get_asset_path=get_asset_path,
                on_set_asset_path=on_set_asset_path,
                key=key,
                on_export_texture=on_export_texture,
                preset_example=self.PRESET_EXAMPLE.get(key),
                defaults=_DEFAULTS.get(key),
                role_notes=(
                    {
                        "Hair cap": "Usually black. Only set this if your dye defines a "
                        "scalp colour (most don't); black = inherit the root."
                    }
                    if key == "hair"
                    else None
                ),
                strength_spec=(
                    ("Cap strength", "cap_strength", 1.0) if key == "hair" else None
                ),
                texture_picker=(key == "warpaint"),
                color_index=(key == "warpaint"),
                roughness=(key == "warpaint"),
                note=(
                    "Not part of Export All \u2014 use the buttons below to export "
                    "Warpaint on its own."
                    if key == "warpaint"
                    else None
                ),
            )
            if self._fill_width:
                # Match View Gear: stretch to fill the column (280 floor) instead of pinning to 280.
                sec.setMinimumWidth(280)
                sec.setSizePolicy(
                    QSizePolicy.Policy.Expanding, sec.sizePolicy().verticalPolicy()
                )
            else:
                sec.setFixedWidth(280)  # all navi sections exactly equal width
            self.sections[key] = sec
            sec.set_load_handler(lambda path, k=key, s=sec: self.on_load(k, s, path))
            sec.overwrite.toggled.connect(self._on_child_overwrite)
            sec.on_validity_change = self._refresh_export_all

        top_row = QHBoxLayout()
        top_row.setSpacing(8)  # same gap as the Banshee Head/Body panels
        _stretch = 1 if self._fill_width else 0
        top_row.addWidget(
            self.sections["skin"], _stretch
        )  # no AlignTop -> both stretch to equal height
        top_row.addWidget(self.sections["eye"], _stretch)
        col.addLayout(top_row)
        col.addSpacing(12)  # Banshee vertical section gap (between the rows)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)
        if self._fill_width:
            # No AlignTop: a non-zero alignment makes Qt size the widget to its hint instead of
            # filling the cell, so the sections only widen when added plain + stretch.
            bottom_row.addWidget(self.sections["hair"], 1)
            bottom_row.addWidget(self.sections["warpaint"], 1)
        else:
            bottom_row.addWidget(self.sections["hair"], 0, Qt.AlignmentFlag.AlignTop)
            bottom_row.addWidget(self.sections["warpaint"], 0, Qt.AlignmentFlag.AlignTop)
        col.addLayout(bottom_row)
        col.addSpacing(12)  # match the Banshee panels -> expbox gap

        # ---- Export All (mirrors the Banshee pattern editor's Export box) ----
        expbox = QGroupBox("Export")
        ebl = QVBoxLayout(expbox)
        ebl.setContentsMargins(10, 8, 10, 8)
        ebl.setSpacing(7)
        hint = QLabel(
            "Non-overwritten sections are written to the Export Folder set in Settings "
            "(rebuilding each file's blue/\u2026 path when 'Replicate the blue folder "
            "structure' is on)."
        )
        hint.setObjectName("legend")
        hint.setWordWrap(True)
        ebl.addWidget(hint)

        self.ov_master = QCheckBox("Overwrite all loaded files")
        self.ov_master.setToolTip(
            "Save every loaded section back over its own file. Ticks and locks the per-"
            "section overwrites and disables the folder name."
        )
        self.ov_master.toggled.connect(self._on_master_overwrite)
        ebl.addWidget(self.ov_master)

        export = QPushButton("Export All")
        export.setObjectName("accent")
        export.setFixedHeight(32)
        export.setToolTip(
            "Save every loaded section's colours (Skin, Eyes, Hair \u2014 not Warpaint)"
        )
        export.clicked.connect(self.export_all)
        ebl.addWidget(export)

        exnote = QLabel(
            "Covers Skin, Eyes and Hair. Warpaint is not included \u2014 export it "
            "from the Warpaint panel's own buttons."
        )
        exnote.setObjectName("subtitle")
        exnote.setWordWrap(True)
        ebl.addWidget(exnote)
        col.addWidget(expbox)
        self.export_all_btn = export
        self._export_all_tip = export.toolTip()

        col.addStretch(1)
        self._refresh_export_all()

    def _changed(self):
        self.on_change(self.state())

    def state(self):
        return {key: sec.state() for key, sec in self.sections.items()}

    def export_preset(self):
        """Per-section paths + colours for a Save/Load preset (all sections, warpaint included)."""
        return {key: sec.export_preset() for key, sec in self.sections.items()}

    def apply_preset(self, data, missing):
        data = data or {}
        for key, sec in self.sections.items():
            sec.apply_preset(data.get(key) or {}, missing)
        self._refresh_export_all()

    def reset_all(self):
        """Reset every section's colours (and its overwrite flag) to defaults."""
        for sec in self.sections.values():
            sec.overwrite.blockSignals(True)
            sec.overwrite.setChecked(False)
            sec.overwrite.blockSignals(False)
            sec.reset_colours()  # clears the swatches -> defaults, and re-emits
        self._refresh_export_all()

    def refresh_asset_rows(self):
        """Re-read each section's asset-picker filenames (after a pick or a viewer reload)."""
        for sec in self.sections.values():
            sec.refresh_assets()

    # ---- Export All wiring (mirrors BansheePatternEditor) -----------------
    def _on_child_overwrite(self, *_):
        if self._ov_guard:
            return
        self._refresh_export_all()

    def _on_master_overwrite(self, checked):
        if self._ov_guard:
            return
        self._ov_guard = True
        if checked:
            for sec in self._export_sections():
                if sec.path:
                    sec.overwrite.setChecked(True)
        self._ov_guard = False
        self._refresh_export_all()

    def _export_sections(self):
        """The sections that 'Export All' acts on - everything except Warpaint, which exports
        only through its own buttons."""
        return [self.sections[k] for k in self.EXPORT_ALL_KEYS if k in self.sections]

    def _refresh_export_all(self):
        export_secs = self._export_sections()
        loaded = [s for s in export_secs if s.path]
        any_loaded = bool(loaded)
        all_loaded = bool(export_secs) and len(loaded) == len(export_secs)
        all_valid = all(s.all_valid() for s in export_secs)

        self._ov_guard = True
        self.ov_master.setEnabled(any_loaded)
        if not self.ov_master.isEnabled() and self.ov_master.isChecked():
            self.ov_master.setChecked(False)
        master_on = self.ov_master.isChecked()
        for sec in export_secs:
            if master_on and sec.path:
                sec.overwrite.setChecked(True)
                sec.overwrite.setEnabled(False)
            else:
                sec.overwrite.setEnabled(bool(sec.path))
        self._ov_guard = False

        self.export_all_btn.setEnabled(all_loaded and all_valid)
        self.export_all_btn.setToolTip(
            self._export_all_tip
            if (all_loaded and all_valid)
            else (
                "Load a .blueitemtype into every section first"
                if not all_loaded
                else "Every colour must be a valid 6-digit hex code"
            )
        )

    def export_all(self):
        import recolor_core

        todo = [s for s in self._export_sections() if s.path]
        if not todo:
            return
        col_over = [s for s in todo if s.overwrite.isChecked()]
        col_new = [s for s in todo if not s.overwrite.isChecked()]
        base = None
        replicated = False
        if col_new:
            base, replicated = resolve_export_base(self)
            if base is None:
                return
        written = []
        try:
            for sec in col_over:
                hex_by_index = dict(zip(sec._color_indices, sec.colors()))
                recolor_core.update_blueitemtype_colors(sec.path, hex_by_index)
                written.append(os.path.basename(sec.path))
            for sec in col_new:
                hex_by_index = dict(zip(sec._color_indices, sec.colors()))
                # replicate on -> the canonical PandoraPaint/blue/.../blueitem path (regardless of
                # where the source was loaded); off -> drop it flat into the chosen folder.
                if replicated:
                    out = os.path.join(
                        base, export_rel(BLUE_DIR_BLUEITEM, os.path.basename(sec.path)))
                else:
                    out = os.path.join(base, os.path.basename(sec.path))
                d = os.path.dirname(out)
                if d:
                    os.makedirs(d, exist_ok=True)
                dest, _ = recolor_core.update_blueitemtype_colors(
                    sec.path, hex_by_index, out_path=out
                )
                sec.path = dest
                sec.path_edit.setText(dest)
                written.append(os.path.basename(dest))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(
                self, "Export failed", "The export did not complete:\n\n%s" % e
            )
            return
        self._refresh_export_all()
        msg = "Export successful.\n\nWrote:\n  " + "\n  ".join(written)
        if base:
            msg += "\n\nInto: %s" % base
        QMessageBox.information(self, "Export", msg)
        log.info("Na'vi export-all wrote %d file(s)%s",
                 len(written), (" into " + base) if base else "")


class _PresetCombo(QComboBox):
    """Dropdown that refreshes its list each time it's opened, so newly saved (or externally added)
    presets show up without needing to rebuild the bar."""

    def __init__(self, on_open, parent=None):
        super().__init__(parent)
        self._on_open = on_open

    def showPopup(self):
        try:
            self._on_open()
        except Exception:  # noqa: BLE001
            pass
        super().showPopup()


class SaveLoadBar(QGroupBox):
    """A compact Save / Load section. Save pops up a name dialog and writes the current paths +
    colours of a tab to a named .json preset (names are unique - saving over an existing name asks
    first); the preset folder is read from config (`preset_dir`). The dropdown loads a preset as
    soon as it's picked; its first entry, 'Default', resets the tab's values to their defaults. On
    load, any file path that no longer exists is flagged and the section falls back to its default.

    `collect()` returns the JSON-able payload dict; `apply_(payload, missing)` restores it (and
    appends missing file paths to `missing`); `reset_()` restores the tab's default values."""

    def __init__(self, title, kind, collect, apply_, reset_, parent=None):
        super().__init__(title, parent)
        self.kind = kind  # 'banshee' | 'navi' - stamped in the file, checked on load
        self._collect = collect
        self._apply = apply_
        self._reset = reset_  # () -> reset this tab's values to their defaults
        self._current_data = "__default__"  # the bar opens on Default; tracks the loaded selection
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 8)
        v.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.combo = _PresetCombo(self._refresh_presets)
        self.combo.setToolTip("Pick a saved preset to load it, or 'Default' to reset")
        self.combo.activated.connect(self._on_pick)
        save = QPushButton("Save\u2026")
        save.setObjectName("accent")
        save.setFixedWidth(86)
        save.setToolTip("Save the current paths and colours as a named preset")
        save.clicked.connect(self._save)
        delete = QPushButton("Delete")
        delete.setFixedWidth(86)
        delete.setStyleSheet(_danger_css())
        delete.setFixedHeight(
            save.sizeHint().height()
        )  # line up with the accent Save button
        delete.setToolTip("Delete the selected preset (asks first)")
        delete.clicked.connect(self._delete)
        row.addWidget(self.combo, 1)
        row.addWidget(save)
        row.addWidget(delete)
        v.addLayout(row)

        note = QLabel(
            "Optional. Saves this tab's current file paths and colours under a name so "
            "you can reload them later. 'Default' resets everything."
        )
        note.setObjectName("subtitle")
        note.setWordWrap(True)
        v.addWidget(note)

        self._refresh_presets()

    # ---- preset list ----
    def _kind_dir(self):
        """This bar's own preset sub-folder (<preset_dir>/banshee or /navi), created on access.
        Foldering by type keeps the Ikran dropdown from listing Na'vi presets and vice-versa, and
        lets an Ikran and a Na'vi preset share a name without clashing on disk."""
        d = os.path.join(assets.preset_dir(), self.kind)
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass
        return d

    @staticmethod
    def _migrate_flat():
        """One-time tidy: any loose .json saved directly in the preset folder (before presets were
        foldered by type) is moved into its type's sub-folder, so the per-tab dropdowns find it."""
        base = assets.preset_dir()
        try:
            flat = [
                f
                for f in os.listdir(base)
                if f.lower().endswith(".json") and os.path.isfile(os.path.join(base, f))
            ]
        except OSError:
            return
        for f in flat:
            src = os.path.join(base, f)
            try:
                with open(src, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                kind = payload.get("type") if isinstance(payload, dict) else None
            except Exception:  # noqa: BLE001
                continue
            if kind not in ("banshee", "navi"):
                continue  # leave anything unrecognised
            try:
                dstdir = os.path.join(base, kind)
                os.makedirs(dstdir, exist_ok=True)
                dst = os.path.join(dstdir, f)
                if not os.path.exists(dst):
                    os.replace(src, dst)
            except OSError:
                continue

    def _list_presets(self):
        """Names (no extension) of this tab's .json presets, case-insensitive sort."""
        self._migrate_flat()
        d = self._kind_dir()
        try:
            names = [
                os.path.splitext(f)[0]
                for f in os.listdir(d)
                if f.lower().endswith(".json")
            ]
        except OSError:
            names = []
        return sorted(names, key=str.lower)

    def _refresh_presets(self):
        names = self._list_presets()
        cur = self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("Default", "__default__")  # index 0: resets to defaults
        for n in names:
            self.combo.addItem(n, n)
        self.combo.setEnabled(True)
        self.combo.setCurrentIndex(1 + names.index(cur) if (cur in names) else 0)
        self.combo.blockSignals(False)

    def _on_pick(self, idx):
        data = self.combo.itemData(idx)
        if data == "__default__":
            if self._current_data == "__default__":
                return  # already on Default - nothing loaded to reset, so leave the work alone
            self._reset()
            self._current_data = "__default__"
        elif data:
            self._load_named(data)
            self._current_data = data

    # ---- save (popup name, unique) ----
    def _save(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Name this preset:")
        if not ok:
            return
        name = name.strip()
        if name.lower().endswith(".json"):
            name = name[:-5].strip()
        if not name:
            QMessageBox.warning(self, "Name needed", "Enter a name for the preset.")
            return
        if any(c in name for c in '/\\:*?"<>|'):
            QMessageBox.warning(
                self,
                "Invalid name",
                "The name can't contain any of  / \\ : * ? \" < > |",
            )
            return
        existing = {n.lower() for n in self._list_presets()}
        if name.lower() in existing:
            r = QMessageBox.question(
                self,
                "Name already used",
                'A preset named "%s" already exists. Overwrite it?' % name,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                return
        out = os.path.join(self._kind_dir(), name + ".json")
        payload = {"type": self.kind, "version": 1, "data": self._collect()}
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            QMessageBox.warning(
                self, "Save failed", "Could not write the preset:\n\n%s" % e
            )
            return
        self._refresh_presets()
        names = self._list_presets()
        if name in names:
            self.combo.setCurrentIndex(1 + names.index(name))
        QMessageBox.information(self, "Saved", 'Saved preset "%s".' % name)

    # ---- delete (with confirmation) ----
    def _delete(self):
        name = self.combo.currentData()
        if not name or name == "__default__":
            QMessageBox.information(
                self, "Nothing to delete", "Pick a saved preset from the list first."
            )
            return
        if (
            QMessageBox.question(
                self,
                "Delete preset",
                'Delete the preset "%s"? This can\'t be undone.' % name,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        path = os.path.join(self._kind_dir(), name + ".json")
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as e:
            QMessageBox.warning(
                self, "Delete failed", "Could not delete the preset:\n\n%s" % e
            )
            return
        self._refresh_presets()
        self.combo.setCurrentIndex(0)  # fall back to 'Default'
        self._reset()  # and actually reset the tab to its defaults

    # ---- load (by name) ----
    def _load_named(self, name):
        path = os.path.join(self._kind_dir(), name + ".json")
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Not found", "That preset no longer exists.")
            self._refresh_presets()
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:  # noqa: BLE001
            log.exception("preset read failed: %s", path)
            QMessageBox.warning(
                self, "Load failed", "Could not read the preset:\n\n%s" % e
            )
            return
        if not isinstance(payload, dict) or payload.get("type") != self.kind:
            QMessageBox.warning(
                self,
                "Wrong preset",
                '"%s" isn\'t a %s Save/Load preset.' % (name, self.kind),
            )
            return
        missing = []
        try:
            self._apply(payload.get("data") or {}, missing)
        except Exception as e:  # noqa: BLE001
            log.exception("preset apply failed")
            QMessageBox.warning(
                self, "Load failed", "The preset did not apply cleanly:\n\n%s" % e
            )
            return
        if missing:
            shown = "\n- ".join(missing[:20])
            extra = "" if len(missing) <= 20 else "\n(+%d more)" % (len(missing) - 20)
            QMessageBox.warning(
                self,
                "Some files were missing",
                "These files referenced by the preset weren't found, so the default was "
                "loaded in their place:\n\n- %s%s" % (shown, extra),
            )


# ===========================================================================
# Gear sub-tab: 4 identical slots (model + textures + 4 colours), plus a shared
# export-format chooser used by every "Export as Texture" button.
# ===========================================================================
def ask_export_format(parent):
    """Pop a small chooser for the export texture format. Returns 'png', 'dds' or None
    (cancelled). Shared by the Na'vi / Ikran section exports and the Gear export."""
    box = QMessageBox(parent)
    box.setWindowTitle("Export as Texture")
    box.setText("Choose the texture format to export:")
    b_png = box.addButton("PNG", QMessageBox.ButtonRole.AcceptRole)
    b_dds = box.addButton("DDS", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.exec()
    c = box.clickedButton()
    return {id(b_png): "png", id(b_dds): "dds"}.get(id(c))


def save_texture(img, base_path, fmt):
    """Save a PIL image as the chosen format. `base_path` has NO extension; returns the written
    path. PNG and (uncompressed) DDS go through Pillow; STF DDS needs the STF/BCn encoder, which
    isn't built yet, so it raises with a clear message."""
    fmt = (fmt or "png").lower()
    if fmt == "png":
        p = base_path + ".png"
        img.save(p)
        return p
    if fmt == "dds":
        p = base_path + ".dds"
        img.convert("RGBA").save(p)
        return p
    if fmt == "stf":
        raise NotImplementedError(
            "STF DDS export is coming next - it needs the STF/BCn encoder. "
            "Export as PNG or DDS for now."
        )
    raise ValueError("unknown export format: %r" % fmt)


class _GearSection(QGroupBox):
    """One gear slot on the View Gear sub-tab: a model picker (or a male/female pair for Na'vi) plus
    the gear texture pickers, sized to match the Na'vi sections. The model loads on top of the
    Na'vi / Ikran in the shared viewer so you can preview gear on the character. A 'Hide Gear'
    tickbox drops just this piece from the preview."""

    BTN_W = 118
    # (slot, file-suffix, description) - buttons read 'Gear (_<suffix>)' like the Na'vi/Ikran rows
    TEX_SLOTS = [
        ("diffuse", "_d", "diffuse"),
        ("material", "_m", "material"),
        ("normal", "_n", "normal"),
        ("regions", "_reg_m", "region mask"),
        ("detail", "_dn", "detail normal"),
        ("crafted", "_cn", "crafted normal"),
        ("emissive", "_e", "emissive"),
    ]

    def __init__(
        self,
        title,
        on_change=None,
        on_pick_model=None,
        on_pick_texture=None,
        on_hide=None,
        on_model_change=None,
        gendered=False,
        gender="m",
        camo=False,
        on_camo=None,
        on_pattern=None,
        expand=False,
    ):
        super().__init__(title)
        self._key = title.lower().replace(" ", "_").replace("'", "").replace("/", "_")
        self._on_change = on_change
        self._on_pick_model = on_pick_model
        self._on_pick_texture = on_pick_texture
        self._on_hide = on_hide
        self._on_model_change = on_model_change
        self._camo = bool(camo)
        self._on_camo = on_camo
        self._on_pattern = on_pattern
        self._busy = False
        self.gendered = bool(gendered)
        self._gender = gender if gender in ("m", "f") else "m"
        # model_paths holds the gendered pair; for the single (Ikran) picker only "m" is used.
        self.model_paths = {"m": None, "f": None}
        self._tex_paths = {}
        self.tex_edits = {}
        self._model_rows = {}  # gender -> (row widget, line edit)

        if expand:
            # Ikran View Gear: fill the (wider-than-584) Edit-Ikran column instead of pinning to 280,
            # so the 2x2 grid stretches edge to edge with the same 8px side margins + column gap as the
            # Edit Ikran sections. 280 stays the floor so it never shrinks below the Na'vi width.
            self.setMinimumWidth(280)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding, self.sizePolicy().verticalPolicy()
            )
        else:
            self.setFixedWidth(280)  # same width as the Na'vi sections
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)

        # ---- Model(s) ----
        mh = QLabel("Model(s)")
        mh.setObjectName("sectiontitle")
        v.addWidget(mh)
        if self.gendered:
            self._add_model_row(v, "m", "Gear (m)", "male gear model (.mmb)")
            self._add_model_row(v, "f", "Gear (f)", "female gear model (.mmb)")
        else:
            self._add_model_row(v, "m", "Gear", "gear model (.mmb)")
        # Hide Gear tickbox, pinned at the bottom of the Model(s) section
        self.hide_cb = QCheckBox("Hide Gear")
        self.hide_cb.setToolTip("Hide just this gear piece in the preview")
        self.hide_cb.toggled.connect(self._hide_toggled)
        v.addWidget(self.hide_cb)

        _hrule(v)

        # ---- Textures ----
        th = QLabel("Textures")
        th.setObjectName("sectiontitle")
        v.addWidget(th)
        for slot, suffix, desc in self.TEX_SLOTS:
            trow = QHBoxLayout()
            trow.setSpacing(6)
            tbtn = QPushButton("Gear (%s)" % suffix)
            tbtn.setFixedWidth(self.BTN_W)
            tbtn.setToolTip("Choose this gear piece's %s texture (%s)" % (desc, suffix))
            tbtn.clicked.connect(
                lambda _=False, s=slot: (
                    self._on_pick_texture(self._key, s)
                    if self._on_pick_texture
                    else None
                )
            )
            edit = QLineEdit()
            edit.setPlaceholderText("required" if slot == "diffuse" else "(optional)")
            edit.setToolTip("Pick a file, or type/paste a path here")
            edit.editingFinished.connect(lambda s=slot: self._tex_text_committed(s))
            self.tex_edits[slot] = edit
            trow.addWidget(tbtn)
            trow.addWidget(edit, 1)
            v.addLayout(trow)

        # ---- Camo (camo tab only) ----
        # Pick which palette to apply to THIS piece, plus a tickbox to turn it off. The colours come
        # from the live, in-memory palette (so edits in Camo Colours show here) - resolved by the
        # window, not read from disk.
        if self._camo:
            _hrule(v)
            ch = QLabel("Camo")
            ch.setObjectName("sectiontitle")
            v.addWidget(ch)
            self.camo_select = QComboBox()
            self.camo_select.setToolTip(
                "Camo palette to apply to this piece (uses your live-edited colours)"
            )
            self.camo_select.currentIndexChanged.connect(self._camo_emit)
            v.addWidget(self.camo_select)
            self.pattern_select = QComboBox()
            self.pattern_select.addItem("Solid (flat colour)", "solid")
            self.pattern_select.addItem("Tiger stripe (pattern)", "tigerstripe")
            self.pattern_select.setToolTip(
                "Region mask for this piece: Solid is the hard-surface camo (flat primary, e.g. "
                "guns); Tiger stripe is the cloth 3-region pattern (e.g. wrapped Na'vi weapons)."
            )
            self.pattern_select.currentIndexChanged.connect(self._pattern_emit)
            v.addWidget(self.pattern_select)
            self.camo_apply = QCheckBox("Apply camo")
            self.camo_apply.setChecked(True)
            self.camo_apply.setToolTip("Untick to preview this piece with no camo")
            self.camo_apply.toggled.connect(self._camo_emit)
            v.addWidget(self.camo_apply)

        self._apply_gender_visibility()

    def populate_camo(self, options):
        """Fill the per-piece camo dropdown. `options` is a list of (label, (subtab, name)); the
        prior selection is kept if it is still present. Signals are blocked during the refill, then
        the (valid) selection is emitted once. No-op for non-camo sections."""
        if not getattr(self, "_camo", False) or not hasattr(self, "camo_select"):
            return
        prev = self.camo_select.currentData()
        self.camo_select.blockSignals(True)
        self.camo_select.clear()
        for label, data in options:
            self.camo_select.addItem(label, data)
        if prev is not None:
            i = self.camo_select.findData(prev)
            if i >= 0:
                self.camo_select.setCurrentIndex(i)
        self.camo_select.blockSignals(False)
        self._camo_emit()

    def _camo_emit(self, *args):
        if self._on_camo and hasattr(self, "camo_select"):
            self._on_camo(
                self._key, self.camo_select.currentData(), self.camo_apply.isChecked()
            )

    def camo_pattern(self):
        """-> 'solid' | 'tigerstripe' for this piece (defaults to 'solid' for non-camo sections)."""
        if not getattr(self, "_camo", False) or not hasattr(self, "pattern_select"):
            return "solid"
        return self.pattern_select.currentData() or "solid"

    def _pattern_emit(self, *args):
        if self._on_pattern and hasattr(self, "pattern_select"):
            self._on_pattern(self._key, self.camo_pattern())

    def _add_model_row(self, v, gender, label, placeholder):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        btn = QPushButton(label)
        btn.setFixedWidth(self.BTN_W)
        btn.setToolTip("Choose a gear model (.mmb) to load on top of the character")
        g = gender if self.gendered else None
        btn.clicked.connect(
            lambda _=False, gg=g: (
                self._on_pick_model(self._key, gg) if self._on_pick_model else None
            )
        )
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setToolTip("Pick a model, or type/paste a path here")
        edit.editingFinished.connect(lambda g=gender: self._model_text_committed(g))
        h.addWidget(btn)
        h.addWidget(edit, 1)
        v.addWidget(row)
        self._model_rows[gender] = (row, edit)

    def _model_text_committed(self, gender):
        """User typed/pasted a path into a model field: apply it (or revert if it's not a file)."""
        if self._busy:
            return
        row = self._model_rows.get(gender)
        if not row:
            return
        txt = row[1].text().strip()
        if txt and not os.path.isfile(txt):
            row[1].setText(self.model_paths.get(gender) or "")  # bad path -> revert
            return
        self.set_model_path(txt or None, gender)
        if self._on_model_change:
            self._on_model_change(self._key, gender)

    def _tex_text_committed(self, slot):
        """User typed/pasted a path into a texture field: apply it (or revert if it's not a file)."""
        if self._busy:
            return
        edit = self.tex_edits.get(slot)
        if edit is None:
            return
        txt = edit.text().strip()
        if txt and not os.path.isfile(txt):
            edit.setText(self._tex_paths.get(slot) or "")  # bad path -> revert
            return
        self.set_texture_path(slot, txt or None)  # fires _changed -> preview

    def _apply_gender_visibility(self):
        # Show BOTH the male and female model pickers (and the single Ikran one). The body type
        # only decides which model the live preview uses, not which picker is visible.
        for _g, (row, _e) in self._model_rows.items():
            row.setVisible(True)

    def set_gender(self, gender):
        """Set which model the preview uses (male/female); both pickers stay visible."""
        if gender in ("m", "f"):
            self._gender = gender

    # --- setters the main window calls after a file dialog ---
    def set_model_path(self, path, gender=None):
        g = gender if (self.gendered and gender in ("m", "f")) else "m"
        self.model_paths[g] = path or None
        row = self._model_rows.get(g)
        if row:
            row[1].setText(path or "")
            row[1].setToolTip(path or "")

    @property
    def model_path(self):
        """The model for the currently-shown gender (the single model for Ikran)."""
        return self.model_paths.get(self._gender if self.gendered else "m")

    def active_model(self):
        return self.model_path

    def set_texture_path(self, slot, path):
        if path:
            self._tex_paths[slot] = path
        else:
            self._tex_paths.pop(slot, None)
        if slot in self.tex_edits:
            self.tex_edits[slot].setText(path or "")
            self.tex_edits[slot].setToolTip(path or "")
        self._changed()

    def hidden(self):
        return self.hide_cb.isChecked()

    def restore(self, st, missing=None):
        """Restore this section from a state() dict; append missing file paths to `missing`."""
        self._busy = True
        try:
            if self.gendered:
                for g in ("m", "f"):
                    mp = st.get("model_%s" % g)
                    self.set_model_path(mp, g)
                    if missing is not None and mp and not os.path.isfile(mp):
                        missing.append(mp)
            else:
                mp = st.get("model")
                self.set_model_path(mp)
                if missing is not None and mp and not os.path.isfile(mp):
                    missing.append(mp)
            for slot in list(self._tex_paths):
                self.set_texture_path(slot, None)
            for slot, p in (st.get("textures") or {}).items():
                if missing is not None and p and not os.path.isfile(p):
                    missing.append(p)
                self.set_texture_path(slot, p)
            self.hide_cb.blockSignals(True)
            self.hide_cb.setChecked(bool(st.get("hidden", False)))
            self.hide_cb.blockSignals(False)
        finally:
            self._busy = False
        self._changed()

    def reset(self):
        """Clear the model(s), textures + hide flag back to defaults."""
        self._busy = True
        try:
            for g in ("m", "f"):
                self.set_model_path(None, g)
            for slot in list(self._tex_paths):
                self.set_texture_path(slot, None)
            self.hide_cb.blockSignals(True)
            self.hide_cb.setChecked(False)
            self.hide_cb.blockSignals(False)
        finally:
            self._busy = False
        self._changed()

    def _hide_toggled(self, _checked):
        if not self._busy and self._on_hide:
            self._on_hide(self._key)

    def _changed(self):
        if not self._busy and self._on_change:
            self._on_change(self._key)

    def state(self):
        st = {"textures": dict(self._tex_paths), "hidden": self.hide_cb.isChecked()}
        if self.gendered:
            st["model_m"] = self.model_paths.get("m")
            st["model_f"] = self.model_paths.get("f")
        else:
            st["model"] = self.model_paths.get("m")
        return st


class GearControls(QWidget):
    """The View Gear sub-tab body: four identical gear slots in a 2x2 grid, sized like NaviControls."""

    def __init__(
        self,
        on_change=None,
        on_pick_model=None,
        on_pick_texture=None,
        on_hide=None,
        on_model_change=None,
        gendered=False,
        gender="m",
        stacked=False,
        title_prefix="Gear",
        camo=False,
        on_camo=None,
        on_pattern=None,
        fill_width=None,
        parent=None,
    ):
        super().__init__(parent)
        self.sections = []
        self.gendered = bool(gendered)
        self._stacked = bool(stacked)
        if self._stacked:
            self.setMaximumWidth(280 + 2 * 8)  # single column of full-width sections
            _sec_stretch = 0
        elif fill_width:
            # Stretch the 2x2 grid to fill a wider column - the Ikran View Gear tab matches the
            # Edit Ikran column width rather than shrinking to the bare 584 section span.
            self.setMaximumWidth(int(fill_width))
            _sec_stretch = 1
        else:
            self.setMaximumWidth(2 * 280 + 8 + 2 * 8)  # = 584, matching NaviControls
            _sec_stretch = 0

        col = QVBoxLayout(self)
        col.setContentsMargins(8, 8, 8, 8)
        col.setSpacing(8)

        intro = QLabel(
            "Gear preview only. Load gear models and textures to see them on the character - "
            "these sections just show changes and don't recolour textures like the other tabs."
        )
        intro.setObjectName("legend")
        intro.setWordWrap(True)
        col.addWidget(intro)
        col.addSpacing(12)  # breathing room before the first row

        for i in range(4):
            sec = _GearSection(
                "%s %d" % (title_prefix, i + 1),
                on_change=on_change,
                on_pick_model=on_pick_model,
                on_pick_texture=on_pick_texture,
                on_hide=on_hide,
                on_model_change=on_model_change,
                gendered=gendered,
                gender=gender,
                camo=camo,
                on_camo=on_camo,
                on_pattern=on_pattern,
                expand=bool(_sec_stretch),
            )
            self.sections.append(sec)

        if self._stacked:  # each gear slot as its own full-width section, stacked vertically
            for sec in self.sections:
                col.addWidget(sec, 0, Qt.AlignmentFlag.AlignTop)
                col.addSpacing(8)
            col.addStretch(1)
        else:
            top_row = QHBoxLayout()
            top_row.setSpacing(8)
            bottom_row = QHBoxLayout()
            bottom_row.setSpacing(8)
            if _sec_stretch:
                # No alignment flag: a non-zero alignment makes Qt size the widget to its hint
                # instead of filling the cell, so the sections only widen when added plain + stretch.
                top_row.addWidget(self.sections[0], 1)
                top_row.addWidget(self.sections[1], 1)
                bottom_row.addWidget(self.sections[2], 1)
                bottom_row.addWidget(self.sections[3], 1)
            else:
                top_row.addWidget(self.sections[0], 0, Qt.AlignmentFlag.AlignTop)
                top_row.addWidget(self.sections[1], 0, Qt.AlignmentFlag.AlignTop)
                bottom_row.addWidget(self.sections[2], 0, Qt.AlignmentFlag.AlignTop)
                bottom_row.addWidget(self.sections[3], 0, Qt.AlignmentFlag.AlignTop)
            col.addLayout(top_row)
            col.addSpacing(12)  # extra gap under Gear 1 / Gear 2 (like the
            # section gap on the Na'vi / Banshee tabs)
            col.addLayout(bottom_row)
            col.addStretch(1)  # keep the rows top-packed

    def set_gender(self, gender):
        """Show the male/female model picker matching the selected body type, across all slots."""
        for s in self.sections:
            s.set_gender(gender)

    def populate_camo(self, options):
        """Fill every section's per-piece camo dropdown with `options` = [(label, (subtab, name))]."""
        for s in self.sections:
            if hasattr(s, "populate_camo"):
                s.populate_camo(options)

    def section(self, key):
        for s in self.sections:
            if s._key == key:
                return s
        return None


# =====================================================================================
# App chrome: left section nav (MainNav) + custom frameless titlebar (TitleBar).
# =====================================================================================
_NAV_FONT_FAMILIES = [
    "Inter",
    "Inter Variable",
    "Segoe UI Variable Text",
    "Segoe UI",
    "Noto Sans",
    "DejaVu Sans",
]
_NAV_FONT = ",".join(f"'{f}'" for f in _NAV_FONT_FAMILIES) + ",sans-serif"


def _nav_font():
    """The font item-view rows must be given explicitly: QListWidget paints item text via its
    delegate using the widget's QFont, so a font-family in the ::item stylesheet is ignored.
    This matches the secondary tabs (Inter, 15px)."""
    f = QFont()
    f.setFamilies(_NAV_FONT_FAMILIES)
    f.setPixelSize(15)
    return f


_SIDEBAR_QSS = """
QFrame#navsidebar { background:#0C0F14; border:none; border-right:1px solid #1A202A; }
QListWidget#mainnav { background:transparent; border:none; outline:0; padding:0; font-size:21px; }
QListWidget#mainnav::item { background:#0F1117; color:#8A93A3; padding:17px 16px 17px 14px; margin:0;
                            border:none; border-left:4px solid #234E57; border-radius:0; }
QListWidget#mainnav::item:hover { background:#171C24; color:#C7D0DC; border-left:4px solid #C7D0DC; }
QListWidget#mainnav::item:selected { background:#0B0D11; color:#22D3EE; border-left:4px solid #22D3EE; }
QListWidget#mainnav::item:disabled { background:#0F1117; color:#3F4754; border-left:4px solid #3F4754; }
"""


# nav icon colours. Normal (inactive) + Selected (active) are driven by the theme accents at build
# time; hover + disabled are fixed (not accents).
_NAV_ICON_HOVER = "#C7D0DC"     # hover: light (same as the hover text + line)
_NAV_ICON_DISABLED = "#3F4754"  # disabled: grey (same as the text)
_NAV_ICON_PX = 22               # icon box size in the sidebar
_NAV_ICON_TEXT_GAP = 10         # transparent pad on the icon's right -> gap between icon and tab text


def _tint_pixmap(px, hexcolor):
    """Recolour a line-art pixmap to a solid colour, preserving its alpha (shape)."""
    out = QPixmap(px.size())
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.drawPixmap(0, 0, px)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    p.fillRect(out.rect(), QColor(hexcolor))
    p.end()
    return out


def _make_nav_icon(path):
    """Build a multi-state QIcon for a sidebar tab from a line-art PNG: cyan when unselected,
    dark when selected, grey when disabled - each tinted to match the tab text. Every icon is
    composited onto a fixed square canvas (left-aligned), so the icon column is one constant
    width and the row text always starts at the same x. Returns None if the file is missing."""
    src = QPixmap(path)
    if src.isNull():
        return None
    scaled = src.scaled(
        _NAV_ICON_PX, _NAV_ICON_PX,
        Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
    )

    def _canvas(color):
        tinted = _tint_pixmap(scaled, color)
        # canvas is wider than the icon by _NAV_ICON_TEXT_GAP: the icon sits left + v-centred, the
        # gap is transparent on the right, so the tab text starts that bit further from the icon
        # while icon and text stay vertically centred on the same row.
        c = QPixmap(_NAV_ICON_PX + _NAV_ICON_TEXT_GAP, _NAV_ICON_PX)
        c.fill(Qt.GlobalColor.transparent)
        p = QPainter(c)
        p.drawPixmap(0, (_NAV_ICON_PX - tinted.height()) // 2, tinted)  # left-aligned, v-centred
        p.end()
        return c

    icon = QIcon()
    icon.addPixmap(_canvas(theme.accent_inactive()), QIcon.Mode.Normal)
    icon.addPixmap(_canvas(_NAV_ICON_HOVER), QIcon.Mode.Active)
    icon.addPixmap(_canvas(theme.accent_active()), QIcon.Mode.Selected)
    icon.addPixmap(_canvas(_NAV_ICON_DISABLED), QIcon.Mode.Disabled)
    return icon


class MainNav(QWidget):
    """Left section menu (Ikran / Na'vi / Item Wiki / Settings) + stacked content.

    Exposes the subset of the QTabWidget API MainWindow uses (addTab / setCurrentIndex /
    currentIndex / setTabEnabled / setTabToolTip / currentChanged), so it drops in for the
    former top-level QTabWidget with almost no change to the surrounding wiring."""

    currentChanged = pyqtSignal(int)

    def __init__(self, header=None, footer=None, top_pad=0, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._side = QFrame()
        self._side.setObjectName("navsidebar")
        self._side.setStyleSheet(theme.apply(_SIDEBAR_QSS))
        self._icon_paths = []  # per-row icon path, so restyle() can re-tint icons to a new accent
        self._actions = {}  # row -> callback for button-style rows (no content page; e.g. Report Bugs)
        sv = QVBoxLayout(self._side)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(0)
        if header is not None:
            sv.addSpacing(14)  # equal breathing room above the logo
            sv.addWidget(header)
            sv.addSpacing(14)  # ...and below, so it doesn't sit on top of the side tabs
        elif top_pad:
            # drop the first side-tab down so its top lines up with the secondary tabs, which sit
            # under the main tab row by the tab-page top gap.
            sv.addSpacing(top_pad)
        self._list = QListWidget()
        self._list.setObjectName("mainnav")
        self._list.setFont(_nav_font())
        self._list.setIconSize(QSize(_NAV_ICON_PX + _NAV_ICON_TEXT_GAP, _NAV_ICON_PX))
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        sv.addWidget(self._list, 1)
        if footer is not None:
            sv.addWidget(footer)  # logo + wordmark pinned to the bottom-left of the sidebar

        self._stack = QStackedWidget()
        lay.addWidget(self._side)
        lay.addWidget(self._stack, 1)

        self._list.currentRowChanged.connect(self._on_row)
        self._list.itemClicked.connect(self._on_item_clicked)

    # ---- internal ----
    def _on_row(self, row):
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)
        self.currentChanged.emit(row)

    def _on_item_clicked(self, item):
        # button-style rows (added via addAction) are enabled but NOT selectable, so they never
        # become the current row - no currentRowChanged, no content switch, and the active tab keeps
        # its selection + accent highlight. itemClicked still fires once per click, so the action
        # runs here (single fire - none of the re-entrant double-firing of a snap-back approach).
        cb = self._actions.get(self._list.row(item))
        if cb is not None:
            cb()

    def _fit_width(self):
        w = max(160, self._list.sizeHintForColumn(0) + 40)
        self._side.setFixedWidth(w)

    # ---- QTabWidget-compatible API used by MainWindow ----
    def addTab(self, widget, label, icon=None):
        self._stack.addWidget(widget)
        item = QListWidgetItem(label)
        if icon:
            nav_icon = _make_nav_icon(icon)
            if nav_icon is not None:
                item.setIcon(nav_icon)
        self._list.addItem(item)
        self._icon_paths.append(icon)
        idx = self._list.count() - 1
        if self._list.currentRow() < 0:
            self._list.setCurrentRow(0)
        self._fit_width()
        return idx

    def addAction(self, label, icon, callback):
        """Add a button-style row below the tabs (e.g. 'Report Bugs') that fires `callback` when
        clicked and does NOT switch content - the selection snaps back to the current tab. Styled
        exactly like a tab (tinted icon + sidebar QSS, re-tinted by restyle()). Returns its row."""
        item = QListWidgetItem(label)
        # Enabled + clickable but NOT selectable: the row acts as a button - it never becomes the
        # current row, so it can't switch content or steal the active tab's selection/accent. The
        # click is handled in _on_item_clicked (via itemClicked), not the selection signal.
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        if icon:
            nav_icon = _make_nav_icon(icon)
            if nav_icon is not None:
                item.setIcon(nav_icon)
        self._list.addItem(item)
        self._icon_paths.append(icon)  # so restyle() re-tints it with the other nav icons
        row = self._list.count() - 1
        self._actions[row] = callback
        self._fit_width()
        return row

    def restyle(self):
        """Re-apply the themed sidebar stylesheet and re-tint every nav icon to the current accents.
        Lets the Settings colour panel re-theme the side menu live, without a restart."""
        self._side.setStyleSheet(theme.apply(_SIDEBAR_QSS))
        # QListWidget caches its resolved ::item rules, so the selected-tab accent (text + the 4px
        # left border) won't repaint on a bare setStyleSheet - force the sidebar + list to re-resolve.
        for w in (self._side, self._list):
            w.style().unpolish(w)
            w.style().polish(w)
            w.update()
        for row, path in enumerate(self._icon_paths):
            it = self._list.item(row)
            if it is None or not path:
                continue
            nav_icon = _make_nav_icon(path)
            if nav_icon is not None:
                it.setIcon(nav_icon)

    def setCurrentIndex(self, idx):
        if 0 <= idx < self._list.count():
            self._list.setCurrentRow(idx)

    def currentIndex(self):
        return self._list.currentRow()

    def count(self):
        return self._list.count()

    def widget(self, idx):
        return self._stack.widget(idx)

    def setTabEnabled(self, idx, enabled):
        it = self._list.item(idx)
        if it is None:
            return
        fl = it.flags()
        sel = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        it.setFlags((fl | sel) if enabled else (fl & ~sel))

    def setTabToolTip(self, idx, text):
        it = self._list.item(idx)
        if it is not None:
            it.setToolTip(text)

    def setTabText(self, idx, text):
        it = self._list.item(idx)
        if it is not None:
            it.setText(text)
        self._fit_width()


_TITLEBAR_QSS = """
QWidget#titlebar { background:#0F1117; }
QLabel#titletext { color:#8A93A3; font-size:13px; background:transparent; }
QPushButton#winbtn { background:transparent; border:none; }
QPushButton#winbtn:hover { background:#1C212A; }
QPushButton#winbtn:pressed { background:#252B36; }
QPushButton#winclose { background:#C0392B; border:none; }
QPushButton#winclose:hover { background:#E04434; }
QPushButton#winclose:pressed { background:#9B2D22; }
"""

_WIN_BTN_W = 46  # window-button width; the title is offset by 3x this so it centres on the bar


class _WinBtn(QPushButton):
    """A window control whose icon (min / max / restore / close) is painted, so it is always
    crisp and perfectly centred regardless of any glyph font's metrics."""

    def __init__(self, kind, name):
        super().__init__()
        self._kind = kind
        self.setObjectName(name)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_kind(self, kind):
        self._kind = kind
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)  # QSS background / hover / pressed
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._kind == "close":
            col = QColor("#FFFFFF")
        else:
            col = QColor("#E6EAF1" if self.underMouse() else "#C7D0DC")
        pen = QPen(col)
        pen.setWidthF(1.3)
        p.setPen(pen)
        r = self.rect()
        cx = r.center().x() + 0.5
        cy = r.center().y() + 0.5
        h = 4.5  # half icon size (9px box)
        if self._kind == "min":
            p.drawLine(QPointF(cx - h, cy), QPointF(cx + h, cy))
        elif self._kind == "max":
            p.drawRect(QRectF(cx - h, cy - h, 2 * h, 2 * h))
        elif self._kind == "restore":
            o = 2.0
            p.drawRect(QRectF(cx - h + o, cy - h - o, 2 * h - o, 2 * h - o))  # back
            p.drawRect(QRectF(cx - h - o, cy - h + o, 2 * h - o, 2 * h - o))  # front
        elif self._kind == "close":
            p.drawLine(QPointF(cx - h, cy - h), QPointF(cx + h, cy + h))
            p.drawLine(QPointF(cx - h, cy + h), QPointF(cx + h, cy - h))
        p.end()


class TitleBar(QWidget):
    """Custom titlebar for the frameless main window."""

    def __init__(self, window, title="Pandora Paint", height=25):
        super().__init__()
        self._win = window
        self.setObjectName("titlebar")
        self.setStyleSheet(_TITLEBAR_QSS)
        self.setFixedHeight(height)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        # left spacer == total button width, so the centred label lands on the true bar centre
        spacer = QWidget()
        spacer.setFixedWidth(_WIN_BTN_W * 3)
        lay.addWidget(spacer)
        self._label = QLabel(title)
        self._label.setObjectName("titletext")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._label, 1)
        self._min = self._btn("min", "winbtn", self._win.showMinimized, height)
        self._max = self._btn("max", "winbtn", self._toggle_max, height)
        self._close = self._btn("close", "winclose", self._win.close, height)
        for b in (self._min, self._max, self._close):
            lay.addWidget(b)
        self.sync_max_glyph()

    def _btn(self, kind, name, slot, height):
        b = _WinBtn(kind, name)
        b.setFixedSize(_WIN_BTN_W, height)
        b.clicked.connect(slot)
        return b

    def _toggle_max(self):
        if self._win.isMaximized():
            self._win.showNormal()
        else:
            self._win.showMaximized()
        self.sync_max_glyph()

    def sync_max_glyph(self):
        self._max.set_kind("restore" if self._win.isMaximized() else "max")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            handle = self._win.windowHandle()
            if handle is not None:
                handle.startSystemMove()

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._toggle_max()


# ======================================================================================
# Camo editor: data layer (gearcamo_colorpalettes.rejuice) + the Camo tab's controls panel.
# The rejuice is the "binjuice" text-binary format; colours are fixed-width ASCII 0xAARRGGBB,
# so edits are an in-place 8-char patch (file length + every other byte preserved -> export
# can't corrupt it). Palettes bucket into the seven gearcamo wiki subtabs (the dropdowns).
# Byte-verified: file RGB matches the wiki for all 130 palettes.
# ======================================================================================

# the seven dropdowns, in wiki order
CAMO_SUBTABS = [
    "Gear (Region)",
    "Weapon (Region)",
    "Player Gear",
    "Player Weapon",
    "Banshee Gear",
    "Windtrader",
    "Ash",
]

_CAMO_RX_BLOCK = re.compile(rb"GearCamoColorPalette\x00([^\x00]+)\x00")
# matches both the camo file's "...Color" fields (myPrimaryColor) and the vanity file's
# "myColorN" fields (myColor1..4): "my" + anything + "Color" + an optional trailing digit/word.
_CAMO_RX_COLOUR = re.compile(rb"(my\w*Color\w*)\x000x([0-9a-fA-F]{8})\x00")
_CAMO_HEX8 = re.compile(r"^[0-9a-fA-F]{8}$")


@dataclass
class CamoColour:
    field: str            # myPrimaryColor / mySecondaryColor / myTertiaryColor
    value: str            # AARRGGBB (lower hex, real alpha from file)
    offset: int           # byte offset of the 8 hex chars in the source file


@dataclass
class CamoPalette:
    name: str
    subtab: str
    colours: list
    region: str = ""
    tier: str = ""
    collection: str = ""

def _camo_wiki_meta(wiki_path: str) -> dict:
    """name -> {subtab, region, tier, collection} from the wiki gearcamo section."""
    with open(wiki_path, encoding="utf-8") as _f:
        w = json.load(_f)
    meta = {}
    for row in w.get("gearcamo", {}).get("items", []):
        f = row.get("fields") or {}
        meta[str(row.get("name") or "").strip()] = {
            "subtab": row.get("type"),
            "region": str(f.get("Region") or ""),
            "tier": str(f.get("Tier") or ""),
            "collection": str(f.get("Collection") or ""),
        }
    return meta


class RejuiceFile:
    """Loads gearcamo_colorpalettes.rejuice, exposes a per-subtab palette model the UI binds
    to, edits colours in memory, and exports by in-place patching (overwrite or copy)."""

    def __init__(self, path: str, wiki_path: str | None = None,
                 block_token: bytes = b"GearCamoColorPalette") -> None:
        self.path = path
        with open(path, "rb") as _f:
            self.data = bytearray(_f.read())
        self._meta = _camo_wiki_meta(wiki_path) if wiki_path else {}
        # block regex built from the token, so the same parser reads the camo rejuice
        # (GearCamoColorPalette) and the vanity gear-colours rejuice (GearColorPalette).
        self._rx_block = re.compile(re.escape(block_token) + rb"\x00([^\x00]+)\x00")
        self.by_subtab: dict = {s: [] for s in CAMO_SUBTABS}
        self._index: dict = {}  # (subtab, name) -> CamoPalette
        self.palettes: list = []  # flat, file order - what the all-visible editor binds to
        self._parse()

    def _parse(self) -> None:
        spans = [(m.group(1).decode("latin-1").lstrip('"').strip(), m.start(), m.end())
                 for m in self._rx_block.finditer(self.data)]
        for i, (name, _s, vstart) in enumerate(spans):
            nxt = spans[i + 1][1] if i + 1 < len(spans) else len(self.data)
            colours = [
                CamoColour(cm.group(1).decode("latin-1"),
                           cm.group(2).decode("latin-1").lower(), cm.start(2))
                for cm in _CAMO_RX_COLOUR.finditer(self.data, vstart, nxt)
            ]
            meta = self._meta.get(name, {})
            sub = meta.get("subtab") or self._guess_subtab(name)
            pal = CamoPalette(name, sub, colours, meta.get("region", ""),
                              meta.get("tier", ""), meta.get("collection", ""))
            self.by_subtab.setdefault(sub, []).append(pal)
            self._index[(sub, name)] = pal
            self.palettes.append(pal)

    @staticmethod
    def _guess_subtab(name: str) -> str:
        n = name.lower()
        if n.startswith("ash"):
            return "Ash"
        if n.startswith("win"):
            return "Windtrader"
        if n.startswith("banshee"):
            return "Banshee Gear"
        if n.startswith("player"):
            return "Player Weapon" if "wpn" in n or "weapon" in n else "Player Gear"
        if "weapon" in n or "wpn" in n:
            return "Weapon (Region)"
        return "Gear (Region)"

    def names(self, subtab: str) -> list:
        return [p.name for p in self.by_subtab.get(subtab, [])]

    def get(self, subtab: str, name: str):
        return self._index.get((subtab, name))

    def total(self) -> int:
        return sum(len(v) for v in self.by_subtab.values())

    def set_colour(self, subtab: str, name: str, idx: int, aarrggbb: str) -> bool:
        v = aarrggbb.strip().lstrip("#").lower()
        if not _CAMO_HEX8.match(v):
            raise ValueError(f"colour must be 8 hex chars (AARRGGBB), got {aarrggbb!r}")
        pal = self.get(subtab, name)
        if not pal or idx >= len(pal.colours):
            return False
        c = pal.colours[idx]
        c.value = v
        self.data[c.offset:c.offset + 8] = v.encode("ascii")  # fixed width -> safe in place
        return True

    def set_value(self, colour, aarrggbb: str) -> bool:
        """Edit one colour in place by its parsed object (used by the flat all-visible editor,
        which holds palette/colour objects directly rather than (subtab, name, idx) keys)."""
        v = aarrggbb.strip().lstrip("#").lower()
        if not _CAMO_HEX8.match(v):
            raise ValueError(f"colour must be 8 hex chars (AARRGGBB), got {aarrggbb!r}")
        colour.value = v
        self.data[colour.offset:colour.offset + 8] = v.encode("ascii")
        return True

    def export(self, dst: str | None = None, overwrite: bool = False) -> str:
        if overwrite:
            target = self.path
        else:
            if not dst:
                raise ValueError("dst required when not overwriting")
            target = dst
        d = os.path.dirname(target)
        if d:
            os.makedirs(d, exist_ok=True)  # reproduce any blue/.../ folders the caller built
        with open(target, "wb") as fh:
            fh.write(self.data)  # self.data is the full file (in-place edits), so this is faithful
        return target

    # preset save/load: snapshot + restore every palette's colours
    def colour_state(self) -> dict:
        return {f"{s}|{p.name}": [c.value for c in p.colours]
                for s, ps in self.by_subtab.items() for p in ps}

    def apply_colour_state(self, state: dict) -> None:
        for key, vals in (state or {}).items():
            sub, _, name = key.partition("|")
            for i, v in enumerate(vals):
                try:
                    self.set_colour(sub, name, i, v)
                except ValueError:
                    pass


# ---- AARRGGBB <-> QColor (the file stores 0xAARRGGBB; we keep the real alpha) ----
def camo_hex_to_qcolor(aarrggbb: str) -> QColor:
    v = aarrggbb.strip().lstrip("#")
    a, r, g, b = (int(v[i:i + 2], 16) for i in (0, 2, 4, 6))
    return QColor(r, g, b, a)


def camo_qcolor_to_hex(c: QColor) -> str:
    return f"{c.alpha():02x}{c.red():02x}{c.green():02x}{c.blue():02x}"


class _CamoColourRow(QWidget):
    """One colour: a clickable swatch + an editable AARRGGBB hex field."""

    def __init__(self, label: str, on_edit) -> None:
        super().__init__()
        self._on_edit = on_edit  # (aarrggbb) -> bool
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        cap = QLabel(label)
        cap.setFixedWidth(70)
        self.swatch = QPushButton()
        self.swatch.setObjectName("swatch")  # match the Na'vi / gear swatch style (2px border, radius)
        self.swatch.setFixedSize(30, 22)
        self.swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.swatch.clicked.connect(self._pick)
        self.edit = QLineEdit()
        self.edit.setMaxLength(8)
        self.edit.setFixedWidth(96)
        self.edit.editingFinished.connect(self._typed)
        row.addWidget(cap)
        row.addWidget(self.swatch)
        row.addWidget(self.edit)
        row.addStretch(1)

    def set_value(self, aarrggbb: str) -> None:
        self.edit.setText(aarrggbb)
        c = camo_hex_to_qcolor(aarrggbb)
        self.swatch.setStyleSheet(
            _swatch_css(f"{c.red():02x}{c.green():02x}{c.blue():02x}")
        )

    def _commit(self, aarrggbb: str) -> None:
        if self._on_edit(aarrggbb):
            self.set_value(aarrggbb)

    def _typed(self) -> None:
        v = self.edit.text().strip().lstrip("#").lower()
        if len(v) == 8 and all(ch in "0123456789abcdef" for ch in v):
            self._commit(v)

    def _pick(self) -> None:
        cur = camo_hex_to_qcolor(self.edit.text() or "ff808080")
        c = QColorDialog.getColor(cur, self, "Pick colour",
                                  QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self._commit(camo_qcolor_to_hex(c))


class _CamoSubtabBlock(QWidget):
    """One camo subtab: a title, a palette dropdown, and the colour picker rows beneath."""

    def __init__(self, subtab: str, on_edit, on_select=None) -> None:
        super().__init__()
        self._subtab = subtab
        self._on_edit = on_edit  # (subtab, name, idx, hex) -> bool
        self._on_pick = on_select  # (subtab, name, [hex,...]) -> None  (active palette)
        self.rf = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 8)
        lay.setSpacing(3)

        title = QLabel(subtab)
        title.setObjectName("sectiontitle")
        lay.addWidget(title)

        self.dropdown = QComboBox()
        self.dropdown.currentTextChanged.connect(self._on_select)
        lay.addWidget(self.dropdown)

        self.rows = [
            _CamoColourRow(name, lambda v, i=i: self._edit(i, v))
            for i, name in enumerate(("Primary", "Secondary", "Tertiary"))
        ]
        for rw in self.rows:
            rw.hide()
            lay.addWidget(rw)

    def populate(self, rf) -> None:
        self.rf = rf
        self.dropdown.blockSignals(True)
        self.dropdown.clear()
        self.dropdown.addItems(rf.names(self._subtab))
        self.dropdown.blockSignals(False)
        self._on_select(self.dropdown.currentText())

    def _on_select(self, name: str) -> None:
        pal = self.rf.get(self._subtab, name) if (self.rf and name) else None
        if not pal:
            for rw in self.rows:
                rw.hide()
            if self._on_pick:
                self._on_pick(self._subtab, name, [])
            return
        for i, rw in enumerate(self.rows):
            if i < len(pal.colours):
                rw.set_value(pal.colours[i].value)
                rw.show()
            else:
                rw.hide()
        if self._on_pick:
            self._on_pick(self._subtab, name, [c.value for c in pal.colours])

    def _edit(self, idx: int, hexv: str) -> bool:
        return self._on_edit(self._subtab, self.dropdown.currentText(), idx, hexv)

    def refresh(self) -> None:
        self._on_select(self.dropdown.currentText())


class _PaletteSwatch(QPushButton):
    """A compact clickable colour swatch. Picking changes only the RGB and keeps the original
    alpha byte - the vanity file stores a shader flag in alpha (mostly 00), and the camo file's
    real alpha is ff, so preserving it on edit protects both. The full AARRGGBB is in the tooltip."""

    def __init__(self, value: str, on_pick) -> None:
        super().__init__()
        self._on_pick = on_pick  # (aarrggbb) -> bool
        self.setObjectName("swatch")  # same 2px-border / radius style as the Na'vi / gear swatches
        self.setFixedSize(34, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._pick)
        self.set_value(value)

    def set_value(self, aarrggbb: str) -> None:
        self._value = aarrggbb
        c = camo_hex_to_qcolor(aarrggbb)
        self.setStyleSheet(_swatch_css(f"{c.red():02x}{c.green():02x}{c.blue():02x}"))
        self.setToolTip(f"#{aarrggbb}  (click to edit)")

    def _pick(self) -> None:
        cur = camo_hex_to_qcolor(self._value)
        c = QColorDialog.getColor(cur, self, "Pick colour")
        if not c.isValid():
            return
        nv = f"{self._value[:2]}{c.red():02x}{c.green():02x}{c.blue():02x}"  # keep the alpha byte
        if self._on_pick(nv):
            self.set_value(nv)


def _wiki_type_map(wiki_path: str, section: str) -> dict:
    """name -> type from a wiki section's items (used to group the camo editor like the wiki)."""
    try:
        with open(wiki_path, encoding="utf-8") as _f:
            w = json.load(_f)
    except Exception:  # noqa: BLE001
        return {}
    return {str(it.get("name") or "").strip(): str(it.get("type") or "")
            for it in (w.get(section, {}) or {}).get("items", [])}


def blue_relpath(src: str) -> str:
    """The path from the 'blue/' game-root component onward, e.g.
    '.../export/blue/game system data/rejuice/x.rejuice' -> 'blue/game system data/rejuice/x.rejuice'.
    Used so a non-overwrite save can rebuild the in-game folder structure under a chosen mod root.
    Falls back to just the basename if there is no 'blue' component."""
    parts = os.path.normpath(src).split(os.sep)
    for i, p in enumerate(parts):
        if p.lower() == "blue":
            return os.path.join(*parts[i:])
    return os.path.basename(src)


# Every replicated export collects under one top-level folder, so the result is a single packable
# mod tree the user can find at a glance. This wrapper is optional (Settings toggle, on by default);
# with it off, exports go straight to <export folder>/blue/...
EXPORT_ROOT = "PandoraPaint"
# Canonical in-game (blue/...) directories per export type. These are HARDCODED rather than derived
# from wherever the user loaded the source - a loaded file (or a renamed copy) may sit anywhere on
# disk, but it always belongs in the same engine folder. Only the filename comes from the source.
BLUE_DIR_REJUICE = "blue/game system data/rejuice"          # gear camo + gear colour .rejuice
BLUE_DIR_BLUEITEM = "blue/game system data/juice/blueitem"  # Na'vi .blueitemtype
BLUE_DIR_IKRAN = "blue/gameplay/vanity/juice"               # Ikran .mcolorpattern (+ controls/data)


def export_dir(blue_dir):
    """The replicated export directory (no filename): [PandoraPaint/]<blue_dir>, where the
    PandoraPaint wrapper is included only when the Settings toggle is on (the default)."""
    import assets
    parts = blue_dir.split("/")
    if assets.export_pandora_folder():
        parts = [EXPORT_ROOT] + parts
    return os.path.join(*parts)


def export_rel(blue_dir, filename):
    """The replicated export path for one file: [PandoraPaint/]<blue_dir>/<filename>."""
    return os.path.join(export_dir(blue_dir), filename)


def _ask_export_folder(parent):
    """One-off folder prompt used when no Export Folder is configured. Returns a folder or ''."""
    return QFileDialog.getExistingDirectory(
        parent, "Choose a folder to export into (no Export Folder is set)")


def resolve_export_path(parent, blue_rel, overwrite, source_path, *,
                        default_ext="", file_filter="All files (*)", title="Export"):
    """Decide where a SINGLE-file export is written, honouring the global Export Folder + the
    'replicate blue folder structure' toggle. Returns (target_path, replicated) where
    ``replicated`` is True whenever the blue/ structure is rebuilt (so the caller shows a success
    message). Returns (None, False) when cancelled.

      overwrite True            -> source_path (write back in place).
      replicate on, folder set  -> <export folder>/<blue_rel>, no dialog.
      replicate on, no folder   -> ask for a folder this time -> <chosen>/<blue_rel>.
      replicate off             -> a Save-as dialog (place the file anywhere; no folders created).

    ``blue_rel`` is the path the file should take under the export folder, e.g.
    ``blue_relpath(source_path)`` for files that already live in a blue/ tree, or a fixed
    ``blue/gameplay/vanity/juice/<name>`` for banshee patterns.
    """
    import assets
    if overwrite:
        return source_path, False
    if assets.export_replicate_blue():
        folder = assets.export_folder().strip()
        if not folder or not os.path.isdir(folder):
            folder = _ask_export_folder(parent)  # not configured -> ask this time
            if not folder:
                return None, False
        return os.path.join(folder, blue_rel), True
    start = source_path or ""
    path, _ = QFileDialog.getSaveFileName(parent, title, start, file_filter)
    if not path:
        return None, False
    if default_ext and not path.lower().endswith(default_ext):
        path += default_ext
    return path, False


def resolve_export_base(parent):
    """Decide the base output FOLDER for an 'export all' operation (many files). Returns
    (base_folder, replicated) or (None, False) when cancelled.

      replicate on, folder set -> the configured Export Folder; files go to base/<blue_rel>.
      replicate on, no folder  -> ask for a folder this time; files go to base/<blue_rel>.
      replicate off            -> a user-picked folder; each file goes flat to base/<basename>.
    """
    import assets
    if assets.export_replicate_blue():
        folder = assets.export_folder().strip()
        if not folder or not os.path.isdir(folder):
            folder = _ask_export_folder(parent)  # not configured -> ask this time
            if not folder:
                return None, False
        return folder, True
    d = QFileDialog.getExistingDirectory(parent, "Choose a folder to export into")
    if not d:
        return None, False
    return d, False


# The vanity gear-colours rejuice isn't in the wiki here, so its palettes are grouped by their name
# prefix - the same Generic / Tier / DLC buckets the Gear Colours wiki tab uses.
VANITY_TYPE_ORDER = ["Generic", "Tier 2", "Tier 3", "Tier 4", "DLC 1", "DLC 2"]


def _vanity_type(name: str) -> str:
    n = name.strip()
    if n.startswith("T2_"):
        return "Tier 2"
    if n.startswith("T3_"):
        return "Tier 3"
    if n.startswith("T4_"):
        return "Tier 4"
    if n.startswith("DLC01"):
        return "DLC 1"
    if n.startswith("DLC02"):
        return "DLC 2"
    return "Generic"


class _PaletteRow(QWidget):
    """One palette as a fixed-width tile: its (elided) name, then a swatch per colour. The width
    holds the widest case (a 4-colour vanity palette: name + 4 swatches) with slack, so Qt never
    has to shrink the fixed swatches to fit - that proportional shrink is what made 4-swatch rows
    misalign against 3-swatch rows. Fixed width also keeps the tiles in clean columns."""

    NAME_W = 150
    WIDTH = NAME_W + 4 * (34 + 6) + 14  # name + four 34px swatches (6px gaps) + slack = 324

    def __init__(self, pal, on_edit) -> None:
        super().__init__()
        self.setFixedWidth(self.WIDTH)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 1, 0, 1)
        h.setSpacing(6)
        name = QLabel()
        name.setFixedWidth(self.NAME_W)
        name.setToolTip(pal.name)
        name.setText(name.fontMetrics().elidedText(
            pal.name, Qt.TextElideMode.ElideRight, self.NAME_W - 4))
        h.addWidget(name)
        for c in pal.colours:
            h.addWidget(_PaletteSwatch(c.value, lambda v, cc=c: on_edit(cc, v)))
        h.addStretch(1)


class _FlowTiles(QWidget):
    """A set of fixed-width palette tiles whose column count is set EXTERNALLY (from the scroll
    viewport width, not this widget's own width). Driving it from the viewport avoids the feedback
    loop where the tiles' own minimum width keeps the content wide and the grid never shrinks."""

    def __init__(self, tiles) -> None:
        super().__init__()
        self._tiles = tiles
        self._ncols = 0
        self._max_cols = 0
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(2)
        self.set_columns(1)

    def set_columns(self, ncols: int) -> None:
        ncols = max(1, int(ncols))
        if ncols == self._ncols:
            return
        self._ncols = ncols
        while self._grid.count():
            self._grid.takeAt(0)
        for c in range(0, self._max_cols + 2):  # clear stretches a wider layout may have set
            self._grid.setColumnStretch(c, 0)
        self._max_cols = max(self._max_cols, ncols)
        for i, w in enumerate(self._tiles):
            self._grid.addWidget(w, i // ncols, i % ncols, Qt.AlignmentFlag.AlignLeft)
        self._grid.setColumnStretch(ncols, 1)  # slack on the right; tiles stay left-packed
        self.updateGeometry()


class _PaletteScroll(QScrollArea):
    """Scroll area that reports its viewport width on resize, so the editor can recompute the tile
    column count from the real available width (the viewport), breaking the min-width feedback loop."""

    def __init__(self, on_width) -> None:
        super().__init__()
        self._on_width = on_width

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._on_width(self.viewport().width())


class PaletteEditor(QWidget):
    """All palettes in a rejuice. A compact Save box (fixed width, with a description) sits at the
    top; below it each TYPE is its own titled section whose tiles reflow across columns as the window
    resizes. The rejuice path comes from Settings. Reads both rejuice formats via RejuiceFile's
    block_token; edits patch the file bytes in place."""

    SAVE_BOX_W = 900          # Export box width (50% wider than the original 600)
    TILE_GAP = 12
    SECTION_PAD_R = 14          # right padding on the scroll content (clear section end)
    SECTION_INNER = 20         # each type section's own left+right group-box margins (10 + 10)

    def __init__(self, block_token: bytes, kind: str = "camo", wiki_path: str | None = None,
                 title: str = "", description: str = "", on_status=None) -> None:
        super().__init__()
        self._block_token = block_token
        self._kind = kind  # 'camo' | 'colors' - for log lines
        self._on_status = on_status
        self._flows = []  # every _FlowTiles, so a resize can re-column them all together
        self.rf = None
        if kind == "camo":
            self._type_order = list(CAMO_SUBTABS)
            tmap = _wiki_type_map(wiki_path, "gearcamo") if wiki_path else {}
            self._classify = lambda n: tmap.get(n, "Other")
            default_desc = (
                "Recolour the gear and weapon camo palettes. Click a swatch to edit, then Export."
            )
        elif kind == "colors":
            self._type_order = list(VANITY_TYPE_ORDER)
            self._classify = _vanity_type
            default_desc = (
                "Recolour the gear and weapon vanity colour palettes. Click a swatch to edit, "
                "then Export."
            )
        else:
            self._type_order = []
            self._classify = lambda n: "All"
            default_desc = ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 12, 4, 4)  # gap above the Save box (under the sub-tabs)
        lay.setSpacing(16)                   # padding beneath the Save box, before the sections

        # ---- compact Export box (above the colours) ----
        savebox = QGroupBox("Export " + (title or "Palettes"))
        savebox.setFixedWidth(self.SAVE_BOX_W)
        sv = QVBoxLayout(savebox)
        sv.setContentsMargins(12, 6, 12, 12)  # small top inside the box (description height is reserved below)
        sv.setSpacing(8)

        # ---- load row: pick a different rejuice; shows the currently-loaded file path ----
        self._loaded_path = ""
        load_row = QHBoxLayout()
        load_row.setSpacing(8)
        _load_label = {"camo": "Camo (.rejuice)", "colors": "Colour (.rejuice)"}.get(
            self._kind, "Load (.rejuice)"
        )
        self.load_btn = QPushButton(_load_label)
        self.load_btn.setObjectName("accent")
        self.load_btn.setToolTip(
            "Load a different .rejuice palette file to edit (defaults to the one set in Settings)."
        )
        self.load_btn.clicked.connect(self._pick_rejuice)
        self.rej_path_edit = QLineEdit()
        self.rej_path_edit.setReadOnly(True)
        self.rej_path_edit.setPlaceholderText("No .rejuice loaded")
        self.rej_path_edit.setToolTip("The .rejuice currently loaded for editing.")
        load_row.addWidget(self.load_btn)
        load_row.addWidget(self.rej_path_edit, 1)
        sv.addLayout(load_row)

        desc = QLabel(description or default_desc)
        desc.setObjectName("legend")
        desc.setWordWrap(True)
        # A word-wrapped QLabel often under-reports its height in a layout (it's sized as if on one
        # line), which is what clipped the text. Reserve the real wrapped height at the box's content
        # width so the box always grows to fit it.
        _dw = self.SAVE_BOX_W - 2 * 12 - 6
        _dh = desc.fontMetrics().boundingRect(
            0, 0, _dw, 0, Qt.TextFlag.TextWordWrap, desc.text()).height()
        desc.setMinimumHeight(_dh + 6)
        sv.addWidget(desc)
        row = QHBoxLayout()
        row.setSpacing(8)
        self._count = QLabel("Not loaded")
        self._count.setObjectName("legend")
        row.addWidget(self._count)
        row.addStretch(1)
        # right side: the Overwrite tickbox sits directly above the Export button
        right = QVBoxLayout()
        right.setSpacing(4)
        self.overwrite_cb = QCheckBox("Overwrite .rejuice")
        self.overwrite_cb.setChecked(False)  # default: export a copy, never touch the original
        self.overwrite_cb.setToolTip(
            "Ticked: Export overwrites the original .rejuice in place.\n"
            "Unticked: Export writes a copy into a chosen mod root, rebuilding the blue/... folders.")
        # Label the export action for what it writes: camo palettes vs vanity gear colours.
        _export_label = {"camo": "Export Camos", "colors": "Export Colours"}.get(
            self._kind, "Export"
        )
        self.export_btn = QPushButton(_export_label)
        self.export_btn.setObjectName("accent")
        self._export_tip = (
            "Export the edited palettes - overwriting the original or writing a blue/... copy, "
            "per the tickbox above.")
        self.export_btn.setToolTip(self._export_tip)
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setEnabled(False)  # nothing to export until a .rejuice is loaded
        right.addWidget(self.overwrite_cb, 0, Qt.AlignmentFlag.AlignRight)
        right.addWidget(self.export_btn, 0, Qt.AlignmentFlag.AlignRight)
        row.addLayout(right)
        sv.addLayout(row)
        lay.addWidget(savebox, 0, Qt.AlignmentFlag.AlignLeft)

        # ---- scrolling stack of per-type sections ----
        self._content = QWidget()
        self._content_lay = QVBoxLayout(self._content)
        # right padding so it's clear where each type section ends (left stays flush with the Save box)
        self._content_lay.setContentsMargins(0, 0, self.SECTION_PAD_R, 0)
        self._content_lay.setSpacing(8)
        self._content_lay.addStretch(1)
        self._scroll = _PaletteScroll(self._reflow)
        self._scroll.setWidget(self._content)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lay.addWidget(self._scroll, 1)

        self.setEnabled(False)

    def _reflow(self, avail: int) -> None:
        """Set the column count for every type section from the viewport width, minus the right
        page padding and each section's own margins, so a full row of tiles always fits."""
        usable = max(0, int(avail) - self.SECTION_PAD_R - self.SECTION_INNER)
        ncols = max(1, (usable + self.TILE_GAP) // (_PaletteRow.WIDTH + self.TILE_GAP))
        for f in self._flows:
            f.set_columns(ncols)

    def load(self, path: str) -> bool:
        try:
            self.rf = RejuiceFile(path, None, block_token=self._block_token)
        except Exception as e:  # noqa: BLE001
            log.exception("%s rejuice load failed: %s", self._kind, path)
            self._status(f"Failed to load: {e}")
            return False
        # clear any previous sections (keep the trailing stretch)
        while self._content_lay.count() > 1:
            it = self._content_lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        self._flows = []
        for tname, pals in self._group(self.rf.palettes):
            sec = QGroupBox(tname)
            sl = QVBoxLayout(sec)
            sl.setContentsMargins(10, 4, 10, 8)
            sl.setSpacing(2)
            flow = _FlowTiles([_PaletteRow(p, self._edit) for p in pals])
            self._flows.append(flow)
            sl.addWidget(flow)
            self._content_lay.insertWidget(self._content_lay.count() - 1, sec)
        self._reflow(self._scroll.viewport().width())  # column count for the current width
        self.setEnabled(True)
        self.export_btn.setEnabled(True)  # a rejuice is now loaded
        self.export_btn.setToolTip(self._export_tip)
        self._count.setText(
            f"{len(self.rf.palettes)} palettes \u2014 {os.path.basename(path)}")
        self._status(f"Loaded {len(self.rf.palettes)} palettes")
        self._loaded_path = path
        if hasattr(self, "rej_path_edit"):
            self.rej_path_edit.setText(path)
            self.rej_path_edit.setToolTip(path)
        log.info("%s rejuice loaded: %d palettes from %s",
                 self._kind, len(self.rf.palettes), path)
        return True

    def _pick_rejuice(self):
        """Let the user load a different .rejuice into this editor. On success the new path is shown
        and persisted to config so it sticks and matches the Settings path."""
        start = (
            os.path.dirname(self._loaded_path)
            if getattr(self, "_loaded_path", "")
            else ""
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a .rejuice palette file", start,
            "Rejuice (*.rejuice);;All files (*)")
        if not path:
            return
        if self.load(path):
            key = {"camo": "gearcamo_rejuice", "colors": "gearcolors_rejuice"}.get(self._kind)
            if key:
                try:
                    assets.update_config(
                        lambda cfg: cfg.setdefault("paths", {}).__setitem__(key, path)
                    )
                except Exception:  # noqa: BLE001
                    log.exception("failed to persist %s rejuice path", self._kind)

    def _group(self, palettes):
        """Bucket palettes by type, ordered as the wiki orders them; unknown types come last."""
        buckets: dict = {}
        for p in palettes:
            buckets.setdefault(self._classify(p.name), []).append(p)
        order = [t for t in self._type_order if t in buckets]
        order += [t for t in buckets if t not in self._type_order]  # e.g. "Other"
        return [(t, buckets[t]) for t in order]

    def _edit(self, colour, hexv: str) -> bool:
        if not self.rf:
            return False
        try:
            return self.rf.set_value(colour, hexv)
        except ValueError as e:
            self._status(str(e))
            return False

    def _export(self) -> None:
        if not self.rf:
            return
        overwrite = self.overwrite_cb.isChecked()
        if overwrite:
            target = self.rf.path
        else:
            # Always let the user pick the destination folder (starting at the configured Export
            # Folder, if any). With "replicate blue structure" on, the file lands under a rebuilt
            # blue/... tree inside the chosen folder; otherwise it's written flat into it.
            start = (assets.export_folder() or "").strip()
            folder = QFileDialog.getExistingDirectory(
                self, "Choose a folder to export the .rejuice into", start)
            if not folder:
                return
            base = os.path.basename(self.rf.path)
            if assets.export_replicate_blue():
                target = os.path.join(folder, export_rel(BLUE_DIR_REJUICE, base))
            else:
                target = os.path.join(folder, base)
            d = os.path.dirname(target)
            if d:
                try:
                    os.makedirs(d, exist_ok=True)
                except Exception:  # noqa: BLE001
                    pass
        try:
            self.rf.export(dst=(None if overwrite else target), overwrite=overwrite)
        except Exception as e:  # noqa: BLE001
            log.exception("%s export failed", self._kind)
            self._status(f"Export failed: {e}")
            return
        log.info("%s export -> %s (overwrite=%s)", self._kind, target, overwrite)
        if overwrite:
            self._status(f"Overwrote {os.path.basename(self.rf.path)}")
        else:
            QMessageBox.information(
                self, "Export successful",
                f"Wrote:\n  {os.path.basename(target)}\n\nInto:\n  {os.path.dirname(target)}")
            self._status(f"Exported {os.path.basename(target)}")

    def _status(self, msg: str) -> None:
        if self._on_status:
            self._on_status(msg)


class CamoColoursSection(QWidget):
    """rejuice picker, then one dropdown per camo subtab; each shows the selected palette's
    colour picker beneath."""

    def __init__(self, wiki_path: str, on_status=None, on_palette=None) -> None:
        super().__init__()
        self._wiki = wiki_path
        self._on_status = on_status
        self._on_palette = on_palette  # (subtab, name, [hex,...]) -> None
        self.rf = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(6)

        pick = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("gearcamo_colorpalettes.rejuice \u2026")
        browse = QPushButton("Browse\u2026")
        browse.clicked.connect(self._browse)
        pick.addWidget(self.path_edit, 1)
        pick.addWidget(browse)
        lay.addLayout(pick)

        self.blocks = []
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        for i, sub in enumerate(CAMO_SUBTABS):  # two columns to use the widened space
            blk = _CamoSubtabBlock(sub, self._edit, on_select=self._palette_picked)
            self.blocks.append(blk)
            grid.addWidget(blk, i // 2, i % 2, Qt.AlignmentFlag.AlignTop)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

        self.setEnabled(False)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select gearcamo_colorpalettes.rejuice", "",
            "Rejuice (*.rejuice);;All files (*)")
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        try:
            self.rf = RejuiceFile(path, self._wiki)
        except Exception as e:  # noqa: BLE001
            self._status(f"Failed to load: {e}")
            return
        self.path_edit.setText(path)
        self.setEnabled(True)
        for blk in self.blocks:
            blk.populate(self.rf)
        self._status(f"Loaded {self.rf.total()} palettes")

    def _edit(self, subtab: str, name: str, idx: int, aarrggbb: str) -> bool:
        if not self.rf:
            return False
        try:
            ok = self.rf.set_colour(subtab, name, idx, aarrggbb)
        except ValueError as e:
            self._status(str(e))
            return False
        if ok and self._on_palette:  # editing a colour changes the active palette's colours
            pal = self.rf.get(subtab, name)
            if pal:
                self._on_palette(subtab, name, [c.value for c in pal.colours])
        return ok

    def _palette_picked(self, subtab: str, name: str, hexes: list) -> None:
        if self._on_palette:
            self._on_palette(subtab, name, hexes)

    def colour_state(self) -> dict:
        return self.rf.colour_state() if self.rf else {}

    def apply_colour_state(self, state: dict) -> None:
        if self.rf:
            self.rf.apply_colour_state(state)
            for blk in self.blocks:
                blk.refresh()

    def _status(self, msg: str) -> None:
        if self._on_status:
            self._on_status(msg)


class CamoExportSection(QWidget):
    """Overwrite tickbox + Export button -> RejuiceFile.export()."""

    def __init__(self, get_rejuice, on_status=None) -> None:
        super().__init__()
        self._get = get_rejuice
        self._on_status = on_status
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.overwrite = QCheckBox("Overwrite original .rejuice")
        btn = QPushButton("Export Camos")
        btn.setObjectName("accent")
        btn.clicked.connect(self._export)
        lay.addWidget(self.overwrite)
        lay.addWidget(btn)

    def _export(self) -> None:
        rf = self._get()
        if not rf:
            self._status("Load a .rejuice first")
            return
        try:
            if self.overwrite.isChecked():
                out = rf.export(overwrite=True)
            else:
                dst, _ = QFileDialog.getSaveFileName(
                    self, "Export camos to\u2026", "gearcamo_colorpalettes.rejuice",
                    "Rejuice (*.rejuice);;All files (*)")
                if not dst:
                    return
                out = rf.export(dst, overwrite=False)
            self._status(f"Exported \u2192 {out}")
        except Exception as e:  # noqa: BLE001
            self._status(f"Export failed: {e}")

    def _status(self, msg: str) -> None:
        if self._on_status:
            self._on_status(msg)


class CamoAssetsPanel(_AssetFolderPanel):
    """Inline asset manager for the two Camo-tab data files: the gear/weapon CAMO colour palettes
    (gearcamo_colorpalettes.rejuice) and the gear/weapon COLOUR palettes (gearcolorpalettes_vanity.
    rejuice). Point it at an asset folder to auto-detect both, or pick each file individually.
    Paths are referenced, never copied; every change writes the config and fires on_changed. Neither
    file is mandatory - the Camo tab is enabled when EITHER resolves, and each sub-tab loads only if
    its own rejuice is present."""

    REJUICE_NAME = "gearcamo_colorpalettes.rejuice"
    COLORS_NAME = "gearcolorpalettes_vanity.rejuice"

    def __init__(self, on_changed=None, on_reset=None, parent=None):
        super().__init__(parent)
        self.on_changed = on_changed
        self.on_reset = on_reset
        self._load_state()

        v = QVBoxLayout(self)
        v.setContentsMargins(2, 6, 2, 2)
        v.setSpacing(8)

        # ---- asset folder picker ----
        self._add_asset_folder_row(
            v,
            "Choose your extracted Avatar asset folder. Both rejuice files are found in it "
            "automatically.",
        )

        # ---- resync + manage row ----
        fr = QHBoxLayout()
        resync_btn = QPushButton("Resync Assets")
        resync_btn.setObjectName("action")
        resync_btn.setToolTip("Re-scan the asset folder above for both rejuice files.")
        resync_btn.clicked.connect(self._resync)
        fr.addWidget(resync_btn)
        clear_btn = QPushButton("Clear manual picks")
        clear_btn.setStyleSheet(_warning_css())
        clear_btn.setToolTip("Drop the per-file overrides and fall back to folder auto-detect.")
        clear_btn.clicked.connect(self._clear_override)
        fr.addWidget(clear_btn)
        if self.on_reset:
            reset_btn = QPushButton("Reset Camo assets\u2026")
            reset_btn.setToolTip(
                "Forget the asset folder and both rejuice overrides. Your files are not deleted."
            )
            reset_btn.setStyleSheet(_danger_css())
            reset_btn.setFixedHeight(clear_btn.sizeHint().height())
            reset_btn.clicked.connect(self.on_reset)
            fr.addWidget(reset_btn)
        fr.addStretch(1)
        v.addLayout(fr)

        # ---- description ----
        intro = QLabel(
            "Point this at your own extracted Avatar: Frontiers of Pandora assets - nothing is "
            "copied, the tool just remembers where the files are. Set an asset folder to "
            "auto-detect both rejuice files, or set each individually. Neither is mandatory: the "
            "Camo tab is enabled when either is found, and each sub-tab loads only if its own "
            "rejuice is present."
        )
        intro.setObjectName("legend")
        intro.setWordWrap(True)
        v.addWidget(intro)

        # ---- the two rejuice files, in a collapsible section (collapsed by default) ----
        self.row = SlotRow(
            "rejuice", "Gear/Weapon Camo palettes (gearcamo_colorpalettes.rejuice)", "",
            "optional", self._pick_one, game=""
        )
        self.colors_row = SlotRow(
            "gearcolors", "Gear/Weapon Colour palettes (gearcolorpalettes_vanity.rejuice)", "",
            "optional", self._pick_colors, game=""
        )
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(10, 0, 10, 4)
        bl.setSpacing(2)
        bl.addWidget(self.row)
        bl.addWidget(self.colors_row)
        self.section = CollapsibleSection("Gear Camo and Colours", body, expanded=False)
        v.addWidget(self.section)
        v.addStretch(1)
        self._refresh()

    # ---- state ----
    def _load_state(self):
        self.cfg = assets.load_config(mutable=True)
        camo = self.cfg.get("camo", {})
        self.folder = camo.get("folder", "") or ""
        self.detected = camo.get("detected", "") or ""
        self.override = camo.get("override", "") or ""
        # the gear/weapon COLOUR (vanity dye) palettes - the second rejuice. Auto-detected from the
        # asset folder or picked individually; resolved path is written to paths.gearcolors_rejuice.
        self.colors_detected = camo.get("colors_detected", "") or ""
        self.colors_override = camo.get("colors_override", "") or ""

    def reload(self):
        self._load_state()
        self._refresh()

    def _resolved(self):
        if self.override and os.path.isfile(self.override):
            return self.override
        if self.detected and os.path.isfile(self.detected):
            return self.detected
        return ""

    def _resolved_colors(self):
        if self.colors_override and os.path.isfile(self.colors_override):
            return self.colors_override
        if self.colors_detected and os.path.isfile(self.colors_detected):
            return self.colors_detected
        return ""

    def _commit(self):
        def _upd(cfg):
            camo = cfg.setdefault("camo", {})
            camo["folder"] = self.folder
            camo["detected"] = self.detected
            camo["override"] = self.override
            camo["colors_detected"] = self.colors_detected
            camo["colors_override"] = self.colors_override
            # drop the retired w_camo mask paths/keys if an older config still carries them
            for k in ("mask_detected", "mask_override", "tiger_detected", "tiger_override"):
                camo.pop(k, None)
            paths = cfg.setdefault("paths", {})
            resolved = self._resolved()
            if resolved:
                paths["gearcamo_rejuice"] = resolved
            else:
                paths.pop("gearcamo_rejuice", None)
            colors = self._resolved_colors()
            if colors:
                paths["gearcolors_rejuice"] = colors
            else:
                paths.pop("gearcolors_rejuice", None)
            paths.pop("camo_mask", None)
            paths.pop("camo_tiger", None)

        self.cfg = assets.update_config(_upd)
        self._refresh()
        if self.on_changed:
            self.on_changed()

    def _refresh(self):
        if hasattr(self, "export_edit"):
            self.export_edit.setText(self.folder)
            self.export_edit.setToolTip(self.folder or "No asset folder set")
        resolved = self._resolved()
        manual = bool(resolved) and resolved == self.override
        self.row.update_state(resolved or self.override or self.detected, override=manual)
        resolved_colors = self._resolved_colors()
        manual_colors = bool(resolved_colors) and resolved_colors == self.colors_override
        self.colors_row.update_state(
            resolved_colors or self.colors_override or self.colors_detected, override=manual_colors
        )

    # ---- actions ----
    def _set_export_folder(self):
        # Same shared "set asset folder" flow as the Ikran + Na'vi panels (pick dir -> set folder ->
        # reflect it -> _resync, which for this panel scans for the rejuice files + commits).
        choose_export_folder(self, "Choose your extracted asset folder")

    def _scan(self):
        found = ""
        colors = ""
        if self.folder and os.path.isdir(self.folder):
            for root, _dirs, files in os.walk(self.folder):
                lower = {f.lower(): f for f in files}
                if not found and self.REJUICE_NAME in lower:
                    found = os.path.join(root, lower[self.REJUICE_NAME])
                if not colors and self.COLORS_NAME in lower:
                    colors = os.path.join(root, lower[self.COLORS_NAME])
                if found and colors:
                    break
        self.detected = found
        self.colors_detected = colors

    def _resync(self):
        if self.folder:
            self._scan()
        self._commit()

    def _pick_one(self, _slot):
        start = self.override or self.folder or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select gearcamo_colorpalettes.rejuice", start,
            "Rejuice (*.rejuice);;All files (*)"
        )
        if not path:
            return
        self.override = path
        self._commit()

    def _pick_colors(self, _slot):
        start = self.colors_override or self.folder or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select gearcolorpalettes_vanity.rejuice", start,
            "Rejuice (*.rejuice);;All files (*)"
        )
        if not path:
            return
        self.colors_override = path
        self._commit()

    def _clear_override(self):
        self.override = ""
        self.colors_override = ""
        self._commit()


class ThemeColorsPanel(QWidget):
    """Settings row: the two UI accent colours on one line, with the description beneath.

    The ACTIVE accent is the bright colour (selected tab, group titles, focus, swatch hover); the
    INACTIVE accent is its dim counterpart (unselected secondary-tab underline, side-tab line/icon).
    Setting the active accent always re-derives the inactive (same hue, dimmed); the inactive swatch
    overrides it afterwards. on_changed fires after any edit so the host can re-apply live."""

    def __init__(self, on_changed=None, parent=None):
        super().__init__(parent)
        self.on_changed = on_changed
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Active accent"))
        self.active_swatch = self._mk_swatch("Pick the active (bright) accent colour.", self._pick_active)
        self.active_hex = QLabel()
        self.active_hex.setObjectName("gamepath")
        row.addWidget(self.active_swatch)
        row.addWidget(self.active_hex)
        row.addSpacing(14)
        row.addWidget(QLabel("Inactive accent"))
        self.inactive_swatch = self._mk_swatch("Pick the inactive (dim) accent colour.", self._pick_inactive)
        self.inactive_hex = QLabel()
        self.inactive_hex.setObjectName("gamepath")
        row.addWidget(self.inactive_swatch)
        row.addWidget(self.inactive_hex)
        row.addSpacing(14)
        reset = QPushButton("Reset colours")
        reset.setStyleSheet(_warning_css())
        reset.setToolTip("Return the active and inactive accents to their defaults.")
        reset.clicked.connect(self._reset)
        row.addWidget(reset)
        row.addStretch(1)
        v.addLayout(row)

        desc = QLabel(
            "The interface accent. Active is the bright colour (selected tab, group titles, focus); "
            "inactive is its dim version (unselected secondary-tab underline, side-tab line and "
            "icons). Setting the active accent re-derives the inactive automatically (same hue, "
            "dimmed) - change the inactive swatch afterwards to override it. Reset returns both to "
            "the defaults."
        )
        desc.setObjectName("legend")
        desc.setWordWrap(True)
        v.addWidget(desc)
        self._refresh()

    def _mk_swatch(self, tip, slot):
        b = QPushButton()
        b.setObjectName("swatch")
        b.setFixedSize(40, 22)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setToolTip(tip)
        b.clicked.connect(slot)
        return b

    def _swatch_style(self, hexv):
        return (
            "QPushButton#swatch{background:%s;border:2px solid #2A2F3A;border-radius:5px;}"
            "QPushButton#swatch:hover{border:2px solid %s;}" % (hexv, theme.accent_active())
        )

    def _refresh(self):
        a, i = theme.accent_active(), theme.accent_inactive()
        self.active_swatch.setStyleSheet(self._swatch_style(a))
        self.active_hex.setText(a)
        self.inactive_swatch.setStyleSheet(self._swatch_style(i))
        self.inactive_hex.setText(i)

    def _pick(self, current):
        col = QColorDialog.getColor(QColor(current), self, "Choose accent colour")
        return col.name().upper() if col.isValid() else None

    def _pick_active(self):
        hx = self._pick(theme.accent_active())
        if hx:
            theme.set_active(hx)  # also re-derives the inactive
            self._changed()

    def _pick_inactive(self):
        hx = self._pick(theme.accent_inactive())
        if hx:
            theme.set_inactive(hx)
            self._changed()

    def _reset(self):
        theme.reset()
        self._changed()

    def _changed(self):
        self._refresh()
        if self.on_changed:
            self.on_changed()
