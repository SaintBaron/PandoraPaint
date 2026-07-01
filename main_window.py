"""Pandora Paint - the MainWindow shell (tabs, viewer wiring, load/export orchestration).

Split out of the original app.py. Builds on the widgets in widgets.py; the entry point is
app.py."""

from __future__ import annotations
import os
import numpy as np

from PyQt6.QtCore import (
    Qt,
    QObject,
    QRunnable,
    QThreadPool,
    pyqtSignal,
    QUrl,
    QByteArray,
    QTimer,
    QEvent,
    QEventLoop,
)
from PyQt6.QtGui import QDesktopServices
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
    QMainWindow,
    QMessageBox,
    QSplitter,
    QScrollArea,
    QFrame,
    QCheckBox,
    QComboBox,
    QTabWidget,
    QProgressDialog,
    QSizeGrip,
    QSizePolicy,
)

import assets
import logging
import theme
from patterns import ColorPattern, PatternControl, BansheePatternData

log = logging.getLogger("pandorapaint.main")

from widgets import (
    PatternPanel,
    NaviControls,
    NaviAssetsPanel,
    AssetsPanel,
    SaveLoadBar,
    _TexDecodeSignals,
    _TexDecodeTask,
    _pil,
    load_rgba,
    GearControls,
    save_texture,
    MainNav,
    TitleBar,
    CamoAssetsPanel,
    PaletteEditor,
    ThemeColorsPanel,
    resolve_export_base,
    export_dir,
    BLUE_DIR_IKRAN,
    ask_export_format,
)


class _LazyTab(QWidget):
    """Builds its content the first time the tab is shown, keeping that work off the startup
    path. Used for tabs that aren't the landing tab and don't share the 3-D viewer."""

    def __init__(self, builder):
        super().__init__()
        self._builder = builder
        self._built = False
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)

    def showEvent(self, ev):
        if not self._built:
            self._built = True
            self._lay.addWidget(self._builder())
        super().showEvent(ev)


# Anti-aliasing + texture-filtering dropdown options, shared by the Settings tab and the first-run
# setup dialog so the two never drift. AA values are mode strings (see MainWindow._aa_params).
_AA_OPTIONS = [
    ("Off (Performance)", "off"),
    ("FXAA (Performance)", "fxaa"),
    ("1.5x - SSAA (Balanced)", "ssaa1.5"),
    ("2x - SSAA (Balanced)", "ssaa2"),
    ("3x - SSAA (Quality)", "ssaa3"),
    ("4x - SSAA (Quality)", "ssaa4"),
]
_AF_OPTIONS = [
    ("Off (Performance)", 1),
    ("2x (Performance)", 2),
    ("4x (Balanced)", 4),
    ("8x (Quality)", 8),
    ("16x (Quality)", 16),
]
# First-run quick-setup toggles: (label, default, settings-tab widget attr, save handler attr, desc).
# The dialog reuses each pref's real handler so a choice writes config + applies live exactly as the
# Settings tab would. Defaults here mirror the Settings tab defaults.
_FIRST_RUN_BOOLS = [
    ("Enable diagnostic logging", False, "pref_logging", "_save_pref_logging",
     "Write activity and errors to a log file so you can share it when reporting a problem. Off by default."),
    ("Replicate the blue folder structure on export", True, "pref_replicate_blue", "_save_pref_replicate_blue",
     "Rebuild each export's blue/\u2026 path inside your Export Folder automatically, with no save dialog. Off: choose a location per export."),
    ("Wrap exports in a top-level PandoraPaint folder", False, "pref_pandora_folder", "_save_pref_pandora_folder",
     "Collect the rebuilt blue/\u2026 tree inside one top-level PandoraPaint folder, so exports gather in one packable place."),
    ("Remember model and texture changes", False, "pref_remember_changes", "_save_pref_remember_changes",
     "Model and texture changes in the Na'vi and Ikran tabs replace the defaults and persist between sessions. Off: they last only this session."),
    ("Open file dialogs at the nearest relevant folder", True, "pref_load_existing", "_save_pref_load_existing",
     "Open each file dialog at the most relevant nearby folder instead of always at the asset root."),
    ("Remember window size and position", True, "pref_remember_geom", "_save_pref_geometry",
     "Save the window's size and position on close and restore them on the next launch."),
    ("Specular highlights in the 3-D preview", True, "pref_specular", "_save_pref_specular",
     "Show the skin clear-coat and hair sheen in the 3-D preview. Off: a flatter, matte look."),
    ("3-D viewer pane on the left", True, "pref_viewer_left", "_save_pref_viewer_side",
     "Put the 3-D viewer on the left and the controls on the right. Off: viewer on the right."),
]
_AA_DESC = (
    "FXAA is a cheap edge smoother; SSAA renders the preview larger and downsamples it - the only "
    "mode that also fixes hair-strand pixelation - at more GPU the higher you go."
)
_AF_DESC = (
    "Sharpen textures at grazing angles (skin, gear, membrane) and cut their shimmer, for a small "
    "GPU cost."
)


class MainWindow(QMainWindow):
    _SUBTAB_TOP_GAP = 12  # top gap under the main tab row (tab-page top margin)
    # Drop the first side-tab so its TOP edge lines up with the top of the viewer legend. The legend
    # sits at: tab-page top (_SUBTAB_TOP_GAP) + rcol top margin (8) + model row (30) + rcol spacing
    # (8) below the content top, so the extra drop past _SUBTAB_TOP_GAP is 8 + 30 + 8 = 46. Tunable.
    _SIDETAB_EXTRA_DROP = 46

    def __init__(self):
        super().__init__()
        # background texture decoding: CPU/Pillow work runs on a thread pool and the
        # finished RGBA array is marshalled back here for the GL upload. OpenGL stays
        # on the main thread (the context has thread affinity).
        self._pool = QThreadPool.globalInstance()
        self._tex_signals = _TexDecodeSignals(self)
        self._tex_signals.done.connect(self._on_texture_decoded)
        self._tex_gen = 0
        self._tex_latest = {}  # (key, role) -> newest generation requested
        self.setWindowTitle("Pandora Paint  -  AFoP Skin Recolour")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.resize(1440, 920)
        # never open larger than the screen, or the right panel runs off-screen
        # (small displays / VMs). Stays freely resizable afterwards.
        _scr = QApplication.primaryScreen()
        if _scr is not None:
            _av = _scr.availableGeometry()
            self.resize(min(1440, _av.width() - 40), min(920, _av.height() - 80))
        self.setAcceptDrops(True)
        from viewer import BansheeViewer
        from app import app_icon  # icon now lives in the entry point (app.py)

        self.setWindowIcon(app_icon())
        self.viewer = BansheeViewer()

        self.head = PatternPanel(
            "Head", "head", self._palette_changed, self.open_texture
        )
        self.body = PatternPanel(
            "Body", "body", self._palette_changed, self.open_texture
        )
        self._coat_engine = {"body": "", "head": ""}  # engine paths of set-loaded coats
        self._pset_path = None  # path of a loaded .mbansheepatterndata (None if none)
        self._pset_data = None  # the loaded BansheePatternData (for in-place overwrite)

        controls = QWidget()
        col = QVBoxLayout(controls)
        col.setContentsMargins(8, 8, 8, 8)
        col.setSpacing(8)

        # ---- Save / Load bar: built here, but pinned ABOVE the scroll region (added to the
        #      left column further down) so it stays put at the top while the controls scroll. ----
        self.banshee_io = SaveLoadBar(
            "Save / Load Ikran",
            "banshee",
            self._banshee_preset_collect,
            self._banshee_preset_apply,
            self._banshee_reset_defaults,
        )

        # ---- pattern-set loader, above the Head/Body panels ----
        pdbox = QGroupBox("Load Banshee Pattern Data")
        pdl = QVBoxLayout(pdbox)
        pdl.setContentsMargins(10, 6, 10, 8)
        pdl.setSpacing(5)
        pdrow = QHBoxLayout()
        pdrow.setSpacing(6)
        pdbtn = QPushButton("Browse...")
        pdbtn.setFixedWidth(86)
        pdbtn.setToolTip(
            "Load a .mbansheepatterndata - applies its body/head colours, "
            "controls and pattern coats at once"
        )
        pdbtn.clicked.connect(self._load_pattern_set)
        self.pset_edit = QLineEdit()
        self.pset_edit.setPlaceholderText(".mbansheepatterndata path (optional)")
        self.pset_edit.setToolTip(
            "Path to a .mbansheepatterndata - type a path and press Enter"
        )
        self.pset_edit.returnPressed.connect(self._load_pattern_set_entered)
        pdrow.addWidget(pdbtn)
        pdrow.addWidget(self.pset_edit, 1)
        pdl.addLayout(pdrow)
        hint = QLabel(
            "Optional: load an existing .mbansheepatterndata here to auto-fill every "
            "field below. The colour patterns, controls and coat .dds it references are "
            "found automatically alongside it (same folder or a sub-folder, blue/\u2026 "
            "layout). Otherwise, fill each section in by hand."
        )
        hint.setObjectName("legend")
        hint.setWordWrap(True)
        pdl.addWidget(hint)
        col.addWidget(pdbox)
        col.addSpacing(12)

        panels = QHBoxLayout()
        panels.setSpacing(8)
        panels.addWidget(self.head)
        arrows = QVBoxLayout()
        arrows.setSpacing(8)
        arrows.addStretch(1)
        to_body = QPushButton("\u2192")
        to_body.setObjectName("arrow")
        to_body.setFixedSize(38, 38)
        to_body.setToolTip("Copy Head colours into Body")
        to_body.clicked.connect(lambda: self._copy_from_other("body"))
        to_head = QPushButton("\u2190")
        to_head.setObjectName("arrow")
        to_head.setFixedSize(38, 38)
        to_head.setToolTip("Copy Body colours into Head")
        to_head.clicked.connect(lambda: self._copy_from_other("head"))
        arrows.addWidget(to_body)
        arrows.addWidget(to_head)
        arrows.addStretch(1)
        panels.addLayout(arrows)
        panels.addWidget(self.body)
        col.addLayout(panels, 1)
        col.addSpacing(12)
        expbox = QGroupBox("Export")
        ebl = QVBoxLayout(expbox)
        ebl.setContentsMargins(10, 8, 10, 8)
        ebl.setSpacing(7)
        hint = QLabel(
            "Non-overwritten files are written to the Export Folder set in Settings "
            "(a blue/gameplay/vanity/juice tree is built inside it when 'Replicate the "
            "blue folder structure' is on)."
        )
        hint.setObjectName("legend")
        hint.setWordWrap(True)
        ebl.addWidget(hint)

        self._ov_guard = False
        self.ov_master = QCheckBox("Overwrite existing pattern files")
        self.ov_master.setToolTip(
            "Save over the loaded files in place (colour patterns, "
            "controls and Banshee Pattern Data). Ticks and locks the "
            "per-panel overwrites and both export options below, and "
            "disables the folder name. Needs both colour patterns loaded."
        )
        self.ov_master.toggled.connect(self._on_master_overwrite)
        ebl.addWidget(self.ov_master)

        self.exp_ctrl_cb = QCheckBox("Export Pattern Control (.mpatterncontrol) files")
        self.exp_ctrl_cb.setToolTip(
            "Also write the Body and Head pattern controls with the "
            "current Level/Invert values (names, uids and file names "
            "preserved). Needs a control loaded in both panels."
        )
        ebl.addWidget(self.exp_ctrl_cb)

        self.exp_pd_cb = QCheckBox("Export Banshee Pattern Data (.mbansheepatterndata)")
        self.exp_pd_cb.setToolTip(
            "Also write the loaded Banshee Pattern Data, preserving its "
            "name/uids/references and updating only the coat paths. "
            "Needs one loaded."
        )
        ebl.addWidget(self.exp_pd_cb)

        export = QPushButton("Export All Patterns")
        export.setObjectName("accent")
        export.setFixedHeight(32)
        export.setToolTip("Save both colour patterns (plus any ticked extras above)")
        export.clicked.connect(self.export_all)
        ebl.addWidget(export)

        export_tex = QPushButton("Export All as Texture")
        export_tex.setObjectName("accent")
        export_tex.setFixedHeight(32)
        export_tex.setToolTip(
            "Bake a recoloured texture for every panel whose Export as Texture is available, "
            "into one chosen folder")
        export_tex.clicked.connect(self._export_all_textures)
        ebl.addWidget(export_tex)
        col.addWidget(expbox)

        self.export_all_btn = export
        self.export_all_tex_btn = export_tex
        self._export_all_tip = export.toolTip()
        self._export_all_tex_tip = export_tex.toolTip()
        self.body.overwrite.toggled.connect(self._on_child_overwrite)
        self.head.overwrite.toggled.connect(self._on_child_overwrite)
        self.body.on_validity_change = self._refresh_export_all
        self.head.on_validity_change = self._refresh_export_all
        self._refresh_export_all()

        # right pane: model path bar + legend (top) | viewer | animation picker (bottom)
        right = QWidget()
        rcol = QVBoxLayout(right)
        rcol.setContentsMargins(8, 8, 8, 8)
        rcol.setSpacing(8)
        modelrow = QHBoxLayout()
        modelrow.setSpacing(8)
        mlbl = QLabel("Model")
        mlbl.setObjectName("subtitle")
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setFixedHeight(30)
        self.model_path_edit.setPlaceholderText("path to a .mmb model")
        self.model_path_edit.setToolTip(
            "Path to the displayed model - type a path and press Enter to load"
        )
        self.model_path_edit.returnPressed.connect(self._model_path_entered)
        browse = QPushButton("Browse...")
        browse.setFixedHeight(30)
        browse.setToolTip("Browse for a model file (.mmb)")
        browse.clicked.connect(self._browse_model)
        modelrow.addWidget(mlbl)
        modelrow.addWidget(self.model_path_edit, 1)
        modelrow.addWidget(browse)
        rcol.addLayout(modelrow)
        legend = QLabel("Orbit   LMB        Pan   MMB / Shift+LMB        Zoom   Wheel")
        legend.setObjectName("legend")
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        legend.setFixedHeight(30)
        rcol.addWidget(legend)
        self._banshee_view_holder = QWidget()
        _bvh = QVBoxLayout(self._banshee_view_holder)
        _bvh.setContentsMargins(0, 0, 0, 0)
        _bvh.addWidget(self.viewer)  # single shared viewer starts here
        rcol.addWidget(self._banshee_view_holder, 1)
        rcol.addLayout(self._build_hide_row("banshee"))

        # wrap the controls column so it scrolls vertically when the window is short.
        # fix its width to the FULL content width plus the vertical scrollbar, so every
        # field is visible horizontally and the (space-reserving) bar never overlaps it.
        controls_scroll = QScrollArea()
        controls_scroll.setWidget(controls)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        _sbw = controls_scroll.verticalScrollBar().sizeHint().width() or 16
        _cw = max(
            controls.sizeHint().width(),
            controls.minimumSizeHint().width(),
            self.banshee_io.sizeHint().width(),
        )
        self._left_width = _cw + _sbw + 4
        controls_scroll.setMaximumWidth(self._left_width)
        controls_scroll.setMinimumWidth(
            0
        )  # so the Gear sub-tab can shrink the column to the
        # gear sections' width (no right-hand gap)

        # left column: Save/Load bar pinned at the top, scrolling controls beneath it. The bar
        # gets a right margin equal to the scrollbar gutter so its right edge lines up with the
        # scrolled content below.
        controls_col = QWidget()
        _ccl = QVBoxLayout(controls_col)
        _ccl.setContentsMargins(0, 0, 0, 0)
        _ccl.setSpacing(8)
        _bar_row = QHBoxLayout()
        _bar_row.setContentsMargins(8, 8, 8 + _sbw, 0)
        _bar_row.addWidget(self.banshee_io)
        _ccl.addLayout(_bar_row)
        _ccl.addWidget(controls_scroll, 1)
        controls_col.setMaximumWidth(self._left_width)
        controls_col.setMinimumWidth(0)
        # The QTabWidget sizes its content stack to the WIDEST page, so the wide Edit-Ikran page
        # keeps the column from narrowing on the View Gear tab. Ignoring its horizontal size hint
        # drops it out of the stack's minimum; its real width is driven by setFixedWidth below.
        controls_col.setSizePolicy(
            QSizePolicy.Policy.Ignored, controls_col.sizePolicy().verticalPolicy()
        )

        self._subtab_orig_w = {}  # which -> original (controls) left-column width
        self._gear_page_w = {}  # which -> gear page width (sized to the gear sections)

        # The left controls become a 2-tab stack (Edit Ikran | Edit Gear). The shared 3-D viewer
        # stays on the RIGHT, so switching to Gear keeps the Ikran in view - preview gear on the model.
        self.ikran_subtabs = QTabWidget()
        self.ikran_subtabs.setObjectName("subtabs")
        self.ikran_subtabs.setDocumentMode(True)
        self.ikran_subtabs.tabBar().setDrawBase(False)
        self.ikran_subtabs.addTab(controls_col, "Edit Ikran")
        self.ikran_subtabs.addTab(self._build_gear_panel("ikran"), "View Gear")
        self._subtab_orig_w["ikran"] = self._left_width
        self.ikran_subtabs.currentChanged.connect(
            lambda _i: self._on_subtab_changed("ikran")
        )
        self._on_subtab_changed("ikran")

        banshee_split = QSplitter()
        self._banshee_split = banshee_split
        self._ikran_box = self._box_subtabs(self.ikran_subtabs)
        self._ikran_viewcol = right
        self._arrange_view_split(banshee_split, right, self._ikran_box, self._left_width, 560)
        self._on_subtab_changed("ikran")  # now the box exists -> size the pane too

        # ---- shell: left section menu (Ikran | Na'vi | Item Wiki | Settings) + content ----
        # Logo + "Pandora Paint" wordmark, pinned to the bottom-left of the side menu.
        self._logo = QLabel()
        self._logo.setToolTip("Pandora Paint")
        self._logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._wordmark = QLabel("Pandora Paint")
        self._wordmark.setObjectName("wordmark")
        self._wordmark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # font-family is set in app.py via a high-specificity 'QLabel#wordmark' rule (the same path
        # the tab bars use) - an inline font-family here loses to the app stylesheet's '*' Inter rule.
        self._wordmark.setStyleSheet(
            "font-size:23px; color:%s; background:transparent;" % theme.accent_active()
        )
        _logo_footer = QWidget()
        _fcol = QVBoxLayout(_logo_footer)
        _fcol.setContentsMargins(0, 8, 0, 14)
        _fcol.setSpacing(8)  # gap between logo and wordmark - lifts the logo a touch above the text
        _fcol.addWidget(self._logo)
        _fcol.addWidget(self._wordmark)
        self.tabs = MainNav(
            footer=_logo_footer,
            top_pad=self._SUBTAB_TOP_GAP + self._SIDETAB_EXTRA_DROP,
        )
        _icondir = assets.resource_path("icons")
        _icon = lambda name: os.path.join(_icondir, name)  # noqa: E731
        self._tab_banshee = self.tabs.addTab(
            self._tab_page(banshee_split), "Ikran", _icon("ikran.png")
        )
        self._tab_navi = self.tabs.addTab(
            self._tab_page(self._build_navi_tab()), "Na'vi", _icon("navi.png")
        )
        def _camo_tab_builder():
            # Built on first show (keeps the two rejuice parses off startup). The Camo tab no longer
            # hosts the shared 3-D viewer, so there is nothing to reparent here - it is purely the
            # two palette editors.
            return self._build_camo_tab()

        self._tab_camo = self.tabs.addTab(
            _LazyTab(_camo_tab_builder), "Camo", _icon("camo.png")
        )
        self._tab_wiki = self.tabs.addTab(
            self._build_wiki_tab(), "Item Wiki", _icon("itemwiki.png")
        )
        self._tab_settings = self.tabs.addTab(
            self._build_settings_tab(), "Settings", _icon("settings.png")
        )
        # Button-style row below Settings: opens the GitHub issues page in the browser. It has no
        # content page - clicking it fires the handler and the selection snaps back to the current
        # tab. The icon + row are themed exactly like the tabs above it.
        self.tabs.addAction(
            "Report Bugs", _icon("reportbug.png"), self._open_bug_report
        )
        self._size_corner_logo()  # fit the logo to the menu header
        QTimer.singleShot(0, self._size_corner_logo)  # re-fit once laid out
        self.tabs.currentChanged.connect(self._on_tab_changed)
        # frameless shell: custom titlebar on top, nav (sidebar meets the window edges) below
        self._titlebar = TitleBar(self, "Pandora Paint")
        shell = QWidget()
        self._shell = shell
        shl = QVBoxLayout(shell)
        shl.setContentsMargins(0, 0, 0, 0)
        shl.setSpacing(0)
        shl.addWidget(self._titlebar)
        shl.addWidget(self.tabs, 1)
        self.setCentralWidget(shell)
        # resize affordance for the frameless window (compositor-driven via QSizeGrip)
        self._grip = QSizeGrip(shell)
        self._grip.setFixedSize(16, 16)
        self._grip.raise_()
        QTimer.singleShot(0, self._reposition_grip)

        # _navi_ready was computed by _recompute_navi_ready() during the Settings build (it gates
        # on the mandatory meshes + textures, one-of-gender-pair, not just whether a folder is set).
        self._navi_ready = bool(getattr(self, "_navi_ready", False))
        self._update_tab_states()
        self._update_camo_tab_enabled()  # Camo tab is enabled only when the rejuice is set
        # open on Banshee when its assets are ready; otherwise land on Settings to set up.
        # On the very first launch with no Export Folder yet, also land on Settings and prompt.
        banshee_ready = not assets.missing_required(
            assets.load_config().get("paths", {})
        )
        # "First run" = the one-time quick-setup hasn't been completed yet. The dialog pre-fills from
        # the CURRENT config (defaults on a fresh install), so showing it to an existing user who
        # simply hasn't seen it can't clobber their settings.
        _cfg_settings = assets.load_config().get("settings") or {}
        first_run = not _cfg_settings.get("first_run_done", False)
        first_export_prompt = (not assets.export_configured()) and not _cfg_settings.get(
            "export_prompted", False
        )
        land_settings = (not banshee_ready) or first_run or first_export_prompt
        self.tabs.setCurrentIndex(
            self._tab_settings if land_settings else self._tab_banshee
        )
        if first_run:
            # the quick-setup dialog now includes the Export-Folder picker (merged in), so the
            # standalone export prompt is skipped while it's showing
            QTimer.singleShot(0, self._show_first_run_setup)
        elif first_export_prompt:
            QTimer.singleShot(0, self._prompt_first_export_folder)

        self._restore_geometry()
        self._on_tab_changed(self.tabs.currentIndex())
        # Force the accent across the freshly-built UI. app.py themes the sheet at startup, but this
        # guarantees it even if that wiring is stale/missing - and any lazy tabs built later inherit
        # the (already themed) app stylesheet at creation time.
        self._reapply_theme()
            # Defer the slow texture decode+upload until AFTER the window is on screen (app appears
            # instantly). A QTimer.singleShot(0) fires once the event loop starts and loads only the
            # active tab's viewer behind one progress bar; the other tab loads lazily when shown.
        self._loaded_tabs = set()  # tab indices whose viewer textures are loaded
        QTimer.singleShot(0, self._load_active_viewer)

    def _load_active_viewer(self):
        """Load textures for whichever viewer is on the current tab (once). Called deferred at
        startup and on first switch to a tab, so each viewer loads lazily and only once."""
        idx = self.tabs.currentIndex()
        if idx in self._loaded_tabs:
            return
        if idx == getattr(self, "_tab_banshee", -2):
            self._loaded_tabs.add(idx)
            self._autoload()
        elif idx == getattr(self, "_tab_navi", -2):
            self._loaded_tabs.add(idx)
            self._load_navi_meshes()

    # ---------------- load from the configured (referenced) asset paths ----------------
    def _autoload(self):
        cfg = assets.load_config()
        slots = dict(cfg.get("paths", {}))
        found, failed = [], []
        stale = assets.invalid_paths(slots)
        tex_slots = {
            "body_color": ("body", "color"),
            "body_material": ("body", "material"),
            "body_pattern": ("body", "pattern"),
            "head_color": ("head", "color"),
            "head_material": ("head", "material"),
            "head_pattern": ("head", "pattern"),
            "body_normal": ("body", "normal"),
            "head_normal": ("head", "normal"),
            "body_dn_mask": ("body", "dn_mask"),
            "head_dn_mask": ("head", "dn_mask"),
            "detail1": ("shared", "detail1"),
            "detail2": ("shared", "detail2"),
            "detail3": ("shared", "detail3"),
            "wing_color": ("wing", "color"),
            "eye_color": ("eye", "color"),
        }
        panels = {"body": self.body, "head": self.head}

        # 1) reflect every configured path in the UI first, independent of whether
        #    the file loads/decodes (so the boxes always mirror the config).
        model_path = slots.get("model")
        if model_path:
            self.model_path_edit.setText(model_path)
        for slot, (key, role) in tex_slots.items():
            p = slots.get(slot)
            panel = panels.get(key)
            if p and panel is not None and role in ("color", "material", "pattern"):
                panel.set_texture_path(role, p)

        # 2) load the model into the viewer
        if model_path and os.path.isfile(model_path):
            try:
                self.viewer.load_model(model_path)
                found.append(os.path.basename(model_path))
            except Exception as e:
                failed.append(f"model ({e})")

        # 3) load textures into the viewer - gated behind the shared progress bar so the model
        #    only appears once every texture is uploaded (same UX as the Na'vi path). The async
        #    single-texture path (_load_texture_async) is still used for live swaps via open_texture.
        tex_items = []
        for slot, (key, role) in tex_slots.items():
            p = slots.get(slot)
            if p and os.path.isfile(p):
                tex_items.append((key, role, p))

        def _decode_banshee(item):
            key, role, p = item
            label = "%s / %s" % (key, role)
            try:
                return np.ascontiguousarray(load_rgba(p)), label
            except Exception as exc:  # noqa: BLE001
                return None, "%s (%s)" % (label, exc)

        def _upload_banshee(item, arr):
            key, role, _p = item
            self.viewer.set_texture(key, role, arr)

        if tex_items:
            _up, fails = self._load_textures_gated(
                "Loading textures\u2026", tex_items, _decode_banshee, _upload_banshee
            )
            # fails are "key / role (...)" labels; the rest uploaded fine.
            failed_labels = {f.split(" (")[0] for f in fails}
            for it in tex_items:
                label = "%s / %s" % (it[0], it[1])
                if label in failed_labels:
                    continue
                found.append("%s/%s" % (it[0], it[1]))
            failed.extend(fails)

        if stale:
            labels = ", ".join(assets.SLOT_HINT.get(s, s) for s in stale)
            self.statusBar().showMessage(
                f"Some files have moved or been deleted ({labels}). Open Assets... to relink.",
                0,
            )
        elif failed:
            self.statusBar().showMessage(
                "Could not read: "
                + ", ".join(failed)
                + "   (for .dds, check that texture2ddecoder is installed)",
                0,
            )
        elif found:
            self.statusBar().showMessage("Loaded: " + ", ".join(found), 9000)
        else:
            self.statusBar().showMessage(
                "No assets loaded. Use the Assets... button to add your extracted files."
            )

    # ---------------- tabbed shell: builders, greying, GL load/unload ----------------
    def _build_navi_tab(self):
        """Na'vi tab - edit-left / view-right, with palette-driven recolour pickers."""
        self.navi = NaviControls(
            on_change=self._on_navi_changed,
            on_load=self._navi_load_blueitem,
            on_pick_asset=self._navi_pick_asset,
            get_asset_path=self._navi_slot_path,
            on_set_asset_path=self._navi_set_asset_path,
            on_export_texture=self._navi_export_texture,
            fill_width=self._left_width,
        )
        cscroll = QScrollArea()
        cscroll.setWidget(self.navi)
        cscroll.setWidgetResizable(True)
        cscroll.setFrameShape(QFrame.Shape.NoFrame)
        _sbw = cscroll.verticalScrollBar().sizeHint().width() or 16
        # Match the Ikran side column exactly: the Na'vi controls now stretch to fill this same
        # width (NaviControls got fill_width=_left_width above), so both side panels line up.
        _naviw = self._left_width

        # Save / Load bar pinned above the (scrolling) sections, so it stays put at the top.
        self.navi_io = SaveLoadBar(
            "Save / Load Na'vi",
            "navi",
            self._navi_preset_collect,
            self._navi_preset_apply,
            self._navi_reset_defaults,
        )
        left = QWidget()
        lcol = QVBoxLayout(left)
        lcol.setContentsMargins(8, 8, 8, 0)
        lcol.setSpacing(8)
        _navi_bar_row = QHBoxLayout()  # right edge lines up with the sections,
        _navi_bar_row.setContentsMargins(
            0, 0, _sbw + 4, 0
        )  # leaving the scrollbar gutter clear
        _navi_bar_row.addWidget(self.navi_io)
        lcol.addLayout(_navi_bar_row)
        lcol.addWidget(cscroll, 1)
        left.setMaximumWidth(_naviw)

        # The Na'vi tab reuses the SINGLE shared viewer (self.viewer), reparented here on tab
        # change and switched to "navi" mode - one GL context, no second-widget conflict.

        right = QWidget()
        rcol = QVBoxLayout(right)
        rcol.setContentsMargins(8, 8, 8, 8)
        rcol.setSpacing(8)

        toolrow = QHBoxLayout()
        toolrow.setSpacing(8)
        toolrow.addWidget(QLabel("Body Type"))
        self.navi_gender = QComboBox()
        self.navi_gender.addItem("Male", "m")
        self.navi_gender.addItem("Female", "f")
        # connect AFTER populating so filling the combo doesn't fire a premature reload
        self.navi_gender.currentIndexChanged.connect(self._load_navi_meshes)
        toolrow.addWidget(self.navi_gender)
        toolrow.addStretch(1)
        rcol.addLayout(toolrow)

        legend = QLabel("Orbit   LMB        Pan   MMB / Shift+LMB        Zoom   Wheel")
        legend.setObjectName("legend")
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        legend.setFixedHeight(30)
        rcol.addWidget(legend)
        self._navi_view_holder = QWidget()
        _nvh = QVBoxLayout(self._navi_view_holder)
        _nvh.setContentsMargins(0, 0, 0, 0)
        rcol.addWidget(
            self._navi_view_holder, 1
        )  # shared viewer reparents in on tab change
        rcol.addLayout(self._build_hide_row("navi"))
        # NOTE: textures are NOT loaded here - they're loaded lazily by _load_active_viewer the
        # first time this tab is shown, so the app window can appear instantly at startup.

        # Left controls become a 2-tab stack (Edit Na'vi | Edit Gear); the viewer stays on the right.
        self.navi_subtabs = QTabWidget()
        self.navi_subtabs.setObjectName("subtabs")
        self.navi_subtabs.setDocumentMode(True)
        self.navi_subtabs.tabBar().setDrawBase(False)
        self.navi_subtabs.addTab(left, "Edit Na'vi")
        self.navi_subtabs.addTab(self._build_gear_panel("navi"), "View Gear")
        # body-type changes flip each gear slot to the matching male/female model + re-preview
        self.navi_gender.currentIndexChanged.connect(
            self._on_navi_gender_changed_for_gear
        )
        self._subtab_orig_w["navi"] = _naviw
        self.navi_subtabs.currentChanged.connect(
            lambda _i: self._on_subtab_changed("navi")
        )
        self._on_subtab_changed("navi")

        split = QSplitter()
        self._navi_split = split
        self._navi_box = self._box_subtabs(self.navi_subtabs)
        self._navi_viewcol = right
        self._arrange_view_split(split, right, self._navi_box, _naviw, 600)
        self._on_subtab_changed("navi")  # now the box exists -> size the pane too
        return split

    # ---------------- Save / Load preset wiring ----------------
    def _banshee_preset_collect(self):
        return {
            "head": self.head.export_preset(),
            "body": self.body.export_preset(),
            "gear": self._gear_preset_collect("ikran"),
        }

    def _banshee_preset_apply(self, data, missing):
        data = data or {}
        self.head.apply_preset(data.get("head") or {}, missing)
        self.body.apply_preset(data.get("body") or {}, missing)
        self._gear_preset_apply("ikran", data.get("gear"), missing)

    def _navi_preset_collect(self):
        out = {
            "sections": self.navi.export_preset(),
            "gear": self._gear_preset_collect("navi"),
        }
        panel = getattr(self, "navi_assets_panel", None)
        if panel is not None:
            out["assets"] = panel.export_overrides()
        return out

    def _navi_preset_apply(self, data, missing):
        data = data or {}
        panel = getattr(self, "navi_assets_panel", None)
        if panel is not None and data.get("assets"):
            panel.apply_overrides(data["assets"], missing)
        self.navi.apply_preset(data.get("sections") or {}, missing)
        self._gear_preset_apply("navi", data.get("gear"), missing)

    def _banshee_reset_defaults(self):
        """'Default' on the Ikran Save/Load bar: head + body colours and pattern controls back to
        their defaults (the per-panel colour reset, plus neutral level/invert), overwrite cleared."""
        for panel in (self.head, self.body):
            panel.overwrite.setChecked(False)
            panel.reset_colours()
            panel.set_control(None)
        self._gear_preset_reset("ikran")
        self.banshee_io._refresh_presets()

    def _navi_reset_defaults(self):
        """'Default' on the Na'vi Save/Load bar: every section's colours and flags back to default."""
        self.navi.reset_all()
        self._gear_preset_reset("navi")
        self.navi_io._refresh_presets()

    def _tab_page(self, content):
        """Wrap a top-level tab's content with a small left indent and a top gap, so the boxed
        sub-tabs sit visually nested below (and slightly in from) the main tab row."""
        page = QWidget()
        pl = QVBoxLayout(page)
        pl.setContentsMargins(14, self._SUBTAB_TOP_GAP, 0, 0)  # left indent + gap under the main tab bar
        pl.setSpacing(0)
        pl.addWidget(content)
        return page

    def _size_corner_logo(self):
        """Size the app logo for the bottom of the left section-menu, tinting its recolourable #E1E9E7
        region to the active accent so the brand mark tracks the theme. The accent is MULTIPLIED onto
        the light region (overlay), which keeps the mark's soft shading and anti-aliasing while taking
        the accent hue - cleaner than replacing a flat fill. The white outline, the brush tip, and
        every other pixel (and the alpha) are left untouched."""
        logo = getattr(self, "_logo", None)
        if logo is None:
            return
        import os
        from PyQt6.QtGui import QImage, QColor, QPixmap

        # Recolourable ("white") logo variant: light #E1E9E7 fills over a kept-white outline, so the
        # accent overlays cleanly. Fall back to the full-colour app icon if it isn't present yet.
        dpr = max(1.0, float(self.devicePixelRatioF()))
        side = int(round(72 * dpr))
        path = assets.resource_path("icons", "pandora-paint-1024-white.png")
        if os.path.isfile(path):
            pm = QPixmap(path).scaled(
                side,
                side,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            from app import app_icon

            pm = app_icon().pixmap(72)
        if pm.isNull():
            return
        img = pm.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        src = QColor("#E1E9E7")  # the recolourable fill
        dst = QColor(theme.accent_active())
        sr, sg, sb = src.red(), src.green(), src.blue()
        dr, dg, db = dst.red(), dst.green(), dst.blue()
        # tol < distance(#E1E9E7, #FFFFFF) == 44.3, so the white outline is never tinted.
        tol = 34.0
        for y in range(img.height()):
            for x in range(img.width()):
                c = img.pixelColor(x, y)
                a = c.alpha()
                if a == 0:
                    continue
                dist = (
                    (c.red() - sr) ** 2 + (c.green() - sg) ** 2 + (c.blue() - sb) ** 2
                ) ** 0.5
                if dist <= tol:
                    k = 1.0 - dist / tol  # 1 at exact #E1E9E7, fading to 0 at the edge
                    # MULTIPLY the accent onto this pixel (overlay), blended in by k for clean AA.
                    mr = c.red() * dr / 255.0
                    mg = c.green() * dg / 255.0
                    mb = c.blue() * db / 255.0
                    nr = round(c.red() * (1 - k) + mr * k)
                    ng = round(c.green() * (1 - k) + mg * k)
                    nb = round(c.blue() * (1 - k) + mb * k)
                    img.setPixelColor(x, y, QColor(nr, ng, nb, a))
        out = QPixmap.fromImage(img)
        out.setDevicePixelRatio(dpr)
        logo.setPixmap(out)

    def _reposition_grip(self):
        """Park the resize grip in the bottom-right corner of the shell."""
        grip = getattr(self, "_grip", None)
        shell = getattr(self, "_shell", None)
        if grip is None or shell is None:
            return
        grip.move(shell.width() - grip.width() - 1, shell.height() - grip.height() - 1)
        grip.setVisible(not self.isMaximized())
        grip.raise_()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._reposition_grip()

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == QEvent.Type.WindowStateChange:
            tb = getattr(self, "_titlebar", None)
            if tb is not None:
                tb.sync_max_glyph()
            self._reposition_grip()

    def _box_subtabs(self, tabs):
        """Wrap a sub-tab widget in a bordered box (QFrame#subtabbox) so the Ikran|Gear / Na'vi|Gear
        tabs read as nested one level under the top-level tabs."""
        box = QFrame()
        box.setObjectName("subtabbox")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(1, 1, 1, 1)
        bl.setSpacing(0)
        bl.addWidget(tabs)
        return box

    def _build_gear_panel(self, which):
        """Gear sub-tab: a Save / Load bar pinned at the top + four gear slots (model + textures + 4
        colour pickers each) in a scroll area below, mirroring the Na'vi / Ikran layout. The shared
        3-D viewer stays on the RIGHT, so gear loaded here previews on top of the character."""
        gc = GearControls(
            on_change=lambda key, w=which: self._gear_changed(w, key),
            on_pick_model=lambda key, gender=None, w=which: self._gear_pick_model(
                w, key, gender
            ),
            on_pick_texture=lambda key, slot, w=which: self._gear_pick_texture(
                w, key, slot
            ),
            on_hide=lambda key, w=which: self._gear_hide_changed(w, key),
            on_model_change=lambda key, gender, w=which: self._gear_preview(w, key),
            gendered=(which == "navi"),
            gender=(self._navi_gender() if which == "navi" else "m"),
            # Both Ikran and Na'vi View Gear stretch their sections to fill the Ikran column width,
            # so View Gear (Na'vi) sections match View Gear (Ikran) sections.
            fill_width=self._left_width,
        )
        if not hasattr(self, "_gear_panels"):
            self._gear_panels = {}
        self._gear_panels[which] = gc

        cscroll = QScrollArea()
        cscroll.setWidget(gc)
        cscroll.setWidgetResizable(True)
        cscroll.setFrameShape(QFrame.Shape.NoFrame)
        # Always reserve the vertical scrollbar gutter (like the Ikran controls tab). Without a
        # diffuse/colour stack the gear sections are short, so an as-needed bar would vanish and
        # leave dead space on the right - this keeps the column flush to the gear content.
        cscroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        cscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _sbw = cscroll.verticalScrollBar().sizeHint().width() or 16

        # The parent Na'vi / Ikran Save/Load section, mirrored here so gear can be saved/loaded
        # straight from View Gear. It writes the SAME preset as the Edit tab's bar (sections + gear
        # together), and both dropdowns refresh from disk on open, so they stay in sync.
        if which == "navi":
            bar = SaveLoadBar(
                "Save / Load Na'vi",
                "navi",
                self._navi_preset_collect,
                self._navi_preset_apply,
                self._navi_reset_defaults,
            )
        else:
            bar = SaveLoadBar(
                "Save / Load Ikran",
                "banshee",
                self._banshee_preset_collect,
                self._banshee_preset_apply,
                self._banshee_reset_defaults,
            )

        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(8, 8, 8, 0)
        col.setSpacing(8)
        bar_row = QHBoxLayout()  # right edge lines up with the sections,
        bar_row.setContentsMargins(
            0, 0, _sbw + 4, 0
        )  # leaving the scrollbar gutter clear
        bar_row.addWidget(bar)
        col.addLayout(bar_row)
        col.addWidget(cscroll, 1)

            # Na'vi gear is sized to its sections' width (584 + gutter) so its scrollbar lines up and
            # matches the Na'vi column. Ikran keeps the full Edit-Ikran width (gear sections stretch to
            # fill). Both View-Gear pages fill the Ikran column width, so all gear pages match.
        self._gear_page_w[which] = self._left_width
        return page

    # ---- shared viewer-side helpers (Na'vi / Ikran / Camo all honour the Settings toggle) ----
    def _viewer_left(self):
        return bool(assets.load_config().get("settings", {}).get("viewer_left", True))

    def _arrange_view_split(self, split, viewer_col, box, box_w, viewer_w=600):
        """Order a view split's two panes per the viewer-side setting: the viewer expands and the
        controls box keeps its width, divider locked. Re-runnable to flip sides live."""
        left_is_viewer = self._viewer_left()
        first, second = (viewer_col, box) if left_is_viewer else (box, viewer_col)
        split.insertWidget(0, first)
        split.insertWidget(1, second)
        vi = 0 if left_is_viewer else 1
        bi = 1 - vi
        split.setStretchFactor(vi, 1)  # viewer takes the freed space
        split.setStretchFactor(bi, 0)  # controls keep their width
        split.setCollapsible(vi, True)
        split.setCollapsible(bi, False)
        sizes = [0, 0]
        sizes[vi], sizes[bi] = viewer_w, box_w
        split.setSizes(sizes)
        if split.count() > 1:
            split.handle(1).setEnabled(False)  # divider not draggable
        split.setHandleWidth(2)

    def _apply_viewer_side(self):
        """Re-order every view split after the Settings toggle changes. (The Camo tab no longer has
        a viewer, so it is not in this list - it ignores the left/right config.)"""
        for split, viewcol, box, w in (
            (getattr(self, "_banshee_split", None), getattr(self, "_ikran_viewcol", None),
             getattr(self, "_ikran_box", None), getattr(self, "_left_width", 600)),
            (getattr(self, "_navi_split", None), getattr(self, "_navi_viewcol", None),
             getattr(self, "_navi_box", None), self._subtab_orig_w.get("navi", 600)),
        ):
            if split is not None and viewcol is not None and box is not None:
                self._arrange_view_split(split, viewcol, box, int(w))

    def _build_camo_tab(self):
        """Camo tab: two all-visible palette editors - no viewer, no save/load, no gear sections,
        and it ignores the viewer-side (left/right) config. 'Edit Gear/Weapon Camo' edits the camo
        rejuice (GearCamoColorPalette); 'Edit Gear/Weapon Colour' edits the vanity gear-colours
        rejuice (GearColorPalette). Each sub-tab loads only if its own rejuice is set in Settings;
        the tab itself is enabled when EITHER is present. The 3-D viewer and gear-camo render logic
        stay in the codebase (dormant) for possible later use."""
        wiki = assets.resource_path("item_wiki.json")
        self._camo_editor = PaletteEditor(
            b"GearCamoColorPalette", kind="camo", wiki_path=wiki,
            title="Gear/Weapon Camo", on_status=self._set_camo_status,
        )
        self._gearcolor_editor = PaletteEditor(
            b"GearColorPalette", kind="colors",
            title="Gear/Weapon Colour", on_status=self._set_camo_status,
        )

        self.camo_subtabs = QTabWidget()
        self.camo_subtabs.setObjectName("subtabs")
        self.camo_subtabs.setDocumentMode(True)
        self.camo_subtabs.tabBar().setDrawBase(False)
        self._camo_tab_idx = self.camo_subtabs.addTab(
            self._camo_editor, "Edit Gear/Weapon Camo"
        )
        self._gearcolor_tab_idx = self.camo_subtabs.addTab(
            self._gearcolor_editor, "Edit Gear/Weapon Colour"
        )
        self._reload_camo_editors()
        return self._tab_page(self._box_subtabs(self.camo_subtabs))

    def _reload_camo_editors(self):
        """Load each rejuice into its editor if configured + present, and enable/disable that
        sub-tab accordingly. Called on tab build and after the Settings assets change."""
        if not hasattr(self, "camo_subtabs"):
            return
        paths = (assets.load_config().get("paths", {}) or {})
        camo_rej = paths.get("gearcamo_rejuice") or ""
        color_rej = paths.get("gearcolors_rejuice") or ""
        camo_ok = bool(camo_rej and os.path.isfile(camo_rej))
        color_ok = bool(color_rej and os.path.isfile(color_rej))
        if camo_ok:
            self._camo_editor.load(camo_rej)
        if color_ok:
            self._gearcolor_editor.load(color_rej)
        self.camo_subtabs.setTabEnabled(self._camo_tab_idx, camo_ok)
        self.camo_subtabs.setTabEnabled(self._gearcolor_tab_idx, color_ok)
        self.camo_subtabs.setTabToolTip(
            self._camo_tab_idx,
            "" if camo_ok else "Set the gear/weapon camo rejuice in Settings (Gear Camo and Colours)",
        )
        self.camo_subtabs.setTabToolTip(
            self._gearcolor_tab_idx,
            "" if color_ok else "Set the gear/weapon colour rejuice in Settings (Gear Camo and Colours)",
        )
        cur = self.camo_subtabs.currentIndex()
        if cur == self._camo_tab_idx and not camo_ok and color_ok:
            self.camo_subtabs.setCurrentIndex(self._gearcolor_tab_idx)
        elif cur == self._gearcolor_tab_idx and not color_ok and camo_ok:
            self.camo_subtabs.setCurrentIndex(self._camo_tab_idx)

    def _set_camo_status(self, msg):
        # the in-panel status label was removed; camo status now shows transiently on the status bar
        self.statusBar().showMessage(msg, 8000)

    def _on_subtab_changed(self, which):
        """Size the left column to the active sub-tab: the Gear tab shrinks to the gear sections'
        width (the viewer takes the freed space); the original tab keeps its full control width."""
        tabs = self.ikran_subtabs if which == "ikran" else self.navi_subtabs
        on_gear = tabs.currentIndex() == 1
        if on_gear and which == "navi":
            # make sure the right male/female picker is showing for the current body type
            panel = self._gear_panels.get("navi")
            if panel is not None:
                panel.set_gender(self._navi_gender())
        w = self._gear_page_w.get(which) if on_gear else self._subtab_orig_w.get(which)
        if w:
            w = int(w)
            tabs.setFixedWidth(w)
            box = (
                getattr(self, "_ikran_box", None)
                if which == "ikran"
                else getattr(self, "_navi_box", None)
            )
            if box is not None:
                box.setFixedWidth(w + 2)  # +2 for the 1px frame on each side
            # A QSplitter keeps its panes at the sizes it last computed and ignores a child's new
            # width hint, so the column looks "locked". Re-distribute explicitly so the pane (and
            # the column) follow the active sub-tab and the viewer reclaims the freed space.
            split = (
                getattr(self, "_banshee_split", None)
                if which == "ikran"
                else getattr(self, "_navi_split", None)
            )
            if split is not None and split.width() > 0:
                rest = max(1, split.width() - (w + 2) - split.handleWidth())
                if self._viewer_left():
                    split.setSizes([rest, w + 2])  # viewer left, controls right
                else:
                    split.setSizes([w + 2, rest])  # controls left, viewer right

    # ---- Gear presets (Save / Load), mirroring the Na'vi / Ikran bars --------------------
    def _gear_preset_collect(self, which):
        panel = self._gear_panels.get(which)
        if panel is None:
            return {}
        return {"sections": {s._key: s.state() for s in panel.sections}}

    def _gear_preset_apply(self, which, data, missing):
        panel = self._gear_panels.get(which)
        if panel is None:
            return
        secs = (data or {}).get("sections", {})
        for s in panel.sections:
            st = secs.get(s._key)
            if st:
                s.restore(st, missing)
        if which in ("navi", "camo"):  # (re)load the meshes + diffuse into the viewer
            for s in panel.sections:
                self._gear_preview(which, s._key)

    def _gear_preset_reset(self, which):
        panel = self._gear_panels.get(which)
        if panel is None:
            return
        for s in panel.sections:
            s.reset()
        if which in ("navi", "camo"):  # clear the now-empty gear meshes from the viewer
            for s in panel.sections:
                self._gear_preview(which, s._key)

    # ---- Gear sub-tab handlers --------------------------------------------
    def _gear_start_dir(self):
        """Where the gear file dialogs open. Prefer the configured asset folder, else home."""
        try:
            cfg = assets.load_config()
            for p in (
                cfg.get("export_folder"),
                cfg.get("paths", {}).get("export_folder"),
            ):
                if p and os.path.isdir(p):
                    return p
        except Exception:
            pass
        return os.path.expanduser("~")

    def _gear_dialog_start(self, existing, model=None):
        """Start dir for a gear pick: this item's own file/folder when the option is on; if it has
        no file yet, fall back to the selected model's folder, else the export root."""
        if assets.get_setting("smart_dialog_start", True):
            s = assets.dialog_start_for(existing, "")  # file / its folder / a dir path
            if not s and model:
                s = assets.dialog_start_for(model, "")  # texture with no folder -> model's folder
            if s:
                return s
        return self._gear_start_dir()

    def _gear_diffuse_rgba(self, state):
        """The gear diffuse as RGBA uint8 for the live preview (None if no diffuse is loaded).
        Gear is shown as-authored now - no recolour."""
        import numpy as np

        dp = state["textures"].get("diffuse")
        if not (dp and os.path.isfile(dp)):
            return None
        try:
            arr = load_rgba(dp)
        except Exception:
            return None
        arr = np.ascontiguousarray(arr).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[2] == 3:
            a = np.full(arr.shape[:2] + (1,), 255, np.uint8)
            arr = np.concatenate([arr, a], axis=2)
        return arr[:, :, :4]

    def _gear_material_rgba(self, state):
        """The gear Material (_m) map as RGBA uint8 - its ALPHA is the camo coverage (Material.a in
        the game camo shader), so camo lands only where the _m alpha allows. None if no _m is loaded;
        a 3-channel _m (no alpha) becomes alpha=255 -> full coverage (same as before)."""
        import numpy as np

        mp = state["textures"].get("material")
        if not (mp and os.path.isfile(mp)):
            return None
        try:
            arr = load_rgba(mp)
        except Exception:
            return None
        arr = np.ascontiguousarray(arr).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[2] == 3:
            a = np.full(arr.shape[:2] + (1,), 255, np.uint8)
            arr = np.concatenate([arr, a], axis=2)
        return arr[:, :, :4]

    def _gear_normal_rgba(self, state):
        """The gear Normal (_n) map as RGBA uint8 for the lit camo path - rgb is the tangent-space
        normal, alpha is baked AO (UnpackNormalAndAO). None if no _n is loaded. This is what gives
        camo'd gear its relief and depth in the preview instead of flat clay."""
        import numpy as np

        npth = state["textures"].get("normal")
        if not (npth and os.path.isfile(npth)):
            return None
        try:
            arr = load_rgba(npth)
        except Exception:
            return None
        arr = np.ascontiguousarray(arr).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[2] == 3:
            a = np.full(arr.shape[:2] + (1,), 255, np.uint8)
            arr = np.concatenate([arr, a], axis=2)
        return arr[:, :, :4]

    def _gear_region_rgba(self, state):
        """The cloth ColorMask (_reg_m) as RGBA uint8 for the cloth/Overlay camo path - its four
        channels are the garment's baked colour regions (camo enters the green .y zone). None if no
        region mask is loaded, in which case the cloth path falls back to a whole-piece overlay."""
        import numpy as np

        rp = state["textures"].get("regions")
        if not (rp and os.path.isfile(rp)):
            return None
        try:
            arr = load_rgba(rp)
        except Exception:
            return None
        arr = np.ascontiguousarray(arr).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[2] == 3:
            a = np.full(arr.shape[:2] + (1,), 255, np.uint8)
            arr = np.concatenate([arr, a], axis=2)
        return arr[:, :, :4]

    def _gear_preview(self, which, key):
        """Push a gear slot's mesh + diffuse to the shared viewer so it previews on top of the
        character (Na'vi or Ikran). Uses the model for the selected body type. No model -> clear it."""
        panel = self._gear_panels.get(which)
        sec = panel.section(key) if panel else None
        if sec is None:
            return
        gkey = "%s:%s" % (which, key)
        model = sec.active_model()
        if not (model and os.path.isfile(model)):
            self.viewer.clear_gear(gkey)
            return
        self.viewer.set_gear(gkey, model, self._gear_diffuse_rgba(sec.state()))
        self.viewer.set_gear_hidden(gkey, sec.hidden())
        if which == "camo":  # _m alpha gates the camo coverage for camo gear
            self.viewer.set_gear_material(gkey, self._gear_material_rgba(sec.state()))
            self.viewer.set_gear_normal(gkey, self._gear_normal_rgba(sec.state()))
            self.viewer.set_gear_region(gkey, self._gear_region_rgba(sec.state()))

    def _gear_changed(self, which, key):
        """A texture changed: refresh just the live preview's diffuse (mesh untouched)."""
        panel = self._gear_panels.get(which)
        sec = panel.section(key) if panel else None
        if sec is None:
            return
        if not (sec.active_model() and os.path.isfile(sec.active_model())):
            return
        rgba = self._gear_diffuse_rgba(sec.state())
        if rgba is not None:
            self.viewer.set_gear_texture("%s:%s" % (which, key), rgba)
        if which == "camo":  # keep the camo coverage (_m alpha) in sync when textures change
            self.viewer.set_gear_material(
                "%s:%s" % (which, key), self._gear_material_rgba(sec.state())
            )
            self.viewer.set_gear_normal(
                "%s:%s" % (which, key), self._gear_normal_rgba(sec.state())
            )
            self.viewer.set_gear_region(
                "%s:%s" % (which, key), self._gear_region_rgba(sec.state())
            )

    def _gear_hide_changed(self, which, key):
        """The Hide Gear tickbox toggled: show/hide just this piece (no mesh rebuild)."""
        panel = self._gear_panels.get(which)
        sec = panel.section(key) if panel else None
        if sec is None:
            return
        self.viewer.set_gear_hidden("%s:%s" % (which, key), sec.hidden())

    def _on_navi_gender_changed_for_gear(self):
        """Body type changed: flip every Na'vi gear slot to the matching male/female model and
        re-preview (the visible picker, and the previewed mesh, follow the body type)."""
        panel = self._gear_panels.get("navi")
        if panel is None:
            return
        panel.set_gender(self._navi_gender())
        for s in panel.sections:
            self._gear_preview("navi", s._key)

    def _gear_pick_model(self, which, key, gender=None):
        sec = self._gear_panels[which].section(key)
        existing = sec.model_paths.get(gender or "m") if sec else None
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose gear model",
            self._gear_dialog_start(existing),
            "Gear model (*.mmb);;All files (*)",
        )
        if not path:
            return
        if sec:
            sec.set_model_path(path, gender)
        self._gear_preview(which, key)

    def _gear_pick_texture(self, which, key, slot):
        sec = self._gear_panels[which].section(key)
        cur = sec._tex_paths.get(slot) if sec else None
        model = sec.active_model() if sec else None
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose %s texture" % slot,
            self._gear_dialog_start(cur, model),
            "Textures (*.dds *.png *.stf);;All files (*)",
        )
        if not path:
            return
        if sec:
            sec.set_texture_path(slot, path)
        # set_texture_path fires on_change -> _gear_changed, which refreshes the preview diffuse

    def _build_wiki_tab(self):
        """Item Wiki tab: a read-only reference browser built from item_wiki.json. Ikran / Na'vi
        sub-tabs, each a row of category tables (Ikran Gear is populated; the rest fill in later).
        Wrapped in the same nested box + indent as the Ikran / Na'vi tabs for a consistent look.

        Built lazily: the wiki is never the landing tab and doesn't touch the shared viewer, so all
        of its construction (JSON read, tab bars, tables) is deferred to the first time it's opened."""

        def _build():
            import wiki
            return self._tab_page(self._box_subtabs(wiki.build_sections()))

        return _LazyTab(_build)

    def _build_settings_tab(self):
        """Settings tab: config location, collapsible asset sections, and preferences."""
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        # ---- configuration location ----
        cfgbox = QGroupBox("Configuration")
        cgl = QVBoxLayout(cfgbox)
        cgl.setContentsMargins(10, 8, 10, 8)
        # 12px between distinct groups = the space BELOW each item+description group, so a description
        # sits well clear of the next item and clearly owns the item above it (which it hugs at 2px
        # inside _desc_group). Subtitles add _GROUP_GAP on top, keeping their separation ~30px.
        cgl.setSpacing(12)
        _BTN_W = 104  # all four path buttons share one width
        _GROUP_GAP = 6  # extra gap above every subtitled group (on top of the 12px cgl spacing)

        def _subtitle_css():
            return "font-weight:700;color:%s;" % theme.accent_active()

        self._cfg_subtitles = []  # subtitle labels, re-tinted to the accent when it changes

        def _subtitle(text, first=False):
            """A bold group subtitle in the active accent colour, with a consistent gap above it
            (none for the first group)."""
            if not first:
                cgl.addSpacing(_GROUP_GAP)
            lbl = QLabel(text)
            lbl.setStyleSheet(_subtitle_css())
            self._cfg_subtitles.append(lbl)
            cgl.addWidget(lbl)
            return lbl

        def _desc_group(*items):
            """Stack an item (a control or a row-layout) with its description tightly so the
            description clearly belongs to the item above it. The group as a whole keeps the normal
            gap below (cgl's spacing) - only the item->description gap is tightened."""
            box = QVBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(2)  # tight: pull the description up to the item it describes
            for it in items:
                if isinstance(it, QWidget):
                    box.addWidget(it)
                else:
                    box.addLayout(it)
            cgl.addLayout(box)
            return box

        # --- Config file (description sits BELOW the picker row) ---
        _subtitle("Config file", first=True)
        crow = QHBoxLayout()
        self.cfg_path_edit = QLineEdit(assets.config_path())
        self.cfg_path_edit.setReadOnly(True)
        self.cfg_path_edit.setToolTip(assets.config_path())
        cfg_chg = QPushButton("Change\u2026")
        cfg_chg.setFixedWidth(_BTN_W)
        cfg_chg.setToolTip(
            "Move config.json to a different folder (remembered across launches)"
        )
        cfg_chg.clicked.connect(self._change_config_dir)
        open_btn = QPushButton("Open folder")
        open_btn.setFixedWidth(_BTN_W)
        open_btn.setToolTip("Open the folder that contains config.json")
        open_btn.clicked.connect(self._open_config_folder)
        crow.addWidget(self.cfg_path_edit, 1)
        crow.addWidget(cfg_chg)
        crow.addWidget(open_btn)
        cap = QLabel(
            "Pandora Paint keeps its preferences and your remembered asset paths in this "
            "file. No game files are stored here - only the locations you point it at."
        )
        cap.setObjectName("legend")
        cap.setWordWrap(True)
        _desc_group(crow, cap)

        # --- Diagnostics (log-file picker, Config-file layout; description BELOW the row) ---
        _subtitle("Diagnostics")
        self.pref_logging = QCheckBox("Enable diagnostic logging")
        self.pref_logging.setChecked(assets.logging_enabled())
        self.pref_logging.setToolTip(
            "Write a diagnostic log (loads, exports, errors) to the file below for troubleshooting. "
            "Off by default."
        )
        self.pref_logging.toggled.connect(self._save_pref_logging)
        lrow = QHBoxLayout()
        self.log_path_edit = QLineEdit(assets.log_file_path())
        self.log_path_edit.setReadOnly(True)
        self.log_path_edit.setToolTip(assets.log_file_path())
        log_chg = QPushButton("Change\u2026")
        log_chg.setFixedWidth(_BTN_W)
        log_chg.setToolTip(
            "Choose a different folder for the log file (remembered across launches)"
        )
        log_chg.clicked.connect(self._change_log_dir)
        log_open = QPushButton("Open file")
        log_open.setFixedWidth(_BTN_W)
        log_open.setToolTip("Open the log file in your default application")
        log_open.clicked.connect(self._open_log_file)
        lrow.addWidget(self.log_path_edit, 1)
        lrow.addWidget(log_chg)
        lrow.addWidget(log_open)
        logcap = QLabel(
            "Off by default. When on, activity and errors (loads, exports, failures) are written "
            "to the file above so you can share it when reporting a problem."
        )
        logcap.setObjectName("legend")
        logcap.setWordWrap(True)
        _desc_group(self.pref_logging, lrow, logcap)

        # --- Presets folder (description BELOW the picker row) ---
        _subtitle("Presets folder")
        prow = QHBoxLayout()
        self.preset_dir_edit = QLineEdit(assets.preset_dir())
        self.preset_dir_edit.setReadOnly(True)
        self.preset_dir_edit.setToolTip(assets.preset_dir())
        pchg = QPushButton("Change\u2026")
        pchg.setFixedWidth(_BTN_W)
        pchg.setToolTip(
            "Choose the folder where Save/Load presets are stored (remembered in config)"
        )
        pchg.clicked.connect(self._change_preset_dir)
        popen = QPushButton("Open folder")
        popen.setFixedWidth(_BTN_W)
        popen.setToolTip("Open the presets folder")
        popen.clicked.connect(self._open_preset_folder)
        prow.addWidget(self.preset_dir_edit, 1)
        prow.addWidget(pchg)
        prow.addWidget(popen)
        pcap = QLabel(
            "Named Save/Load presets from the Ikran and Na'vi tabs (their file paths and "
            "colours) are stored as .json files here. Change it to keep them somewhere you "
            "back up or sync."
        )
        pcap.setObjectName("legend")
        pcap.setWordWrap(True)
        _desc_group(prow, pcap)

        # --- Export folder (where exports land; description + replicate toggle below the row) ---
        _subtitle("Export Folder")
        exrow = QHBoxLayout()
        self.export_dir_edit = QLineEdit(assets.export_folder())
        self.export_dir_edit.setReadOnly(True)
        self.export_dir_edit.setPlaceholderText("(no export folder set)")
        self.export_dir_edit.setToolTip(assets.export_folder() or "No export folder set")
        exchg = QPushButton("Change\u2026")
        exchg.setFixedWidth(_BTN_W)
        exchg.setToolTip("Choose the folder your exported files are written into.")
        exchg.clicked.connect(self._change_export_dir)
        exopen = QPushButton("Open folder")
        exopen.setFixedWidth(_BTN_W)
        exopen.setToolTip("Open the export folder")
        exopen.clicked.connect(self._open_export_folder)
        exrow.addWidget(self.export_dir_edit, 1)
        exrow.addWidget(exchg)
        exrow.addWidget(exopen)
        ecap = QLabel(
            "Where the Na'vi, Ikran and Gear Camo/Colour exports are written when the per-file "
            "Overwrite tickbox is off. With 'Replicate the blue folder structure' on, each file's "
            "blue/\u2026 path is rebuilt inside this folder and no save dialog appears - just a "
            "success message."
        )
        ecap.setObjectName("legend")
        ecap.setWordWrap(True)
        _desc_group(exrow, ecap)
        self.pref_replicate_blue = QCheckBox("Replicate the blue folder structure on export")
        self.pref_replicate_blue.setChecked(assets.export_replicate_blue())
        self.pref_replicate_blue.setToolTip(
            "On (default): exports rebuild each file's blue/\u2026 path inside the Export Folder "
            "above, with no save dialog. Off: every export opens a save dialog so you can place "
            "the file anywhere, with no folders created. The per-file Overwrite tickboxes always "
            "win - they save back over the original file in place."
        )
        self.pref_replicate_blue.toggled.connect(self._save_pref_replicate_blue)
        rcap = QLabel(
            "Turn off to choose where each export goes via a save dialog, with no blue/\u2026 "
            "folders created."
        )
        rcap.setObjectName("legend")
        rcap.setWordWrap(True)
        _desc_group(self.pref_replicate_blue, rcap)

        self.pref_pandora_folder = QCheckBox("Wrap exports in a top-level PandoraPaint folder")
        self.pref_pandora_folder.setChecked(assets.export_pandora_folder())
        self.pref_pandora_folder.setToolTip(
            "On: the replicated blue/\u2026 tree is placed inside a PandoraPaint folder so every "
            "export collects in one packable place. Off (default): the blue/\u2026 tree is written "
            "straight into the Export Folder. Only applies while 'Replicate the blue folder "
            "structure' is on."
        )
        self.pref_pandora_folder.setEnabled(self.pref_replicate_blue.isChecked())
        self.pref_pandora_folder.toggled.connect(self._save_pref_pandora_folder)
        # Indent: this is a sub-setting of 'Replicate the blue folder structure' above.
        _pandora_row = QHBoxLayout()
        _pandora_row.setContentsMargins(0, 0, 0, 0)
        _pandora_row.addSpacing(22)
        _pandora_row.addWidget(self.pref_pandora_folder)
        cgl.addLayout(_pandora_row)

        # --- behaviour & viewer preferences (each checkbox gets a short description) ---
        s = assets.load_config().get("settings", {})

        def _pref_cb(text, desc, cfg_key, default, on_toggle):
            cb = QCheckBox(text)
            cb.setChecked(bool(s.get(cfg_key, default)))
            cb.setToolTip(desc)
            cb.toggled.connect(on_toggle)
            dl = QLabel(desc)
            dl.setObjectName("legend")
            dl.setWordWrap(True)
            _desc_group(cb, dl)  # description hugs the checkbox above it
            return cb

        def _pref_combo(text, desc, cfg_key, default, options, on_change):
            """A labelled dropdown pref. `options` is a list of (label, value); the stored config
            value is matched against the option values. `on_change` receives the chosen value. The
            dropdown sits left-aligned directly under its label (not pushed to the right margin)."""
            lab = QLabel(text)
            combo = QComboBox()
            cur = s.get(cfg_key, default)
            sel = 0
            for i, (lbl, val) in enumerate(options):
                combo.addItem(lbl, val)
                if val == cur:
                    sel = i
            combo.setCurrentIndex(sel)
            combo.setToolTip(desc)
            combo.setSizePolicy(
                QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
            )  # shrink to content so it hugs the left, under the label
            combo.currentIndexChanged.connect(
                lambda i, c=combo: on_change(c.itemData(i))
            )
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(combo)
            row.addStretch(1)  # keep the dropdown left-aligned, filling nothing to its right
            dl = QLabel(desc)
            dl.setObjectName("legend")
            dl.setWordWrap(True)
            _desc_group(lab, row, dl)  # label + dropdown + description as one tight group
            return combo

        _subtitle("Behaviour")
        self.pref_remember_changes = _pref_cb(
            "Remember model and texture changes",
            "Changes made in the Na'vi and Ikran tabs will replace the default Na'vi and Ikran "
            "assets, persisting through sessions. When off, those changes apply only for the "
            "current session and are forgotten next launch.",
            "remember_changes",
            False,
            self._save_pref_remember_changes,
        )
        self._remember_changes = self.pref_remember_changes.isChecked()

        self.pref_load_existing = _pref_cb(
            "Open file dialogs at the most relevant nearby folder",
            "Open each file dialog at the most relevant nearby folder instead of always at the "
            "asset root: the file's own folder if it is already set, otherwise a related file's "
            "location - its paired hair/kuru model, its part's mesh folder, or another already-set "
            "slot. On by default. When off, every dialog opens at the asset folder.",
            "smart_dialog_start",
            True,
            self._save_pref_load_existing,
        )
        self._load_from_existing_path = self.pref_load_existing.isChecked()

        self.pref_remember_geom = _pref_cb(
            "Remember window size and position",
            "Save the window's size and position when you close Pandora Paint and restore them on "
            "the next launch.",
            "remember_geometry",
            True,
            self._save_pref_geometry,
        )

        _subtitle("Viewer")
        # Two independent dropdowns (anti-aliasing + texture filtering); each has its own "Off", so
        # no master switch is needed. The AA value is a mode string ("off"/"fxaa"/"ssaa2"/...).
        # Migrate old configs: `aa_scale` float, or the even-older `ssaa` bool, -> a mode string.
        _legacy_ssaa = s.get("ssaa", True)
        _legacy_scale = float(s.get("aa_scale", 2.0 if _legacy_ssaa else 1.0))
        _aa_default = s.get("aa_mode") or {
            1.5: "ssaa1.5", 2.0: "ssaa2", 3.0: "ssaa3", 4.0: "ssaa4"
        }.get(_legacy_scale, "off" if _legacy_scale <= 1.0 else "ssaa2")

        self.pref_aa = _pref_combo(
            "Anti-aliasing",
            "FXAA is a cheap edge smoother (Performance). SSAA renders the whole preview larger and "
            "downsamples it - the only mode that also removes hair-strand pixelation on the alpha-cut "
            "hair/membrane - at more GPU the higher you go. Off renders at native resolution.",
            "aa_mode",
            _aa_default,
            _AA_OPTIONS,
            self._save_pref_aa,
        )
        self.pref_af = _pref_combo(
            "Texture filtering",
            "Anisotropic filtering sharpens textures viewed at grazing angles (skin, gear, "
            "membrane) and cuts their shimmer. Higher = sharper at steep angles for a small GPU "
            "cost. Off falls back to plain trilinear.",
            "anisotropy",
            s.get("anisotropy", 16),
            _AF_OPTIONS,
            self._save_pref_af,
        )
        if hasattr(self, "viewer") and self.viewer is not None:
            _scale, _fxaa = self._aa_params(_aa_default)
            self.viewer.set_ssaa_scale(_scale)
            self.viewer.set_fxaa(_fxaa)
            self.viewer.set_anisotropy(float(s.get("anisotropy", 16)))


        self.pref_specular = _pref_cb(
            "Specular highlights",
            "Show the skin clear-coat sheen and the hair sheen. Turn off for a flatter, fully "
            "matte preview.",
            "specular",
            True,
            self._save_pref_specular,
        )
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.set_specular(self.pref_specular.isChecked())

        self.pref_viewer_left = _pref_cb(
            "Viewer pane on the left",
            "Put the 3-D viewer on the left and the controls on the right in the Na'vi, Ikran and "
            "Camo tabs. On by default. Turn off to put the viewer on the right instead.",
            "viewer_left",
            True,
            self._save_pref_viewer_side,
        )
        # accent colours, merged at the bottom of the Configuration box
        _subtitle("Accent colours")
        self.theme_panel = ThemeColorsPanel(on_changed=self._reapply_theme)
        cgl.addWidget(self.theme_panel)

        # ---- Item Wiki export ----
        _subtitle("Item Wiki")
        wiki_export = QPushButton("Export Item Wiki")
        wiki_export.setObjectName("accent")
        wiki_export.clicked.connect(self._export_item_wiki)
        wrow = QHBoxLayout()
        wrow.addWidget(wiki_export)
        wrow.addStretch(1)  # keep the button at its natural width, left-aligned
        wiki_cap = QLabel(
            "Export the entire Item Wiki as a .zip of .csv files - one CSV per category sub-tab "
            "(Na'vi Colours, Ikran Patterns, Weapons, and so on), each with the same columns shown "
            "in the wiki plus the item UID. Handy for spreadsheets, diffing, or sharing the data."
        )
        wiki_cap.setObjectName("legend")
        wiki_cap.setWordWrap(True)
        _desc_group(wrow, wiki_cap)

        v.addWidget(cfgbox)

        # ---- Ikran assets: a titled section (QGroupBox), matching the Configuration box ----
        self.assets_panel = AssetsPanel(
            on_changed=self._on_assets_changed, on_reset=self._reset_banshee_assets
        )
        ikran_box = QGroupBox("Ikran Assets")
        ibl = QVBoxLayout(ikran_box)
        ibl.setContentsMargins(10, 8, 10, 8)
        ibl.setSpacing(6)
        ibl.addWidget(self.assets_panel)
        v.addWidget(ikran_box)

        # ---- Na'vi assets: a titled section (QGroupBox), matching the Configuration box ----
        self.navi_assets_panel = NaviAssetsPanel(
            get_gender=self._navi_gender,
            on_changed=self._on_navi_assets_changed,
            on_reset=self._reset_navi_assets,
        )
        navi_box = QGroupBox("Na'vi Assets")
        nbl = QVBoxLayout(navi_box)
        nbl.setContentsMargins(10, 8, 10, 8)
        nbl.setSpacing(6)
        nbl.addWidget(self.navi_assets_panel)
        v.addWidget(navi_box)
        # gender changes affect texture resolution + the expected-path hints - refresh the panel
        if hasattr(self, "navi_gender"):
            self.navi_gender.currentIndexChanged.connect(
                lambda *_: self.navi_assets_panel._refresh()
            )
        self._recompute_navi_ready()

        # ---- Camo assets: the gearcamo_colorpalettes.rejuice (enables the Camo tab) ----
        v.addWidget(self._build_camo_assets_box())

        v.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        return page

    def _reapply_theme(self):
        """Re-apply the accent colours across the running app: the app-wide stylesheet, the side
        menu (sheet + re-tinted icons) and the wordmark. Called when the Appearance panel changes,
        so accent edits take effect without a restart.

        Qt's QSS engine caches resolved rules per widget; calling app.setStyleSheet() with a new
        sheet updates simple widgets but often leaves complex sub-controls (QTabBar tab underlines,
        QGroupBox::title pills, QCheckBox::indicator ticks, themed QPushButtons) painting the old
        colour. Clearing the sheet to "" first forces Qt to drop those cached rules, then re-setting
        rebuilds them from scratch; the unpolish/polish pass repaints anything still holding state."""
        app = QApplication.instance()
        base = theme.base_stylesheet()
        if app is not None and base:
            sheet = theme.apply(base)
            app.setStyleSheet("")  # drop the cached QSS rules so sub-controls re-resolve
            app.setStyleSheet(sheet)
            for w in app.allWidgets():
                try:
                    w.style().unpolish(w)
                    w.style().polish(w)
                    w.update()
                    if hasattr(w, "restyle_arrow"):
                        w.restyle_arrow()  # QPainter arrows don't follow a stylesheet swap
                except Exception:  # noqa: BLE001
                    pass
        tabs = getattr(self, "tabs", None)
        if tabs is not None and hasattr(tabs, "restyle"):
            try:
                tabs.restyle()
            except Exception:  # noqa: BLE001
                pass  # a failed live re-tint shouldn't crash settings; restart applies it
        self._size_corner_logo()  # re-tint the #E1E9E7 logo region to the new accent
        for lbl in getattr(self, "_cfg_subtitles", []):
            try:
                lbl.setStyleSheet("font-weight:700;color:%s;" % theme.accent_active())
            except Exception:  # noqa: BLE001
                pass  # a deleted label shouldn't break the live re-tint
        if hasattr(self, "_wordmark"):
            self._wordmark.setStyleSheet(
                "font-size:23px; color:%s; background:transparent;" % theme.accent_active()
            )

    def _export_item_wiki(self):
        """Export the whole Item Wiki as a .zip of .csv files - one CSV per category sub-tab, columns
        matching the wiki tables (plus the item UID). Prompts for a save location; no-op on cancel."""
        import csv
        import io
        import zipfile

        try:
            import wiki as _wiki

            data = _wiki.load_wiki_data()
        except Exception:  # noqa: BLE001
            data = None
        if not isinstance(data, dict) or not data:
            QMessageBox.warning(self, "Export Item Wiki", "Couldn't load the item wiki data.")
            return

        start = assets.get_setting("export_folder", "") or os.path.expanduser("~")
        default = os.path.join(start, "item_wiki_csv.zip")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Item Wiki", default, "Zip archive (*.zip)"
        )
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"

        hidden = getattr(_wiki, "_HIDDEN_COLS", set())

        def _cols(rows):
            # same column derivation as the wiki table: field keys (first-seen), 'UI Name' lead when
            # any row is named, minus the hidden columns.
            cols = []
            for it in rows:
                for k in it.get("fields") or {}:
                    if k not in cols:
                        cols.append(k)
            has_name = any((it.get("name") or "").strip() for it in rows)
            lead = (["Type"] if "Type" in cols else []) + (["UI Name"] if has_name else [])
            return [c for c in (lead + [c for c in cols if c != "Type"]) if c not in hidden]

        def _safe(name):
            return "".join(c if (c.isalnum() or c in " -_'") else "_" for c in name).strip() or "sheet"

        written = 0
        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                for skey, section in data.items():
                    if not isinstance(section, dict):
                        continue
                    label = section.get("label") or skey
                    items = section.get("items") or []
                    # every type that actually has items, in first-seen order (nothing dropped)
                    types = []
                    for it in items:
                        t = it.get("type", "")
                        if t not in types:
                            types.append(t)
                    for t in types:
                        rows = [it for it in items if it.get("type") == t]
                        if not rows:
                            continue
                        cols = _cols(rows)
                        buf = io.StringIO()
                        w = csv.writer(buf)
                        w.writerow(cols + ["uid"])
                        for it in rows:
                            f = it.get("fields") or {}
                            line = [
                                it.get("name", "") if c == "UI Name"
                                else it.get("type", "") if c == "Type"
                                else f.get(c, "")
                                for c in cols
                            ]
                            line.append(it.get("uid", ""))
                            w.writerow(line)
                        zf.writestr("%s - %s.csv" % (_safe(label), _safe(t)), buf.getvalue())
                        written += 1
        except OSError as e:
            QMessageBox.warning(self, "Export Item Wiki", "Couldn't write the archive:\n%s" % e)
            return

        QMessageBox.information(
            self, "Export Item Wiki", "Exported %d CSV file(s) to:\n%s" % (written, path)
        )

    def _build_hide_row(self, which):
        """View toggles below the shared 3-D viewer: hide the creature/character body and hide all
        gear at once. Both tabs drive the same viewer, so each tab keeps its own checkbox state and
        that state is re-applied whenever its tab becomes active (keeping the two tabs independent)."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 4, 0, 2)
        cb_body = QCheckBox("Hide Ikran" if which == "banshee" else "Hide Na'vi")
        cb_gear = QCheckBox("Hide all gear")
        cb_body.toggled.connect(self.viewer.set_body_hidden)
        cb_gear.toggled.connect(self._reapply_gear_visibility)
        row.addStretch(1)
        row.addWidget(cb_body)
        row.addSpacing(18)
        row.addWidget(cb_gear)
        if which != "banshee":  # the tail curl is a Na'vi-only pose
            cb_tail = QCheckBox("Straighten tail")
            cb_tail.setToolTip(
                "Turn off the built-in Na'vi tail curl and show the raw bind-pose tail. "
                "Bodies the pose doesn't fit are left un-posed automatically."
            )
            cb_tail.toggled.connect(lambda on: self.viewer.set_navi_tail_pose(not on))
            row.addSpacing(18)
            row.addWidget(cb_tail)
            self._nv_straighten_tail = cb_tail
        row.addStretch(1)
        if which == "banshee":
            self._ik_hide_body, self._ik_hide_gear = cb_body, cb_gear
        else:
            self._nv_hide_body, self._nv_hide_gear = cb_body, cb_gear
        return row

    def _isolate_viewer_for(self, active, hide_all_active=False):
        """Show only the active tab's gear in the shared viewer; every other tab's gear is hidden
        per-key. This keeps the Na'vi / Ikran / Camo tabs visually independent on the one viewer
        (so camo gear never shows on Na'vi/Ikran, and vice versa)."""
        self.viewer.set_all_gear_hidden(False)  # global off; visibility is per-key below
        for which, panel in getattr(self, "_gear_panels", {}).items():
            for sec in panel.sections:
                gkey = "%s:%s" % (which, sec._key)
                if which != active:
                    self.viewer.set_gear_hidden(gkey, True)
                else:
                    self.viewer.set_gear_hidden(gkey, hide_all_active or sec.hidden())

    def _reapply_gear_visibility(self, *_):
        """Re-run isolation for whichever tab is active (driven by the 'Hide all gear' toggles)."""
        idx = self.tabs.currentIndex()
        if idx == getattr(self, "_tab_navi", -1) and hasattr(self, "_nv_hide_gear"):
            self._isolate_viewer_for("navi", self._nv_hide_gear.isChecked())
        elif idx == getattr(self, "_tab_banshee", -1) and hasattr(self, "_ik_hide_gear"):
            self._isolate_viewer_for("ikran", self._ik_hide_gear.isChecked())

    def _on_tab_changed(self, idx):
            # One shared GL viewer (self.viewer), reparented into the active tab's holder and switched
            # between "banshee"/"navi" modes. Reparenting within the window preserves the GL context, so
            # there's only ever ONE context - no second-widget detect failure / GLX conflict.
        on_banshee = idx == getattr(self, "_tab_banshee", -1)
        on_navi = idx == getattr(self, "_tab_navi", -1)
        if on_banshee and hasattr(self, "_banshee_view_holder"):
            self._banshee_view_holder.layout().addWidget(self.viewer)
            self.viewer.set_mode("banshee")
            self.viewer.set_body_hidden(
                self._ik_hide_body.isChecked() if hasattr(self, "_ik_hide_body") else False
            )
            self._isolate_viewer_for(
                "ikran", self._ik_hide_gear.isChecked() if hasattr(self, "_ik_hide_gear") else False
            )
            self.viewer.show()
            self.viewer.update()
        elif on_navi and hasattr(self, "_navi_view_holder"):
            self._navi_view_holder.layout().addWidget(self.viewer)
            self.viewer.set_mode("navi")
            self.viewer.set_body_hidden(
                self._nv_hide_body.isChecked() if hasattr(self, "_nv_hide_body") else False
            )
            self._isolate_viewer_for(
                "navi", self._nv_hide_gear.isChecked() if hasattr(self, "_nv_hide_gear") else False
            )
            self.viewer.show()
            self.viewer.update()
        # Camo is applied per gear piece (set_gear_camo) and camo gear is hidden on the Na'vi/Ikran
        # tabs by _isolate_viewer_for, so it can never bleed onto them - no global push/clear needed.
        if hasattr(self, "cfg_path_edit"):
            self.cfg_path_edit.setText(assets.config_path())
            # Lazily load this viewer's textures the first time its tab is shown (deferred so the switch
            # is instant; gated progress bar then runs). Guarded by _loaded_tabs (once). It may not exist
            # yet during the initial __init__ _on_tab_changed - that first load is driven by the startup timer.
        if (
            (on_banshee or on_navi)
            and hasattr(self, "_loaded_tabs")
            and idx not in self._loaded_tabs
        ):
            QTimer.singleShot(0, self._load_active_viewer)

    def _update_tab_states(self):
        """Grey out a tab until its assets are configured."""
        banshee_ready = not assets.missing_required(
            assets.load_config().get("paths", {})
        )
        navi_ready = bool(getattr(self, "_navi_ready", False))
        self.tabs.setTabEnabled(self._tab_banshee, banshee_ready)
        self.tabs.setTabEnabled(self._tab_navi, navi_ready)
        self.tabs.setTabEnabled(self._tab_settings, True)
        self.tabs.setTabToolTip(
            self._tab_banshee,
            "" if banshee_ready else "Set up Ikran assets in Settings first",
        )
        self.tabs.setTabToolTip(
            self._tab_navi,
            "" if navi_ready else "Set up Na'vi assets in Settings first",
        )

    def _on_assets_changed(self):
        self._update_tab_states()
        if hasattr(self, "_loaded_tabs"):
            self._loaded_tabs.add(getattr(self, "_tab_banshee", -2))
        self._autoload()
        if (
            not assets.missing_required(assets.load_config().get("paths", {}))
            and self.tabs.currentIndex() == self._tab_settings
        ):
            self.statusBar().showMessage(
                "Ikran assets ready - the Ikran tab is now enabled.", 6000
            )

    def _navi_load_blueitem(self, key, section, explicit_path=None):
        """Load a single .blueitemtype item and apply it to one Na'vi category. Colour items
        (myColorData2) fill the section's swatches; texture-select items (myTextureData - hair
        caps, skin pattern/stripe selection, etc.) resolve their target(s) against the scanned
        export and bind straight into the viewer. A single file can carry either or both.
        Wired to each section's Load button / path-field Enter key."""
        try:
            import recolor_core
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(
                "Colour loader unavailable (recolor_core import failed): %s" % exc, 9000
            )
            return
        if explicit_path:
            path = explicit_path
        else:
            cur = getattr(section, "path", None)
            opt = assets.get_setting("smart_dialog_start", True)
            root = assets.load_config().get("navi", {}).get("folder", "") or ""
            start = assets.dialog_start_for(cur, root) if opt else root
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load %s item (.blueitemtype)" % key.title(),
                start,
                "Blue item type (*.blueitemtype);;All files (*)",
            )
            if not path:
                return
        if not os.path.isfile(path):
            self.statusBar().showMessage("File not found: %s" % path, 8000)
            return
        try:
            rec = recolor_core.parse_blueitemtype(path)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(
                "Could not parse %s: %s" % (os.path.basename(path), exc), 8000
            )
            return

        cols = [c for c in rec.get("colors", []) if c.get("hex")]
        targets = rec.get("texture_targets") or []
        if not cols and not targets:
            self.statusBar().showMessage(
                "No colours or textures found in %s (not a recognised customization item?)."
                % os.path.basename(path),
                8000,
            )
            return

        notes = []
        if cols:
            colors = [c["hex"] for c in cols]
            indices = [c["index"] for c in cols]
            section.apply_loaded(path, colors, indices)
            notes.append("%d colour(s)" % len(cols))

        if targets:
            cache = assets.load_config().get("navi", {}).get("cache", {}) or {}
            bound, missing = 0, []
            for t in targets:
                engine_path = t.get("path")
                real = assets.navi_resolve_engine_path(cache, engine_path)
                if not real:
                    missing.append(os.path.basename(engine_path or "?"))
                    continue
                try:
                    arr = self._decode_navi_texture(real)
                except Exception:  # noqa: BLE001 - skip a bad map
                    missing.append(os.path.basename(real))
                    continue
                self.viewer.set_navi_texture(t["bucket"], t["role"], arr)
                bound += 1
            if bound:
                notes.append("%d texture(s) bound" % bound)
            if missing:
                notes.append("could not find in your export: %s" % ", ".join(missing))

        cat = rec.get("category")
        slot_note = (
            ("  (file's slot is '%s', applied to %s)" % (cat, key))
            if (cat and cat != key)
            else ""
        )
        self.statusBar().showMessage(
            "Loaded %s from %s: %s%s"
            % (
                key,
                os.path.basename(path),
                ", ".join(notes) or "nothing usable",
                slot_note,
            ),
            8000,
        )

    def _navi_gender(self):
        cb = getattr(self, "navi_gender", None)
        return (cb.currentData() if cb is not None else "m") or "m"

    @staticmethod
    def _decode_navi_texture(path):
        """Decode an STF/standard .dds or a plain image to a contiguous HxWx4 RGBA array.
        Shares the single decode path with the Banshee viewer (widgets.load_rgba)."""
        return np.ascontiguousarray(load_rgba(path))

    def _load_textures_gated(self, title, items, decode_fn, upload_fn):
        """Decode (OFF the GUI thread) + upload a model's textures behind a modal progress bar,
        keeping the shared viewer HIDDEN until every texture is in, then revealing it. Used by
        BOTH viewers (Banshee + Na'vi) so a model never appears half-textured, and the window
        stays responsive during the (CPU-heavy, pure-Python) BCn decode.

        `items`     : list of opaque task descriptors (whatever decode_fn/upload_fn expect).
        `decode_fn` : item -> (decoded_or_None, label).  Runs on a WORKER thread; must do NO GL.
        `upload_fn` : (item, decoded) -> None.  Runs on the MAIN thread (GL upload).
        Returns (uploaded, failures) when every item is done (the call is synchronous to the
        caller, but pumps the event loop so the UI stays live).
        """
        viewer = getattr(self, "viewer", None)
        total = len(items)
        was_visible = bool(viewer and viewer.isVisible())
        if total == 0:
            if viewer is not None and (was_visible or self._viewer_on_active_tab()):
                viewer.show()
                viewer.update()
            return 0, []
        if viewer is not None:
            viewer.hide()

        dlg = QProgressDialog(title, None, 0, total, self)
        dlg.setWindowTitle("Loading")
        dlg.setMinimumDuration(0)  # show immediately
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setCancelButton(None)  # not cancellable - a half-load is worse than waiting
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setValue(0)

        # Decode each item on the shared QThreadPool; collect results back here. We keep this
        # method synchronous (callers expect a return value) by pumping the event loop until the
        # worker results arrive - the GUI stays responsive because the heavy decode is off-thread.
        results = {}  # index -> (decoded_or_None, label)
        state = {"done": 0}

        class _Sig(QObject):
            one = pyqtSignal(int, object, str)  # index, decoded|None, label

        sig = _Sig()

        def _on_one(i, decoded, label):
            results[i] = (decoded, label)
            # upload on the main thread (this slot runs on the main thread)
            if decoded is not None:
                try:
                    upload_fn(items[i], decoded)
                except Exception as exc:  # noqa: BLE001
                    results[i] = (None, "%s (upload: %s)" % (label, exc))
            state["done"] += 1
            dlg.setLabelText("%s\n%s" % (title, label))
            dlg.setValue(state["done"])

        sig.one.connect(_on_one)

        class _DecodeTask(QRunnable):
            def __init__(self, i, item):
                super().__init__()
                self._i, self._item = i, item

            def run(self):
                try:
                    decoded, label = decode_fn(self._item)
                except Exception as exc:  # noqa: BLE001
                    decoded, label = None, "item %d (%s)" % (self._i, exc)
                sig.one.emit(self._i, decoded, label)

        pool = getattr(self, "_pool", None) or QThreadPool.globalInstance()
        for i, item in enumerate(items):
            pool.start(_DecodeTask(i, item))

        # Pump until all workers have reported back (uploads happen in _on_one on this thread).
        try:
            while state["done"] < total:
                QApplication.processEvents(
                    QEventLoop.ProcessEventsFlag.WaitForMoreEvents, 50
                )
                QApplication.processEvents()
        finally:
            dlg.close()
            if viewer is not None and (was_visible or self._viewer_on_active_tab()):
                viewer.show()
                viewer.update()

        uploaded = sum(
            1 for i in range(total) if results.get(i, (None,))[0] is not None
        )
        failures = [
            results[i][1] for i in range(total) if results.get(i, (None,))[0] is None
        ]
        return uploaded, failures

    def _viewer_on_active_tab(self):
        """True if the shared viewer belongs on the currently selected tab (so it should be
        shown after a gated load even if it was hidden during loading)."""
        idx = self.tabs.currentIndex() if hasattr(self, "tabs") else -1
        return idx in (
            getattr(self, "_tab_banshee", -2),
            getattr(self, "_tab_navi", -2),
        )

    def _load_navi_meshes(self, *_):
        """Resolve the Na'vi preview meshes + base textures from the cached scan (+ per-slot
        overrides) and push them into the viewer. Reads the cache only - never walks the export
        folder - so it's cheap on launch and on gender changes. No-ops cleanly when nothing is
        configured/found (e.g. in the dev sandbox, where the meshes live on the VM)."""
        if not hasattr(self, "viewer"):
            return
        navi = assets.load_config().get("navi", {})
        cache = navi.get("cache", {}) or {}
        # Prefer the panel's in-memory overrides: when "Remember model and texture changes" is off,
        # a tab pick lives only here (not config), so reading them keeps the live preview correct.
        _ap = getattr(self, "navi_assets_panel", None)
        overrides = (
            dict(_ap.overrides) if _ap is not None else (navi.get("paths", {}) or {})
        )
        gender = self._navi_gender()
        if not cache.get("slots") and not overrides:
            self.viewer.set_navi_meshes({})
            self._recompute_navi_ready()
            return
        try:
            found = assets.navi_viewer_assets(cache, overrides, gender)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(
                "Could not resolve Na'vi assets: %s" % exc, 8000
            )
            return
        meshes = {
            k: v for k, v in found.get("meshes", {}).items() if v and os.path.isfile(v)
        }
        self.viewer.set_navi_meshes(meshes)
        # Build the list of texture tasks, then decode + bind them behind the shared progress bar
        # so the viewer only appears once every texture is in (see _load_textures_gated).
        items = []
        for bucket, roles in found.get("textures", {}).items():
            for role, path in roles.items():
                if path and os.path.isfile(path):
                    items.append((bucket, role, path))
        resolved = len(items)

        def _decode(item):
            bucket, role, path = item
            label = "%s / %s" % (bucket, role)
            try:
                return self._decode_navi_texture(path), label
            except Exception as exc:  # noqa: BLE001
                return None, "%s (%s): %s" % (label, os.path.basename(path), exc)

        def _upload(item, arr):
            bucket, role, _path = item
            self.viewer.set_navi_texture(bucket, role, arr)

        bound, decode_fail = self._load_textures_gated(
            "Loading Na'vi textures\u2026", items, _decode, _upload
        )
        # re-apply current colour picks so the freshly built meshes show them straight away
        if hasattr(self, "navi"):
            self.viewer.set_navi_colors(self.navi.state())
            self.navi.refresh_asset_rows()  # populate picker path fields on (re)load
        nmesh = sum(1 for v in meshes.values() if v)
        if nmesh:
            miss = assets.navi_missing_required(cache, overrides, gender)
            msg = "Na'vi preview: %d mesh(es), %d texture(s) loaded." % (nmesh, bound)
            if resolved == 0:
                # meshes resolved but NO texture slot pointed at a real file - the classic
                # stale-cache signature (cache built by an older scan format). Tell the user the
                # exact fix rather than leaving a silently-grey model.
                msg += (
                    '  No texture slots resolved - press "Add Asset Folder" on the Na\'vi '
                    "Assets panel to rebuild the texture cache in the current format."
                )
            elif decode_fail:
                msg += "  %d texture(s) failed to decode." % len(decode_fail)
            if miss:
                msg += "  Missing required: " + ", ".join(miss)
            self.statusBar().showMessage(msg, 12000)
        self._recompute_navi_ready()

    def _on_navi_changed(self, state):
        """A Na'vi picker changed - push the live colour state into the 3D preview."""
        if hasattr(self, "viewer"):
            try:
                self.viewer.set_navi_colors(state)
            except Exception:  # noqa: BLE001
                pass
            self._apply_warpaint_texture(state)
        try:
            skin = state.get("skin", {}).get("colors", ["?"])[0]
            self.statusBar().showMessage("Na'vi: skin #%s" % skin, 3000)
        except Exception:  # noqa: BLE001
            pass

    def _apply_warpaint_texture(self, state):
        """Bind/unbind the warpaint section's own preview textures (the self-contained slots, not
        part of the asset config). Head paints the head bucket; Body 1-4 layer onto the body bucket
        (paint .. paint4). Only changed slots are re-decoded."""
        wp = state.get("warpaint", {}) or {}
        textures = wp.get("textures")
        if not isinstance(textures, dict):
            old = (
                wp.get("texture") or ""
            ).strip()  # back-compat: a single path -> Body 1
            textures = {"body1": old} if old else {}
        target = {
            "head": ("head", "paint"),
            "body1": ("body", "paint"),
            "body2": ("body", "paint2"),
            "body3": ("body", "paint3"),
            "body4": ("body", "paint4"),
        }
        cache = getattr(self, "_warpaint_tex_paths", None)
        if cache is None:
            cache = self._warpaint_tex_paths = {}
        for slot, (bucket, role) in target.items():
            path = (textures.get(slot) or "").strip()
            if path == cache.get(slot, ""):
                continue  # unchanged -> no re-decode
            cache[slot] = path
            if not path:
                self.viewer.clear_navi_texture(bucket, role)
                continue
            try:
                arr = self._decode_navi_texture(path)
            except Exception:  # noqa: BLE001
                self.statusBar().showMessage(
                    "Could not load %s texture: %s" % (slot, os.path.basename(path)),
                    6000,
                )
                continue
            self.viewer.set_navi_texture(bucket, role, arr)
            self.statusBar().showMessage(
                "Warpaint %s: %s" % (slot, os.path.basename(path)), 4000
            )

    def _navi_pick_asset(self, slot):
        """A viewer-section file picker: delegate to the Na'vi Assets panel's per-slot picker
        (which opens the dialog, sets the override, commits + reloads the viewer), then refresh
        the in-viewer picker labels to show the new file."""
        panel = getattr(self, "navi_assets_panel", None)
        if panel is None:
            return
        panel._pick_one(slot, persist=getattr(self, "_remember_changes", False))
        if hasattr(self, "navi"):
            self.navi.refresh_asset_rows()

    def _navi_slot_path(self, slot):
        """Currently-resolved file for a Na'vi slot (override or cache), for picker labels."""
        panel = getattr(self, "navi_assets_panel", None)
        if panel is None:
            return None
        try:
            return panel.current_path(slot)
        except Exception:
            return None

    def _navi_set_asset_path(self, slot, path):
        """A viewer-section path field: set the slot override to a typed path (if it exists),
        commit + reload, then refresh the picker labels. Mirrors the Banshee texture field."""
        panel = getattr(self, "navi_assets_panel", None)
        if panel is None:
            return
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "File not found", "That path doesn't point to a file:\n%s" % path
            )
            if hasattr(self, "navi"):
                self.navi.refresh_asset_rows()  # restore the previous value
            return
        panel.set_override(
            slot, path, persist=getattr(self, "_remember_changes", False)
        )
        if hasattr(self, "navi"):
            self.navi.refresh_asset_rows()

    def _navi_export_texture(self, key, colors, state, fmt="png", out_dir=None):
        """Bake a section's recoloured texture(s) to PNG, reusing the tested CPU recolour
        (recolor_core). Source textures are resolved from the current Na'vi assets. Guarded so a
        bad/missing input shows a message instead of crashing. When ``out_dir`` is given the files
        are written straight there (no dialogs, no per-file popups) and the written paths are
        returned - used by 'Export All as Texture'."""
        import numpy as np
        import recolor_core

        if _pil() is None:
            QMessageBox.warning(
                self, "Missing dependency", "Pillow is required to save a texture."
            )
            return
        from PIL import Image  # local: avoids depending on the widgets module global

        panel = getattr(self, "navi_assets_panel", None)
        if panel is None:
            return

        # Per-section "Export as Texture" (out_dir is None) now batches like Export All: ask once for
        # a destination folder and write every baked map straight into it - no per-file save dialogs.
        # "Export All as Texture" already passes a folder, so it skips this prompt + its own summary.
        interactive = out_dir is None
        if interactive:
            out_dir = QFileDialog.getExistingDirectory(
                self, "Choose a folder for the baked texture(s)"
            )
            if not out_dir:
                return

        def rgb(h):
            h = h.lstrip("#")
            return (
                int(h[0:2], 16) / 255.0,
                int(h[2:4], 16) / 255.0,
                int(h[4:6], 16) / 255.0,
            )

        def src(slot):
            p = panel.current_path(slot)
            return p if (p and os.path.isfile(p)) else None

        def chan(slot, idx, default):
            p = src(slot)
            return (load_rgba(p).astype(np.float32) / 255.0)[..., idx] if p else default

        def tex_rgb(slot):
            p = src(slot)
            if p is None:
                raise FileNotFoundError("missing source texture for '%s'" % slot)
            return (load_rgba(p).astype(np.float32) / 255.0)[..., :3]

        def _fit(arr, w, h):
            """Resize a float map (HxW mask or HxWxC image, 0..1) to (h, w) so source maps decoded
            at different resolutions can be combined - the same resize-to-base step the Banshee bake
            uses. None and scalars pass through unchanged (they broadcast)."""
            if arr is None or np.ndim(arr) < 2:
                return arr
            if arr.shape[0] == h and arr.shape[1] == w:
                return arr
            a8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            if a8.ndim == 2:
                im = Image.fromarray(a8, "L")
            elif a8.shape[2] == 1:
                im = Image.fromarray(a8[..., 0], "L")
            else:
                im = Image.fromarray(a8[..., :3], "RGB")
            return (
                np.asarray(im.resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0
            )

        written_paths = []
        bake_error = None

        def save(rgb01, suggested):
            img8 = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            ext = ".png" if fmt == "png" else ".dds"
            sug = suggested
            for e in (".png", ".dds"):
                if sug.lower().endswith(e):
                    sug = sug[:-4]
            if out_dir is not None:  # batch: write straight to the folder, no dialog/popup
                base = os.path.join(out_dir, sug)
                try:
                    written = save_texture(Image.fromarray(img8, "RGB"), base, fmt)
                except NotImplementedError:
                    return
                written_paths.append(written)
                return
            filt = {
                "png": "PNG image (*.png)",
                "dds": "DDS texture (*.dds)",
                "stf": "STF DDS texture (*.dds)",
            }[fmt]
            out, _ = QFileDialog.getSaveFileName(
                self, "Export %s texture" % key, sug + ext, filt
            )
            if not out:
                return
            base = out
            for e in (".png", ".dds"):
                if base.lower().endswith(e):
                    base = base[:-4]
            try:
                written = save_texture(Image.fromarray(img8, "RGB"), base, fmt)
            except NotImplementedError as exc:
                QMessageBox.information(self, "Export as Texture", str(exc))
                return
            written_paths.append(written)
            QMessageBox.information(
                self, "Exported", "Saved recoloured texture:\n%s" % written
            )

        try:
            if key == "hair":
                c1, c2, c3 = rgb(colors[0]), rgb(colors[1]), rgb(colors[2])
                length_t = chan(
                    "nav_hair_m", 1, 0.5
                )  # HairMaps green = root->tip gradient
                ao = chan("nav_hair_ao", 0, 1.0)
                if np.ndim(length_t) >= 2 and np.ndim(ao) >= 2:
                    ao = _fit(ao, length_t.shape[1], length_t.shape[0])
                save(
                    recolor_core.recolor_hair(c1, c2, c3, length_t, ao=ao),
                    "hair_recoloured.png",
                )
            elif key == "eye":
                iris = tex_rgb("nav_eye_d")
                h, w = iris.shape[:2]
                height = _fit(chan("nav_eye_h", 0, 0.9), w, h)
                # colours are [Right outer, Right inner, Left outer, Left inner]. Bake a single eye
                # texture when both sides match, else a separate right + left texture.
                right = (colors[0], colors[1])
                left = (colors[2], colors[3]) if len(colors) >= 4 else right
                if tuple(c.lower() for c in right) == tuple(c.lower() for c in left):
                    save(
                        recolor_core.recolor_eye(
                            iris, rgb(right[0]), rgb(right[1]), height
                        ),
                        "eye_recoloured.png",
                    )
                else:
                    save(
                        recolor_core.recolor_eye(
                            iris, rgb(right[0]), rgb(right[1]), height
                        ),
                        "eye_right_recoloured.png",
                    )
                    save(
                        recolor_core.recolor_eye(
                            iris, rgb(left[0]), rgb(left[1]), height
                        ),
                        "eye_left_recoloured.png",
                    )
            elif key == "skin":
                g = self._navi_gender()
                skin_c, pat_c = rgb(colors[0]), rgb(colors[1])
                did = False
                for region, alb_slot, pat_slot in (
                    ("head", "nav_head_d_" + g, "nav_head_pat"),
                    ("body", "nav_body_d_" + g, "nav_body_pat"),
                ):
                    if src(alb_slot) is None:
                        continue
                    alb = tex_rgb(alb_slot)
                    h, w = alb.shape[:2]
                    pmask = _fit(
                        chan(pat_slot, 0, None), w, h
                    )  # match pattern mask to the albedo
                    out, _emission = recolor_core.recolor_skin(
                        alb, skin_c, pattern_color=pat_c, pattern_mask=pmask
                    )
                    save(out, region + "_skin_recoloured.png")
                    did = True
                if not did and out_dir is None:
                    QMessageBox.warning(
                        self,
                        "No source texture",
                        "Couldn't find the %s skin albedo to bake. Set it in "
                        "Settings > Na'vi Assets." % g,
                    )
            elif key == "warpaint":
                    # Warpaint isn't a single recolourable texture: each paint _m map (Head + Body 1..4)
                    # encodes per pixel WHICH of the four colours (R, 4-stop via paintSelect) at WHAT
                    # coverage (G). "Export as Texture" bakes one coloured map PER slot with a source set
                    # (like the skin branch, one per region); colours are the section's four swatches,
                    # uncovered pixels stay black.
                paints = [
                    np.asarray(rgb(colors[i]), np.float32)
                    if i < len(colors)
                    else np.zeros(3, np.float32)
                    for i in range(4)
                ]
                c1, c2, c3, c4 = paints
                wp_tex = state.get("textures") or {}

                def _paint_select(sel_r):
                    # mirror gl_shaders.paintSelect: t = R*3, 4-stop lerp c1->c2->c3->c4
                    t = np.clip(sel_r * 3.0, 0.0, 3.0)[..., None]
                    p = np.broadcast_to(c1, sel_r.shape + (3,)).astype(np.float32).copy()
                    w = np.clip(t, 0.0, 1.0)
                    p = p * (1.0 - w) + c2 * w
                    w = np.clip(t - 1.0, 0.0, 1.0)
                    p = p * (1.0 - w) + c3 * w
                    w = np.clip(t - 2.0, 0.0, 1.0)
                    p = p * (1.0 - w) + c4 * w
                    return p

                did = False
                for slot in ("head", "body1", "body2", "body3", "body4"):
                    p = wp_tex.get(slot)
                    if not (p and os.path.isfile(p)):
                        continue
                    m = load_rgba(p).astype(np.float32) / 255.0
                    col = _paint_select(m[..., 0])       # selected paint colour per pixel
                    out = col * m[..., 1][..., None]      # weighted by the G-channel coverage
                    save(out, "warpaint_%s_recoloured.png" % slot)
                    did = True
                if not did and out_dir is None:
                    QMessageBox.warning(
                        self,
                        "No source texture",
                        "Set at least one warpaint _m map (Head or Body 1\u20134) in the "
                        "Warpaint section before baking.",
                    )
            elif out_dir is None:
                QMessageBox.information(
                    self,
                    "Export as Texture",
                    "Texture baking isn't defined for the %s section." % key,
                )
        except Exception as e:  # noqa: BLE001
            bake_error = str(e)
            log.exception("texture bake failed for section %s", key)
        if interactive:
            if written_paths:
                QMessageBox.information(
                    self,
                    "Export as Texture",
                    "Baked %d texture(s) into:\n  %s" % (len(written_paths), out_dir),
                )
            elif bake_error:
                QMessageBox.warning(
                    self,
                    "Export as Texture",
                    "Couldn't bake the %s texture(s):\n%s" % (key, bake_error),
                )
            else:
                QMessageBox.information(
                    self,
                    "Export as Texture",
                    "No textures were written - check that this section's source textures "
                    "are set in Settings > Na'vi Assets.",
                )
        return written_paths

    def _on_navi_assets_changed(self):
        """Called by NaviAssetsPanel after any change (folder / per-slot / clear / reset).
        Re-resolve the viewer meshes and re-gate the Na'vi tab (colour items are loaded
        per-section now, not bulk-scanned). The explicit recompute covers the cases where
        _load_navi_meshes early-returns (e.g. assets just cleared)."""
        log.info("Na'vi assets changed; reloading meshes")
        if hasattr(self, "_loaded_tabs"):
            self._loaded_tabs.add(getattr(self, "_tab_navi", -2))
        self._load_navi_meshes()
        self._recompute_navi_ready()
        if hasattr(self, "navi"):
            self.navi.refresh_asset_rows()

    def _recompute_navi_ready(self):
        """Recompute Na'vi-tab readiness from the cached scan + overrides (mandatory meshes +
        textures, one-of-gender-pair for head/body; colour items excluded) and re-gate the tab.
        Cache lookup only - no disk walk."""
        navi = assets.load_config().get("navi", {})
        cache = navi.get("cache", {}) or {}
        _ap = getattr(self, "navi_assets_panel", None)
        overrides = (
            dict(_ap.overrides) if _ap is not None else (navi.get("paths", {}) or {})
        )
        gender = self._navi_gender() if hasattr(self, "navi_gender") else "m"
        has_assets = bool(cache.get("slots") or overrides)
        if has_assets:
            missing = assets.navi_missing_required(cache, overrides, gender)
        else:
            missing = ["(no assets set)"]
        self._navi_ready = has_assets and not missing
        if hasattr(self, "_tab_settings"):  # all tabs assigned; safe to re-gate
            self._update_tab_states()

    def _change_config_dir(self):
        d = QFileDialog.getExistingDirectory(
            self,
            "Choose a folder to keep config.json in",
            os.path.dirname(assets.config_path()),
        )
        if not d:
            return
        assets.set_config_dir(d)
        if hasattr(self, "cfg_path_edit"):
            self.cfg_path_edit.setText(assets.config_path())
            self.cfg_path_edit.setToolTip(assets.config_path())
        self.statusBar().showMessage("Config moved to: %s" % assets.config_path(), 6000)

    def _open_config_folder(self):
        d = os.path.dirname(assets.config_path())
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def _change_log_dir(self):
        """Pick a different folder for the diagnostic log, persist it, and re-point the live file
        handler there (if logging is on). Mirrors the Config file's Change button."""
        start = os.path.dirname(assets.log_file_path())
        d = QFileDialog.getExistingDirectory(self, "Choose a folder for the log file", start)
        if not d:
            return
        assets.update_config(
            lambda cfg: cfg.setdefault("paths", {}).__setitem__("log_dir", d)
        )
        try:
            assets.configure_logging(assets.logging_enabled())  # reopen the handler at the new path
        except Exception:
            log.exception("re-pointing the log file failed")
        if hasattr(self, "log_path_edit"):
            self.log_path_edit.setText(assets.log_file_path())
            self.log_path_edit.setToolTip(assets.log_file_path())

    def _open_log_file(self):
        p = assets.log_file_path()
        if not os.path.isfile(p):
            self.statusBar().showMessage(
                "No log file yet - enable diagnostic logging first.", 4000
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def _change_preset_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose Save/Load preset folder", assets.preset_dir()
        )
        if not d:
            return
        assets.set_preset_dir(d)
        if hasattr(self, "preset_dir_edit"):
            self.preset_dir_edit.setText(d)
            self.preset_dir_edit.setToolTip(d)
        # refresh both tabs' preset dropdowns so they list the new folder's presets
        for bar in (getattr(self, "banshee_io", None), getattr(self, "navi_io", None)):
            if bar is not None:
                bar._refresh_presets()
        self.statusBar().showMessage("Save/Load preset folder updated.", 4000)

    def _open_preset_folder(self):
        d = assets.preset_dir()
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def _change_export_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Choose your Export Folder", assets.export_folder() or ""
        )
        if not d:
            return
        assets.set_export_folder(d)
        if hasattr(self, "export_dir_edit"):
            self.export_dir_edit.setText(d)
            self.export_dir_edit.setToolTip(d)
        self.statusBar().showMessage("Export Folder updated.", 4000)

    def _open_export_folder(self):
        d = assets.export_folder()
        if not d or not os.path.isdir(d):
            QMessageBox.information(
                self, "Export Folder",
                "No export folder set yet - use Change\u2026 to choose one.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def _open_bug_report(self):
        """Open the project's GitHub issues page in the user's default browser."""
        QDesktopServices.openUrl(
            QUrl("https://github.com/SaintBaron/PandoraPaint/issues")
        )

    def _save_pref_replicate_blue(self, on):
        assets.set_export_replicate_blue(bool(on))
        # the PandoraPaint wrapper only matters while the structure is being replicated
        if hasattr(self, "pref_pandora_folder"):
            self.pref_pandora_folder.setEnabled(bool(on))

    def _save_pref_pandora_folder(self, on):
        assets.set_export_pandora_folder(bool(on))

    def _save_pref_logging(self, on):
        assets.set_logging_enabled(bool(on))
        if on:
            assets.configure_logging(True)  # logs its own "logging enabled" line
        else:
            log.info("logging disabled by user")
            assets.configure_logging(False)

    @staticmethod
    def _set_combo_value(combo, val):
        """Select the option whose data == val, without firing currentIndexChanged."""
        if combo is None:
            return
        idx = combo.findData(val)
        if idx >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _show_first_run_setup(self):
        """First-launch quick-setup dialog. Presents the enable/disable options (with descriptions),
        the AA/AF dropdowns, and the Export Folder picker - all pre-set to their defaults. Save
        applies each choice through its normal Settings handler (config + live apply, so the Settings
        tab reflects it); Cancel applies the defaults instead. Either way the one-time flag is set so
        it won't show again, and the separate export prompt is suppressed (it's merged in here)."""
        from PyQt6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QVBoxLayout,
            QHBoxLayout,
            QCheckBox,
            QLabel,
            QComboBox,
            QLineEdit,
            QPushButton,
            QScrollArea,
            QWidget,
            QFrame,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Welcome to Pandora Paint \u2014 Quick Setup")
        dlg.setMinimumWidth(600)
        # Open tall so most options are visible without scrolling (clamped to the screen).
        _dlg_h = 900
        _dscr = QApplication.primaryScreen()
        if _dscr is not None:
            _dlg_h = min(_dlg_h, _dscr.availableGeometry().height() - 80)
        dlg.resize(620, max(480, _dlg_h))
        outer = QVBoxLayout(dlg)
        intro = QLabel(
            "Pick what you'd like enabled to start with. These are the recommended defaults, so "
            "you can just press Save. Everything here can be changed later in Settings."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # scrollable body (there are a lot of options + descriptions)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setSpacing(5)  # more breathing room between items (they were cramped)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        def _legend(text):
            lb = QLabel(text)
            lb.setObjectName("legend")
            lb.setWordWrap(True)
            lb.setContentsMargins(22, 0, 0, 16)  # indent under its control + a clear gap below
            return lb

        # Pre-fill from the CURRENT settings-tab state (which is the defaults on a fresh install),
        # so the dialog reflects reality and Save/Cancel never overwrites an existing config.
        def _cur_bool(wattr, fallback):
            w = getattr(self, wattr, None)
            return bool(w.isChecked()) if w is not None else bool(fallback)

        boxes = []
        for label, default, wattr, hattr, desc in _FIRST_RUN_BOOLS:
            initial = _cur_bool(wattr, default)
            cb = QCheckBox(label)
            cb.setChecked(initial)  # current value (defaults on a fresh config)
            lay.addWidget(cb)
            lay.addWidget(_legend(desc))
            boxes.append((cb, initial, wattr, hattr))

        def _combo(label_text, options, default_val, desc):
            lay.addWidget(QLabel(label_text))  # the "subtitle" on its own line
            c = QComboBox()
            sel = 0
            for i, (lbl, val) in enumerate(options):
                c.addItem(lbl, val)
                if val == default_val:
                    sel = i
            c.setCurrentIndex(sel)
            crow = QHBoxLayout()
            crow.setContentsMargins(0, 0, 0, 0)
            crow.addWidget(c)
            crow.addStretch(1)  # left-align the dropdown directly under its label
            lay.addLayout(crow)
            lay.addWidget(_legend(desc))
            return c

        _aa_now = (
            self.pref_aa.currentData()
            if getattr(self, "pref_aa", None) is not None
            else "ssaa2"
        )
        _af_now = (
            self.pref_af.currentData()
            if getattr(self, "pref_af", None) is not None
            else 16
        )
        aa = _combo("Anti-aliasing", _AA_OPTIONS, _aa_now, _AA_DESC)
        af = _combo("Texture filtering", _AF_OPTIONS, _af_now, _AF_DESC)

        # merged Export Folder picker (was a separate prompt after setup)
        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Export folder"))
        exp_edit = QLineEdit(assets.export_folder() or "")
        exp_edit.setReadOnly(True)
        exp_edit.setPlaceholderText("(optional - where exports are written; set later in Settings)")
        exp_btn = QPushButton("Choose\u2026")
        exp_btn.setFixedWidth(104)

        def _pick_export():
            d = QFileDialog.getExistingDirectory(
                dlg, "Choose your Export Folder", exp_edit.text() or ""
            )
            if d:
                exp_edit.setText(d)
                exp_edit.setToolTip(d)

        exp_btn.clicked.connect(_pick_export)
        exp_row.addWidget(exp_edit, 1)
        exp_row.addWidget(exp_btn)
        lay.addLayout(exp_row)
        lay.addWidget(
            _legend(
                "Where the Na'vi, Ikran and Gear Camo/Colour exports are written when the per-file "
                "Overwrite tickbox is off. Optional - you can set or change it any time in Settings."
            )
        )
        lay.addWidget(QLabel("Accent colours"))
        _dlg_theme = ThemeColorsPanel(on_changed=self._reapply_theme)
        lay.addWidget(_dlg_theme)
        lay.addStretch(1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        outer.addWidget(btns)

        accepted = dlg.exec() == QDialog.DialogCode.Accepted

        # Apply each setting through its real Settings handler: on Save use the chosen value, on
        # Cancel keep the current/pre-filled value (defaults on a fresh config). Setting the
        # Settings-tab widget too keeps that tab in sync.
        for cb, initial, wattr, hattr in boxes:
            v = bool(cb.isChecked()) if accepted else bool(initial)
            w = getattr(self, wattr, None)
            if w is not None:
                w.blockSignals(True)
                w.setChecked(v)
                w.blockSignals(False)
            handler = getattr(self, hattr, None)
            if handler is not None:
                handler(v)  # writes config + applies live, exactly like the Settings checkbox
        aa_val = aa.currentData() if accepted else _aa_now
        af_val = af.currentData() if accepted else _af_now
        self._set_combo_value(getattr(self, "pref_aa", None), aa_val)
        self._save_pref_aa(aa_val)
        self._set_combo_value(getattr(self, "pref_af", None), af_val)
        self._save_pref_af(af_val)

        # Export Folder (merged in): apply only if one was chosen; leave unset otherwise.
        exp_path = exp_edit.text().strip() if accepted else ""
        if exp_path:
            assets.set_export_folder(exp_path)
            if hasattr(self, "export_dir_edit"):
                self.export_dir_edit.setText(exp_path)
                self.export_dir_edit.setToolTip(exp_path)

        assets.set_setting("first_run_done", True)
        # merged in -> make sure the old standalone export prompt never fires
        assets.set_setting("export_prompted", True)
        log.info(
            "first-run quick-setup applied (saved=%s, export_folder=%s)",
            accepted, bool(exp_path),
        )
        # the dialog's accent picker may have changed the accent - sync the Settings swatches
        if getattr(self, "theme_panel", None) is not None:
            self.theme_panel._refresh()

    def _prompt_first_export_folder(self):
        """First-launch nudge: land the user on Settings and offer to set the Export Folder now."""
        assets.set_setting("export_prompted", True)
        r = QMessageBox.information(
            self,
            "Set an Export Folder",
            "Before exporting, choose an Export Folder - the place your recoloured files are "
            "written. With 'Replicate the blue folder structure' on (the default), each file's "
            "blue/\u2026 path is rebuilt inside it automatically, with no save dialog.\n\n"
            "Choose it now?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if r == QMessageBox.StandardButton.Ok:
            self._change_export_dir()

    def _reset_banshee_assets(self):
        if (
            QMessageBox.question(
                self,
                "Reset Ikran assets",
                "Forget all remembered Ikran asset paths? Your files are not deleted - "
                "you'll just need to point the tool at them again.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        def _upd(cfg):
            cfg["paths"] = {}
            cfg["models"] = []
            cfg["manual_slots"] = []

        assets.update_config(_upd)
        if hasattr(self, "assets_panel"):
            self.assets_panel.reload()
        self._on_assets_changed()
        self.statusBar().showMessage("Ikran asset paths cleared.", 5000)

    def _reset_navi_assets(self):
        if (
            QMessageBox.question(
                self,
                "Reset Na'vi assets",
                "Forget the Na'vi asset folder and all per-slot file overrides? Your files are "
                "not deleted - you'll just need to point the tool at them again.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        assets.update_config(
            lambda cfg: cfg.__setitem__(
                "navi", {"folder": "", "paths": {}, "cache": {}}
            )
        )
        if hasattr(self, "navi_assets_panel"):
            self.navi_assets_panel.reload()
        self._on_navi_assets_changed()
        self.statusBar().showMessage("Na'vi asset paths cleared.", 5000)

    # ---------------- window geometry persistence ----------------
    def _restore_geometry(self):
        try:
            s = assets.load_config().get("settings", {})
            if s.get("remember_geometry", True) and s.get("geometry"):
                self.restoreGeometry(
                    QByteArray.fromBase64(s["geometry"].encode("ascii"))
                )
        except Exception:
            pass

    def _save_geometry(self):
        try:

            def _upd(cfg):
                s = cfg.setdefault("settings", {})
                if s.get("remember_geometry", True):
                    s["geometry"] = bytes(self.saveGeometry().toBase64()).decode(
                        "ascii"
                    )
                else:
                    s.pop("geometry", None)

            assets.update_config(_upd)
        except Exception:
            pass

    def _save_pref_remember_changes(self, on):
        self._remember_changes = bool(on)
        assets.set_setting("remember_changes", bool(on))

    def _save_pref_load_existing(self, on):
        self._load_from_existing_path = bool(on)
        assets.set_setting("smart_dialog_start", bool(on))

    def _save_pref_geometry(self, on):
        def _upd(cfg):
            s = cfg.setdefault("settings", {})
            s["remember_geometry"] = bool(on)
            if not on:
                s.pop("geometry", None)

        assets.update_config(_upd)

    @staticmethod
    def _aa_params(mode):
        """Map an anti-aliasing mode string to (ssaa_scale, fxaa_on)."""
        mode = str(mode)
        if mode == "fxaa":
            return 1.0, True
        if mode.startswith("ssaa"):
            try:
                return float(mode[4:]), False
            except ValueError:
                return 1.0, False
        return 1.0, False  # "off" or anything unrecognised

    def _save_pref_aa(self, mode):
        mode = str(mode)
        assets.set_setting("aa_mode", mode)
        if hasattr(self, "viewer") and self.viewer is not None:
            scale, fxaa = self._aa_params(mode)
            self.viewer.set_ssaa_scale(scale)
            self.viewer.set_fxaa(fxaa)

    def _save_pref_af(self, level):
        level = int(level)
        assets.set_setting("anisotropy", level)
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.set_anisotropy(float(level))

    def _save_pref_specular(self, on):
        assets.set_setting("specular", bool(on))
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.set_specular(bool(on))

    def _save_pref_viewer_side(self, on):
        assets.set_setting("viewer_left", bool(on))
        self._apply_viewer_side()  # flip the Na'vi / Ikran / Camo splits live

    # ---- Camo assets (gearcamo_colorpalettes.rejuice): mirrors the other asset sections ----
    def _build_camo_assets_box(self):
        self.camo_assets_panel = CamoAssetsPanel(
            on_changed=self._on_camo_assets_changed, on_reset=self._reset_camo_assets
        )
        box = QGroupBox("Gear Camo and Colours")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(10, 8, 10, 8)
        bl.setSpacing(6)
        bl.addWidget(self.camo_assets_panel)
        return box

    def _on_camo_assets_changed(self):
        """Called by CamoAssetsPanel after any change: (re)load each rejuice into its editor sub-tab
        and re-gate the Camo tab. Neither rejuice is mandatory - the tab is enabled when either is
        set, and each sub-tab loads only if its own rejuice resolves."""
        log.info("Camo/Colour assets changed; reloading rejuice editors")
        self._reload_camo_editors()
        self._update_camo_tab_enabled()
        paths = (assets.load_config().get("paths", {}) or {})
        any_rej = bool(paths.get("gearcamo_rejuice") or paths.get("gearcolors_rejuice"))
        if any_rej and self.tabs.currentIndex() == getattr(self, "_tab_settings", -1):
            self.statusBar().showMessage(
                "Rejuice loaded - the Camo tab is now enabled.", 6000
            )

    def _reset_camo_assets(self):
        if (
            QMessageBox.question(
                self,
                "Reset Camo assets",
                "Forget the asset folder and both rejuice overrides? Your files are not "
                "deleted - you'll just need to point the tool at them again.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        def _upd(cfg):
            cfg["camo"] = {
                "folder": "", "detected": "", "override": "",
                "colors_detected": "", "colors_override": "",
            }
            paths = cfg.get("paths", {})
            paths.pop("gearcamo_rejuice", None)
            paths.pop("gearcolors_rejuice", None)

        assets.update_config(_upd)
        if hasattr(self, "camo_assets_panel"):
            self.camo_assets_panel.reload()
        self._on_camo_assets_changed()
        self.statusBar().showMessage("Camo asset paths cleared.", 5000)

    def _update_camo_tab_enabled(self):
        """Enable the Camo tab when EITHER rejuice is present (the gear/weapon camo palettes or the
        gear/weapon colour palettes); otherwise grey it out (and step off it if it's current). Each
        sub-tab is separately gated on its own rejuice by _reload_camo_editors."""
        if not hasattr(self, "_tab_camo"):
            return
        paths = (assets.load_config().get("paths", {}) or {})
        camo_rej = paths.get("gearcamo_rejuice") or ""
        color_rej = paths.get("gearcolors_rejuice") or ""
        ok = bool(camo_rej and os.path.isfile(camo_rej)) or bool(
            color_rej and os.path.isfile(color_rej)
        )
        self.tabs.setTabEnabled(self._tab_camo, ok)
        self.tabs.setTabToolTip(
            self._tab_camo,
            "" if ok else "Set a rejuice in Settings (Gear Camo and Colours) first",
        )
        if not ok and self.tabs.currentIndex() == self._tab_camo:
            self.tabs.setCurrentIndex(getattr(self, "_tab_settings", 0))

    def closeEvent(self, e):
        self._save_geometry()
        super().closeEvent(e)

    def _assets_dir(self):
        cfg = assets.load_config()
        for p in cfg.get("paths", {}).values():
            if p and os.path.isdir(os.path.dirname(p)):
                return os.path.dirname(p)
        return ""

    # ---------------- asset setup (button + drag-and-drop) ----------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return
        cfg = assets.load_config(mutable=True)
        pm = dict(cfg.get("paths", {}))
        models = list(cfg.get("models", []))
        n = 0
        for p in paths:
            if os.path.isdir(p):
                s, m = assets.scan_folder(p)
                pm.update(s)
                n += len(s)
                if m:
                    models = m
            elif os.path.isfile(p):
                slot = assets.classify(os.path.basename(p))
                if slot:
                    pm[slot] = p
                    n += 1
                    if slot == "model" and p not in models:
                        models.append(p)
        cfg["paths"] = pm
        cfg["models"] = models
        assets.save_config(cfg)
        self._autoload()
        self.statusBar().showMessage(
            f"Linked {n} file(s)." if n else "No matching assets found.", 6000
        )

    # ---------------- model selection ----------------
    def _load_texture_async(self, key, role, path):
        """Queue a texture decode off-thread; the GL upload runs on the main thread
        once it finishes, and results superseded by a newer request are dropped."""
        self._tex_gen += 1
        self._tex_latest[(key, role)] = self._tex_gen
        self._pool.start(
            _TexDecodeTask(key, role, path, self._tex_gen, self._tex_signals)
        )

    def _on_texture_decoded(self, key, role, gen, arr):
        if self._tex_latest.get((key, role)) != gen:
            return  # a newer request for this slot superseded it
        if arr is None:
            self.statusBar().showMessage(f"Could not decode {key} {role} texture", 5000)
            return
        self.viewer.set_texture(key, role, arr)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open model",
            self._assets_dir(),
            "MMB (*.mmb);;All files (*)",
        )
        if path:
            self._activate_model(path)

    def _model_path_entered(self):
        path = self.model_path_edit.text().strip()
        if not path:
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Not found", f"No file at:\n{path}")
            return
        self._activate_model(path)

    def _activate_model(self, path):
        try:
            self.viewer.load_model(path)
        except Exception as e:
            QMessageBox.warning(self, "Model load failed", str(e))
            return
        self.model_path_edit.setText(path)
        self.statusBar().showMessage(f"Loaded {os.path.basename(path)}", 4000)

    def _banshee_current_tex(self, key, role):
        """The texture path currently shown for a Banshee (key, role), from its panel field."""
        panel = getattr(self, key, None)
        edits = getattr(panel, "tex_edits", None) if panel is not None else None
        if isinstance(edits, dict):
            e = edits.get(role)
            if e is not None:
                t = e.text().strip()
                if t:
                    return t
        return None

    def open_texture(self, key, role, path=None):
        if _pil() is None:
            QMessageBox.warning(
                self, "Missing dependency", "Pillow is required to load textures."
            )
            return None
        if path is None:
            start = self._assets_dir()
            if assets.get_setting("smart_dialog_start", True):
                s = assets.dialog_start_for(self._banshee_current_tex(key, role), "")
                if s:
                    start = s  # file / its folder / a dir path
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"{key} {role} texture",
                start,
                "Images (*.png *.tga *.dds *.jpg)",
            )
            if not path:
                return None
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Not found", f"No file at:\n{path}")
            return None
        try:
            self._load_texture_async(key, role, path)
            if getattr(self, "_remember_changes", False):
                self._persist_banshee_texture(key, role, path)
            self.statusBar().showMessage(
                f"Loading {key} {role}: {os.path.basename(path)}", 5000
            )
            return path
        except Exception as e:
            QMessageBox.warning(self, "Texture load failed", str(e))
            return None

    # banshee viewer (key, role) -> config slot, for persisting live texture swaps when
    # "Remember model and texture changes" is on.
    _BANSHEE_TEX_SLOT = {
        ("body", "color"): "body_color",
        ("body", "material"): "body_material",
        ("body", "pattern"): "body_pattern",
        ("body", "normal"): "body_normal",
        ("head", "color"): "head_color",
        ("head", "material"): "head_material",
        ("head", "pattern"): "head_pattern",
        ("head", "normal"): "head_normal",
        ("wing", "color"): "wing_color",
        ("eye", "color"): "eye_color",
    }

    def _persist_banshee_texture(self, key, role, path):
        """Write a tab-made Ikran texture swap into config so it becomes the default next launch."""
        slot = self._BANSHEE_TEX_SLOT.get((key, role))
        if not slot:
            return
        assets.update_config(
            lambda cfg: cfg.setdefault("paths", {}).__setitem__(slot, path)
        )
        panel = getattr(self, "assets_panel", None)
        if panel is not None:
            try:
                panel.reload()  # mirror the new default in the Settings asset panel
            except Exception:  # noqa: BLE001
                pass

    def _load_pattern_set_entered(self):
        p = self.pset_edit.text().strip()
        if p:
            self._load_pattern_set(path=p)

    def _load_pattern_set(self, checked=False, path=None):
        """Load a .mbansheepatterndata and apply its colour patterns, controls and coats, resolving each member via assets.find_related."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open pattern set",
                self._assets_dir(),
                "Banshee pattern data (*.mbansheepatterndata)",
            )
        if not path:
            return
        try:
            data = BansheePatternData.load(path)
        except Exception as e:
            QMessageBox.warning(self, "Pattern set load failed", str(e))
            return
        self.pset_edit.setText(path)
        self._pset_path, self._pset_data = path, data
        members = data.member_paths()
        panels = {"body": self.body, "head": self.head}
        # The colour patterns + controls are what a successful find needs; the pattern
        # coats (_pc) usually live under blue/baked and may be absent - they're optional.
        applied, missing_req, missing_coat = [], [], []

        def resolve(part, role, bucket):
            ep = members.get((part, role))
            if not ep:
                return None, None
            real = assets.find_related(path, ep)
            if real is None:
                bucket.append(os.path.basename(ep))
            return ep, real

        # 1) colours
        for part in ("body", "head"):
            _ep, real = resolve(part, "color", missing_req)
            if real:
                try:
                    panels[part].set_pattern(
                        ColorPattern.load(real), real, as_default=True
                    )
                    applied.append(f"{part} colours")
                except Exception:
                    missing_req.append(os.path.basename(real))
        # 2) controls (fill the Pattern Control fields)
        for part in ("body", "head"):
            _ep, real = resolve(part, "control", missing_req)
            if real:
                try:
                    panels[part].set_control(PatternControl.load(real), real)
                    applied.append(f"{part} control")
                except Exception:
                    missing_req.append(os.path.basename(real))
        # 3) pattern coats (the _pc this set uses) - optional, never fails the find
        for part in ("body", "head"):
            ep, real = resolve(part, "coat", missing_coat)
            if real:
                try:
                    self._load_texture_async(part, "pattern", real)
                    panels[part].set_texture_path("pattern", real)
                    self._coat_engine[part] = ep  # remember the engine path
                    applied.append(f"{part} coat")
                except Exception:
                    missing_coat.append(os.path.basename(real))

        msg = f"Pattern set '{data.name}': " + (
            ", ".join(applied) if applied else "nothing applied"
        )
        if missing_coat and not missing_req:
            msg += (
                "  -  pattern coats not found (load the _pc textures by hand if needed)"
            )
        self.statusBar().showMessage(msg, 0 if missing_req else 8000)
        self._refresh_export_all()  # manifest now loaded -> pattern-data export enabled
        if missing_req:
            QMessageBox.warning(
                self,
                "Pattern set",
                "Unable to find related files. You will need to manually populate the "
                "data below or choose an .mbansheepatterndata in the correct location.",
            )

    def _palette_changed(self, key, palette, params):
        self.viewer.set_palette(key, palette, params)

    def _copy_from_other(self, key):
        src = self.body if key == "head" else self.head
        dst = self.head if key == "head" else self.body
        if src.cp is None:
            self.statusBar().showMessage(
                "Nothing to copy - the other panel has no pattern.", 4000
            )
            return
        for i in range(10):
            dst.rows[i].set_hex(src.cp.rgb_hex(i), notify=True)
        other = "Body" if key == "head" else "Head"
        self.statusBar().showMessage(
            f"Copied {other} colours into {key.title()}.", 4000
        )

    def _refresh_export_all(self):
        body_loaded, head_loaded = bool(self.body.path), bool(self.head.path)
        ok = self.body.all_valid() and self.head.all_valid()
        ctrl_ready = (
            ok and bool(self.body.control_path) and bool(self.head.control_path)
        )
        pd_ready = ok and bool(self._pset_path)

        self._ov_guard = True
        # master is available once both colour patterns are loaded
        self.ov_master.setEnabled(body_loaded and head_loaded)
        if not self.ov_master.isEnabled() and self.ov_master.isChecked():
            self.ov_master.setChecked(False)
        master_on = self.ov_master.isChecked()
        # per-panel overwrite: master forces them ticked + greyed; otherwise per loaded
        for ov, loaded in (
            (self.body.overwrite, body_loaded),
            (self.head.overwrite, head_loaded),
        ):
            if master_on:
                ov.setChecked(True)
                ov.setEnabled(False)
            else:
                ov.setEnabled(loaded)
                if not loaded and ov.isChecked():
                    ov.setChecked(False)
        # the two extra-export tickboxes: master forces both ticked + greyed
        if master_on:
            for cb in (self.exp_ctrl_cb, self.exp_pd_cb):
                cb.setChecked(True)
                cb.setEnabled(False)
        else:
            self.exp_ctrl_cb.setEnabled(ctrl_ready)
            if not ctrl_ready and self.exp_ctrl_cb.isChecked():
                self.exp_ctrl_cb.setChecked(False)
            self.exp_pd_cb.setEnabled(pd_ready)
            if not pd_ready and self.exp_pd_cb.isChecked():
                self.exp_pd_cb.setChecked(False)
        self._ov_guard = False

        both_loaded = body_loaded and head_loaded
        self.export_all_btn.setEnabled(ok and both_loaded)
        self.export_all_btn.setToolTip(
            self._export_all_tip
            if (ok and both_loaded)
            else (
                "Load both the Body and Head patterns first"
                if not both_loaded
                else "Every Head and Body colour must be a valid 6-digit hex code"
            )
        )
        # Export All as Texture follows the per-panel Export as Texture buttons.
        tex_ready = self.body.export_tex_btn.isEnabled() or self.head.export_tex_btn.isEnabled()
        self.export_all_tex_btn.setEnabled(tex_ready)
        self.export_all_tex_btn.setToolTip(
            self._export_all_tex_tip if tex_ready
            else "Enable at least one panel's Export as Texture first"
        )

    def _on_child_overwrite(self, *_):
        if self._ov_guard:
            return
        self._refresh_export_all()

    def _on_master_overwrite(self, checked):
        if self._ov_guard:
            return
        self._ov_guard = True
        if checked:
            for ov in (self.body.overwrite, self.head.overwrite):
                ov.setChecked(True)
        self._ov_guard = False
        self._refresh_export_all()

    def export_all(self):
        panels = [(self.body, "body"), (self.head, "head")]
        todo = [(p, key) for (p, key) in panels if p.cp is not None]
        want_ctrl = self.exp_ctrl_cb.isChecked()
        want_pd = self.exp_pd_cb.isChecked()
        if not todo and not want_ctrl and not want_pd:
            self.statusBar().showMessage("Nothing to export.", 4000)
            return
        ow = self.ov_master.isChecked() and self.ov_master.isEnabled()
        # Overwrite writes back over the loaded files; otherwise everything is written into a
        # new <name>/blue/gameplay/vanity/juice/ tree. Either way the names, uids and file
        # names are preserved exactly - only colours, control values and the coat paths differ.
        col_over, col_new = [], []
        for p, key in todo:
            (col_over if (p.overwrite.isChecked() and p.path) else col_new).append(
                (p, key)
            )
        ctrl_over = ow
        pd_over = ow
        need_dest = (
            bool(col_new) or (want_ctrl and not ctrl_over) or (want_pd and not pd_over)
        )
        juice = None
        if need_dest:
            base, replicated = resolve_export_base(self)
            if base is None:
                return
            # replicate on -> [PandoraPaint/]blue/gameplay/vanity/juice inside the export folder
            # (the PandoraPaint wrapper follows the Settings toggle); off -> flat in the folder.
            juice = (
                os.path.join(base, export_dir(BLUE_DIR_IKRAN))
                if replicated else base
            )
            try:
                os.makedirs(juice, exist_ok=True)
            except OSError as e:
                QMessageBox.warning(
                    self, "Export failed", f"Could not create folder:\n\n{e}"
                )
                return
        written, notes = [], []
        try:
            for p, key in col_over:
                p.cp.save(p.path)
                written.append(os.path.basename(p.path))
            for p, key in col_new:
                base = (
                    os.path.basename(p.path) if p.path else f"{key}_color.mcolorpattern"
                )
                out = os.path.join(juice, base)
                p.cp.save(out)
                written.append(os.path.basename(out))
            if want_ctrl:
                written += self._write_pattern_controls(juice, ctrl_over)
            if want_pd:
                fn, note = self._write_pattern_data(juice, pd_over)
                if fn:
                    written.append(fn)
                if note:
                    notes.append(note)
        except Exception as e:
            QMessageBox.warning(
                self, "Export failed", f"The export did not complete:\n\n{e}"
            )
            self.statusBar().showMessage("Export failed.", 6000)
            return
        if not written:
            QMessageBox.information(self, "Export", "Nothing was exported.")
            self.statusBar().showMessage("Nothing exported.", 4000)
            return
        body = "Export successful.\n\nWrote:\n  " + "\n  ".join(written)
        if juice:
            body += f"\n\nInto: {juice}"
        if want_pd:
            body += (
                "\n\nCheck that myBodyPatternCoat and myHeadPatternCoat in the "
                ".mbansheepatterndata match the location of your pattern-coat textures "
                "in your blue directory."
            )
        if notes:
            body += "\n\n" + "; ".join(notes)
        QMessageBox.information(self, "Export successful", body)
        self.statusBar().showMessage("Exported: " + ", ".join(written), 8000)
        log.info("Ikran export-all wrote %d file(s): %s", len(written), ", ".join(written))

    def _export_all_textures(self):
        """Bake a recoloured texture for every panel whose Export as Texture is available, into
        one chosen folder - one format prompt, one folder prompt, no per-file dialogs."""
        ready = [p for p in (self.body, self.head) if p.export_tex_btn.isEnabled()]
        if not ready:
            self.statusBar().showMessage("Nothing to export as texture.", 4000)
            return
        fmt = ask_export_format(self)
        if fmt is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the baked textures")
        if not folder:
            return
        written = []
        for p in ready:
            res = p._export_texture(fmt=fmt, out_dir=folder)
            if res:
                written.append(res)
        if not written:
            QMessageBox.information(
                self, "Export as Texture",
                "No textures were written - check that each panel's source textures are loaded.")
            return
        QMessageBox.information(
            self, "Export as Texture",
            "Baked %d texture(s) into:\n  %s" % (len(written), folder))
        log.info("Ikran export-all-textures baked %d into %s", len(written), folder)
        self.statusBar().showMessage(
            "Exported %d texture(s)." % len(written), 6000)

    def _coat_engine_path(self, part, panel):
        """Engine path ('blue/...') for a panel's pattern coat: derive from the loaded _pc path if it sits in a 'blue/' tree, else the loaded set's path, else None."""
        p = panel.tex_edits["pattern"].text().strip().replace("\\", "/")
        low = p.lower()
        i = low.rfind("/blue/")
        if i != -1:
            return p[i + 1 :]
        if low.startswith("blue/"):
            return p
        return self._coat_engine.get(part) or None

    def _write_pattern_controls(self, juice, overwrite):
        """Write Body and Head pattern controls, preserving name/uid/filename (only Level/Invert change). Overwrite writes over the loaded file, else into `juice`."""
        written = []
        for panel, part in ((self.body, "body"), (self.head, "head")):
            if not panel.control_path:
                continue
            ctrl = panel.current_control()  # loaded name/uid + current field values
            out = (
                panel.control_path
                if overwrite
                else os.path.join(juice, os.path.basename(panel.control_path))
            )
            ctrl.save(out)
            written.append(os.path.basename(out))
        return written

    def _write_pattern_data(self, juice, overwrite):
        """Write the loaded .mbansheepatterndata, preserving name/uid/sub-uids/references; only the coat paths are refreshed. Overwrite saves over the loaded file, else into `juice`."""
        if self._pset_data is None or not self._pset_path:
            return None, "pattern data skipped (load a .mbansheepatterndata first)"
        for key, part, panel in (
            ("myBodyPatternCoat", "body", self.body),
            ("myHeadPatternCoat", "head", self.head),
        ):
            ep = self._coat_engine_path(part, panel)
            if ep:
                self._pset_data.coats[key] = ep
        out = (
            self._pset_path
            if overwrite
            else os.path.join(juice, os.path.basename(self._pset_path))
        )
        self._pset_data.save(out)
        return os.path.basename(out), None
