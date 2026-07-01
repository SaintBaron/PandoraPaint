"""Pandora Paint - AFoP Na'vi / banshee skin recolour editor with an OpenGL viewport.

Entry point. The UI widgets and styling live in widgets.py; the main window lives in
main_window.py. This module wires up the QApplication, fonts and stylesheet, then shows it.
"""

from __future__ import annotations
import sys
import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QSurfaceFormat, QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication

from widgets import QSS
from main_window import MainWindow
import assets
import logging
import theme

log = logging.getLogger("pandorapaint.app")


# Resolved at startup to the family name Qt assigns the bundled BebasNeue-Regular face (or None).
BEBAS_FAMILY = None


# App / window icon: a file in the icons/ folder (PNG preferred, then ICO).
_ICON_FILE_NAMES = ("pandora-paint-1024.png", "pandora-paint.png", "pandora-paint.ico")


def _icon_file_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in _ICON_FILE_NAMES:
        p = os.path.join(here, "icons", name)
        if os.path.isfile(p):
            return p
    return None


_ICON_CACHE = None


def app_icon():
    """Return the app QIcon from the branding file in icons/ (PNG preferred, then ICO). Returns an
    empty QIcon if the file is missing (the app simply runs without a custom icon). Imports Qt lazily.
    Cached: the source PNG is large, and this is called several times at startup (window icon + the
    corner logo), so it is built once and reused."""
    global _ICON_CACHE
    if _ICON_CACHE is None:
        from PyQt6.QtGui import QIcon

        path = _icon_file_path()
        _ICON_CACHE = QIcon(path) if path else QIcon()
    return _ICON_CACHE


def main():
    assets.configure_logging()  # honour the saved logging toggle (off by default)
    log.info("Pandora Paint starting")

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    # The app now hosts two QOpenGLWidgets (Banshee + Na'vi viewers). On X11/GLX, creating a
    # second widget's context without sharing can leave moderngl unable to detect a current
    # context in initializeGL ("glXGetCurrentContext: cannot detect OpenGL context"). Sharing
    # contexts across the app is Qt's supported way to run multiple QOpenGLWidgets. Must be set
    # before the QApplication is constructed.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    # Crisp UI at Windows 11 fractional display scaling (125% / 150% / 175%): use the exact scale
    # factor rather than Qt's default rounding, which would otherwise snap 150% up to 2x (oversized
    # UI) or 125% down to 1x (blurry). No effect at the integer scales typical on Linux. Must be set
    # before the QApplication is constructed.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("PandoraPaint")
    # modern UI font with graceful fallbacks (Inter where present, then the platform's
    # newest system UI face). The QSS font-family carries the same preference list.
    _font = QFont()
    _font.setFamilies(
        [
            "Inter",
            "Inter Variable",
            "Segoe UI Variable Text",
            "Segoe UI",
            "Noto Sans",
            "DejaVu Sans",
        ]
    )
    app.setFont(_font)

    # Bundled UI face (SIL Open Font License - OFL.txt ships alongside): Bebas Neue Regular, used
    # for the tab bars and the collapsible Settings section headers. Register it and read back the
    # family name Qt assigns, then point the selectors at it - everything else stays on the Inter
    # stack set above.
    def _load_family(filename):
        try:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "fonts", filename
            )
            if os.path.isfile(path):
                fid = QFontDatabase.addApplicationFont(path)
                fams = QFontDatabase.applicationFontFamilies(fid) if fid != -1 else []
                return fams[0] if fams else None
        except Exception:  # never let a missing font abort startup
            pass
        return None

    # Only Bebas Neue Regular is injected (tab bars + the Settings 'dropdown' section headers);
    # everything else - group-box titles, comboboxes, asset headings, section sub-titles - uses
    # the Inter stack set via app.setFont() above.
    _bebas_regular = _load_family("BebasNeue-Regular.ttf")
    global BEBAS_FAMILY
    BEBAS_FAMILY = (
        _bebas_regular  # exposed so MainWindow's wordmark uses the same registered face
    )

    _extra = ""
    if _bebas_regular:
        _extra += (
            "\nQPushButton#sectiontoggle { font-family:'%s'; }"  # collapsible section headers
            "\nQTabBar::tab { font-family:'%s'; }"  # secondary / wiki tab rows
            "\nQListWidget#mainnav { font-family:'%s'; }"  # left section menu
            "\nQLabel#wordmark { font-family:'%s'; }"  # the 'Pandora Paint' wordmark
            "\nQProgressBar { font-family:'%s'; }"  # loading-bar number / % text
            % (
                _bebas_regular,
                _bebas_regular,
                _bebas_regular,
                _bebas_regular,
                _bebas_regular,
            )
        )
    theme.set_base_stylesheet(
        QSS + _extra
    )  # stashed so the Settings colour panel can re-apply live
    app.setStyleSheet(theme.apply(QSS + _extra))
    app.setWindowIcon(app_icon())

    win = MainWindow()
    win.show()
    log.info("main window shown")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
