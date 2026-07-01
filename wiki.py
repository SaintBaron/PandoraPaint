"""Item Wiki tab for Pandora Paint.

A read-only reference browser built from item_wiki.json. Three nav levels, matching the app's
new structure: the section tabs (Ikran / Na'vi) sit at the top of the Item Wiki page; under them,
indented, sit the category tabs (Ikran Gear, Ikran Paint / Hair, Gear, Gear Mods, ...); each
category is a searchable table whose columns adapt to that category's fields (gear shows model +
textures, Ikran Paint shows its colour-pattern files).

Data shape (item_wiki.json):
    { "<section>": { "label", "types":[...], "not_included":[...],
                     "items":[ {"type","name","fields":{col: value}}, ... ] } }
Categories listed in `types` but absent from the items (or in `not_included`) render a
placeholder tab so the structure is visible before the data is filled in.
"""

from __future__ import annotations

import json
import os

import assets

from PyQt6.QtCore import Qt, QSize, QRect, QEvent
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QStyle,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

_MUTED = "#8A93A3"
# Columns kept in the data (item_wiki.json) but never rendered. Hidden per request; the
# values stay on disk and remain searchable via _haystack.
_HIDDEN_COLS = {"Camo UID", "Rarity", "Gear Colour UID"}


_BOUNDARY = QColor("#3A4658")  # thicker line between whole items


def _paint_item_boundary(painter, option, index, boundaries):
    """Draw a 2px line along the top edge of a cell that starts a new item."""
    if index.row() in boundaries:
        painter.save()
        pen = QPen(_BOUNDARY)
        pen.setWidth(2)
        painter.setPen(pen)
        y = option.rect.top() + 1
        painter.drawLine(option.rect.left(), y, option.rect.right(), y)
        painter.restore()


class _WrapDelegate(QStyledItemDelegate):
    """Sizes a cell to its actual text: width = the widest line, height = the number of lines.

    With columns then resized to contents (no width cap), every line fits without wrapping, so
    nothing is truncated and rows are exactly tall enough for their explicit line breaks.
    """

    def __init__(self, table):
        super().__init__(table)
        self._table = table
        self._boundaries = set()  # table rows that start a new item -> draw a thicker top line

    def set_boundaries(self, rows):
        self._boundaries = rows

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        _paint_item_boundary(painter, option, index, self._boundaries)

    def sizeHint(self, option, index):
        text = index.data(Qt.ItemDataRole.DisplayRole)
        if not text:
            return super().sizeHint(option, index)
        fm = option.fontMetrics
        lines = str(text).split("\n")
        w = max((fm.horizontalAdvance(ln) for ln in lines), default=0)
        h = fm.height() * len(lines)
        return QSize(w + 18, h + 12)  # padding to match the cell's 8px/5px padding

    # --- read-only editor: lets the user drag-select a portion of the cell text and copy it ---
    _EDIT_QSS = ("background:#11161D; color:#D7DEE8; border:none;"
                 " selection-background-color:#22D3EE; selection-color:#0B0D11;")

    def createEditor(self, parent, option, index):
        multiline = "\n" in str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        if multiline:
            ed = QPlainTextEdit(parent)
            ed.setReadOnly(True)
            ed.setFrameShape(QFrame.Shape.NoFrame)
            ed.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            ed.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        else:
            ed = QLineEdit(parent)
            ed.setReadOnly(True)
            ed.setFrame(False)
        ed.setFont(option.font)
        ed.setStyleSheet(self._EDIT_QSS)
        return ed

    def setEditorData(self, editor, index):
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        if isinstance(editor, QPlainTextEdit):
            editor.setPlainText(text)
        else:
            editor.setText(text)

    def setModelData(self, editor, model, index):
        pass  # read-only — never write the selection back to the cell


_SW = 20   # swatch size (px)
_GAP = 6   # gap between swatches
_PADX = 8  # left padding


def _parse_swatches(text):
    """'ffrrggbb ffrrggbb ...' -> [(QColor, '#RRGGBB'/'#AARRGGBB'), ...]; [] if not swatch data."""
    out = []
    for tok in str(text or "").split():
        h = tok.strip().lower()
        if len(h) != 8 or any(ch not in "0123456789abcdef" for ch in h):
            return []
        a, r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
        label = f"#{r:02X}{g:02X}{b:02X}" if a == 255 else f"#{a:02X}{r:02X}{g:02X}{b:02X}"
        out.append((QColor(r, g, b, a), label))
    return out


class _SwatchDelegate(QStyledItemDelegate):
    """Paints up to four colour boxes in a cell; hovering a box shows that colour's hex."""

    def __init__(self, table):
        super().__init__(table)
        self._boundaries = set()

    def set_boundaries(self, rows):
        self._boundaries = rows

    def paint(self, painter, option, index):
        swatches = _parse_swatches(index.data(Qt.ItemDataRole.DisplayRole))
        if not swatches:
            super().paint(painter, option, index)
            _paint_item_boundary(painter, option, index, self._boundaries)
            return
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        painter.save()
        y = option.rect.y() + (option.rect.height() - _SW) // 2
        x = option.rect.x() + _PADX
        for color, _label in swatches:
            box = QRect(x, y, _SW, _SW)
            painter.fillRect(box, color)
            painter.setPen(QColor("#2A313C"))
            painter.drawRect(box)
            x += _SW + _GAP
        painter.restore()
        _paint_item_boundary(painter, option, index, self._boundaries)

    def sizeHint(self, option, index):
        swatches = _parse_swatches(index.data(Qt.ItemDataRole.DisplayRole))
        if not swatches:
            return super().sizeHint(option, index)
        n = len(swatches)
        return QSize(_PADX * 2 + n * _SW + (n - 1) * _GAP, _SW + 12)

    def helpEvent(self, event, view, option, index):
        if event.type() == QEvent.Type.ToolTip:
            swatches = _parse_swatches(index.data(Qt.ItemDataRole.DisplayRole))
            if swatches:
                rel = event.pos().x() - option.rect.x() - _PADX
                if rel >= 0:
                    i = rel // (_SW + _GAP)
                    if 0 <= i < len(swatches) and rel - i * (_SW + _GAP) <= _SW:
                        QToolTip.showText(event.globalPos(), swatches[i][1], view)
                        return True
            QToolTip.hideText()
            return True
        return super().helpEvent(event, view, option, index)


_ROW_BASE = "#0E1116"   # per-item stripe (even)
_ROW_ALT = "#0F141A"    # per-item stripe (odd)


_TABLE_QSS = """
QTableWidget { background:#0E1116; alternate-background-color:#0F141A; border:none;
               gridline-color:#1A202A; color:#D7DEE8; }
QTableWidget::item { padding:5px 8px; }
QTableWidget::item:selected { background:#10333A; color:#E6EAF1; }
QHeaderView::section { background:#1C212A; color:#A9B6C4; border:none;
                       border-right:1px solid #11151B; padding:6px 8px; font-weight:600; }
QTableCornerButton::section { background:#1C212A; border:none; }
"""


def _data_path():
    return assets.resource_path("item_wiki.json")


def load_wiki_data(path=None):
    """Load item_wiki.json -> dict. Returns an empty-but-valid shell if missing/unreadable so the
    tab still renders rather than crashing the app."""
    try:
        with open(path or _data_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {
            "ikran": {"label": "Ikran", "types": [], "not_included": [], "items": []},
            "navi": {"label": "Na'vi", "types": [], "not_included": [], "items": []},
        }


def _placeholder(text):
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(24, 28, 24, 24)
    lab = QLabel(text)
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lab.setWordWrap(True)
    lab.setStyleSheet(f"color:{_MUTED}; font-size:13px;")
    lay.addStretch(1)
    lay.addWidget(lab)
    lay.addStretch(2)
    return w


def _filename(p):
    """Show just the filename for a stored path, but only strip directories from a genuinely deep
    path (>= 2 separators). Placeholders like 'n/a' have a single slash and must show verbatim."""
    return os.path.basename(p) if p.count("/") >= 2 else p


def _columns_for(items):
    """Field keys present across `items` (first-seen order), with 'UI Name' prepended only when at
    least one item is actually named (so e.g. the Colours tab can lead with 'Part' instead)."""
    cols = []
    for it in items:
        for k in it.get("fields") or {}:
            if k not in cols:
                cols.append(k)
    has_name = any((it.get("name") or "").strip() for it in items)
    lead = (["Type"] if "Type" in cols else []) + (["UI Name"] if has_name else [])
    ordered = lead + [c for c in cols if c != "Type"]
    return [c for c in ordered if c not in _HIDDEN_COLS]


class _CategoryView(QWidget):
    """Search box + adaptive table for one category (all items share a type/field schema)."""

    def __init__(self, items):
        super().__init__()
        self._items = items
        self._cols = _columns_for(items)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(8)

        bar = QHBoxLayout()
        bar.setContentsMargins(2, 0, 2, 0)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search this category\u2026")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        self._count = QLabel()
        self._count.setStyleSheet(f"color:{_MUTED};")
        bar.addWidget(self._search, 1)
        bar.addWidget(self._count, 0)
        lay.addLayout(bar)

        # --- lay out: each multi-value column gets one cell per value down a block of rows;
        # single-value columns span the block. So every asset is its own real, selectable cell. ---
        def split(col, raw):
            if col == "Colours":  # swatch string stays one cell
                return [str(raw)] if str(raw).strip() else [""]
            vals = [ln for ln in str(raw).split("\n") if ln.strip()]
            return vals or [""]

        layouts = []          # per item: (height, {col: [values]})
        self._row_item = []   # table row -> item index (for filter + context)
        total = 0
        for it in items:
            fields = it.get("fields") or {}
            colvals = {}
            h = 1
            for col in self._cols:
                raw = it.get("name", "") if col == "UI Name" else fields.get(col, "")
                colvals[col] = split(col, raw)
                h = max(h, len(colvals[col]))
            layouts.append((h, colvals))
            self._row_item += [len(layouts) - 1] * h
            total += h

        self._table = QTableWidget(total, len(self._cols))
        self._table.setHorizontalHeaderLabels(self._cols)
        self._table.setStyleSheet(_TABLE_QSS)
        self._table.setAlternatingRowColors(False)  # we stripe per item (a block of rows) instead
        self._table.setWordWrap(True)
        self._table.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._context_menu)
        self._table.verticalHeader().setVisible(False)
        self._table.setViewportMargins(0, 0, 0, 16)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        hdr = self._table.horizontalHeader()
        for c in range(len(self._cols)):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)

        def make_item(col, value):
            if col.startswith("."):  # file: show filename, keep full path for Copy path
                cell = QTableWidgetItem(_filename(value) if value else "")
                if value:
                    cell.setData(Qt.ItemDataRole.UserRole, value)
                    cell.setToolTip(value)
            else:
                cell = QTableWidgetItem(value)
                if value and col != "Colours" and len(value) > 28:
                    cell.setToolTip(value)
                if col == "Colours":
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
            cell.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            return cell

        r = 0
        boundaries = set()
        for idx, (h, colvals) in enumerate(layouts):
            if idx:
                boundaries.add(r)  # first row of every item after the first
            bg = QColor(_ROW_BASE if idx % 2 == 0 else _ROW_ALT)
            for c, col in enumerate(self._cols):
                vals = colvals[col]
                n = len(vals)
                # partition the item's block of h rows into n contiguous segments and centre
                # each value in its own segment (1 value -> spans whole block; n == h -> one per
                # row; in between -> evenly split, e.g. 2 meshes beside 11 textures -> two halves)
                base, rem = divmod(h, n)
                off = 0
                for i, v in enumerate(vals):
                    seg = base + (1 if i < rem else 0)
                    cell = make_item(col, v)
                    cell.setBackground(bg)
                    self._table.setItem(r + off, c, cell)
                    if seg > 1:
                        self._table.setSpan(r + off, c, seg, 1)
                    off += seg
            r += h

        # delegates first so column/row auto-sizing uses their per-line measurements
        wrap = _WrapDelegate(self._table)
        wrap.set_boundaries(boundaries)
        self._table.setItemDelegate(wrap)
        if "Colours" in self._cols:
            swatch = _SwatchDelegate(self._table)
            swatch.set_boundaries(boundaries)
            self._table.setItemDelegateForColumn(self._cols.index("Colours"), swatch)
        self._table.resizeColumnsToContents()  # baseline sizing
        # explicit width pass: spanned cells can confuse resizeColumnsToContents, so size each
        # column to its widest actual value so nothing is clipped
        fm = self._table.fontMetrics()
        for c, col in enumerate(self._cols):
            if col == "Colours":
                continue
            widest = 0
            for _h, colvals in layouts:
                for v in colvals[col]:
                    disp = _filename(v) if col.startswith(".") else v
                    if disp:
                        widest = max(widest, fm.horizontalAdvance(disp))
            if widest:
                self._table.setColumnWidth(c, max(self._table.columnWidth(c), widest + 26))
        if ".mcolorpattern" in self._cols:  # a touch of extra breathing room, per request
            ci = self._cols.index(".mcolorpattern")
            self._table.setColumnWidth(ci, self._table.columnWidth(ci) + 48)
        last = len(self._cols) - 1  # trailing space so the last column ends before the window edge
        self._table.setColumnWidth(last, self._table.columnWidth(last) + 18)
        self._table.resizeRowsToContents()
        lay.addWidget(self._table, 1)
        self._update_count(len(items))

    def _context_menu(self, pos):
        item = self._table.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self._table)
        clip = QApplication.clipboard()
        path = item.data(Qt.ItemDataRole.UserRole)
        text = item.text()
        if path:  # file cell: full path is the asset, filename is shown
            menu.addAction("Copy path").triggered.connect(
                lambda *_a: clip.setText(path))
            menu.addAction("Copy filename").triggered.connect(
                lambda *_a: clip.setText(text))
        elif text:
            menu.addAction("Copy").triggered.connect(
                lambda *_a: clip.setText(text))
        if not menu.isEmpty():
            menu.exec(self._table.viewport().mapToGlobal(pos))

    def _haystack(self, it):
        return " ".join(
            [it.get("name", "")] + list((it.get("fields") or {}).values())
        ).lower()

    def _update_count(self, shown):
        total = len(self._items)
        self._count.setText(
            f"{shown} of {total}" if shown != total else f"{total} items"
        )

    def _apply_filter(self, text):
        q = (text or "").strip().lower()
        vis = [not (q and q not in self._haystack(it)) for it in self._items]
        for row, idx in enumerate(self._row_item):
            self._table.setRowHidden(row, not vis[idx])
        self._update_count(sum(vis))


class _LazyCategory(QWidget):
    """Defers building a category's (expensive) table until its tab is first shown, so app start
    only pays for the first visible category instead of every table across both sections."""

    def __init__(self, items):
        super().__init__()
        self._items = items
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self._view = None

    def showEvent(self, ev):
        if self._view is None:
            self._view = _CategoryView(self._items)
            self._lay.addWidget(self._view)
        super().showEvent(ev)


def _section_page(section):
    """A section page: an indented row of category (3rd-layer) tabs under the section tab."""
    by_type = {}
    for it in section.get("items", []):
        by_type.setdefault(it.get("type", ""), []).append(it)
    not_inc = set(section.get("not_included") or [])
    types = section.get("types") or list(by_type.keys())

    cats = QTabWidget()
    cats.setObjectName("wikicats")
    cats.setDocumentMode(True)
    for t in types:
        items = by_type.get(t, [])
        label = t
        if t in not_inc or not items:
            body = _placeholder(
                f"\u201c{t}\u201d isn't populated yet.\nIt'll be added in a later pass."
                if t in not_inc
                else f"No \u201c{t}\u201d items yet."
            )
        else:
            body = _LazyCategory(items)
        cats.addTab(body, label)

    # indent the 3rd-layer tab row so it reads as nested under its section tab; the trailing
    # margin gives the tables breathing room underneath
    page = QWidget()
    pl = QVBoxLayout(page)
    pl.setContentsMargins(18, 18, 18, 24)
    pl.setSpacing(0)
    pl.addWidget(cats)
    return page


def build_sections(data=None):
    """The Item Wiki's section (2nd-layer) tab bar. Renders the known sections in order
    (Ikran / Na'vi / Weapons) plus any other section-shaped top-level key, so adding a new
    section to item_wiki.json shows up here with no code change."""
    data = data if data is not None else load_wiki_data()
    tabs = QTabWidget()
    tabs.setDocumentMode(True)
    order = ["ikran", "navi", "weapons"]
    keys = [k for k in order if k in data]
    for k in data:
        if (
            k not in keys
            and isinstance(data.get(k), dict)
            and ("items" in data[k] or "types" in data[k])
        ):
            keys.append(k)
    for key in keys:
        section = data.get(key) or {"label": key.title(), "items": []}
        tabs.addTab(_section_page(section), section.get("label", key.title()))
    return tabs


class ItemWikiTab(QWidget):
    """Standalone Item Wiki content (section tabs). Kept for direct embedding / testing."""

    def __init__(self, data=None):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.sections = build_sections(data)
        lay.addWidget(self.sections)
