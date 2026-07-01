"""Theme accents for Pandora Paint.

Two colours drive the whole UI's accent: the ACTIVE accent (the bright cyan used for selected tabs,
group-box titles, focus, etc.) and the INACTIVE accent (the dim version used for the unselected
secondary-tab underline and the side-tab line/icon). Both live in the config under "theme" and are
substituted into the stylesheets at apply time, so changing them re-themes everything that used the
default hex. The inactive accent defaults to a value DERIVED from the active one (same hue, dimmed),
matching the original #234E57 - it can be overridden manually, or left on auto so it tracks the
active accent.
"""

import colorsys
import re

import assets

# defaults reproduce the current look exactly: substituting these with themselves is a no-op.
DEFAULT_ACTIVE = "#22D3EE"  # bright cyan accent
DEFAULT_INACTIVE = "#234E57"  # dim teal accent (the literal the QSS shipped with)
DEFAULT_HOVER = "#67E8F9"  # accent button :hover (lighter active)
DEFAULT_PRESSED = "#1BA8BE"  # accent button :pressed (darker active)
DEFAULT_TEXT = (
    "#04181E"  # text on accent buttons (auto -> #0B0D11 or #FFFFFF by luminance)
)

# Inactive = active with saturation x _S_SCALE and value x _V_SCALE in HSV (hue preserved). These
# ratios are measured from #22D3EE -> #234E57 (S 0.857->0.598, V 0.933->0.341), so derive(#22D3EE)
# lands on ~#235057 - perceptually identical to #234E57, and the right "dim same-hue" for any accent.
_S_SCALE = 0.698
_V_SCALE = 0.366
_HOVER_S, _HOVER_V = 0.684, 1.046  # active -> #67E8F9 (lighter, slightly desaturated)
_PRESSED_V = 0.799  # active -> #1BA8BE (darker, same saturation)

_HEX_RX = re.compile(r"^#?([0-9a-fA-F]{6})$")
_base_stylesheet = (
    ""  # the app-wide sheet (with default hexes), stashed so it can be re-applied
)


def normalize_hex(value, fallback=DEFAULT_ACTIVE):
    """Return a '#RRGGBB' upper-hex string, or `fallback` if `value` isn't a valid 6-hex colour."""
    m = _HEX_RX.match(str(value or "").strip())
    return ("#" + m.group(1).upper()) if m else fallback


def derive_inactive(active_hex):
    """The default inactive accent for a given active accent: same hue, dimmed + slightly
    desaturated (HSV S x 0.698, V x 0.366). Preserves hue, so any accent yields a matching dim.
    Anchored so the stock cyan returns the exact shipped #234E57 (the formula gives ~#235057, a
    perceptually identical value), keeping the default look pixel-for-pixel unchanged."""
    active = normalize_hex(active_hex)
    if active == DEFAULT_ACTIVE:
        return DEFAULT_INACTIVE
    h = active.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    hh, s, v = colorsys.rgb_to_hsv(r, g, b)
    nr, ng, nb = colorsys.hsv_to_rgb(hh, min(1.0, s * _S_SCALE), min(1.0, v * _V_SCALE))
    return "#%02X%02X%02X" % (round(nr * 255), round(ng * 255), round(nb * 255))


# ---- config-backed accents ----
def _cfg():
    return assets.load_config().get("theme", {}) or {}


def accent_active():
    return normalize_hex(_cfg().get("accent_active"), DEFAULT_ACTIVE)


def accent_inactive():
    raw = _cfg().get("accent_inactive")
    if _HEX_RX.match(str(raw or "").strip()):
        return normalize_hex(raw)
    return derive_inactive(accent_active())


def _scale_hsv(active_hex, s_scale, v_scale):
    h = normalize_hex(active_hex).lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    hh, s, v = colorsys.rgb_to_hsv(r, g, b)
    nr, ng, nb = colorsys.hsv_to_rgb(hh, min(1.0, s * s_scale), min(1.0, v * v_scale))
    return "#%02X%02X%02X" % (round(nr * 255), round(ng * 255), round(nb * 255))


def accent_hover():
    """The accent button :hover shade (lighter active). Anchored to the shipped #67E8F9 at default."""
    a = accent_active()
    return DEFAULT_HOVER if a == DEFAULT_ACTIVE else _scale_hsv(a, _HOVER_S, _HOVER_V)


def accent_pressed():
    """The accent button :pressed shade (darker active). Anchored to the shipped #1BA8BE at default."""
    a = accent_active()
    return DEFAULT_PRESSED if a == DEFAULT_ACTIVE else _scale_hsv(a, 1.0, _PRESSED_V)


def accent_text():
    """Text/foreground colour to sit ON the active accent (accent buttons). Auto-picked from the
    accent's perceived luminance: near-black (#0B0D11) on light/bright accents, white on dark ones,
    so the label stays legible whatever accent is chosen."""
    a = accent_active().lstrip("#")
    r, g, b = (int(a[i : i + 2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b  # sRGB-weighted, 0..255
    return "#0B0D11" if luminance >= 140 else "#FFFFFF"


def set_active(hex_value):
    """Set the active accent AND re-derive the inactive from it (same hue, dimmed). The inactive is
    always recomputed when the active changes; pick the inactive swatch afterwards to override it."""
    hx = normalize_hex(hex_value)
    inactive = derive_inactive(hx)

    def _upd(cfg):
        t = cfg.setdefault("theme", {})
        t["accent_active"] = hx
        t["accent_inactive"] = inactive

    assets.update_config(_upd)


def set_inactive(hex_value):
    """Override just the inactive accent (leaves the active untouched)."""
    hx = normalize_hex(hex_value)

    def _upd(cfg):
        cfg.setdefault("theme", {})["accent_inactive"] = hx

    assets.update_config(_upd)


def reset():
    """Back to defaults: active = DEFAULT_ACTIVE, inactive = auto (derived)."""

    def _upd(cfg):
        cfg.pop("theme", None)

    assets.update_config(_upd)


# ---- applying the accents to stylesheets ----
def apply(qss):
    """Substitute the default accent hexes in a stylesheet with the current accents. QSS strings
    ship with the default hexes, so this is a no-op until the accents are changed."""
    subs = (
        (DEFAULT_ACTIVE, accent_active()),
        (DEFAULT_HOVER, accent_hover()),
        (DEFAULT_PRESSED, accent_pressed()),
        (DEFAULT_TEXT, accent_text()),
        (DEFAULT_INACTIVE, accent_inactive()),
    )
    for token, value in subs:
        qss = re.sub(re.escape(token), value, qss, flags=re.IGNORECASE)
    return qss


def set_base_stylesheet(s):
    """Stash the app-wide sheet (default hexes) so the Settings panel can re-apply it live."""
    global _base_stylesheet
    _base_stylesheet = s or ""


def base_stylesheet():
    """The app-wide sheet with default hexes. If app.py never stashed it (old build / import
    order), fall back to widgets.QSS so the live re-apply still has a sheet to theme - the only
    thing lost in that case is the runtime Bebas font on a couple of headers until restart, never
    the accent colours. Imported lazily to avoid a widgets <-> theme import cycle."""
    if _base_stylesheet:
        return _base_stylesheet
    try:
        import widgets

        return widgets.QSS
    except Exception:  # pragma: no cover - widgets always imports in the running app
        return ""
