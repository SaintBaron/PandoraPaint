"""Reimplementation of px_wildlife_skin_banshee's colour path (10 palette colours ->
recoloured albedo), ported from the compiled .mshader. Sampled at UV0:

  PatternCoat.B/A -> 5-stop gradients across Coat1 (myColor1..5) / Coat2 (myColor6..10)
  PatternCoat.R/G -> Coat1/Coat2 placement masks (Level/Invert)
  coat = lerp(coat1, coat2, saturate(mask1+mask2)); out = Overlay(coat, albedo, Material.B)

The shader's sqrt/overlay/square gamma is replicated; values are sRGB-normalised for display.
"""

from __future__ import annotations
import os
import re
import sys
import json
import glob
import datetime
import numpy as np

GREY = np.array([0.698, 0.686, 0.663], np.float32)  # desaturation target (from shader)


def _sat(x):
    return np.clip(x, 0.0, 1.0)


def _lerp(a, b, t):
    return a + (b - a) * t


def _overlay_channel(base, blend):
    # standard overlay (snowdrop overlay.h)
    return np.where(
        base < 0.5, 2 * base * blend, 1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    )


def recolor(
    color,
    material,
    patterncoat,
    palette,
    invert1=None,
    invert2=1.0,
    level1=1.0,
    level2=1.0,
    desat=(0.0, 0.0),
):
    """color/material/patterncoat: HxWx(3|4) float arrays in [0,1] (sampled at UV0).
    palette: 10 (r,g,b) floats in [0,1] (myColor1..10). invert1: 1.0 (neutral) if None.
    Returns recoloured albedo HxWx3 in [0,1]."""
    pc = patterncoat.astype(np.float32)
    R, G, B, A = pc[..., 0], pc[..., 1], pc[..., 2], pc[..., 3]
    C = [np.asarray(c, np.float32) for c in palette]
    if invert1 is None:
        invert1 = 1.0

    # ---- Coat 1 gradient from B ----
    t = B[..., None] * 4.0
    c1 = _lerp(C[0], C[1], _sat(t - 0.0))
    c1 = _lerp(c1, C[2], _sat(t - 1.0))
    c1 = _lerp(c1, C[3], _sat(t - 2.0))
    c1 = _lerp(c1, C[4], _sat(t - 3.0))
    # ---- Coat 2 gradient from A ----
    t2 = A[..., None] * 4.0
    c2 = _lerp(C[5], C[6], _sat(t2 - 0.0))
    c2 = _lerp(c2, C[7], _sat(t2 - 1.0))
    c2 = _lerp(c2, C[8], _sat(t2 - 2.0))
    c2 = _lerp(c2, C[9], _sat(t2 - 3.0))

    # ---- pattern placement masks ----
    hi1 = level1 * 0.25
    m1 = _smoothstep(hi1 - 0.25, hi1, R) * invert1 - min(0.0, invert1)
    hi2 = level2 * 0.25
    m2 = _smoothstep(hi2 - 0.25, hi2, G) * invert2 - min(0.0, invert2)
    mask = _sat(m1 + m2)[..., None]

    coat = _lerp(c1, c2, mask)
    coat = np.sqrt(np.clip(coat, 0, None))

    # ---- albedo path (detail texture, optional desaturation) ----
    alb = color[..., :3].astype(np.float32)
    ds = _sat(desat[1] + 1.0)
    alb = _lerp(GREY, alb, ds)
    alb = np.sqrt(np.clip(alb, 0, None))

    # ---- overlay coat onto albedo, masked by Material.B ----
    om = _sat(material[..., 2] + desat[0])[..., None]
    blended = _overlay_channel(alb, coat)  # base=albedo, blend=coat
    out = _lerp(alb, blended, om)
    out = out * out  # undo sqrt (shader pow 2)
    return _sat(out)


def _smoothstep(e0, e1, x):
    t = _sat((x - e0) / np.maximum(e1 - e0, 1e-6))
    return t * t * (3.0 - 2.0 * t)


def palette_from_pattern(cp):
    """Build the 10 (r,g,b) palette from a ColorPattern object."""
    pal = []
    for v in cp.colors:
        pal.append(
            ((v >> 16 & 0xFF) / 255.0, (v >> 8 & 0xFF) / 255.0, (v & 0xFF) / 255.0)
        )
    return pal


# =====================================================================================
# Na'vi (player) recolour - CPU reference, merged in from the former navi_recolor.py.
# Decoded from the AFoP player shaders (px_character_navi_face / px_character_skin_player /
# px_eye2 / px_hair2_3color_tousle). The matching GLSL lives in gl_shaders.NAVI_GLSL and is
# kept in lock-step with these functions.
# Public: BIO_GREEN, hex_to_rgb01, srgb_to_linear, linear_to_srgb, lerp, smoothstep,
#         overlay_blend, overlay, overlay_color_base, multi_lerp, recolor_skin,
#         eye_height_blend, recolor_eye, hair_gradient, recolor_hair.
# (banshee recolor()/palette_from_pattern() above are a separate, unrelated colour path.)
# =====================================================================================


# the characteristic Na'vi bioluminescence green the face/skin shaders lerp toward
BIO_GREEN = np.array([0.467784, 0.930111, 0.508881], dtype=np.float32)


# ----------------------------------------------------------------------------- colour utils
def hex_to_rgb01(hexstr):
    """'#rrggbb' or 'rrggbb' -> float32 (3,) in 0..1 (sRGB)."""
    h = hexstr.lstrip("#")
    return np.array(
        [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32
    )


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


def linear_to_srgb(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(
        c <= 0.0031308,
        c * 12.92,
        1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055,
    ).astype(np.float32)


def _c(x):
    """Coerce a colour (hex str / sequence) to float32 (3,)."""
    if isinstance(x, str):
        return hex_to_rgb01(x)
    return np.asarray(x, dtype=np.float32)[:3]


def _mask(m, ref):
    """Broadcast a mask to blend against `ref` (H,W,3): scalars pass through, (H,W) -> (H,W,1)."""
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == ref.ndim - 1:
        return m[..., None]
    return m


def lerp(a, b, t):
    """a + (b-a)*t, with t a scalar or per-pixel mask."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    t = (
        _mask(t, a)
        if (np.ndim(t) and np.ndim(t) == a.ndim - 1)
        else np.asarray(t, np.float32)
    )
    return (a + (b - a) * t).astype(np.float32)


def smoothstep(lo, hi, x):
    x = np.asarray(x, dtype=np.float32)
    t = np.clip((x - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


# ----------------------------------------------------------------------------- blends
def overlay_blend(base, blend):
    """Photoshop overlay, element-wise. overlay(b, 0.5) == b (50% grey is identity)."""
    base = np.asarray(base, dtype=np.float32)
    blend = np.asarray(blend, dtype=np.float32)
    lo = 2.0 * base * blend
    hi = 1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    return np.where(base < 0.5, lo, hi).astype(np.float32)


def overlay(base, tint, mask=1.0):
    """lerp(base, overlay_blend(base, tint), mask) - tint `base` by `tint` where `mask` is high.
    `base` is (H,W,3); `tint` a (3,) colour; `mask` a scalar or (H,W)."""
    base = np.asarray(base, dtype=np.float32)
    blended = overlay_blend(base, _c(tint))
    m = _mask(mask, base)
    return (base * (1.0 - m) + blended * m).astype(np.float32)


def overlay_color_base(color, image, mask=1.0):
    """lerp(color, overlay_blend(color, image), mask) - the COLOUR is the overlay base and the
    image is the blend, matching the real shader's `Overlay(colorParam, texture, mask)` call
    convention (confirmed against px_character_navi(_face).mshader). This is NOT the same as
    overlay() with swapped arguments: overlay() assumes its first arg is the (H,W,3) image and
    its second a (3,) colour, so the roles can't just be swapped at the call site - this version
    handles a (3,) base broadcasting against an (H,W,3) image correctly.

    `color` may also be a per-pixel (H,W,3) colour map (e.g. the eye's inner/outer hue selected by
    a 2-D height map); only a flat colour is coerced to (3,), an image-shaped colour passes through.
    """
    color = (
        _c(color)
        if isinstance(color, str) or np.ndim(color) < 2
        else np.asarray(color, dtype=np.float32)
    )
    image = np.asarray(image, dtype=np.float32)
    blended = overlay_blend(color, image)
    m = _mask(mask, image)
    return (color * (1.0 - m) + blended * m).astype(np.float32)


def multi_lerp(c1, c2, c3, c4, sel):
    """Warpaint 4-stop selector driven by a SINGLE channel (PaintTexture.R), decoded from
    px_character_skin_player / px_character_navi_face:
        t = sel.R * 3 ; p=c1 ; p=lerp(p,c2,sat(t)) ; p=lerp(p,c3,sat(t-1)) ; p=lerp(p,c4,sat(t-2))
    so R = 0, 1/3, 2/3, 1 select c1, c2, c3, c4. `sel` is (...,3) (the paint texture rgb); only
    its R channel matters. Returns (...,3)."""
    c1, c2, c3, c4 = _c(c1), _c(c2), _c(c3), _c(c4)
    sel = np.asarray(sel, dtype=np.float32)
    t = sel[..., 0:1] * 3.0
    p = np.broadcast_to(c1, sel.shape).astype(np.float32).copy()
    p = p + (c2 - p) * np.clip(t, 0.0, 1.0)
    p = p + (c3 - p) * np.clip(t - 1.0, 0.0, 1.0)
    p = p + (c4 - p) * np.clip(t - 2.0, 0.0, 1.0)
    return p.astype(np.float32)


# ----------------------------------------------------------------------------- skin / face
def recolor_skin(
    albedo,
    skin_color,
    skin_mask=1.0,
    pattern_color=None,
    pattern_mask=None,
    paint_color=None,
    paint_coverage=None,
    bio_mask=None,
    bio_color=BIO_GREEN,
    bio_brightness=1.0,
    bio_pulsation=0.0,
    time=0.0,
    hair_cap_color=None,
    hair_cap_mask=None,
    color_weight=1.0,
):
    """Apply the full skin/face recolour chain. `albedo` is (H,W,3|4) in 0..1 sRGB.
    Returns (albedo_out (H,W,3) sRGB, emission (H,W,3) or None).

    The colour chain runs in LINEAR light (mirrors the GLSL recolorSkin + skin tail): the sRGB
    albedo and every sRGB colour swatch are linearised in, overlay/lerp happen in linear, and the
    result is encoded back to sRGB. This is what makes saturated blues stay blue (not magenta) and
    keeps the surface from over-brightening. Emission is returned in the swatch's own (sRGB) space,
    unchanged, since the baker composites the glow separately.

    color_weight : 0 = saturation-gated hybrid (lean to texture-base for near-primary picks),
                   1 = pure game colour-base order (default; most faithful).
    skin_mask    : where the skin tint applies (shader uses Material.g)
    pattern_mask : skin-pattern coverage (from PatternTexture)
    paint_color  : the already-selected warpaint colour (see multi_lerp); coverage = paint_coverage
    bio_mask     : bioluminescence coverage (from Bioluminescence texture)
    hair_cap_*   : scalp tint (face shader only)
    """
    a_srgb = np.asarray(albedo, dtype=np.float32)[..., :3]
    c = srgb_to_linear(a_srgb).copy()  # work in LINEAR
    # SKIN colour - EXACT px_character_navi order: Overlay(blend=skinColor, base=diffuse, Material.G^2)
    # = lerp(diffuse, overlay_blend(base=diffuse, blend=skinColor), mask). overlay_blend branches on
    # its FIRST arg (the base), so the diffuse must be first - branching on the colour floods saturated
    # channels (the "too blue" bug). No saturation hybrid exists in the game; color_weight is unused.
    sk = srgb_to_linear(_c(skin_color))
    sk_b = np.broadcast_to(sk, c.shape)
    recol = overlay_blend(c, sk_b.copy())  # base = diffuse, blend = skin colour
    m = _mask(skin_mask, c)
    c = (c * (1.0 - m) + recol * m).astype(np.float32)  # skin tint
    if pattern_color is not None and pattern_mask is not None:
        c = overlay(c, srgb_to_linear(_c(pattern_color)), pattern_mask)  # skin pattern
    if bio_mask is not None:
        c = lerp(c, srgb_to_linear(_c(bio_color)), bio_mask)  # bio green tint
    if paint_color is not None and paint_coverage is not None:
        pc = paint_color if np.ndim(paint_color) >= c.ndim else _c(paint_color)
        c = lerp(c, srgb_to_linear(pc), paint_coverage)  # warpaint
    if hair_cap_color is not None and hair_cap_mask is not None:
        c = lerp(c, srgb_to_linear(_c(hair_cap_color)), hair_cap_mask)  # scalp cap
    c = linear_to_srgb(np.clip(c, 0.0, 1.0))  # encode -> display
    c = np.clip(c, 0.0, 1.0)

    emission = None
    if bio_mask is not None:
        puls = 1.0 + (abs(_triangle(time)) - 1.0) * float(
            bio_pulsation
        )  # lerp(1, |tri|, puls)
        e = _mask(bio_mask, c) * (float(bio_brightness) * puls)
        if hair_cap_mask is not None:
            e = e * (1.0 - _mask(hair_cap_mask, c))  # scalp doesn't glow
        emission = np.clip(e * _c(bio_color), 0.0, None).astype(np.float32)
    return c, emission


def _triangle(t):
    """lerp(-1, 1, frac(t)) -> a -1..1 ramp, matching the shader's time-based pulsation."""
    return -1.0 + 2.0 * (float(t) - np.floor(float(t)))

    # ----------------------------------------------------------------------------- eye
    # Per the real px_eye2.mshader: the inner/outer iris hue is driven by the iris HEIGHT map
    # (p_eyes_01_h.dds) through a parallax node, NOT a UV-radial gradient. We use the raw height value
    # and skip the parallax UV offset (shifts where you sample, not the blend shape) - a documented
    # simplification for a static preview. Two values derive from the same "remap231" term:
    #   mix    : 0(outer)..1(inner) hue selector
    #   ovmask : strength of Overlay(hue, irisTexture) - ~1 almost everywhere, ->0 only where
    #            height >= 0.95 (a narrow peak band). "Outer iris" affecting only that band is faithful
    #            to the real shader, so expect it to stay subtle.


def eye_height_blend(height, spread=0.7, height_lo=0.85, height_hi=0.99, blend=1.2):
    """Inner/outer hue selector (mix) and overlay mask from the RAW iris height map.

    Matches the corrected GLSL EYE branch. `mask` (overlay strength) = 1 - smoothstep(lo, hi, h):
    full on the low-height iris detail, fading to 0 on the high-height flat white surround (so the
    hue never colours the white outer eye). The inner/outer `mix` is driven by the same raw height
    shaped by `spread`, with `blend` setting the transition steepness (lower = softer, more gradual
    blend between the two hues), then eased with a smoothstep so the seam isn't hard."""
    height = np.asarray(height, dtype=np.float32)
    t = np.clip((height - height_lo) / max(1e-6, height_hi - height_lo), 0.0, 1.0)
    mask = 1.0 - (t * t * (3.0 - 2.0 * t))  # 1 on iris, 0 on flat white
    hs = np.power(np.clip(height, 0.0, 1.0), spread)
    mix = np.clip((1.0 - hs) * blend, 0.0, 1.0)
    mix = mix * mix * (3.0 - 2.0 * mix)  # ease (smoothstep), matches GLSL
    return mix, mask


def _box_blur(a, radius):
    """Separable edge-clamped box (mean) blur of a 2-D map. Mip-blurs the iris HEIGHT so the
    inner/outer hue split reads as a smooth radial gradient (matching the viewer's uEyeHeightBlur LOD
    bias) instead of following the raw per-fibre height noise."""
    a = np.asarray(a, dtype=np.float32)
    r = int(radius)
    if r < 1 or a.ndim != 2:
        return a
    w = 2 * r + 1

    def _ma(x):  # moving average along axis 1; separable -> apply to rows, then columns
        xp = np.pad(x, ((0, 0), (r, r)), mode="edge")
        c = np.cumsum(xp, axis=1, dtype=np.float32)
        c = np.concatenate([np.zeros((x.shape[0], 1), np.float32), c], axis=1)
        return (c[:, w:] - c[:, :-w]) / np.float32(w)

    out = _ma(a)
    out = _ma(np.ascontiguousarray(out.T)).T
    return np.ascontiguousarray(out, dtype=np.float32)


def recolor_eye(
    iris_diffuse,
    outer_hue,
    inner_hue,
    height,
    mask=None,
    height_lo=0.95,
    height_hi=1.0,
    select_gain=1.8,
    height_blur=True,
    **_legacy,
):
    """Recolour the iris diffuse to match the live viewer's px_eye2 GLSL branch EXACTLY.

    The shader chain, in LINEAR light, is:
        h       = height            (mip-blurred to average out fibre noise; uEyeHeightBlur)
        mask    = saturate(1 - (h - 0.95) / 0.05)     # 1 on the iris, 0 on the flat white sclera
        selectT = saturate((1 - h) * mask * 1.8)      # iris centre -> inner hue, rim -> outer hue
        hue     = lerp(outerHue, innerHue, selectT)
        recol   = Overlay(base = iris, blend = hue)   # branch on the IRIS so its fibre detail shows
        albedo  = lerp(iris, recol, mask)

    The iris diffuse and both hue swatches are sRGB, so they are linearised in and the result is
    re-encoded to sRGB on the way out, exactly like recolor_skin. The previous version branched the
    overlay on the COLOUR and layered on contrast-normalisation / adaptive-opacity / radial-falloff
    heuristics that are NOT in the shader, so its bake diverged from the preview - hence this replace.
    `**_legacy` harmlessly swallows the old heuristic kwargs (spread/detail/opacity/uv_radius/...)."""
    iris = srgb_to_linear(np.asarray(iris_diffuse, dtype=np.float32)[..., :3])
    h = np.asarray(height, dtype=np.float32)
    if height_blur and h.ndim == 2:
        # ~ uEyeHeightBlur LOD 3: average an ~8x8 footprint (resolution-independent)
        h = _box_blur(h, max(2, int(round(max(h.shape) / 128))))
    denom = max(float(height_hi) - float(height_lo), 1e-6)
    m = np.clip(1.0 - (h - float(height_lo)) / denom, 0.0, 1.0)
    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
    select_t = np.clip((1.0 - h) * m * float(select_gain), 0.0, 1.0)
    outer_lin = srgb_to_linear(_c(outer_hue))
    inner_lin = srgb_to_linear(_c(inner_hue))
    sel = select_t[..., None] if np.ndim(select_t) >= 2 else select_t
    hue = outer_lin + (inner_lin - outer_lin) * sel  # (H,W,3) or (3,), LINEAR
    recol = overlay_blend(iris, hue)  # base = iris, blend = hue (LINEAR)
    cov = m[..., None] if np.ndim(m) >= 2 else m
    albedo = iris + (recol - iris) * cov  # LINEAR
    return linear_to_srgb(np.clip(albedo, 0.0, 1.0)).astype(np.float32)


# ----------------------------------------------------------------------------- hair
def hair_gradient(c1, c2, c3, length_t, smoothness=0.0):
    """3-stop root->mid->tip gradient, decoded from px_hair2_3color_tousle:
        r1 = saturate((t - lo1)/(hi1 - lo1)) ; r2 = saturate((t - lo2)/(hi2 - lo2))
        colour = c1 + (c2-c1)*r1 + (c3-c2)*r2          # additive 3-weight blend
    Stops ~0.33 / ~0.665, widened by `smoothness` (0..1). Ramps are LINEAR (not smoothstep),
    matching the shader. `length_t` is (H,W) along-strand 0(root)..1(tip) (HairMaps.G in-game).
    Returns (H,W,3)."""
    c1, c2, c3 = _c(c1), _c(c2), _c(c3)
    lo1 = lerp(0.33, 0.0, smoothness)
    hi1 = lerp(0.34, 0.5, smoothness)
    lo2 = lerp(0.665, 0.5, smoothness)
    hi2 = lerp(0.666, 1.0, smoothness)
    t = np.asarray(length_t, dtype=np.float32)
    r1 = np.clip((t - float(lo1)) / max(float(hi1 - lo1), 1e-6), 0.0, 1.0)[..., None]
    r2 = np.clip((t - float(lo2)) / max(float(hi2 - lo2), 1e-6), 0.0, 1.0)[..., None]
    return (c1 + (c2 - c1) * r1 + (c3 - c2) * r2).astype(np.float32)


def recolor_hair(c1, c2, c3, length_t, ao=1.0, smoothness=0.0, root_darkening=0.0):
    """Hair colour = 3-stop gradient * root darkening * ao/mask."""
    col = hair_gradient(c1, c2, c3, length_t, smoothness)
    ao = np.asarray(ao, dtype=np.float32)
    rd = 1.0 + (ao - 1.0) * float(root_darkening)  # lerp(1, ao, root_darkening)
    col = col * _mask(rd, col) * _mask(ao, col)
    return np.clip(col, 0.0, 1.0).astype(np.float32)


# =====================================================================================
# Na'vi customization colour scanner - merged in from the former navi_colors.py.
# Parses AFoP `*.blueitemtype` BlueItemTypeCustomizationFeature items (skin / warpaint /
# eye / hair colour swatches) into the UI palette consumed by the Na'vi tab + Load buttons.
# Pure text parsing (no numpy); kept here so all colour logic lives in recolor_core.
# Public: parse_blueitemtype, parse_text, scan_folder, select_players,
#         to_palette, summarize, to_markdown, write_report, write_palette_json.
# Also runnable as a CLI: `python3 recolor_core.py <folder-or-files...> [--out FILE.md] ...`
# =====================================================================================

# slot -> (category, [(color_index, role, shader_param), ...]) ; index 1-based
SLOT_ROLES = {
    "CustomizationSkinColor": (
        "skin",
        [
            (1, "base skin", "myBaseColorOverlay"),
            (2, "skin pattern", "myPatternColor"),
            (3, "unused", ""),
            (4, "unused", ""),
        ],
    ),
    "CustomizationWarpaintColor": (
        "warpaint",
        [
            (1, "paint 1", "myPaintColor1"),
            (2, "paint 2", "myPaintColor2"),
            (3, "paint 3", "myPaintColor3"),
            (4, "paint 4", "myPaintColor4"),
        ],
    ),
    "CustomizationEyeColor": (
        "eye",
        [
            (1, "right outer", "myOuterIrisHue"),
            (2, "right inner", "myInnerIrisHue"),
            (3, "left outer", "myOuterIrisHue"),
            (4, "left inner", "myInnerIrisHue"),
        ],
    ),
    "CustomizationHairColor": (
        "hair",
        [
            (1, "root", "myHairColor1"),
            (2, "mid", "myHairColor2"),
            (3, "tip", "myHairColor3"),
            (
                4,
                "hair cap",
                "myHairCapColor",
            ),  # scalp/cap tint; 0x000000 = unset -> inherit the root
        ],
    ),
}
CATEGORY_ORDER = [
    "skin",
    "warpaint",
    "eye",
    "hair",
    "haircap",
    "pattern",
    "texture",
    "other",
]

# Texture-customization slots: the chosen texture arrives via myTextureData (myHeadTexture /
# myBodyTexture), NOT myColorData, so these items have no colour roles. slot -> (category, role).
# Confirmed from real items: CustomizationHairCapTexture points the head at a hairstyle's
# *_mask.dds (scalp cap). CustomizationSkinPatternTexture is documented in the customization notes.
SLOT_TEXTURES = {
    "CustomizationHairCapTexture": ("haircap", "haircap"),
    "CustomizationSkinPatternTexture": ("pattern", "pattern"),
}
_TEX_BUCKET = {"myHeadTexture": "head", "myBodyTexture": "body"}


# ----------------------------------------------------------------------------- colour
def parse_argb(token):
    """'0xffc5e4f1' (AARRGGBB) -> structured colour, or None if not a colour token."""
    t = token.strip().lower()
    if not (t.startswith("0x") and len(t) == 10):
        return None
    try:
        v = int(t, 16)
    except ValueError:
        return None
    a, r, g, b = (v >> 24) & 255, (v >> 16) & 255, (v >> 8) & 255, v & 255
    return {
        "argb": token.strip(),
        "a": a,
        "r": r,
        "g": g,
        "b": b,
        "hex": "#%02x%02x%02x" % (r, g, b),
        "rgb255": [r, g, b],
        "rgb01": [round(r / 255.0, 6), round(g / 255.0, 6), round(b / 255.0, 6)],
        "is_black": (r == 0 and g == 0 and b == 0),
    }


# ----------------------------------------------------------------------------- tagging
# hair dyes ship as a base plus per-biome variants; collapse them by stripping the suffix
_REGION_SUFFIX = {"_rnf": "Rainforest", "_tmp": "Temperate", "_vlt": "Veldt"}


def _region_and_dye(name, explicit_region):
    """Return (dye_id, region): strip a trailing _rnf/_tmp/_vlt so variants of one dye share
    a dye_id; fill region from the suffix when myCustomizationData didn't supply one."""
    dye_id, region = name, explicit_region
    if name:
        for suf, reg in _REGION_SUFFIX.items():
            if name.endswith(suf):
                dye_id = name[: -len(suf)]
                region = region or reg
                break
    return dye_id, region


def _source_tag(name):
    """Best-effort audience tag from the item name (matters mostly for warpaint)."""
    n = (name or "").lower()
    if n.startswith("item_mtx_"):
        return "named"  # iconic-character paints (jake/neytiri/toruk/tsutey)
    if "npc" in n:
        return "npc"  # enemy weapon-skin paints - not player selectable
    if "dlc" in n:
        return "dlc"
    if "mtx" in n:
        return "mtx"
    if "bodypaint" in n:
        return "bodypaint"
    if "quest" in n:
        return "quest"
    return "player"


# ----------------------------------------------------------------------------- parser
def _parse_body(lines, idx):
    """Recursively parse a { ... } block (one key/value per line; '{' and '}' on own lines).
    Returns (dict, next_index). `idx` points at the line just after the opening '{'."""
    out = {}
    pending = None
    while idx < len(lines):
        raw = lines[idx].strip()
        idx += 1
        if raw == "" or raw.startswith("include "):
            continue
        if raw == "}":
            return out, idx
        if raw == "{":
            block, idx = _parse_body(lines, idx)
            if pending is not None:
                out[pending] = block
                pending = None
            continue
        # key [value...]
        parts = raw.split(None, 1)
        key = parts[0]
        if len(parts) == 1:
            pending = key  # a block header; its '{' is on the next line
            out.setdefault(key, None)
        else:
            out[key] = _clean_value(parts[1])
            pending = None
    return out, idx


def _clean_value(v):
    """Strip a quoted string (and unescape), else return the raw token string."""
    v = v.strip()
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        return v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return v


_HEADER_RE = re.compile(
    r"BlueItemTypeCustomizationFeature\s+(\S+)\s*<\s*uid=([0-9A-Fa-f]+)\s*>"
)
_TEXT_RE = re.compile(r'text\s*=\s*"((?:[^"\\]|\\.)*)"')
_CTX_RE = re.compile(r'contextComment\s*=\s*"((?:[^"\\]|\\.)*)"')
_BASE_RE = re.compile(r"<\s*uid=([0-9A-Fa-f]+)\s*>\s*=\s*(\S+)\s+([0-9A-Fa-f]+)")


def parse_blueitemtype(path):
    """Parse one .blueitemtype customization item (read from disk)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return parse_text(fh.read(), path)


_COLOR_LINE_RE = re.compile(r"^(\s*)(myColor)(\d+)(\s+)0x[0-9A-Fa-f]{8}(\s*)$")


def _fmt_rough(v):
    """Format a roughness float the way the game files do (no trailing zeros): 0.5, 0.825, 1."""
    return ("%.4f" % float(v)).rstrip("0").rstrip(".") or "0"


def _set_main_color_index(text, idx):
    """Set/insert `myMainColorIndex <idx>` inside the myColorData2 block (text-level patch)."""
    m = re.search(r"(myColorData2[ \t]*\r?\n[ \t]*\{)(.*?)(\r?\n[ \t]*\})", text, re.S)
    if not m:
        return text
    head, inner, tail = m.group(1), m.group(2), m.group(3)
    cm = re.search(r"(\r?\n)([ \t]*)myColor1", inner)
    nl = cm.group(1) if cm else ("\r\n" if "\r\n" in text else "\n")
    indent = cm.group(2) if cm else "\t\t"
    if re.search(r"[ \t]*myMainColorIndex[ \t]+\d+", inner):
        inner = re.sub(
            r"([ \t]*myMainColorIndex[ \t]+)\d+",
            lambda mm: mm.group(1) + str(int(idx)),
            inner,
        )
    else:
        inner = inner + nl + indent + "myMainColorIndex " + str(int(idx))
    return text[: m.start()] + head + inner + tail + text[m.end() :]


def _set_color_roughness(text, vals):
    """Set/insert the sibling `myColorRoughness { myRoughness1..4 }` block (text-level patch)."""
    nl = "\r\n" if "\r\n" in text else "\n"
    bm = re.search(
        r"(myColorRoughness[ \t]*\r?\n[ \t]*\{)(.*?)(\r?\n[ \t]*\})", text, re.S
    )
    if bm:
        inner = bm.group(2)
        rm = re.search(r"(\r?\n)([ \t]*)myRoughness1", inner)
        rnl = rm.group(1) if rm else nl
        rindent = rm.group(2) if rm else "\t\t"
        for i in range(4):
            key = "myRoughness%d" % (i + 1)
            if re.search(r"[ \t]*%s[ \t]+[-0-9.]+" % key, inner):
                inner = re.sub(
                    r"([ \t]*%s[ \t]+)[-0-9.]+" % key,
                    lambda mm: mm.group(1) + _fmt_rough(vals[i]),
                    inner,
                )
            else:
                inner = inner + rnl + rindent + "%s %s" % (key, _fmt_rough(vals[i]))
        return text[: bm.start()] + bm.group(1) + inner + bm.group(3) + text[bm.end() :]
    dm = re.search(r"myColorData2[ \t]*\r?\n[ \t]*\{.*?\r?\n([ \t]*)\}", text, re.S)
    if not dm:
        return text
    indent = dm.group(1)
    inner_indent = indent + "\t"
    block = (
        nl
        + indent
        + "myColorRoughness"
        + nl
        + indent
        + "{"
        + "".join(
            nl + inner_indent + "myRoughness%d %s" % (i + 1, _fmt_rough(vals[i]))
            for i in range(4)
        )
        + nl
        + indent
        + "}"
    )
    return text[: dm.end()] + block + text[dm.end() :]


def update_blueitemtype_colors(
    path, hex_by_index, out_path=None, main_color_index=None, roughness=None
):
    """Patch a .blueitemtype's myColorN 0xAARRGGBB tokens with new colours and write the result
    (defaulting to `path` itself, i.e. an overwrite).

    This is a TEXT-LEVEL patch, not a from-scratch writer: only the matched colour-value tokens
    are replaced (alpha byte preserved from the original); every other byte of the file - name,
    uid, slot, textures, comments, line endings - is left exactly as it was. This is deliberate:
    we have a validated PARSER for these files but not a full from-scratch serializer, and a
    line-level patch of an already-real file is far safer than re-synthesising one.

    `hex_by_index` maps the file's own 1-based myColorN index (NOT row position - see
    rec["colors"][i]["index"] from parse_blueitemtype) to a '#rrggbb' string. Indices with no
    entry are left untouched. Returns the path written to.

    main_color_index : int 1..4 -> set/insert myMainColorIndex; None -> leave the file's as-is.
    roughness        : [r1..r4] floats -> set/insert myColorRoughness; None -> leave as-is.
    Both default to "don't touch", so disabling the warpaint authoring toggles never strips
    fields a loaded file already had.
    """
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        text = fh.read()
    lines = text.splitlines(keepends=True)
    changed = 0
    out_lines = []
    for ln in lines:
        m = _COLOR_LINE_RE.match(ln)
        if m:
            idx = int(m.group(3))
            hexcol = hex_by_index.get(idx)
            if hexcol:
                hexcol = hexcol.strip().lstrip("#").lower()
                if len(hexcol) == 6:
                    # find + preserve the original alpha byte (first 2 hex digits after 0x)
                    orig = re.search(r"0x([0-9A-Fa-f]{8})", ln).group(1)
                    alpha = orig[:2]
                    new_token = "0x" + alpha + hexcol
                    ln = (
                        m.group(1)
                        + m.group(2)
                        + m.group(3)
                        + m.group(4)
                        + new_token
                        + m.group(5)
                    )
                    changed += 1
        out_lines.append(ln)
    text = "".join(out_lines)
    if main_color_index is not None:
        text = _set_main_color_index(text, main_color_index)
    if roughness is not None:
        text = _set_color_roughness(text, roughness)
    dest = out_path or path
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return dest, changed


def parse_text(text, path=""):
    """Parse .blueitemtype source text into a structured dict (path is for reporting only)."""
    lines = text.splitlines()

    name = uid = None
    body, body_start = {}, None
    for i, ln in enumerate(lines):
        m = _HEADER_RE.search(ln)
        if m:
            name, uid = m.group(1), m.group(2).upper()
            # find the opening brace (next non-empty line that is '{')
            j = i + 1
            while j < len(lines) and lines[j].strip() != "{":
                j += 1
            if j < len(lines):
                body, _ = _parse_body(lines, j + 1)
            body_start = i
            break

    rec = {
        "file": os.path.basename(path),
        "path": path,
        "name": name,
        "uid": uid,
        "slot": body.get("myEquipmentSlot"),
        "category": "other",
        "display_name": "",
        "context": "",
        "ui_image": body.get("myUIImage", "") or "",
        "region": None,
        "dye_id": None,
        "source": None,
        "player_selectable": True,
        "base_item": None,
        "flags": {},
        "textures": {},
        "texture_targets": [],
        "colors": [],
        "roughness": None,
        "raw": body,
        "errors": [],
    }
    if body_start is None or name is None:
        rec["errors"].append("no BlueItemTypeCustomizationFeature header found")
        return rec

    # display name / context come from the embedded menu-line metadata in myUIName
    uiname = body.get("myUIName", "") or ""
    mt = _TEXT_RE.search(uiname)
    mc = _CTX_RE.search(uiname)
    if mt:
        rec["display_name"] = mt.group(1)
    if mc:
        rec["context"] = mc.group(1)

    # flags (top-level TRUE/FALSE booleans)
    for k, v in body.items():
        if isinstance(v, str) and v in ("TRUE", "FALSE"):
            rec["flags"][k] = v == "TRUE"
    if isinstance(body.get("myIsCustomizationSharedWithGameMode"), str):
        rec["flags"]["sharedWithGameMode"] = body["myIsCustomizationSharedWithGameMode"]

    # texture overrides
    td = body.get("myTextureData")
    if isinstance(td, dict):
        rec["textures"] = {k: v for k, v in td.items() if v not in (None, "")}

    # inheritance / region
    cd = body.get("myCustomizationData")
    if isinstance(cd, dict):
        rec["region"] = cd.get("myRegion")
        base = cd.get("myBaseItemType")
        if isinstance(base, str):
            mb = _BASE_RE.search(base)
            if mb:
                rec["base_item"] = {
                    "uid": mb.group(1).upper(),
                    "name": mb.group(2),
                    "hash": mb.group(3).upper(),
                }

    # category + roles. Texture-customization items (HairCap / SkinPattern / any *Texture slot)
    # carry their pick in myTextureData, not myColorData, so they have no colour roles - instead
    # they yield a normalised bind target (bucket, role, engine path) for the viewer.
    slot = rec["slot"]
    tex_info = SLOT_TEXTURES.get(slot)
    is_texture_item = bool(tex_info) or bool(slot and slot.endswith("Texture"))
    if is_texture_item:
        rec["category"], tex_role = tex_info if tex_info else ("texture", "texture")
        roles = []
        for k, v in rec["textures"].items():
            bucket = _TEX_BUCKET.get(k)
            if bucket and v:
                rec["texture_targets"].append(
                    {"bucket": bucket, "role": tex_role, "path": v}
                )
    else:
        category, roles = SLOT_ROLES.get(
            slot, ("other", [(i, "color %d" % i, "") for i in range(1, 5)])
        )
        rec["category"] = category

    # region-variant grouping (hair) + audience tag (warpaint)
    rec["dye_id"], rec["region"] = _region_and_dye(rec["name"], rec["region"])
    rec["source"] = _source_tag(rec["name"])
    rec["player_selectable"] = rec["source"] != "npc"

    # colours
    cdat = body.get("myColorData2")
    if isinstance(cdat, dict):
        for idx, role, param in roles:
            col = parse_argb(cdat.get("myColor%d" % idx, "") or "")
            if col is None:
                continue
            col.update({"index": idx, "role": role, "param": param})
            rec["colors"].append(col)
    else:
        if not is_texture_item:
            rec["errors"].append("no myColorData2 block")

    # roughness (warpaint)
    rdat = body.get("myColorRoughness")
    if isinstance(rdat, dict):
        rough = []
        for idx in range(1, 5):
            val = rdat.get("myRoughness%d" % idx)
            if val is not None:
                try:
                    rough.append(float(val))
                except ValueError:
                    pass
        if rough:
            rec["roughness"] = rough

    return rec


# ----------------------------------------------------------------------------- scan
_SLOT_SNIFF_RE = re.compile(r"myEquipmentSlot\s+(Customization\w+)")


def _iter_blueitem_paths(paths, recursive=True):
    """Yield every *.blueitemtype path reachable from `paths` (files and/or folders).
    Folders are searched recursively by default; results are de-duplicated."""
    seen = set()
    for p in paths:
        if os.path.isdir(p):
            pattern = (
                os.path.join(p, "**", "*.blueitemtype")
                if recursive
                else os.path.join(p, "*.blueitemtype")
            )
            for fp in glob.glob(pattern, recursive=recursive):
                rp = os.path.normpath(fp)
                if rp not in seen:
                    seen.add(rp)
                    yield rp
        elif p.endswith(".blueitemtype") and os.path.isfile(p):
            rp = os.path.normpath(p)
            if rp not in seen:
                seen.add(rp)
                yield rp


def _scan_files(paths, only_colors=True, dedupe=True):
    """Read + parse the given paths. When only_colors, a cheap content sniff skips any file
    that isn't a customization-colour item (skin/warpaint/eye/hair) before the full parse."""
    out, seen_uid = [], set()
    for p in sorted(paths):
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if "BlueItemTypeCustomizationFeature" not in text:
            continue  # not a customization feature at all
        if only_colors:
            m = _SLOT_SNIFF_RE.search(text)
            if not m or m.group(1) not in SLOT_ROLES:
                continue  # customization, but not a colour slot
        rec = parse_text(text, p)
        if only_colors and rec["category"] == "other":
            continue
        if dedupe and rec["uid"]:
            if rec["uid"] in seen_uid:
                continue
            seen_uid.add(rec["uid"])
        out.append(rec)
    return out


def scan_folder(folder, recursive=True, only_colors=True):
    """Scan a folder (recursively by default) for customization-colour items.
    Set only_colors=False to also include other customization features (bucketed 'other')."""
    return _scan_files(
        _iter_blueitem_paths([folder], recursive=recursive), only_colors=only_colors
    )


# ----------------------------------------------------------------------------- report
def select_players(items, drop_sources=("npc",)):
    """Drop items whose audience tag is in drop_sources (NPC weapon paints by default)."""
    return [it for it in items if it["source"] not in drop_sources]


def to_palette(items):
    """Compact {category: [ {name, uid, display, colors:[{role,hex,rgb01}], ...} ]} for the UI."""
    pal = {c: [] for c in CATEGORY_ORDER}
    for it in items:
        pal.setdefault(it["category"], []).append(
            {
                "name": it["name"],
                "uid": it["uid"],
                "display": it["display_name"] or it["name"],
                "context": it["context"],
                "dye_id": it["dye_id"],
                "region": it["region"],
                "source": it["source"],
                "player_selectable": it["player_selectable"],
                "base_item": it["base_item"]["name"] if it["base_item"] else None,
                "roughness": it["roughness"],
                "colors": [
                    {
                        "index": c["index"],
                        "role": c["role"],
                        "param": c["param"],
                        "hex": c["hex"],
                        "rgb01": c["rgb01"],
                    }
                    for c in it["colors"]
                ],
            }
        )
    return {k: v for k, v in pal.items() if v}


def summarize(items):
    """Human-readable report grouped by category."""
    by_cat = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)

    lines = []
    lines.append("Na'vi customization colour scan - %d item(s)" % len(items))
    lines.append("=" * 64)
    for cat in CATEGORY_ORDER:
        group = by_cat.get(cat)
        if not group:
            continue
        slot = next((s for s, (c, _r) in SLOT_ROLES.items() if c == cat), "?")
        lines.append("")
        lines.append("%s  (%s)  -  %d item(s)" % (cat.upper(), slot, len(group)))
        lines.append("-" * 64)
        for it in group:
            tail = []
            if it["source"] not in ("player", None):
                tail.append("src=%s" % it["source"])
            if it["region"]:
                tail.append("region=%s" % it["region"])
            if it["base_item"]:
                tail.append("base=%s" % it["base_item"]["name"])
            if it["roughness"]:
                tail.append("rough=%s" % "/".join("%.3g" % r for r in it["roughness"]))
            tails = ("   [" + ", ".join(tail) + "]") if tail else ""
            lines.append("  %s   uid=%s%s" % (it["name"], it["uid"], tails))
            # description: prefer the designer's contextComment, then the UI text label
            desc = it["context"] or (
                it["display_name"] if it["display_name"] != it["name"] else ""
            )
            if desc:
                lines.append("      \u201c%s\u201d" % desc)
            for c in it["colors"]:
                flag = " (black/unused)" if c["is_black"] else ""
                lines.append(
                    "      Color%d  %-18s %s  rgb01=%.3f,%.3f,%.3f  -> %s%s"
                    % (
                        c["index"],
                        c["role"],
                        c["hex"],
                        c["rgb01"][0],
                        c["rgb01"][1],
                        c["rgb01"][2],
                        c["param"] or "-",
                        flag,
                    )
                )
            for it_err in it["errors"]:
                lines.append("      ! %s" % it_err)
    return "\n".join(lines)


# ----------------------------------------------------------------------------- markdown doc
def _roles_for_category(cat):
    for _slot, (c, roles) in SLOT_ROLES.items():
        if c == cat:
            return roles
    return []


def _md(s):
    """Escape a cell for a markdown table."""
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def to_markdown(items, sources=None, files_scanned=None, only_colors=True):
    """Render the scan as a markdown document: one table per category, with each item's
    colours under role-named columns mapped to their shader params."""
    by_cat = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)

    L = ["# Na\u2019vi Customization Colour Catalog", ""]
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = "_Generated by `recolor_core.py` on %s \u2014 %d item(s)" % (ts, len(items))
    if files_scanned is not None:
        meta += " from %d scanned file(s)" % files_scanned
    L.append(meta + "._")
    if sources:
        L += ["", "Sources: " + ", ".join("`%s`" % s for s in sources)]
    counts = ", ".join(
        "%s\u00a0%d" % (c, len(by_cat[c])) for c in CATEGORY_ORDER if c in by_cat
    )
    L += ["", "Totals: " + (counts or "none"), ""]

    for cat in CATEGORY_ORDER:
        group = by_cat.get(cat)
        if not group:
            continue
        slot = next((s for s, (c, _r) in SLOT_ROLES.items() if c == cat), "")
        roles = [
            (i, role, param)
            for (i, role, param) in _roles_for_category(cat)
            if "unused" not in role and "dup" not in role
        ]
        if not roles:  # 'other' -> generic colour columns
            maxn = max(
                (
                    max(
                        (c["index"] for c in it["colors"] if not c["is_black"]),
                        default=0,
                    )
                    for it in group
                ),
                default=0,
            )
            roles = [(i, "Color%d" % i, "") for i in range(1, maxn + 1)]

        col_heads = [
            "%s%s" % (role.capitalize(), (" `%s`" % param) if param else "")
            for (_i, role, param) in roles
        ]
        cols = ["Item", "Description"] + col_heads + ["Extras", "uid"]
        L.append(
            "## %s%s \u2014 %d item(s)"
            % (cat.capitalize(), (" (`%s`)" % slot) if slot else "", len(group))
        )
        L += [
            "",
            "| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|",
        ]
        for it in sorted(group, key=lambda r: r["name"] or ""):
            cmap = {c["index"]: c for c in it["colors"]}
            desc = it["context"] or (
                it["display_name"]
                if it["display_name"] and it["display_name"] != it["name"]
                else ""
            )
            cells = ["`%s`" % _md(it["name"]), _md(desc)]
            for i, _role, _param in roles:
                c = cmap.get(i)
                cells.append(
                    "`%s`" % c["hex"] if (c and not c["is_black"]) else "\u2014"
                )
            extras = []
            if it["source"] not in ("player", None):
                extras.append("src " + _md(it["source"]))
            if it["region"]:
                extras.append("region " + _md(it["region"]))
            if it["base_item"]:
                extras.append("base `%s`" % _md(it["base_item"]["name"]))
            if it["roughness"]:
                extras.append("rough " + "/".join("%.3g" % r for r in it["roughness"]))
            cells.append(_md("; ".join(extras)))
            cells.append("`%s`" % (it["uid"] or ""))
            L.append("| " + " | ".join(cells) + " |")
        L.append("")

    # self-contained legend
    L += [
        "## Colour role reference",
        "",
        "Each item's `myColorData2` colours map to the decoded shader parameters:",
        "",
        "- **Skin** \u2014 Color1 base skin (`myBaseColorOverlay`), Color2 pattern (`myPatternColor`)",
        "- **Warpaint** \u2014 Color1\u20134 \u2192 `myPaintColor1..4` (+ `myColorRoughness1..4`)",
        "- **Eye** \u2014 Color1 outer iris (`myOuterIrisHue`), Color2 inner iris (`myInnerIrisHue`)",
        "- **Hair** \u2014 Color1/2/3 root/mid/tip (`myHairColor1..3`), Color4 scalp/hair-cap "
        "(`myHairCapColor`)",
        "",
        "Colours are 8-bit sRGB (`0xAARRGGBB`); convert sRGB\u2192linear before feeding the "
        "`Overlay`/`lerp` shader math. A `#000000` entry is an unset sentinel: for hair, a black "
        "Color4 means no explicit scalp colour, so the cap inherits the hair root.",
    ]
    return "\n".join(L) + "\n"


def write_report(items, path, **meta):
    """Write the markdown catalog to `path`."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(items, **meta))
    return path


def write_palette_json(items, path):
    """Write the UI-ready palette JSON to `path`."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_palette(items), fh, indent=2)
    return path


# ----------------------------------------------------------------------------- CLI
def main(argv):
    only_colors = "--all" not in argv
    drop_npc = "--no-npc" in argv
    out_md = out_json = None
    targets = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--all":
            pass
        elif a == "--out":
            i += 1
            out_md = argv[i] if i < len(argv) else "navi_colors_catalog.md"
        elif a.startswith("--out="):
            out_md = a.split("=", 1)[1] or "navi_colors_catalog.md"
        elif a == "--json":
            i += 1
            out_json = argv[i] if i < len(argv) else "navi_colors_palette.json"
        elif a.startswith("--json="):
            out_json = a.split("=", 1)[1] or "navi_colors_palette.json"
        elif a.startswith("--"):
            pass  # unknown flag, ignore
        else:
            targets.append(a)
        i += 1

    if not targets:
        print(
            "usage: python3 recolor_core.py <folder-or-files...> [--all] [--no-npc] "
            "[--out [FILE.md]] [--json [FILE.json]]\n"
            "  Recurses into sub-folders and keeps skin/warpaint/eye/hair colour items.\n"
            "  --out    writes a markdown catalog (default navi_colors_catalog.md)\n"
            "  --json   writes the UI palette  (default navi_colors_palette.json)\n"
            "  --no-npc drops NPC weapon-skin warpaints (keeps player-selectable items)\n"
            "  --all    also include other customization features\n"
            "  With no --out/--json, prints the report to stdout."
        )
        return 2

    all_paths = list(_iter_blueitem_paths(targets, recursive=True))
    if not all_paths:
        print("no .blueitemtype files found under: " + ", ".join(targets))
        return 1
    items = _scan_files(all_paths, only_colors=only_colors)
    if drop_npc:
        before = len(items)
        items = select_players(items)
        dropped = before - len(items)
    kind = "customization-colour" if only_colors else "customization"
    print(
        "Scanned %d .blueitemtype file(s) under %d path(s); found %d %s item(s)%s."
        % (
            len(all_paths),
            len(targets),
            len(items),
            kind,
            (" (dropped %d npc)" % dropped) if drop_npc else "",
        )
    )

    if out_md:
        write_report(
            items,
            out_md,
            sources=targets,
            files_scanned=len(all_paths),
            only_colors=only_colors,
        )
        print("  wrote markdown catalog -> %s" % out_md)
    if out_json:
        write_palette_json(items, out_json)
        print("  wrote palette JSON     -> %s" % out_json)
    if not out_md and not out_json:
        print()
        print(summarize(items))
    return 0

    # ---------------------------------------------------------------------------
    # Camo (gear/weapon) colour blend, ported from "blinn-phong camouflage_rse.mshader". Samples a
    # 3-channel mask (w_camo_solid.dds) triplanar-projected in model space, then per pixel:
    #       camo = colorR*mask.r + colorG*mask.g + colorB*mask.b + mask.r*mask.g*mask.b
    #       camo = clamp(camo, 0.025, 0.7)
    # and lerps onto the base where material.a * myCamoEnable is set. myCamoColorR/G/B are the palette's
    # Primary/Secondary/Tertiary in linear light (rejuice stores them sRGB 0xAARRGGBB).
    # ---------------------------------------------------------------------------


CAMO_CLAMP_LO = 0.025
CAMO_CLAMP_HI = 0.7


def _camo_hex_to_lin(h):
    """0xAARRGGBB / AARRGGBB / RRGGBB / #RRGGBB -> linear-light float3 (alpha dropped)."""
    if h is None:
        return None
    h = h.strip().lower().lstrip("#")
    if h.startswith("0x"):
        h = h[2:]
    if len(h) == 8:  # AARRGGBB -> RRGGBB
        h = h[2:]
    return srgb_to_linear(hex_to_rgb01(h))


def camo_colors_from_palette(primary, secondary=None, tertiary=None):
    """Return (R, G, B) linear-light float3 colours for the camo shader's
    myCamoColorR / G / B slots, from a palette's Primary / Secondary / Tertiary
    0xAARRGGBB values. When a palette defines fewer than three colours, the missing
    slots fall back to the primary (the game reuses it the same way)."""
    r = _camo_hex_to_lin(primary)
    g = _camo_hex_to_lin(secondary)
    b = _camo_hex_to_lin(tertiary)
    if r is None:
        r = np.zeros(3, np.float32)
    if g is None:
        g = r
    if b is None:
        b = r
    return r, g, b


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
