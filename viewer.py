"""moderngl viewport embedded in a PyQt6 QOpenGLWidget.

Starts empty; model and textures load at runtime via load_model(path) and
set_texture(key, role, np_rgba). Body/Head meshes use the recolour shader, others render flat.
Colours update live via set_palette(); orbit = LMB drag, zoom = wheel.
"""

from __future__ import annotations
import numpy as np
import moderngl
import sys
import logging
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

import gl_shaders as gs

log = logging.getLogger("pandorapaint.viewer")
import mmb_loader as ml
import recolor_core as rc

RECOLOUR_MESHES = {"Banshee_Body": "body", "Banshee_Head": "head"}

# Compact luma-based FXAA (Timothy Lottes' console variant), run as a post-process on the resolved
# image - a cheap "Performance" anti-aliasing that works with SSAA off (unlike full supersampling it
# only smooths luma edges, so it can't remove alpha-cut hair pixelation the way SSAA does).
_FXAA_FRAG = (
    "#version 330\n"
    "uniform sampler2D uTex;\n"
    "uniform vec2 uRcpFrame;   // 1.0 / source texel size\n"
    "in vec2 vUV; out vec4 f;\n"
    "void main(){\n"
    "  const float SPAN_MAX = 8.0, REDUCE_MUL = 1.0/8.0, REDUCE_MIN = 1.0/128.0;\n"
    "  vec3 luma = vec3(0.299, 0.587, 0.114);\n"
    "  vec3 nw = texture(uTex, vUV + vec2(-1.0,-1.0)*uRcpFrame).rgb;\n"
    "  vec3 ne = texture(uTex, vUV + vec2( 1.0,-1.0)*uRcpFrame).rgb;\n"
    "  vec3 sw = texture(uTex, vUV + vec2(-1.0, 1.0)*uRcpFrame).rgb;\n"
    "  vec3 se = texture(uTex, vUV + vec2( 1.0, 1.0)*uRcpFrame).rgb;\n"
    "  vec3 m  = texture(uTex, vUV).rgb;\n"
    "  float lnw = dot(nw,luma), lne = dot(ne,luma), lsw = dot(sw,luma), lse = dot(se,luma), lm = dot(m,luma);\n"
    "  float lmin = min(lm, min(min(lnw,lne), min(lsw,lse)));\n"
    "  float lmax = max(lm, max(max(lnw,lne), max(lsw,lse)));\n"
    "  vec2 dir = vec2(-((lnw+lne)-(lsw+lse)), ((lnw+lsw)-(lne+lse)));\n"
    "  float reduce = max((lnw+lne+lsw+lse)*0.25*REDUCE_MUL, REDUCE_MIN);\n"
    "  float rcpMin = 1.0/(min(abs(dir.x),abs(dir.y)) + reduce);\n"
    "  dir = clamp(dir*rcpMin, vec2(-SPAN_MAX), vec2(SPAN_MAX)) * uRcpFrame;\n"
    "  vec3 rgbA = 0.5*(texture(uTex, vUV+dir*(1.0/3.0-0.5)).rgb + texture(uTex, vUV+dir*(2.0/3.0-0.5)).rgb);\n"
    "  vec3 rgbB = rgbA*0.5 + 0.25*(texture(uTex, vUV+dir*-0.5).rgb + texture(uTex, vUV+dir*0.5).rgb);\n"
    "  float lb = dot(rgbB, luma);\n"
    "  f = vec4((lb < lmin || lb > lmax) ? rgbA : rgbB, 1.0);\n"
    "}\n"
)

# Mesh parts to drop entirely from the Banshee preview, matched by NAME PREFIX so it covers both
# a bare name and a "_LOD0"-suffixed one (the real export may carry the suffix or not, depending
# on source format - same as the Wing1/Wing2 aliasing below).
BANSHEE_SKIP_MESH = ("Banshee_Head_part", "Banshee_weakpoint")


# Which texture atlas every non-recolour mesh samples. Per the engine graph object,
# wings use the shared insect-wing texture (DragonflyWing shader) and eyes use the eye
# textures (Eye shader) - NOT the body/head skin atlas. Head_part is inner-mouth head
# skin; the weakpoint is a body skin patch.
ATLAS_OF = {
    "Banshee_Head_part": "head",
    "Banshee_weakpoint": "body",
    "Banshee_SmallEyes": "eye",
    "Banshee_Eyes": "eye",
    "Banshee_Wing": "wing",  # MMB merges the two wings into one mesh
    "Banshee_Wing1": "wing",  # the export splits them
    "Banshee_Wing2": "wing",
}


    # Na'vi mode: BansheeViewer renders BOTH the banshee and the Na'vi model from ONE GL context
    # (switched by set_mode()) - a single QOpenGLWidget avoids the two-context conflict seen when the
    # Na'vi viewer was separate. The Na'vi fragment shader is gl_shaders.NAVI_GLSL (matches the CPU ref).
NAVI_KIND_SKIN, NAVI_KIND_EYE, NAVI_KIND_HAIR, NAVI_KIND_FLAT, NAVI_KIND_LASH = (
    0,
    1,
    2,
    3,
    4,
)
NAVI_KIND_EYESHADOW = (
    6  # "eyeshell" mesh -> PX_Eye_Shadow: transparent dark contact-shadow
)
NAVI_KIND_EYESHELL = (
    7  # "eyeEdge" mesh  -> PX_Eye_Shell: cornea shell, white/alpha from vtx colour
)
NAVI_BUCKETS = ("head", "body", "eye", "hair", "kuru", "accessory")
    # Mesh parts to drop from the Na'vi preview (substring match, lowercased). "eyeshell" (PX_Eye_Shadow)
    # is a transparent contact-shadow overlay that needs real alpha blending; skipped until the preview
    # does ordered transparency (an opaque fill was just wrong). "eyeEdge" (PX_Eye_Shell) is the
    # cornea/sclera shell whose white is mostly transparent via a per-vertex alpha mmb_loader skips - so
    # it too would render as an opaque blob; the iris mesh already carries the sclera, so skip it.
NAVI_SKIP_SUBMESH = ("eyeshell", "eyeedge")
# fixed texture units for the navi program (match the sampler uniforms set in initializeGL)
NU_COLOR, NU_MATERIAL, NU_PATTERN, NU_PAINT, NU_BIO, NU_HAIRCAP = 0, 1, 2, 3, 4, 5
NU_IRIS, NU_HAIRMASK, NU_HAIRAO, NU_EYE_HEIGHT = 6, 7, 8, 9
NU_EYE_NORMAL = 10
NU_SKIN_NORMAL, NU_DETAIL_NORMAL = (
    11,
    12,
)  # skin normal (alpha=AO) + tiled detail normal
NU_PAINT2, NU_PAINT3, NU_PAINT4 = 13, 14, 15  # extra body warpaint layers (Body 2/3/4)
_NGREY = (0.5, 0.5, 0.5)
# Skin shades in linear now, so the overlay-identity neutral is the sRGB value that linearises to
# 0.5 (overlayBlend(0.5_lin, x) == x). linear_to_srgb(0.5) = 0.735357. Skin-branch swatches default
# to this so "no recolour loaded" still shows the raw texture; eye/hair stay on the 0.5 sRGB path.
_NGREY_LIN = (0.735357, 0.735357, 0.735357)


def navi_classify(part, name):
    """Map a Na'vi submesh (source part + mesh name) to (kind, texture-bucket).
    Validated against the real p_head_01_m / rnf_body_01_m / p_rnf_hair_01 submesh names AND the
    submesh->shader bindings in p_head_01_f.mgraphobject. NB the mesh NAMES are misleading:
      eyeLeft/eyeRight -> PX_Eye2            (iris, gets the hue overlay)
      eyeshell         -> PX_Eye_Shadow      (TRANSPARENT dark contact-shadow overlay, NOT white)
      eyeEdge          -> PX_Eye_Shell       (the cornea/sclera SHELL; carries the vertex colours)
      eyelashes        -> px_basic           (lash cards)
    so we classify by the real shader role, not the name."""
    n = (name or "").lower()
    iris = (
        "eye" in n
        and any(k in n for k in ("left", "right", "ball", "iris"))
        and not any(k in n for k in ("lash", "shell", "edge", "shadow", "lid", "brow"))
    )
    if iris:
        return NAVI_KIND_EYE, "eye"
    if part == "hair":
        # the 'Accessories' submesh is a flat px_basic piece with its OWN diffuse (feathers /
        # hairbands / leatherwork, named in the hairstyle's .mgraphobject) - give it the dedicated
        # 'accessory' bucket so it stops sampling the hair bucket (which has no colour -> flat grey).
        if "accessor" in n:
            return NAVI_KIND_FLAT, "accessory"
        if any(k in n for k in ("kuru", "bead", "band", "cap", "tie", "feather")):
            return NAVI_KIND_FLAT, "hair"
        return NAVI_KIND_HAIR, "hair"
    if part == "kuru":
        # the Na'vi neural queue/braid: the strands - including the tapering hair END/TIP - render like
        # hair (3-colour gradient). Only the wrapping hardware (bands, beads, ties) is a flat px_basic
        # accessory piece that samples the band atlas; the tip is NOT hardware, so it stays hair.
        if any(k in n for k in ("band", "bead", "tie", "accessor", "feather")):
            return NAVI_KIND_FLAT, "kuru"
        return NAVI_KIND_HAIR, "kuru"
    if part == "body":
        # the tail-hair tuft at the tip is a separate submesh that uses the RNF hair shader
        # (px_hair2_3color_tousle) + the p_rnf_hair_01 textures, per p_body_01_f.mgraphobject - route
        # it to the hair bucket so it samples the hair textures instead of the body skin map.
        if "tailhair" in n or ("tail" in n and "hair" in n):
            return NAVI_KIND_HAIR, "hair"
        return NAVI_KIND_SKIN, "body"
    if part == "head":
        if "lash" in n:  # eyelash card: own alpha texture, transparent pass
            return NAVI_KIND_LASH, "lash"
        # "eyeshell" mesh -> PX_Eye_Shadow: a transparent dark contact-shadow overlay (blend, Fog
        # colour, alpha-discard). It is NOT a white sclera. Routed to the eye-shadow kind.
        if "shell" in n:
            return NAVI_KIND_EYESHADOW, "eyeshadow"
        # "eyeEdge" mesh -> PX_Eye_Shell: the cornea/sclera shell. Its white + opacity come from
        # vertex colours (R = alpha, G = shade); loaded into SubMesh.colors. Routed to the shell kind.
        if "edge" in n:
            return NAVI_KIND_EYESHELL, "eyeshell"
        # genuinely-flat head parts: teeth/tongue/gums/nails and the small eye-accessory meshes
        # (eyelid, eyebrow, eyesocket, eyeshadow). NB: do NOT match a bare "eye" - the main FACE mesh
        # can contain "eye" in its name and must classify as SKIN (gets the skin recolour) like the
        # body, otherwise the head shows the raw texture while the body is recoloured (a colour seam).
        flat = any(k in n for k in ("tooth", "teeth", "tongue", "gum", "nail")) or any(
            k in n
            for k in (
                "eyelid",
                "eyebrow",
                "eyesocket",
                "eyeshadow",
                "eye_lid",
                "eye_brow",
                "brow",
                "lid",
            )
        )
        return (NAVI_KIND_FLAT, "head") if flat else (NAVI_KIND_SKIN, "head")
    return NAVI_KIND_FLAT, (part if part in NAVI_BUCKETS else "head")


NAVI_VERTEX = """
#version 330
uniform mat4 uMVP;
uniform mat3 uNormalMat;
uniform float uFlipV;
in vec3 in_pos;
in vec3 in_nrm;
in vec3 in_tan;
in vec2 in_uv;
in vec2 in_uv1;
out vec3 vN;
out vec3 vTan;
out vec2 vUV;
out vec2 vUV1;
out vec3 vWorldPos;
void main(){
    vN  = normalize(uNormalMat * in_nrm);
    vTan = uNormalMat * in_tan;                          // strand tangent (may be 0 -> no sheen)
    vUV  = vec2(in_uv.x,  mix(in_uv.y,  1.0 - in_uv.y,  uFlipV));
    vUV1 = vec2(in_uv1.x, mix(in_uv1.y, 1.0 - in_uv1.y, uFlipV));
    vWorldPos = in_pos;                                  // object space is fine for the tangent frame
    gl_Position = uMVP * vec4(in_pos, 1.0);
}
"""

NAVI_FRAG = """
#version 330
in vec3 vN;
in vec3 vTan;
in vec2 vUV;
in vec2 vUV1;
in vec3 vWorldPos;
out vec4 frag;

uniform int  uKind;            // 0 skin,1 eye,2 hair,3 flat,4 eyelash,5/7 cornea-shell,6 eye-shadow
uniform vec3 uLightDir;

uniform sampler2D uColorTex;
uniform sampler2D uMaterialTex;
uniform sampler2D uPatternTex;
uniform sampler2D uPaintTex;
uniform sampler2D uPaintTex2;
uniform sampler2D uPaintTex3;
uniform sampler2D uPaintTex4;
uniform sampler2D uBioTex;
uniform sampler2D uHairCapTex;
uniform sampler2D uIrisTex;
uniform sampler2D uHairMask;
uniform sampler2D uHairAO;
uniform sampler2D uEyeHeightTex;
uniform sampler2D uEyeNormalTex;
uniform sampler2D uSkinNormalTex;   // main skin normal; alpha = baked AO (UnpackNormalAndAO)
uniform sampler2D uDetailNormalTex; // tiled detail normal (fine pore relief, masks UV seams)

uniform int uHasColor, uHasMaterial, uHasPattern, uHasPaint, uHasBio, uHasHairCap;
uniform int uHasPaint2, uHasPaint3, uHasPaint4;
uniform int uHasIris, uHasHairMask, uHasHairAO, uHasEyeHeight, uHasEyeNormal;
uniform int uHasSkinNormal, uHasDetailNormal;
uniform float uSkinAOStrength;      // how strongly the baked AO darkens the skin (0..1)
uniform float uSkinAmbient;         // legacy flat floor (unused by skin now - IBL drives ambient)
uniform float uSkinLightWrap;       // legacy half-Lambert wrap (unused by skin now)
uniform float uIblIntensity;        // master scale on the SH irradiance ambient (scene exposure)
uniform float uKeyLight;            // optional frontal directional fill (0 = pure IBL)
uniform float uIblSaturation;       // 1 = raw cubemap chroma, 0 = neutral grey ambient (sky-off rig)
uniform vec3  uIblTint;             // white-balance on the ambient (derived game/viewer channel ratio)
uniform float uClearCoat;           // skin clear-coat strength (px_character_navi: ~0.08-0.20)
uniform float uClearCoatRough;      // clear-coat roughness (px_character_navi: ~0.2-0.7, soft sheen)
uniform float uSpecular;            // master specular enable (1 = on, 0 = off): clear-coat + hair sheen
uniform float uSkinAOMean;          // this mesh's mean AO (AO applied relative to it, no global step)
uniform float uSkinNormalStr;       // main skin-normal xy strength (damps the baked musculature)
uniform float uDetailNormalStr;     // detail-normal relief strength
uniform float uDetailNormalTiling;  // detail-normal UV tiling (shader default myDetailTiling = 1)

uniform vec3  uSkinColor, uPatternColor;
uniform float uSkinColorWeight; // 0 = sat-gated hybrid (current), 1 = pure game colour-base order
uniform float uPatternStrength; // scales the skin-pattern (stripe) coverage; 1 = full, <1 = subtler
uniform float uSkinDesaturate;  // pulls the diffuse toward its own luminance BEFORE the skin overlay,
                                // but only for DARK skin colours - kills the blue diffuse texels that
                                // otherwise survive the overlay's screen branch as blue chest speckles.
uniform vec3  uToneCorrect;    // per-mesh skin tone match (1,1,1 = none; body scaled toward head)
uniform vec3  uPaintColor1, uPaintColor2, uPaintColor3, uPaintColor4;
uniform vec3  uBioColor, uHairCapColor;
uniform int   uHasHairCapColor;
uniform float uHairCapStrength;
uniform float uBioBrightness, uBioPulsation, uTime;
uniform int   uBioEnabled;
uniform vec3  uOuterIris, uInnerIris;
uniform float uEyeHeightBlur;  // mip LOD bias for the iris height map (averages out fiber noise)
uniform float uIrisSpread;     // <1 pulls inner iris hue inward, widening the outer rim
uniform float uIrisBlend;      // inner/outer transition steepness (lower = softer, more blended)
uniform float uIrisDetail;     // how strongly the iris fibre detail shows through the colour
uniform float uIrisMeanLum;    // mean luminance of the iris texture (centre for the detail term)
uniform float uIrisHeightLo;   // height where the overlay starts fading out (toward the flat white)
uniform float uIrisHeightHi;   // height where the overlay is fully off (the white sclera region)
uniform vec2  uIrisUVCenter;   // UV centre of the iris disc (~0.46,0.46 for the Na'vi eyeball)
uniform float uIrisRadius;     // UV radius where the hue overlay starts fading out (outer rim)
uniform float uIrisRimSoft;    // softness of that radial fade
uniform float uIrisNormalStr;  // strength of the iris normal-map fibre relief
uniform float uIrisOpacity;    // base colour opacity over the iris (1=full colour, <1 lets detail show)
uniform float uIrisOpacityFalloff;  // how much extreme hues reduce opacity (detail-protect for white/black)
uniform vec3  uHair1, uHair2, uHair3;
uniform float uSmoothness, uRootDarkening;
uniform float uHairCoverage;   // strand coverage scale for the dithered alpha (1=raw, >1 = fuller)
uniform vec3  uCamPos;         // camera position (object space) - view vector for the hair sheen
uniform float uHairSpec;       // anisotropic hair-sheen strength (0 = off); preview-only, no export effect
uniform float uHairRough;      // hair roughness -> sheen breadth (the shader's myRoughness)
uniform int   uHairSoftAlpha;  // 1 = soft coverage (overlapping cards blend), 0 = hard cutoff
uniform float uTonemap;        // 0 = raw clamp (current look), 1 = full ACES filmic rolloff (game-like)
uniform vec3  uLashTint;       // multiply on eyelash albedo - the lash card is near-white; game lashes are dark

// ---- Camouflage (gear flat path only; mirrors blinn-phong camouflage_rse.mshader) ----
uniform int   uCamoEnable;     // 1 = blend the camo colours into this gear piece (default 0 -> no-op)
uniform sampler2D uCamoMask;   // w_camo_solid 3-channel region mask (shares the material unit)
uniform int   uHasCamoMask;    // 1 = a camo mask is bound; without it the block is skipped
uniform vec3  uCamoColR;       // myCamoColorR = Primary   (LINEAR light)
uniform vec3  uCamoColG;       // myCamoColorG = Secondary (LINEAR light)
uniform vec3  uCamoColB;       // myCamoColorB = Tertiary  (LINEAR light)
uniform vec2  uCamoTiling;     // myCamoTiling   (game default 4,4)
uniform float uCamoRotation;   // myCamoRotation (degrees)
uniform sampler2D uGearMatTex; // gear Material (_m) map; .r=metalness .g=roughness .b=detailStr .a=camo coverage
uniform int   uHasGearMat;     // 1 = a gear _m map is bound (shares the pattern unit on the flat path)
uniform int   uCamoAlphaEnable;// myCamoAlphaEnable: 1 = let the mask alpha punch the diffuse back (default 0)
uniform int   uCamoBlendMode;  // 0 = weapon (replace, camouflage_rse), 1 = cloth (Overlay, 3coloroverlay)
uniform sampler2D uGearRegionTex; // cloth ColorMask (_reg_m), UV0: 4 baked regions; camo enters the .y zone
uniform int   uHasGearRegion;  // 1 = a region ColorMask is bound (shares the detail-normal unit)
uniform vec3  uCamoColA;       // cloth 4th overlay (myColorOverlay_A base); default = Overlay identity
uniform sampler2D uGearNormalTex; // gear Normal (_n) map (UnpackNormalAndAO: rgb=normal, a=AO); camo path only
uniform int   uHasGearNormal;  // 1 = a gear _n map is bound (shares the skin-normal unit on the flat path)
uniform float uGearNormalStr;  // gear normal-map strength (xy scale; default 1)

// ---- gl_shaders.NAVI_GLSL is concatenated above this main() at load time ----

void main(){
    vec3 albedo;
    float alpha = 1.0;
    vec3 emission = vec3(0.0);
    vec3 hairSpec = vec3(0.0);                            // anisotropic strand sheen (hair branch only)
    vec3 shadeN = normalize(vN);                         // per-branch normal (eye perturbs it)
    float camoCover = 0.0;                                // camo coverage at this texel (Material.a*enable)

    if (uKind == 0) {                                   // SKIN / FACE
        // Skin shades in LINEAR light (see srgb2lin/lin2srgb in NAVI_GLSL). The diffuse is an sRGB
        // texture, so linearise on sample; every colour param (skin/pattern/paint/bio/cap) is a sRGB
        // swatch, so linearise it too before the overlay chain. The lit/tonemap/encode tail (below)
        // finishes the skin branch in linear and writes sRGB.
        vec3 base = (uHasColor == 1) ? srgb2lin(texture(uColorTex, vUV).rgb) : vec3(0.5);
        base *= uToneCorrect;                            // per-mesh tone match (body -> head) to hide the neck step
        float g   = (uHasMaterial == 1) ? texture(uMaterialTex, vUV).g : 1.0;
        float skinM = g * g;
        float patM  = (uHasPattern == 1) ? texture(uPatternTex, vUV).r * g : 0.0;
        // Warpaint: up to four paint _m maps layered onto the body (Body 1..4 from the tool). Each
        // uses its R channel to pick one of the four paint colours (paintSelect) and its G as
        // coverage. Later layers paint over earlier ones; coverage accumulates (max) so recolorSkin
        // blends the right amount. The head uses only the first layer (paint2..4 stay unbound there).
        vec3 pCol1 = srgb2lin(uPaintColor1), pCol2 = srgb2lin(uPaintColor2),
             pCol3 = srgb2lin(uPaintColor3), pCol4 = srgb2lin(uPaintColor4);
        vec3 paint = vec3(0.0);
        float paintCov = 0.0;
        if (uHasPaint == 1)  { vec4 pt = texture(uPaintTex,  vUV); float pc = pt.g * g;
            paint = mix(paint, paintSelect(pCol1, pCol2, pCol3, pCol4, pt.rgb), pc); paintCov = max(paintCov, pc); }
        if (uHasPaint2 == 1) { vec4 pt = texture(uPaintTex2, vUV); float pc = pt.g * g;
            paint = mix(paint, paintSelect(pCol1, pCol2, pCol3, pCol4, pt.rgb), pc); paintCov = max(paintCov, pc); }
        if (uHasPaint3 == 1) { vec4 pt = texture(uPaintTex3, vUV); float pc = pt.g * g;
            paint = mix(paint, paintSelect(pCol1, pCol2, pCol3, pCol4, pt.rgb), pc); paintCov = max(paintCov, pc); }
        if (uHasPaint4 == 1) { vec4 pt = texture(uPaintTex4, vUV); float pc = pt.g * g;
            paint = mix(paint, paintSelect(pCol1, pCol2, pCol3, pCol4, pt.rgb), pc); paintCov = max(paintCov, pc); }
        vec3 bioCol = srgb2lin(uBioColor);
        float bioRaw = (uBioEnabled == 1 && uHasBio == 1) ? texture(uBioTex, vUV1).r : 0.0;
        float bioM = clamp((bioRaw - 0.08) * 4.0, 0.0, 1.0) * g;
        float capM = (uHasHairCap == 1) ? texture(uHairCapTex, vUV1).r : 0.0;
        capM *= clamp(uHairCapStrength, 0.0, 1.0);     // 'Cap strength' dial (0=off, 1=full)
        // Scalp cap (the patch under/around the hair on the head UV). The dye's slot-4 colour
        // (myHairCapColor) is the explicit scalp tint when present; otherwise fall back to inheriting
        // the HAIR ROOT (uHair1, darkened by uRootDarkening) so the scalp still reads as the hair base.
        vec3 capCol = (uHasHairCapColor == 1)
                      ? srgb2lin(uHairCapColor)
                      : srgb2lin(uHair1) * (1.0 - uRootDarkening);
        // DARK-SKIN BLUE FIX: the body diffuse is blue-dominant. With a dark skin colour the bulk
        // goes down the overlay's MULTIPLY branch (taking the dark skin hue), but the few bright
        // texels (chest freckle/marking clusters) hit the SCREEN branch, which preserves the diffuse
        // hue - so they survive as saturated-blue speckles. The game hides this with deep-red SSS;
        // we approximate by pulling the diffuse toward its own luminance (detail stays, blue chroma
        // goes) so the overlay then tints those texels to the skin colour like everything else. Gate
        // it to dark skin only (smoothstep on skin-colour luminance) so the light default look that
        // was already signed off is left completely untouched.
        vec3 skinLin = srgb2lin(uSkinColor);
        float skinLum = dot(skinLin, vec3(0.2126, 0.7152, 0.0722));
        float desatAmt = uSkinDesaturate * (1.0 - smoothstep(0.06, 0.35, skinLum));
        base = mix(base, vec3(dot(base, vec3(0.2126, 0.7152, 0.0722))), desatAmt);
        albedo = recolorSkin(base, skinM, skinLin,
                             srgb2lin(uPatternColor), patM,
                             paint, paintCov,
                             bioM, bioCol,
                             capCol, capM,
                             uSkinColorWeight, uPatternStrength);
        if (bioM > 0.0) {
            float puls = 1.0 + (abs(-1.0 + 2.0*fract(uTime)) - 1.0) * uBioPulsation;
            emission = max(bioCol * (bioM * uBioBrightness * puls) * (1.0 - capM), 0.0);
        }
        // SKIN NORMAL + baked AMBIENT OCCLUSION (UnpackNormalAndAO: rgb=normal, a=AO). The AO is the
        // soft cavity shading the game bakes in (neck crease, collarbone, chest contour); applying it
        // unifies the head/body surface and removes the flat-lit patchiness that makes the seam pop.
        // The main normal already bakes the musculature, so its xy is scaled by uSkinNormalStr to
        // avoid over-cranking the abs/pecs (the previous full-strength + 20x detail was way too harsh).
        if (uHasSkinNormal == 1) {
            vec4 sn = texture(uSkinNormalTex, vUV);
            // AO applied RELATIVE to this mesh's own mean: (ao - mean) so it averages to 0 -> only
            // local cavity darkening, no global brightness shift that would step at the head/body neck.
            float aoRel = (sn.a - uSkinAOMean);
            float ao = clamp(1.0 + aoRel * uSkinAOStrength, 0.0, 1.0);
            albedo *= ao;
            vec3 nm = vec3((sn.xy * 2.0 - 1.0) * uSkinNormalStr, 1.0);   // damped main normal
            vec3 Ng = normalize(vN);
            vec3 dp1 = dFdx(vWorldPos); vec3 dp2 = dFdy(vWorldPos);
            vec2 du1 = dFdx(vUV);       vec2 du2 = dFdy(vUV);
            vec3 T = normalize(dp1 * du2.y - dp2 * du1.y);
            T = normalize(T - Ng * dot(Ng, T));
            vec3 B = cross(Ng, T);
            // DETAIL NORMAL: per the body shader, detail strength = myDetailStrength * Material.z
            // (a per-texel MASK), tiled at myDetailTiling (default 1x - the sampler's '20' is only a
            // mip hint, not the tiling). So detail is gated by the material map and subtle, not a
            // uniform high-frequency overlay (that was the harsh look).
            vec2 dn = vec2(0.0);
            if (uHasDetailNormal == 1) {
                float dmask = (uHasMaterial == 1) ? texture(uMaterialTex, vUV).b : 1.0;
                vec3 dtex = texture(uDetailNormalTex, vUV * uDetailNormalTiling).rgb * 2.0 - 1.0;
                dn = dtex.xy * (uDetailNormalStr * dmask);
            }
            vec3 perturbed = normalize(T * (nm.x + dn.x) + B * (nm.y + dn.y) + Ng);
            shadeN = perturbed;
        }
    }
    else if (uKind == 1) {                              // EYE / IRIS  -  exact px_eye2 recolour
        // px_eye2.mshader, in LINEAR (deferred). The earlier build branched the Overlay on the COLOUR
        // and wrapped it in contrast-normalisation / adaptive-opacity / radial-falloff heuristics -
        // none of which is in the shader. The real chain is:
        //   mask    = 1 - saturate((h - 0.95)/0.05)        (remap_231: 1 on iris, 0 on the white sclera)
        //   selectT = saturate((1 - h) * mask * 1.8)       (iris centre -> inner hue, rim -> outer hue)
        //   hue     = lerp(outerHue, innerHue, selectT)
        //   colour  = Overlay(blend=hue, base=iris, mask) == lerp(iris, overlayBlend(iris,hue), mask)
        // Overlay branches on the IRIS (base) - so the iris fibre detail comes through on its own and
        // no contrast hack is needed (branching on the colour was the same bug as the skin). The game
        // parallax-traces the height for h; we read the raw height (uEyeHeightBlur mip-blurs it to
        // stand in for the traced smoothing).
        vec3 iris = (uHasIris == 1) ? srgb2lin(texture(uIrisTex, vUV).rgb) : vec3(0.214);
        float h = (uHasEyeHeight == 1) ? textureLod(uEyeHeightTex, vUV, uEyeHeightBlur).r : 1.0;
        float mask    = clamp(1.0 - (h - 0.95) / 0.05, 0.0, 1.0);
        float selectT = clamp((1.0 - h) * mask * 1.8, 0.0, 1.0);
        vec3 hue = mix(srgb2lin(uOuterIris), srgb2lin(uInnerIris), selectT);
        vec3 recol = overlayBlend(iris, hue);           // base = iris, blend = hue (branch on iris)
        albedo = mix(iris, recol, mask);                // linear albedo
        // iris fibre relief from the normal map (px_eye2 samples Normal -> eyeIrisNormal), gated to
        // the iris disc by the same mask. Tangent frame from screen-space UV derivatives.
        if (uHasEyeNormal == 1) {
            vec3 nm = texture(uEyeNormalTex, vUV).rgb * 2.0 - 1.0;
            nm = normalize(vec3(nm.xy * uIrisNormalStr, 1.0));
            vec3 Ng = normalize(vN);
            vec3 dp1 = dFdx(vWorldPos); vec3 dp2 = dFdy(vWorldPos);
            vec2 du1 = dFdx(vUV);       vec2 du2 = dFdy(vUV);
            vec3 T = normalize(dp1 * du2.y - dp2 * du1.y);
            T = normalize(T - Ng * dot(Ng, T));            // Gram-Schmidt orthonormalise
            vec3 B = cross(Ng, T);
            vec3 perturbed = normalize(T * nm.x + B * nm.y + Ng * nm.z);
            shadeN = normalize(mix(Ng, perturbed, mask));
        }
    }
    else if (uKind == 2) {                              // HAIR
        // The hair mask (strands + the green-channel gradient + alpha) is sampled on UV0, exactly as
        // the game shader px_hair2_3color_tousle does: add_191 = myTiling * Gfx_UV0 with myTiling = 1,
        // so it samples raw UV0. UV0 deliberately runs past [0,1] (~ -8..10) to tile the strand maps
        // across the cards - that tiling IS the strand detail. Only the AO uses UV1 (UVset=1 in the
        // shader). Sampling the mask on UV1 instead flattens the strands to grey ribbons.
        vec4 hm = (uHasHairMask == 1) ? texture(uHairMask, vUV)
                                      : vec4(1.0, clamp(vUV.y, 0.0, 1.0), 0.0, 1.0);
        // The alpha CUT reads the sharp top mip: a minified/angled card otherwise samples a coarse mip
        // whose averaged strand alpha, once hard-thresholded, breaks into chunky "pixelated" edges.
        // LOD 0 keeps strand silhouettes crisp at every distance; colour still uses the mipmapped hm.
        float aTest = (uHasHairMask == 1) ? textureLod(uHairMask, vUV, 0.0).a : 1.0;
        // Hair shades in LINEAR now (matching px_hair2_3color_tousle / the deferred pass), consistent
        // with skin + eye. The gradient is a weighted blend of the root/mid/tip colours, so linearising
        // the three swatches makes the whole blend linear-correct; rd and ao are 0..1 factors.
        vec3 col = hairGradient(srgb2lin(uHair1), srgb2lin(uHair2), srgb2lin(uHair3), hm.g, uSmoothness);
        float ao = (uHasHairAO == 1) ? texture(uHairAO, vUV1).r : 1.0;
        float rd = mix(1.0, hm.r, uRootDarkening);
        albedo = col * rd * ao;                          // linear albedo (gradient * root-dark * AO)
        // --- anisotropic strand sheen (Kajiya-Kay / Scheuermann). This is the piece the GAME does
        // in its deferred lighting pass from extendedOut.hairTangent + roughness; it's what makes the
        // flat albedo (gradient * AO * root-dark) read as HAIR instead of dark plates, and it's why
        // the baked AO/strand variation stops looking like "random colour steps" once it's present.
        // The strand tangent is now the AUTHORED per-vertex tangent (vTan) decoded from the mesh -
        // the same strand-flow vector the game feeds its hair sheen. This replaces the old
        // screen-space-derivative guess (which, with the previously broken normals, produced the chrome
        // look). Falls back to no sheen where vTan is zero. Preview-only: touches nothing the exporter
        // writes.
        if (uHairSpec > 0.0) {
            vec3 Tdir = vTan;
            if (dot(Tdir, Tdir) > 1e-6) {
                Tdir = normalize(Tdir);
                vec3 Lh = -normalize(uLightDir);
                vec3 Vh = normalize(uCamPos - vWorldPos);
                vec3 Hh = normalize(Lh + Vh);
                float dotTH = dot(Tdir, Hh);
                float sinTH = sqrt(max(1.0 - dotTH * dotTH, 1e-4));   // sign-free, so T orientation doesn't matter
                float rough = mix(0.5 * uHairRough, uHairRough, hm.g);  // rougher toward the tip (per shader)
                float expo  = mix(160.0, 16.0, clamp(rough, 0.0, 1.0));
                float gate  = max(dot(normalize(shadeN), Lh), 0.0) * 0.7 + 0.3;  // fade on the shadow side
                vec3  tint  = mix(vec3(1.0), srgb2lin(uHair3), 0.2);  // near-white, a touch of the tip colour (linear)
                hairSpec = pow(sinTH, expo) * gate * uHairSpec * tint * uSpecular;
            }
        }
        if (uHasHairMask == 1) {
            if (uHairSoftAlpha == 1) {
                // Soft coverage: with depth-WRITE off for this pass, overlapping cards blend their
                // colours instead of each pixel snapping to one frontmost card. That's what stops the
                // per-card gradient phases tiling into hard colour blocks. Boost partial coverage so
                // thin strands still read; drop only the genuinely empty texels.
                alpha = clamp(pow(aTest, 0.7) * 1.1, 0.0, 1.0);
                if (alpha < 0.02) discard;
            } else {
                // Hard cutoff (original): crisp silhouette, one opaque card per pixel.
                float cutoff = clamp(0.5 / max(uHairCoverage, 0.1), 0.05, 0.95);
                if (aTest < cutoff) discard;
                alpha = 1.0;
            }
        }
    }
    else if (uKind == 4) {                              // EYELASH card (own alpha texture, gender-shared)
        vec4 c = (uHasColor == 1) ? texture(uColorTex, vUV) : vec4(0.05, 0.05, 0.05, 1.0);
        albedo = c.rgb * uLashTint;
        // Lashes are fine, sparse, soft-alpha strands: keep true alpha blending and gently boost the
        // partial-alpha texels so thin strands read fuller (NOT a hard cutoff, which discards the soft
        // tips and leaves them thin/broken - that was the regression).
        alpha = clamp(pow(c.a, 0.65) * 1.1, 0.0, 1.0);
        if (alpha < 0.01) discard;
    }
    else if (uKind == 7) {                              // CORNEA / SCLERA SHELL (PX_Eye_Shell)
        // The "eyeEdge" mesh is the cornea/sclera shell; in-game its white + opacity come from
        // vertex colour (R=alpha, G=shade) over the material's myColor/myShadowColor. Until the
        // vertex-colour attribute is wired through, a faithful near-white sclera fill is the right
        // approximation (the shell reads white where vertex colour is high, which is most of it).
        albedo = vec3(0.92, 0.91, 0.88);
    }
    else if (uKind == 6) {                              // EYE-SHADOW overlay (PX_Eye_Shadow)
        // Transparent dark contact shadow around the eye; normally skipped (needs alpha blending).
        // If ever drawn, keep it dark and let the FLAT alpha path handle discard.
        albedo = vec3(0.05, 0.05, 0.05);
    }
    else {                                              // FLAT / textured
        if (uHasColor == 1) {
            vec4 c = texture(uColorTex, vUV);
            albedo = c.rgb;
            alpha = c.a;
            if (alpha < 0.02) discard;
        } else {
            albedo = vec3(0.45, 0.44, 0.42);
        }
        // camo coverage (1 = whole surface). Set from the _m alpha inside the camo block below; left
        // at 1 for plain accessory / band pieces so their _n AO applies across the whole piece.
        float cover = 1.0;
        // Camouflage preview: triplanar-project the region mask in MODEL space (matches the game
        // camo shader - it never uses the gear's own UVs) and blend the three palette colours by
        // the mask channels. Gated by uCamoEnable, so non-camo gear and every other uKind are
        // untouched. The flat path's albedo is sRGB-space, so the linear camo result is encoded.
        if (uCamoEnable == 1 && uHasCamoMask == 1) {
            vec3 gearDiffuse = albedo;                    // sRGB gear diffuse, before any camo
            vec3 cn = abs(normalize(vN));
            float cca = cos(radians(uCamoRotation));
            float csa = sin(radians(uCamoRotation));
            mat2 crot = mat2(cca, -csa, csa, cca);
            vec4 mTop   = texture(uCamoMask, crot * (vWorldPos.xz * uCamoTiling));  // top   (x,z)
            vec4 mSide  = texture(uCamoMask, crot * (vWorldPos.xy * uCamoTiling));  // side  (x,y)
            vec4 mFront = texture(uCamoMask, crot * (vWorldPos.zy * uCamoTiling));  // front (z,y)
            vec4 cm = mix(mTop, mSide, cn.z);
            cm = mix(cm, mFront, cn.x);
            cover = (uHasGearMat == 1) ? texture(uGearMatTex, vUV).a : 1.0;
            camoCover = cover;                            // default; the cloth-region branch overrides
            if (uCamoBlendMode == 1) {
                // CLOTH (blinn-phong cloth 3coloroverlay.mshader): tint the diffuse with the palette via
                // Photoshop "Overlay" blend (keeps the base luminance, so weave/wear/AO survive; no 0.7
                // clamp). Overlay R/G/B/A onto the diffuse, each gated by a region weight; A is the base.
                if (uHasGearRegion == 1) {
                    // Faithful: the garment's baked ColorMask (UV0) defines up to 4 regions. Camo (the
                    // triplanar pattern) only enters the GREEN region (.y), re-driving the R/G/B mix
                    // there - exactly m = lerp(ColorMask, vec4(camo.rgb,1), ColorMask.y * camoEnable).
                    vec4 colorMask = texture(uGearRegionTex, vUV);
                    vec4 m = mix(colorMask, vec4(cm.rgb, 1.0), colorMask.y);
                    vec3 col = gearDiffuse;
                    col = mix(col, overlayBlend(gearDiffuse, lin2srgb(uCamoColA)), m.w);
                    col = mix(col, overlayBlend(gearDiffuse, lin2srgb(uCamoColB)), m.z);
                    col = mix(col, overlayBlend(gearDiffuse, lin2srgb(uCamoColG)), m.y);
                    col = mix(col, overlayBlend(gearDiffuse, lin2srgb(uCamoColR)), m.x);
                    albedo = col;
                    camoCover = clamp(m.x + m.y + m.z + m.w, 0.0, 1.0);
                } else {
                    // No ColorMask loaded: treat the whole covered area as the camo zone (the triplanar
                    // pattern drives the 3-colour overlay), gated by the _m alpha. A base first (identity).
                    vec3 col = gearDiffuse;
                    col = mix(col, overlayBlend(gearDiffuse, lin2srgb(uCamoColA)), 1.0);
                    col = mix(col, overlayBlend(gearDiffuse, lin2srgb(uCamoColB)), cm.b);
                    col = mix(col, overlayBlend(gearDiffuse, lin2srgb(uCamoColG)), cm.g);
                    col = mix(col, overlayBlend(gearDiffuse, lin2srgb(uCamoColR)), cm.r);
                    albedo = mix(gearDiffuse, col, cover);
                }
            } else {
                // WEAPON (blinn-phong camouflage_rse.mshader): REPLACE the diffuse with the flat clamped
                // region colour. graph tail: lerp(diffuse, lerp(camo, diffuse, mask.a*alphaEnable), cover).
                vec3 camo = uCamoColR * cm.r + uCamoColG * cm.g + uCamoColB * cm.b
                            + (cm.r * cm.g * cm.b);
                camo = clamp(camo, 0.025, 0.7);
                vec3 camoSrgb = lin2srgb(camo);
                vec3 camoOrDiff = mix(camoSrgb, gearDiffuse, cm.a * float(uCamoAlphaEnable));
                albedo = mix(gearDiffuse, camoOrDiff, cover);
            }
            // The game's camo shader is a DEFERRED G-buffer pass: it writes the flat camo albedo with
            // roughness 0.7 / metalness 0 (matte paint) PLUS the real normal map + baked AO, and the
            // lighting pass shades all of it. The unlit flat path threw the normal/AO away (-> flat
            // clay). The reconstruction now lives just below the camo block so it also runs for plain
            // accessory / band pieces (uCamoEnable 0), not only camo'd gear.
        }
        // Material (_m) + normal (_n) for accessory / band pieces (and any flat gear). The _n is
        // UnpackNormalAndAO (rgb = tangent normal, a = baked AO): apply the AO (scaled by `cover`, so
        // camo only darkens its painted zones while plain pieces darken fully) and perturb the shading
        // normal so beads / feathers / leather read as relief instead of flat clay. Validated skin-
        // normal math; gated by uHasGearNormal, so a piece with no _n bound is unchanged.
        if (uHasGearNormal == 1) {
            vec4 gn = texture(uGearNormalTex, vUV);
            float ao = clamp(gn.a, 0.0, 1.0);
            albedo *= mix(1.0, ao, cover);
            vec3 nm = vec3((gn.xy * 2.0 - 1.0) * uGearNormalStr, 1.0);
            vec3 Ng = normalize(vN);
            vec3 dp1 = dFdx(vWorldPos); vec3 dp2 = dFdy(vWorldPos);
            vec2 du1 = dFdx(vUV);       vec2 du2 = dFdy(vUV);
            vec3 T = normalize(dp1 * du2.y - dp2 * du1.y);
            T = normalize(T - Ng * dot(Ng, T));       // Gram-Schmidt
            vec3 B = cross(Ng, T);
            shadeN = normalize(T * nm.x + B * nm.y + Ng);
        }
    }

    vec3 N = normalize(shadeN);
    // Two-sided meshes (hair cards, body membrane, eyelashes) render with cull OFF, so a card's BACK
    // face is drawn with the same authored normal as the front - which points away from the viewer and
    // usually away from the light, collapsing ndl and turning the card black. Flip the normal to face
    // the camera on back faces (the textbook two-sided-lighting fix). Front-face-only meshes (skin,
    // head: cull ON) always see gl_FrontFacing == true, so this is a no-op for them.
    if (!gl_FrontFacing) N = -N;
    float ndl = max(dot(N, -normalize(uLightDir)), 0.0);
    vec3 outRGB;
    if (uKind == 0) {
        // SKIN lit by the game's own environment irradiance (SH projected from the character-render
        // scene's cubemap): a per-normal ambient - top-down dominant, cool sky tint, dark underside -
        // replacing the old flat ambient+wrap floor that pinned every texel to ~0.9 and washed the
        // form out. uIblIntensity is the master exposure (the scene's absolute exposure isn't in the
        // env file); uKeyLight adds a frontal directional fill only if the scene has a key light.
        vec3 amb  = shAmbient(N) * uIblIntensity;
        // The only baked cubemap in the export is an outdoor blue-sky map, but this render scene has
        // the sky DISABLED - so its fill shouldn't carry that blue. uIblSaturation pulls the ambient
        // toward neutral grey (keeping the top-down form), and uIblTint white-balances it onto the
        // game's measured skin channel ratio. Both default to remove the excess blue.
        amb = mix(vec3(dot(amb, vec3(0.2126, 0.7152, 0.0722))), amb, uIblSaturation) * uIblTint;
        float key = ndl * uKeyLight;
        vec3 hdr  = albedo * (amb + key) + emission;                          // linear HDR
        // --- CLEAR-COAT SKIN SHEEN (px_character_navi: extendedOut.clearCoat 0.08-0.20,
        // clearCoatRoughness 0.2-0.7, clearCoatSkin=1). A thin dielectric coat over the diffuse+SSS
        // that REFLECTS the environment, weighted by Schlick Fresnel (F0=0.04): nearly invisible
        // head-on, brightening at grazing angles - the soft waxy sheen the game gives Na'vi skin along
        // cheekbones / shoulders / brow. The coat is rough, so the blurry SH irradiance sampled in the
        // reflection direction is a fair stand-in for a prefiltered specular probe. uClearCoatRough
        // blends the reflection toward the diffuse-normal ambient as it gets rougher (broader sheen).
        vec3 V   = normalize(uCamPos - vWorldPos);
        float NoV = max(dot(N, V), 1e-4);
        float fres = 0.04 + 0.96 * pow(1.0 - NoV, 5.0);                       // Schlick, dielectric F0
        vec3 Rv  = reflect(-V, N);
        vec3 envR = shAmbient(mix(Rv, N, clamp(uClearCoatRough, 0.0, 1.0))) * uIblIntensity;
        envR = mix(vec3(dot(envR, vec3(0.2126, 0.7152, 0.0722))), envR, uIblSaturation) * uIblTint;
        vec3 coat = envR * fres * uClearCoat;
        // a soft key-light glint too, if a key light is enabled (Blinn lobe widened by roughness)
        if (uKeyLight > 0.0) {
            vec3 H = normalize(-normalize(uLightDir) + V);
            float a = max(uClearCoatRough, 0.04);
            float expo = max(2.0 / (a * a) - 2.0, 1.0);                       // roughness -> Blinn exponent
            coat += vec3(pow(max(dot(N, H), 0.0), expo) * fres) * uKeyLight * uClearCoat * ndl;
        }
        hdr += coat * uSpecular;                                             // add the sheen (linear)
        vec3 tm   = (hdr * (2.51*hdr + 0.03)) / (hdr * (2.43*hdr + 0.59) + 0.14);  // ACES, linear
        vec3 lin  = mix(hdr, tm, clamp(uTonemap, 0.0, 1.0));
        outRGB = lin2srgb(clamp(lin, 0.0, 1.0));                              // -> display
    } else if (uKind == 1) {
        // EYE: albedo is LINEAR now (px_eye2). Light + tonemap in linear, encode once - same as skin.
        float lit = 0.30 + 0.70 * ndl;
        vec3 hdr = albedo * lit + emission;
        vec3 tm  = (hdr * (2.51*hdr + 0.03)) / (hdr * (2.43*hdr + 0.59) + 0.14);
        vec3 lin = mix(hdr, tm, clamp(uTonemap, 0.0, 1.0));
        outRGB = lin2srgb(clamp(lin, 0.0, 1.0));
    } else if (uKind == 2) {
        // HAIR: albedo + sheen are LINEAR now. Light + tonemap in linear, encode once - like skin/eye.
        float lit = 0.30 + 0.70 * ndl;
        vec3 hdr = albedo * lit + hairSpec + emission;
        vec3 tm  = (hdr * (2.51*hdr + 0.03)) / (hdr * (2.43*hdr + 0.59) + 0.14);
        vec3 lin = mix(hdr, tm, clamp(uTonemap, 0.0, 1.0));
        outRGB = lin2srgb(clamp(lin, 0.0, 1.0));
    } else {
        // FLAT / LASH / shells: unchanged sRGB-space path.
        float lit = 0.30 + 0.70 * ndl;
        vec3 hdr = albedo * lit + hairSpec + emission;
        // CAMO matte-paint sheen: the game writes camo zones as roughness 0.7 / metalness 0, so add a
        // broad, dim dielectric Blinn lobe over the (now normal-mapped) surface, gated to the painted
        // zones by camoCover. This plus the perturbed normal is what turns flat camo into a lit surface.
        if (uCamoEnable == 1 && camoCover > 0.0) {
            vec3 Vc = normalize(uCamPos - vWorldPos);
            vec3 Lc = -normalize(uLightDir);
            vec3 Hc = normalize(Lc + Vc);
            float ac   = 0.7;                                  // camo-zone roughness
            float expo = max(2.0 / (ac * ac) - 2.0, 1.0);      // roughness -> Blinn exponent (~2)
            float sp   = pow(max(dot(N, Hc), 0.0), expo) * ndl;
            hdr += vec3(sp * 0.10 * camoCover) * uSpecular;    // F0~0.04 dielectric, lifted for preview
        }
        // Accessory / band material (_m): a dim dielectric Blinn sheen driven by the _m roughness
        // (.g -> lobe width) and metalness (.r -> a little more punch), so polished beads / metal
        // bands catch the key light while leather / cloth stay matte. Non-camo only (camo has its own
        // sheen above); gated by uHasGearMat + uSpecular, so a piece with no _m bound is unchanged.
        else if (uHasGearMat == 1) {
            vec4 gm = texture(uGearMatTex, vUV);
            float rough = clamp(gm.g, 0.06, 1.0);
            vec3 Va = normalize(uCamPos - vWorldPos);
            vec3 La = -normalize(uLightDir);
            vec3 Ha = normalize(La + Va);
            float expo = max(2.0 / (rough * rough) - 2.0, 1.0);     // roughness -> Blinn exponent
            float sp   = pow(max(dot(N, Ha), 0.0), expo) * ndl;
            hdr += vec3(sp * (0.06 + 0.10 * gm.r)) * uSpecular;     // dim, preview-lifted
        }
        vec3 tm  = clamp((hdr * (2.51*hdr + 0.03)) / (hdr * (2.43*hdr + 0.59) + 0.14), 0.0, 1.0);
        outRGB = mix(clamp(hdr, 0.0, 1.0), tm, clamp(uTonemap, 0.0, 1.0));
    }
    frag = vec4(outRGB, alpha);
}
"""


def navi_fragment_source():
    """NAVI_GLSL helper block spliced in right after the #version line of NAVI_FRAG."""
    main = NAVI_FRAG.lstrip("\n")
    head, rest = main.split("\n", 1)
    return head + "\n" + gs.NAVI_GLSL + "\n" + rest


def navi_default():
    """Identity colour state (0.5 grey overlay no-op, no warpaint/bio, visible hair/iris)."""
    return dict(
        # Open on the real default-Na'vi look: item_customization_player_skin_color_001.blueitemtype
        # myColor1 #ADCBD7 (base skin) / myColor2 #848B9A (pattern), ARGB. So the skin + stripes read
        # as the game's default Na'vi until the user changes a colour. _NGREY_LIN is still the overlay-
        # identity value if a colour is reset to neutral.
        skin=(0.6784, 0.7961, 0.8431),
        pattern=(0.5176, 0.5451, 0.6039),
        paint1=_NGREY_LIN,
        paint2=_NGREY_LIN,
        paint3=_NGREY_LIN,
        paint4=_NGREY_LIN,
        hide_warpaint=False,
        outer_r=(0.807843, 0.658824, 0.431373),
        inner_r=(0.85098, 0.901961, 0.513725),
        outer_l=(0.807843, 0.658824, 0.431373),
        inner_l=(0.85098, 0.901961, 0.513725),
        hair1=(0.419608, 0.419608, 0.419608),
        hair2=(0.121569, 0.121569, 0.121569),
        hair3=(0.196078, 0.196078, 0.196078),
        haircap=_NGREY_LIN,
        has_haircap_color=False,
        hair_cap_strength=1.0,
        bio_color=tuple(float(x) for x in rc.BIO_GREEN),
        bio_enabled=False,
        bio_brightness=1.0,
        bio_pulsation=0.0,
        smoothness=1.0,
        root_darkening=0.35,
        tonemap=0.0,
        lash_tint=(0.22, 0.20, 0.18),
        skin_color_weight=1.0,
        pattern_strength=1.0,
        skin_desaturate=0.85,
        ibl_intensity=2.5,
        key_light=0.0,
        ibl_saturation=0.0,
        ibl_tint=(1.0, 1.0, 1.0),
        clear_coat=0.6,
        clear_coat_rough=0.6,
    )


    # =====================================================================================
    # Armature binding: the .mmb embeds a skeleton + per-mesh bind matrix. For a STATIC preview the
    # head/body/hair are authored in one shared character space, so we draw raw positions; the per-mesh
    # "bind" matrices are skinning data and applying them as placement shears the parts. Calibration
    # switches (no code edits): NAVI_APPLY_BIND (False = raw positions), NAVI_BIND_MODE ("mesh_bind" |
    # "root_bone" | "none"), NAVI_BIND_TRANSPOSE (flip if matrices are row-vector p*M). Leave bind OFF
    # until per-vertex skinning is wired and the .mmb matrix layout is verified against a real file.
    # =====================================================================================
NAVI_APPLY_BIND = False
NAVI_BIND_MODE = "none"
NAVI_BIND_TRANSPOSE = False
_NFLIP_X = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32)

_NFLIP_X = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32)

# Hardcoded Na'vi tail curl: every tail joint (C_tail_02..15) rotates about its LOCAL Y axis (this
# rig: Y = pitch/down, X = roll, Z = yaw). Forward kinematics (navi_pose_skin) accumulates them into a
# smooth droop with each downstream bone following. Applied once as a static CPU skin at load (only
# tail-weighted verts move; normals untouched). NAVI_POSE_ENABLED = False for the raw bind pose.
NAVI_POSE_ENABLED = True
NAVI_POSE = {"C_tail_%02d" % k: (-8.0, "y") for k in range(2, 16)}


def navi_world_bones(skeleton):
    """World bind matrix per bone, assuming each bone['matrix'] is LOCAL (relative to parent).
    parent index out of range (e.g. 0xFFFF) = root. Returns list of 4x4 float32."""
    n = len(skeleton)
    world = [None] * n
    stack = []

    def resolve(i):
        if world[i] is not None:
            return world[i]
        b = skeleton[i]
        M = np.asarray(b.get("matrix"), np.float32).reshape(4, 4)
        p = b.get("parent", -1)
        if 0 <= p < n and p != i and i not in stack:
            stack.append(i)
            world[i] = (resolve(p) @ M).astype(np.float32)
            stack.pop()
        else:
            world[i] = M
        return world[i]

    return [resolve(i) for i in range(n)]


def _navi_sane(M):
    """Reject a non-finite or absurdly-translated matrix (bad parse) -> fall back to identity."""
    if M is None or getattr(M, "shape", None) != (4, 4) or not np.all(np.isfinite(M)):
        return False
    return float(np.max(np.abs(M[:3, 3]))) < 1.0e6


def navi_placement(mesh_extra, skeleton_world):
    """4x4 placement matrix for one mesh, in the viewer's FLIP_X frame. Returns identity when
    binding is off / data is missing / the matrix looks invalid."""
    if not NAVI_APPLY_BIND or NAVI_BIND_MODE == "none" or not mesh_extra:
        return np.eye(4, dtype=np.float32)
    if NAVI_BIND_MODE == "root_bone":
        rb = mesh_extra.get("root_bone", -1)
        M = (
            skeleton_world[rb]
            if (skeleton_world and 0 <= rb < len(skeleton_world))
            else None
        )
    else:  # "mesh_bind"
        M = mesh_extra.get("bind")
    if not _navi_sane(M):
        return np.eye(4, dtype=np.float32)
    M = np.asarray(M, np.float32).reshape(4, 4)
    if NAVI_BIND_TRANSPOSE:
        M = M.T.copy()
    Mp = (_NFLIP_X @ M @ _NFLIP_X).astype(np.float32)  # conjugate into the FLIP_X frame
    return Mp if _navi_sane(Mp) else np.eye(4, dtype=np.float32)


def navi_apply_placement(P, N, M):
    """Apply a 4x4 placement to positions (n,3) and normals (n,3). No-op for identity."""
    if np.allclose(M, np.eye(4, dtype=np.float32)):
        return P, N
    Ph = np.concatenate([P, np.ones((P.shape[0], 1), np.float32)], 1)
    P2 = (Ph @ M.T)[:, :3].astype(np.float32)  # column-vector: p' = M p
    N2 = (N @ M[:3, :3].T).astype(np.float32)
    ln = np.linalg.norm(N2, axis=1, keepdims=True)
    N2 = np.divide(N2, ln, out=np.zeros_like(N2), where=ln > 1e-9)
    return np.ascontiguousarray(P2), np.ascontiguousarray(N2)


def _tail_rot_mat4(deg, axis="x"):
    """4x4 rotation of `deg` degrees about the given LOCAL axis ('x'/'y'/'z'), column-vector
    convention (p' = M p)."""
    r = np.radians(float(deg))
    c, s = np.cos(r), np.sin(r)
    M = np.eye(4, dtype=np.float32)
    a = (axis or "x").lower()
    if a == "x":
        M[1, 1], M[1, 2], M[2, 1], M[2, 2] = c, -s, s, c
    elif a == "y":
        M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s, -s, c
    else:  # "z"
        M[0, 0], M[0, 1], M[1, 0], M[1, 1] = c, -s, s, c
    return M


def navi_pose_skin(skeleton):
    """Skinning matrices for NAVI_POSE, in the viewer's FLIP_X mesh frame. Returns {global_bone_idx:
    4x4} for every bone the pose actually moves - the directly-rotated joints AND every bone
    downstream of them, which inherit the rotation through forward kinematics (so the whole tail
    follows, not just the base). Empty {} when the pose is off or none of the posed bones are here.

    The .mmb stores bone matrices ROW-VECTOR (translation in the bottom row), so they're transposed
    to the column-vector convention the rest of the skinning uses - without that the tail rotates
    about the world origin and shears apart. Skin = pose_world @ inverse(rest_world), conjugated into
    the FLIP_X frame like navi_placement."""
    if not (NAVI_POSE_ENABLED and NAVI_POSE and skeleton):
        return {}
    name_to_idx = {(b.get("name") or "").lower(): i for i, b in enumerate(skeleton)}
    pose_l = {k.lower(): v for k, v in NAVI_POSE.items()}
    if not any(nm in name_to_idx for nm in pose_l):
        log.info("navi tail pose: none of the posed bones are in this skeleton - left un-posed")
        return {}
    n = len(skeleton)
    # local bind per bone (relative to parent), transposed row-vector -> column-vector
    L = [np.asarray(b.get("matrix"), np.float32).reshape(4, 4).T for b in skeleton]

    def _fk(locals_):
        w = [None] * n

        def resolve(i, stack):
            if w[i] is not None:
                return w[i]
            p = skeleton[i].get("parent", -1)
            if 0 <= p < n and p != i and i not in stack:
                w[i] = (resolve(p, stack + [i]) @ locals_[i]).astype(np.float32)
            else:
                w[i] = locals_[i]
            return w[i]

        for i in range(n):
            resolve(i, [])
        return w

    rest_world = _fk(L)
    # posed local: rotate each posed bone about its own local axis (L_rest @ R)
    posed_local = list(L)
    matched = []
    for i, b in enumerate(skeleton):
        spec = pose_l.get((b.get("name") or "").lower())
        if spec is not None:
            deg, axis = spec if isinstance(spec, tuple) else (spec, "y")
            posed_local[i] = (L[i] @ _tail_rot_mat4(deg, axis)).astype(np.float32)
            matched.append(b.get("name"))
    pose_world = _fk(posed_local)
    # keep skin only for bones that actually move (posed bones + their descendants)
    out = {}
    I4 = np.eye(4, dtype=np.float32)
    for i in range(n):
        if not _navi_sane(rest_world[i]):
            continue
        try:
            S = (pose_world[i] @ np.linalg.inv(rest_world[i])).astype(np.float32)
        except np.linalg.LinAlgError:
            continue
        Sf = (_NFLIP_X @ S @ _NFLIP_X).astype(np.float32)
        if not np.allclose(Sf, I4, atol=1e-5):
            out[i] = Sf
    log.info(
        "navi tail pose: %d joints rotated (%s%s), %d bones affected",
        len(matched), ", ".join(matched[:4]), "..." if len(matched) > 4 else "", len(out),
    )
    return out


def navi_apply_pose(P, N, weights, influences, skin_by_bone):
    """Deform (n,3) positions P and normals N by NAVI_POSE. `weights` is the per-vertex list of
    {mesh_slot: weight} from mmb_loader; `influences` maps slot -> (inv_bind, global_bone_idx);
    `skin_by_bone` = {global_bone_idx: 4x4} from navi_pose_skin. Only vertices weighted to a posed
    bone move (delta form - every other bone is identity), so this is a no-op for meshes/verts the
    pose doesn't touch. Returns P, N unchanged when there's nothing to apply."""
    if not skin_by_bone or not weights or not influences:
        return P, N
    n = P.shape[0]
    if len(weights) != n:
        return P, N
    slot_bone = [inf[1] for inf in influences]  # mesh slot -> global bone index
    # per-posed-bone vertex weight column (0 where a vertex isn't weighted to that bone)
    wcol = {bi: np.zeros(n, np.float32) for bi in skin_by_bone}
    for v, w in enumerate(weights):
        if not w:
            continue
        for slot, wt in w.items():
            if 0 <= slot < len(slot_bone):
                bi = slot_bone[slot]
                if bi in wcol:
                    wcol[bi][v] += wt
    Ph = np.concatenate([P, np.ones((n, 1), np.float32)], 1)
    Pout = P.astype(np.float32).copy()
    Nout = N.astype(np.float32).copy()
    I4 = np.eye(4, dtype=np.float32)
    touched = False
    moved = {}
    for bi, S in skin_by_bone.items():
        wb = wcol[bi]
        moved[bi] = int(np.count_nonzero(wb))
        if not np.any(wb):
            continue
        touched = True
        d = S - I4  # blend the delta (skin - identity) by the vertex's weight to this bone
        Pout += wb[:, None] * (Ph @ d.T)[:, :3]
        Nout += wb[:, None] * (N @ d[:3, :3].T)
    # verts moved per posed bone: if child bones show 0 here, the mesh isn't weighted to them (so
    # they can't curl - it's the model's rigging, not the FK). {bone_idx: vertex_count}
    if any(moved.values()):
        log.info("navi tail pose: verts moved per bone %s", moved)
    if not touched:
        return P, N
    ln = np.linalg.norm(Nout, axis=1, keepdims=True)
    Nout = np.divide(Nout, ln, out=Nout, where=ln > 1e-9)
    return np.ascontiguousarray(Pout), np.ascontiguousarray(Nout)


def navi_comb_normals(P, N, center, blend=0.85):
    """Re-comb hair-card normals toward the scalp.

    The .mmb loader recomputes normals from faces, which for flat hair cards yields one flat
    facet normal per card - so adjacent cards catch light at random angles and the hair reads as
    abrupt per-card tone steps rather than a coherent volume. Real hair-card normals are authored
    'combed' radially outward from the head. We approximate that: blend each normal toward the
    outward radial direction from `center` (the head centroid). blend 0 = keep card facets,
    1 = pure radial dome. Positions/normals are in the same (placed) space.
    """
    out = P - np.asarray(center, np.float32)[None, :]
    ln = np.linalg.norm(out, axis=1, keepdims=True)
    out = np.divide(out, ln, out=np.zeros_like(out), where=ln > 1e-6)
    mixed = (1.0 - blend) * N + blend * out
    mn = np.linalg.norm(mixed, axis=1, keepdims=True)
    mixed = np.divide(mixed, mn, out=N.copy(), where=mn > 1e-6)
    return np.ascontiguousarray(mixed.astype(np.float32))


class BansheeViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = self.format()
        # No default-framebuffer MSAA: SSAA (the offscreen resolve below) is the real AA here - it
        # antialiases the alpha-discard hair/membrane cutouts, which MSAA cannot. Keeping MSAA off
        # lets the "Anti-aliasing: Off" setting be genuinely off instead of a residual 4x.
        fmt.setSamples(0)
        self.setFormat(fmt)
        self.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )  # so the viewer receives key presses
        self.ctx = None
        self.prog = None
        self.grid_prog = None
        self.grid_vao = None
        self.grid_vbo = None
        self.floor_y = 0.0
        self.synth = None  # shared synthetic pattern texture
        self.gl_objs = []  # (vao,vbo,ibo,name,key)
        self.tex = {
            "body": {},
            "head": {},
            "wing": {},
            "eye": {},
        }  # key -> role -> Texture
        self.palettes = {"body": [(0, 0, 0)] * 10, "head": [(0, 0, 0)] * 10}
        # neutral defaults: invert2=0 keeps the pre-load surface on Coat 1 (no splotch);
        # the real per-skin constants are resolved from the colour pattern on load.
        self.params = {
            k: dict(invert1=1.0, invert2=0.0, level1=1.0, level2=1.0)
            for k in ("body", "head")
        }
        self.center = np.zeros(3, np.float32)
        self.radius = 1.0
        self.az, self.el, self.dist = 35.0, 12.0, 3.0
        self.pan = np.zeros(3, np.float32)
        self.flip_v = False
        # normal-mapping controls (tweak if detail reads inverted / too strong).
        # normal_strength = base _n shape; detail_weight = tiled micro-grain overlay.
        # Both were 1.0 (raw game values); dialled down because the combined base +
        # 3 detail layers read too strong. Raise toward 1.0 for more relief.
        self.detail_tiling = (8.0, 8.0, 8.0)  # UV tiling per skin_detail normal
        self.normal_strength = 0.5  # base-normal (_n) strength
        self.detail_weight = 0.35  # overall detail-normal strength
        self.normal_yflip = -1.0  # DirectX-style green; flip to +1 if inverted
        self._last = QPoint()
        self._w, self._h = 1, 1
        self._pending_model = None
        self._pending_tex = []
        # ---- Na'vi mode (second render mode in this one widget/context) ----
        self.mode = "banshee"  # "banshee" | "navi"
        self._frames = {}  # mode -> saved camera framing
        self.navi_prog = None
        self.navi_objs = []  # (vao,vbo,ibo,name,kind,bucket)
        self.navi_armature = []  # master skeleton (from the body mesh)
        self.navi_armature_world = []  # cached world bind matrices (for posing)
        self.navi_tex = {b: {} for b in NAVI_BUCKETS}  # bucket -> role -> Texture
        self.navi_fallback = None  # 1x1 white
        self.navi = navi_default()
        self._specular = (
            True  # specular highlights (clear-coat + hair sheen); config-toggled
        )
        self.navi_time = 0.0
        self._navi_pending_meshes = None
        self._navi_pending_tex = []
        self._navi_built_for = None
        self.navi_tail_pose = True  # apply the hardcoded Na'vi tail curl (toggle below the viewer)
        # ---- gear pieces rendered on top of the Na'vi (flat-textured with a baked diffuse) ----
        self.gear_objs = []  # (vao,vbo,ibo,name,key)
        self.gear_hidden = set()  # keys currently hidden via 'Hide Gear'
        self.body_hidden = False  # 'Hide Ikran' / 'Hide Na'vi' — skip the creature/character body
        self.all_gear_hidden = False  # 'Hide all gear' — skip every gear piece at once
        self.gear_tex = {}  # key -> Texture (recoloured diffuse)
        self.gear_mat_tex = {}  # key -> Texture (gear _m map; alpha is the camo coverage)
        self.gear_normal_tex = {}  # key -> Texture (gear _n map; rgb=normal, a=AO; lit camo path)
        self.gear_region_tex = {}  # key -> Texture (cloth ColorMask _reg_m; 4 baked regions, UV0)
        self._gear_pending = {}  # key -> (mesh_path, diffuse_rgba) before GL init
        self._gear_mat_pending = {}  # key -> material_rgba before GL init
        self._gear_normal_pending = {}  # key -> normal_rgba before GL init
        self._gear_region_pending = {}  # key -> region_rgba before GL init
        # ---- camo preview state (applied only to 'camo:' gear keys; see _draw_gear_pieces) ----
        self._gear_camo = {}  # gear key -> (R, G, B) linear float3 (per-piece camo); absent = off
        self._camo_mask_tex = None  # the w_camo_solid region mask Texture, or None
        self._camo_mask_pending = None  # RGBA array set before GL init
        self._camo_tiger_tex = None  # the w_camo_tigerstripe region mask Texture, or None
        self._camo_tiger_pending = None
        self._gear_camo_pattern = {}  # gkey -> "solid" | "tigerstripe" (per-piece mask choice)
        self._camo_tiling = (4.0, 4.0)  # myCamoTiling
        self._camo_rotation = 0.0  # myCamoRotation (degrees)
        self._camo_alpha_enable = False  # myCamoAlphaEnable: mask-alpha punches the diffuse back
        self._gear_normal_str = 1.0  # lit camo path: gear _n xy strength (raise/lower for more relief)
        # cloth 4th overlay (myColorOverlay_A) base: linear value whose lin2srgb == 0.5, so Overlay() is
        # an identity (the A region shows the diffuse) until/unless a real 4th palette colour is wired.
        self._camo_col_a = (0.214041, 0.214041, 0.214041)
        self._iris_mean_lum = (
            0.38  # updated from the iris texture on upload (fibre-detail centre)
        )
        self._skin_mean = {}  # bucket -> mean skin RGB, for head/body tone-matching
        self._skin_ao_mean = {}  # bucket -> mean AO, for relative (centred) AO application
            # ---- SSAA (supersampling) ----------------------------------------------------------------
            # Render into an offscreen buffer at ssaa_scale x pixel size, then downsample. This
            # supersamples the whole fragment (hair-mask lookup, alpha-test cutout, lighting) - the only
            # thing that removes hair-strand pixelation (MSAA only AAs triangle edges, not alpha discard).
            # On by default; set_ssaa() toggles. Falls back to direct render if the resolve prog fails to
            # build or the target would exceed GL_MAX_TEXTURE_SIZE.
        self.ssaa = True
        self.ssaa_scale = 2.0
        self._ssaa_fbo = None
        self._ssaa_color = None
        self._ssaa_depth = None
        self._ssaa_size = (0, 0)
        self._ssaa_cap = 8192
        self._resolve_prog = None
        self._resolve_vao = None
        self._resolve_vbo = None
        # FXAA: cheap post-process AA, its own resolve program over the offscreen buffer. On its own
        # it renders the scene to a 1x offscreen buffer then FXAA-resolves to the screen.
        self.fxaa = False
        self._fxaa_prog = None
        self._fxaa_vao = None

        # ---- anisotropic texture filtering --------------------------------------------------------
        # Every mipmapped material texture (built through _tex_from_np) is registered here so
        # set_anisotropy() can re-apply a new level to the live textures without a reload. 1.0 =
        # isotropic (off). Clamped to the GPU max on apply. Applies to ALL viewers (there is one
        # shared BansheeViewer) since every texture routes through _tex_from_np.
        self.aniso = 16.0
        self._live_textures = []

    # ---------------- public API ----------------
    def set_ssaa(self, enabled):
        """Enable/disable supersampling. Safe to call before GL init (just sets the flag)."""
        self.ssaa = bool(enabled)
        self.update()

    def set_ssaa_scale(self, scale):
        """Set the anti-aliasing supersample scale. 1.0 (or less) = off (direct render, no AA);
        1.5/2/3/4 supersample the whole fragment (hair/cutout-safe). The offscreen buffer rebuilds
        itself on the next frame when the target size changes. Safe before GL init."""
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        self.ssaa_scale = max(1.0, scale)
        self.ssaa = self.ssaa_scale > 1.0
        if self.ssaa:
            log.info("anti-aliasing: SSAA %.3gx enabled", self.ssaa_scale)
        else:
            log.info("anti-aliasing: off")
        self.update()

    def set_fxaa(self, enabled):
        """Enable/disable FXAA - a cheap single-pass post-process anti-aliasing on the resolved
        image. On its own it renders to a 1x offscreen buffer and FXAA-resolves to the screen; it
        smooths luma edges but (unlike SSAA) cannot fix alpha-cut hair pixelation. Safe before GL
        init (the program builds on init; if it isn't available FXAA silently no-ops)."""
        self.fxaa = bool(enabled)
        log.info("anti-aliasing: FXAA %s", "on" if self.fxaa else "off")
        self.update()

    def set_anisotropy(self, level):
        """Set anisotropic filtering (1 = off/isotropic, up to the GPU max, typically 16). Applied
        live to every registered mipmapped texture; new textures pick it up from self.aniso. Safe
        before GL init (stored; applied when textures exist)."""
        try:
            level = float(level)
        except (TypeError, ValueError):
            level = 1.0
        cap = float(getattr(self.ctx, "max_anisotropy", 16.0)) if self.ctx is not None else 16.0
        requested = level
        self.aniso = max(1.0, min(level, cap))
        applied = 0
        failed = 0
        alive = []
        for t in self._live_textures:
            try:
                t.anisotropy = self.aniso  # moderngl clamps to the GL max internally too
                alive.append(t)
                applied += 1
            except Exception as e:  # noqa: BLE001 - drop released/dead textures, but log why
                failed += 1
                log.debug("anisotropy apply skipped a texture: %s", e)
        self._live_textures = alive
        clamp_note = (
            " (requested %.3g, clamped to GPU max %.3g)" % (requested, cap)
            if requested > cap
            else ""
        )
        if failed:
            log.warning(
                "texture filtering: anisotropy %.3gx applied to %d texture(s), %d failed%s",
                self.aniso, applied, failed, clamp_note,
            )
        else:
            log.info(
                "texture filtering: anisotropy %.3gx applied to %d texture(s)%s",
                self.aniso, applied, clamp_note,
            )
        self.update()

    def set_specular(self, enabled):
        """Enable/disable specular highlights (skin clear-coat + hair sheen). Safe before GL init."""
        self._specular = bool(enabled)
        self.update()

    def set_navi_tail_pose(self, enabled):
        """Enable/disable the hardcoded Na'vi tail curl. The pose is baked into the vertices at build
        time, so this rebuilds the currently-loaded Na'vi meshes to apply it (textures are kept - the
        rebuild only touches geometry). No-op before GL / with no Na'vi loaded."""
        enabled = bool(enabled)
        if enabled == getattr(self, "navi_tail_pose", True):
            return
        self.navi_tail_pose = enabled
        log.info("navi tail pose: %s", "on" if enabled else "off")
        built = getattr(self, "_navi_built_for", None)
        if built and self.ctx is not None:
            self.makeCurrent()
            self._build_navi_meshes(dict(built))
            self.doneCurrent()
        self.update()

    def load_model(self, path):
        if self.ctx is None:
            self._pending_model = path
            return
        self.makeCurrent()
        self._build_meshes(path)
        self.doneCurrent()
        self.update()

    def set_texture(self, key, role, arr):
        arr = np.ascontiguousarray(arr)
        if self.ctx is None:
            self._pending_tex.append((key, role, arr))
            return
        self.makeCurrent()
        slot = self.tex.setdefault(key, {})
        if role in slot:
            self._release_tex(slot[role])
        slot[role] = self._tex_from_np(arr)
        self.doneCurrent()
        self.update()

    def set_palette(self, key, palette, params=None):
        self.palettes[key] = [tuple(c) for c in palette]
        if params:
            self.params[key].update(params)
        self.update()

    def set_mode(self, mode):
        """Switch the single viewer between 'banshee' and 'navi' rendering."""
        if mode not in ("banshee", "navi") or mode == self.mode:
            return
        self._save_frame()
        self.mode = mode
        if self.ctx is not None:
            self.makeCurrent()
            self._load_frame(mode)  # restore that mode's camera + rebuild grid
            self.doneCurrent()
        self.update()

    def set_navi_meshes(self, mapping):
        """mapping: {"head": path, "body": path, "hair": path} (any subset)."""
        paths = tuple(sorted((k, v) for k, v in (mapping or {}).items() if v))
        if self.ctx is None:
            self._navi_pending_meshes = paths
            return
        self.makeCurrent()
        self._clear_navi_textures()  # drop stale textures so a removed role can't keep binding
        self._build_navi_meshes(dict(paths))
        self._navi_built_for = paths
        self.doneCurrent()
        self.update()

    def _clear_navi_textures(self):
        """Release and drop all bound navi textures. Called before a fresh texture load so a role that
        no longer resolves (e.g. a hair cap that was removed) doesn't linger and keep being bound."""
        for roles in self.navi_tex.values():
            for tex in roles.values():
                try:
                    if tex is not None:
                        self._release_tex(tex)
                except Exception:
                    pass
        self.navi_tex = {b: {} for b in NAVI_BUCKETS}

    def clear_navi_texture(self, bucket, role):
        """Release and drop a single bucket/role texture (e.g. when the warpaint texture is removed)
        so it stops binding. Safe to call before GL init or when the slot is already empty."""
        slot = self.navi_tex.get(bucket)
        if not slot or role not in slot:
            return
        try:
            if self.ctx is not None and slot[role] is not None:
                self._release_tex(slot[role])
        except Exception:
            pass
        slot.pop(role, None)
        self.update()

    def set_navi_texture(self, bucket, role, arr):
        """Upload an RGBA uint8 texture into a navi bucket/role slot (head/body/eye/hair)."""
        arr = np.ascontiguousarray(arr)
        if bucket == "eye" and role == "iris":
            # mean luminance of the iris diffuse, used to centre the fibre-detail term so striations
            # read through the recolour instead of washing out to a flat gradient.
            rgb = arr[..., :3].astype(np.float32)
            self._iris_mean_lum = float(
                (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).mean()
                / 255.0
            )
        if bucket in ("head", "body") and role == "color":
            # mean skin tone of this mesh's diffuse atlas, used to tone-match the body to the head so
            # the neck doesn't show a tonal STEP where the two separate atlases meet. We sample only
            # the mid-tone skin (ignore near-black UV gutter / near-white specular packing) so the
            # mean reflects actual skin, not atlas padding.
            rgb = arr[..., :3].astype(np.float32) / 255.0
            lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
            skin = (lum > 0.08) & (lum < 0.95)
            mean = (
                rgb[skin].mean(axis=0)
                if skin.any()
                else rgb.reshape(-1, 3).mean(axis=0)
            )
            self._skin_mean[bucket] = mean.astype(np.float32)
        if bucket in ("head", "body") and role == "normal":
            # mean of the AO (normal alpha) for this mesh, so AO is applied RELATIVE to its own mean
            # (only local cavity contrast, no global brightness step that would mismatch at the neck).
            a = arr[..., 3].astype(np.float32) / 255.0
            self._skin_ao_mean[bucket] = float(a.mean())
        if self.ctx is None:
            self._navi_pending_tex.append((bucket, role, arr))
            return
        self.makeCurrent()
        self._navi_upload(bucket, role, arr)
        self.doneCurrent()
        self.update()

    def set_navi_colors(self, state):
        """Drive the navi preview from NaviControls.state() (hex sRGB -> 0..1 uniforms)."""
        state = state or {}

        def cols(cat):
            return (state.get(cat, {}) or {}).get("colors", []) or []

        def rgb(hexstr, default):
            try:
                return tuple(float(x) for x in rc.hex_to_rgb01(hexstr))
            except Exception:  # noqa: BLE001 - bad/short hex -> keep default
                return default

        d = self.navi
        sk = cols("skin")
        if len(sk) >= 1:
            d["skin"] = rgb(sk[0], d["skin"])
        if len(sk) >= 2:
            d["pattern"] = rgb(sk[1], d["pattern"])
        ey = cols("eye")
        # Per the AFoP customization format: myColor1/2 = right eye outer/inner,
        # myColor3/4 = left eye outer/inner. With only 2 swatches present (or a preset that omits
        # 3/4) the left eye mirrors the right.
        if len(ey) >= 1:
            d["outer_r"] = rgb(ey[0], d["outer_r"])
        if len(ey) >= 2:
            d["inner_r"] = rgb(ey[1], d["inner_r"])
        d["outer_l"] = rgb(ey[2], d["outer_r"]) if len(ey) >= 3 else d["outer_r"]
        d["inner_l"] = rgb(ey[3], d["inner_r"]) if len(ey) >= 4 else d["inner_r"]
        ha = cols("hair")
        if len(ha) >= 1:
            d["hair1"] = rgb(ha[0], d["hair1"])
        if len(ha) >= 2:
            d["hair2"] = rgb(ha[1], d["hair2"])
        if len(ha) >= 3:
            d["hair3"] = rgb(ha[2], d["hair3"])
        # Hair colour slot 4 = myHairCapColor (scalp patch). 0x000000 is the "unset" sentinel:
        # leave it off so the scalp keeps inheriting the hair root (the existing fallback). A real,
        # non-black slot 4 (e.g. the DLC 'Summer Brown' dye) is the explicit scalp tint.
        cap_rgb = rgb(ha[3], None) if len(ha) >= 4 else None
        has_cap = cap_rgb is not None and any(v > 1e-4 for v in cap_rgb)
        d["has_haircap_color"] = has_cap
        if has_cap:
            d["haircap"] = cap_rgb
        hair_state = state.get("hair", {}) or {}
        if "cap_strength" in hair_state:
            try:
                d["hair_cap_strength"] = float(hair_state["cap_strength"])
            except (TypeError, ValueError):
                pass
        wp = cols("warpaint")
        for i, key in enumerate(("paint1", "paint2", "paint3", "paint4")):
            if len(wp) > i:
                d[key] = rgb(wp[i], d[key])
        wp_state = state.get("warpaint", {}) or {}
        d["bio_enabled"] = bool(wp_state.get("bioluminescent", False))
        d["hide_warpaint"] = bool(wp_state.get("hide_warpaint", False))
        self.update()

    def _save_frame(self):
        self._frames[self.mode] = dict(
            center=np.array(self.center, np.float32),
            radius=float(self.radius),
            floor_y=float(self.floor_y),
            dist=float(self.dist),
            pan=np.array(self.pan, np.float32),
            az=float(self.az),
            el=float(self.el),
        )

    def _load_frame(self, mode):
        fr = self._frames.get(mode)
        if not fr:
            return
        self.center = np.array(fr["center"], np.float32)
        self.radius = fr["radius"]
        self.floor_y = fr["floor_y"]
        self.dist = fr["dist"]
        self.pan = np.array(fr["pan"], np.float32)
        self.az, self.el = fr["az"], fr["el"]
        self._build_grid()

    def _frame_model(self, mode, center, radius, floor_y, default_el):
        """Store framing for `mode`; apply to the active camera only if that mode is active
        (so loading one mode's model never clobbers the other mode's view)."""
        fr = self._frames.get(mode) or {}
        fr.update(
            center=np.array(center, np.float32),
            radius=float(radius),
            floor_y=float(floor_y),
            dist=float(radius) * 2.4,
            pan=np.zeros(3, np.float32),
        )
        fr.setdefault("az", 35.0)
        fr.setdefault("el", default_el)
        self._frames[mode] = fr
        if mode == self.mode:
            self._load_frame(mode)

    # ---------------- GL lifecycle ----------------
    def initializeGL(self):
        self.ctx = moderngl.create_context()
        info = self.ctx.info
        log.info(
            "GL context ready: %s | %s | GLSL %s",
            info.get("GL_VERSION", "?"),
            info.get("GL_RENDERER", "?"),
            info.get("GL_SHADING_LANGUAGE_VERSION", "?"),
        )
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.prog = self.ctx.program(
            vertex_shader=gs.VERTEX_SHADER, fragment_shader=gs.FRAGMENT_SHADER
        )
        self.synth = self._tex_from_np(gs.synthetic_pattern(512), mip=False)
        self.grid_prog = self.ctx.program(
            vertex_shader=gs.GRID_VERTEX_SHADER, fragment_shader=gs.GRID_FRAGMENT_SHADER
        )
        # SSAA resolve: a fullscreen triangle that samples the offscreen colour buffer with linear
        # filtering (a 2x source sampled at output-pixel centres is the exact 2x2 box downsample).
        try:
            self._resolve_prog = self.ctx.program(
                vertex_shader="#version 330\nin vec2 in_pos; in vec2 in_uv; out vec2 vUV;"
                "void main(){ vUV = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }",
                fragment_shader="#version 330\nuniform sampler2D uTex; in vec2 vUV; out vec4 f;"
                "void main(){ f = texture(uTex, vUV); }",
            )
            tri = np.array(
                [-1.0, -1.0, 0.0, 0.0, 3.0, -1.0, 2.0, 0.0, -1.0, 3.0, 0.0, 2.0],
                dtype="f4",
            )
            self._resolve_vbo = self.ctx.buffer(tri.tobytes())
            self._resolve_vao = self.ctx.vertex_array(
                self._resolve_prog, [(self._resolve_vbo, "2f 2f", "in_pos", "in_uv")]
            )
            self._ssaa_cap = int(self.ctx.info.get("GL_MAX_TEXTURE_SIZE", 8192))
        except Exception:  # noqa: BLE001 - degrade to direct rendering
            log.warning("SSAA resolve setup failed; using direct rendering", exc_info=True)
            self._resolve_prog = None
        # FXAA resolve program (shares the fullscreen-triangle VBO). Separate try so a FXAA build
        # failure can't take out plain SSAA resolve; FXAA just becomes unavailable + logged.
        try:
            if self._resolve_vbo is not None:
                self._fxaa_prog = self.ctx.program(
                    vertex_shader="#version 330\nin vec2 in_pos; in vec2 in_uv; out vec2 vUV;"
                    "void main(){ vUV = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }",
                    fragment_shader=_FXAA_FRAG,
                )
                self._fxaa_vao = self.ctx.vertex_array(
                    self._fxaa_prog, [(self._resolve_vbo, "2f 2f", "in_pos", "in_uv")]
                )
                log.info("FXAA resolve program ready")
            else:
                log.warning("FXAA unavailable: resolve VBO missing")
        except Exception:  # noqa: BLE001
            log.warning("FXAA resolve setup failed; FXAA unavailable", exc_info=True)
            self._fxaa_prog = None
            self._fxaa_vao = None
        if self._pending_model:
            self._build_meshes(self._pending_model)
            self._pending_model = None
        for key, role, arr in self._pending_tex:
            slot = self.tex.setdefault(key, {})
            if role in slot:
                self._release_tex(slot[role])
            slot[role] = self._tex_from_np(arr)
        self._pending_tex.clear()

        # ---- Na'vi program (same context, second mode) ----
        # Compile is wrapped so a GLSL error surfaces loudly instead of failing silently mid-
        # initializeGL(): an unhandled throw here leaves navi_prog undefined and every Na'vi mesh
        # silently undrawn, which is indistinguishable from "no textures at all". On failure we
        # print the driver's actual shader log (with line numbers) and leave navi_prog = None so
        # the draw path degrades gracefully and the cause is visible in the terminal.
        navi_src = navi_fragment_source()
        try:
            self.navi_prog = self.ctx.program(
                vertex_shader=NAVI_VERTEX, fragment_shader=navi_src
            )
        except Exception as exc:  # noqa: BLE001
            self.navi_prog = None
            import sys

            print("\n=== Na'vi shader FAILED to compile ===", file=sys.stderr)
            print(exc, file=sys.stderr)
            numbered = "\n".join(
                "%4d | %s" % (i + 1, ln) for i, ln in enumerate(navi_src.splitlines())
            )
            print(numbered, file=sys.stderr)
            print("=== end Na'vi shader dump ===\n", file=sys.stderr)
        if self.navi_prog is not None:
            for name, unit in (
                ("uColorTex", NU_COLOR),
                ("uMaterialTex", NU_MATERIAL),
                ("uPatternTex", NU_PATTERN),
                ("uPaintTex", NU_PAINT),
                ("uPaintTex2", NU_PAINT2),
                ("uPaintTex3", NU_PAINT3),
                ("uPaintTex4", NU_PAINT4),
                ("uBioTex", NU_BIO),
                ("uHairCapTex", NU_HAIRCAP),
                ("uIrisTex", NU_IRIS),
                ("uHairMask", NU_HAIRMASK),
                ("uHairAO", NU_HAIRAO),
                ("uEyeHeightTex", NU_EYE_HEIGHT),
                ("uEyeNormalTex", NU_EYE_NORMAL),
                ("uSkinNormalTex", NU_SKIN_NORMAL),
                ("uDetailNormalTex", NU_DETAIL_NORMAL),
                ("uCamoMask", NU_MATERIAL),  # camo mask shares the material unit (flat path frees it)
                ("uGearMatTex", NU_PATTERN),  # gear _m coverage shares the pattern unit (flat path frees it)
                ("uGearNormalTex", NU_SKIN_NORMAL),  # gear _n shares the skin-normal unit (flat path frees it)
                ("uGearRegionTex", NU_DETAIL_NORMAL),  # cloth ColorMask shares the detail-normal unit (free here)
            ):
                u = self.navi_prog.get(name, None)
                if u is not None:
                    u.value = unit
        self.navi_fallback = self._tex_from_np(
            np.full((1, 1, 4), 255, np.uint8), mip=False
        )
        if self._navi_pending_meshes is not None:
            self._build_navi_meshes(dict(self._navi_pending_meshes))
            self._navi_built_for = self._navi_pending_meshes
            self._navi_pending_meshes = None
        for bucket, role, arr in self._navi_pending_tex:
            self._navi_upload(bucket, role, arr)
        self._navi_pending_tex.clear()
        if self._gear_pending:
            for gkey, (mpath, garr) in list(self._gear_pending.items()):
                if mpath:
                    self._build_gear(gkey, mpath)
                if garr is not None:
                    self._set_gear_tex(gkey, garr)
            self._gear_pending.clear()
        if self._gear_mat_pending:
            for gkey, marr in list(self._gear_mat_pending.items()):
                if marr is not None:
                    self._set_gear_mat_tex(gkey, marr)
            self._gear_mat_pending.clear()
        if self._gear_normal_pending:
            for gkey, narr in list(self._gear_normal_pending.items()):
                if narr is not None:
                    self._set_gear_normal_tex(gkey, narr)
            self._gear_normal_pending.clear()
        if self._gear_region_pending:
            for gkey, rarr in list(self._gear_region_pending.items()):
                if rarr is not None:
                    self._set_gear_region_tex(gkey, rarr)
            self._gear_region_pending.clear()
        if self._camo_mask_pending is not None:
            self.set_camo_mask(self._camo_mask_pending)
            self._camo_mask_pending = None
        if self._camo_tiger_pending is not None:
            self.set_camo_tiger_mask(self._camo_tiger_pending)
            self._camo_tiger_pending = None
        # if we initialised while already in navi mode, frame it now
        if self.mode == "navi":
            self._load_frame("navi")

    def _release_tex(self, t):
        """Release a GL texture and drop it from the live-texture registry, so replaced/cleared
        textures don't accumulate as dead handles in ``_live_textures`` (a slow memory leak, plus
        wasted try/except work in set_anisotropy). Safe on any GL object or None."""
        if t is None:
            return
        try:
            self._live_textures.remove(t)
        except ValueError:
            pass  # not a registry-tracked texture (e.g. a plain ctx.texture); just release it
        try:
            t.release()
        except Exception:  # noqa: BLE001
            pass

    def _tex_from_np(self, arr, mip=True, clamp=False):
        h, w = arr.shape[:2]
        comp = arr.shape[2]
        t = self.ctx.texture((w, h), comp, arr.tobytes())
        # most navi maps tile (repeat); the hair-cap map is wrap=clamp in-game (UVset=1 is tiled,
        # so clamping keeps the mask on the scalp instead of tiling it across the face).
        t.repeat_x = t.repeat_y = not clamp
        if mip:
            t.build_mipmaps()
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            t.anisotropy = self.aniso  # reduce grazing-angle aliasing (membrane sparkle); 1 = off
        else:
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._live_textures.append(t)  # so set_anisotropy() can re-tune it live
        return t

    def _build_meshes(self, path):
        for vao, vbo, ibo, *_ in self.gl_objs:
            vao.release()
            vbo.release()
            ibo.release()
        self.gl_objs.clear()
        meshes, _ = ml.load_model(path)
        # drop mesh parts that shouldn't be in the preview (tweak BANSHEE_SKIP_MESH)
        meshes = [m for m in meshes if not (m.name or "").startswith(BANSHEE_SKIP_MESH)]
        center, radius = ml.model_bounds(meshes)
        floor_y = float(min(m.positions[:, 1].min() for m in meshes))
        self._frame_model("banshee", center, radius, floor_y, 12.0)
        for m in meshes:
            inter = np.concatenate([m.positions, m.normals, m.uv0], 1).astype("f4")
            vbo = self.ctx.buffer(inter.tobytes())
            ibo = self.ctx.buffer(m.indices.astype("i4").tobytes())
            vao = self.ctx.vertex_array(
                self.prog,
                [(vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv")],
                index_buffer=ibo,
            )
            key = RECOLOUR_MESHES.get(m.name)
            atlas = key or ATLAS_OF.get(m.name)  # which texture set to sample
            self.gl_objs.append((vao, vbo, ibo, m.name, key, atlas))

    def _build_grid(self):
        if self.grid_vao is not None:
            self.grid_vao.release()
            self.grid_vbo.release()
            self.grid_vao = self.grid_vbo = None
        data = gs.grid_lines(self.center, self.radius, self.floor_y)
        self.grid_vbo = self.ctx.buffer(data.tobytes())
        self.grid_vao = self.ctx.vertex_array(
            self.grid_prog, [(self.grid_vbo, "3f 3f", "in_pos", "in_col")]
        )

    def resizeGL(self, w, h):
        self._w, self._h = max(w, 1), max(h, 1)

    def paintGL(self):
        if self.ctx is None:
            return
        screen = self.ctx.detect_framebuffer()
        target = self._ssaa_begin(
            screen
        )  # offscreen 2x buffer, or the screen when SSAA is off
        target.use()
        self.ctx.clear(0.10, 0.11, 0.14, 1.0)
        self._render_scene()
        if target is not screen:
            self._ssaa_resolve(screen)  # linear downsample 2x -> widget

    def _render_scene(self):
        objs = self.navi_objs if self.mode == "navi" else self.gl_objs
        if not objs:
            return
        aspect = self._w / self._h
        M, _eye = gs.mvp(
            self.center, self.az, self.el, self.dist, aspect, self.radius, self.pan
        )
        if self.grid_vao is not None:
            self.grid_prog["uMVP"].write(np.ascontiguousarray(M.T))
            self.grid_vao.render(moderngl.LINES)
        if self.mode == "navi":
            self._draw_navi_scene(M, _eye)
            return
        self.prog["uMVP"].write(np.ascontiguousarray(M.T))
        self.prog["uNormalMat"].write(np.eye(3, dtype="f4").tobytes())
        self.prog["uLightDir"].value = (-0.4, -0.7, -0.55)
        self.prog[
            "uIblIntensity"
        ].value = 2.5  # same character-render scene exposure as the Na'vi
        self.prog["uKeyLight"].value = 0.0  # pure IBL by default (key light optional)
        self.prog["uFlipV"].value = 1.0 if self.flip_v else 0.0
        self.prog["uColor"].value = 0
        self.prog["uMaterial"].value = 1
        self.prog["uPatternCoat"].value = 2
        self.prog["uNormalTex"].value = 4
        self.prog["uDetail1"].value = 5
        self.prog["uDetail2"].value = 6
        self.prog["uDetail3"].value = 7
        self.prog["uDetailMask"].value = 8

        # opaque meshes first, then transparent ones (body membrane + wing) blended & two-sided
        if not self.body_hidden:
            transp = []
            for obj in self.gl_objs:
                if (
                    obj[5] == "wing" or obj[4] == "body"
                ):  # wing mesh, or body (membrane in _d alpha)
                    transp.append(obj)
                    continue
                self._draw_mesh(obj, transparent=False)
            if transp:
                self.ctx.enable(moderngl.BLEND)
                self.ctx.disable(moderngl.CULL_FACE)  # membrane visible from both sides
                for obj in transp:
                    self._draw_mesh(obj, transparent=True)
                self.ctx.enable(moderngl.CULL_FACE)
                self.ctx.disable(moderngl.BLEND)

        # gear pieces preview on top of the Ikran via the Na'vi flat path. The gear VAOs are bound
        # to navi_prog, so aim its MVP at this scene's camera before drawing (the meshes above used
        # self.prog). Opaque state here (blend off, cull on) is what the flat gear wants.
        if self.navi_prog is not None and self.gear_objs:
            self.navi_prog["uMVP"].write(np.ascontiguousarray(M.T))
            self._navi_setu("uFlipV", 1.0 if self.flip_v else 0.0)
            # the flat gear path lights with uLightDir and finishes with uTonemap; the navi scene
            # sets these but this (banshee) path didn't - so the gear rendered dark until a Na'vi
            # draw happened to set them on the shared navi_prog. Set them here to match.
            self._navi_setu("uLightDir", (-0.4, -0.7, -0.55))
            self._navi_setu("uTonemap", float(self.navi.get("tonemap", 0.0)))
            self._draw_gear_pieces()

    # ---------------- SSAA offscreen target ----------------
    def _ssaa_begin(self, screen):
        """Return the render target: an offscreen buffer (at ssaa_scale x the widget for SSAA, or 1x
        for FXAA-only) that gets resolved to the screen, or the screen itself when no post-process is
        active / available. Rebuilds the buffer when the size changes."""
        want_ssaa = self.ssaa and self.ssaa_scale > 1.0 and self._resolve_prog is not None
        want_fxaa = self.fxaa and self._fxaa_prog is not None
        if not (want_ssaa or want_fxaa):
            return screen
        sw, sh = screen.size
        scale = self.ssaa_scale if want_ssaa else 1.0  # FXAA-only renders at native res
        tw = max(1, min(int(round(sw * scale)), self._ssaa_cap))
        th = max(1, min(int(round(sh * scale)), self._ssaa_cap))
        if self._ssaa_fbo is None or self._ssaa_size != (tw, th):
            self._ssaa_rebuild(tw, th)
        # a failed rebuild leaves _ssaa_fbo None (and logs) -> render direct to the screen instead
        return self._ssaa_fbo if self._ssaa_fbo is not None else screen

    def _ssaa_rebuild(self, w, h):
        for o in (self._ssaa_fbo, self._ssaa_color, self._ssaa_depth):
            if o is not None:
                try:
                    o.release()
                except Exception:  # noqa: BLE001
                    pass
        try:
            self._ssaa_color = self.ctx.texture((w, h), 4)
            self._ssaa_color.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._ssaa_color.repeat_x = self._ssaa_color.repeat_y = False
            self._ssaa_depth = self.ctx.depth_renderbuffer((w, h))
            self._ssaa_fbo = self.ctx.framebuffer(
                color_attachments=[self._ssaa_color], depth_attachment=self._ssaa_depth
            )
            self._ssaa_size = (w, h)
            log.debug("SSAA offscreen buffer built at %dx%d", w, h)
        except Exception:  # noqa: BLE001 - degrade to direct rendering rather than crash paintGL
            log.warning(
                "SSAA offscreen buffer %dx%d failed; falling back to direct rendering",
                w, h, exc_info=True,
            )
            self._ssaa_fbo = self._ssaa_color = self._ssaa_depth = None
            self._ssaa_size = (0, 0)
            self.ssaa = False  # stop retrying this frame; a smaller size may succeed later

    def _ssaa_resolve(self, screen):
        screen.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self._ssaa_color.use(location=0)
        if self.fxaa and self._fxaa_prog is not None and self._fxaa_vao is not None:
            self._fxaa_prog["uTex"].value = 0
            w, h = self._ssaa_size
            self._fxaa_prog["uRcpFrame"].value = (1.0 / max(1, w), 1.0 / max(1, h))
            self._fxaa_vao.render(moderngl.TRIANGLES)
        elif self._resolve_prog is not None:
            self._resolve_prog["uTex"].value = 0
            self._resolve_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def _draw_mesh(self, obj, transparent):
        vao, vbo, ibo, name, key, atlas = obj
        ts = self.tex.get(key, {})
        bts = self.tex.get("body", {})
        col = ts.get("color") or bts.get("color")  # head falls back to body
        mat = ts.get("material") or bts.get("material")
        pat = ts.get("pattern") or bts.get("pattern") or self.synth
        self.prog["uUseTexAlpha"].value = 1 if transparent else 0
        if key is not None and col is not None and mat is not None:
            col.use(0)
            mat.use(1)
            pat.use(2)
            self.prog["uRecolor"].value = 1
            self.prog["uTextured"].value = 0
            self.prog["uDesat"].value = (0.0, 0.0)
            self.prog["uColors"].write(np.array(self.palettes[key], "f4").tobytes())
            p = self.params[key]
            self.prog["uInvert1"].value = float(p["invert1"])
            self.prog["uInvert2"].value = float(p["invert2"])
            self.prog["uLevel1"].value = float(p["level1"])
            self.prog["uLevel2"].value = float(p["level2"])
            # normal maps: base (_n) + three shared tiling detail normals masked by _dn_mask
            nrm = ts.get("normal") or bts.get("normal")
            if nrm is not None:
                sh = self.tex.get("shared", {})
                d1 = sh.get("detail1")
                d2 = sh.get("detail2")
                d3 = sh.get("detail3")
                dmk = ts.get("dn_mask") or bts.get("dn_mask")
                nrm.use(4)
                (d1 or nrm).use(5)
                (d2 or nrm).use(6)
                (d3 or nrm).use(7)
                (dmk or nrm).use(8)
                self.prog["uUseNormalMap"].value = 1
                self.prog["uNormalStrength"].value = self.normal_strength
                self.prog["uDetailTiling"].value = self.detail_tiling
                have_detail = d1 is not None and dmk is not None
                self.prog["uDetailWeight"].value = (
                    self.detail_weight if have_detail else 0.0
                )
                self.prog["uNormalYFlip"].value = self.normal_yflip
            else:
                self.prog["uUseNormalMap"].value = 0
        else:
            self.prog["uUseNormalMap"].value = 0
            # non-recolour mesh: sample its atlas albedo if we have it, else flat grey
            acol = self.tex.get(atlas, {}).get("color") if atlas else None
            if acol is not None:
                acol.use(0)
                self.prog["uRecolor"].value = 0
                self.prog["uTextured"].value = 1
            else:
                self.prog["uRecolor"].value = 0
                self.prog["uTextured"].value = 0
                self.prog["uFlat"].value = (0.45, 0.44, 0.42)
        vao.render()

    # ---------------- Na'vi rendering ----------------
    def _navi_upload(self, bucket, role, arr):
        slot = self.navi_tex.setdefault(bucket, {})
        if role in slot:
            self._release_tex(slot[role])
        # the head hair-cap samples the tiled uv1; the game clamps it (wrap=clamp) so the mask lands
        # on the scalp once instead of tiling across the face. Match that for the cap only.
        clamp = bucket == "head" and role == "haircap"
        slot[role] = self._tex_from_np(arr, clamp=clamp)

    def set_gear(self, key, mesh_path, diffuse_rgba=None):
        """Load a gear .mmb and (optionally) its already-recoloured diffuse, to render on top of the
        Na'vi. `key` identifies the slot so re-picking the same slot replaces it. diffuse_rgba is
        RGBA uint8 or None (None -> the mesh renders flat-grey until a diffuse is set)."""
        if self.ctx is None:
            self._gear_pending[key] = (mesh_path, diffuse_rgba)
            return
        self.makeCurrent()
        self._build_gear(key, mesh_path)
        if diffuse_rgba is not None:
            self._set_gear_tex(key, diffuse_rgba)
        self.doneCurrent()
        self.update()

    def set_gear_texture(self, key, diffuse_rgba):
        """Update just the recoloured diffuse for an already-loaded gear slot (on a colour change)."""
        if self.ctx is None:
            mp = self._gear_pending.get(key, (None, None))[0]
            self._gear_pending[key] = (mp, diffuse_rgba)
            return
        self.makeCurrent()
        self._set_gear_tex(key, diffuse_rgba)
        self.doneCurrent()
        self.update()

    def set_gear_material(self, key, material_rgba):
        """Set (or clear, with None) a gear slot's Material (_m) map. Only its ALPHA is used - as the
        camo coverage mask (Material.a in the game shader), so camo lands only on the camo'd zones."""
        if self.ctx is None:
            if material_rgba is None:
                self._gear_mat_pending.pop(key, None)
            else:
                self._gear_mat_pending[key] = material_rgba
            return
        self.makeCurrent()
        if material_rgba is None:
            old = self.gear_mat_tex.pop(key, None)
            if old is not None:
                try:
                    self._release_tex(old)
                except Exception:  # noqa: BLE001
                    pass
        else:
            self._set_gear_mat_tex(key, material_rgba)
        self.doneCurrent()
        self.update()

    def set_gear_normal(self, key, normal_rgba):
        """Set (or clear, with None) a gear slot's Normal (_n) map. Used only by the lit camo path:
        rgb perturbs the shading normal, alpha is baked AO - this is what gives camo'd gear its relief
        and depth instead of flat clay. No-op for non-camo gear (the flat path ignores uGearNormalTex)."""
        if self.ctx is None:
            if normal_rgba is None:
                self._gear_normal_pending.pop(key, None)
            else:
                self._gear_normal_pending[key] = normal_rgba
            return
        self.makeCurrent()
        if normal_rgba is None:
            old = self.gear_normal_tex.pop(key, None)
            if old is not None:
                try:
                    self._release_tex(old)
                except Exception:  # noqa: BLE001
                    pass
        else:
            self._set_gear_normal_tex(key, normal_rgba)
        self.doneCurrent()
        self.update()

    def set_gear_region(self, key, region_rgba):
        """Set (or clear, with None) a gear slot's cloth ColorMask (_reg_m) map. Used only by the cloth
        Overlay camo path: its RGBA channels are the four baked garment regions (camo enters the .y/green
        zone). No-op for the weapon path and non-camo gear (the shader ignores uGearRegionTex there)."""
        if self.ctx is None:
            if region_rgba is None:
                self._gear_region_pending.pop(key, None)
            else:
                self._gear_region_pending[key] = region_rgba
            return
        self.makeCurrent()
        if region_rgba is None:
            old = self.gear_region_tex.pop(key, None)
            if old is not None:
                try:
                    self._release_tex(old)
                except Exception:  # noqa: BLE001
                    pass
        else:
            self._set_gear_region_tex(key, region_rgba)
        self.doneCurrent()
        self.update()

    def set_gear_hidden(self, key, hidden):
        """Show/hide a single gear slot without destroying its mesh (the 'Hide Gear' tickbox)."""
        if hidden:
            self.gear_hidden.add(key)
        else:
            self.gear_hidden.discard(key)
        self.update()

    def set_gear_camo(self, key, colors):
        """Arm or clear the camo for ONE gear piece (key = the full 'camo:...' gear key).

        colors = (R, G, B), three LINEAR float3 colours (Primary/Secondary/Tertiary, as produced by
        recolor_core.camo_colors_from_palette), or None to clear that piece. Stored per key so each
        camo gear/weapon previews its own selected palette; a mask must also be set (set_camo_mask)
        before anything shows."""
        if colors is None:
            self._gear_camo.pop(key, None)
        else:
            try:
                self._gear_camo[key] = tuple(
                    (float(c[0]), float(c[1]), float(c[2])) for c in colors
                )
            except (TypeError, ValueError, IndexError):
                self._gear_camo.pop(key, None)
        self.update()

    def set_camo_mask(self, rgba):
        """Set (or clear, with None) the camo region mask: the decoded w_camo_solid.dds as an HxWx4
        uint8 RGBA array. The shader reads its RGB channels triplanar-projected in model space."""
        if rgba is None:
            if self.ctx is not None and self._camo_mask_tex is not None:
                self.makeCurrent()
                try:
                    self._release_tex(self._camo_mask_tex)
                except Exception:
                    pass
                self.doneCurrent()
            self._camo_mask_tex = None
            self._camo_mask_pending = None
            self.update()
            return
        if self.ctx is None:
            self._camo_mask_pending = np.ascontiguousarray(rgba)
            return
        self.makeCurrent()
        old = self._camo_mask_tex
        # repeat-wrapped + mipmapped so the mask tiles cleanly under the triplanar projection
        self._camo_mask_tex = self._tex_from_np(
            np.ascontiguousarray(rgba), mip=True, clamp=False
        )
        if old is not None:
            try:
                self._release_tex(old)
            except Exception:
                pass
        self.doneCurrent()
        self.update()

    def set_camo_tiger_mask(self, rgba):
        """Set (or clear, with None) the tiger-stripe region mask (decoded w_camo_tigerstripe.dds),
        the alternate to the solid mask. Pieces set to the 'tigerstripe' pattern sample this one."""
        if rgba is None:
            if self.ctx is not None and self._camo_tiger_tex is not None:
                self.makeCurrent()
                try:
                    self._release_tex(self._camo_tiger_tex)
                except Exception:
                    pass
                self.doneCurrent()
            self._camo_tiger_tex = None
            self._camo_tiger_pending = None
            self.update()
            return
        if self.ctx is None:
            self._camo_tiger_pending = np.ascontiguousarray(rgba)
            return
        self.makeCurrent()
        old = self._camo_tiger_tex
        self._camo_tiger_tex = self._tex_from_np(
            np.ascontiguousarray(rgba), mip=True, clamp=False
        )
        if old is not None:
            try:
                self._release_tex(old)
            except Exception:
                pass
        self.doneCurrent()
        self.update()

    def set_body_hidden(self, hidden):
        """'Hide Ikran' / 'Hide Na'vi' — skip drawing the creature/character body (gear still shows)."""
        self.body_hidden = bool(hidden)
        self.update()

    def set_all_gear_hidden(self, hidden):
        """'Hide all gear' — skip every gear piece in one toggle (the body still shows)."""
        self.all_gear_hidden = bool(hidden)
        self.update()

    def clear_gear(self, key=None):
        """Remove one gear slot (or all if key is None)."""
        if self.ctx is not None:
            self.makeCurrent()
        kept = []
        for obj in self.gear_objs:
            if key is None or obj[4] == key:
                for r in obj[:3]:
                    try:
                        r.release()
                    except Exception:
                        pass
            else:
                kept.append(obj)
        self.gear_objs = kept
        for k in list(self.gear_tex) if key is None else [key]:
            self._release_tex(self.gear_tex.pop(k, None))
            self._release_tex(self.gear_mat_tex.pop(k, None))
            self._release_tex(self.gear_normal_tex.pop(k, None))
            self._release_tex(self.gear_region_tex.pop(k, None))
            self._gear_pending.pop(k, None)
            self._gear_mat_pending.pop(k, None)
            self._gear_normal_pending.pop(k, None)
            self._gear_region_pending.pop(k, None)
            self.gear_hidden.discard(k)
        if self.ctx is not None:
            self.doneCurrent()
        self.update()

    def _set_gear_tex(self, key, arr):
        arr = np.ascontiguousarray(arr)
        old = self.gear_tex.get(key)
        if old is not None:
            try:
                self._release_tex(old)
            except Exception:
                pass
        self.gear_tex[key] = self._tex_from_np(arr)

    def _set_gear_mat_tex(self, key, arr):
        arr = np.ascontiguousarray(arr)
        old = self.gear_mat_tex.get(key)
        if old is not None:
            try:
                self._release_tex(old)
            except Exception:
                pass
        self.gear_mat_tex[key] = self._tex_from_np(arr)

    def _set_gear_normal_tex(self, key, arr):
        arr = np.ascontiguousarray(arr)
        old = self.gear_normal_tex.get(key)
        if old is not None:
            try:
                self._release_tex(old)
            except Exception:
                pass
        self.gear_normal_tex[key] = self._tex_from_np(arr)

    def _set_gear_region_tex(self, key, arr):
        arr = np.ascontiguousarray(arr)
        old = self.gear_region_tex.get(key)
        if old is not None:
            try:
                self._release_tex(old)
            except Exception:
                pass
        self.gear_region_tex[key] = self._tex_from_np(arr)

    def _build_gear(self, key, path):
        """Load + place a gear .mmb (replacing any meshes already loaded for this key). Placement
        reuses the Na'vi path with the gear's OWN skeleton, so a gear authored in bind pose lands on
        the bind-pose character. Alignment is the thing to eyeball on the first run."""
        if self.navi_prog is None:
            return
        kept = []
        for obj in self.gear_objs:
            if obj[4] == key:
                for r in obj[:3]:
                    try:
                        r.release()
                    except Exception:
                        pass
            else:
                kept.append(obj)
        self.gear_objs = kept
        try:
            meshes, extra = ml.load_model(path)
        except Exception as exc:
            print("gear load failed (%s): %s" % (path, exc), file=sys.stderr)
            return
        extra = extra if isinstance(extra, dict) else {}
        world = navi_world_bones(extra.get("skeleton") or [])
        nverts = 0
        for m in meshes:
            mx = (extra.get("meshes") or {}).get(m.name)
            M = navi_placement(mx, world)
            P, N = navi_apply_placement(m.positions, m.normals, M)
            uv0 = m.uv0
            uv1 = m.uv1 if getattr(m, "uv1", None) is not None else uv0
            T = (
                m.tangents
                if getattr(m, "tangents", None) is not None
                else np.zeros((len(P), 3), np.float32)
            )
            inter = np.concatenate([P, N, T, uv0, uv1], 1).astype("f4")
            vbo = self.ctx.buffer(inter.tobytes())
            ibo = self.ctx.buffer(m.indices.astype("i4").tobytes())
            vao = self.ctx.vertex_array(
                self.navi_prog,
                [
                    (
                        vbo,
                        "3f 3f 3f 2f 2f",
                        "in_pos",
                        "in_nrm",
                        "in_tan",
                        "in_uv",
                        "in_uv1",
                    )
                ],
                index_buffer=ibo,
            )
            self.gear_objs.append((vao, vbo, ibo, m.name, key))
            nverts += len(P)
        print(
            "gear '%s': %d submesh(es), %d verts placed" % (key, len(meshes), nverts),
            file=sys.stderr,
        )

    def _build_navi_meshes(self, mapping):
        if self.navi_prog is None:  # shader failed to compile - nothing to bind VAOs to
            return
        for vao, vbo, ibo, *_ in self.navi_objs:
            vao.release()
            vbo.release()
            ibo.release()
        self.navi_objs.clear()
        self.navi_armature = []
        self.navi_armature_world = []
        all_meshes = []  # (part, submesh, part_extra)
        part_extra = {}  # part -> extra dict from the loader (skeleton + binds)
        for part in ("head", "body", "hair", "kuru"):
            path = mapping.get(part)
            if not path:
                continue
            try:
                meshes, extra = ml.load_model(path)
            except Exception:  # noqa: BLE001 - skip a part that fails to load
                continue
            extra = extra if isinstance(extra, dict) else {}
            part_extra[part] = extra
            for m in meshes:
                all_meshes.append((part, m, extra))
        if not all_meshes:
            return

        # master armature = the body's skeleton (fall back to head, then whatever loaded first)
        for pref in ("body", "head", "hair"):
            sk = (part_extra.get(pref) or {}).get("skeleton")
            if sk:
                self.navi_armature = sk
                self.navi_armature_world = navi_world_bones(sk)
                break

        # place each submesh via its bind transform, then frame from the PLACED geometry
        placed = []  # (part, submesh, P, N, uv0, uv1)
        for part, m, extra in all_meshes:
            mx = (extra.get("meshes") or {}).get(m.name) or {}
                # static display pose (tail curl): skin the mesh's own verts before placement. A no-op for
                # parts/verts the pose doesn't touch and it never rewrites normals wholesale (shading
                # unaffected). Off when disabled; any body whose skeleton/weights don't fit is left un-posed.
            Pp, Np = m.positions, m.normals
            if getattr(self, "navi_tail_pose", True):
                try:
                    pskin = navi_pose_skin(extra.get("skeleton") or [])
                    Pp, Np = navi_apply_pose(
                        m.positions, m.normals, mx.get("weights"), mx.get("influences"), pskin
                    )
                except Exception:  # noqa: BLE001 - incompatible model: just render it un-posed
                    log.debug("navi tail pose skipped for %s", getattr(m, "name", "?"), exc_info=True)
                    Pp, Np = m.positions, m.normals
            M = navi_placement(mx, self.navi_armature_world)
            P, N = navi_apply_placement(Pp, Np, M)
            uv1 = m.uv1 if m.uv1 is not None else m.uv0
            placed.append((part, m, P, N, m.uv0, uv1))

        allP = np.concatenate([p[2] for p in placed], 0)
        center = ((allP.min(0) + allP.max(0)) * 0.5).astype(np.float32)
        radius = float(np.linalg.norm(allP.max(0) - allP.min(0)) * 0.5) or 1.0
        floor_y = float(allP[:, 1].min())

            # The loader reads AUTHORED smooth normals from the .mmb, so hair cards already carry proper
            # scalp-combed shading - combing is OFF by default (blending toward the head dome would flatten
            # real detail). Kept as a tunable (self.navi["hair_normal_comb"]) for the fallback where a mesh
            # lacks authored normals. Head centroid is the radial origin; fall back to bbox.
        kinds = [navi_classify(part, m.name) for (part, m, P, N, uv0, uv1) in placed]
        head_pts = [
            P
            for (part, m, P, N, uv0, uv1), (k, b) in zip(placed, kinds)
            if b == "head" and k == NAVI_KIND_SKIN
        ]
        head_center = (
            np.concatenate(head_pts, 0).mean(0).astype(np.float32)
            if head_pts
            else center
        )
        comb = float(
            self.navi.get("hair_normal_comb", 0.0)
        )  # 0 = use authored normals as-is

        for (part, m, P, N, uv0, uv1), (kind, bucket) in zip(placed, kinds):
            # the transparent cornea/wet shell renders opaque with the head skin map and just
            # occludes the actual eyeball; skip it for the preview (tweak NAVI_SKIP_SUBMESH).
            if any(k in (m.name or "").lower() for k in NAVI_SKIP_SUBMESH):
                continue
            if kind == NAVI_KIND_HAIR and comb > 0.0:
                N = navi_comb_normals(P, N, head_center, comb)
            # Authored strand tangent (hair sheen direction). Zeros where a mesh lacks one -> the
            # shader skips the sheen for those verts. Placement is identity (NAVI_APPLY_BIND off), so
            # the raw tangent stays consistent with the raw P/N.
            T = (
                m.tangents
                if getattr(m, "tangents", None) is not None
                else np.zeros((len(P), 3), np.float32)
            )
            inter = np.concatenate([P, N, T, uv0, uv1], 1).astype("f4")
            vbo = self.ctx.buffer(inter.tobytes())
            ibo = self.ctx.buffer(m.indices.astype("i4").tobytes())
            vao = self.ctx.vertex_array(
                self.navi_prog,
                [
                    (
                        vbo,
                        "3f 3f 3f 2f 2f",
                        "in_pos",
                        "in_nrm",
                        "in_tan",
                        "in_uv",
                        "in_uv1",
                    )
                ],
                index_buffer=ibo,
            )
            self.navi_objs.append((vao, vbo, ibo, m.name, kind, bucket))
        self._frame_model("navi", center, radius, floor_y, 8.0)

    def _navi_setu(self, name, value):
        if self.navi_prog is None:
            return
        u = self.navi_prog.get(name, None)
        if u is not None:
            u.value = value

    def _draw_navi_scene(self, M, eye=None):
        if self.navi_prog is None:  # shader failed to compile - error already printed
            return
        self.navi_prog["uMVP"].write(np.ascontiguousarray(M.T))
        if eye is not None:
            cp = self.navi_prog.get("uCamPos", None)
            if cp is not None:
                cp.value = (float(eye[0]), float(eye[1]), float(eye[2]))
        nm = self.navi_prog.get("uNormalMat", None)
        if nm is not None:
            nm.write(np.eye(3, dtype="f4").tobytes())
        self._navi_setu("uFlipV", 1.0 if self.flip_v else 0.0)
        self._navi_setu("uLightDir", (-0.4, -0.7, -0.55))
        self._write_navi_uniforms()
        # opaque (skin/eye/flat) first, then alpha (hair + eyelash cards), two-sided & blended
        transp = []
        for obj in self.navi_objs:
            if obj[4] == NAVI_KIND_HAIR or obj[5] == "lash":
                transp.append(obj)
            else:
                self._draw_navi_mesh(obj)
        if transp:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.disable(moderngl.CULL_FACE)
            # soft-alpha hair needs depth-WRITE off so overlapping cards blend (depth TEST stays on,
            # so the opaque head still occludes hair behind it). Hard-cutoff keeps the original write.
            soft = bool(self.navi.get("hair_soft_alpha", False))
            if soft:
                self.ctx.depth_mask = False
            for obj in transp:
                self._draw_navi_mesh(obj)
            self.ctx.depth_mask = True
            self.ctx.enable(moderngl.CULL_FACE)
            self.ctx.disable(moderngl.BLEND)
        # gear pieces drawn once on top of the character (independent of body-hide)
        self._draw_gear_pieces()

    def _write_navi_uniforms(self):
        d = self.navi
        # camo is armed per-piece only in _draw_gear_pieces; keep the shared program disarmed for the
        # body/head flat parts (teeth, nails, accessories) so they are never camo-tinted.
        self._navi_setu("uCamoEnable", 0)
        for name, key in (
            ("uSkinColor", "skin"),
            ("uPatternColor", "pattern"),
            ("uPaintColor1", "paint1"),
            ("uPaintColor2", "paint2"),
            ("uPaintColor3", "paint3"),
            ("uPaintColor4", "paint4"),
            ("uHair1", "hair1"),
            ("uHair2", "hair2"),
            ("uHair3", "hair3"),
            ("uBioColor", "bio_color"),
            ("uHairCapColor", "haircap"),
        ):
            self._navi_setu(name, tuple(d[key]))
        self._navi_setu("uHasHairCapColor", 1 if d.get("has_haircap_color") else 0)
        self._navi_setu("uHairCapStrength", float(d.get("hair_cap_strength", 1.0)))
            # Hair cap: the game applies lerp(skin, myHairCapColor, HairCapMap.r) on UV1. When the dye
            # sets slot 4 (non-black), uHasHairCapColor=1 and the shader uses uHairCapColor; the 0x000000
            # sentinel falls back to inheriting the hair root. uHairCapStrength is the 'Cap strength' dial.
            # Eye iris colours (uOuterIris/uInnerIris) are set per-mesh in _draw_navi_mesh so left/right
            # eyes can take independent colours (myColor1/2 vs myColor3/4).
        self._navi_setu("uBioEnabled", 1 if d["bio_enabled"] else 0)
        self._navi_setu("uBioBrightness", float(d["bio_brightness"]))
        self._navi_setu("uBioPulsation", float(d["bio_pulsation"]))
        self._navi_setu("uTime", float(self.navi_time))
        self._navi_setu("uSmoothness", float(d["smoothness"]))
        self._navi_setu("uRootDarkening", float(d["root_darkening"]))
        self._navi_setu("uHairCoverage", float(self.navi.get("hair_coverage", 1.15)))
        self._navi_setu(
            "uHairSoftAlpha", 1 if self.navi.get("hair_soft_alpha", False) else 0
        )
        self._navi_setu("uHairSpec", float(self.navi.get("hair_spec", 0.09)))
        self._navi_setu("uHairRough", float(self.navi.get("hair_rough", 0.40)))
        # Fidelity knobs (tune against in-game shots): tonemap 0=off (current), ramp toward 1 for the
        # game's filmic rolloff; lash_tint darkens the near-white eyelash cards so they recede.
        self._navi_setu("uTonemap", float(self.navi.get("tonemap", 0.0)))
        self._navi_setu(
            "uLashTint", tuple(self.navi.get("lash_tint", (0.22, 0.20, 0.18)))
        )
        self._navi_setu(
            "uSkinAOStrength", float(self.navi.get("skin_ao_strength", 0.6))
        )
        # Skin fidelity dials (tune live against the game): skin_color_weight 0=current hybrid ->
        # 1=pure game colour-base; pattern_strength scales the stripe markings (1=full, lower=subtler).
        self._navi_setu(
            "uSkinColorWeight", float(self.navi.get("skin_color_weight", 1.0))
        )
        self._navi_setu(
            "uPatternStrength", float(self.navi.get("pattern_strength", 1.0))
        )
        self._navi_setu(
            "uSkinDesaturate", float(self.navi.get("skin_desaturate", 0.85))
        )
        self._navi_setu("uSkinAmbient", float(self.navi.get("skin_ambient", 0.5)))
        self._navi_setu("uSkinLightWrap", float(self.navi.get("skin_light_wrap", 0.8)))
        self._navi_setu("uIblIntensity", float(self.navi.get("ibl_intensity", 1.0)))
        self._navi_setu("uKeyLight", float(self.navi.get("key_light", 0.0)))
        self._navi_setu("uIblSaturation", float(self.navi.get("ibl_saturation", 0.0)))
        self._navi_setu("uIblTint", tuple(self.navi.get("ibl_tint", (1.0, 1.0, 1.0))))
        self._navi_setu("uClearCoat", float(self.navi.get("clear_coat", 0.6)))
        self._navi_setu(
            "uClearCoatRough", float(self.navi.get("clear_coat_rough", 0.6))
        )
        self._navi_setu("uSpecular", 1.0 if getattr(self, "_specular", True) else 0.0)
        self._navi_setu("uSkinNormalStr", float(self.navi.get("skin_normal_str", 0.25)))
        self._navi_setu(
            "uDetailNormalStr", float(self.navi.get("detail_normal_str", 0.15))
        )
        self._navi_setu(
            "uDetailNormalTiling", float(self.navi.get("detail_normal_tiling", 1.0))
        )
        self._navi_setu("uEyeHeightBlur", float(self.navi.get("eye_height_blur", 3.0)))
        self._navi_setu("uIrisSpread", float(self.navi.get("iris_spread", 0.7)))
        self._navi_setu("uIrisBlend", float(self.navi.get("iris_blend", 1.2)))
        self._navi_setu("uIrisDetail", float(self.navi.get("iris_detail", 1.0)))
        self._navi_setu("uIrisMeanLum", float(getattr(self, "_iris_mean_lum", 0.38)))
        self._navi_setu("uIrisHeightLo", float(self.navi.get("iris_height_lo", 0.85)))
        self._navi_setu("uIrisHeightHi", float(self.navi.get("iris_height_hi", 0.99)))
        cx, cy = self.navi.get("iris_uv_center", (0.46, 0.46))
        self._navi_setu("uIrisUVCenter", (float(cx), float(cy)))
        self._navi_setu("uIrisRadius", float(self.navi.get("iris_radius", 0.52)))
        self._navi_setu("uIrisRimSoft", float(self.navi.get("iris_rim_soft", 0.05)))
        self._navi_setu("uIrisNormalStr", float(self.navi.get("iris_normal_str", 0.3)))
        self._navi_setu("uIrisOpacity", float(self.navi.get("iris_opacity", 0.9)))
        self._navi_setu(
            "uIrisOpacityFalloff", float(self.navi.get("iris_opacity_falloff", 0.5))
        )

    def _navi_bind(self, bucket, role, unit):
        t = self.navi_tex.get(bucket, {}).get(role)
        (t or self.navi_fallback).use(unit)
        return 1 if t is not None else 0

    def _draw_navi_mesh(self, obj):
        if self.body_hidden:
            return
        _vao, _vbo, _ibo, _name, kind, bucket = obj
        try:
            self._navi_setu("uKind", kind)
            # per-mesh tone match: nudge the BODY skin toward the HEAD's mean tone so the two
            # separate atlases don't show a tonal step at the neck. Head is the reference (identity);
            # body is scaled by head_mean/body_mean (clamped so it's a gentle correction, not a recolour).
            tc = (1.0, 1.0, 1.0)
            if kind == NAVI_KIND_SKIN and bucket == "body":
                hm = self._skin_mean.get("head")
                bm = self._skin_mean.get("body")
                if hm is not None and bm is not None:
                    ratio = [float(hm[i] / max(bm[i], 1e-3)) for i in range(3)]
                    blend = float(
                        self.navi.get("skin_tone_match", 0.85)
                    )  # 0 = off, 1 = full match
                    tc = tuple(
                        1.0 + (min(max(r, 0.6), 1.6) - 1.0) * blend for r in ratio
                    )
            self._navi_setu("uToneCorrect", tc)
            if kind == NAVI_KIND_SKIN:
                self._navi_setu(
                    "uSkinAOMean", float(self._skin_ao_mean.get(bucket, 0.5))
                )
            if kind == NAVI_KIND_EYE:
                # left eye takes myColor3/4, right (default) takes myColor1/2 - per the AFoP
                # customization format. Detect the left eye by submesh name.
                nm = (_name or "").lower()
                is_left = (
                    "left" in nm or nm.endswith("_l") or "_l_" in nm or "eyel" in nm
                )
                d = self.navi
                self._navi_setu(
                    "uOuterIris", tuple(d["outer_r" if is_left else "outer_l"])
                )
                self._navi_setu(
                    "uInnerIris", tuple(d["inner_r" if is_left else "inner_l"])
                )
            # eyelash cards (bucket 'lash') reuse the FLAT path but sample the dedicated eyelash
            # texture (its own alpha) instead of the opaque head skin map.
            if bucket == "lash":
                self._navi_setu("uHasColor", self._navi_bind("eye", "lash", NU_COLOR))
            else:
                self._navi_setu("uHasColor", self._navi_bind(bucket, "color", NU_COLOR))
            self._navi_setu(
                "uHasMaterial", self._navi_bind(bucket, "material", NU_MATERIAL)
            )
            self._navi_setu(
                "uHasPattern", self._navi_bind(bucket, "pattern", NU_PATTERN)
            )
            hide = bool(self.navi.get("hide_warpaint"))
            pb = self._navi_bind(
                bucket, "paint", NU_PAINT
            )  # always bind (fallback) so samplers
            pb2 = self._navi_bind(
                bucket, "paint2", NU_PAINT2
            )  # stay valid even when a slot is empty
            pb3 = self._navi_bind(bucket, "paint3", NU_PAINT3)
            pb4 = self._navi_bind(bucket, "paint4", NU_PAINT4)
            self._navi_setu("uHasPaint", 0 if hide else pb)
            self._navi_setu("uHasPaint2", 0 if hide else pb2)
            self._navi_setu("uHasPaint3", 0 if hide else pb3)
            self._navi_setu("uHasPaint4", 0 if hide else pb4)
            self._navi_setu("uHasBio", self._navi_bind(bucket, "bio", NU_BIO))
            self._navi_setu(
                "uHasHairCap", self._navi_bind(bucket, "haircap", NU_HAIRCAP)
            )
            self._navi_setu("uHasIris", self._navi_bind(bucket, "iris", NU_IRIS))
            self._navi_setu(
                "uHasEyeHeight", self._navi_bind(bucket, "height", NU_EYE_HEIGHT)
            )
            self._navi_setu(
                "uHasEyeNormal", self._navi_bind(bucket, "normal", NU_EYE_NORMAL)
            )
            # skin normal (alpha=AO) + tiled detail normal - only for the skin (head/body) meshes
            if kind == NAVI_KIND_SKIN:
                self._navi_setu(
                    "uHasSkinNormal", self._navi_bind(bucket, "normal", NU_SKIN_NORMAL)
                )
                self._navi_setu(
                    "uHasDetailNormal",
                    self._navi_bind(bucket, "detail", NU_DETAIL_NORMAL),
                )
            else:
                self._navi_setu("uHasSkinNormal", 0)
                self._navi_setu("uHasDetailNormal", 0)
            self._navi_setu(
                "uHasHairMask", self._navi_bind(bucket, "mask", NU_HAIRMASK)
            )
            self._navi_setu("uHasHairAO", self._navi_bind(bucket, "ao", NU_HAIRAO))
                # Accessory / band (FLAT) material (_m) + normal (_n): reuse the gear flat-path units
                # (uGearMatTex/uGearNormalTex, free here), bound AFTER the pattern/skin-normal binds so the
                # unit ends holding the accessory map. Cleared for every non-flat mesh so a prior draw can't
                # leak _m/_n onto the face. uCamoEnable=0 on the Na'vi path, so this feeds the plain-flat
                # material/normal application.
            if kind == NAVI_KIND_FLAT:
                self._navi_setu(
                    "uHasGearMat", self._navi_bind(bucket, "material", NU_PATTERN)
                )
                self._navi_setu(
                    "uHasGearNormal", self._navi_bind(bucket, "normal", NU_SKIN_NORMAL)
                )
                self._navi_setu("uGearNormalStr", 1.0)
            else:
                self._navi_setu("uHasGearMat", 0)
                self._navi_setu("uHasGearNormal", 0)
            _vao.render()
        except Exception as exc:  # noqa: BLE001
            # one mesh failing (e.g. a driver rejecting a high texture unit) must not blank the
            # whole model - report it once per session and carry on with the rest.
            if not getattr(self, "_navi_draw_err_shown", False):
                self._navi_draw_err_shown = True
                import sys

                print(
                    "Na'vi mesh draw failed on '%s' (kind=%s bucket=%s): %s"
                    % (_name, kind, bucket, exc),
                    file=sys.stderr,
                )

    def _draw_gear_pieces(self):
        """Render every visible gear slot via the Na'vi flat path - just its baked diffuse, no
        lighting/material slots. The gear VAOs are bound to navi_prog, so the caller must already
        have navi_prog's uMVP set (the Na'vi scene does it; the Ikran scene sets it before calling).
        Gear keys are prefixed by their tab ('navi:' / 'ikran:'), so each viewer only draws its own
        tab's gear - Ikran gear stays on the Ikran viewer, Na'vi gear on the Na'vi viewer."""
        if self.all_gear_hidden:
            return
        other = "ikran:" if self.mode == "navi" else "navi:"
        for gvao, _gvbo, _gibo, gname, gkey in self.gear_objs:
            if str(gkey).startswith(other):  # gear from the other tab - not this viewer
                continue
            if gkey in self.gear_hidden:
                continue
            try:
                self._navi_setu("uKind", NAVI_KIND_FLAT)
                tex = self.gear_tex.get(gkey)
                (tex or self.navi_fallback).use(NU_COLOR)
                self._navi_setu("uHasColor", 1 if tex is not None else 0)
                # flat path: silence every other slot so it samples only the gear diffuse
                for _h in (
                    "uHasMaterial",
                    "uHasPattern",
                    "uHasPaint",
                    "uHasPaint2",
                    "uHasPaint3",
                    "uHasPaint4",
                    "uHasBio",
                    "uHasHairCap",
                    "uHasIris",
                    "uHasEyeHeight",
                    "uHasEyeNormal",
                    "uHasSkinNormal",
                    "uHasDetailNormal",
                    "uHasHairMask",
                    "uHasHairAO",
                ):
                    self._navi_setu(_h, 0)
                # Camo preview: only 'camo:' gear, and only when both colours + mask are armed. Set
                # per-piece and reset after, so it never leaks onto the next piece or the body.
                cam = (
                    self._gear_camo.get(gkey)
                    if str(gkey).startswith("camo:")
                    else None
                )
                # per-piece region mask: tiger stripe if this piece is set to it AND the tiger mask
                # is loaded, otherwise the solid mask. Falls back to solid so a missing tiger never
                # blanks the preview.
                pattern = self._gear_camo_pattern.get(gkey, "solid")
                mask_tex = self._camo_mask_tex
                if pattern == "tigerstripe" and self._camo_tiger_tex is not None:
                    mask_tex = self._camo_tiger_tex
                camo_on = cam is not None and mask_tex is not None
                if camo_on:
                    mask_tex.use(NU_MATERIAL)  # uCamoMask shares the material unit
                    r, g, b = cam
                    self._navi_setu("uCamoColR", r)
                    self._navi_setu("uCamoColG", g)
                    self._navi_setu("uCamoColB", b)
                    self._navi_setu("uCamoTiling", self._camo_tiling)
                    self._navi_setu("uCamoRotation", self._camo_rotation)
                    self._navi_setu("uHasCamoMask", 1)
                    self._navi_setu("uCamoEnable", 1)
                    # gear _m alpha is the camo coverage (Material.a): camo only lands where it allows,
                    # so the diffuse shows on the un-camo'd zones (wood/cord) instead of one flat colour.
                    mat = self.gear_mat_tex.get(gkey)
                    if mat is not None:
                        mat.use(NU_PATTERN)  # uGearMatTex shares the pattern unit
                    self._navi_setu("uHasGearMat", 1 if mat is not None else 0)
                    self._navi_setu("uCamoAlphaEnable", 1 if self._camo_alpha_enable else 0)
                    # tigerstripe piece -> cloth shader (Overlay onto the diffuse, detail preserved);
                    # solid piece -> weapon shader (flat replace + clamp). Matches the two game shaders.
                    self._navi_setu(
                        "uCamoBlendMode", 1 if pattern == "tigerstripe" else 0
                    )
                    # gear _n drives the lit camo path: normal-mapped relief + baked AO, so camo reads
                    # as a lit surface instead of flat clay (matches the game's deferred normal/AO).
                    nrm = self.gear_normal_tex.get(gkey)
                    if nrm is not None:
                        nrm.use(NU_SKIN_NORMAL)  # uGearNormalTex shares the skin-normal unit
                    self._navi_setu("uHasGearNormal", 1 if nrm is not None else 0)
                    self._navi_setu("uGearNormalStr", self._gear_normal_str)
                    # cloth ColorMask (_reg_m): 4 baked regions; camo enters its green zone. Only used
                    # by the cloth/Overlay path; the 4th overlay (A) defaults to Overlay identity.
                    reg = self.gear_region_tex.get(gkey)
                    if reg is not None:
                        reg.use(NU_DETAIL_NORMAL)  # uGearRegionTex shares the detail-normal unit
                    self._navi_setu("uHasGearRegion", 1 if reg is not None else 0)
                    self._navi_setu("uCamoColA", self._camo_col_a)
                else:
                    self._navi_setu("uCamoEnable", 0)
                gvao.render()
                if camo_on:
                    self._navi_setu("uCamoEnable", 0)  # leave the shared program disarmed
                    self._navi_setu("uHasGearMat", 0)
                    self._navi_setu("uHasGearNormal", 0)
                    self._navi_setu("uHasGearRegion", 0)
            except Exception as gexc:  # noqa: BLE001
                if not getattr(self, "_gear_draw_err_shown", False):
                    self._gear_draw_err_shown = True
                    print(
                        "gear draw failed on '%s' (%s): %s" % (gname, gkey, gexc),
                        file=sys.stderr,
                    )

    # ---------------- interaction ----------------
    def mousePressEvent(self, e):
        self._last = e.position().toPoint()

    def mouseMoveEvent(self, e):
        p = e.position().toPoint()
        dx, dy = p.x() - self._last.x(), p.y() - self._last.y()
        self._last = p
        btns = e.buttons()
        shift = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        # Pan: middle mouse, or Shift + left mouse. Plain left mouse orbits.
        if (btns & Qt.MouseButton.MiddleButton) or (
            (btns & Qt.MouseButton.LeftButton) and shift
        ):
            s, u = gs.camera_basis(self.center + self.pan, self.az, self.el, self.dist)
            scale = self.dist / max(self._h, 1)
            self.pan = self.pan + (-dx * s + dy * u) * scale
            self.update()
        elif btns & Qt.MouseButton.LeftButton:
            self.az -= dx * 0.4
            self.el = max(-89.0, min(89.0, self.el + dy * 0.4))
            self.update()

    def wheelEvent(self, e):
        f = 0.9 if e.angleDelta().y() > 0 else 1.1
        self.dist = max(self.radius * 0.2, min(self.radius * 8, self.dist * f))
        self.update()
