"""GLSL for the recolour viewport, camera math, and a synthetic Pattern Coat generator.

The fragment shader transcribes recolor_core.recolor(): PatternCoat.B/A drive the two 5-stop
colour gradients, R/G the placement masks, overlaid onto the detail albedo. Simple
lambert+ambient preview lighting (not the game's PBR).
"""

from __future__ import annotations
import numpy as np

VERTEX_SHADER = """
#version 330
uniform mat4 uMVP;
uniform mat3 uNormalMat;
uniform float uFlipV;
in vec3 in_pos;
in vec3 in_nrm;
in vec2 in_uv;
out vec3 vN;
out vec3 vP;            // model-space position (for screen-space-derivative tangent frame)
out vec2 vUV;
void main() {
    vN = normalize(uNormalMat * in_nrm);
    vP = in_pos;
    vUV = vec2(in_uv.x, mix(in_uv.y, 1.0 - in_uv.y, uFlipV));
    gl_Position = uMVP * vec4(in_pos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D uColor;       // _d  (detail albedo)
uniform sampler2D uMaterial;    // _m  (PBR; .b drives overlay show-through)
uniform sampler2D uPatternCoat; // 4ch zone texture (R,G pattern masks; B,A coat selectors)
uniform vec3  uColors[10];      // myColor1..10 (rgb, 0..1)
uniform float uInvert1;
uniform float uInvert2;
uniform float uLevel1;
uniform float uLevel2;
uniform vec2  uDesat;
uniform int   uRecolor;         // 1 = recolour, 0 = flat/textured
uniform int   uTextured;        // 1 = sample uColor as plain albedo (non-recolour meshes)
uniform int   uUseTexAlpha;     // 1 = output sampled texture alpha (transparent meshes, e.g. wings)
uniform vec3  uFlat;            // flat colour when uRecolor==0 && uTextured==0
uniform vec3  uLightDir;        // world-space, normalised, points toward surface
uniform float uIblIntensity;    // master scale on the SH irradiance ambient (same scene as Na'vi)
uniform float uKeyLight;        // optional directional fill on top of the IBL (0 = pure IBL)

uniform sampler2D uNormalTex;        // _n       base normal (X=R, Y=A; Z reconstructed)
uniform sampler2D uDetail1;          // skin_detail_1_nr (X=R, Y=G), tiled
uniform sampler2D uDetail2;          // skin_detail_2_nr
uniform sampler2D uDetail3;          // skin_detail_4_nr
uniform sampler2D uDetailMask;       // _dn_mask  (rgb = per-detail blend weights)
uniform int   uUseNormalMap;    // 1 = perturb the geometric normal with the normal maps
uniform float uNormalStrength;  // base-normal XY scale
uniform vec3  uDetailTiling;    // UV tiling per detail normal
uniform float uDetailWeight;    // overall detail-normal strength
uniform float uNormalYFlip;     // +1 keep green, -1 flip (DirectX-style normal maps)

in vec3 vN;
in vec3 vP;
in vec2 vUV;
out vec4 frag;

// reconstruct Z from a tangent XY (snowdrop "unpack normal xy")
vec3 unpackXY(vec2 xy) { return vec3(xy, sqrt(clamp(1.0 - dot(xy, xy), 0.0, 1.0))); }
// reoriented normal mapping (snowdrop "combine normal maps.h")
vec3 rnm(vec3 a, vec3 b) { a += vec3(0.0, 0.0, 1.0); b *= vec3(-1.0, -1.0, 1.0); return a * dot(a, b) / a.z - b; }

const vec3 GREY = vec3(0.698, 0.686, 0.663);

float smoothstep_(float e0, float e1, float x){
    float t = clamp((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0);
    return t*t*(3.0 - 2.0*t);
}
vec3 overlay(vec3 base, vec3 blend){
    return mix(2.0*base*blend, 1.0 - 2.0*(1.0-base)*(1.0-blend), step(0.5, base));
}

// --- LINEAR-light helpers (mirrors the Na'vi pipeline) ----------------------
// The banshee skin shader (px_wildlife_skin_banshee) outputs a LINEAR albedo to the deferred
// buffer and is lit by the same character-render scene as the Na'vi. So the preview now runs the
// recolour + lighting in linear and encodes once at the end, instead of the old flat sRGB-space
// hemi (0.30 + 0.70*ndl with no encode).
vec3 srgb2lin(vec3 c){
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}
vec3 lin2srgb(vec3 c){
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
// Order-2 SH irradiance of the character-render scene (cubemap_park2_d_*), identical to the Na'vi
// IBL: top-down dominant, cool sky tint, dark ground.
const vec3 IBL_SH[9] = vec3[9](
    vec3( 0.110727,  0.145911,  0.157346),
    vec3( 0.083777,  0.122881,  0.172150),
    vec3( 0.002414,  0.010347,  0.009372),
    vec3( 0.003921, -0.002763, -0.021293),
    vec3(-0.002234, -0.010764, -0.025318),
    vec3( 0.005120,  0.010209,  0.009343),
    vec3( 0.006033,  0.002683, -0.007406),
    vec3( 0.021286,  0.030104,  0.038401),
    vec3( 0.015050,  0.011310, -0.006772)
);
vec3 shAmbient(vec3 n){
    float x = n.x, y = n.y, z = n.z;
    vec3 e = IBL_SH[0]
           + IBL_SH[1]*y + IBL_SH[2]*z + IBL_SH[3]*x
           + IBL_SH[4]*(x*y) + IBL_SH[5]*(y*z) + IBL_SH[6]*(3.0*z*z - 1.0)
           + IBL_SH[7]*(x*z) + IBL_SH[8]*(x*x - y*y);
    return max(e, 0.0);
}

vec3 recolour(){
    vec4 pc = texture(uPatternCoat, vUV);
    // Palette swatches are sRGB; linearise them (the shader's ColorBox colours are linear) so the
    // gradient blend, overlay and lighting all run in linear, then encode once at output.
    // Coat 1 gradient (B)
    float t = pc.b * 4.0;
    vec3 c1 = mix(srgb2lin(uColors[0]), srgb2lin(uColors[1]), clamp(t-0.0,0.0,1.0));
    c1 = mix(c1, srgb2lin(uColors[2]), clamp(t-1.0,0.0,1.0));
    c1 = mix(c1, srgb2lin(uColors[3]), clamp(t-2.0,0.0,1.0));
    c1 = mix(c1, srgb2lin(uColors[4]), clamp(t-3.0,0.0,1.0));
    // Coat 2 gradient (A)
    float t2 = pc.a * 4.0;
    vec3 c2 = mix(srgb2lin(uColors[5]), srgb2lin(uColors[6]), clamp(t2-0.0,0.0,1.0));
    c2 = mix(c2, srgb2lin(uColors[7]), clamp(t2-1.0,0.0,1.0));
    c2 = mix(c2, srgb2lin(uColors[8]), clamp(t2-2.0,0.0,1.0));
    c2 = mix(c2, srgb2lin(uColors[9]), clamp(t2-3.0,0.0,1.0));
    // placement masks
    float hi1 = uLevel1 * 0.25;
    float m1 = smoothstep_(hi1-0.25, hi1, pc.r) * uInvert1 - min(0.0, uInvert1);
    float hi2 = uLevel2 * 0.25;
    float m2 = smoothstep_(hi2-0.25, hi2, pc.g) * uInvert2 - min(0.0, uInvert2);
    float mask = clamp(m1 + m2, 0.0, 1.0);
    vec3 coat = sqrt(max(mix(c1, c2, mask), 0.0));
    // detail albedo path (base diffuse linearised; GREY is the shader's linear desaturate target)
    vec3 alb = srgb2lin(texture(uColor, vUV).rgb);
    float ds = clamp(uDesat.y + 1.0, 0.0, 1.0);
    alb = sqrt(max(mix(GREY, alb, ds), 0.0));
    // overlay coat onto albedo, masked by Material.b
    float om = clamp(texture(uMaterial, vUV).b + uDesat.x, 0.0, 1.0);
    vec3 outc = mix(alb, overlay(alb, coat), om);
    vec3 result = clamp(outc*outc, 0.0, 1.0);
    return clamp(result, 0.0, 1.0);
}

void main(){
    vec3 albedo;
    float alpha = 1.0;
    if (uRecolor == 1) {
        albedo = recolour();
        if (uUseTexAlpha == 1) {              // body membrane transparency from _d alpha
            alpha = texture(uColor, vUV).a;
            if (alpha < 0.02) discard;        // fully-transparent membrane: cut out cleanly
        }
    }
    else if (uTextured == 1) {
        vec4 c = texture(uColor, vUV);
        albedo = srgb2lin(c.rgb);
        if (uUseTexAlpha == 1) {              // wing membrane transparency from texture alpha
            alpha = c.a;
            if (alpha < 0.02) discard;
        }
    }
    else                     albedo = srgb2lin(uFlat);
    vec3 N = normalize(vN);
    if (uUseNormalMap == 1) {
        // base normal (_n): tangent XY in R and A
        vec2 bxy = (texture(uNormalTex, vUV).ra * 2.0 - 1.0) * uNormalStrength;
        bxy.y *= uNormalYFlip;
        vec3 nTS = unpackXY(bxy);
        // three tiling detail normals (skin_detail_*_nr, XY in R/G), weighted by _dn_mask
        vec3 dm = texture(uDetailMask, vUV).rgb * uDetailWeight;
        vec2 d1 = texture(uDetail1, vUV * uDetailTiling.x).rg * 2.0 - 1.0; d1.y *= uNormalYFlip;
        vec2 d2 = texture(uDetail2, vUV * uDetailTiling.y).rg * 2.0 - 1.0; d2.y *= uNormalYFlip;
        vec2 d3 = texture(uDetail3, vUV * uDetailTiling.z).rg * 2.0 - 1.0; d3.y *= uNormalYFlip;
        nTS = rnm(nTS, unpackXY(d1 * dm.r));   // RNM, scaled by mask channel (0 => identity)
        nTS = rnm(nTS, unpackXY(d2 * dm.g));
        nTS = rnm(nTS, unpackXY(d3 * dm.b));
        // tangent frame from screen-space derivatives (snowdrop "tangent from uv.h")
        vec3 dpx = dFdx(vP), dpy = dFdy(vP);
        vec2 dux = dFdx(vUV), duy = dFdy(vUV);
        vec3 r1 = cross(N, dpy), r2 = cross(dpx, N);
        vec3 T = normalize(r1 * dux.x + r2 * duy.x);
        vec3 B = normalize(r1 * dux.y + r2 * duy.y);
        N = normalize((-T) * nTS.x + B * nTS.y + N * nTS.z);   // ColumnMatrix(-t, b, n)
    }
    float ndl = max(dot(N, -normalize(uLightDir)), 0.0);
    vec3 amb = shAmbient(N) * uIblIntensity;          // game-scene IBL irradiance (same as Na'vi)
    vec3 hdr = albedo * (amb + ndl * uKeyLight);      // linear HDR (key light optional)
    vec3 tm  = (hdr * (2.51*hdr + 0.03)) / (hdr * (2.43*hdr + 0.59) + 0.14);   // ACES, linear
    frag = vec4(lin2srgb(clamp(tm, 0.0, 1.0)), alpha);   // encode to display
}
"""


# ---------------- camera / matrix math (numpy, column-major for moderngl) ---


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    eye = np.asarray(eye, np.float32)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def orbit_eye(center, azimuth_deg, elevation_deg, distance):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    d = np.array(
        [np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], np.float32
    )
    return np.asarray(center, np.float32) + d * distance


def camera_basis(center, az, el, dist):
    """Return (right, up) world vectors of the orbit camera, for panning."""
    eye = orbit_eye(center, az, el, dist)
    f = np.asarray(center, np.float32) - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, np.array([0, 1, 0], np.float32))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return s, u


def mvp(center, az, el, dist, aspect, radius, pan=None):
    c = np.asarray(center, np.float32)
    if pan is not None:
        c = c + np.asarray(pan, np.float32)
    eye = orbit_eye(c, az, el, dist)
    view = look_at(eye, c, np.array([0, 1, 0], np.float32))
    near = max(dist - radius * 2.0, radius * 0.02)
    far = dist + radius * 2.0
    proj = perspective(45.0, aspect, near, far)
    return (proj @ view).astype(np.float32), eye


# ---------------- synthetic pattern coat (placeholder) ----------------------


def synthetic_pattern(size=512):
    """Stand-in Pattern Coat: B ramps across U, A across V, R/G soft masks. Replace with the real texture later."""
    u = np.linspace(0, 1, size, dtype=np.float32)[None, :].repeat(size, 0)
    v = np.linspace(0, 1, size, dtype=np.float32)[:, None].repeat(size, 1)
    R = np.clip((u - 0.5) * 3.0 + 0.5, 0, 1)  # coat1 placement
    G = (np.sin(v * np.pi * 3) * 0.5 + 0.5).astype(np.float32)  # coat2 placement
    B = u  # coat1 selector
    A = v  # coat2 selector
    return (np.stack([R, G, B, A], -1) * 255).astype(np.uint8)


# ---------------- ground grid (Blender-style) -------------------------------

GRID_VERTEX_SHADER = """
#version 330
uniform mat4 uMVP;
in vec3 in_pos;
in vec3 in_col;
out vec3 vcol;
void main(){ vcol = in_col; gl_Position = uMVP * vec4(in_pos, 1.0); }
"""

GRID_FRAGMENT_SHADER = """
#version 330
in vec3 vcol;
out vec4 frag;
void main(){ frag = vec4(vcol, 1.0); }
"""


def grid_lines(center, radius, floor_y, divisions=40):
    """Interleaved [x,y,z,r,g,b] float32 line list for a floor grid on the XZ plane at floor_y, with brighter every-5th lines and coloured centre axes (X red, Z blue)."""
    span = radius * 3.2
    step = (2.0 * span) / divisions
    cx, cz = float(center[0]), float(center[2])
    y = float(floor_y)
    minor = (0.26, 0.27, 0.30)
    major = (0.38, 0.40, 0.44)
    x_axis = (0.62, 0.24, 0.26)
    z_axis = (0.24, 0.40, 0.62)
    verts = []

    def seg(p0, p1, c):
        verts.extend(
            (
                p0[0],
                p0[1],
                p0[2],
                c[0],
                c[1],
                c[2],
                p1[0],
                p1[1],
                p1[2],
                c[0],
                c[1],
                c[2],
            )
        )

    for i in range(divisions + 1):
        t = -span + i * step
        on_axis = abs(t) < step * 0.5
        c = major if (i % 5 == 0) else minor
        # lines parallel to Z (vary x); the centre one IS the Z axis -> blue
        seg((cx + t, y, cz - span), (cx + t, y, cz + span), z_axis if on_axis else c)
        # lines parallel to X (vary z); the centre one IS the X axis -> red
        seg((cx - span, y, cz + t), (cx + span, y, cz + t), x_axis if on_axis else c)

    return np.asarray(verts, np.float32)


# =====================================================================================
# Na'vi player recolour GLSL - merged in from the former navi_recolor.py. Spliced after the
# #version line of the Na'vi fragment shader in viewer.py. Mirrors recolor_core's Na'vi math.
# =====================================================================================
# ----------------------------------------------------------------------------- GLSL mirror
# The same functions for the moderngl viewport. Keep in lock-step with the CPU reference
# in recolor_core.py (recolor_skin / multi_lerp / hair_gradient / overlay).
NAVI_GLSL = r"""
const vec3 BIO_GREEN = vec3(0.467784, 0.930111, 0.508881);

// sRGB <-> linear (IEC 61966-2-1). The skin chain runs in LINEAR light to match the game's
// shading order; albedo + colour params are linearised on the way in, the result is encoded back
// to sRGB at output. Doing overlay + lighting + tonemap in sRGB-encoded values (the old path)
// both over-brightened the skin and pulled saturated blues toward magenta.
vec3 srgb2lin(vec3 c){
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}
vec3 lin2srgb(vec3 c){
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

// Diffuse irradiance of the game's character-render-scene environment, projected to order-2 SH
// (folded coeffs: ambient(n) = sum Ck * monomial_k(n), in LINEAR light, already /pi so it reads as
// a white-albedo reflectance). Derived from blue/baked/.../cubemaps/cubemap_park2_d_*.dds. Profile:
// top-down dominant, cool sky tint, dark ground - i.e. the actual hemispheric sky the game lights
// the customization character with, not a flat ambient floor.
const vec3 IBL_SH[9] = vec3[9](
    vec3( 0.110727,  0.145911,  0.157346),   // 1
    vec3( 0.083777,  0.122881,  0.172150),   // y
    vec3( 0.002414,  0.010347,  0.009372),   // z
    vec3( 0.003921, -0.002763, -0.021293),   // x
    vec3(-0.002234, -0.010764, -0.025318),   // xy
    vec3( 0.005120,  0.010209,  0.009343),   // yz
    vec3( 0.006033,  0.002683, -0.007406),   // 3z^2-1
    vec3( 0.021286,  0.030104,  0.038401),   // xz
    vec3( 0.015050,  0.011310, -0.006772)    // x^2-y^2
);
vec3 shAmbient(vec3 n){
    float x = n.x, y = n.y, z = n.z;
    vec3 e = IBL_SH[0]
           + IBL_SH[1]*y + IBL_SH[2]*z + IBL_SH[3]*x
           + IBL_SH[4]*(x*y) + IBL_SH[5]*(y*z) + IBL_SH[6]*(3.0*z*z - 1.0)
           + IBL_SH[7]*(x*z) + IBL_SH[8]*(x*x - y*y);
    return max(e, 0.0);
}

vec3 overlayBlend(vec3 b, vec3 t){
    return mix(2.0*b*t, 1.0 - 2.0*(1.0-b)*(1.0-t), step(0.5, b));
}
vec3 overlayTint(vec3 base, vec3 tint, float mask){     // 0.5 grey tint = identity
    return mix(base, overlayBlend(base, tint), mask);
}
vec3 paintSelect(vec3 c1, vec3 c2, vec3 c3, vec3 c4, vec3 sel){
    float t = sel.r * 3.0;                       // single-channel (PaintTexture.R) 4-stop selector
    vec3 p = c1;
    p = mix(p, c2, clamp(t,       0.0, 1.0));
    p = mix(p, c3, clamp(t - 1.0, 0.0, 1.0));
    p = mix(p, c4, clamp(t - 2.0, 0.0, 1.0));
    return p;
}
vec3 hairGradient(vec3 c1, vec3 c2, vec3 c3, float t, float smoothness){
    float lo1 = mix(0.33, 0.0, smoothness), hi1 = mix(0.34, 0.5, smoothness);
    float lo2 = mix(0.665,0.5, smoothness), hi2 = mix(0.666,1.0, smoothness);
    float r1 = clamp((t - lo1) / max(hi1 - lo1, 1e-6), 0.0, 1.0);   // linear ramps (per shader)
    float r2 = clamp((t - lo2) / max(hi2 - lo2, 1e-6), 0.0, 1.0);
    return c1 + (c2 - c1) * r1 + (c3 - c2) * r2;                    // additive 3-weight blend
}

// uniforms the viewer should supply (defaults = identity / no-op):
//   uSkinColor (0.5 grey), uPatternColor, uPaintColor1..4, uBioColor (BIO_GREEN),
//   uOuterIris,uInnerIris, uHair1..3, uBioBrightness, uBioPulsation, uTime,
//   uHairCapColor, uSmoothness, uRootDarkening
// decoded texture/channel map (from the .mshader source):
//   skin tint mask   = Material.G (squared: remap(G)*G)
//   pattern mask     = PatternTexture.R * Material.G
//   warpaint select  = PaintTexture.R  (single channel, 4-stop via paintSelect)
//   warpaint coverage= PaintTexture.G * Material.G
//   bioluminescence  = Bioluminescence.R on UVset1 (smoothed UV), tinted toward BIO_GREEN
//   hair length t    = HairMaps.G ; root-darken = HairMaps.R ; alpha = HairMaps.A ; AO on UVset1
//   hair-cap (face)  = HairCapMap.R on UVset1 -> lerp toward uHairCapColor

vec3 recolorSkin(vec3 albedo, float skinMask, vec3 skinColor,
                 vec3 patternColor, float patternMask,
                 vec3 paintColor, float paintCoverage,
                 float bioMask, vec3 bioColor,
                 vec3 hairCapColor, float hairCapMask,
                 float colorWeight, float patStrength){
    // SKIN colour - EXACT px_character_navi order: Overlay(blend=skinColor, base=albedo, Material.G^2)
    //   = lerp(albedo, overlayBlend(base=albedo, blend=skinColor), mask).
    // Overlay is NOT commutative: it branches on the BASE. Branching on the colour (the old bug)
    // sent saturated channels down the screen branch - skinColor.b=1 forced B to ~1 ("too blue").
    // overlayBlend(albedo, skinColor) below branches on albedo, matching the shader. No saturation
    // hybrid exists in the game; colorWeight is kept only for signature compatibility (unused).
    vec3 recol = overlayBlend(albedo, skinColor);   // base = albedo (diffuse), blend = skin colour
    vec3 c = mix(albedo, recol, skinMask);
    c = overlayTint(c, patternColor, clamp(patternMask * max(patStrength, 0.0), 0.0, 1.0));
    c = mix(c, bioColor, bioMask);
    c = mix(c, paintColor, paintCoverage);
    c = mix(c, hairCapColor, hairCapMask);
    return c;
}
"""
