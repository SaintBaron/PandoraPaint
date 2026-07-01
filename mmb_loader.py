"""Headless reader for Snowdrop .mmb skeletal-mesh assets.

Ported (read path, LOD0) from the AFoP Mesh Tool Blender addon (AlexPo, JasperZebra, J-Lyt,
SaintBaron) so it runs without Blender. Returns render-ready SubMesh geometry objects.
Supports versions 11-17; reads positions, uv0 and indices, recomputing normals from faces.
Older versions (11-14) differ only in the header/mesh-record layout (header skip, pre-LOD
root_bone/lod_info_type fields, LOD record size, and the tail); once stride/type/offsets are
parsed, the geometry decode is identical across versions.
"""

from __future__ import annotations
import io
import struct
import numpy as np


class SubMesh:
    """Render-ready geometry for one mesh section (positions, normals, uv0/uv1, indices, ...)."""

    __slots__ = (
        "name", "positions", "normals", "uv0", "uv1",
        "indices", "material_hash", "colors", "tangents",
    )

    def __init__(self, name, positions, normals, uv0, indices, material_hash,
                 uv1=None, colors=None, tangents=None):
        self.name = name
        self.positions = positions  # (N,3) float32
        self.normals = normals  # (N,3) float32
        self.uv0 = uv0  # (N,2) float32 - skin / pattern coat UV
        # uv1: second UV set; falls back to a copy of uv0 when only one channel exists.
        self.uv1 = uv1 if uv1 is not None else np.array(uv0, np.float32)
        self.indices = indices  # (M,) uint32 flat triangle list
        self.material_hash = material_hash
        # colors: (N,4) float32 RGBA vertex colours in 0..1, or None. The eye-shell shader uses
        # R as opacity (dither-discard) and G as a shading blend; most meshes have none.
        self.colors = colors
        # tangents: (N,3) float32 authored per-vertex strand tangent (hair sheen direction), or None.
        self.tangents = tangents


def model_bounds(meshes):
    """Bounding sphere (center, radius) over every submesh's vertices."""
    allpos = np.concatenate([m.positions for m in meshes], 0)
    lo = allpos.min(0)
    hi = allpos.max(0)
    return (lo + hi) * 0.5, float(np.linalg.norm(hi - lo) * 0.5)


FLIP_X = True  # match the viewer orientation (engine X is mirrored)


    # --- UV encoding from the binary divisor table (per-UV-set float32 divisor; AFoP Mesh Tool) ---
    # Table location: v12-v15 = 4-byte block after the post-uv 'unk' (else read as colour hashes);
    # v16/v17 = a dedicated count_c block after the colour hashes. count == uv_count. Divisor values:
    #   0.0 -> float32 (8 B/vert)   4095.0 -> compact ((u16 % 4096)/4095)
    #   4096.0 -> wide (i16/4096, hair/accessory x8)   32767.0 -> int16_norm (i16/32767, [-1,1])
    # Authoritative (replaces the old magnitude heuristic). Accepted only when every entry is a known
    # divisor AND count == uv_count, so genuine hashes are never mistaken for a table.
_UV_KNOWN_DIVISORS = (0.0, 4095.0, 4096.0, 32767.0)


def _uv_divisor_candidates(raw4_list):
    """Decode a list of 4-byte chunks to float32 divisors, or None if any value is not a recognised
    divisor (i.e. the block is hashes, not a divisor table)."""
    out = []
    for b in raw4_list:
        if len(b) != 4:
            return None
        v = struct.unpack("<f", b)[0]
        if v not in _UV_KNOWN_DIVISORS:
            return None
        out.append(v)
    return out


def _encoding_from_divisor(div):
    """Map a divisor float to an encoding name ('float32'|'wide'|'int16_norm'|'compact') or None."""
    if div == 0.0:
        return "float32"
    if div == 4096.0:
        return "wide"
    if div == 32767.0:
        return "int16_norm"
    if div == 4095.0:
        return "compact"
    return None


def _decode_uv_with_divisor(raw, div):
    """Decode ONE packed UV set using the authoritative file divisor (4096 / 32767 / 4095).
    raw: (N,2) uint16 = [u_raw, v_raw]. Returns (N,2) float32 - matches the AFoP Mesh Tool get_uvs:
    wide -> signed int16 / 4096, int16_norm -> signed int16 / 32767, compact -> (uint16 % 4096)/4095."""
    ru = raw[:, 0].astype(np.int64)
    rv = raw[:, 1].astype(np.int64)
    if div == 4095.0:  # 12-bit compact unorm
        u = (ru % 4096) / 4095.0
        v = (rv % 4096) / 4095.0
    else:  # wide (4096) or int16_norm (32767): signed int16 / div
        su = (ru ^ 32768) - 32768
        sv = (rv ^ 32768) - 32768
        u = su / div
        v = sv / div
    return np.ascontiguousarray(np.stack([u, v], 1).astype(np.float32))


def _mat4_from_16f(b):
    """64 bytes -> 4x4 float32 (row-major as stored)."""
    return np.frombuffer(b, np.float32, count=16).reshape(4, 4).astype(np.float32)


def _mat4_from_12f(b):
    """48 bytes -> 4x4: a 3x4 row-major affine (rotation 3x3 + translation col), bottom row 0001.
    NOTE: layout (3x4 vs 4x3, row vs column major) is an assumption to verify on real meshes;
    the viewer can transpose / switch placement mode via its NAVI_BIND_* constants."""
    a = np.frombuffer(b, np.float32, count=12).reshape(3, 4).astype(np.float32)
    M = np.eye(4, dtype=np.float32)
    M[:3, :] = a
    return M


class _R:
    """Cursor reader over a bytes buffer."""

    def __init__(self, buf):
        self.f = io.BytesIO(buf)

    def seek(self, o, w=0):
        self.f.seek(o, w)

    def tell(self):
        return self.f.tell()

    def read(self, n):
        return self.f.read(n)

    def u8(self):
        return self.f.read(1)[0]

    def u16(self):
        return struct.unpack("<H", self.f.read(2))[0]

    def i16(self):
        return struct.unpack("<h", self.f.read(2))[0]

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def f32(self):
        return struct.unpack("<f", self.f.read(4))[0]

    def name(self):
        n = self.u16()
        return self.f.read(n).decode("latin1").rstrip("\x00")

    def int16_norm(self):
        i = self.u16()
        v = (i ^ 0x8000) - 0x8000
        return v / 32767.0


class Mesh:
    def __init__(self):
        self.name = ""
        self.lods = []
        self.vertex_stride = 0
        self.normals_stride = 0
        self.uv_count = 0
        self.color_count = 0
        self.uv_divisors = None  # per-UV-set divisor table read from the file (authoritative)
        self.normal_type = 0
        self.position_type = 0
        self.color_in_normals = True
        self.bind = np.eye(
            4, dtype=np.float32
        )  # per-mesh bind/placement (48-byte block)
        self.influences = []  # [(4x4 inverse-bind, bone_index), ...]
        self.root_bone = -1  # bone this mesh is parented to


class LOD:
    lod_unk = 0  # v11 only: extra uint32 after vertex_count


def _parse(buf):
    r = _R(buf)
    magic = r.read(3)
    if magic != b"MMB":
        raise ValueError("not an MMB file")
    version = r.u8()
    size = r.u32()
    if version >= 15:
        r.seek(4, 1)  # 4-byte header field present only on v15+
    if version not in (11, 12, 13, 14, 15, 16, 17):
        raise ValueError(f"unsupported .mmb version {version} (this loader does 11-17)")

    bone_count = r.u32()
    bones = []
    for _ in range(bone_count):  # skeleton: name + 4x4 matrix + parent (u16)
        bname = r.name()
        bmat = _mat4_from_16f(r.read(64))
        bparent = r.u16()
        bones.append({"name": bname, "matrix": bmat, "parent": bparent})

    mesh_count = r.u32()
    meshes = []
    for _ in range(mesh_count):
        meshes.append(_parse_mesh(r, version))
    return version, size, meshes, bones


def _parse_mesh(r, version):
    m = Mesh()
    m.name = r.name()
    m.bind = _mat4_from_12f(r.read(48))  # per-mesh bind/placement
    r.seek(1, 1)
    if version == 11:
        r.seek(1, 1)  # skip x_count byte
        r.seek(4 * r.u16(), 1)  # then a u16 count of 4-byte entries
    else:
        x_count = r.u8()
        r.seek(1 + 4 * x_count, 1)
    u_count = r.u16()
    for _ in range(u_count):  # per-influence 4x4 matrix + bone index
        inf_mat = _mat4_from_16f(r.read(64))
        inf_bone = r.u16()
        m.influences.append((inf_mat, inf_bone))

        # Pre-LOD: root_bone_index present only when u_count>0 and not v11/v12 (1 B on v13/v14,
        # 2 B on v15+). lod_info_type is v14-v17 only (else 0). See AFoP Mesh Tool Mesh.parse.
    if u_count > 0 and version not in (11, 12):
        if version in (13, 14):
            m.root_bone = r.u8()  # 1-byte root_bone_index
        else:  # v15/16/17
            m.root_bone = r.u16()  # 2-byte root_bone_index
        lod_info_type = r.u8()
    elif version in (11, 12, 13):
        lod_info_type = 0  # no lod_info_type byte on these versions
    else:  # v14/15/16/17 with u_count == 0
        lod_info_type = r.u8()

    lod_count = r.u8()
    r.seek(4, 1)
    for li in range(lod_count):
        lod = LOD()
        lod.index = li
        lod.vertex_count = r.u32()
        if version == 11:
            lod.lod_unk = (
                r.u32()
            )  # v11 only: extra u32 after vertex_count (40-byte LOD)
        lod.index_count = r.u32()
        lod.size_a = r.u32()
        lod.vertex_data_offset_a = r.u32()
        lod.vertex_data_offset_b = r.u32()
        lod.face_block_offset = r.u32()
        lod.data_offset = r.u32()
        lod.data_size = r.u32()
        lod.screen_size = r.f32()
        if lod_info_type == 2:
            r.seek(28, 1)
        m.lods.append(lod)

    # tail: uv hashes, color hashes, strides (per-version layout)
    if version == 11:
        r.seek(8, 1)  # v11 only: two unknown 4-byte fields before uv_count
    m.uv_count = r.u8()
    r.seek(4 * m.uv_count, 1)  # skip the uv hashes
        # uv_divisors: per-UV-set divisor table read from the file (block position is version-specific;
        # same cursor advance as the old seeks). Accepted only when every entry is a known divisor and
        # count == uv_count. Authoritative for UV decode; magnitude heuristic is the fallback.
    m.uv_divisors = None
    if version == 11:
        m.color_count = 0  # v11 does not store color_count
    elif version in (16, 17):
        m.color_count = r.u8()
        r.seek(4 * m.color_count, 1)  # skip the real colour hashes
        r.seek(4, 1)  # unk after colour (v16/v17)
        count_c = r.u8()
        div_raw = [r.read(4) for _ in range(count_c)]  # dedicated divisor block
        if count_c == m.uv_count:
            m.uv_divisors = _uv_divisor_candidates(div_raw)
    else:  # v12/v13/v14/v15: the divisor table IS the post-uv 'colour' block
        r.seek(4, 1)  # unk before colour
        m.color_count = r.u8()
        div_raw = [r.read(4) for _ in range(m.color_count)]
        if m.color_count == m.uv_count:
            m.uv_divisors = _uv_divisor_candidates(div_raw)

    m.vertex_stride = r.u16()
    m.normals_stride = r.u16()

    nb_with_color = m.normals_stride - 4 * m.color_count - 4 * m.uv_count
    m.color_in_normals = nb_with_color >= 8
    normals_base = m.normals_stride - 4 * m.uv_count - 4 * m.color_count
    m.normal_type = 1 if normals_base >= 28 else 0
    if m.vertex_stride in (32, 40):
        m.position_type = 0
    elif m.vertex_stride in (28, 36):
        m.position_type = 1
    elif normals_base >= 28:
        m.position_type = 1
    else:
        m.position_type = 0

    r.seek(20 if version == 17 else 16, 1)
    return m


def _extract(buf, lods):
    out = io.BytesIO()
    for lod in reversed(lods):  # engine stores LOD blocks reversed
        out.write(buf[lod.data_offset : lod.data_offset + lod.data_size])
    return out.getvalue()


def _positions(ext, m, lod):
    n = lod.vertex_count
    stride = m.vertex_stride
    off = lod.vertex_data_offset_a
    region = np.frombuffer(ext, np.uint8, count=n * stride, offset=off).reshape(
        n, stride
    )
    if m.position_type == 0:
        q = region[:, :8].copy().view(np.int16).reshape(n, 4).astype(np.float64)
        return ((q[:, :3] / 32767.0) * q[:, 3:4]).astype(np.float32)
    return region[:, :12].copy().view(np.float32).reshape(n, 3).astype(np.float32)


def _decode_packed_uv_set(raw):
    """Decode ONE non-float32 UV set (raw: (N,2) uint16 = [u_raw, v_raw]) the way the AFoP Mesh
    Tool get_uvs() does. A set with any raw value >32767 is signed: either 4.12 fixed-point (int16/4096,
    peak near 4096 - Na'vi accessories) or a signed normalized short (int16/32767, peak near 32768 -
    banshee skin, whose V runs [-1,1] as mirrored left/right islands), split on peak magnitude. A set
    with no value >32767 is a 15-bit normalized short (int16/32767, any |s|>8191) or a 12-bit compact
    set ((uint16 % 4096)/4095). The genuinely tiled hair-card set is float32 and handled in _uvs()."""
    ru = raw[:, 0].astype(np.int64)
    rv = raw[:, 1].astype(np.int64)
    has_large_u = bool(np.any(ru > 32767))
    has_large_v = bool(np.any(rv > 32767))
        # A packed set with any raw value >32767 (negative as signed int16) is signed 4.12 fixed-point:
        # int16/4096 (verified vs .cast export, err 0). Modulo-wrap/fold is never geometrically correct
        # here (genuinely tiled sets are float32, not packed), so both are removed.
    if has_large_u or has_large_v:
        su = (ru ^ 32768) - 32768
        sv = (rv ^ 32768) - 32768
            # Signed data. 4.12 fixed-point (Na'vi accessories): 1.0 at 4096, [0,1]-ish set peaks
            # ~4096-8192 -> int16/4096. Else signed normalized short -> int16/32767 (e.g. banshee skin:
            # U in [0,1], V in [-1,1] as mirrored L/R islands). uint16/65535 squashes it; /4096 tiles 8x.
        peak = int(max(np.abs(su).max(initial=0), np.abs(sv).max(initial=0)))
        div = 4096.0 if peak <= 16384 else 32767.0
        u = su / div
        v = sv / div
    else:
        su = (ru ^ 32768) - 32768  # signed int16
        sv = (rv ^ 32768) - 32768
            # int16_norm if EITHER axis uses >~13 bits. Genuine 12-bit compact values are <=4095 on
            # BOTH axes; testing only U mis-routed small-U/large-V sets (e.g. Harness_flap Vmax 32479)
            # into the compact branch, whose `% 4096` shredded the island. Test U and V both.
        if np.any(np.abs(su) > 8191) or np.any(
            np.abs(sv) > 8191
        ):  # int16_norm -> [0,1]
            u = su / 32767.0
            v = sv / 32767.0
        else:  # compact (12-bit) -> [0,1]
            u = (ru % 4096) / 4095.0
            v = (rv % 4096) / 4095.0
    return np.ascontiguousarray(np.stack([u, v], 1).astype(np.float32))


def _uvs(ext, m, lod):
    """Read UV0 and (if present) UV1 from the normals stream, replicating the AFoP Mesh Tool decode.
    UV sets are walked in order: a float32 set is 8 bytes wide, every packed set 4 bytes, so an
    earlier float32 set shifts the field offset of the next - which the previous fixed-stride reader
    got wrong. UV1 is the body/head decal channel."""
    n = lod.vertex_count
    stride = m.normals_stride
    cc = m.color_count if m.color_in_normals else 0
    color_prefix = 4 * cc
    normal_block = 8 if m.normal_type == 0 else (12 + 12 + 4)
    base = lod.vertex_data_offset_b
    region = np.frombuffer(ext, np.uint8, count=n * stride, offset=base).reshape(
        n, stride
    )
    nuv = max(1, m.uv_count)
    divs = m.uv_divisors  # authoritative per-set divisors, or None when the file has no table

    sets = []
        # UV region start. Packed UV sets sit at the END of the normals stride (after colour prefix +
        # normal/tangent block). The forward guess (colour + FIXED normal-block size) mis-sizes meshes
        # whose normal block isn't 8/28 B (e.g. some v15 kuru use 16). With a divisor table we know each
        # set's width (8 B float32, 4 B packed) and anchor to the stride end instead. No table -> guess.
    if divs:
        total_uv_bytes = sum(8 if d == 0.0 else 4 for d in divs[:nuv])
        cur = max(color_prefix + normal_block, stride - total_uv_bytes)
    else:
        cur = color_prefix + normal_block
    for _s in range(nuv):
        div = divs[_s] if (divs is not None and _s < len(divs)) else None
            # float32 plausibility probe: the slot reads as a plausible float (0 or 1e-4<|f|<500) for
            # >90% of verts only if truly float32. Confirms a 0.0 divisor; drives the legacy guess.
        plausible_f32 = False
        if n > 0 and cur + 4 <= stride:
            fv = region[:, cur : cur + 4].copy().view(np.float32).reshape(n)
            plausible = np.isfinite(fv) & (
                (fv == 0.0) | ((np.abs(fv) > 1e-4) & (np.abs(fv) < 500.0))
            )
            plausible_f32 = plausible.mean() > 0.90
            # Decide encoding: the divisor table is authoritative; magnitude heuristic only when absent.
        if div is not None:
            enc = _encoding_from_divisor(div)
            if enc == "float32" and not plausible_f32:
                enc = "int16_norm"  # stray 0.0 entry: bytes aren't real floats -> signed short
        elif m.normal_type == 0 and plausible_f32:
            enc = "float32"  # legacy float32 detection (only for normal_type 0, per the importer)
        else:
            enc = "legacy"  # no table -> magnitude-heuristic decode below
        if enc == "float32":
            uv = (
                region[:, cur : cur + 8]
                .copy()
                .view(np.float32)
                .reshape(n, 2)
                .astype(np.float32)
                if cur + 8 <= stride
                else np.zeros((n, 2), np.float32)
            )
            sets.append(np.ascontiguousarray(uv))
            cur += 8
        else:
            raw = (
                region[:, cur : cur + 4].copy().view(np.uint16).reshape(n, 2)
                if cur + 4 <= stride
                else np.zeros((n, 2), np.uint16)
            )
            if enc == "legacy":
                sets.append(_decode_packed_uv_set(raw))  # heuristic fallback (no divisor table)
            else:  # 'wide' | 'int16_norm' | 'compact' - use the chosen encoding's own divisor, NOT
                # the raw file value (which is 0.0 when a float32 set was reclassified to int16_norm).
                eff_div = {"wide": 4096.0, "int16_norm": 32767.0, "compact": 4095.0}[enc]
                sets.append(_decode_uv_with_divisor(raw, eff_div))
            cur += 4

    uv0 = sets[0]
    uv1 = sets[1] if len(sets) > 1 else None
    return uv0, uv1


def _faces(ext, lod):
    r = _R(ext)
    r.seek(lod.face_block_offset)
    n = lod.index_count
    use32 = False
    if n > 0:
        if (
            lod.size_a == lod.face_block_offset // 4
            and lod.size_a != lod.face_block_offset // 2
        ):
            use32 = True
        else:
            peek = ext[lod.face_block_offset : lod.face_block_offset + 16]
            if len(peek) >= 16:
                hi = [struct.unpack("<H", peek[i : i + 2])[0] for i in range(2, 16, 4)]
                use32 = all(v == 0 for v in hi)
    if use32:
        idx = np.frombuffer(ext, np.uint32, count=n, offset=lod.face_block_offset)
    else:
        idx = np.frombuffer(
            ext, np.uint16, count=n, offset=lod.face_block_offset
        ).astype(np.uint32)
    return idx.copy()


def _normals_from_faces(P, idx):
    nv = P.shape[0]
    tri = idx.reshape(-1, 3)
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    fn = np.cross(b - a, c - a)  # area-weighted face normals
    flat = tri.reshape(-1)  # corner -> vertex
    w = np.repeat(fn, 3, axis=0)  # face normal per corner
    nrm = np.empty((nv, 3), np.float32)
    for k in range(3):  # scatter-add via bincount (fast, C-level)
        nrm[:, k] = np.bincount(flat, weights=w[:, k], minlength=nv)[:nv]
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    return np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 1e-12)


def _authored_normals(ext, m, lod):
    """Decode the AUTHORED per-vertex normals from the normals stream, instead of recomputing flat
    face normals (which discards the smooth, scalp-combed normals the artist authored - the cause of
    harsh per-facet hair lighting). The normal sits right after the optional colour prefix, at the
    front of the same `normal_block` that _uvs() steps over: 3x signed int8 / 127 when normal_type==0
    (8-byte block = normal[4] + tangent[4]), or 3x float32 when normal_type!=0 (12+12+4 block).
    Returns (N,3) unit float32, or None if the layout doesn't yield plausible unit normals (then the
    caller falls back to face normals)."""
    n = lod.vertex_count
    stride = m.normals_stride
    base = lod.vertex_data_offset_b
    if n == 0 or base + n * stride > len(ext):
        return None
    region = np.frombuffer(ext, np.uint8, count=n * stride, offset=base).reshape(
        n, stride
    )
    off = 4 * (m.color_count if m.color_in_normals else 0)
    if m.normal_type == 0:
        if off + 3 > stride:
            return None
        N = region[:, off : off + 3].astype(np.int8).astype(np.float32) / 127.0
    else:
        if off + 12 > stride:
            return None
        N = region[:, off : off + 12].copy().view(np.float32).reshape(n, 3)
    ln = np.linalg.norm(N, axis=1)
    if (
        np.mean(np.abs(ln - 1.0) < 0.1) < 0.80
    ):  # not unit normals -> wrong layout, bail to faces
        return None
        # Tangent-space guard (AFoP Mesh Tool get_normals): if raw X never dips below ~0, these are
        # tangent-space normals (unit-length but wrong in object space) - bail to computed face normals,
        # we can't resolve them without the per-vertex tangent basis.
    if N[:, 0].min() > -0.05:
        return None
    N = np.divide(N, ln[:, None], out=np.zeros_like(N), where=ln[:, None] > 1e-6)
    return np.ascontiguousarray(N.astype(np.float32))


def _authored_tangents(ext, m, lod):
    """Decode the authored per-vertex strand TANGENT - the second half of the normal_block (bytes
    4:7 as signed int8 / 127 when normal_type==0, or the 12:24 float3 slot otherwise). This is the
    strand-flow direction the game feeds its anisotropic hair sheen; without it the stretched strand
    cards render as flat texture. Returns (N,3) unit float32 or None (then no sheen tangent)."""
    n = lod.vertex_count
    stride = m.normals_stride
    base = lod.vertex_data_offset_b
    if n == 0 or base + n * stride > len(ext):
        return None
    region = np.frombuffer(ext, np.uint8, count=n * stride, offset=base).reshape(
        n, stride
    )
    off = 4 * (m.color_count if m.color_in_normals else 0)
    if m.normal_type == 0:
        if off + 7 > stride:
            return None
        T = region[:, off + 4 : off + 7].astype(np.int8).astype(np.float32) / 127.0
    else:
        if off + 24 > stride:
            return None
        T = region[:, off + 12 : off + 24].copy().view(np.float32).reshape(n, 3)
    ln = np.linalg.norm(T, axis=1)
    if np.mean(np.abs(ln - 1.0) < 0.1) < 0.80:
        return None
    T = np.divide(T, ln[:, None], out=np.zeros_like(T), where=ln[:, None] > 1e-6)
    return np.ascontiguousarray(T.astype(np.float32))


def _colors(ext, m, lod):
    """Read vertex colours from the normals stream, if present. They sit at the very START of that
    stream (the same `4 * color_count` bytes that _uvs() skips over via its `pre` offset), stored as
    RGBA8 (4 bytes per colour). Returns (N,4) float32 in 0..1, or None when the mesh has no colours.

    The eye-shell shader (px_character_eye_shell) reads R as opacity (dither-discard) and G as a
    shading blend, so this is what we need to render the white-of-eye / cornea correctly."""
    if not (m.color_in_normals and m.color_count >= 1):
        return None
    n = lod.vertex_count
    stride = m.normals_stride
    base = lod.vertex_data_offset_b
    need = base + n * stride
    if need > len(ext):
        return None
    region = np.frombuffer(ext, np.uint8, count=n * stride, offset=base).reshape(
        n, stride
    )
    # colour layer 0 = first 4 bytes (RGBA8). Higher layers follow but the shell only uses VC0.
    rgba8 = region[:, 0:4].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgba8)


def _weights(ext, m, lod):
    """Per-vertex bone skin weights, ported from the AFoP Mesh Tool get_bone_weights. Reads ONLY the
    position/skin stream (offset_a) - it never touches the normal stream, so it has no effect on
    shading. Layout depends on vertex_stride (and, for stride 32, a sub-layout probed from the data).
    Returns a list of {mesh_bone_slot: weight} dicts (one per vertex), or None when absent."""
    n = lod.vertex_count
    stride = m.vertex_stride
    base = lod.vertex_data_offset_a
    if n == 0 or stride == 0 or base + n * stride > len(ext):
        return None
    ptype = m.position_type
    n_slots = len(m.influences)

    def u16(o):
        return struct.unpack_from("<H", ext, o)[0]

    layout32 = None
    if stride == 32:
        p = base + 8
        w8 = [u16(p + i * 2) for i in range(8)]
        if sum(w8) == 32767:
            layout32 = "A"
        else:
            c12 = ext[p + 12 : p + 24]
            layout32 = "C" if (n_slots <= 256 and all(0 <= x < n_slots for x in c12)) else "B"

    out = []
    for v in range(n):
        vo = base + v * stride
        iw = {}
        if stride == 12:
            iw[ext[vo + 8]] = 1.0
        elif stride == 16:
            if ptype == 1:
                iw[ext[vo + 12]] = 1.0
            else:
                w0, w1 = ext[vo + 8] / 255.0, ext[vo + 9] / 255.0
                i0, i1 = u16(vo + 12), u16(vo + 14)
                if w0 > 0.0:
                    iw[i0] = iw.get(i0, 0.0) + w0
                if w1 > 0.0:
                    iw[i1] = iw.get(i1, 0.0) + w1
        elif stride == 20:
            base_o = 12 if ptype == 1 else 8
            o = vo + base_o
            wc = ((stride - base_o) - 4) // 2
            ws = [u16(o + 2 * k) / 32767.0 for k in range(wc)]
            io = o + 2 * wc
            for k in range(wc):
                if ws[k] > 0.0:
                    iw[ext[io + k]] = ws[k]
        elif stride == 32:
            o = vo + 8
            if layout32 == "A":
                ws = [u16(o + 2 * k) / 32767.0 for k in range(8)]
                for k in range(8):
                    if ws[k] > 0.0:
                        idx = ext[o + 16 + k]
                        iw[idx] = iw.get(idx, 0.0) + ws[k]
            elif layout32 == "C":
                for k in range(12):
                    wt = ext[o + k] / 255.0
                    if wt > 0.0:
                        idx = ext[o + 12 + k]
                        iw[idx] = iw.get(idx, 0.0) + wt
            else:
                for k in range(6):
                    wt = ext[o + k] / 255.0
                    if wt > 0.0:
                        idx = u16(o + 8 + 2 * k)
                        iw[idx] = iw.get(idx, 0.0) + wt
        elif stride == 36:
            o = vo + 12
            ws = [u16(o + 2 * k) / 32767.0 for k in range(8)]
            for k in range(8):
                if ws[k] > 0.0:
                    iw[ext[o + 16 + k]] = ws[k]
        elif stride == 40:
            o = vo + 8
            ws = [u16(o + 2 * k) / 32767.0 for k in range(8)]
            for k in range(8):
                if ws[k] > 0.0:
                    iw[u16(o + 16 + 2 * k)] = ws[k]
        elif stride == 44:
            base_o = 12 if ptype == 1 else 8
            o = vo + base_o
            ws = [ext[o + k] / 255.0 for k in range(12)]
            for k in range(12):
                if ws[k] > 0.0:
                    iw[u16(o + 12 + 2 * k)] = ws[k]
        else:
            pl = 12 if ptype == 1 else 8
            o = vo + pl
            wc = (stride - pl) // 2
            ws = [wt for wt in (ext[o + k] / 255.0 for k in range(wc)) if wt > 0.0]
            io = o + wc
            for k in range(wc):
                if k < len(ws):
                    iw[ext[io + k]] = ws[k]
        out.append(iw)
    return out


def load_model(path):
    """Load a Snowdrop .mmb preview model. Returns (list[SubMesh], extra). Only .mmb is supported."""
    if not str(path).lower().endswith(".mmb"):
        raise ValueError("unsupported model format (only .mmb is supported): %s" % path)
    with open(path, "rb") as _f:
        buf = _f.read()
    version, size, meshes, bones = _parse(buf)
    out = []
    extra = {"skeleton": bones, "meshes": {}}
    for m in meshes:
        extra["meshes"][m.name] = {
            "bind": m.bind,
            "influences": m.influences,
            "root_bone": m.root_bone,
        }
        if not m.lods:
            continue
        lod = m.lods[0]  # LOD0 = highest detail
        if lod.vertex_count == 0 or lod.index_count == 0:
            continue
        ext = _extract(buf, m.lods)
        # per-vertex skin weights (position stream only; does not affect normals/shading)
        extra["meshes"][m.name]["weights"] = _weights(ext, m, lod)
        P = _positions(ext, m, lod)
        UV, UV1 = _uvs(ext, m, lod)
        VC = _colors(ext, m, lod)
        idx = _faces(ext, lod)
        if FLIP_X:
            P[:, 0] = -P[:, 0]
            idx = idx.reshape(-1, 3)[:, ::-1].reshape(-1)  # keep winding after mirror
        N = _authored_normals(ext, m, lod)  # real smooth normals from the file...
        if N is None:
            N = _normals_from_faces(
                P, idx
            )  # ...fall back to flat face normals only if absent
            T = None
        else:
            T = _authored_tangents(ext, m, lod)
            if FLIP_X:
                N[:, 0] = -N[:, 0]  # mirror authored normals to match the flipped X
                if T is not None:
                    T[:, 0] = -T[:, 0]
        out.append(
            SubMesh(
                m.name,
                P,
                N,
                UV,
                idx.astype(np.uint32),
                None,
                uv1=UV1,
                colors=VC,
                tangents=T,
            )
        )
    if not out:
        raise ValueError("no LOD0 geometry found in MMB")
    return out, extra
