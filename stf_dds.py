"""Read AFoP ".dds" textures -> HxWx4 uint8 RGBA.  (single self-contained file)

AFoP ships textures in a Snowdrop "STF" container (magic b"STF\\x02") that uses the .dds
extension but isn't a standard DirectDraw Surface. This module decodes them with a fully
self-contained, deterministic pipeline, with NO external module or Pillow/C dependency:

  * the codec is resolved from the STF class byte via a ground-truth CLASS_CODEC map
    (built from 132k converted-DDS headers) - no blockiness guessing;
  * the largest resident mip is decoded by a pure-Python BCn decoder (BC1/BC2/BC3/BC4/
    BC5/BC7 + uncompressed RGBA8/BGRA8/RGBA16/RGBA16F/RGBA32F/R8/R16/RG8), numpy-accelerated.

The BCn decoder and STF format model are embedded below (verbatim from afop_bcn.py /
afop_stf_scan.py) and exec'd into in-process modules so they stay easy to update from source.
Standard b"DDS " files are also handled. `load_dds(path, fmt=None)` returns a contiguous
HxWx4 uint8 RGBA numpy array (prior public API preserved).

BC6H (HDR, STF class 0x49) is not yet decodable and raises.
"""

from __future__ import annotations
import os
import sys
import struct
import types
import numpy as np

# ----------------------------------------------------------------------------
# Embedded decoder + format model. Exec'd into real in-process modules so the
# stf model's `from afop_bcn import _decode` resolves, and so the code below is a
# verbatim copy of afop_bcn.py / afop_stf_scan.py (update by re-pasting those).
# ----------------------------------------------------------------------------
_AFOP_BCN_SRC = r'''
#!/usr/bin/env python3
"""Pure-Python BC1/BC4/BC5/BC7 decoder for AFOP/Snowdrop STF textures.

No C extension, no cffi, no compiler - just Python, with numpy used for speed
when present (BC1/BC4/BC5 are fully vectorised; BC7 is per-block). API-compatible
with the previous bcdec build: bc1_decode / bc4_decode / bc5_decode / bc7_decode /
decode_best / _decode, all returning RGBA8 bytes of length w*h*4.

    from afop_bcn import bc7_decode, decode_best
    rgba = bc7_decode(mip_bytes, w, h)
    rgba, fmt = decode_best(mip_bytes, w, h, ("BC1", "BC4"))
"""
try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


def _expand(v, bits):
    """Replicate a `bits`-bit value up to 8 bits."""
    if bits >= 8:
        return v & 0xFF
    return ((v << (8 - bits)) | (v >> (2 * bits - 8))) & 0xFF


def _crop(buf, pw, ph, w, h):
    if pw == w and ph == h:
        return bytes(buf)
    out = bytearray(w * h * 4)
    row = w * 4
    for y in range(h):
        out[y * row:(y + 1) * row] = buf[y * pw * 4:y * pw * 4 + row]
    return bytes(out)


# ----------------------------------------------------------------- BC1 -------
def bc1_decode(data, w, h, force_four=False):
    bw, bh = (w + 3) // 4, (h + 3) // 4
    nb = bw * bh
    need = nb * 8
    if len(data) < need:
        raise ValueError("BC1 data too short")
    if not _NP:
        return _bc1_py(data, w, h, bw, bh)
    a = np.frombuffer(bytes(data[:need]), np.uint8).reshape(nb, 8).astype(np.uint32)
    c0 = a[:, 0] | (a[:, 1] << 8)
    c1 = a[:, 2] | (a[:, 3] << 8)

    def e565(c):
        r = (c >> 11) & 0x1F
        g = (c >> 5) & 0x3F
        b = c & 0x1F
        return ((r << 3) | (r >> 2)), ((g << 2) | (g >> 4)), ((b << 3) | (b >> 2))

    r0, g0, b0 = e565(c0)
    r1, g1, b1 = e565(c1)
    four = (c0 > c1) | force_four          # BC2/BC3 colour blocks are always 4-colour
    col = np.zeros((nb, 4, 4), np.uint16)
    col[:, 0] = np.stack([r0, g0, b0, np.full(nb, 255)], 1)
    col[:, 1] = np.stack([r1, g1, b1, np.full(nb, 255)], 1)
    r2 = np.where(four, (2 * r0 + r1) // 3, (r0 + r1) // 2)
    g2 = np.where(four, (2 * g0 + g1) // 3, (g0 + g1) // 2)
    b2 = np.where(four, (2 * b0 + b1) // 3, (b0 + b1) // 2)
    col[:, 2] = np.stack([r2, g2, b2, np.full(nb, 255)], 1)
    r3 = np.where(four, (r0 + 2 * r1) // 3, 0)
    g3 = np.where(four, (g0 + 2 * g1) // 3, 0)
    b3 = np.where(four, (b0 + 2 * b1) // 3, 0)
    a3 = np.where(four, 255, 0)
    col[:, 3] = np.stack([r3, g3, b3, a3], 1)
    idx = a[:, 4] | (a[:, 5] << 8) | (a[:, 6] << 16) | (a[:, 7] << 24)
    out = np.zeros((nb, 16, 4), np.uint8)
    rng = np.arange(nb)
    for p in range(16):
        out[:, p, :] = col[rng, (idx >> (2 * p)) & 3, :]
    img = out.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return img[:h, :w].tobytes()


def _bc1_py(data, w, h, bw, bh):
    pw, ph = bw * 4, bh * 4
    buf = bytearray(pw * ph * 4)
    for bi in range(bw * bh):
        o = bi * 8
        c0 = data[o] | (data[o + 1] << 8)
        c1 = data[o + 2] | (data[o + 3] << 8)

        def ex(c):
            r = (c >> 11) & 0x1F
            g = (c >> 5) & 0x3F
            b = c & 0x1F
            return [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2), 255]
        cols = [ex(c0), ex(c1)]
        if c0 > c1:
            cols.append([(2 * cols[0][k] + cols[1][k]) // 3 for k in range(3)] + [255])
            cols.append([(cols[0][k] + 2 * cols[1][k]) // 3 for k in range(3)] + [255])
        else:
            cols.append([(cols[0][k] + cols[1][k]) // 2 for k in range(3)] + [255])
            cols.append([0, 0, 0, 0])
        idx = data[o + 4] | (data[o + 5] << 8) | (data[o + 6] << 16) | (data[o + 7] << 24)
        bx, by = (bi % bw) * 4, (bi // bw) * 4
        for p in range(16):
            c = cols[(idx >> (2 * p)) & 3]
            x, y = bx + (p % 4), by + (p // 4)
            q = (y * pw + x) * 4
            buf[q:q + 4] = bytes(c)
    return _crop(buf, pw, ph, w, h)


# ----------------------------------------------------------- BC4 / BC5 -------
_W6 = (9363, 18724, 28086, 37450, 46812, 56173)
_W4 = (13107, 26215, 39321, 52429)


def _bc4_block_vals(r0, r1):
    v = [r0, r1]
    if r0 > r1:
        for k in range(6):
            v.append((_W6[5 - k] * r0 + _W6[k] * r1 + 32768) >> 16)
    else:
        for k in range(4):
            v.append((_W4[3 - k] * r0 + _W4[k] * r1 + 32768) >> 16)
        v += [0, 255]
    return v


def _bc4_channel_np(data, base, stride, bw, bh):
    """Vectorised BC4 channel -> (bh*4, bw*4) uint8 image (numpy)."""
    nb = bw * bh
    raw = np.frombuffer(bytes(data), np.uint8)
    starts = base + np.arange(nb) * stride
    blk = np.stack([raw[starts + k] for k in range(8)], 1).astype(np.int64)
    r0, r1 = blk[:, 0], blk[:, 1]
    vgt = np.zeros((nb, 8), np.int64); vle = np.zeros((nb, 8), np.int64)
    vgt[:, 0] = vle[:, 0] = r0; vgt[:, 1] = vle[:, 1] = r1
    for k in range(6):
        vgt[:, 2 + k] = (_W6[5 - k] * r0 + _W6[k] * r1 + 32768) >> 16
    for k in range(4):
        vle[:, 2 + k] = (_W4[3 - k] * r0 + _W4[k] * r1 + 32768) >> 16
    vle[:, 6] = 0; vle[:, 7] = 255
    vals = np.where((r0 > r1)[:, None], vgt, vle)
    bits = np.zeros(nb, np.uint64)
    for k in range(6):
        bits |= blk[:, 2 + k].astype(np.uint64) << np.uint64(8 * k)
    grid = np.zeros((nb, 16), np.uint8)
    rng = np.arange(nb)
    for p in range(16):
        idx = ((bits >> np.uint64(3 * p)) & np.uint64(7)).astype(np.int64)
        grid[:, p] = vals[rng, idx]
    return grid.reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4)


def _bc4_channel_py(data, base, stride, bw, bh):
    pw, ph = bw * 4, bh * 4
    grid = bytearray(pw * ph)
    for bi in range(bw * bh):
        o = base + bi * stride
        vals = _bc4_block_vals(data[o], data[o + 1])
        bits = int.from_bytes(data[o + 2:o + 8], "little")
        bx, by = (bi % bw) * 4, (bi // bw) * 4
        for p in range(16):
            grid[(by + (p >> 2)) * pw + bx + (p & 3)] = vals[(bits >> (3 * p)) & 7]
    return grid, pw, ph


def bc4_decode(data, w, h):
    bw, bh = (w + 3) // 4, (h + 3) // 4
    if len(data) < bw * bh * 8:
        raise ValueError("BC4 data too short")
    pw, ph = bw * 4, bh * 4
    if _NP:
        g = _bc4_channel_np(data, 0, 8, bw, bh)
        rgba = np.empty((ph, pw, 4), np.uint8)
        rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = g
        rgba[:, :, 3] = 255
        return rgba[:h, :w].tobytes()
    g, _, _ = _bc4_channel_py(data, 0, 8, bw, bh)
    buf = bytearray(pw * ph * 4)
    for i in range(pw * ph):
        v = g[i]; buf[i * 4:i * 4 + 4] = bytes((v, v, v, 255))
    return _crop(buf, pw, ph, w, h)


def _reconstruct_z(gr, gg):
    """Given R,G of a tangent-space normal (uint8 grids), rebuild B = sqrt(1-x^2-y^2)
    and return it as a uint8 grid (encoded 0..255)."""
    nx = gr.astype(np.float32) / 127.5 - 1.0
    ny = gg.astype(np.float32) / 127.5 - 1.0
    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
    return np.clip(nz * 127.5 + 127.5, 0, 255).astype(np.uint8)


def bc5_decode(data, w, h, reconstruct_z=False):
    bw, bh = (w + 3) // 4, (h + 3) // 4
    if len(data) < bw * bh * 16:
        raise ValueError("BC5 data too short")
    pw, ph = bw * 4, bh * 4
    if _NP:
        gr = _bc4_channel_np(data, 0, 16, bw, bh)
        gg = _bc4_channel_np(data, 8, 16, bw, bh)
        rgba = np.empty((ph, pw, 4), np.uint8)
        rgba[:, :, 0] = gr; rgba[:, :, 1] = gg
        rgba[:, :, 2] = _reconstruct_z(gr, gg) if reconstruct_z else 0
        rgba[:, :, 3] = 255
        return rgba[:h, :w].tobytes()
    gr, _, _ = _bc4_channel_py(data, 0, 16, bw, bh)
    gg, _, _ = _bc4_channel_py(data, 8, 16, bw, bh)
    buf = bytearray(pw * ph * 4)
    for i in range(pw * ph):
        if reconstruct_z:
            x = gr[i] / 127.5 - 1.0; y = gg[i] / 127.5 - 1.0
            z = max(0.0, 1.0 - x * x - y * y) ** 0.5
            b = int(min(255, max(0, z * 127.5 + 127.5)))
        else:
            b = 0
        buf[i * 4:i * 4 + 4] = bytes((gr[i], gg[i], b, 255))
    return _crop(buf, pw, ph, w, h)


def bc5n_decode(data, w, h):
    """BC5 normal map with the blue channel reconstructed."""
    return bc5_decode(data, w, h, reconstruct_z=True)


_BC7_PART = [
  [
    [128, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 129],
    [128, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 129],
    [128, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 129],
    [128, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 129],
    [128, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 129],
    [128, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 129],
    [128, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 129],
    [128, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 129],
    [128, 1, 129, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [128, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 0, 1, 1, 1, 0],
    [128, 1, 129, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [128, 0, 129, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [128, 0, 0, 0, 1, 0, 0, 0, 129, 1, 0, 0, 1, 1, 1, 0],
    [128, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 0, 1, 1, 0, 0],
    [128, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 129],
    [128, 0, 129, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [128, 0, 0, 0, 1, 0, 0, 0, 129, 0, 0, 0, 1, 1, 0, 0],
    [128, 1, 129, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [128, 0, 129, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [128, 0, 0, 1, 0, 1, 1, 1, 129, 1, 1, 0, 1, 0, 0, 0],
    [128, 0, 0, 0, 1, 1, 1, 1, 129, 1, 1, 1, 0, 0, 0, 0],
    [128, 1, 129, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [128, 0, 129, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [128, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 129],
    [128, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 129],
    [128, 1, 0, 1, 1, 0, 129, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    [128, 0, 1, 1, 0, 0, 1, 1, 129, 1, 0, 0, 1, 1, 0, 0],
    [128, 0, 129, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [128, 1, 0, 1, 0, 1, 0, 1, 129, 0, 1, 0, 1, 0, 1, 0],
    [128, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 129],
    [128, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 129],
    [128, 1, 129, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    [128, 0, 0, 1, 0, 0, 1, 1, 129, 1, 0, 0, 1, 0, 0, 0],
    [128, 0, 129, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [128, 0, 129, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [128, 1, 129, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [128, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 129],
    [128, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 129],
    [128, 0, 0, 0, 0, 1, 129, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [128, 1, 0, 0, 1, 1, 129, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [128, 0, 129, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [128, 0, 0, 0, 0, 0, 129, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [128, 0, 0, 0, 0, 1, 0, 0, 129, 1, 1, 0, 0, 1, 0, 0],
    [128, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 129],
    [128, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 129],
    [128, 1, 129, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [128, 0, 129, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    [128, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 129],
    [128, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 129],
    [128, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 129],
    [128, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 129],
    [128, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 129],
    [128, 0, 129, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [128, 0, 129, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [128, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 129],
  ],
  [
    [128, 0, 1, 129, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 130],
    [128, 0, 0, 129, 0, 0, 1, 1, 130, 2, 1, 1, 2, 2, 2, 1],
    [128, 0, 0, 0, 2, 0, 0, 1, 130, 2, 1, 1, 2, 2, 1, 129],
    [128, 2, 2, 130, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 0, 129, 1, 2, 2, 1, 1, 2, 130],
    [128, 0, 1, 129, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 130],
    [128, 0, 2, 130, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 129],
    [128, 0, 1, 1, 0, 0, 1, 1, 130, 2, 1, 1, 2, 2, 1, 129],
    [128, 0, 0, 0, 0, 0, 0, 0, 129, 1, 1, 1, 2, 2, 2, 130],
    [128, 0, 0, 0, 1, 1, 1, 1, 129, 1, 1, 1, 2, 2, 2, 130],
    [128, 0, 0, 0, 1, 1, 129, 1, 2, 2, 2, 2, 2, 2, 2, 130],
    [128, 0, 1, 2, 0, 0, 129, 2, 0, 0, 1, 2, 0, 0, 1, 130],
    [128, 1, 1, 2, 0, 1, 129, 2, 0, 1, 1, 2, 0, 1, 1, 130],
    [128, 1, 2, 2, 0, 129, 2, 2, 0, 1, 2, 2, 0, 1, 2, 130],
    [128, 0, 1, 129, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 130],
    [128, 0, 1, 129, 2, 0, 0, 1, 130, 2, 0, 0, 2, 2, 2, 0],
    [128, 0, 0, 129, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 130],
    [128, 1, 1, 129, 0, 0, 1, 1, 130, 0, 0, 1, 2, 2, 0, 0],
    [128, 0, 0, 0, 1, 1, 2, 2, 129, 1, 2, 2, 1, 1, 2, 130],
    [128, 0, 2, 130, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 129],
    [128, 1, 1, 129, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 130],
    [128, 0, 0, 129, 0, 0, 0, 1, 130, 2, 2, 1, 2, 2, 2, 1],
    [128, 0, 0, 0, 0, 0, 129, 1, 0, 1, 2, 2, 0, 1, 2, 130],
    [128, 0, 0, 0, 1, 1, 0, 0, 130, 2, 129, 0, 2, 2, 1, 0],
    [128, 1, 2, 130, 0, 129, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0],
    [128, 0, 1, 2, 0, 0, 1, 2, 129, 1, 2, 2, 2, 2, 2, 130],
    [128, 1, 1, 0, 1, 2, 130, 1, 129, 2, 2, 1, 0, 1, 1, 0],
    [128, 0, 0, 0, 0, 1, 129, 0, 1, 2, 130, 1, 1, 2, 2, 1],
    [128, 0, 2, 2, 1, 1, 0, 2, 129, 1, 0, 2, 0, 0, 2, 130],
    [128, 1, 1, 0, 0, 129, 1, 0, 2, 0, 0, 2, 2, 2, 2, 130],
    [128, 0, 1, 1, 0, 1, 2, 2, 0, 1, 130, 2, 0, 0, 1, 129],
    [128, 0, 0, 0, 2, 0, 0, 0, 130, 2, 1, 1, 2, 2, 2, 129],
    [128, 0, 0, 0, 0, 0, 0, 2, 129, 1, 2, 2, 1, 2, 2, 130],
    [128, 2, 2, 130, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 129],
    [128, 0, 1, 129, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 130],
    [128, 1, 2, 0, 0, 129, 2, 0, 0, 1, 130, 0, 0, 1, 2, 0],
    [128, 0, 0, 0, 1, 1, 129, 1, 2, 2, 130, 2, 0, 0, 0, 0],
    [128, 1, 2, 0, 1, 2, 0, 1, 130, 0, 129, 2, 0, 1, 2, 0],
    [128, 1, 2, 0, 2, 0, 1, 2, 129, 130, 0, 1, 0, 1, 2, 0],
    [128, 0, 1, 1, 2, 2, 0, 0, 1, 1, 130, 2, 0, 0, 1, 129],
    [128, 0, 1, 1, 1, 1, 130, 2, 2, 2, 0, 0, 0, 0, 1, 129],
    [128, 1, 0, 129, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 130],
    [128, 0, 0, 0, 0, 0, 0, 0, 130, 1, 2, 1, 2, 1, 2, 129],
    [128, 0, 2, 2, 1, 129, 2, 2, 0, 0, 2, 2, 1, 1, 2, 130],
    [128, 0, 2, 130, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 129],
    [128, 2, 2, 0, 1, 2, 130, 1, 0, 2, 2, 0, 1, 2, 2, 129],
    [128, 1, 0, 1, 2, 2, 130, 2, 2, 2, 2, 2, 0, 1, 0, 129],
    [128, 0, 0, 0, 2, 1, 2, 1, 130, 1, 2, 1, 2, 1, 2, 129],
    [128, 1, 0, 129, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 130],
    [128, 2, 2, 130, 0, 1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 129],
    [128, 0, 0, 2, 1, 129, 1, 2, 0, 0, 0, 2, 1, 1, 1, 130],
    [128, 0, 0, 0, 2, 129, 1, 2, 2, 1, 1, 2, 2, 1, 1, 130],
    [128, 2, 2, 2, 0, 129, 1, 1, 0, 1, 1, 1, 0, 2, 2, 130],
    [128, 0, 0, 2, 1, 1, 1, 2, 129, 1, 1, 2, 0, 0, 0, 130],
    [128, 1, 1, 0, 0, 129, 1, 0, 0, 1, 1, 0, 2, 2, 2, 130],
    [128, 0, 0, 0, 0, 0, 0, 0, 2, 1, 129, 2, 2, 1, 1, 130],
    [128, 1, 1, 0, 0, 129, 1, 0, 2, 2, 2, 2, 2, 2, 2, 130],
    [128, 0, 2, 2, 0, 0, 1, 1, 0, 0, 129, 1, 0, 0, 2, 130],
    [128, 0, 2, 2, 1, 1, 2, 2, 129, 1, 2, 2, 0, 0, 2, 130],
    [128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 129, 1, 130],
    [128, 0, 0, 130, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 129],
    [128, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 129, 2, 2, 130],
    [128, 1, 0, 129, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 130],
    [128, 1, 1, 129, 2, 0, 1, 1, 130, 2, 0, 1, 2, 2, 2, 0],
  ],
]

# ----------------------------------------------------------------- BC7 -------
_NS = (3, 2, 3, 2, 1, 1, 1, 2)
_PB = (4, 6, 6, 6, 0, 0, 0, 6)
_RB = (0, 0, 0, 0, 2, 2, 0, 0)
_ISB = (0, 0, 0, 0, 1, 0, 0, 0)
_CB = (4, 6, 5, 7, 5, 7, 7, 5)
_AB = (0, 0, 0, 0, 6, 8, 7, 5)
_EPB = (1, 0, 0, 1, 0, 0, 1, 1)
_SPB = (0, 1, 0, 0, 0, 0, 0, 0)
_IB = (3, 3, 2, 2, 2, 2, 4, 2)
_IB2 = (0, 0, 0, 0, 3, 2, 0, 0)
_WT = {2: (0, 21, 43, 64),
       3: (0, 9, 18, 27, 37, 46, 55, 64),
       4: (0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64)}


def _interp(a, b, w):
    return (a * (64 - w) + b * w + 32) >> 6


def _bc7_block(blk):
    v = int.from_bytes(blk, "little")
    pos = [0]

    def rd(n):
        r = (v >> pos[0]) & ((1 << n) - 1)
        pos[0] += n
        return r

    mode = 0
    while mode < 8 and rd(1) == 0:
        mode += 1
    if mode == 8:
        return [(0, 0, 0, 0)] * 16
    NS, CB, AB, IB, IB2 = _NS[mode], _CB[mode], _AB[mode], _IB[mode], _IB2[mode]
    part = rd(_PB[mode]) if _PB[mode] else 0
    rot = rd(_RB[mode]) if _RB[mode] else 0
    isel = rd(_ISB[mode]) if _ISB[mode] else 0
    ne = 2 * NS
    R = [rd(CB) for _ in range(ne)]
    G = [rd(CB) for _ in range(ne)]
    B = [rd(CB) for _ in range(ne)]
    A = [rd(AB) for _ in range(ne)] if AB else None
    if _EPB[mode]:
        pb = [rd(1) for _ in range(ne)]
    elif _SPB[mode]:
        sp = [rd(1) for _ in range(NS)]
        pb = [sp[i // 2] for i in range(ne)]
    else:
        pb = None
    cb = CB + (1 if pb is not None else 0)
    ab = AB + (1 if (pb is not None and AB) else 0)
    eps = []
    for i in range(ne):
        if pb is not None:
            r = _expand((R[i] << 1) | pb[i], cb)
            g = _expand((G[i] << 1) | pb[i], cb)
            b = _expand((B[i] << 1) | pb[i], cb)
            a = _expand((A[i] << 1) | pb[i], ab) if AB else 255
        else:
            r = _expand(R[i], CB)
            g = _expand(G[i], CB)
            b = _expand(B[i], CB)
            a = _expand(A[i], AB) if AB else 255
        eps.append((r, g, b, a))
    ptab = _BC7_PART[NS - 2][part] if NS >= 2 else None

    def subset(p):
        return 0 if NS == 1 else (ptab[p] & 3)

    def anchor(p):
        if p == 0:
            return True
        return bool(ptab[p] & 0x80) if NS >= 2 else False

    cidx = [rd((IB - 1) if anchor(p) else IB) for p in range(16)]
    aidx = [rd((IB2 - 1) if p == 0 else IB2) for p in range(16)] if IB2 else None
    cw = _WT[IB]
    aw = _WT[IB2] if IB2 else None
    out = []
    for p in range(16):
        s = subset(p)
        e0, e1 = eps[2 * s], eps[2 * s + 1]
        if IB2:
            ci, ai = cidx[p], aidx[p]
            Wc, Wa = (aw, cw) if isel else (cw, aw)
            if isel:
                ci, ai = ai, ci
            r = _interp(e0[0], e1[0], Wc[ci])
            g = _interp(e0[1], e1[1], Wc[ci])
            b = _interp(e0[2], e1[2], Wc[ci])
            a = _interp(e0[3], e1[3], Wa[ai])
        else:
            wt = cw[cidx[p]]
            r = _interp(e0[0], e1[0], wt)
            g = _interp(e0[1], e1[1], wt)
            b = _interp(e0[2], e1[2], wt)
            a = _interp(e0[3], e1[3], wt)
        if rot == 1:
            a, r = r, a
        elif rot == 2:
            a, g = g, a
        elif rot == 3:
            a, b = b, a
        out.append((r, g, b, a))
    return out


def _getbits(blocks, off, width):
    """Extract a fixed-offset, fixed-width little-endian field across all blocks.
    blocks: (n,16) uint8 -> (n,) uint64."""
    b0 = off >> 3
    bit = off & 7
    nby = (bit + width + 7) >> 3
    acc = np.zeros(len(blocks), np.uint64)
    for k in range(nby):
        acc |= blocks[:, b0 + k].astype(np.uint64) << np.uint64(8 * k)
    return (acc >> np.uint64(bit)) & np.uint64((1 << width) - 1)


def _bc7_modes(blocks):
    """Mode per block = index of lowest set bit in byte 0 (8 == invalid)."""
    b0 = blocks[:, 0].astype(np.int32)
    mode = np.full(len(blocks), 8, np.int32)
    for m in range(8):
        mode = np.where((mode == 8) & ((b0 >> m) & 1 == 1), m, mode)
    return mode


def _exp(v, bits):
    if bits >= 8:
        return v
    return (v << (8 - bits)) | (v >> (2 * bits - 8))


# fixed bit layouts for the single-subset modes (anchor only at pixel 0)
#   mode: (color_bits, alpha_bits, has_pbit, ib, ib2, rot_off, isel_off, ep_off)
_VEC = {6: (7, 7, True, 4, 0, None, None, 7),
        5: (7, 8, False, 2, 2, 6, None, 8),
        4: (5, 6, False, 2, 3, 5, 7, 8)}


def _bc7_vec_single(blocks, mode):
    """Vectorised decode of single-subset modes 4/5/6 -> (n,16,4) uint8."""
    n = len(blocks)
    cb, ab, pbit, ib, ib2, rot_off, isel_off, ep = _VEC[mode]
    # endpoints (two each), components R,G,B then A
    vals = {}
    o = ep
    for comp in "RGB":
        vals[comp + "0"] = _getbits(blocks, o, cb); o += cb
        vals[comp + "1"] = _getbits(blocks, o, cb); o += cb
    A0 = _getbits(blocks, o, ab); o += ab
    A1 = _getbits(blocks, o, ab); o += ab
    if pbit:
        P0 = _getbits(blocks, o, 1); o += 1
        P1 = _getbits(blocks, o, 1); o += 1
    cidx_off = o

    def ep_pair(c, a, p):
        if pbit:
            comp = (c << 1) | p
            return comp.astype(np.int32)          # cb+1 == 8 bits, identity expand
        return _exp(c, cb).astype(np.int32)

    e0 = np.stack([ep_pair(vals["R0"], A0, P0 if pbit else 0),
                   ep_pair(vals["G0"], A0, P0 if pbit else 0),
                   ep_pair(vals["B0"], A0, P0 if pbit else 0),
                   ((A0 << 1) | P0).astype(np.int32) if pbit else _exp(A0, ab).astype(np.int32)], 1)
    e1 = np.stack([ep_pair(vals["R1"], A1, P1 if pbit else 0),
                   ep_pair(vals["G1"], A1, P1 if pbit else 0),
                   ep_pair(vals["B1"], A1, P1 if pbit else 0),
                   ((A1 << 1) | P1).astype(np.int32) if pbit else _exp(A1, ab).astype(np.int32)], 1)
    Wc = np.array(_WT[ib], np.int32)
    Wa = np.array(_WT[ib2], np.int32) if ib2 else None

    # primary (colour) index offsets: pixel 0 has ib-1 bits, rest ib
    cidx = np.zeros((n, 16), np.int64)
    p_off = cidx_off
    for p in range(16):
        wbits = ib - 1 if p == 0 else ib
        cidx[:, p] = _getbits(blocks, p_off, wbits).astype(np.int64)
        p_off += wbits
    aidx = None
    if ib2:
        aidx = np.zeros((n, 16), np.int64)
        for p in range(16):
            wbits = ib2 - 1 if p == 0 else ib2
            aidx[:, p] = _getbits(blocks, p_off, wbits).astype(np.int64)
            p_off += wbits

    out = np.zeros((n, 16, 4), np.uint8)
    if ib2:
        isel = _getbits(blocks, isel_off, 1).astype(bool) if isel_off is not None \
            else np.zeros(n, bool)
        # when isel: colour uses aidx/Wa, alpha uses cidx/Wc
        col_w = np.where(isel[:, None], Wa[aidx], Wc[cidx]).astype(np.int32)
        alp_w = np.where(isel[:, None], Wc[cidx], Wa[aidx]).astype(np.int32)
        for ci in range(3):
            w = col_w
            out[:, :, ci] = ((e0[:, None, ci] * (64 - w) + e1[:, None, ci] * w + 32) >> 6)
        out[:, :, 3] = ((e0[:, None, 3] * (64 - alp_w) + e1[:, None, 3] * alp_w + 32) >> 6)
    else:
        w = Wc[cidx].astype(np.int32)         # (n,16)
        for ci in range(4):
            out[:, :, ci] = ((e0[:, None, ci] * (64 - w) + e1[:, None, ci] * w + 32) >> 6)

    if rot_off is not None:
        rot = _getbits(blocks, rot_off, 2)
        for k, ch in ((1, 0), (2, 1), (3, 2)):
            m = rot == k
            if m.any():
                a = out[m][:, :, 3].copy()
                sub = out[m]
                sub[:, :, 3] = sub[:, :, ch]
                sub[:, :, ch] = a
                out[m] = sub
    return out


def _bc7_vec_multi(blocks, mode):
    """Vectorised decode of multi-subset modes 0/1/2/3/7 -> (n,16,4) uint8.
    Endpoints are read globally; indices/subset assignment are grouped by the
    6-bit partition (<=64 groups), each of which has fixed anchor offsets."""
    n = len(blocks)
    NS, PB, CB, AB, IB = _NS[mode], _PB[mode], _CB[mode], _AB[mode], _IB[mode]
    epb, spb = _EPB[mode], _SPB[mode]
    o = mode + 1                               # mode prefix bits
    part = _getbits(blocks, o, PB).astype(np.int64); o += PB
    ne = 2 * NS
    R = [_getbits(blocks, o + i * CB, CB) for i in range(ne)]; o += ne * CB
    G = [_getbits(blocks, o + i * CB, CB) for i in range(ne)]; o += ne * CB
    B = [_getbits(blocks, o + i * CB, CB) for i in range(ne)]; o += ne * CB
    if AB:
        A = [_getbits(blocks, o + i * AB, AB) for i in range(ne)]; o += ne * AB
    else:
        A = None
    if epb:
        P = [_getbits(blocks, o + i, 1) for i in range(ne)]; o += ne
    elif spb:
        SP = [_getbits(blocks, o + i, 1) for i in range(NS)]; o += NS
        P = [SP[i // 2] for i in range(ne)]
    else:
        P = None
    idx_off = o
    cb = CB + (1 if P is not None else 0)
    ab = AB + (1 if (P is not None and AB) else 0)
    eps = np.zeros((ne, n, 4), np.int32)
    for i in range(ne):
        if P is not None:
            p = P[i]
            eps[i, :, 0] = _exp((R[i] << 1) | p, cb)
            eps[i, :, 1] = _exp((G[i] << 1) | p, cb)
            eps[i, :, 2] = _exp((B[i] << 1) | p, cb)
            eps[i, :, 3] = _exp((A[i] << 1) | p, ab) if AB else 255
        else:
            eps[i, :, 0] = _exp(R[i], CB)
            eps[i, :, 1] = _exp(G[i], CB)
            eps[i, :, 2] = _exp(B[i], CB)
            eps[i, :, 3] = _exp(A[i], AB) if AB else 255
    W = np.array(_WT[IB], np.int32)
    ptab = _BC7_PART[NS - 2]
    out = np.zeros((n, 16, 4), np.uint8)
    for pv in np.unique(part):
        mask = part == pv
        sb = blocks[mask]
        pat = ptab[pv]
        anchors = {p for p in range(16) if pat[p] & 0x80}
        bo = idx_off
        for p in range(16):
            wb = IB - 1 if p in anchors else IB
            idx = _getbits(sb, bo, wb).astype(np.int64)
            bo += wb
            s = pat[p] & 3
            e0 = eps[2 * s][mask]; e1 = eps[2 * s + 1][mask]
            w = W[idx][:, None]
            out[mask, p, :] = ((e0 * (64 - w) + e1 * w + 32) >> 6).astype(np.uint8)
    return out


def bc7_decode(data, w, h):
    bw, bh = (w + 3) // 4, (h + 3) // 4
    nb = bw * bh
    if len(data) < nb * 16:
        raise ValueError("BC7 data too short")
    if not _NP:
        return _bc7_decode_py(data, w, h, bw, bh)
    blocks = np.frombuffer(bytes(data[:nb * 16]), np.uint8).reshape(nb, 16)
    modes = _bc7_modes(blocks)
    px = np.zeros((nb, 16, 4), np.uint8)
    for mode in (6, 5, 4):                      # single-subset
        m = modes == mode
        if m.any():
            px[m] = _bc7_vec_single(blocks[m], mode)
    for mode in (1, 3, 7, 0, 2):                # multi-subset
        m = modes == mode
        if m.any():
            px[m] = _bc7_vec_multi(blocks[m], mode)
    # mode 8 (invalid) stays zero
    img = px.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return img[:h, :w].tobytes()


def _bc7_decode_py(data, w, h, bw, bh):
    pw, ph = bw * 4, bh * 4
    buf = bytearray(pw * ph * 4)
    for by in range(bh):
        rowbase = by * bw
        for bx in range(bw):
            px = _bc7_block(data[(rowbase + bx) * 16:(rowbase + bx) * 16 + 16])
            x0, y0 = bx * 4, by * 4
            for p in range(16):
                q = ((y0 + (p >> 2)) * pw + x0 + (p & 3)) * 4
                buf[q:q + 4] = bytes(px[p])
    return _crop(buf, pw, ph, w, h)


# --------------------------------------------------- uncompressed formats ----
def rgba8_decode(data, w, h):
    n = w * h * 4
    if len(data) < n:
        raise ValueError("RGBA8 data too short")
    return bytes(data[:n])


def bgra8_decode(data, w, h):
    n = w * h * 4
    if len(data) < n:
        raise ValueError("BGRA8 data too short")
    if _NP:
        a = np.frombuffer(bytes(data[:n]), np.uint8).reshape(-1, 4)
        return a[:, [2, 1, 0, 3]].tobytes()
    b = bytearray(data[:n])
    b[0::4], b[2::4] = b[2::4], b[0::4]
    return bytes(b)


def rgba16_decode(data, w, h):
    n = w * h * 8
    if len(data) < n:
        raise ValueError("RGBA16 data too short")
    a = np.frombuffer(bytes(data[:n]), np.uint16).reshape(-1, 4)
    return (a >> 8).astype(np.uint8).tobytes()      # unorm16 high byte -> 8-bit


def rgba16f_decode(data, w, h, tonemap=True):
    """Half-float (float16) RGBA, e.g. HDR skies. tonemap=True applies Reinhard
    (x/(1+x)) so highlights don't blow out; tonemap=False just clamps to [0,1]."""
    n = w * h * 8
    if len(data) < n:
        raise ValueError("RGBA16F data too short")
    a = np.frombuffer(bytes(data[:n]), np.float16).reshape(-1, 4).astype(np.float32)
    rgb = a[:, :3]
    rgb = rgb / (1.0 + rgb) if tonemap else np.clip(rgb, 0.0, 1.0)
    out = np.empty_like(a, np.uint8)
    out[:, :3] = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[:, 3] = np.clip(np.clip(a[:, 3], 0, 1) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out.tobytes()


def rgba32f_decode(data, w, h):
    n = w * h * 16
    if len(data) < n:
        raise ValueError("RGBA32F data too short")
    a = np.frombuffer(bytes(data[:n]), np.float32).reshape(-1, 4)
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).tobytes()


def _split_blocks(data, bw, bh):
    nb = bw * bh
    if len(data) < nb * 16:
        raise ValueError("block data too short")
    return np.frombuffer(bytes(data[:nb * 16]), np.uint8).reshape(nb, 16)


def bc2_decode(data, w, h):
    """BC2 / DXT3: 8B explicit 4-bit alpha + 8B BC1 colour (4-colour)."""
    bw, bh = (w + 3) // 4, (h + 3) // 4
    blk = _split_blocks(data, bw, bh)
    rgb = np.frombuffer(bc1_decode(blk[:, 8:16].tobytes(), w, h, force_four=True),
                        np.uint8).reshape(h, w, 4).copy()
    # explicit alpha: 16 nibbles per block (bytes 0-7), 4-bit -> 8-bit
    av = blk[:, 0:8]
    nib = np.empty((blk.shape[0], 16), np.uint8)
    nib[:, 0::2] = av & 0x0F
    nib[:, 1::2] = av >> 4
    nib = (nib << 4) | nib
    af = nib.reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4)
    rgb[:, :, 3] = af[:h, :w]
    return rgb.tobytes()


def bc3_decode(data, w, h):
    """BC3 / DXT5: 8B BC4-style alpha + 8B BC1 colour (4-colour)."""
    bw, bh = (w + 3) // 4, (h + 3) // 4
    blk = _split_blocks(data, bw, bh)
    rgb = np.frombuffer(bc1_decode(blk[:, 8:16].tobytes(), w, h, force_four=True),
                        np.uint8).reshape(h, w, 4).copy()
    af = np.frombuffer(bc4_decode(blk[:, 0:8].tobytes(), w, h),
                       np.uint8).reshape(h, w, 4)[:, :, 0]
    rgb[:, :, 3] = af
    return rgb.tobytes()


def r8_decode(data, w, h):
    """R8_UNORM: single 8-bit channel -> grey RGB, A=255."""
    n = w * h
    if len(data) < n:
        raise ValueError("R8 data too short")
    g = np.frombuffer(bytes(data[:n]), np.uint8).reshape(h, w)
    out = np.empty((h, w, 4), np.uint8)
    out[:, :, 0] = out[:, :, 1] = out[:, :, 2] = g
    out[:, :, 3] = 255
    return out.tobytes()


def r16_decode(data, w, h):
    """R16_UNORM: single 16-bit channel (high byte) -> grey RGB, A=255."""
    n = w * h * 2
    if len(data) < n:
        raise ValueError("R16 data too short")
    g = (np.frombuffer(bytes(data[:n]), np.uint16).reshape(h, w) >> 8).astype(np.uint8)
    out = np.empty((h, w, 4), np.uint8)
    out[:, :, 0] = out[:, :, 1] = out[:, :, 2] = g
    out[:, :, 3] = 255
    return out.tobytes()


def rg8_decode(data, w, h, snorm=True, reconstruct_z=True):
    """R8G8 (typically a tangent normal, often SNORM). Maps to R,G; optionally
    reconstructs blue from the unit normal."""
    n = w * h * 2
    if len(data) < n:
        raise ValueError("RG8 data too short")
    a = np.frombuffer(bytes(data[:n]), np.uint8).reshape(h, w, 2).astype(np.int16)
    if snorm:                                   # signed -128..127 stored biased
        rg = a.astype(np.float32)
        rg = np.where(rg > 127, rg - 256, rg)   # interpret as int8
        r8 = np.clip((rg[:, :, 0] + 128), 0, 255).astype(np.uint8)
        g8 = np.clip((rg[:, :, 1] + 128), 0, 255).astype(np.uint8)
    else:
        r8 = a[:, :, 0].astype(np.uint8)
        g8 = a[:, :, 1].astype(np.uint8)
    out = np.empty((h, w, 4), np.uint8)
    out[:, :, 0] = r8
    out[:, :, 1] = g8
    out[:, :, 2] = _reconstruct_z(r8, g8) if reconstruct_z else 0
    out[:, :, 3] = 255
    return out.tobytes()


# ----------------------------------------------------------------- API -------
_DECODERS = {"BC1": bc1_decode, "BC2": bc2_decode, "BC3": bc3_decode,
             "BC4": bc4_decode, "BC5": bc5_decode, "BC5N": bc5n_decode,
             "BC7": bc7_decode, "RGBA8": rgba8_decode, "BGRA8": bgra8_decode,
             "RGBA16": rgba16_decode, "RGBA16F": rgba16f_decode,
             "RGBA32F": rgba32f_decode, "R8": r8_decode, "R16": r16_decode,
             "RG8": rg8_decode}


def _decode(data, w, h, fmt):
    return _DECODERS[fmt](data, w, h)


def _coherence(rgba, w, h):
    if not _NP or w < 8 or h < 8:
        return 1.0
    g = np.frombuffer(rgba, np.uint8).reshape(h, w, 4)[:, :, :3].astype(float).mean(2)
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    cx = np.arange(dx.shape[1]) % 4 == 3
    cy = np.arange(dy.shape[0]) % 4 == 3
    bx = dx[:, cx].mean() if cx.any() else 0.0
    ix = dx[:, ~cx].mean() if (~cx).any() else 0.0
    by = dy[cy].mean() if cy.any() else 0.0
    iy = dy[~cy].mean() if (~cy).any() else 0.0
    return ((bx + by) / 2) / ((ix + iy) / 2 + 1e-6)


def decode_best(data, w, h, candidates):
    """Decode with each candidate codec name; return (rgba, fmt) for the most
    coherent (lowest 4-pixel blockiness). candidates e.g. ("BC1","BC4")."""
    best = None
    for fmt in candidates:
        try:
            rgba = _decode(data, w, h, fmt)
        except Exception:
            continue
        sc = _coherence(rgba, w, h)
        if best is None or sc < best[0]:
            best = (sc, rgba, fmt)
    if best is None:
        raise ValueError("no candidate codec fit the data")
    return best[1], best[2]
'''

_AFOP_STF_SRC = r'''
#!/usr/bin/env python3
"""afop_stf_scan.py - validate our AFOP/Snowdrop STF .dds model against real files.

For every STF texture under a root it:
  1. parses the header  (magic, dimensions, texture-class byte),
  2. infers the block layout from payload size
       (byte-class 8/16 bpp, surface count, whether mip0 is resident),
  3. and - if the bcdec decoder (afop_bcn) is importable - DECODES the largest
     resident mip with each candidate codec (BC1/BC4 for 8bpp, BC7/BC5 for 16bpp)
     and keeps the most coherent one, so the reported codec is decode-verified
     rather than guessed.

Everything we learned is encoded here:
  width  = (u16@5 - 1)//4 ,  height = u16@7//2          (content-independent)
  payload is smallest-mip-first, mip0 LAST, + a small trailer
  BC7 chain == 2 x BC1 chain  -> size alone can't separate the pairs
  8 bpp blocks: BC1 (colour) | BC4 (single channel)
  16 bpp blocks: BC7 (colour) | BC5 (two channel / normals)

Usage:
  python3 afop_stf_scan.py <root> [--limit N] [--no-decode] [--verbose]
  python3 afop_stf_scan.py one_file.dds --verbose
"""
import os
import sys
import struct

STF_MAGIC = b"STF\x02"
HDR = 76

# optional decoder (bcdec via cffi). Header/size analysis works without it.
try:
    from afop_bcn import _decode as _bcn_decode      # (data,w,h,"BC1"/"BC4"/"BC5"/"BC7")
    HAVE_DECODER = True
except Exception:
    HAVE_DECODER = False

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False

# codecs afop_bcn can currently decode (BC6H/0x49 HDR is not yet supported)
_DECODABLE = {"BC1", "BC2", "BC3", "BC4", "BC5", "BC7", "RGBA8", "BGRA8",
              "RGBA16", "RGBA16F", "RGBA32F", "R8", "R16", "RG8"}


# ---------- geometry ----------
def block_size(w, h, bpb):
    return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * bpb


def chain_size(w, h, bpb):
    t = 0
    while True:
        t += block_size(w, h, bpb)
        if w <= 1 and h <= 1:
            break
        w = max(1, w // 2)
        h = max(1, h // 2)
    return t


def parse_dims(data):
    if data[:4] != STF_MAGIC:
        raise ValueError("not STF")
    wf = struct.unpack_from("<H", data, 5)[0]
    hf = struct.unpack_from("<H", data, 7)[0]
    if (wf - 1) % 4 != 0:
        raise ValueError("bad width field")
    w, h = (wf - 1) // 4, hf // 2
    if not (0 < w <= 16384 and 0 < h <= 16384):
        raise ValueError("bad dims")
    return w, h


# ---------- layout inference (which mips/surfaces are resident) ----------
def _levels(w, h, bpb):
    """All mip levels largest-first: list of (mw, mh, byte_size) down to 1x1."""
    out = []
    while True:
        out.append((w, h, block_size(w, h, bpb)))
        if w <= 1 and h <= 1:
            break
        w = max(1, w // 2)
        h = max(1, h // 2)
    return out


# Byte-class per STF class byte, learned from a 295k-file corpus parity profile
# (afop_stf_classmap.py). 8 = BC1/BC4, 16 = BC7/BC5. Only entries proven by an
# odd-multiple share (-> 8bpp) or 0% odd + a clean BC7/BC5 decode (-> 16bpp) are
# locked here; ambiguous all-even classes (e.g. 0x47, 0x49) are left to decode-
# verify, and uncompressed classes are skipped. The class byte fixes the
# byte-class (the size-ambiguous 8-vs-16 axis); BC1-vs-BC4 / BC7-vs-BC5 stays
# content-level and is still decided by the decoder.
CLASS_BYTECLASS = {
    0x0d: 8, 0x0e: 8, 0x1c: 8, 0x1d: 8, 0x45: 8,
    0x19: 16, 0x1b: 16, 0x1e: 16, 0x1f: 16, 0x20: 16, 0x21: 16,
    0x47: 16, 0x49: 16, 0x4b: 16, 0x4c: 16,
}
CLASS_UNCOMPRESSED = {0x05, 0x0a}      # mostly fit no BC chain -> RGBA16/32F etc.

# Ground-truth class byte -> EXACT codec, from 132,255 converted-DDS headers
# (afop_stf_compare --fast). The class byte fully determines the format; the old
# blockiness colour-vs-data guess is unnecessary. 0x21 is 99.7% BC3 (2 stray BC7
# in 680). 0x49 is BC6H (HDR) - not yet decodable here.
CLASS_CODEC = {
    0x04: "BGRA8", 0x05: "RGBA32F", 0x07: "RGBA16F", 0x0a: "RGBA16",
    0x0d: "RGBA8", 0x0e: "RGBA8", 0x17: "RG8", 0x19: "R16", 0x1b: "R8",
    0x1c: "BC1", 0x1d: "BC1", 0x1e: "BC2", 0x1f: "BC2", 0x20: "BC3", 0x21: "BC3",
    0x45: "BC4", 0x47: "BC5", 0x49: "BC6H", 0x4b: "BC7", 0x4c: "BC7",
}
# codec -> ("block", bytes_per_4x4_block) | ("pixel", bytes_per_pixel)
CODEC_GEOM = {
    "BC1": ("block", 8), "BC4": ("block", 8),
    "BC2": ("block", 16), "BC3": ("block", 16), "BC5": ("block", 16),
    "BC6H": ("block", 16), "BC7": ("block", 16),
    "R8": ("pixel", 1), "RG8": ("pixel", 2), "R16": ("pixel", 2),
    "RGBA8": ("pixel", 4), "BGRA8": ("pixel", 4),
    "RGBA16": ("pixel", 8), "RGBA16F": ("pixel", 8), "RGBA32F": ("pixel", 16),
}

# preferred uncompressed format per class byte (size still arbitrates), and the
# classes that are predominantly uncompressed (try uncompressed before BC).
CLASS_UNCOMP_FMT = {0x05: "RGBA32F", 0x04: "RGBA8", 0x0e: "RGBA8",
                    0x07: "RGBA16", 0x0a: "RGBA16"}
CLASS_UNCOMP_FIRST = {0x04, 0x05, 0x07, 0x0a}
_UBPP = {"RGBA8": 4, "RGBA16": 8, "RGBA32F": 16}


def _ulevels(w, h, bpp):
    lv, ww, hh = [], w, h
    while True:
        lv.append((ww, hh, ww * hh * bpp))
        if ww <= 1 and hh <= 1:
            break
        ww, hh = max(1, ww // 2), max(1, hh // 2)
    return lv


def detect_uncompressed(payload, w, h, cls, data, do_decode):
    """Try uncompressed layouts (single mip0 or full chain, N surfaces). Format is
    decided by payload size with the class byte breaking ties. Returns a rec or None."""
    hint = CLASS_UNCOMP_FMT.get(cls)
    order = ([hint] if hint else []) + [f for f in ("RGBA8", "RGBA16", "RGBA32F")
                                        if f != hint]
    for fmt in order:
        bpp = _UBPP[fmt]
        lv = _ulevels(w, h, bpp)
        nlv = len(lv)
        for tail in (nlv - 1, 0):              # mip0-only, then full chain
            sub = lv[:nlv - tail] if tail else lv
            base = sum(s[2] for s in sub)
            if base <= 0:
                continue
            n = round(payload / base)
            if not (1 <= n <= 256 and 0 <= payload - n * base <= max(128, 16 * n)):
                continue
            mw, mh, mlen = sub[0]              # mip0, stored last in surface 0
            moff = HDR + base - mlen
            rec = dict(w=w, h=h, cls=cls, size=payload + HDR, payload=payload,
                       status="UNCOMP", codec=fmt, bpb=bpp * 8, surfaces=n,
                       mip0=True, mw=mw, mh=mh)
            if do_decode and HAVE_DECODER and len(data) >= moff + mlen:
                try:
                    _bcn_decode(data[moff:moff + mlen], mw, mh, fmt)
                except Exception:
                    continue                   # size fit but decode failed; try next
            return rec
    return None


def infer_layouts(payload, w, h, only_bpb=None):
    """Candidate layouts as dicts. A surface is a mip chain stored smallest-first;
    real files may (a) truncate the smallest `tail` mips, (b) drop mip0 (streamed),
    and (c) repeat over N array surfaces. We enumerate a bounded set and let the
    decode step arbitrate - only the true layout yields a coherent surface 0.
    If only_bpb is set (from the class byte), restrict to that byte-class so the
    size-ambiguous 8-vs-16 bpp axis is decided deterministically, not by blockiness."""
    out = []
    seen = set()
    for bpb in ((only_bpb,) if only_bpb else (8, 16)):
        lv = _levels(w, h, bpb)
        nlv = len(lv)
        for drop in (0, 1):                 # mip0 present / streamed out
            if drop and nlv < 2:
                continue
            tails = set(range(0, min(7, nlv - drop)))
            mip0_only = nlv - drop - 1          # a single mip, no chain
            tails.add(mip0_only)
            tails.add(nlv - drop - 2)           # baked: top 2 mips (mip0+mip1)
            tails.add(nlv - drop - 3)           # baked: top 3 mips
            for tail in sorted(t for t in tails if t >= 0):   # omit `tail` smallest mips
                sub = lv[drop: nlv - tail] if tail else lv[drop:]
                if not sub:
                    continue
                base = sum(s[2] for s in sub)
                if base <= 0:
                    continue
                # a truncated-to-mip0-only candidate only makes sense as a single
                # surface; allowing N>1 would let uncompressed textures (e.g. RGBA8
                # == 4x a BC7 mip0) masquerade as BC arrays and hide real gaps.
                single_only = (len(sub) == 1 and tail > 0)
                n = round(payload / base)
                for N in (n - 1, n, n + 1):
                    if N < 1 or N > 256:
                        continue
                    if single_only and N != 1:
                        continue
                    tr = payload - N * base
                    # trailer scales a little with surface count (per-surface align)
                    if not (0 <= tr <= 96 + 8 * N):
                        continue
                    mw, mh, mlen = sub[0]       # mip0: largest resident, stored last
                    moff = HDR + base - mlen    # within surface 0
                    key = (bpb, drop, tail, N)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(dict(bpb=bpb, surfaces=N, mip0=(drop == 0),
                                    tail=tail, mw=mw, mh=mh,
                                    mip_off=moff, mip_len=mlen, trailer=tr,
                                    sub=sub, base=base))
    # rank: fewer surfaces, mip0 present, fewer truncated mips, smaller trailer
    out.sort(key=lambda d: (d["surfaces"], 0 if d["mip0"] else 1,
                            d["tail"], abs(d["trailer"])))
    return out


# ---------- decode verification ----------
def coherence(rgba, w, h):
    """Blockiness: gradient at 4-pixel block boundaries vs within blocks. A wrong
    codec decodes each 4x4 block independently -> sharp 4px-periodic seams -> high
    ratio; a correct decode is continuous across them -> ~1.0. Lower == better,
    and it's robust to genuinely high-frequency content (stripes, normals)."""
    if not HAVE_NUMPY or w < 8 or h < 8:
        return 1.0
    g = np.frombuffer(rgba, np.uint8).reshape(h, w, 4)[:, :, :3].astype(float).mean(2)
    dx = np.abs(np.diff(g, axis=1)); dy = np.abs(np.diff(g, axis=0))
    cx = np.arange(dx.shape[1]) % 4 == 3      # boundary columns
    cy = np.arange(dy.shape[0]) % 4 == 3      # boundary rows
    bx = dx[:, cx].mean() if cx.any() else 0.0
    ix = dx[:, ~cx].mean() if (~cx).any() else 0.0
    by = dy[cy].mean() if cy.any() else 0.0
    iy = dy[~cy].mean() if (~cy).any() else 0.0
    return ((bx + by) / 2) / ((ix + iy) / 2 + 1e-6)


_CODECS = {8: ("BC1", "BC4"), 16: ("BC7", "BC5")}


def detect_codec(data, layout):
    """Decode a whole resident mip (the largest one <=512px, for speed without a
    flat-strip trap) with the colour and data codec of this byte-class. Prefer the
    COLOUR codec (BC1/BC7) unless the DATA codec (BC4/BC5) is clearly cleaner.
    Returns (codec, score, other_score)."""
    bpb = layout["bpb"]
    sub = layout["sub"]                        # resident levels, largest-first
    # choose verify level: largest with max-dim <= 2048 (whole mip; big enough to
    # carry content and to expose a wrong layout as garbage at its shifted offset)
    vidx = None
    for j, (lw, lh, _) in enumerate(sub):
        if max(lw, lh) <= 1024 and lw >= 4 and lh >= 4:
            vidx = j
            break
    if vidx is None:
        # nothing <=512; use the smallest resident mip that is still >=4px
        for j in range(len(sub) - 1, -1, -1):
            if sub[j][0] >= 4 and sub[j][1] >= 4:
                vidx = j
                break
    if vidx is None:
        return (None, None, None)
    vw, vh, vlen = sub[vidx]
    voff = HDR + sum(s[2] for s in sub[vidx + 1:])   # smaller mips stored before it
    mip = data[voff:voff + vlen]
    if len(mip) < vlen:
        return (None, None, None)
    colour_codec, data_codec = _CODECS[bpb]    # (BC1|BC7, BC4|BC5)
    s = {}
    for codec in (colour_codec, data_codec):
        try:
            s[codec] = coherence(_bcn_decode(mip, vw, vh, codec), vw, vh)
        except Exception:
            pass
    if not s:
        return (None, None, None)
    if colour_codec in s and data_codec in s:
        if s[data_codec] < s[colour_codec] - 0.15:
            return (data_codec, s[data_codec], s[colour_codec])
        return (colour_codec, s[colour_codec], s[data_codec])
    codec = next(iter(s))
    return (codec, s[codec], None)


# ---------- per-file ----------
def resolve(head, payload):
    """Deterministic: class byte -> exact codec (ground-truth CLASS_CODEC) ->
    largest resident mip. No blockiness. `head` is the 76-byte header, `payload`
    is filesize-HDR. Returns a dict or None when the class is unknown (caller may
    fall back to the legacy decode-verify path)."""
    if head[:4] != STF_MAGIC:
        return None
    try:
        w, h = parse_dims(head)
    except ValueError:
        return None
    cls = head[4]
    codec = CLASS_CODEC.get(cls)
    if codec is None:
        return None
    kind, unit = CODEC_GEOM[codec]
    if kind == "block":
        layouts = infer_layouts(payload, w, h, only_bpb=unit)
        full = [L for L in layouts if L["mip0"]] or layouts
        if not full:
            return None
        L = full[0]
        return dict(codec=codec, w=w, h=h, mw=L["mw"], mh=L["mh"],
                    mip_off=L["mip_off"], mip_len=L["mip_len"],
                    surfaces=L["surfaces"], mip0=L["mip0"], cls=cls)
    # uncompressed pixel format
    lv = _ulevels(w, h, unit)
    nlv = len(lv)
    for tail in (nlv - 1, 0):                  # mip0-only, then full chain
        sub = lv[:nlv - tail] if tail else lv
        base = sum(s[2] for s in sub)
        if base <= 0:
            continue
        n = round(payload / base)
        if not (1 <= n <= 256 and 0 <= payload - n * base <= max(128, 16 * n)):
            continue
        mw, mh, mlen = sub[0]
        return dict(codec=codec, w=w, h=h, mw=mw, mh=mh,
                    mip_off=HDR + base - mlen, mip_len=mlen, surfaces=n,
                    mip0=True, cls=cls)
    return None


def scan_file(path, do_decode):
    with open(path, "rb") as f:
        head = f.read(HDR)
        if head[:4] != STF_MAGIC:
            return None
        size = os.path.getsize(path)
        data = head + f.read() if (do_decode and HAVE_DECODER) else head
    try:
        w, h = parse_dims(head)
    except ValueError as e:
        return dict(status="BADHDR", note=str(e), size=size, w=0, h=0,
                    cls=head[4] if len(head) > 4 else 0)
    payload = size - HDR
    cls = head[4]

    # deterministic ground-truth path: class byte -> exact codec (no blockiness)
    det = resolve(head, payload)
    if det is not None:
        rec = dict(w=w, h=h, cls=cls, size=size, payload=payload,
                   codec=det["codec"], surfaces=det["surfaces"],
                   mip0=det["mip0"], mw=det["mw"], mh=det["mh"])
        if do_decode and HAVE_DECODER and det["codec"] in _DECODABLE \
                and len(data) >= det["mip_off"] + det["mip_len"]:
            try:
                px = _bcn_decode(data[det["mip_off"]:det["mip_off"] + det["mip_len"]],
                                 det["mw"], det["mh"], det["codec"])
                rec["status"] = "OK"
                rec["score"] = round(coherence(px, det["mw"], det["mh"]), 2)
            except Exception as e:
                rec.update(status="NODEC", note=str(e))
        else:
            rec["status"] = "DET" if det["codec"] in _DECODABLE else "NODEC"
        return rec

    if cls in CLASS_UNCOMP_FIRST:              # predominantly uncompressed classes
        u = detect_uncompressed(payload, w, h, cls, data, do_decode)
        if u:
            return u
    only_bpb = CLASS_BYTECLASS.get(cls)        # lock byte-class from the class byte
    layouts = infer_layouts(payload, w, h, only_bpb=only_bpb)
    rec = dict(w=w, h=h, cls=cls, size=size, payload=payload,
               nlayouts=len(layouts), classbpb=only_bpb)
    if not layouts:
        u = detect_uncompressed(payload, w, h, cls, data, do_decode)
        if u:                                  # e.g. 0x0e VAT stored as RGBA8
            return u
        rec.update(status="NOFIT", codec="?", bpb="?", surfaces="?", mip0="?")
        return rec
    L = layouts[0]
    rec.update(bpb=L["bpb"], surfaces=L["surfaces"], mip0=L["mip0"],
               mw=L["mw"], mh=L["mh"])
    if do_decode and HAVE_DECODER and len(data) >= L["mip_off"] + L["mip_len"]:
        # The 2x size ambiguity (8bpp 2-surface vs 16bpp 1-surface) means several
        # full-res layouts can fit. Decode-verify and, within the full-res
        # (mip0-present) group, take the genuinely lowest blockiness; only fall to
        # mip0-dropped/half-res if nothing full-res decodes coherently.
        def pick(group):
            best = None
            for cand in group:
                if len(data) < cand["mip_off"] + cand["mip_len"]:
                    continue
                codec, score, second = detect_codec(data, cand)
                if codec is None:
                    continue
                if best is None or score < best[0]:
                    best = (score, codec, second, cand)
            return best

        p_full = pick([c for c in layouts if c["mip0"]])
        p_drop = pick([c for c in layouts if not c["mip0"]])
        if p_full and p_full[0] < 1.45:
            pick_ = p_full
        elif p_drop and p_drop[0] < 1.45:
            pick_ = p_drop
        elif p_full:
            pick_ = p_full          # busy-but-correct full-res beats a blurry misread
        elif p_drop:
            pick_ = p_drop
        else:
            pick_ = None
        best = (pick_[1], pick_[0], pick_[2], pick_[3]) if pick_ else None
        if best:
            codec, score, second, cand = best
            rec.update(bpb=cand["bpb"], surfaces=cand["surfaces"], mip0=cand["mip0"],
                       mw=cand["mw"], mh=cand["mh"], codec=codec,
                       score=round(score, 2))
            # blockiness: correct decodes sit ~1.0-1.4, wrong-codec >= ~1.5
            if score < 1.45:
                rec["status"] = "OK"
            elif score < 1.6:
                rec["status"] = "OK?"      # borderline; check it
            else:
                rec["status"] = "NOISY"    # nothing decoded cleanly (array/odd layout)
        else:
            # a layout fits the payload cleanly but we couldn't decode-verify it
            # (mip too small/skinny, or no decoder) -> it still PARSES by structure
            rec.update(status="FIT", codec=_CODECS[L["bpb"]][0] + "?")
    else:
        # header/size only: report inferred class + colour-codec default
        rec.update(codec=_CODECS[L["bpb"]][0] + "?", status="INFER")
    return rec


'''


def _load_embedded():
    bcn = types.ModuleType("afop_bcn")
    bcn.__name__ = "afop_bcn"
    exec(compile(_AFOP_BCN_SRC, "afop_bcn (embedded)", "exec"), bcn.__dict__)
    sys.modules["afop_bcn"] = bcn
    stf = types.ModuleType("afop_stf_scan")
    stf.__name__ = "afop_stf_scan"
    exec(compile(_AFOP_STF_SRC, "afop_stf_scan (embedded)", "exec"), stf.__dict__)
    sys.modules["afop_stf_scan"] = stf
    return bcn, stf


_bcn, _stf = _load_embedded()

STF_MAGIC = b"STF\x02"
_STF_HEADER = _stf.HDR  # 76-byte STF front header

# Optional Pillow fallback, only for exotic standard-DDS payloads we can't name.
try:
    from PIL import Image as _PILImage
except Exception:  # pragma: no cover
    _PILImage = None


def _as_rgba(rgba_bytes, w, h):
    """afop_bcn returns w*h*4 RGBA8 bytes; reshape to a contiguous HxWx4 array."""
    arr = np.frombuffer(rgba_bytes, np.uint8)
    if arr.size != w * h * 4:
        raise RuntimeError(
            "decoder returned %d bytes, expected %d (%dx%d RGBA)"
            % (arr.size, w * h * 4, w, h)
        )
    return np.ascontiguousarray(arr.reshape(h, w, 4))


def _load_stf(data, stem, fmt):
    """Decode an STF container. Codec is taken deterministically from the class byte
    (afop_stf_scan.resolve); `fmt` optionally forces it (e.g. 'bc7')."""
    head = data[:_STF_HEADER]
    payload = len(data) - _STF_HEADER

    det = _stf.resolve(head, payload)
    if det is not None:
        codec = fmt.upper() if fmt else det["codec"]
        if codec == "BC6H":
            raise RuntimeError("BC6H (HDR, STF class 0x49) isn't supported yet.")
        if codec == "BC5" and stem.lower().endswith("_n"):
            codec = "BC5N"  # normal map: reconstruct blue
        mip = data[det["mip_off"] : det["mip_off"] + det["mip_len"]]
        return _as_rgba(
            _bcn._decode(mip, det["mw"], det["mh"], codec), det["mw"], det["mh"]
        )

    # Unknown class byte: last-resort uncompressed probe (RGBA8/16/32F by size).
    w, h = _stf.parse_dims(head)
    res = _try_uncompressed(data, payload, w, h, head[4])
    if res:
        mw, mh, _fmt, rgba = res
        return _as_rgba(rgba, mw, mh)
    raise RuntimeError("Unknown STF class byte %#04x (no codec mapping)." % head[4])


def _try_uncompressed(data, payload, w, h, cls):
    """Probe uncompressed layouts (mip0-only or full chain, N surfaces); format decided
    by payload size with the class byte breaking ties. Returns (mw, mh, fmt, rgba) | None."""
    hint = _stf.CLASS_UNCOMP_FMT.get(cls)
    order = ([hint] if hint else []) + [
        f for f in ("RGBA8", "RGBA16", "RGBA32F") if f != hint
    ]
    for fmt in order:
        bpp = _stf._UBPP[fmt]
        lv = _stf._ulevels(w, h, bpp)
        nlv = len(lv)
        for tail in (nlv - 1, 0):
            sub = lv[: nlv - tail] if tail else lv
            base = sum(s[2] for s in sub)
            if base <= 0:
                continue
            n = round(payload / base)
            if not (1 <= n <= 256 and 0 <= payload - n * base <= max(128, 16 * n)):
                continue
            mw, mh, mlen = sub[0]
            moff = _STF_HEADER + base - mlen
            if len(data) < moff + mlen:
                continue
            try:
                rgba = _bcn._decode(data[moff : moff + mlen], mw, mh, fmt)
            except Exception:
                continue
            return mw, mh, fmt, rgba
    return None


# ---- standard DDS ----------------------------------------------------------
_DXGI_BC = {
    70: "BC1",
    71: "BC1",
    72: "BC1",
    73: "BC2",
    74: "BC2",
    75: "BC2",
    76: "BC3",
    77: "BC3",
    78: "BC3",
    79: "BC4",
    80: "BC4",
    81: "BC4",
    82: "BC5",
    83: "BC5",
    84: "BC5",
    97: "BC7",
    98: "BC7",
    99: "BC7",
}
_FOURCC = {
    b"DXT1": "BC1",
    b"DXT3": "BC2",
    b"DXT5": "BC3",
    b"BC4U": "BC4",
    b"ATI1": "BC4",
    b"BC5U": "BC5",
    b"ATI2": "BC5",
}


def _load_standard_dds(data, stem, fmt):
    h, w = struct.unpack_from("<II", data, 12)
    fourcc = data[84:88]
    off = 128
    codec = _FOURCC.get(fourcc)
    if fourcc == b"DX10":
        dxgi = struct.unpack_from("<I", data, 128)[0]
        codec = _DXGI_BC.get(dxgi)
        off = 148
    codec = fmt.upper() if fmt else codec
    if codec is None:
        if _PILImage is not None:
            import io

            im = _PILImage.open(io.BytesIO(data))
            im.load()
            return np.ascontiguousarray(np.asarray(im.convert("RGBA")))
        raise RuntimeError("unrecognised standard .dds format for %s" % stem)
    if codec == "BC5" and stem.lower().endswith("_n"):
        codec = "BC5N"
    bb = 8 if codec in ("BC1", "BC4") else 16
    mw, mh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    mip0 = mw * mh * bb
    return _as_rgba(_bcn._decode(data[off : off + mip0], w, h, codec), w, h)


def load_dds(path, fmt=None):
    """Load an AFoP STF .dds (or standard .dds) -> contiguous HxWx4 uint8 RGBA.

    `fmt` optionally forces a codec ('bc1'/'bc2'/'bc3'/'bc4'/'bc5'/'bc7'); normally the
    codec is resolved deterministically from the STF class byte."""
    with open(path, "rb") as f:
        data = f.read()
    stem = os.path.splitext(os.path.basename(path))[0]
    if data[:4] == STF_MAGIC or data[:3] == b"STF":
        return _load_stf(data, stem, fmt)
    if data[:4] == b"DDS ":
        return _load_standard_dds(data, stem, fmt)
    raise RuntimeError("not an STF or DDS texture: %s" % os.path.basename(path))
