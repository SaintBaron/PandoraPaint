#!/usr/bin/env python3
"""
copy_assets.py - copy every asset Pandora Paint actually resolves out of a raw-files export into a
folder called ASSETS next to this script.

Usage:
    python3 copy_assets.py <raw_files_root>
    python3 copy_assets.py                 # tries to infer the root from your remembered asset paths

It drives the tool's OWN resolution (assets.scan_folder for the Ikran, assets.find_navi_assets for
the Na'vi - both genders - plus the gear-camo .rejuice), so it copies exactly the meshes/textures/
palettes the tool binds, and nothing else. Each file is copied under ASSETS/ preserving its path
relative to the raw-files root, so the blue/... layout is kept. On a big export the scan can take a
minute or two.
"""

import os
import sys
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "ASSETS")


def _collect(obj, out):
    """Recursively pull existing-file string values out of a nested dict/list/tuple."""
    if isinstance(obj, str):
        if obj and os.path.isfile(obj):
            out.add(os.path.abspath(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _collect(v, out)


def _infer_root(assets):
    """Best-effort raw-files root from the common ancestor of the tool's remembered asset paths."""
    remembered = set()
    _collect(assets.load_config(), remembered)
    remembered = [p for p in remembered if os.path.isfile(p)]
    if not remembered:
        return ""
    common = os.path.commonpath(remembered) if len(remembered) > 1 else os.path.dirname(remembered[0])
    return common


def main():
    try:
        import assets
    except Exception as e:  # noqa: BLE001
        print("Could not import the tool's assets.py (run this from the Pandora Paint folder):", e)
        return 1

    root = sys.argv[1] if len(sys.argv) > 1 else _infer_root(assets)
    if not root or not os.path.isdir(root):
        print("Raw-files root not found. Pass it explicitly:")
        print("    python3 copy_assets.py <path-to-raw_files>")
        return 1
    root = os.path.abspath(root)
    print("Scanning raw files under:", root)

    used = set()

    # Ikran / Banshee: best file per slot + every banshee model
    try:
        slots, models = assets.scan_folder(root)
        _collect(slots, used)
        _collect(models, used)
        print("  Ikran: %d slot files, %d models" % (len(slots), len(models)))
    except Exception as e:  # noqa: BLE001
        print("  (Ikran scan skipped:", e, ")")

    # Na'vi: resolve for both genders (meshes are gender-specific; textures overlap)
    for g in ("m", "f"):
        try:
            before = len(used)
            _collect(assets.find_navi_assets(root, g), used)
            print("  Na'vi (%s): +%d files" % (g, len(used) - before))
        except Exception as e:  # noqa: BLE001
            print("  (Na'vi %s scan skipped: %s)" % (g, e))

    # Gear-camo colour palette(s) - classify() doesn't tag .rejuice, so grab them directly
    for dp, _d, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".rejuice") and ("camo" in f.lower() or "palette" in f.lower()):
                used.add(os.path.abspath(os.path.join(dp, f)))

    if not used:
        print("No assets resolved - is that the right raw-files root?")
        return 1

    print("Copying %d asset(s) into %s ..." % (len(used), DEST))
    copied = 0
    for src in sorted(used):
        rel = os.path.relpath(src, root)
        if rel.startswith(".."):  # outside the root (other drive/tree) -> flatten to basename
            rel = os.path.basename(src)
        dst = os.path.join(DEST, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.copy2(src, dst)
            copied += 1
        except OSError as e:
            print("  skip", os.path.basename(src), "-", e)

    print("Done: copied %d file(s) into %s" % (copied, DEST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
