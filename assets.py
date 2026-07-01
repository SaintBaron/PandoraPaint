"""User-supplied game-asset management for Banshee Brush.

The tool ships no AFoP files; the user points it at their own extracted files and we remember
each path in <AppConfigLocation>/config.json (never copying them), validating on load.
classify / scan_folder decide which file fills which slot.
"""

from __future__ import annotations
import json
import logging
import os
import sys


def resource_dir():
    """Directory holding the app's bundled resources (item_wiki.json, icons/, fonts/).

    PyInstaller unpacks bundled data files to sys._MEIPASS at runtime, so prefer that when the app is
    frozen; otherwise fall back to this module's own folder (running from source). This keeps resource
    loading working identically from source and from a compiled onefile/onedir build."""
    meipass = getattr(sys, "_MEIPASS", None)
    return meipass if meipass else os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts):
    """Absolute path to a bundled resource, e.g. resource_path("item_wiki.json") or
    resource_path("icons", "pandora-paint.png")."""
    return os.path.join(resource_dir(), *parts)


IMG_EXT = (".dds", ".png", ".tga", ".jpg", ".jpeg")


def dialog_start_for(path, fallback=""):
    """Best starting location for a file dialog given a possibly-empty/invalid path:
    - an existing file        -> the file itself (Qt opens its folder with the file selected)
    - an existing directory   -> that directory
    - a missing file whose parent folder exists -> that parent folder
    - anything else (empty / nonexistent)       -> `fallback`."""
    if path:
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            return path
        d = os.path.dirname(path)
        if d and os.path.isdir(d):
            return d
    return fallback


# slot, human label, expected-filename hint, tier
SLOTS = [
    (
        "model",
        "Model mesh",
        "wildlife_banshee_*.mmb",
        "required",
    ),
    ("body_color", "Body base colour", "wildlife_banshee_*_body_d.dds", "required"),
    ("body_pattern", "Body pattern coat", "wildlife_banshee_*_body_pc.dds", "required"),
    ("head_color", "Head base colour", "wildlife_banshee_*_head_d.dds", "required"),
    ("head_pattern", "Head pattern coat", "wildlife_banshee_*_head_pc.dds", "required"),
    ("body_material", "Body material", "wildlife_banshee_*_body_m.dds", "recommended"),
    ("head_material", "Head material", "wildlife_banshee_*_head_m.dds", "recommended"),
    ("wing_color", "Wing albedo", "insect_wing_d.dds (shared)", "optional"),
    ("eye_color", "Eye albedo", "wildlife_eye_grayscale.dds (shared)", "optional"),
    ("body_normal", "Body normal", "wildlife_banshee_*_body_n.dds", "optional"),
    ("head_normal", "Head normal", "wildlife_banshee_*_head_n.dds", "optional"),
    (
        "body_dn_mask",
        "Body detail mask",
        "wildlife_banshee_*_body_dn_mask.dds",
        "optional",
    ),
    (
        "head_dn_mask",
        "Head detail mask",
        "wildlife_banshee_*_head_dn_mask.dds",
        "optional",
    ),
    ("detail1", "Detail normal 1", "skin_detail_1_nr.dds (shared)", "optional"),
    ("detail2", "Detail normal 2", "skin_detail_2_nr.dds (shared)", "optional"),
    ("detail3", "Detail normal 3", "skin_detail_4_nr.dds (shared)", "optional"),
]
SLOT_HINT = {s: h for s, _l, h, _t in SLOTS}
REQUIRED = [s for s, _l, _h, tier in SLOTS if tier == "required"]

# The tool is banshee-specific; creature-named files must match this so an export
# dump full of other wildlife (thanator, crawler, bully, ...) can't be picked up.
CREATURE = "banshee"
# When several banshee variants exist (wl_banshee_01, corpse_banshee_01, ...),
# prefer the standard one.
DEFAULT_VARIANT = "wildlife_banshee_01"

# Canonical in-game locations (paths are relative to the extracted-bundle root,
# i.e. below the extractor-specific prefix). Shown under each row so users know
# where to look in their own extraction.
_BANSHEE_DIR = "characterart/wildlife/banshee/wildlife_banshee_01"
_SHARED_DIR = "characterart/sharedtexture"
GAME_PATH = {
    "model": "characterart/wildlife/banshee/wl_banshee_01/wl_banshee_01.mmb",
    "body_color": _BANSHEE_DIR + "/wildlife_banshee_01_body_d.dds",
    "body_pattern": _BANSHEE_DIR + "/wildlife_banshee_01_body_pc.dds",
    "body_material": _BANSHEE_DIR + "/wildlife_banshee_01_body_m.dds",
    "head_color": _BANSHEE_DIR + "/wildlife_banshee_01_head_d.dds",
    "head_pattern": _BANSHEE_DIR + "/wildlife_banshee_01_head_pc.dds",
    "head_material": _BANSHEE_DIR + "/wildlife_banshee_01_head_m.dds",
    "body_normal": _BANSHEE_DIR + "/wildlife_banshee_01_body_n.dds",
    "head_normal": _BANSHEE_DIR + "/wildlife_banshee_01_head_n.dds",
    "body_dn_mask": _BANSHEE_DIR + "/wildlife_banshee_01_body_dn_mask.dds",
    "head_dn_mask": _BANSHEE_DIR + "/wildlife_banshee_01_head_dn_mask.dds",
    "wing_color": _SHARED_DIR + "/insect_wing_d.dds",
    "eye_color": _SHARED_DIR + "/wildlife_eye_grayscale.dds",
    "detail1": _SHARED_DIR + "/skin_detail_1_nr.dds",
    "detail2": _SHARED_DIR + "/skin_detail_2_nr.dds",
    "detail3": _SHARED_DIR + "/skin_detail_4_nr.dds",
}


def slot_filter(slot):
    """Qt file-dialog filter for a given slot."""
    if slot == "model":
        return "Model (*.mmb)"
    return "Texture (*.dds *.png *.tga *.jpg *.jpeg)"


# ----------------------------------------------------------------- paths/config
def _default_config_dir():
    """The OS-default config directory (XDG/APPDATA/PandoraPaint). The relocation pointer always
    lives HERE so the app can always find where the real config was moved to."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Roaming"
        )
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
            os.path.expanduser("~"), ".config"
        )
    return os.path.join(base, "PandoraPaint")


def config_dir():
    """Return the app config directory (creating it). By default this is the OS location below, but
    it can be relocated (Settings -> Config file -> Change...): a '.config_location' pointer file at
    the default location records the chosen directory, so the GUI and any helper script resolve the
    SAME config.json.
        Linux/macOS : $XDG_CONFIG_HOME/PandoraPaint  (default ~/.config/PandoraPaint)
        Windows     : %APPDATA%\\PandoraPaint
    """
    global _RESOLVED_CONFIG_DIR
    if _RESOLVED_CONFIG_DIR is None:
        default = _default_config_dir()
        os.makedirs(default, exist_ok=True)
        d = default
        try:  # follow the relocation pointer, if one is set
            ptr = os.path.join(default, ".config_location")
            if os.path.isfile(ptr):
                with open(ptr, encoding="utf-8") as _f:
                    loc = _f.read().strip()
                if loc and os.path.isdir(loc):
                    d = loc
        except OSError:
            pass
        _RESOLVED_CONFIG_DIR = d
    d = _RESOLVED_CONFIG_DIR
    if d not in _MADE_DIRS:  # makedirs is a syscall; only do it once per dir
        os.makedirs(d, exist_ok=True)
        _MADE_DIRS.add(d)
    return d


def set_config_dir(new_dir):
    """Relocate config.json to new_dir and remember it via the pointer at the default location.
    The current config is written into the new directory; future reads resolve there."""
    global _RESOLVED_CONFIG_DIR
    cfg = load_config(mutable=True)  # snapshot the current config before switching
    os.makedirs(new_dir, exist_ok=True)
    try:
        with open(os.path.join(new_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass
    default = _default_config_dir()
    os.makedirs(default, exist_ok=True)
    try:
        with open(
            os.path.join(default, ".config_location"), "w", encoding="utf-8"
        ) as f:
            f.write(new_dir)
    except OSError:
        pass
    _RESOLVED_CONFIG_DIR = new_dir
    _CFG_CACHE["key"] = None  # next load_config re-reads from the new location


_MADE_DIRS = set()
_RESOLVED_CONFIG_DIR = None


def config_path():
    return os.path.join(config_dir(), "config.json")


    # Memoize the parsed config keyed by (path, mtime, size) so the ~20 read-only load_config() calls
    # at startup don't each re-read/re-parse the (~1 MB) file. Callers that MUTATE and save back pass
    # mutable=True for an independent copy; read-only callers share the cached object. save_config()
    # invalidates.
_CFG_CACHE = {"key": None, "cfg": None}


def _read_config_file():
    try:
        with open(config_path(), "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg.setdefault("paths", {})
    cfg.setdefault("models", [])
    return cfg


def load_config(mutable=False):
    """Return the app config. Cached by file mtime so repeated reads are cheap. Pass mutable=True
    if you will modify the returned dict in place and save_config() it back - that returns a fresh,
    independent copy so the cache can't be corrupted by your edits."""
    p = config_path()
    try:
        st = os.stat(p)
        key = (p, st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and _CFG_CACHE["key"] == key:
        cfg = _CFG_CACHE["cfg"]
    else:
        cfg = _read_config_file()
        if key is not None:
            _CFG_CACHE["key"] = key
            _CFG_CACHE["cfg"] = cfg
    if mutable:
        import copy

        return copy.deepcopy(cfg)
    return cfg


def save_config(cfg):
    try:
        with open(config_path(), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        logging.getLogger("pandorapaint.assets").debug("config saved -> %s", config_path())
    except Exception:
        logging.getLogger("pandorapaint.assets").exception("failed to save config")
    _CFG_CACHE["key"] = None  # invalidate: next load_config re-reads the new file


def update_config(updater):
    """Load the config FRESH, let updater(cfg) mutate it in place, save it, and return it.

    Loading fresh on every write is what keeps two long-lived editors - the Ikran assets panel
    and the Na'vi assets panel each hold their own copy - from clobbering each other: a panel that
    saved a snapshot taken at startup would wipe whatever the other panel changed in the meantime.
    Every caller that persists a slice of the config should go through here and touch only its keys.
    """
    cfg = load_config(mutable=True)
    updater(cfg)
    save_config(cfg)
    return cfg


def get_setting(key, default=None):
    """Read a single value from the persisted ``[settings]`` config block (fresh each call)."""
    return load_config().get("settings", {}).get(key, default)


def set_setting(key, value):
    """Write a single value into the persisted ``[settings]`` config block (creating it if needed)."""
    update_config(lambda cfg: cfg.setdefault("settings", {}).__setitem__(key, value))


def preset_dir():
    """Directory where Save/Load presets (.json) are stored. Configurable via the 'preset_dir'
    key in config.json; defaults to <config_dir>/presets. Created on access."""
    cfg = load_config()
    d = cfg.get("preset_dir") or os.path.join(config_dir(), "presets")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def set_preset_dir(path):
    """Persist a new preset directory into config.json."""
    update_config(lambda cfg: cfg.__setitem__("preset_dir", path))


# ------------------------------------------------------------------ exports
# The unified export destination. Every exporter (Na'vi, Ikran, Gear Camo/Colour) writes here
# when the per-file Overwrite tickbox is off. With replicate_blue on (the default) each file's
# blue/... engine path is rebuilt inside this folder and no save dialog is shown; with it off a
# save dialog is opened instead so the file can be placed anywhere, with no folders created.
def export_folder():
    """The configured export destination folder, or '' if never set."""
    exp = load_config().get("export", {})
    return (exp.get("folder") or "") if isinstance(exp, dict) else ""


def export_replicate_blue():
    """Whether exports rebuild each file's blue/... path inside the export folder (default True)."""
    exp = load_config().get("export", {})
    return bool(exp.get("replicate_blue", True)) if isinstance(exp, dict) else True


def export_pandora_folder():
    """Whether replicated exports are wrapped in a top-level PandoraPaint/ folder (default False)."""
    exp = load_config().get("export", {})
    return bool(exp.get("pandora_folder", False)) if isinstance(exp, dict) else False


def export_configured():
    """True once an export folder has been chosen (used to gate first-launch setup)."""
    return bool(export_folder().strip())


def set_export_folder(path):
    """Persist the export destination folder into config.json."""
    logging.getLogger("pandorapaint.assets").info("export folder set -> %s", path)
    def _up(cfg):
        cfg.setdefault("export", {})["folder"] = path
    update_config(_up)


def set_export_replicate_blue(flag):
    """Persist the 'replicate blue folder structure' toggle into config.json."""
    def _up(cfg):
        cfg.setdefault("export", {})["replicate_blue"] = bool(flag)
    update_config(_up)


def set_export_pandora_folder(flag):
    """Persist the 'wrap exports in a PandoraPaint folder' toggle into config.json."""
    def _up(cfg):
        cfg.setdefault("export", {})["pandora_folder"] = bool(flag)
    update_config(_up)


# ------------------------------------------------------------------ logging
# Opt-in diagnostic logging (off by default). When enabled, every module's "pandorapaint" logger
# writes to a rotating-ish single file in the config dir; when disabled the logger is silenced so
# the calls are cheap no-ops. The toggle can be flipped live from Settings.
LOGGER_NAME = "pandorapaint"
_log_handler = None


def logging_enabled():
    """Whether diagnostic logging is on (default False)."""
    return bool(load_config().get("logging", False))


def set_logging_enabled(flag):
    """Persist the logging toggle into config.json."""
    update_config(lambda cfg: cfg.__setitem__("logging", bool(flag)))


def log_file_path():
    """Where the log is written: <log dir>/pandorapaint.log. The log dir is the config dir unless
    an explicit paths.log_dir override is set (via Settings > Diagnostics > Change)."""
    d = (load_config().get("paths", {}) or {}).get("log_dir") or config_dir()
    return os.path.join(d, "pandorapaint.log")


def get_logger(name=None):
    """The shared app logger (or a named child, e.g. get_logger('viewer'))."""
    base = logging.getLogger(LOGGER_NAME)
    return base.getChild(name) if name else base


def configure_logging(enabled=None):
    """Install or tear down the file log handler to match the toggle. Safe to call repeatedly
    (e.g. once at startup and again whenever the Settings checkbox changes)."""
    global _log_handler
    if enabled is None:
        enabled = logging_enabled()
    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())  # avoid 'no handlers' noise when off
    if _log_handler is not None:  # remove any previous file handler first
        logger.removeHandler(_log_handler)
        try:
            _log_handler.close()
        except Exception:
            pass
        _log_handler = None
    if enabled:
        try:
            os.makedirs(os.path.dirname(log_file_path()), exist_ok=True)
            h = logging.FileHandler(log_file_path(), encoding="utf-8")
            h.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                    "%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(h)
            _log_handler = h
        except Exception:
            pass  # never let logging setup break the app
        logger.setLevel(logging.DEBUG)
        logger.info("logging enabled -> %s", log_file_path())
    else:
        logger.setLevel(logging.CRITICAL + 1)  # silence info/debug/warning when off


# ----------------------------------------------------------------- matching
def classify(fname):
    """Return the asset slot a filename fills, or None. Creature slots require 'banshee'; wing/eye also accept their shared textures."""
    stem, ext = os.path.splitext(fname.lower())
    if ext == ".mmb":
        return "model" if CREATURE in stem else None
    if ext in IMG_EXT:
        if CREATURE in stem:
            # accept the plural "heads" shared-texture naming as well as "head"
            for key, toks in (("body", ("body",)), ("head", ("heads", "head"))):
                for t in toks:
                    if stem.endswith(t + "_d"):
                        return key + "_color"
                    if stem.endswith(t + "_m"):
                        return key + "_material"
                    if stem.endswith(t + "_dn_mask"):
                        return key + "_dn_mask"
                    if stem.endswith(t + "_n"):
                        return key + "_normal"
                    if t in stem and (
                        "pattern" in stem or "coat" in stem or stem.endswith(t + "_pc")
                    ):
                        return key + "_pattern"
            if stem.endswith("wing_d"):
                return "wing_color"
            if stem.endswith("eye_d") or stem.endswith("eyes_d"):
                return "eye_color"
        # shared (not creature-named) textures the banshee uses
        if stem.endswith("insect_wing_d") or stem.endswith("insect_wing"):
            return "wing_color"
        if stem.endswith("wildlife_eye_grayscale"):
            return "eye_color"
        if stem.endswith("skin_detail_1_nr"):
            return "detail1"
        if stem.endswith("skin_detail_2_nr"):
            return "detail2"
        if stem.endswith("skin_detail_4_nr"):
            return "detail3"
    return None


def _rank(slot, path):
    """Sort key for candidate files (lower preferred). For the model slot the render
    mesh (wl_banshee_*) is preferred over wildlife_banshee_* skeleton/animation files;
    for texture slots the default variant (wildlife_banshee_01) wins. .dds over .png,
    and animation/pose/corpse/lod variants last."""
    stem = os.path.splitext(os.path.basename(path).lower())[0]
    if slot == "model":
        # the actual render mesh is named wl_banshee_*; wildlife_banshee_* .mmb files
        # are skeleton/animation, so prefer the wl_ mesh (wl_banshee_01 first).
        if "wl_" + CREATURE + "_01" in stem:
            var = 0
        elif "wl_" + CREATURE in stem:
            var = 1
        elif CREATURE in stem:
            var = 2
        else:
            var = 3
    elif DEFAULT_VARIANT in stem:
        var = 0
    elif (
        CREATURE + "_01" in stem
        or "_" + CREATURE + "_01" in stem
        or "banshee_01" in stem
    ):
        var = 1
    elif CREATURE in stem:
        var = 2
    else:
        var = 3  # shared (insect_wing, wildlife_eye_grayscale)
    if any(
        b in stem
        for b in (
            "corpse",
            "lastlod",
            "_lod",
            "crashed",
            "ragdoll",
            "death",
            "idle",
            "_walk",
            "_run",
            "_fly",
            "_glide",
            "_attack",
            "_pose",
        )
    ):
        var += 5  # animation/pose/damaged variant, not the base mesh                                 # animation/pose/damaged variant, not the base mesh
    pl = path.lower()
    if slot == "model":
        typ = 0 if pl.endswith(".mmb") else 1
    elif pl.endswith(".dds"):
        typ = 0
    elif pl.endswith(".png"):
        typ = 1
    else:
        typ = 2
    return (var, typ, stem)


def scan_folder(folder, progress=None):
    """Recursively scan `folder`; return (slots, models) - best path per slot plus every banshee model found.

    `progress`, if given, is called as progress(files_seen) periodically during the walk (off the UI
    thread) so a caller can drive a progress bar; keep the callback cheap/safe."""
    slots, models = {}, []
    seen = 0
    if folder and os.path.isdir(folder):
        for dp, _d, fs in os.walk(folder):
            for f in sorted(fs):
                seen += 1
                if progress is not None and (seen & 0x3FF) == 0:
                    progress(seen)
                slot = classify(f)
                if slot is None:
                    continue
                p = os.path.join(dp, f)
                if slot == "model":
                    models.append(p)
                cur = slots.get(slot)
                if cur is None or _rank(slot, p) < _rank(slot, cur):
                    slots[slot] = p
    if progress is not None:
        progress(seen)
    return slots, models


# ----------------------------------------------------------------- validation
def missing_required(pathmap):
    """Required slots that are absent or point at a file that no longer exists."""
    return [s for s in REQUIRED if not (pathmap.get(s) and os.path.isfile(pathmap[s]))]


def invalid_paths(pathmap):
    """Tracked slots whose stored path no longer exists (unknown/legacy slots are ignored)."""
    known = {s for s, _l, _h, _t in SLOTS}
    return [s for s, p in pathmap.items() if s in known and p and not os.path.isfile(p)]


def afop_blue_root(path):
    """If `path` is inside an AFOP 'blue/...' tree, return the directory containing 'blue' (so 'blue/...' engine paths resolve under it), else None."""
    parts = os.path.abspath(path).replace("\\", "/").split("/")
    idx = None
    for i, seg in enumerate(parts[:-1]):  # skip the filename itself
        if seg.lower() == "blue":
            idx = i  # take the last 'blue' if nested
    if idx is None:
        return None
    root = "/".join(parts[:idx])
    return root or "/"


def find_related(manifest_path, engine_path):
    """Resolve a file referenced by a .mbansheepatterndata, searching in strict order and never
    outside these scopes: the manifest's folder, its sub-folders, then (if it sits in a
    'blue/...' tree) the engine path under the blue root. Returns an absolute path or None."""
    base = os.path.dirname(os.path.abspath(manifest_path))
    rel = engine_path.replace("\\", "/").lstrip("/")
    name = os.path.basename(rel)
    lname = name.lower()
    # 1) the manifest's own folder
    cand = os.path.join(base, name)
    if os.path.isfile(cand):
        return cand
    # 2) sub-folders of the manifest's own folder
    for dp, _sub, files in os.walk(base):
        if dp == base:
            continue  # own folder already checked in (1)
        for f in files:
            if f.lower() == lname:
                return os.path.join(dp, f)
    # 3) relative to the AFOP 'blue' root, using the engine path verbatim
    root = afop_blue_root(manifest_path)
    if root is not None:
        cand = os.path.join(root, *rel.split("/"))
        if os.path.isfile(cand):
            return cand
    return None


# =====================================================================================
# Na'vi (player) asset resolution - merged in from the former navi_assets.py.
# Banshee asset config/scan/classify lives above; this section resolves the Na'vi preview
# meshes + textures by filename.
# Public API: find_navi_assets(), navi_summarize().
# =====================================================================================

NAVI_MESH_EXT = ".mmb"
NAVI_TEX_EXT = (".dds", ".png", ".tga")

# mesh stems per part and gender (lowercase, no extension). The viewer loads one per part.
_MESH_STEMS = {
    "head": {"m": ("p_head_01_m",), "f": ("p_head_01_f",)},
    # player body: prefer the player p_body_01_* mesh, fall back to the NPC rnf base body
    "body": {
        "m": ("p_body_01_m", "rnf_body_01_m"),
        "f": ("p_body_01_f", "rnf_body_01_f"),
    },
    # hair is a shared/style mesh - prefer the rnf default, then a vlt style LOD or shared kuru.
    # (the strands .mmb bundles its own "Accessories" submesh, so one file is enough.)
    "hair": {
        "m": ("p_rnf_hair_01", "p_vlt_hair_07_lq", "kuru_lowpoly_01"),
        "f": ("p_rnf_hair_01", "p_vlt_hair_07_lq", "kuru_lowpoly_01"),
    },
    # the kuru (neural queue/braid) is its own mesh + maps
    "kuru": {"m": ("p_rnf_kuru_01",), "f": ("p_rnf_kuru_01",)},
}


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0].lower()


def _is_lod_variant(stem):
    return any(
        b in stem for b in ("_lod", "lastlod", "_lq", "_mq", "ragdoll", "corpse")
    )


def _mesh_rank(part, gender, path):
    """Lower = preferred. Exact gender stem first, then declared order, lod variants last."""
    stem = _stem(path)
    wanted = _MESH_STEMS[part][gender]
    var = next((i for i, w in enumerate(wanted) if stem == w), None)
    if var is None:
        var = next((10 + i for i, w in enumerate(wanted) if stem.startswith(w)), 99)
    lod = 5 if (_is_lod_variant(stem) and "_lq" not in wanted[0]) else 0
    return (var + lod, stem)


def _find_mesh(folder, part, gender):
    wanted = _MESH_STEMS[part][gender]
    best = None
    for dp, _d, fs in os.walk(folder):
        for f in fs:
            if not f.lower().endswith(NAVI_MESH_EXT):
                continue
            stem = os.path.splitext(f.lower())[0]
            if not any(stem == w or stem.startswith(w) for w in wanted):
                continue
            p = os.path.join(dp, f)
            if best is None or _mesh_rank(part, gender, p) < _mesh_rank(
                part, gender, best
            ):
                best = p
    return best


# ---- textures: (bucket, role) decided from the filename suffix -----------------------------
def _classify_texture(stem):
    """Return (bucket, role) for a texture stem, or None. bucket in head/body/eye/hair."""
    s = stem.lower()
    # --- (#5) customization-library + cross-part maps, resolved best-effort by keyword so they
    #     bind if present in the export. Pattern/paint/bio are normally runtime-selected (the
    #     player picks them; an item's myTextureData can also point at them); haircap is written
    #     by the hair graph and READ by the head/face shader. These keyword rules are heuristic -
    #     refine once real library filenames are confirmed. Checked first so they win over the
    #     suffix rules below. ---
    _is_head = s.startswith(("p_head_01", "p_rnf_head", "rnf_head_01", "p_face"))
    _skin_bucket = "head" if _is_head else "body"
        # Head hair-cap (scalp tint on the head's UV1): only a PLAYER-prefixed cap (p_rnf*/p_navi*) that
        # pairs to this hair (p_rnf_hair_01) is correct - OR the hairstyle's own p_rnf_*/p_navi_* "*_mask"
        # (e.g. p_rnf_hair_01_mask), which IS this character's scalp cap despite the "_mask" name. Human
        # caps (hmn_*) and other hairstyles' NPC caps (rnf_haircap_01_f_d) must NOT bind (they land on the
        # face). No correct cap for p_rnf_hair_01 exists in the export, so nothing binds - correct, and a
        # real p_ cap drops straight in later. The strand mask p_rnf_hair_01_m ends "_m" not "_mask", so
        # it's not caught. Cap colour stays its own ~0.5 grey (tinting from the hair root patches the face).
    if (
        (
            ("haircap" in s or "hair_cap" in s or "scalp_cap" in s)
            or (s.endswith("_mask") and "hair" in s)
        )
        and not s.startswith(("hmn_", "human"))
        and "human" not in s
        and s.startswith(("p_rnf", "p_navi"))
    ):
        return ("head", "haircap")
    if "biolum" in s or "bioluminescen" in s:
        return (_skin_bucket, "bio")
    if "warpaint" in s or s.startswith(("p_warpaint", "warpaint", "item_warpaint")):
        return (_skin_bucket, "paint")
    if "pattern" in s and (
        "skin" in s
        or _is_head
        or s.startswith(
            (
                "p_body",
                "rnf_body",
                "p_customization",
                "p_rnf_body",
                "p_rnf_arms",
                "p_rnf_head",
                "rnf_arms",
            )
        )
    ):
        # arms patterns belong to the BODY mesh (the arms are part of the body skin); head pattern
        # to the head. _is_head already routed p_rnf_head_*; force the body bucket for body/arms.
        if s.startswith(("p_rnf_body", "rnf_body", "p_body", "p_rnf_arms", "rnf_arms")):
            return ("body", "pattern")
        return (_skin_bucket, "pattern")

    body_head = s.startswith(
        ("p_body_01", "rnf_body_01", "p_head_01", "p_rnf_head", "rnf_head_01")
    )
    # eyelash card has its own alpha diffuse (p_eyelashes_*_d) - route it to its own role so the
    # lash submesh can sample it (with alpha) instead of the opaque head skin.
    if "eyelash" in s and s.endswith("_d"):
        return ("eye", "lash")
    # eyes first (iris diffuse drives the eye recolour) - but a body/head-prefixed map that merely
    # contains "eyes" (e.g. p_body_01_m_eyes_n) belongs to that skin set, not the eye bucket.
    if (
        not body_head
        and "eye" in s
        and not any(k in s for k in ("eyelash", "eyeshadow", "eyeshell", "eyebrow"))
    ):
        if s.endswith("_d"):
            return ("eye", "iris")
        if s.endswith("_n"):
            return ("eye", "normal")
        if s.endswith("_h"):  # iris height / parallax (p_eyes_01_h)
            return ("eye", "height")
        return None
    # the kuru (neural queue/braid) has its own maps in a dedicated bucket
    if s.startswith("p_rnf_kuru") or s.startswith("kuru"):
        if s.endswith("_dir"):
            return ("kuru", "dir")
        if s.endswith("_ao"):
            return ("kuru", "ao")
        if s.endswith("_m"):
            return ("kuru", "mask")
        return None
    # hair maps (no albedo for strands - mask/dir/ao only)
    if (
        ("hair" in s or s.startswith(("p_vlt_hair", "p_rnf_hair")))
        and not s.startswith(("hmn_", "human"))
        and "hair_cap" not in s
        and "haircap" not in s
    ):
        if s.endswith("_dir"):
            return ("hair", "dir")
        if s.endswith("_ao"):
            return ("hair", "ao")
        if s.endswith("_m"):
            return ("hair", "mask")
        return None
    # body skin
    if s.startswith(("p_body_01", "rnf_body_01")):
        if "detail" in s and s.endswith("_n"):
            return ("body", "detail")
        if s.endswith("_d"):
            return ("body", "color")
        if s.endswith("_m"):
            return ("body", "material")
        if s.endswith("_n"):
            return ("body", "normal")
        return None
    # generic detail-normal library (dn_*_n) - shared detail normal usable by either skin bucket;
    # classify to head 'detail' by default (the head graph binds dn_plastic_04_n); the body has its
    # own p_body_*_detail_n which wins for the body bucket.
    if s.startswith("dn_") and s.endswith("_n"):
        return ("head", "detail")
    # ch_dn_skin_* is the shared skin detail normal the rnf BODY graph binds (ch_dn_skin_01_n).
    if s.startswith("ch_dn_skin") and s.endswith("_n"):
        return ("body", "detail")
    # head/face skin (player p_head_01_*; female player head reuses npc rnf_head_01_f)
    if s.startswith(("p_head_01", "p_rnf_head", "rnf_head_01", "p_face")):
        if "detail" in s and s.endswith("_n"):
            return ("head", "detail")
        if s.endswith("_d"):
            return ("head", "color")
        if s.endswith("_m"):
            return ("head", "material")
        if s.endswith("_n"):
            return ("head", "normal")
        return None
    return None


def _tex_rank(gender, stem):
    """Prefer the active gender, then non-detail base maps.

    Gender lives as a `_m_`/`_f_` token before the role suffix (e.g. p_body_01_m_d);
    the trailing role `_m` (material) has no following underscore, so it never collides.
    """
    s = stem.lower()
    other = "_f_" if gender == "m" else "_m_"
    wrong = 1 if other in s else 0
    # Iris textures: the AUTHORITATIVE binding (p_head_01_f.mgraphobject -> PX_Eye2) is a deliberate
    # MIXED set - the diffuse is p_eyes_02_d, paired with p_eyes_01_n (normal) and p_eyes_01_h
    # (height). It is NOT a same-numbered set. So for the iris DIFFUSE (_d) prefer eyes_02; for the
    # normal/height prefer eyes_01. (Earlier we preferred eyes_01 for the diffuse on the theory the
    # diffuse and height had to share a number - the material graph disproves that.)
    if s.endswith("_d") and "eyes_0" in s:
        eye_pref = 0 if "eyes_02" in s else 1  # diffuse: eyes_02 wins
    elif ("eyes_0" in s) and (s.endswith("_n") or s.endswith("_h")):
        eye_pref = 0 if "eyes_01" in s else 1  # normal/height: eyes_01 wins
    else:
        eye_pref = 0
    detail = 1 if "detail" in s else 0
    # the rnf BODY graph authoritatively binds ch_dn_skin_01_n as the body detail normal, so prefer
    # it over a p_body_*_detail_n that might also be present. 0 = ch_dn_skin (preferred detail).
    dn_pref = 0 if s.startswith("ch_dn_skin") else 1
    # the rnf HEAD graph binds dn_plastic_04_n specifically; prefer it over other dn_* library
    # normals (e.g. dn_bark) that would otherwise win on the shorter-stem tiebreaker.
    dn_head_pref = 0 if s.startswith("dn_plastic") else 1
    # body pattern: prefer the main body coverage (p_rnf_body_pattern) over the arms pattern.
    pat_pref = 0 if "body_pattern" in s else (1 if "arms_pattern" in s else 0)
    # Head/body SKIN textures: prefer the NPC rnf_head_01_*/rnf_body_01_* maps over the player
    # p_head_01_*/p_body_01_* ones - the rnf set is the correct skin albedo/material/normal for the
    # preview. Only applies to the skin families (rnf_head/rnf_body vs p_head/p_body); eye/hair are
    # untouched. 0 = rnf (preferred), 1 = player p_ skin, neutral otherwise.
    if s.startswith(("rnf_head_01", "rnf_body_01")):
        rnf_pref = 0
    elif s.startswith(("p_head_01", "p_body_01")):
        rnf_pref = 1
    else:
        rnf_pref = 0
    # shorter stem = more canonical (p_body_01_m_n beats p_body_01_m_eyes_n for the body normal)
    return (
        wrong,
        eye_pref,
        rnf_pref,
        dn_pref,
        dn_head_pref,
        pat_pref,
        detail,
        len(s),
        s,
    )


def find_navi_assets(folder, gender="m"):
    """Resolve preview meshes + textures under `folder`. `gender` in {'m','f'}.

    Returns {"meshes": {part: path|None}, "textures": {bucket: {role: path}}}.
    Missing entries are simply absent (head/body/hair are independent; any subset is fine).
    """
    gender = "f" if str(gender).lower().startswith("f") else "m"
    out = {
        "meshes": {},
        "textures": {"head": {}, "body": {}, "eye": {}, "hair": {}, "kuru": {}},
    }
    if not folder or not os.path.isdir(folder):
        out["meshes"] = {"head": None, "body": None, "hair": None, "kuru": None}
        return out

    for part in ("head", "body", "hair", "kuru"):
        out["meshes"][part] = _find_mesh(folder, part, gender)

    best_paths = {}  # (bucket, role) -> path (kept if better-ranked)
    for dp, _d, fs in os.walk(folder):
        for f in fs:
            ext = os.path.splitext(f.lower())[1]
            if ext not in NAVI_TEX_EXT:
                continue
            stem = os.path.splitext(f.lower())[0]
            br = _classify_texture(stem)
            if br is None:
                continue
            bucket, role = br
            # HEAD skin maps: the male rnf head reuses the FEMALE rnf head textures and a male rnf
            # head set may be absent, so the head's d/m/n must be rnf_head_01 of EITHER gender and
            # never the player p_head_* textures (which would otherwise win on gender match).
            if bucket == "head" and role in ("color", "material", "normal"):
                if not stem.startswith("rnf_head_01"):
                    continue
            p = os.path.join(dp, f)
            cur = best_paths.get(br)
            if cur is None or _head_aware_rank(
                gender, bucket, role, stem
            ) < _head_aware_rank(gender, bucket, role, _stem(cur)):
                best_paths[br] = p
    for (bucket, role), p in best_paths.items():
        out["textures"][bucket][role] = p
    return out


def _head_aware_rank(gender, bucket, role, stem):
    """_tex_rank, but for head skin maps the female rnf head texture is NOT penalised as
    wrong-gender when resolving the male head (the male head legitimately reuses female maps)."""
    if (
        bucket == "head"
        and role in ("color", "material", "normal")
        and stem.startswith("rnf_head_01")
    ):
        # rank rnf head textures of either gender, preferring the requested gender only as a
        # tiebreaker (so a complete other-gender set still wins over nothing).
        sg = _stem_gender(stem)
        gender_miss = 1 if (sg is not None and sg != gender) else 0
        return (gender_miss,) + tuple(_tex_rank(gender, stem)[1:])
    return (0,) + tuple(_tex_rank(gender, stem))


def navi_summarize(found):
    """One-line-per-entry human summary (for the CLI / status messages)."""
    lines = ["meshes:"]
    for part, p in found.get("meshes", {}).items():
        lines.append("  %-5s %s" % (part, os.path.basename(p) if p else "(none)"))
    lines.append("textures:")
    for bucket, roles in found.get("textures", {}).items():
        for role, p in roles.items():
            lines.append("  %-5s %-9s %s" % (bucket, role, os.path.basename(p)))
    return "\n".join(lines)


# =====================================================================================
# Na'vi asset SLOTS for the Settings panel (mirrors the Banshee SLOTS / AssetsPanel model).
# Per-gender mesh slots give "options for both genders"; a part is satisfied if EITHER gender
# is populated (one-of-the-pair). Textures are single slots resolved for the active gender.
# Fields: (slot, label, tier, kind, a, b)  - mesh: a=part b=gender|None ; tex: a=bucket b=role
# =====================================================================================
NAVI_SLOTS = [
    # (slot, label, tier, kind, a, b, g)
    #   mesh: a=part,  b=None, g=gender|None(hair-shared)
    #   tex : a=bucket, b=role, g=gender|None(eye/hair-shared)
    ("nav_head_m", "Head mesh - Male", "required", "mesh", "head", None, "m"),
    ("nav_head_f", "Head mesh - Female", "required", "mesh", "head", None, "f"),
    ("nav_body_m", "Body mesh - Male", "required", "mesh", "body", None, "m"),
    ("nav_body_f", "Body mesh - Female", "required", "mesh", "body", None, "f"),
    ("nav_hair", "Hair mesh", "required", "mesh", "hair", None, None),
    ("nav_kuru", "Kuru mesh", "required", "mesh", "kuru", None, None),
    ("nav_kuru_m", "Kuru mask", "required", "tex", "kuru", "mask", ""),
    ("nav_kuru_ao", "Kuru AO", "required", "tex", "kuru", "ao", ""),
    ("nav_kuru_dir", "Kuru direction", "required", "tex", "kuru", "dir", ""),
    ("nav_kuru_acc", "Kuru Decor (_d)", "required", "tex", "kuru", "color", ""),
    ("nav_kuru_acc_m", "Kuru Decor (_m)", "optional", "tex", "kuru", "material", ""),
    ("nav_kuru_acc_n", "Kuru Decor (_n)", "optional", "tex", "kuru", "normal", ""),
    ("nav_body_d_m", "Body albedo - Male", "required", "tex", "body", "color", "m"),
    ("nav_body_d_f", "Body albedo - Female", "required", "tex", "body", "color", "f"),
    (
        "nav_body_mt_m",
        "Body material - Male",
        "required",
        "tex",
        "body",
        "material",
        "m",
    ),
    (
        "nav_body_mt_f",
        "Body material - Female",
        "required",
        "tex",
        "body",
        "material",
        "f",
    ),
    ("nav_head_d_m", "Head albedo - Male", "required", "tex", "head", "color", "m"),
    ("nav_head_d_f", "Head albedo - Female", "required", "tex", "head", "color", "f"),
    (
        "nav_head_mt_m",
        "Head material - Male",
        "required",
        "tex",
        "head",
        "material",
        "m",
    ),
    (
        "nav_head_mt_f",
        "Head material - Female",
        "required",
        "tex",
        "head",
        "material",
        "f",
    ),
    ("nav_eye_d", "Eye iris", "required", "tex", "eye", "iris", None),
    ("nav_hair_m", "Hair mask", "required", "tex", "hair", "mask", None),
    ("nav_body_n_m", "Body normal - Male", "required", "tex", "body", "normal", "m"),
    ("nav_body_n_f", "Body normal - Female", "required", "tex", "body", "normal", "f"),
    ("nav_head_n_m", "Head normal - Male", "required", "tex", "head", "normal", "m"),
    ("nav_head_n_f", "Head normal - Female", "required", "tex", "head", "normal", "f"),
    (
        "nav_body_dn_m",
        "Body detail normal - Male",
        "required",
        "tex",
        "body",
        "detail",
        "m",
    ),
    (
        "nav_body_dn_f",
        "Body detail normal - Female",
        "required",
        "tex",
        "body",
        "detail",
        "f",
    ),
    (
        "nav_head_dn_m",
        "Head detail normal - Male",
        "required",
        "tex",
        "head",
        "detail",
        "m",
    ),
    (
        "nav_head_dn_f",
        "Head detail normal - Female",
        "required",
        "tex",
        "head",
        "detail",
        "f",
    ),
    ("nav_head_pat", "Head pattern coverage", "required", "tex", "head", "pattern", ""),
    ("nav_body_pat", "Body pattern coverage", "required", "tex", "body", "pattern", ""),
    ("nav_head_cap", "Hair cap (_mask)", "required", "tex", "head", "haircap", ""),
    ("nav_eye_n", "Eye normal", "required", "tex", "eye", "normal", None),
    ("nav_eye_h", "Eye height", "required", "tex", "eye", "height", None),
    ("nav_lash", "Eyelash", "required", "tex", "eye", "lash", None),
    ("nav_hair_ao", "Hair AO", "required", "tex", "hair", "ao", None),
    ("nav_hair_dir", "Hair direction", "required", "tex", "hair", "dir", None),
    ("nav_hair_acc", "Hair Decor (_d)", "required", "tex", "accessory", "color", ""),
    ("nav_hair_acc_m", "Hair Decor (_m)", "optional", "tex", "accessory", "material", ""),
    ("nav_hair_acc_n", "Hair Decor (_n)", "optional", "tex", "accessory", "normal", ""),
]

# Edit-Na'vi-only slots: these exist so a user CAN pick a file for them in the Edit Na'vi texture
# pickers, but they are deliberately NOT shown in the Settings > Na'vi Assets panel and are NEVER
# auto-resolved by the folder scan - so the decor _m/_n maps are opt-in extras, never auto-added and
# never given a default. (The decor _d diffuse stays a normal, auto-resolved, Settings-listed slot.)
NAVI_EDIT_ONLY_SLOTS = {
    "nav_hair_acc_m",
    "nav_hair_acc_n",
    "nav_kuru_acc_m",
    "nav_kuru_acc_n",
}
_NAVI_SLOT_META = {
    s: {"label": lbl, "tier": t, "kind": k, "a": a, "b": b, "g": g}
    for (s, lbl, t, k, a, b, g) in NAVI_SLOTS
}

_NV_CHAR = "blue/baked/characterart"
_NV_NPC = _NV_CHAR + "/npc/rnf/default"
_NV_DETAIL_N = "blue/baked/art/[characters]/[assets]/[detailnormals]/dn_plastic_n.dds"
NAVI_GAME_PATH = {
    "nav_head_m": _NV_CHAR + "/player/head/male/p_head_01_m/p_head_01_m.mmb",
    "nav_head_f": _NV_CHAR + "/player/head/female/p_head_01_f/p_head_01_f.mmb",
    "nav_body_m": _NV_CHAR + "/player/body/male/p_body_01_m/p_body_01_m.mmb",
    "nav_body_f": _NV_CHAR + "/player/body/female/p_body_01_f/p_body_01_f.mmb",
    "nav_hair": _NV_CHAR + "/player/hair/p_rnf_hair_01/p_rnf_hair_01.mmb",
    "nav_kuru": _NV_CHAR + "/player/hair/p_rnf_kuru_01/p_rnf_kuru_01.mmb",
    # skin textures: the correct head/body maps are the NPC rnf set (not the player p_ textures)
    "nav_body_d_m": _NV_NPC + "/body/male/rnf_body_01_m/textures/rnf_body_01_m_d.dds",
    "nav_body_d_f": _NV_NPC + "/body/female/rnf_body_01_f/textures/rnf_body_01_f_d.dds",
    "nav_body_mt_m": _NV_NPC + "/body/male/rnf_body_01_m/textures/rnf_body_01_m_m.dds",
    "nav_body_mt_f": _NV_NPC
    + "/body/female/rnf_body_01_f/textures/rnf_body_01_f_m.dds",
    "nav_head_d_m": _NV_NPC + "/head/male/rnf_head_01_m/textures/rnf_head_01_m_d.dds",
    "nav_head_d_f": _NV_NPC + "/head/female/rnf_head_01_f/textures/rnf_head_01_f_d.dds",
    "nav_head_mt_m": _NV_NPC + "/head/male/rnf_head_01_m/textures/rnf_head_01_m_m.dds",
    "nav_head_mt_f": _NV_NPC
    + "/head/female/rnf_head_01_f/textures/rnf_head_01_f_m.dds",
    "nav_eye_d": _NV_CHAR + "/player/head/shared/eyes/p_eyes_01_d.dds",
    "nav_hair_m": _NV_CHAR + "/player/hair/p_rnf_hair_01/p_rnf_hair_01_m.dds",
    "nav_body_n_m": _NV_NPC + "/body/male/rnf_body_01_m/textures/rnf_body_01_m_n.dds",
    "nav_body_n_f": _NV_NPC + "/body/female/rnf_body_01_f/textures/rnf_body_01_f_n.dds",
    "nav_head_n_m": _NV_NPC + "/head/male/rnf_head_01_m/textures/rnf_head_01_m_n.dds",
    "nav_head_n_f": _NV_NPC + "/head/female/rnf_head_01_f/textures/rnf_head_01_f_n.dds",
    "nav_eye_n": _NV_CHAR + "/player/head/shared/eyes/p_eyes_01_n.dds",
    "nav_eye_h": _NV_CHAR + "/player/head/shared/eyes/p_eyes_01_h.dds",
    "nav_lash": _NV_CHAR + "/player/head/shared/eyes/p_eyelashes_d.dds",
    "nav_hair_ao": _NV_CHAR + "/player/hair/p_rnf_hair_01/p_rnf_hair_01_ao.dds",
    "nav_hair_dir": _NV_CHAR + "/player/hair/p_rnf_hair_01/p_rnf_hair_01_dir.dds",
    # kuru textures (share the player hair tree)
    "nav_kuru_m": _NV_CHAR + "/player/hair/p_rnf_kuru_01/p_rnf_kuru_01_m.dds",
    "nav_kuru_ao": _NV_CHAR + "/player/hair/p_rnf_kuru_01/p_rnf_kuru_01_ao.dds",
    # kuru direction map: defaults to the rnf hair-01 strand-direction map
    "nav_kuru_dir": _NV_CHAR + "/player/hair/p_rnf_hair_01/p_rnf_hair_01_dir.dds",
    # hair / kuru decor (hairband) diffuse: the shared hairbands atlas
    "nav_hair_acc": _NV_CHAR + "/player/hair/shared/p_hairbands_01_d.dds",
    "nav_kuru_acc": _NV_CHAR + "/player/hair/shared/p_hairbands_01_d.dds",
    # skin pattern coverage masks (shared per body region)
    "nav_head_pat": _NV_CHAR + "/player/head/shared/pattern/p_rnf_head_pattern_01.dds",
    "nav_body_pat": _NV_CHAR + "/player/body/shared/pattern/p_rnf_body_pattern_01.dds",
    # hair cap (scalp tint on the head, sampled on uv1) - the cap item's myHeadTexture
    "nav_head_cap": _NV_CHAR + "/player/hair/p_rnf_hair_01/p_rnf_hair_01_mask.dds",
    # detail normal: a shared tech texture the skin shaders bind for both genders
    "nav_head_dn_m": _NV_DETAIL_N,
    "nav_head_dn_f": _NV_DETAIL_N,
    "nav_body_dn_m": _NV_DETAIL_N,
    "nav_body_dn_f": _NV_DETAIL_N,
}


def navi_game_path(slot, gender=None):
    """Expected in-game path for a slot. Gender is already baked into gendered slots, so the
    optional `gender` arg is ignored (kept for call-site compatibility)."""
    return NAVI_GAME_PATH.get(slot, "")


def _stem_gender(stem):
    """Gender token in a texture stem: 'm'/'f' from the `_m_`/`_f_` infix, else None. The trailing
    role suffix (material '_m', mask '_m') is NOT a gender token (no following underscore)."""
    s = stem.lower()
    if "_m_" in s:
        return "m"
    if "_f_" in s:
        return "f"
    return None


def _nav_path_pref(path):
    """Rank a texture by its location: 0 = the player asset tree (what we want) OR the NPC rnf
    default skin set (the correct head/body skin maps), 2 = clearly some other character's textures
    (other npc / fauna / wildlife / prototype / shared eye atlases), 1 = other. A full-game export
    contains thousands of 'eye'/'hair' textures, so without this the matchers happily grab an NPC's
    eyes; this keeps us inside .../player/... plus the one whitelisted rnf skin path."""
    pl = path.lower().replace("\\", "/")
    if "/player/" in pl:
        return 0
    # the NPC rnf default skin set is the CORRECT head/body skin - rank it as preferred too
    if "/npc/rnf/default/" in pl and ("rnf_head_01" in pl or "rnf_body_01" in pl):
        return 0
    # the generic detail-normal library (dn_*_n) is a shared tech texture the skin shaders bind
    if "/detailnormals/" in pl or "/dn_" in pl or "ch_dn_skin" in pl:
        return 0
    bad = (
        "/npc/",
        "/fauna/",
        "/wildlife/",
        "/prototype/",
        "/sharedtexture/",
        "/basehead/",
        "/character_art/",
        "/[npc]/",
        "/[fauna]/",
        "/effects/",
        "/gear/",
        "/vfxtextures",
    )
    if any(k in pl for k in bad):
        return 2
    return 1


def navi_slot_filter(slot):
    """Qt file-dialog filter for a slot (meshes vs textures)."""
    if _NAVI_SLOT_META.get(slot, {}).get("kind") == "mesh":
        return "Na'vi mesh (*.mmb);;All files (*)"
    return "Texture (*.dds *.png *.tga);;All files (*)"


def scan_navi_folder(folder, progress=None):
    """Walk `folder` ONCE and resolve every Na'vi slot (both genders for head/body, shared for
    eye/hair) + collect colour-item paths. Returns {"folder", "slots": {slot: path}, "colors": []}.

    This is the ONLY function that walks the tree. The panel/viewer/launch path read the cached
    result (cfg["navi"]["cache"]) so they never re-scan a (potentially enormous) export folder.

    `progress`, if given, is called as progress(files_seen) periodically during the walk so a caller
    can drive a progress bar; it is invoked off the UI thread, so the callback must be cheap/safe.
    """
    out = {"folder": folder or "", "slots": {}, "colors": [], "tex_by_basename": {}}
    if not folder or not os.path.isdir(folder):
        return out
    meshes, texes = [], []  # (stem, path)
    by_base = {}  # basename (no ext, lower) -> [path, ...] for item texture_targets
    seen = 0
    for dp, _d, fs in os.walk(folder):
        for f in fs:
            seen += 1
            if progress is not None and (seen & 0x3FF) == 0:  # every 1024 files
                progress(seen)
            low = f.lower()
            ext = os.path.splitext(low)[1]
            p = os.path.join(dp, f)
            if ext == NAVI_MESH_EXT:
                meshes.append((os.path.splitext(low)[0], p))
            elif ext in NAVI_TEX_EXT:
                texes.append((os.path.splitext(low)[0], p))
                by_base.setdefault(os.path.splitext(low)[0], []).append(p)
            elif ext == ".blueitemtype":
                out["colors"].append(p)
    if progress is not None:
        progress(seen)
    # one path per basename, preferring the player asset tree on any collision (same rule used
    # for the auto-resolved slots, so an item's texture pointer doesn't accidentally bind an
    # NPC/fauna file that happens to share a name).
    out["tex_by_basename"] = {
        base: min(paths, key=_nav_path_pref) for base, paths in by_base.items()
    }
    slots = out["slots"]
    for slot, meta in _NAVI_SLOT_META.items():
        if meta["kind"] != "mesh":
            continue
        part, g = meta["a"], (meta["g"] or "m")
        wanted = _MESH_STEMS[part][g]
        best = None
        for stem, p in meshes:
            if any(stem == w or stem.startswith(w) for w in wanted):
                if best is None or _mesh_rank(part, g, p) < _mesh_rank(part, g, best):
                    best = p
        if best:
            slots[slot] = best

    def _mesh_stem(slot_key):
        p = slots.get(slot_key)
        return os.path.splitext(os.path.basename(p))[0].lower() if p else None

    for slot, meta in _NAVI_SLOT_META.items():
        if meta["kind"] != "tex":
            continue
        if slot in NAVI_EDIT_ONLY_SLOTS:
            continue  # opt-in extras: never auto-resolved, so they have no default
        bucket, role, g = meta["a"], meta["b"], meta["g"]
        g = g or None  # '' -> genderless (patterns are shared)
        # the texture should belong to THIS character's part. For the SKIN (head/body) the correct
        # maps are the NPC rnf_head_01_*/rnf_body_01_* set (not the player p_head/p_body textures),
        # so expect that stem; the player mesh is still used for geometry. For eyes the player p_eyes
        # / p_eyelashes families. This is what keeps a whole-game export from binding the wrong maps.
        if bucket == "body":
            expect = (
                "rnf_body_01_%s" % (g or "m")
                if role not in ("detail", "pattern")
                else None
            )
        elif bucket == "head":
            expect = (
                "rnf_head_01_%s" % (g or "m")
                if role not in ("detail", "pattern")
                else None
            )
        elif bucket == "hair":
            expect = _mesh_stem("nav_hair")
        elif bucket == "eye":
            expect = "p_eyelashes" if role == "lash" else "p_eyes"
        else:
            expect = None
        best, best_key = None, None
        for stem, p in texes:
            if _classify_texture(stem) != (bucket, role):
                continue
            sg = _stem_gender(stem)
            # HEAD skin maps: the male rnf head mesh reuses the FEMALE rnf head textures (its
            # mgraphobject references rnf_head_01_f_*), and a male rnf head set may be incomplete or
            # absent. So for the head's d/m/n roles accept rnf_head_01 of EITHER gender and NEVER the
            # player p_head_* textures; prefer the requested gender, then the other rnf gender.
            if bucket == "head" and role in ("color", "material", "normal"):
                if not stem.startswith("rnf_head_01"):
                    continue  # exclude p_head_* / anything non-rnf
                gender_miss = 1 if (g is not None and sg is not None and sg != g) else 0
                prefix_ok = 0  # already constrained to rnf_head
                key = (gender_miss, _nav_path_pref(p), _tex_rank(g or "m", stem))
                if best is None or key < best_key:
                    best, best_key = p, key
                continue
            # detail normals are often genderless (shared dn_*_n library), so a gendered detail slot
            # accepts a stem of its own gender OR a genderless one; other roles want their own gender.
            if role == "detail":
                if g is not None and sg is not None and sg != g:
                    continue
            else:
                if g is not None and sg != g:  # gendered slot wants its own gender only
                    continue
                if g is None and bucket in ("body", "head") and sg is not None:
                    continue  # shared body/head slot skips gendered stems
            prefix_ok = 0 if (expect and stem.startswith(expect)) else 1
            key = (_nav_path_pref(p), prefix_ok, _tex_rank(g or "m", stem))
            if best is None or key < best_key:
                best, best_key = p, key
        if best:
            slots[slot] = best
    # Named-asset defaults: a few slots default to a SPECIFIC game asset rather than whatever the
    # (bucket, role) heuristic happens to pick - the hair/kuru decor diffuse is the shared hairbands
    # atlas, and the kuru direction map is the rnf hair-01 strand dir. Resolve each by basename
    # against the scanned tree (so on-disk casing doesn't matter) and let it win over the heuristic;
    # a user's manual pick (an override) still wins over this in navi_resolve.
    for _slot in ("nav_hair_acc", "nav_kuru_acc", "nav_kuru_dir"):
        _hit = navi_resolve_engine_path(out, NAVI_GAME_PATH.get(_slot, ""))
        if _hit:
            slots[_slot] = _hit
    return out


def navi_resolve_engine_path(cache, engine_path):
    """Resolve an in-game-style relative path (as found in a .blueitemtype's myTextureData,
    e.g. 'blue/baked/characterart/Player/hair/p_rnf_hair_03/p_rnf_hair_03_mask.dds') to a real
    file under the scanned export folder. Matches by BASENAME against the cache's basename
    index (built during the one folder walk) rather than the literal relative path, since the
    export's on-disk casing doesn't always match the engine path's casing. Returns None if no
    file with that name was found in the export."""
    if not engine_path:
        return None
    base = os.path.splitext(os.path.basename(engine_path.replace("\\", "/")))[0].lower()
    by_base = (cache or {}).get("tex_by_basename", {}) or {}
    return by_base.get(base)


def navi_resolve(cache, overrides=None, gender=None):
    """Resolve every slot to a path from a scan cache (scan_navi_folder) + manual overrides. An
    override wins, else the cached path. NO disk walk. `gender` is ignored (slots are explicit)."""
    overrides = overrides or {}
    slots = (cache or {}).get("slots", {}) or {}
    return {slot: (overrides.get(slot) or slots.get(slot)) for slot in _NAVI_SLOT_META}


def navi_viewer_assets(cache, overrides=None, gender="m"):
    """Build the viewer's {"meshes":{part:path}, "textures":{bucket:{role:path}}} for the active
    gender, falling back to the other gender per part/role when the active one is absent."""
    sp = navi_resolve(cache, overrides)
    other = "f" if gender == "m" else "m"
    meshes = {
        "head": sp.get("nav_head_%s" % gender) or sp.get("nav_head_%s" % other),
        "body": sp.get("nav_body_%s" % gender) or sp.get("nav_body_%s" % other),
        "hair": sp.get("nav_hair"),
        "kuru": sp.get("nav_kuru"),
    }
    textures = {
        "head": {},
        "body": {},
        "eye": {},
        "hair": {},
        "kuru": {},
        "accessory": {},
    }
    paired = {}  # (bucket, role) -> {gender: path} for gendered tex slots
    for slot, meta in _NAVI_SLOT_META.items():
        if meta["kind"] != "tex":
            continue
        path = sp.get(slot)
        if not meta["g"]:  # None or '' -> genderless (shared) slot
            if path:
                textures[meta["a"]][meta["b"]] = path
        else:
            paired.setdefault((meta["a"], meta["b"]), {})[meta["g"]] = path
    for (bucket, role), gmap in paired.items():
        path = gmap.get(gender) or gmap.get(other)
        if path:
            textures[bucket][role] = path
        # Head hair-cap (scalp tint on the head's UV1): a PER-HAIRSTYLE texture (each hair ships its own
        # <hair>_mask.dds). A user override always wins; otherwise DERIVE the cap from the loaded
        # hairstyle's sibling mask so it follows the hair. Only if that hair has no sibling mask do we fall
        # back to keeping an auto-resolved cap that classifies as a real player head cap (so a stray mask
        # can't auto-flood the face via UV1).
    if not (overrides or {}).get("nav_head_cap"):
        derived_cap = None
        hair_mesh = meshes.get("hair")
        if hair_mesh:
            cand = os.path.splitext(hair_mesh)[0] + "_mask.dds"
            if os.path.isfile(cand):
                derived_cap = cand
        if derived_cap:
            textures["head"]["haircap"] = derived_cap
        else:
            cap = textures.get("head", {}).get("haircap")
            if cap:
                cap_stem = os.path.splitext(os.path.basename(cap))[0].lower()
                if _classify_texture(cap_stem) != ("head", "haircap"):
                    textures["head"].pop("haircap", None)
    # The hair-accessory ('accessory' bucket) and kuru-band ('kuru' colour) diffuses are now set
    # explicitly by the user via the nav_hair_acc / nav_kuru_acc texture slots (resolved by the
    # generic tex loop above), rather than auto-resolved from a sibling .mgraphobject / catalogue.
    return {"meshes": meshes, "textures": textures}


def navi_missing_required(cache, overrides=None, gender="m"):
    """Required assets that aren't satisfied for the active gender (after the one-of-gender-pair
    fallback). Colour items are excluded - they are chosen in the Na'vi tab."""
    va = navi_viewer_assets(cache, overrides, gender)

    def have(p):
        return bool(p and os.path.isfile(p))

    missing = []
    for part, label in (
        ("head", "head mesh"),
        ("body", "body mesh"),
        ("hair", "hair mesh"),
    ):
        if not have(va["meshes"].get(part)):
            missing.append(label)
    for bucket, role, label in (
        ("body", "color", "body albedo"),
        ("body", "material", "body material"),
        ("head", "color", "head albedo"),
        ("head", "material", "head material"),
        ("eye", "iris", "eye iris"),
        ("hair", "mask", "hair mask"),
    ):
        if not have(va["textures"].get(bucket, {}).get(role)):
            missing.append(label)
    return missing
