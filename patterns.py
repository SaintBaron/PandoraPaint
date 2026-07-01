"""Read/write the AFoP "vanity" pattern config files (Snowdrop CRLF plain text):

  ColorPattern (.mcolorpattern)             - the 10 palette colours
  PatternControl (.mpatterncontrol)         - per-skin level/invert constants
  BansheePatternData (.mbansheepatterndata) - manifest binding a whole skin

All three round-trip byte-for-byte (includes, newline style and trailing bytes preserved).
"""

from __future__ import annotations
import re
import secrets
from dataclasses import dataclass, field


def _f(v):
    """Format a float like the game files do: compact but always with a decimal point (1.0 -> '1.0')."""
    s = "%g" % float(v)
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s


def mint_uid():
    """A fresh 32-hex-char (128-bit) uppercase uid for newly-authored assets."""
    return secrets.token_hex(16).upper()


_TOKEN = re.compile(r"(rnf|tmp|veldt|temperate|young|default)\s*0*(\d+)?", re.I)


_CP_INCLUDE = re.compile(r"^\s*include\s+(?P<path>\S+)\s*$")
_CP_HEADER = re.compile(
    r'^\s*ColorPattern\s+"(?P<name>[^"]*)"\s*<\s*uid=(?P<uid>[0-9A-Fa-f]+)\s*>\s*$'
)
_CP_COLOR = re.compile(r"^\s*myColor(?P<idx>\d+)\s+0x(?P<hex>[0-9A-Fa-f]{8})\s*$")


@dataclass
class ColorPattern:
    name: str
    uid: str
    include: str = "blue/gameplay/vanity/fruit/colorpattern.fruit"
    colors: list[int] = field(default_factory=lambda: [0xFF000000] * 10)
    newline: str = "\r\n"
    trailing: str = "\r\n\r\n"

    @staticmethod
    def argb(v):
        return (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF

    def rgb_hex(self, i):
        _, r, g, b = self.argb(self.colors[i])
        return f"{r:02X}{g:02X}{b:02X}"

    def alpha(self, i):
        return (self.colors[i] >> 24) & 0xFF

    def set_rgb(self, i, rrggbb, keep_alpha=True):
        rgb = int(rrggbb, 16) & 0xFFFFFF
        a = self.alpha(i) if keep_alpha else 0xFF
        self.colors[i] = (a << 24) | rgb

    @classmethod
    def loads(cls, text):
        name = uid = None
        include = cls.include
        colors = [0xFF000000] * 10
        nl = "\r\n" if "\r\n" in text else "\n"
        for raw in text.splitlines():
            if m := _CP_INCLUDE.match(raw):
                include = m["path"]
            elif m := _CP_HEADER.match(raw):
                name, uid = m["name"], m["uid"]
            elif m := _CP_COLOR.match(raw):
                idx = int(m["idx"])
                if 1 <= idx <= 10:
                    colors[idx - 1] = int(m["hex"], 16)
        if name is None or uid is None:
            raise ValueError("not a valid .mcolorpattern (missing ColorPattern header)")
        tail = text[text.rfind("}") + 1 :]
        return cls(
            name=name,
            uid=uid,
            include=include,
            colors=colors,
            newline=nl,
            trailing=tail,
        )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls.loads(f.read().decode("utf-8"))

    def dumps(self):
        nl = self.newline
        body = [
            f"include {self.include}",
            "",
            f'ColorPattern "{self.name}" < uid={self.uid} >',
            "{",
        ]
        for i, c in enumerate(self.colors, 1):
            body.append(f"\t myColor{i} 0x{c:08x}".replace("\t ", "\t"))
        return nl.join(body) + nl + "}" + self.trailing

    def save(self, path):
        with open(path, "wb") as f:
            f.write(self.dumps().encode("utf-8"))


# ============================================================ PatternControl
_PC_INCLUDE = re.compile(r"^\s*include\s+(?P<path>\S+)\s*$")
_PC_HEADER = re.compile(
    r'^\s*PatternControl\s+"(?P<name>[^"]*)"\s*<\s*uid=(?P<uid>[0-9A-Fa-f]+)\s*>\s*$'
)
_PC_VAL = re.compile(r"^\s*(?P<key>my\w+)\s+(?P<val>-?\d+(?:\.\d+)?)\s*$")

# attr <-> file key, in the order the game writes them
_PC_FIELDS = [
    ("invert1", "myPattern1Invert"),
    ("invert2", "myPattern2Invert"),
    ("level1", "myPattern1LevelControl"),
    ("level2", "myPattern2LevelControl"),
]
_PC_KEY2ATTR = {k: a for a, k in _PC_FIELDS}


@dataclass
class PatternControl:
    name: str
    uid: str
    invert1: float = 1.0
    invert2: float = 1.0
    level1: float = 1.0
    level2: float = 1.0
    include: str = "blue/gameplay/vanity/fruit/patterncontrol.fruit"
    newline: str = "\r\n"
    trailing: str = "\r\n\r\n"

    @classmethod
    def loads(cls, text):
        name = uid = None
        include = cls.include
        vals = {}
        nl = "\r\n" if "\r\n" in text else "\n"
        for raw in text.splitlines():
            if (m := _PC_INCLUDE.match(raw)) and m["path"].endswith(".fruit"):
                include = m["path"]
            elif m := _PC_HEADER.match(raw):
                name, uid = m["name"], m["uid"]
            elif m := _PC_VAL.match(raw):
                attr = _PC_KEY2ATTR.get(m["key"])
                if attr:
                    vals[attr] = float(m["val"])
        if name is None:
            raise ValueError(
                "not a valid .mpatterncontrol (missing PatternControl header)"
            )
        tail = text[text.rfind("}") + 1 :]
        return cls(
            name=name, uid=uid or "", include=include, newline=nl, trailing=tail, **vals
        )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls.loads(f.read().decode("utf-8", "replace"))

    def params(self):
        return {
            "invert1": self.invert1,
            "invert2": self.invert2,
            "level1": self.level1,
            "level2": self.level2,
        }

    def dumps(self):
        nl = self.newline
        body = [
            f"include {self.include}",
            "",
            f'PatternControl "{self.name}" < uid={self.uid} >',
            "{",
        ]
        for attr, key in _PC_FIELDS:
            body.append(f"\t{key} {_f(getattr(self, attr))}")
        return nl.join(body) + nl + "}" + self.trailing

    def save(self, path):
        with open(path, "wb") as f:
            f.write(self.dumps().encode("utf-8"))


# ============================================================ BansheePatternData
_PD_INCLUDE = re.compile(r'^\s*include\s+(?:"(?P<q>[^"]+)"|(?P<p>.+?))\s*$')
_PD_HEADER = re.compile(
    r"^\s*BansheePatternData\s+(?P<name>\S+)\s*<\s*uid=(?P<uid>[0-9A-Fa-f]+)\s*>\s*$"
)
_PD_REF = re.compile(
    r"^\s*(?P<key>my\w+)\s*<\s*uid=(?P<sub>[0-9A-Fa-f]+)\s*>\s*=\s*"
    r'"(?P<name>[^"]*)"\s+(?P<target>[0-9A-Fa-f]+)\s*$'
)
_PD_COAT = re.compile(r"^\s*(?P<key>my\w+Coat)\s+(?P<path>\S+)\s*$")

_PD_REF_KEYS = {
    "myBodyColorPattern": ("body", "color"),
    "myHeadColorPattern": ("head", "color"),
    "myBodyPatternControl": ("body", "control"),
    "myHeadPatternControl": ("head", "control"),
}
_PD_COAT_KEYS = {"myBodyPatternCoat": "body", "myHeadPatternCoat": "head"}
# canonical emit order
_PD_ORDER = [
    "myBodyColorPattern",
    "myHeadColorPattern",
    "myBodyPatternControl",
    "myHeadPatternControl",
]
_PD_COAT_ORDER = ["myBodyPatternCoat", "myHeadPatternCoat"]
_PD_SCHEMA = "blue/gameplay/vanity/fruit/bansheepatterndata.fruit"


@dataclass
class BansheePatternData:
    name: str
    uid: str
    includes: list[str] = field(default_factory=list)
    refs: dict = field(default_factory=dict)  # key -> {sub, name, target}
    coats: dict = field(default_factory=dict)  # key -> engine path
    newline: str = "\r\n"
    trailing: str = "\r\n\r\n"
    quote_includes: bool = True  # loaded files keep their quoting; new builds drop it

    @classmethod
    def loads(cls, text):
        name = uid = None
        includes, refs, coats = [], {}, {}
        nl = "\r\n" if "\r\n" in text else "\n"
        for raw in text.splitlines():
            if m := _PD_INCLUDE.match(raw):
                includes.append(m["q"] or m["p"])
            elif m := _PD_HEADER.match(raw):
                name, uid = m["name"], m["uid"]
            elif m := _PD_REF.match(raw):
                refs[m["key"]] = {
                    "sub": m["sub"],
                    "name": m["name"],
                    "target": m["target"],
                }
            elif m := _PD_COAT.match(raw):
                coats[m["key"]] = m["path"]
        if name is None:
            raise ValueError(
                "not a valid .mbansheepatterndata (missing BansheePatternData header)"
            )
        tail = text[text.rfind("}") + 1 :]
        return cls(
            name=name,
            uid=uid or "",
            includes=includes,
            refs=refs,
            coats=coats,
            newline=nl,
            trailing=tail,
        )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls.loads(f.read().decode("utf-8", "replace"))

    def ref(self, part, role):
        for key, (p, r) in _PD_REF_KEYS.items():
            if p == part and r == role:
                return self.refs.get(key)
        return None

    def coat(self, part):
        for key, p in _PD_COAT_KEYS.items():
            if p == part:
                return self.coats.get(key)
        return None

    def member_paths(self):
        out = {}
        for inc in self.includes:
            low = inc.lower()
            if low.endswith("bansheepatterndata.fruit"):
                continue
            part = "body" if "body" in low else ("head" if "head" in low else None)
            if part is None:
                continue
            if low.endswith(".mpatterncontrol"):
                out[(part, "control")] = inc
            elif low.endswith(".mcolorpattern"):
                out[(part, "color")] = inc
        for part in ("body", "head"):
            c = self.coat(part)
            if c:
                out[(part, "coat")] = c
        return out

    # ---- authoring ----
    @classmethod
    def build(
        cls,
        name,
        body_color,
        head_color,
        body_control,
        head_control,
        body_coat,
        head_coat,
        member_includes=None,
        uid=None,
        sub_uids=None,
    ):
        """Assemble a manifest. *_color/*_control are (name, target_uid) pairs; *_coat are engine
        paths. `uid` is minted unless given; `sub_uids` may preserve existing per-member sub-uids
        (missing ones minted). member_includes are the four member engine paths (emitted unquoted)."""
        uid = uid or mint_uid()
        sub_uids = sub_uids or {}
        refs = {}
        for key, (nm, tgt) in (
            ("myBodyColorPattern", body_color),
            ("myHeadColorPattern", head_color),
            ("myBodyPatternControl", body_control),
            ("myHeadPatternControl", head_control),
        ):
            refs[key] = {
                "sub": sub_uids.get(key) or mint_uid(),
                "name": nm,
                "target": tgt,
            }
        coats = {"myBodyPatternCoat": body_coat, "myHeadPatternCoat": head_coat}
        includes = [_PD_SCHEMA] + list(member_includes or [])
        return cls(
            name=name,
            uid=uid,
            includes=includes,
            refs=refs,
            coats=coats,
            quote_includes=False,
        )

    def dumps(self):
        nl = self.newline
        out = []
        for inc in self.includes:
            out.append(
                f'include "{inc}"'
                if (self.quote_includes and " " in inc)
                else f"include {inc}"
            )
        out.append("")
        out.append(f"BansheePatternData {self.name} < uid={self.uid} >")
        out.append("{")
        for key in _PD_ORDER:
            r = self.refs.get(key)
            if r:
                out.append(f'\t{key} < uid={r["sub"]} > = "{r["name"]}" {r["target"]}')
        for key in _PD_COAT_ORDER:
            p = self.coats.get(key)
            if p:
                out.append(f"\t{key} {p}")
        return nl.join(out) + nl + "}" + self.trailing

    def save(self, path):
        with open(path, "wb") as f:
            f.write(self.dumps().encode("utf-8"))
