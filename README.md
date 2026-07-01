# Pandora Paint
<img width="512" height="512" alt="pandora-paint-1024" src="https://github.com/user-attachments/assets/406a2c5c-5425-4b89-a238-bbcc35cac025" />

**A recolouring studio and item reference browser for *Avatar: Frontiers of Pandora* (Snowdrop engine) assets.**

Pandora Paint loads the game's Na'vi, Ikran, weapon and gear assets, lets you recolour and re-pattern them against a live, shader-accurate 3D preview, and ships with a searchable Item Wiki built from the game's own customization data. It runs entirely on your own extracted game files — no game code or assets are distributed with it.

> ⚠️ **Unofficial fan project.** Not affiliated with, endorsed by, or supported by Ubisoft, Massive Entertainment, Lightstorm, or Disney. *Avatar: Frontiers of Pandora* and all related assets are the property of their respective owners. This tool operates only on files **you** have legally extracted from your own copy of the game.

---

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Windows (download & run)](#windows-download--run)
  - [Linux (from source)](#linux-from-source)
- [Building from source](#building-from-source)
- [First run & usage](#first-run--usage)
- [Supported formats](#supported-formats)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Credits](#credits)

---

## Features

### Na'vi character recolouring
- Live GPU preview of the player Na'vi (head, body, hair, neural queue / *kuru*).
- Independent control of **skin**, **hair** (3-colour root → mid → tip gradient plus scalp cap), **eyes** (inner/outer iris, left and right independently), and **warpaint**.
- Shader-accurate colour maths ported line-for-line from the game's `px_character` / `px_eye` / `px_hair` shaders, so the preview matches in-game rendering.
- Hair rendered with authored strand tangents (sheen), alpha-blended lashes, and proper scalp-cap blending.

### Ikran (banshee) recolouring
- Recolour and re-pattern the banshee mount (head and body atlases), plus its gear.
- Pattern/coat controls with a live preview on the real mesh.

### Gear & camo
- Edit **Gear Camo** and **Gear Colour** palettes (`.rejuice`) in place.
- Per-region camo composition using the game's triplanar camo shader.

### Item Wiki
- Built-in, read-only reference browser for AFoP customization items: **Ikran**, **Na'vi**, **Weapons**, **Gear Camo**, and **Gear Colours**.
- Per-category tables with UI names, colour swatches, source `.blueitemtype` paths, base colours, and UIDs.
- **Export the entire wiki** as a `.zip` of CSV files (one per category sub-tab) from *Settings → Item Wiki*.

### Real-time 3D viewer
- Single shared **ModernGL** context (OpenGL 4.x).
- **SSAA** supersampling (the AA that actually cleans up alpha-tested hair/membrane cutouts), optional **FXAA**, and anisotropic texture filtering.
- **ACES** tonemapping, base + detail normal mapping, hair sheen, and alpha-discard cutouts.
- Adjustable display pose (e.g. the Na'vi tail curl).

### Presets, export & workflow
- **Save / Load** named presets per tab (Ikran and Na'vi).
- Export colour patterns (`.mcolorpattern`) and pattern controls (`.mpatterncontrol`).
- **Bake and export recoloured textures** per region.
- Remembers your asset folders and per-slot file overrides; configurable export folder with optional `blue/…` path replication.
- Dark UI with a customizable accent colour.

---

## Requirements

- A GPU/driver with **OpenGL 4.x** (Mesa on Linux, or vendor drivers on Windows).
- Your **own extracted** *Avatar: Frontiers of Pandora* asset files.

The pre-built Windows release is self-contained and needs **no Python install**. Running from source (Linux) additionally requires **Python 3.10+** and the packages in `requirements.txt` (`PyQt6`, `moderngl`, `numpy`, `Pillow`).

Developed and tested on Arch Linux (Wayland, AMD RX 9070 XT / Mesa) and Windows 10/11.

---

## Installation

### Windows (download & run)

1. Open the [**Releases**](https://github.com/SaintBaron/PandoraPaint/releases) page.
2. Download **`PandoraPaint.zip`** from the latest release.
3. Extract it — you'll get a **`PandoraPaint`** folder.
4. Open that folder and run **`PandoraPaint.exe`**.

That's it — no Python, no dependencies to install. The first launch may show a Windows SmartScreen prompt because the app is unsigned; click **More info → Run anyway**. Keep the extracted `PandoraPaint` folder intact (the exe loads its resources from alongside it).

### Linux (from source)

```bash
git clone https://github.com/SaintBaron/PandoraPaint.git
cd PandoraPaint
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

On subsequent runs just `source venv/bin/activate` and `python app.py`.

> Wayland users: PyQt6 auto-detects the platform. If you hit a platform-plugin error, try `QT_QPA_PLATFORM=xcb python app.py`.

---

## Building from source

To produce your own standalone build (this is how the Windows release is made), install [PyInstaller](https://pyinstaller.org/) into the same environment and package `app.py`. `--onedir` gives a near-instant launch.

```bash
pip install pyinstaller
```

> The `--add-data` separator differs by OS: **`;`** on Windows, **`:`** on Linux/macOS.

**Windows:**

```bat
python -m PyInstaller --onedir --windowed --name PandoraPaint ^
  --icon "icons\pandora-paint.ico" ^
  --add-data "icons;icons" --add-data "fonts;fonts" --add-data "item_wiki.json;." ^
  app.py
```

**Linux:**

```bash
python -m PyInstaller --onedir --windowed --name PandoraPaint \
  --add-data "icons:icons" --add-data "fonts:fonts" --add-data "item_wiki.json:." \
  app.py
```

The result is in `dist/PandoraPaint/`; zip that folder to distribute it. Bundled data (`icons/`, `fonts/`, `item_wiki.json`) is located at runtime via a frozen-aware resolver, so it loads correctly from both source and the build. On Windows, verify Pillow's native decoder made it in with `Get-ChildItem -Recurse dist\PandoraPaint -Filter "_imaging*"` (it should list `_imaging...pyd`).

---

## First run & usage

1. Launch the app and open **Settings**.
2. Point the three asset panels at your extracted game files:
   - **Ikran Assets** — the banshee meshes/textures.
   - **Na'vi Assets** — the player head/body/hair/kuru meshes and textures.
   - **Camo / Colour** — the `gearcamo_colorpalettes.rejuice` / `gearcolors_colorpalettes.rejuice` palettes.
3. Switch to the **Na'vi** or **Ikran** tab, pick colours/patterns, and watch the live preview update.
4. **Save** a preset, or **export** the recoloured textures / colour pattern when you're happy.
5. Browse the **Item Wiki** tab for a reference of every customization item, or export it to CSV from *Settings → Item Wiki*.

Pandora Paint never modifies your source files — it reads them and writes new output to your chosen export folder.

---

## Supported formats

| Type | Formats |
|------|---------|
| Meshes | Snowdrop **`.mmb`** skeletal meshes (versions 11–17, LOD0) |
| Textures (read) | **STF `.dds`** — pure-Python BC1/BC2/BC3/BC4/BC5/BC7 decoder; `.png` and other formats via Pillow |
| Palettes | Gear camo / colour **`.rejuice`** |
| Presets / export | `.json` presets, **`.mcolorpattern`**, **`.mpatterncontrol`**, exported texture images |
| Reference data | `item_wiki.json` (the built-in Item Wiki) |

---

## Troubleshooting

**"Pillow is required to load textures"**
Pillow isn't installed in the environment you're running/building from. Install it (`pip install Pillow`) — and if it persists in a build, you likely have an interpreter mismatch: build inside a venv and call it by full path (`venv\Scripts\python.exe -m PyInstaller …`) so pip and PyInstaller share one interpreter. `.dds` textures load without Pillow (via the STF reader); only `.png`/other formats and texture export need it.

**Item Wiki tab is empty**
`item_wiki.json` failed to parse or wasn't found. Enable *Settings → Diagnostics* logging to see the exact path/reason. A common cause is illegal **trailing commas** in a hand-edited/regenerated `item_wiki.json` — strict JSON rejects them; clean it with the included `fix_wiki_json.py`.

**Blank 3D viewport / OpenGL errors**
Ensure your driver exposes OpenGL 4.x (`glxinfo | grep "OpenGL version"` on Linux). In a build, if the viewport is blank, rebuild with `--collect-all moderngl`.

**Slow launch of a build**
That's `--onefile` re-extracting itself each start. Rebuild with `--onedir` for near-instant launch.

---

## Project layout

```
app.py            entry point (window, theming, fonts, icon)
main_window.py    main window, tabs, Settings, Item Wiki tab
widgets.py        controls, panels, Save/Load bar, lazy texture loader
viewer.py         ModernGL renderer (Na'vi + Ikran), shaders, pose skinning
mmb_loader.py     pure-numpy Snowdrop .mmb mesh reader
assets.py         config, asset resolution, resource paths
recolor_core.py   CPU-reference recolour maths (mirrors the GLSL)
wiki.py           Item Wiki data + rendering
requirements.txt  runtime dependencies
item_wiki.json    Item Wiki reference data
icons/  fonts/     bundled UI resources
```

---

## Credits

- **Pandora Paint** by [SaintBaron](https://github.com/SaintBaron).
- `.mmb` mesh format reverse-engineering adapted from the **AFoP Mesh Tool** (AlexPo, JasperZebra, J-Lyt, SaintBaron).
- Colour/shader behaviour derived from the game's own Snowdrop material shaders, reimplemented for previewing.

*Pandora Paint is a fan-made asset tool and does not include or redistribute any* Avatar: Frontiers of Pandora *game content. All game assets and trademarks remain the property of their respective owners.*
