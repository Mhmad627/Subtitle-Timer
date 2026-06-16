# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitle Timer GUI.

Builds a windowed (no console) onedir app. The heavy ML libraries ship data
files and dynamically-imported submodules that PyInstaller can't find on its
own, so we pull them in explicitly with collect_all.

Build:   pyinstaller subtitle_extractor.spec
Result:  dist/SubtitleExtractor/SubtitleExtractor.exe
"""

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# Packages that need their data files / dynamic submodules collected.
for pkg in ("whisper", "easyocr", "rapidfuzz", "cv2", "skimage"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# torch is huge; collect_submodules keeps it importable without grabbing
# everything twice. (collect_all on torch can blow up build time.)
from PyInstaller.utils.hooks import collect_submodules
hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("torchvision")


block_cipher = None

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'PyQt5', 'PySide2'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SubtitleExtractor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # windowed app (no terminal)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SubtitleExtractor',
)
