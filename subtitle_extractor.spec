# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitle Timer GUI.

Builds a windowed (no console) onedir app. Only OpenCV (+ numpy) is bundled:
the trained detection model runs through cv2.dnn, so no PyTorch or other ML
frameworks are needed at runtime.

Build:   pyinstaller subtitle_extractor.spec
Result:  dist/SubtitleExtractor/SubtitleExtractor.exe
"""

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("cv2")

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
    # Heavy packages that must never sneak back into the bundle.
    excludes=[
        'matplotlib', 'PyQt5', 'PySide2',
        'torch', 'torchvision', 'easyocr', 'whisper', 'rapidfuzz', 'skimage',
        'scipy', 'PIL',
    ],
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
