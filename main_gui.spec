# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui\\main_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('gui', 'gui'), ('known_faces', 'known_faces'), ('attendance', 'attendance'), ('reports', 'reports'), ('snapshots', 'snapshots'), ('best_emotion_model.h5', '.'), ('face_recognition_models', 'face_recognition_models')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['gui\\DrishtiAI_Logo.ico'],
)
