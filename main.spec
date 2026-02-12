# -*- mode: python ; coding: utf-8 -*-
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['cupy_backends.cuda.api._runtime_enum', 'cupy.cuda.common', 'pydicom.encoders.pylibjpeg', 'pydicom.encoders.gdcm','scipy._cyutility', 'cupy_backends.cuda.stream', 'cupy_backends.cuda._softlink', 'cupy._core._carray', 'fastrlock', 'fastrlock.rlock', 'cupy_backends.cuda.api._driver_enum', 'cupy._core._ufuncs', 'cupy._core._cub_reduction', 'cupy._core._routines_sorting', 'cupy._core.flags', 'cupy._core.new_fusion', 'cupy._core._fusion_trace', 'cupy._core._fusion_variable', 'cupy._core._fusion_op', 'cupy._core._fusion_optimization', 'cupy._core._fusion_kernel'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pyqt5', 'alabaster', 'sphinx', 'pyopencl', 'pycuda'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    
    )
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
a.datas += Tree("C:\\Users\\xufen\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cupy\\_core\\include\\cupy",".\\cupy\\_core\\include\\cupy")

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VisOCTExplorer',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='./icon/Logo.png'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)