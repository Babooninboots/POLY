# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for POLY — Polycrystalline LAMMPS Generator.

Usage:
    pyinstaller gui_main.spec

The bundled executable is placed in dist/POLY/gui_main.exe.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_PROJ = Path(SPECPATH).resolve().absolute()

# ---------------------------------------------------------------------------
# Hidden imports — Python packages that PyInstaller can't auto-detect
# ---------------------------------------------------------------------------

# SciPy (.spatial, .spatial.transform, etc.)
_hidden_scipy = [
    "scipy.spatial", "scipy.spatial.transform",
    "scipy.spatial.transform._rotation_groups",
    "scipy.spatial._ckdtree",
    "scipy.optimize", "scipy.linalg",
    "scipy.sparse", "scipy.sparse.csgraph",
    "scipy.interpolate", "scipy.ndimage",
    "scipy.special", "scipy.stats",
]

# VTK — the vtkmodules package is the standard entry point
_hidden_vtk = [
    "vtkmodules", "vtkmodules.all", "vtkmodules.vtkCommonCore",
    "vtkmodules.vtkCommonDataModel", "vtkmodules.vtkCommonMath",
    "vtkmodules.vtkCommonTransforms", "vtkmodules.vtkCommonExecutionModel",
    "vtkmodules.vtkFiltersCore", "vtkmodules.vtkFiltersGeneral",
    "vtkmodules.vtkFiltersGeometry", "vtkmodules.vtkFiltersSources",
    "vtkmodules.vtkFiltersExtraction", "vtkmodules.vtkRenderingCore",
    "vtkmodules.vtkRenderingOpenGL2", "vtkmodules.vtkInteractionStyle",
    "vtkmodules.vtkInteractionWidgets", "vtkmodules.vtkIOImage",
    "vtkmodules.vtkIOLegacy", "vtkmodules.vtkIOXML",
    "vtkmodules.vtkIOCore", "vtkmodules.vtkIOExport",
    "vtkmodules.vtkRenderingAnnotation", "vtkmodules.vtkRenderingUI",
    "vtkmodules.vtkRenderingVolume", "vtkmodules.vtkRenderingVolumeOpenGL2",
    "vtkmodules.vtkChartsCore", "vtkmodules.vtkViewsCore",
    "vtkmodules.vtkViewsContext2D",
    "vtkmodules.vtkRenderingContext2D", "vtkmodules.vtkRenderingContextOpenGL2",
    "vtkmodules.vtkRenderingLabel", "vtkmodules.vtkRenderingFreeType",
    "vtkmodules.vtkImagingCore", "vtkmodules.vtkImagingGeneral",
    "vtkmodules.vtkImagingHybrid",
    "vtkmodules.vtkRenderingLOD",
]

# PySide6 Qt plugins and helpers
_hidden_pyside6 = [
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets",
    "PySide6.QtSvg", "PySide6.QtNetwork",
]

# ASE
_hidden_ase = [
    "ase", "ase.atoms", "ase.data", "ase.io", "ase.lattice",
    "ase.spacegroup", "ase.build",
]

# pyqtgraph
_hidden_pyqtgraph = [
    "pyqtgraph", "pyqtgraph.graphicsItems",
    "pyqtgraph.exporters",
]

# pyvista / pyvistaqt
_hidden_pyvista = [
    "pyvista", "pyvistaqt",
    "pyvista.plotting", "pyvista.core",
    "pyvista.utilities",
]

# Our local modules (relative to project root are handled by pathex)
_hidden_local = [
    "gui_views", "workers", "grain_seeds",
    "orientation", "pristine_crystal", "pc_assembly",
]

hiddenimports = (
    _hidden_scipy
    + _hidden_vtk
    + _hidden_pyside6
    + _hidden_ase
    + _hidden_pyqtgraph
    + _hidden_pyvista
    + _hidden_local
)

# ---------------------------------------------------------------------------
# PySide6 plugin / DLL collections
# ---------------------------------------------------------------------------
# Let PyInstaller auto-collect Qt plugins via hooks — but add explicit paths
# for platform plugins since they're critical.
_pyside6_dir = None
try:
    import PySide6
    _pyside6_dir = Path(PySide6.__file__).parent
except Exception:
    pass

_qt_plugins = []
if _pyside6_dir and _pyside6_dir.exists():
    _plugins_dir = _pyside6_dir / "Qt" / "plugins"
    if _plugins_dir.exists():
        _qt_plugins.append((str(_plugins_dir / "platforms"), "PySide6/Qt/plugins/platforms"))
        # styles, imageformats
        for _sub in ("styles", "imageformats"):
            _sp = _plugins_dir / _sub
            if _sp.exists():
                _qt_plugins.append((str(_sp), f"PySide6/Qt/plugins/{_sub}"))

# ASE data files (spacegroup.dat for ase.spacegroup.crystal)
_ase_datas = []
try:
    import ase
    _ase_dir = Path(ase.__file__).parent
    _sg_dat = _ase_dir / "spacegroup" / "spacegroup.dat"
    if _sg_dat.exists():
        _ase_datas.append((str(_sg_dat), "ase/spacegroup"))
except Exception:
    pass

_datas = _qt_plugins + _ase_datas

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [str(_PROJ / "gui_main.py")],
    pathex=[str(_PROJ)],
    binaries=[],
    datas=_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavyweight packages we don't use
        "tkinter", "Tkinter", "tcl",
        "pandas",
        "jedi", "parso", "IPython", "jupyter",
        "notebook", "nbformat",
        "sphinx", "docutils",
        "Cython",
        "setuptools", "pip",
        "pkg_resources",
        "wx", "gtk",
    ],
    noarchive=False,
    optimize=0,
)

# ---------------------------------------------------------------------------
# Collect additional VTK data files (shaders, etc.)
# ---------------------------------------------------------------------------
# VTK modules live in site-packages/vtkmodules/ — add shader subdirs
_extra_datas = []
try:
    import vtkmodules
    _vtkmod_dir = Path(vtkmodules.__file__).parent
    # Shaders
    _shaders_dir = _vtkmod_dir / "Rendering" / "OpenGL2"
    if _shaders_dir.exists():
        for _item in _shaders_dir.rglob("*"):
            if _item.is_file():
                _target = str(Path("vtkmodules") / _item.relative_to(_vtkmod_dir))
                _extra_datas.append((str(_item), str(_target.parent)))
except Exception:
    pass

a.datas += TOC(_extra_datas)

# ---------------------------------------------------------------------------
# PYZ
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
# EXE
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="POLY_v1.6.1",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,       # UPX compression for smaller output
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,   # Windows GUI app — no console window
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(_PROJ / "icon.ico") if (_PROJ / "icon.ico").exists() else None,
)
