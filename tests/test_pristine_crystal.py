"""
Tests for Module 2 -- pristine_crystal.py

Run:
    python test_pristine_crystal.py
"""

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pristine_crystal import (
    PristineCrystal,
    CrystalResult,
    generate_pristine_bravais,
    generate_pristine_cif,
)


BOX_SMALL = ((0, 0, 0), (30, 30, 30))
BOX_RECT  = ((0, 0, 0), (80, 40, 20))


# ===================================================================
# 1. Constructor & factory validation
# ===================================================================

def test_constructor_stores_attrs():
    from ase.build import bulk
    uc = bulk("Al", "fcc", a=4.05)
    pc = PristineCrystal(uc, box_start=BOX_SMALL[0], box_end=BOX_SMALL[1])
    assert np.allclose(pc.box_size, [30, 30, 30])
    assert len(pc.unit_cell) == 1  # fcc primitive


def test_from_bravais_auto_lookup():
    pc = PristineCrystal.from_bravais("Fe", box_start=(0, 0, 0), box_end=(50, 50, 50))
    assert pc.unit_cell is not None
    # Fe defaults to bcc with a ~2.8665
    cell = pc.unit_cell.get_cell()
    assert np.linalg.norm(cell).sum() > 0


def test_from_bravais_explicit():
    pc = PristineCrystal.from_bravais(
        "Cu", "fcc", a=3.61, box_start=(0, 0, 0), box_end=(40, 40, 40)
    )
    assert len(pc.unit_cell) >= 1


def test_from_bravais_unknown_element():
    """Element not in ELEMENT_DEFAULTS -- let ASE guess."""
    pc = PristineCrystal.from_bravais(
        "Rh", box_start=(0, 0, 0), box_end=(30, 30, 30)
    )
    assert pc.unit_cell is not None


def test_from_spacegroup():
    pc = PristineCrystal.from_spacegroup(
        symbols="Al",
        basis=[(0, 0, 0)],
        spacegroup=225,
        cellpar=[4.05, 4.05, 4.05, 90, 90, 90],
        box_start=(0, 0, 0),
        box_end=(40, 40, 40),
    )
    assert pc.unit_cell is not None


def test_from_cif():
    from ase.build import bulk
    from ase.io import write as ase_write
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
        tmp = f.name
    try:
        ase_write(tmp, bulk("Ni", "fcc", a=3.52))
        pc = PristineCrystal.from_cif(tmp, box_start=(0, 0, 0), box_end=(30, 30, 30))
        assert pc.unit_cell is not None
    finally:
        import os
        os.unlink(tmp)


def test_from_custom():
    pc = PristineCrystal.from_custom(
        cell=[[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]],
        scaled_positions=[[0, 0, 0]],
        symbols="W",
        box_start=(0, 0, 0),
        box_end=(30, 30, 30),
    )
    assert len(pc.unit_cell) == 1


# ===================================================================
# 2. Supercell construction
# ===================================================================

def test_supercell_fills_box_bravais():
    pc = PristineCrystal.from_bravais("Al", "fcc", a=4.05, box_start=BOX_SMALL[0], box_end=BOX_SMALL[1])
    pc.build_supercell(margin=2)
    pos = pc.atoms.get_positions()
    extent = pos.max(axis=0) - pos.min(axis=0)
    assert all(extent >= pc.box_size), f"{extent} < {pc.box_size}"


def test_supercell_fills_box_spacegroup():
    pc = PristineCrystal.from_spacegroup(
        symbols="Al", basis=[(0, 0, 0)], spacegroup=225,
        cellpar=[4.05, 4.05, 4.05, 90, 90, 90],
        box_start=(0, 0, 0), box_end=(35, 35, 35),
    )
    pc.build_supercell(margin=2)
    pos = pc.atoms.get_positions()
    extent = pos.max(axis=0) - pos.min(axis=0)
    assert all(extent >= pc.box_size), f"{extent} < {pc.box_size}"


def test_supercell_fills_box_custom():
    pc = PristineCrystal.from_custom(
        cell=[[2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5]],
        scaled_positions=[[0, 0, 0]],
        symbols="Fe",
        box_start=(0, 0, 0), box_end=(25, 25, 25),
    )
    pc.build_supercell(margin=1)
    pos = pc.atoms.get_positions()
    extent = pos.max(axis=0) - pos.min(axis=0)
    assert all(extent >= pc.box_size), f"{extent} < {pc.box_size}"


def test_supercell_rectangular_box():
    pc = PristineCrystal.from_bravais("Cu", "fcc", a=3.61, box_start=BOX_RECT[0], box_end=BOX_RECT[1])
    pc.build_supercell(margin=2)
    pos = pc.atoms.get_positions()
    extent = pos.max(axis=0) - pos.min(axis=0)
    assert all(extent >= pc.box_size), f"{extent} < {pc.box_size}"


def test_repeats_are_reasonable():
    """Repeats should be within a sane range (not 10^6 for a 50A box)."""
    pc = PristineCrystal.from_bravais("Al", "fcc", a=4.05, box_start=BOX_SMALL[0], box_end=BOX_SMALL[1])
    pc.build_supercell(margin=2)
    for r in pc.repeats:
        assert 1 <= r <= 50, f"repeat {r} out of range"


# ===================================================================
# 3. COM centering
# ===================================================================

def test_com_at_origin_bravais():
    pc = PristineCrystal.from_bravais("Al", "fcc", a=4.05, box_start=BOX_SMALL[0], box_end=BOX_SMALL[1])
    pc.build_supercell()
    pc.center_to_origin()
    com = pc.atoms.get_center_of_mass()
    assert np.allclose(com, [0, 0, 0], atol=1e-9), f"COM = {com}"


def test_com_at_origin_spacegroup():
    pc = PristineCrystal.from_spacegroup(
        symbols="Al", basis=[(0, 0, 0)], spacegroup=225,
        cellpar=[4.05, 4.05, 4.05, 90, 90, 90],
        box_start=(0, 0, 0), box_end=(30, 30, 30),
    )
    pc.build_supercell()
    pc.center_to_origin()
    com = pc.atoms.get_center_of_mass()
    assert np.allclose(com, [0, 0, 0], atol=1e-9), f"COM = {com}"


def test_center_to_origin_without_supercell_raises():
    pc = PristineCrystal.from_bravais("Al", "fcc", a=4.05, box_start=BOX_SMALL[0], box_end=BOX_SMALL[1])
    try:
        pc.center_to_origin()
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


# ===================================================================
# 4. Full pipeline (run)
# ===================================================================

def test_run_bravais():
    r = generate_pristine_bravais("Al", "fcc", a=4.05, box_start=BOX_SMALL[0], box_end=BOX_SMALL[1], verbose=False)
    assert isinstance(r, CrystalResult)
    assert r.n_atoms > 0
    assert all(r.repeats[i] >= 1 for i in range(3))
    com = r.atoms.get_center_of_mass()
    assert np.allclose(com, [0, 0, 0], atol=1e-9)


def test_run_cif():
    from ase.build import bulk
    from ase.io import write as ase_write
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
        tmp = f.name
    try:
        ase_write(tmp, bulk("Ni", "fcc", a=3.52))
        r = generate_pristine_cif(tmp, box_start=(0, 0, 0), box_end=(30, 30, 30), verbose=False)
        assert r.n_atoms > 0
        extent = r.atoms.get_positions().max(axis=0) - r.atoms.get_positions().min(axis=0)
        assert all(extent >= r.box_size)
    finally:
        import os
        os.unlink(tmp)


def test_run_verbose_output():
    """Ensure verbose mode runs without error (output goes to stdout)."""
    r = generate_pristine_bravais("Al", "fcc", a=4.05, box_start=BOX_SMALL[0], box_end=BOX_SMALL[1], verbose=True)
    assert r.n_atoms > 0


# ===================================================================
# 5. CrystalResult properties
# ===================================================================

def test_result_properties():
    r = generate_pristine_bravais("Al", "fcc", a=4.05, box_start=BOX_SMALL[0], box_end=BOX_SMALL[1], verbose=False)
    assert r.n_atoms == len(r.atoms)
    assert r.chemical_formula == r.atoms.get_chemical_formula()
    shape = r.supercell_shape
    assert shape[0] == r.n_atoms
    assert shape[1:] == r.repeats


# ===================================================================
# 6. Persistence
# ===================================================================

def test_save_and_load():
    import os
    pc = PristineCrystal.from_bravais("Al", "fcc", a=4.05, box_start=BOX_SMALL[0], box_end=BOX_SMALL[1])
    pc.run(verbose=False)
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        tmp = f.name
    try:
        pc.save(tmp)
        loaded = PristineCrystal.load(tmp)
        assert len(loaded) == len(pc.atoms)
    finally:
        os.unlink(tmp)


def test_save_without_run_raises():
    pc = PristineCrystal.from_bravais("Al", "fcc", a=4.05, box_start=BOX_SMALL[0], box_end=BOX_SMALL[1])
    try:
        pc.save("nowhere.xyz")
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


# ===================================================================
# 7. Element defaults table
# ===================================================================

def test_element_defaults_coverage():
    """Every element in the table should build without error."""
    for symbol in PristineCrystal.ELEMENT_DEFAULTS:
        pc = PristineCrystal.from_bravais(symbol, box_start=BOX_SMALL[0], box_end=BOX_SMALL[1])
        assert pc.unit_cell is not None, f"Failed for {symbol}"


def test_element_hcp_has_two_params():
    """HCP elements should have a and c in the table."""
    for symbol, info in PristineCrystal.ELEMENT_DEFAULTS.items():
        if info["structure"] == "hcp":
            assert "c" in info, f"{symbol} HCP missing c"
            assert "a" in info, f"{symbol} HCP missing a"


# ===================================================================
# 8. Edge cases
# ===================================================================

def test_margin_zero():
    """margin=0 with +1 COM-padding still fills the box."""
    pc = PristineCrystal.from_custom(
        cell=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        scaled_positions=[[0, 0, 0]],
        symbols="Au",
        box_start=(0, 0, 0), box_end=(20, 20, 20),
    )
    pc.build_supercell(margin=0)
    pos = pc.atoms.get_positions()
    extent = pos.max(axis=0) - pos.min(axis=0)
    # ceil(20/5) + 1 + 0 = 5 repeats, extent = 20 after centering
    assert all(extent >= pc.box_size), f"{extent} < {pc.box_size}"


def test_multispecies_bravais():
    """NaCl rocksalt structure."""
    pc = PristineCrystal.from_bravais(
        "NaCl", "rocksalt", a=5.64,
        box_start=(0, 0, 0), box_end=(40, 40, 40),
    )
    pc.run(verbose=False)
    symbols = pc.atoms.get_chemical_symbols()
    assert "Na" in symbols
    assert "Cl" in symbols


# ===================================================================
# 9. Intermetallics
# ===================================================================

def test_intermetallic_all_entries_build():
    for name in PristineCrystal.INTERMETALLIC_DEFAULTS:
        pc = PristineCrystal.from_intermetallic(name, box_start=(0,0,0), box_end=(30,30,30))
        pc.run(margin=1, verbose=False)
        assert pc.atoms is not None, f"Failed for {name}"
        assert len(pc.atoms) > 0


def test_intermetallic_unknown_raises():
    try:
        PristineCrystal.from_intermetallic("BOGUS", box_start=(0,0,0), box_end=(30,30,30))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_intermetallic_wrong_symbol_count_raises():
    try:
        PristineCrystal.from_intermetallic(
            "L1_2", symbols=("Ni",), box_start=(0,0,0), box_end=(30,30,30)
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_intermetallic_custom_elements():
    pc = PristineCrystal.from_intermetallic(
        "L1_2", symbols=("Al", "Ni"), a=3.57,
        box_start=(0,0,0), box_end=(30,30,30),
    )
    pc.run(margin=1, verbose=False)
    syms = pc.atoms.get_chemical_symbols()
    # Ni should be 3x Al
    assert syms.count("Ni") == 3 * syms.count("Al")


def test_intermetallic_custom_cellpar():
    pc = PristineCrystal.from_intermetallic(
        "B2", symbols=("Fe", "Al"), cellpar=[2.90, 2.90, 2.90, 90, 90, 90],
        box_start=(0,0,0), box_end=(30,30,30),
    )
    pc.run(margin=1, verbose=False)
    syms = pc.atoms.get_chemical_symbols()
    assert syms.count("Fe") == syms.count("Al")  # B2 is 1:1


def test_intermetallic_L10_tetragonal():
    """L1_0 is tetragonal; c != a."""
    pc = PristineCrystal.from_intermetallic(
        "L1_0", symbols=("Ti", "Ti", "Al"), a=2.80, c=4.10,
        box_start=(0,0,0), box_end=(30,30,30),
    )
    pc.run(margin=1, verbose=False)
    extent = pc.atoms.get_positions().max(axis=0) - pc.atoms.get_positions().min(axis=0)
    assert all(extent >= pc.box_size)


def test_intermetallic_convenience_function():
    from pristine_crystal import generate_pristine_intermetallic
    r = generate_pristine_intermetallic(
        "A15", symbols=("Sn", "Nb"), a=5.23,
        box_start=(0,0,0), box_end=(30,30,30), verbose=False,
    )
    assert r.n_atoms > 0
    syms = r.atoms.get_chemical_symbols()
    # Nb3Sn: Nb=3x Sn
    assert syms.count("Nb") == 3 * syms.count("Sn")


def test_intermetallic_stoichiometry():
    """Verify expected stoichiometry for each prototype."""
    from collections import Counter
    expected = {
        "L1_2": {"Au": 1, "Cu": 3},
        "L1_0": {"Au": 2, "Cu": 2},
        "B2":   {"Cs": 1, "Cl": 1},
        "D0_3": {"Bi": 4, "Fe": 12},
        "L2_1": {"Al": 4, "Mn": 4, "Cu": 8},
        "D0_19":{"Sn": 2, "Ni": 6},
        "A15":  {"Si": 2, "Cr": 6},
        "C15":  {"Mg": 8, "Cu": 16},
    }
    for name, exp in expected.items():
        pc = PristineCrystal.from_intermetallic(name, box_start=(0,0,0), box_end=(20,20,20))
        r = pc.run(margin=0, verbose=False)
        # Count only in one unit cell (before centering, use unit_cell)
        counts = Counter(pc.unit_cell.get_chemical_symbols())
        assert counts == exp, f"{name}: expected {exp}, got {dict(counts)}"


# ===================================================================
# Runner
# ===================================================================

if __name__ == "__main__":
    import traceback

    module = sys.modules[__name__]
    tests = sorted(
        (name, obj)
        for name, obj in vars(module).items()
        if name.startswith("test_") and callable(obj)
    )

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
