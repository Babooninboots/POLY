"""
Tests for Module 5 -- pc_assembly.py

Run:
    python test_pc_assembly.py
"""

import sys
import tempfile
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grain_seeds import generate_grains
from pristine_crystal import (
    generate_pristine_bravais,
    PristineCrystal,
)
from orientation import generate_orientations
from pc_assembly import (
    PolycrystalAssembly,
    AssemblyResult,
    assemble_polycrystal,
)


BOX_S = (0, 0, 0)
BOX_E = (50, 50, 50)
BOX_S_NP = np.array(BOX_S, dtype=float)
BOX_E_NP = np.array(BOX_E, dtype=float)


def _make_components(
    n_grains=6,
    crystal_args=None,
    ori_mode="random",
    seed=42,
):
    """Create seeds, crystal, and orientations for testing."""
    if crystal_args is None:
        crystal_args = ("Al", "fcc", 4.05)

    grains = generate_grains(
        box_start=BOX_S, box_end=BOX_E,
        n_grains=n_grains, distribution="random",
        random_seed=seed, verbose=False,
    )

    crystal = generate_pristine_bravais(
        *crystal_args,
        box_start=BOX_S, box_end=BOX_E,
        verbose=False,
    )

    ori_kwargs = {}
    if ori_mode in ("low_angle", "high_angle", "custom_misorientation"):
        ori_kwargs["neighbors"] = grains.neighbors

    ori = generate_orientations(
        ori_mode,
        n_grains=grains.n_grains,
        random_seed=seed + 1,
        verbose=False,
        **ori_kwargs,
    )

    return grains, crystal, ori


# ===================================================================
# 1. Constructor validation
# ===================================================================

def test_constructor_seed_ori_count_mismatch():
    grains, crystal, ori = _make_components(n_grains=5)
    # Truncate seeds so count != orientation count
    try:
        PolycrystalAssembly(
            seeds=grains.seeds[:3],
            crystal_atoms=crystal.atoms,
            orientations=ori,
            box_start=BOX_S,
            box_end=BOX_E,
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ===================================================================
# 2. Assembly basics
# ===================================================================

def test_assemble_returns_correct_shapes():
    grains, crystal, ori = _make_components(n_grains=6)
    assembler = PolycrystalAssembly(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
    )
    pos, typ, gid, euler, counts = assembler.assemble(verbose=False)

    assert pos.ndim == 2 and pos.shape[1] == 3
    assert typ.ndim == 1
    assert gid.ndim == 1
    assert euler.ndim == 2 and euler.shape[1] == 3
    assert len(pos) == len(typ) == len(gid) == len(euler)
    assert len(counts) == 6
    assert counts.sum() == len(pos)


def test_assemble_atoms_within_box():
    grains, crystal, ori = _make_components(n_grains=4)
    assembler = PolycrystalAssembly(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
    )
    pos, _, _, _, _ = assembler.assemble(verbose=False)

    assert np.all(pos >= BOX_S_NP)
    assert np.all(pos <= BOX_E_NP)


def test_assemble_no_empty_grains():
    """Every grain should retain at least some atoms."""
    grains, crystal, ori = _make_components(n_grains=8)
    assembler = PolycrystalAssembly(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
    )
    _, _, _, _, counts = assembler.assemble(verbose=False)
    assert np.all(counts >= 1), f"Empty grain found: {counts}"


# ===================================================================
# 3. Voronoi truncation (nearest-seed property)
# ===================================================================

def test_atoms_closest_to_assigned_seed():
    """Every atom must be closer to its assigned grain seed (PBC-aware)."""
    grains, crystal, ori = _make_components(n_grains=6, seed=10)
    assembler = PolycrystalAssembly(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
    )
    pos, _, gid, _, _ = assembler.assemble(verbose=False)
    seeds = grains.seeds
    box_size = assembler.box_size

    idx = np.random.default_rng(0).choice(
        len(pos), size=min(500, len(pos)), replace=False
    )
    for j in idx:
        # PBC-aware minimum-image distance
        delta = seeds - pos[j]
        delta -= box_size * np.round(delta / box_size)
        dists = np.linalg.norm(delta, axis=1)
        assert np.argmin(dists) == gid[j], (
            f"Atom {j} (grain {gid[j]}) closest to seed {np.argmin(dists)}"
        )


def test_no_duplicate_atoms():
    """Each atom coordinate should be unique (no boundary duplicates)."""
    grains, crystal, ori = _make_components(n_grains=4)
    assembler = PolycrystalAssembly(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
    )
    pos, _, _, _, _ = assembler.assemble(verbose=False)

    rounded = np.round(pos, decimals=6)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    assert len(unique_idx) == len(pos), (
        f"{len(pos) - len(unique_idx)} duplicate atoms found"
    )


# ===================================================================
# 4. Grain ID and Euler angle consistency
# ===================================================================

def test_grain_ids_in_range():
    grains, crystal, ori = _make_components(n_grains=5)
    assembler = PolycrystalAssembly(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
    )
    _, _, gid, _, _ = assembler.assemble(verbose=False)
    assert gid.min() >= 0
    assert gid.max() < 5


def test_euler_per_atom_matches_grain():
    """All atoms in a grain share the grain's Euler angles."""
    grains, crystal, ori = _make_components(n_grains=3, seed=20)
    assembler = PolycrystalAssembly(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
    )
    _, _, gid, euler, _ = assembler.assemble(verbose=False)

    for i in range(3):
        mask = gid == i
        if mask.sum() > 0:
            grain_euler = euler[mask]
            expected = ori.euler_angles[i]
            assert np.allclose(grain_euler, expected, atol=1e-12), (
                f"Grain {i}: Euler mismatch"
            )


# ===================================================================
# 5. AssemblyResult
# ===================================================================

def test_assembly_result_fields():
    grains, crystal, ori = _make_components(n_grains=4)
    result = assemble_polycrystal(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
        verbose=False,
    )
    assert isinstance(result, AssemblyResult)
    assert result.n_atoms == len(result.positions)
    assert result.n_grains == 4
    assert len(result.symbols) >= 1
    assert len(result.type_to_symbol) == len(result.symbols)
    assert len(result.type_masses) == len(result.symbols)
    assert len(result.keep_counts) == 4
    assert result.keep_counts.sum() == result.n_atoms
    assert np.allclose(result.box_start, BOX_S_NP)
    assert np.allclose(result.box_end, BOX_E_NP)


# ===================================================================
# 6. LAMMPS file output
# ===================================================================

def test_write_lammps_data():
    grains, crystal, ori = _make_components(n_grains=4)
    assembler = PolycrystalAssembly(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
    )
    pos, typ, gid, euler, _ = assembler.assemble(verbose=False)
    assembler._positions = pos
    assembler._types = typ
    assembler._grain_ids = gid
    assembler._euler_per_atom = euler
    assembler._n_total = len(pos)

    with tempfile.NamedTemporaryFile(suffix=".data", delete=False) as f:
        tmp = f.name
    try:
        assembler.write_lammps_data(tmp)
        with open(tmp) as fh:
            content = fh.read()
        assert "LAMMPS data file via POLY" in content
        assert f"{len(pos)} atoms" in content
        assert "Masses" in content
        assert "Atoms" in content
        assert "xlo xhi" in content
        # Count atom data lines (id type x y z format)
        in_atoms = False
        atom_count = 0
        for line in content.split("\n"):
            line = line.strip()
            if line == "Atoms":
                in_atoms = True
                continue
            if in_atoms and line:
                parts = line.split()
                if len(parts) == 5 and parts[0].isdigit():
                    atom_count += 1
        assert atom_count == len(pos), f"{atom_count} != {len(pos)}"
    finally:
        os.unlink(tmp)


def test_write_lammps_dump():
    grains, crystal, ori = _make_components(n_grains=3)
    assembler = PolycrystalAssembly(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
    )
    pos, typ, gid, euler, _ = assembler.assemble(verbose=False)
    assembler._positions = pos
    assembler._types = typ
    assembler._grain_ids = gid
    assembler._euler_per_atom = euler
    assembler._n_total = len(pos)

    with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as f:
        tmp = f.name
    try:
        assembler.write_lammps_dump(tmp)
        with open(tmp) as fh:
            content = fh.read()
        assert "ITEM: TIMESTEP" in content
        assert "ITEM: NUMBER OF ATOMS" in content
        assert f"{len(pos)}" in content
        assert "ITEM: BOX BOUNDS pp pp pp" in content
        assert "ITEM: ATOMS id_POLY type x y z grain_id euler_angle_1 euler_angle_2 euler_angle_3" in content
        # Verify grain IDs are 1-based
        for line in content.split("\n")[-10:-1]:
            if line.strip() and line[0].isdigit():
                parts = line.split()
                gid_val = int(parts[5])
                assert gid_val >= 1 and gid_val <= 3, f"Grain ID {gid_val} out of range"
    finally:
        os.unlink(tmp)


def test_write_without_assemble_raises():
    grains, crystal, ori = _make_components(n_grains=3)
    assembler = PolycrystalAssembly(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
    )
    try:
        assembler.write_lammps_data("nowhere.data")
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


def test_run_with_file_output():
    grains, crystal, ori = _make_components(n_grains=3)
    with tempfile.NamedTemporaryFile(suffix=".data", delete=False) as f:
        tmp_data = f.name
    with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as f:
        tmp_dump = f.name
    try:
        result = assemble_polycrystal(
            seeds=grains.seeds, crystal_atoms=crystal.atoms,
            orientations=ori, box_start=BOX_S, box_end=BOX_E,
            data_path=tmp_data, dump_path=tmp_dump,
            verbose=False,
        )
        assert os.path.getsize(tmp_data) > 0
        assert os.path.getsize(tmp_dump) > 0
    finally:
        os.unlink(tmp_data)
        os.unlink(tmp_dump)


# ===================================================================
# 7. Multi-species
# ===================================================================

def test_multispecies_crystal():
    """Assembly should handle multi-element crystals (e.g. NaCl)."""
    grains, crystal, ori = _make_components(
        n_grains=3,
        crystal_args=("NaCl", "rocksalt", 5.64),
        seed=30,
    )
    result = assemble_polycrystal(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
        verbose=False,
    )
    assert len(result.symbols) == 2
    assert set(result.symbols) == {"Cl", "Na"}
    assert result.types.min() >= 1
    assert result.types.max() <= 2
    # Both types should be present
    assert 1 in result.types
    assert 2 in result.types


# ===================================================================
# 8. Integration with orientation modes
# ===================================================================

def test_low_angle_assembly():
    grains, crystal, ori = _make_components(
        n_grains=6, ori_mode="low_angle", seed=40,
    )
    result = assemble_polycrystal(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
        verbose=False,
    )
    assert result.n_atoms > 0


def test_z_alignment_assembly():
    grains, crystal, _ = _make_components(n_grains=4, seed=50)
    ori = generate_orientations(
        "z_alignment", n_grains=grains.n_grains,
        hkl=(1, 0, 0), random_seed=51, verbose=False,
    )
    result = assemble_polycrystal(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
        verbose=False,
    )
    assert result.n_atoms > 0


# ===================================================================
# 9. Edge cases
# ===================================================================

def test_single_grain():
    grains, crystal, ori = _make_components(n_grains=1, seed=60)
    result = assemble_polycrystal(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
        verbose=False,
    )
    # Single grain: all atoms within box are kept (atoms outside are
    # periodic images and must be discarded, not wrapped).
    assert result.n_atoms > 0
    assert result.n_atoms <= len(crystal.atoms)
    assert np.all(result.positions >= BOX_S_NP)
    assert np.all(result.positions < BOX_E_NP)


def test_verbose_output():
    grains, crystal, ori = _make_components(n_grains=3)
    result = assemble_polycrystal(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
        verbose=True,
    )
    assert result.n_atoms > 0


def test_reproducibility():
    """Same inputs -> same output."""
    grains, crystal, ori = _make_components(n_grains=4, seed=70)

    r1 = assemble_polycrystal(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
        verbose=False,
    )
    r2 = assemble_polycrystal(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
        verbose=False,
    )
    assert r1.n_atoms == r2.n_atoms
    assert np.allclose(r1.positions, r2.positions)
    assert np.array_equal(r1.types, r2.types)
    assert np.array_equal(r1.grain_ids, r2.grain_ids)


def test_convenience_function():
    grains, crystal, ori = _make_components(n_grains=3)
    result = assemble_polycrystal(
        seeds=grains.seeds, crystal_atoms=crystal.atoms,
        orientations=ori, box_start=BOX_S, box_end=BOX_E,
        verbose=False,
    )
    assert isinstance(result, AssemblyResult)
    assert result.n_atoms > 0


# ===================================================================
# 10. Full pipeline (Modules 1->2->3->5)
# ===================================================================

def test_full_pipeline():
    """End-to-end: seeds -> crystal -> orientations -> assembly."""
    grains = generate_grains(
        box_start=BOX_S, box_end=BOX_E,
        n_grains=8, distribution="random",
        random_seed=80, verbose=False,
    )
    crystal = generate_pristine_bravais(
        "Ni", "fcc", a=3.52,
        box_start=grains.box_start, box_end=grains.box_end,
        verbose=False,
    )
    ori = generate_orientations(
        "low_angle",
        n_grains=grains.n_grains,
        neighbors=grains.neighbors,
        random_seed=81, verbose=False,
    )
    result = assemble_polycrystal(
        seeds=grains.seeds,
        crystal_atoms=crystal.atoms,
        orientations=ori,
        box_start=grains.box_start,
        box_end=grains.box_end,
        verbose=False,
    )
    assert result.n_atoms > 0
    # Count grain IDs
    unique_grains = np.unique(result.grain_ids)
    assert len(unique_grains) == 8
    assert np.allclose(result.box_start, grains.box_start)
    assert np.allclose(result.box_end, grains.box_end)


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
