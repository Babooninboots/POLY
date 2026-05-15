"""
Tests for Module 3 -- orientation.py

Run:
    python test_orientation.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grain_seeds import generate_grains
from orientation import (
    OrientationAssigner,
    OrientationResult,
    generate_orientations,
)
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_neighbors(n_grains=12):
    """Generate a small set of grains with Voronoi neighbors."""
    grains = generate_grains(
        box_start=(0, 0, 0),
        box_end=(100, 100, 100),
        n_grains=n_grains,
        distribution="random",
        random_seed=42,
        verbose=False,
    )
    return grains.neighbors, grains.n_grains


# ===================================================================
# 1. Constructor validation
# ===================================================================

def test_constructor_rejects_bad_mode():
    try:
        OrientationAssigner("bogus", n_grains=10)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_low_angle_requires_neighbors():
    try:
        OrientationAssigner("low_angle", n_grains=10)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_high_angle_requires_neighbors():
    try:
        OrientationAssigner("high_angle", n_grains=10)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_custom_misorientation_requires_neighbors():
    try:
        OrientationAssigner("custom_misorientation", n_grains=10)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ===================================================================
# 2. Random mode
# ===================================================================

def test_random_mode_shape():
    assigner = OrientationAssigner("random", n_grains=20, random_seed=1)
    result = assigner.run(verbose=False)
    assert result.euler_angles.shape == (20, 3)
    assert result.rotation_matrices.shape == (20, 3, 3)
    assert np.allclose(np.linalg.det(result.rotation_matrices), 1.0)


def test_random_mode_reproducible():
    r1 = generate_orientations("random", n_grains=15, random_seed=42, verbose=False)
    r2 = generate_orientations("random", n_grains=15, random_seed=42, verbose=False)
    assert np.allclose(r1.euler_angles, r2.euler_angles)


def test_random_mode_covers_sphere():
    """A large number of random orientations should span the full Euler range."""
    r = generate_orientations("random", n_grains=100, random_seed=3, verbose=False)
    # theta should cover [0, 180] roughly
    theta = r.euler_angles[:, 1]
    assert np.min(theta) < 5
    assert np.max(theta) > 170


# ===================================================================
# 3. Z-axis alignment
# ===================================================================

def test_z_alignment_hkl_111():
    result = generate_orientations(
        "z_alignment", n_grains=10, random_seed=5, hkl=(1, 1, 1), verbose=False
    )
    z = np.array([0.0, 0.0, 1.0])
    n_111 = np.array([1, 1, 1], dtype=float)
    n_111 /= np.linalg.norm(n_111)

    # Each grain's rotation should map the hkl normal to z
    for i in range(10):
        R = Rotation.from_matrix(result.rotation_matrices[i])
        rotated = R.apply(n_111)
        assert np.allclose(rotated, z, atol=1e-9), f"Grain {i}: {rotated} != {z}"


def test_z_alignment_hkl_100():
    result = generate_orientations(
        "z_alignment", n_grains=5, random_seed=6, hkl=(1, 0, 0), verbose=False
    )
    z = np.array([0.0, 0.0, 1.0])
    n_100 = np.array([1.0, 0.0, 0.0])

    for i in range(5):
        R = Rotation.from_matrix(result.rotation_matrices[i])
        rotated = R.apply(n_100)
        assert np.allclose(rotated, z, atol=1e-9)


def test_z_alignment_hkl_110():
    result = generate_orientations(
        "z_alignment", n_grains=3, random_seed=7, hkl=(1, 1, 0), verbose=False
    )
    z = np.array([0.0, 0.0, 1.0])
    n_110 = np.array([1, 1, 0], dtype=float)
    n_110 /= np.linalg.norm(n_110)

    for i in range(3):
        R = Rotation.from_matrix(result.rotation_matrices[i])
        rotated = R.apply(n_110)
        assert np.allclose(rotated, z, atol=1e-9)


def test_z_alignment_in_plane_varies():
    """The random in-plane rotation should differ across grains."""
    result = generate_orientations(
        "z_alignment", n_grains=20, random_seed=8, hkl=(1, 1, 1), verbose=False
    )
    # phi2 (rotation about Z) should vary
    phi2 = result.euler_angles[:, 2]
    assert np.std(phi2) > 0.1, f"phi2 std = {np.std(phi2):.4f}"


def test_z_alignment_requires_hkl():
    assigner = OrientationAssigner("z_alignment", n_grains=10)
    try:
        assigner.run(verbose=False)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ===================================================================
# 4. Low angle (BFS, < 10 deg)
# ===================================================================

def test_low_angle_constraint_satisfied():
    nbrs, n = _make_neighbors(20)
    assigner = OrientationAssigner("low_angle", n_grains=n, neighbors=nbrs, random_seed=9)
    result = assigner.run(verbose=False)

    assert result.misorientation_angles is not None
    assert len(result.misorientation_angles) > 0
    max_miso = np.max(result.misorientation_angles)
    assert max_miso <= 10.0, f"Max misorientation = {max_miso:.2f} > 10 deg"


def test_low_angle_all_grains_assigned():
    nbrs, n = _make_neighbors(15)
    assigner = OrientationAssigner("low_angle", n_grains=n, neighbors=nbrs, random_seed=10)
    result = assigner.run(verbose=False)
    assert not np.any(np.isnan(result.euler_angles))
    assert not np.any(np.isnan(result.rotation_matrices))


def test_low_angle_rotations_are_valid():
    nbrs, n = _make_neighbors(12)
    assigner = OrientationAssigner("low_angle", n_grains=n, neighbors=nbrs, random_seed=11)
    result = assigner.run(verbose=False)
    dets = np.linalg.det(result.rotation_matrices)
    assert np.allclose(dets, 1.0, atol=1e-9)


# ===================================================================
# 5. High angle (BFS, > 20 deg)
# ===================================================================

def test_high_angle_constraint_satisfied():
    nbrs, n = _make_neighbors(12)
    assigner = OrientationAssigner("high_angle", n_grains=n, neighbors=nbrs, random_seed=12)
    result = assigner.run(verbose=False)

    assert result.misorientation_angles is not None
    assert len(result.misorientation_angles) > 0
    min_miso = np.min(result.misorientation_angles)
    assert min_miso >= 20.0, f"Min misorientation = {min_miso:.2f} < 20 deg"


def test_high_angle_all_grains_assigned():
    nbrs, n = _make_neighbors(15)
    assigner = OrientationAssigner("high_angle", n_grains=n, neighbors=nbrs, random_seed=13)
    result = assigner.run(verbose=False)
    assert not np.any(np.isnan(result.euler_angles))


# ===================================================================
# 6. Custom misorientation (MC optimization)
# ===================================================================

def test_custom_misorientation_runs():
    nbrs, n = _make_neighbors(10)
    # Target: unimodal around 30 degrees
    target = np.random.default_rng(0).normal(30.0, 5.0, size=500)
    target = np.abs(target)  # angles must be >= 0

    assigner = OrientationAssigner(
        "custom_misorientation", n_grains=n, neighbors=nbrs, random_seed=14
    )
    result = assigner.run(
        target_angles=target,
        max_steps=200,
        verbose=False,
    )
    assert result.euler_angles.shape == (n, 3)
    assert result.mc_energy_history is not None
    assert len(result.mc_energy_history) >= 2
    assert result.mc_acceptance_rate is not None
    assert 0.0 <= result.mc_acceptance_rate <= 1.0


def test_custom_misorientation_energy_decreases():
    """Over a modest run, final energy should not be worse than start."""
    nbrs, n = _make_neighbors(8)
    target = np.random.default_rng(1).normal(45.0, 8.0, size=300)
    target = np.abs(target)

    assigner = OrientationAssigner(
        "custom_misorientation", n_grains=n, neighbors=nbrs, random_seed=15
    )
    result = assigner.run(
        target_angles=target,
        max_steps=300,
        cooling_rate=0.97,
        verbose=False,
    )
    hist = result.mc_energy_history
    assert hist[-1] <= hist[0] + 0.05, (
        f"Energy rose from {hist[0]:.4f} to {hist[-1]:.4f}"
    )


def test_custom_misorientation_requires_target():
    nbrs, n = _make_neighbors(10)
    assigner = OrientationAssigner(
        "custom_misorientation", n_grains=n, neighbors=nbrs, random_seed=16
    )
    try:
        assigner.run(verbose=False)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ===================================================================
# 7. Custom orientation profile
# ===================================================================

def test_custom_profile_explicit():
    euler_map = {
        0: (0.0, 0.0, 0.0),
        1: (45.0, 90.0, 30.0),
        2: (120.0, 60.0, 15.0),
    }
    assigner = OrientationAssigner("custom_profile", n_grains=5, random_seed=17)
    result = assigner.run(euler_map=euler_map, verbose=False)

    for i, (phi1, theta, phi2) in euler_map.items():
        R_expected = Rotation.from_euler("zxz", [phi1, theta, phi2], degrees=True)
        R_actual = Rotation.from_matrix(result.rotation_matrices[i])
        # Check via the difference rotation angle
        diff = (R_actual * R_expected.inv()).magnitude()
        assert np.degrees(diff) < 1e-6, f"Grain {i} mismatch: diff = {np.degrees(diff):.2e}"


def test_custom_profile_unmapped_grains_get_identity():
    euler_map = {0: (10.0, 20.0, 30.0)}
    assigner = OrientationAssigner("custom_profile", n_grains=4, random_seed=18)
    result = assigner.run(euler_map=euler_map, verbose=False)

    # Unmapped grains should be identity (zero rotation)
    for i in [1, 2, 3]:
        R = Rotation.from_matrix(result.rotation_matrices[i])
        angle = np.degrees(R.magnitude())
        assert angle < 1e-9, f"Grain {i}: expected identity, got angle={angle:.2e}"


def test_custom_profile_requires_map():
    assigner = OrientationAssigner("custom_profile", n_grains=5)
    try:
        assigner.run(verbose=False)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ===================================================================
# 8. OrientationResult properties
# ===================================================================

def test_result_fields_random():
    result = generate_orientations("random", n_grains=8, random_seed=19, verbose=False)
    assert isinstance(result, OrientationResult)
    assert result.mode == "random"
    assert result.mc_energy_history is None
    assert result.mc_acceptance_rate is None


def test_result_contains_rotations_object():
    result = generate_orientations("random", n_grains=5, random_seed=20, verbose=False)
    assert isinstance(result.rotations, Rotation)
    assert len(result.rotations) == 5


def test_result_neighbors_preserved():
    nbrs, n = _make_neighbors(10)
    assigner = OrientationAssigner("low_angle", n_grains=n, neighbors=nbrs, random_seed=21)
    result = assigner.run(verbose=False)
    assert result.neighbors == nbrs


# ===================================================================
# 9. Convenience function
# ===================================================================

def test_convenience_function_random():
    result = generate_orientations("random", n_grains=6, random_seed=22, verbose=False)
    assert result.euler_angles.shape == (6, 3)


def test_convenience_function_z_alignment():
    result = generate_orientations(
        "z_alignment", n_grains=4, random_seed=23, hkl=(1, 0, 0), verbose=False
    )
    z = np.array([0.0, 0.0, 1.0])
    n_100 = np.array([1.0, 0.0, 0.0])
    R = Rotation.from_matrix(result.rotation_matrices[0])
    assert np.allclose(R.apply(n_100), z, atol=1e-9)


# ===================================================================
# 10. Edge cases
# ===================================================================

def test_single_grain():
    """Single grain should work for all simple modes."""
    for mode in ["random", "z_alignment", "low_angle", "high_angle"]:
        nbrs = [[]]  # one grain, no neighbors
        assigner = OrientationAssigner(
            mode, n_grains=1, neighbors=nbrs, random_seed=24
        )
        kwargs = {}
        if mode == "z_alignment":
            kwargs["hkl"] = (1, 1, 1)
        result = assigner.run(verbose=False, **kwargs)
        assert result.euler_angles.shape == (1, 3), f"Failed for {mode}"
        det = np.linalg.det(result.rotation_matrices[0])
        assert np.allclose(det, 1.0), f"Det != 1 for {mode}"


def test_two_grains_low_angle():
    nbrs = [[1], [0]]
    assigner = OrientationAssigner(
        "low_angle", n_grains=2, neighbors=nbrs, random_seed=25
    )
    result = assigner.run(verbose=False)
    miso = result.misorientation_angles[0]
    assert miso <= 10.0, f"Miso = {miso:.2f} > 10"


def test_two_grains_high_angle():
    nbrs = [[1], [0]]
    assigner = OrientationAssigner(
        "high_angle", n_grains=2, neighbors=nbrs, random_seed=26
    )
    result = assigner.run(verbose=False)
    miso = result.misorientation_angles[0]
    assert miso >= 20.0, f"Miso = {miso:.2f} < 20"


def test_disconnected_components():
    """Grains with no edges should still get an orientation (handled by BFS)."""
    nbrs = [[], [], []]  # three isolated grains
    assigner = OrientationAssigner(
        "low_angle", n_grains=3, neighbors=nbrs, random_seed=27
    )
    result = assigner.run(verbose=False)
    assert result.euler_angles.shape == (3, 3)
    assert not np.any(np.isnan(result.euler_angles))


def test_verbose_output():
    """Verbose mode should run without error (output goes to stdout)."""
    result = generate_orientations("random", n_grains=5, random_seed=28, verbose=True)
    assert result.euler_angles.shape == (5, 3)


# ===================================================================
# 11. Integration with grain_seeds module
# ===================================================================

def test_integration_seeds_to_orientations():
    """Full Module 1 + Module 3 pipeline."""
    grains = generate_grains(
        box_start=(0, 0, 0),
        box_end=(50, 50, 50),
        n_grains=15,
        distribution="random",
        random_seed=29,
        verbose=False,
    )
    assert grains.neighbors is not None
    assert len(grains.neighbors) == 15

    # Low angle using the real neighbor list
    result = generate_orientations(
        "low_angle",
        n_grains=grains.n_grains,
        neighbors=grains.neighbors,
        random_seed=30,
        verbose=False,
    )
    assert result.euler_angles.shape == (15, 3)
    assert result.misorientation_angles is not None
    max_miso = np.max(result.misorientation_angles)
    assert max_miso <= 10.0, f"Max miso = {max_miso:.2f}"


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
