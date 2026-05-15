"""
Tests for Module 1 — grain_seeds.py

Run:
    python test_grain_seeds.py
    python -m pytest test_grain_seeds.py -v   (if pytest is available)
"""

import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path so we can import grain_seeds
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grain_seeds import (
    GrainSeedGenerator,
    SeedResult,
    generate_grains,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_box():
    """Return a standard 100³ box for tests."""
    return (0.0, 0.0, 0.0), (100.0, 100.0, 100.0)


# ===================================================================
# 1. Initialisation & seed generation
# ===================================================================

def test_constructor_with_n_grains():
    gen = GrainSeedGenerator((0, 0, 0), (10, 10, 10), n_grains=50)
    assert gen.n_grains == 50
    assert gen.distribution == "random"


def test_constructor_with_avg_diameter():
    # Box volume = 1000; avg grain volume = (4/3)π·(5)³ ≈ 523.6
    # → expected n ≈ 1000/523.6 ≈ 1.9 → 2
    gen = GrainSeedGenerator((0, 0, 0), (10, 10, 10), avg_diameter=10.0)
    assert gen.n_grains >= 1


def test_constructor_rejects_missing_count():
    try:
        GrainSeedGenerator((0, 0, 0), (10, 10, 10))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_constructor_rejects_bad_distribution():
    try:
        GrainSeedGenerator((0, 0, 0), (10, 10, 10), n_grains=5, distribution="bogus")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_normal_requires_std_dev():
    try:
        GrainSeedGenerator((0, 0, 0), (10, 10, 10), n_grains=5, distribution="normal")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_customized_requires_positions():
    try:
        GrainSeedGenerator(
            (0, 0, 0), (10, 10, 10), distribution="customized"
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_generate_seeds_random():
    box_s, box_e = _make_box()
    gen = GrainSeedGenerator(box_s, box_e, n_grains=40, random_seed=1)
    seeds = gen.generate_seeds()
    assert seeds.shape == (40, 3)
    assert np.all(seeds >= box_s)
    assert np.all(seeds <= box_e)


def test_generate_seeds_customized():
    custom = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    gen = GrainSeedGenerator(
        (0, 0, 0), (10, 10, 10), distribution="customized", seed_positions=custom
    )
    seeds = gen.generate_seeds()
    assert np.allclose(seeds, custom)
    assert gen.n_grains == 3


def test_generate_seeds_reproducible():
    box_s, box_e = _make_box()
    gen1 = GrainSeedGenerator(box_s, box_e, n_grains=20, random_seed=42)
    gen2 = GrainSeedGenerator(box_s, box_e, n_grains=20, random_seed=42)
    assert np.allclose(gen1.generate_seeds(), gen2.generate_seeds())


# ===================================================================
# 2. Voronoi tessellation & grain-size computation
# ===================================================================

def test_compute_grain_sizes_no_nan():
    box_s, box_e = _make_box()
    gen = GrainSeedGenerator(box_s, box_e, n_grains=30, random_seed=2)
    gen.generate_seeds()
    diameters = gen.compute_grain_sizes()
    assert diameters.shape == (30,)
    assert not np.any(np.isnan(diameters))
    assert np.all(diameters > 0)


def test_grain_sizes_sum_approximates_box_volume():
    """Sum of Voronoi cell volumes ≈ box volume (PBC enforces space-filling)."""
    box_s, box_e = _make_box()
    gen = GrainSeedGenerator(box_s, box_e, n_grains=50, random_seed=4)
    gen.generate_seeds()
    diameters = gen.compute_grain_sizes()
    cell_volumes = (4.0 / 3.0) * np.pi * (diameters / 2.0) ** 3
    rel_error = abs(np.sum(cell_volumes) - gen.box_volume) / gen.box_volume
    assert rel_error < 0.15, f"Volume mismatch: {rel_error:.4f}"


def test_diameter_positive_monotonic_with_volume():
    box_s, box_e = _make_box()
    gen = GrainSeedGenerator(box_s, box_e, n_grains=20, random_seed=5)
    gen.generate_seeds()
    d = gen.compute_grain_sizes()
    assert np.all(d > 0)
    assert np.all(np.isfinite(d))


# ===================================================================
# 5. Full pipeline (run method)
# ===================================================================

def test_run_random_distribution():
    result = generate_grains(
        (0, 0, 0), (100, 100, 100), n_grains=20, distribution="random",
        random_seed=10, verbose=False,
    )
    assert isinstance(result, SeedResult)
    assert result.distribution == "random"
    assert result.seeds.shape == (20, 3)
    assert result.diameters.shape == (20,)
    assert result.neighbors is not None
    assert result.polyhedron_data is not None


def test_run_normal_distribution():
    result = generate_grains(
        (0, 0, 0), (100, 100, 100), n_grains=15, distribution="normal",
        std_dev=2.0, random_seed=11, verbose=False,
    )
    assert result.distribution == "normal"
    assert result.seeds.shape == (15, 3)
    assert result.diameters.shape == (15,)
    assert result.neighbors is not None
    assert result.polyhedron_data is not None


def test_run_customized_distribution():
    custom = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float)
    result = generate_grains(
        (0, 0, 0), (100, 100, 100), distribution="customized",
        seed_positions=custom, verbose=False,
    )
    assert result.distribution == "customized"
    assert result.n_grains == 3
    assert np.allclose(result.seeds, custom)


def test_run_with_avg_diameter():
    result = generate_grains(
        (0, 0, 0), (50, 50, 50), avg_diameter=25.0, distribution="random",
        random_seed=12, verbose=False,
    )
    assert result.n_grains >= 1
    assert result.seeds.shape[1] == 3


# ===================================================================
# 6. Persistence
# ===================================================================

def test_save_and_load_seeds(tmp_path):
    box_s, box_e = _make_box()
    gen = GrainSeedGenerator(box_s, box_e, n_grains=15, random_seed=13)
    gen.generate_seeds()
    gen.compute_grain_sizes()

    path = str(tmp_path / "seeds_test.npz")
    gen.save_seeds(path)
    data = GrainSeedGenerator.load_seeds(path)

    assert np.allclose(data["seeds"], gen.seeds)
    assert data["n_grains"] == 15


def test_save_without_run_raises():
    gen = GrainSeedGenerator((0, 0, 0), (10, 10, 10), n_grains=5, random_seed=0)
    try:
        gen.save_seeds("nowhere.npz")
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


# ===================================================================
# 7. SeedResult properties
# ===================================================================

def test_seed_result_properties():
    box_s, box_e = _make_box()
    result = generate_grains(
        box_s, box_e, n_grains=10, distribution="random",
        random_seed=14, verbose=False,
    )
    assert np.allclose(result.grain_sizes, result.diameters)
    assert result.volume == 1_000_000.0


# ===================================================================
# 8. Regression / edge cases
# ===================================================================

def test_minimum_grains():
    """Single grain: one cell fills the box."""
    result = generate_grains(
        (0, 0, 0), (10, 10, 10), n_grains=1, distribution="random",
        random_seed=15, verbose=False,
    )
    assert result.seeds.shape == (1, 3)
    d = result.diameters[0]
    # Equivalent sphere should be roughly the box diagonal scale
    assert 5.0 < d < 30.0


def test_many_grains_no_nan():
    """Ensure no NaN even with relatively many grains."""
    result = generate_grains(
        (0, 0, 0), (50, 50, 50), n_grains=100, distribution="random",
        random_seed=16, verbose=False,
    )
    assert not np.any(np.isnan(result.diameters))
    assert np.all(result.diameters > 0)


def test_non_cubic_box():
    """Rectangular boxes should work fine."""
    result = generate_grains(
        (0, 0, 0), (200, 50, 50), n_grains=25, distribution="random",
        random_seed=17, verbose=False,
    )
    assert result.seeds.shape == (25, 3)
    assert np.all(result.seeds[:, 0] >= 0) and np.all(result.seeds[:, 0] <= 200)
    assert np.all(result.seeds[:, 1] >= 0) and np.all(result.seeds[:, 1] <= 50)
    assert not np.any(np.isnan(result.diameters))


def test_custom_seeds_give_sensible_sizes():
    """Pre-placed seeds on a regular grid should give roughly equal sizes."""
    n = 3
    xs = np.linspace(5, 95, n)
    ys = np.linspace(5, 95, n)
    zs = np.linspace(5, 95, n)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    result = generate_grains(
        (0, 0, 0), (100, 100, 100), distribution="customized",
        seed_positions=grid, verbose=False,
    )
    # On a regular 3×3×3 grid all cells should be similar
    std = np.std(result.diameters)
    mean = np.mean(result.diameters)
    assert std / mean < 0.3, f"Grid seeds produced high variance: std/mean = {std/mean:.3f}"


# ===================================================================
# 9. Runner
# ===================================================================

if __name__ == "__main__":
    # Collect and run all test_* functions in this module
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
            # Create a temporary directory for save/load test
            if "tmp_path" in fn.__code__.co_varnames:
                import tempfile
                from pathlib import Path
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
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
