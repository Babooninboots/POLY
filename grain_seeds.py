"""
Module 1: Grain Seed Initialization & Voronoi Tessellation.

Generates 3D grain seed positions and computes grain size distributions
via Voronoi tessellation with periodic boundary conditions (PBC).

For 'normal' grain-size distribution targets, a Force-Biased Sphere
Packing algorithm relaxes seed positions to match the specified
standard deviation.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyvoro2 as pv


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SeedResult:
    """Container returned by GrainSeedGenerator.run()."""

    seeds: np.ndarray          # (N, 3) final seed coordinates
    diameters: np.ndarray      # (N,) equivalent-sphere diameters
    distribution: str          # "random" | "normal" | "customized"
    n_grains: int
    box_start: np.ndarray
    box_end: np.ndarray
    neighbors: list[list[int]] | None = None  # Voronoi adjacency (populated by run())
    polyhedron_data: list[tuple[int, np.ndarray, int]] | None = None  # (grain_id, vertices, image_index) — all 27 PBC images
    target_radii: np.ndarray | None = None  # Laguerre-Voronoi radii (populated by FBSP)

    @property
    def grain_sizes(self) -> np.ndarray:
        """Alias for diameters (equivalent-sphere diameter per grain)."""
        return self.diameters

    @property
    def volume(self) -> float:
        return float(np.prod(self.box_end - self.box_start))


# ---------------------------------------------------------------------------
# Force-Biased Sphere Packing
# ---------------------------------------------------------------------------

class ForceBiasedPacker:
    """Force-Biased Sphere Packing for matching a target grain-size distribution.

    Assigns each grain a target radius sampled from N(target_mean, target_std),
    scales total size (3D volume or 2D area) to fill the domain, then
    iteratively pushes overlapping spheres apart using KDTree neighbour
    search with PBC wrapping on the active axes.

    Parameters
    ----------
    box_start, box_end : np.ndarray
        Simulation box bounds.
    n_grains : int
        Number of grains.
    target_mean : float
        Target mean grain diameter.
    target_std : float
        Target standard deviation of grain diameters.
    laminate_direction : str or None
        ``"x"``, ``"y"``, ``"z"`` for 2D in-plane packing; ``None`` for 3D.
    rng : np.random.Generator or None
        Random number generator.
    """

    def __init__(
        self,
        box_start: np.ndarray,
        box_end: np.ndarray,
        n_grains: int,
        target_mean: float,
        target_std: float,
        laminate_direction: str | None = None,
        rng: np.random.Generator | None = None,
        bimodal_params: tuple | None = None,
    ):
        self.box_start = box_start
        self.box_end = box_end
        self.box_size = box_end - box_start
        self.n_grains = n_grains
        self.target_mean = target_mean
        self.target_std = target_std
        self.rng = rng or np.random.default_rng()
        self.bimodal_params = bimodal_params  # (frac, m1, s1, m2, s2) or None

        self.laminate_direction = laminate_direction
        if laminate_direction is not None:
            axis_map = {"x": 0, "y": 1, "z": 2}
            self.locked_axis = axis_map[laminate_direction]
            self.active_axes = [i for i in range(3) if i != self.locked_axis]
            self.active_box_size = self.box_size[self.active_axes]
            self.plane_area = float(np.prod(self.active_box_size))
        else:
            self.locked_axis = None
            self.active_axes = [0, 1, 2]
            self.active_box_size = self.box_size

    def run(
        self,
        initial_seeds: np.ndarray,
        progress_callback: Callable[[int, int, float], None] | None = None,
        pause_callback: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run FBSP relaxation and return final seed positions and target radii.

        Parameters
        ----------
        initial_seeds : (N, 3) array
        progress_callback : callable or None
            ``progress_callback(step, max_steps, max_overlap)`` every 10 steps
            and on early stop.
        pause_callback : callable or None
            ``pause_callback(current_seeds, target_diameters) -> bool`` at the
            start of every step.  Return ``True`` to stop early.
        """
        from scipy.spatial import KDTree as FBSPKDTree

        is_2d = self.laminate_direction is not None
        ndim = 2 if is_2d else 3

        # 1. Sample target diameters
        if self.bimodal_params is not None:
            frac, m1, s1, m2, s2 = self.bimodal_params
            n1 = max(1, int(round(self.n_grains * frac)))
            n2 = self.n_grains - n1
            d1 = self.rng.normal(m1, s1, size=n1)
            d2 = self.rng.normal(m2, s2, size=n2)
            target_diameters = np.concatenate([d1, d2])
            self.rng.shuffle(target_diameters)
            target_diameters = np.clip(target_diameters, 0.1 * min(m1, m2), None)
        else:
            target_diameters = self.rng.normal(
                self.target_mean, self.target_std, size=self.n_grains
            )
            target_diameters = np.clip(target_diameters, 0.1 * self.target_mean, None)

        # 2. Compute target sizes and radii
        if is_2d:
            target_areas = np.pi * (target_diameters / 2.0) ** 2
            packing_fraction = 0.78  # 2D random close packing
            area_scale = (self.plane_area * packing_fraction) / np.sum(target_areas)
            target_areas *= area_scale
            target_radii = np.sqrt(target_areas / np.pi)
        else:
            target_volumes = (4.0 / 3.0) * np.pi * (target_diameters / 2.0) ** 3
            box_vol = float(np.prod(self.box_size))
            packing_fraction = 0.58  # 3D random close packing
            volume_scale = (box_vol * packing_fraction) / np.sum(target_volumes)
            target_volumes *= volume_scale
            target_radii = (3.0 * target_volumes / (4.0 * np.pi)) ** (1.0 / 3.0)

        # 3. Initialise positions
        positions = initial_seeds.copy().astype(float)

        # 4. Relaxation loop
        lr = 0.5
        lr_decay = 0.995
        max_iters = 2000
        threshold = 0.01

        for step in range(max_iters):
            # ---- pause / stop hook ----
            if pause_callback is not None:
                if pause_callback(positions, target_diameters):
                    break

            active_coords = positions[:, self.active_axes]
            tree = FBSPKDTree(active_coords, boxsize=self.active_box_size)

            cutoff = 2.0 * np.max(target_radii)
            pairs = tree.query_pairs(r=cutoff, output_type="ndarray")

            if len(pairs) == 0:
                break

            forces = np.zeros((self.n_grains, ndim))
            max_overlap = 0.0

            for i, j in pairs:
                ri = target_radii[i]
                rj = target_radii[j]

                # PBC-aware separation on active axes
                delta = active_coords[j] - active_coords[i]
                delta = delta - self.active_box_size * np.round(
                    delta / self.active_box_size
                )

                dist = np.linalg.norm(delta)
                if dist < 1e-12:
                    delta = self.rng.normal(0, 1e-6, size=ndim)
                    dist = np.linalg.norm(delta)
                    if dist < 1e-12:
                        continue

                overlap = (ri + rj) - dist
                if overlap > 0:
                    max_overlap = max(max_overlap, overlap)
                    direction = delta / dist
                    total_r = ri + rj
                    forces[i] -= direction * overlap * lr * (rj / total_r)
                    forces[j] += direction * overlap * lr * (ri / total_r)

            positions[:, self.active_axes] += forces

            # PBC wrapping on active axes
            positions[:, self.active_axes] = (
                self.box_start[self.active_axes]
                + (positions[:, self.active_axes] - self.box_start[self.active_axes])
                % self.active_box_size
            )

            lr *= lr_decay

            # ---- progress callback (every 10 steps) ----
            if progress_callback is not None and step % 10 == 0:
                progress_callback(step, max_iters, max_overlap)

            if max_overlap < threshold:
                if progress_callback is not None:
                    progress_callback(max_iters, max_iters, max_overlap)
                break
        else:
            # Loop exhausted without early stop — fire final 100% callback
            if progress_callback is not None:
                progress_callback(max_iters, max_iters, max_overlap)

        return positions, target_radii


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class GrainSeedGenerator:
    """Generate and optionally optimize grain seed positions.

    Parameters
    ----------
    box_start, box_end : array-like (3,)
        Simulation box bounds.
    n_grains : int, optional
        Exact number of grains.
    avg_diameter : float, optional
        Approximate average grain diameter; *n_grains* is estimated from
        the box volume.
    distribution : str
        ``"random"``, ``"normal"``, or ``"customized"``.
    std_dev : float, optional
        Target standard deviation (required for ``distribution="normal"``).
    seed_positions : array-like (N, 3), optional
        Pre-defined seed positions (required for ``distribution="customized"``).
    random_seed : int, optional
        Seed for the numpy random generator.
    """

    def __init__(
        self,
        box_start: tuple[float, float, float] | list[float],
        box_end: tuple[float, float, float] | list[float],
        n_grains: int | None = None,
        avg_diameter: float | None = None,
        distribution: str = "random",
        std_dev: float | None = None,
        seed_positions: np.ndarray | None = None,
        seed_radii: np.ndarray | None = None,
        bimodal_params: tuple | None = None,
        random_seed: int | None = None,
        laminate_in_plane_dist: str = "random",
        laminate_direction: str = "z",
    ):
        # -- box geometry --
        self.box_start = np.asarray(box_start, dtype=float)
        self.box_end = np.asarray(box_end, dtype=float)
        self.box_size = self.box_end - self.box_start
        self.box_volume = float(np.prod(self.box_size))

        # -- distribution validation --
        if distribution not in ("random", "normal", "customized", "laminate", "even", "bimodal"):
            raise ValueError(
                f"distribution must be 'random', 'normal', 'customized', "
                f"'laminate', 'even', or 'bimodal', got '{distribution}'"
            )
        self.distribution = distribution
        self.std_dev = std_dev
        self._bimodal_params = bimodal_params
        self._custom_seeds: np.ndarray | None = None

        # -- laminate parameters --
        self.laminate_in_plane_dist = laminate_in_plane_dist
        self.laminate_direction = laminate_direction

        if distribution == "customized":
            if seed_positions is None:
                raise ValueError("seed_positions is required for 'customized' distribution.")
            self._custom_seeds = np.asarray(seed_positions, dtype=float)
            self.n_grains = len(self._custom_seeds)
        else:
            if distribution == "laminate":
                if laminate_in_plane_dist not in ("random", "normal"):
                    raise ValueError(
                        "laminate_in_plane_dist must be 'random' or 'normal', "
                        f"got '{laminate_in_plane_dist}'"
                    )
                if laminate_in_plane_dist == "normal" and std_dev is None:
                    raise ValueError(
                        "std_dev is required for 'laminate' with "
                        "'normal' in-plane distribution."
                    )
            elif distribution == "normal" and std_dev is None:
                raise ValueError("std_dev is required for 'normal' distribution.")

            if n_grains is not None:
                self.n_grains = int(n_grains)
            elif avg_diameter is not None:
                avg_vol = (4.0 / 3.0) * np.pi * (avg_diameter / 2.0) ** 3
                self.n_grains = max(1, int(np.rint(self.box_volume / avg_vol)))
            else:
                raise ValueError("Either n_grains or avg_diameter must be provided.")

        self.rng = np.random.default_rng(random_seed)
        self.seeds: np.ndarray | None = None
        self.target_radii: np.ndarray | None = None

        if seed_radii is not None:
            self.target_radii = np.asarray(seed_radii, dtype=float)

    # ------------------------------------------------------------------
    # Seed generation
    # ------------------------------------------------------------------

    def generate_seeds(self) -> np.ndarray:
        """Place seeds uniformly at random inside the simulation box."""
        if self._custom_seeds is not None:
            self.seeds = self._custom_seeds.copy()
        elif self.distribution == "laminate":
            self.seeds = self._generate_laminate_seeds()
        elif self.distribution == "even":
            axis_map = {"x": 0, "y": 1, "z": 2}
            n_axis = axis_map[self.laminate_direction]
            self.seeds = np.empty((self.n_grains, 3), dtype=float)
            for ax in range(3):
                self.seeds[:, ax] = self.box_start[ax] + self.box_size[ax] / 2.0
            spacing = self.box_size[n_axis] / self.n_grains
            self.seeds[:, n_axis] = self.box_start[n_axis] + spacing * (np.arange(self.n_grains) + 0.5)
        else:
            self.seeds = self.rng.uniform(
                low=self.box_start, high=self.box_end, size=(self.n_grains, 3)
            )
        self._enforce_min_separation()
        return self.seeds

    def _enforce_min_separation(self, min_dist: float = 10.0) -> None:
        """Jitter any seed that lies within *min_dist* of another seed.

        Iterates until all pairwise distances exceed the threshold (max 50
        passes).  For laminate the normal-axis coordinate is pinned to the
        mid-plane so seeds stay on the layer.
        """
        if self.seeds is None or self.n_grains < 2:
            return
        if self.distribution == "even":
            return  # evenly spaced seeds need no separation enforcement
        from scipy.spatial import KDTree

        n_axis: int | None = None
        mid_val: float | None = None
        if self.distribution == "laminate":
            n_axis = {"x": 0, "y": 1, "z": 2}[self.laminate_direction]
            mid_val = self.box_start[n_axis] + self.box_size[n_axis] / 2.0

        for _ in range(50):
            tree = KDTree(self.seeds)
            pairs = list(tree.query_pairs(r=min_dist))
            if not pairs:
                break
            moved: set[int] = set()
            for i, j in pairs:
                if i in moved or j in moved:
                    continue
                new_pos = self.rng.uniform(self.box_start, self.box_end)
                if n_axis is not None:
                    new_pos[n_axis] = mid_val
                self.seeds[j] = new_pos
                moved.add(j)

    # ------------------------------------------------------------------
    # Laminate seed generation (layered structure, PBC-commensurate)
    # ------------------------------------------------------------------

    def _generate_laminate_seeds(self) -> np.ndarray:
        """Generate seeds on a single mid-plane for 2D laminate Voronoi.

        All seeds share the same coordinate along the laminate normal axis
        (the box mid-plane).  In-plane coordinates are drawn uniformly.
        PBC along the normal axis is provided by the 27-image scheme, so
        the single layer tiles infinitely in that direction.
        """
        axis_map = {"x": 0, "y": 1, "z": 2}
        n_axis = axis_map[self.laminate_direction]

        mid_val = self.box_start[n_axis] + self.box_size[n_axis] / 2.0

        seeds = np.empty((self.n_grains, 3), dtype=float)
        seeds[:, n_axis] = mid_val

        other_axes = [ax for ax in range(3) if ax != n_axis]
        for ax in other_axes:
            seeds[:, ax] = self.rng.uniform(
                self.box_start[ax], self.box_end[ax], size=self.n_grains
            )

        return seeds

    # ------------------------------------------------------------------
    # Voronoi analysis
    # ------------------------------------------------------------------

    def _pyvoro_cells(self, seeds: np.ndarray) -> list[dict]:
        """Run pyvoro2 tessellation and return cell list."""
        domain = pv.OrthorhombicCell((
            (self.box_start[0], self.box_end[0]),
            (self.box_start[1], self.box_end[1]),
            (self.box_start[2], self.box_end[2]),
        ), periodic=(True, True, True))
        if self.target_radii is not None:
            return pv.compute(
                seeds, domain=domain, mode="power",
                radii=self.target_radii, include_empty=False,
            )
        return pv.compute(
            seeds, domain=domain, mode="standard", include_empty=False,
        )

    @staticmethod
    def _diameters_from_volumes(
        volumes: np.ndarray,
        distribution: str,
        box_size: np.ndarray,
        laminate_direction: str,
    ) -> np.ndarray:
        """Convert cell volumes to equivalent diameters, handling laminate."""
        if distribution == "laminate":
            axis_map = {"x": 0, "y": 1, "z": 2}
            thickness = box_size[axis_map[laminate_direction]]
            areas = volumes / thickness
            with np.errstate(invalid="ignore"):
                diameters = 2.0 * np.sqrt(areas / np.pi)
        else:
            with np.errstate(invalid="ignore"):
                diameters = 2.0 * (3.0 * volumes / (4.0 * np.pi)) ** (1.0 / 3.0)

        nan_mask = np.isnan(diameters)
        if nan_mask.any():
            diameters[nan_mask] = np.nanmean(diameters)
        return diameters

    def compute_grain_sizes(self, seeds: np.ndarray | None = None) -> np.ndarray:
        """Compute equivalent diameters via pyvoro2 Voronoi tessellation."""
        if seeds is None:
            seeds = self.seeds
        if seeds is None:
            raise RuntimeError("No seeds available; call generate_seeds() first.")

        cells = self._pyvoro_cells(seeds)
        volumes = np.array([c["volume"] for c in cells])
        return self._diameters_from_volumes(
            volumes, self.distribution, self.box_size, self.laminate_direction,
        )

    def compute_grain_cells(
        self, seeds: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[tuple[int, np.ndarray, int]]]:
        """Return (diameters, polyhedron_data) via pyvoro2 tessellation.

        polyhedron_data entries are (grain_id, vertices, image_index) where
        image_index runs 0..26, matching the 3×3×3 PBC shift grid iterated
        as ``for dx in (-1,0,1): for dy in (-1,0,1): for dz in (-1,0,1)``
        (index 13 = shift [0,0,0]).
        All 27 images per grain are returned so callers can select the most
        representative image via Delaunay neighbour count.
        """
        if seeds is None:
            seeds = self.seeds
        if seeds is None:
            raise RuntimeError("No seeds available; call generate_seeds() first.")

        cells = self._pyvoro_cells(seeds)
        volumes = np.array([c["volume"] for c in cells])
        diameters = self._diameters_from_volumes(
            volumes, self.distribution, self.box_size, self.laminate_direction,
        )

        box_size = self.box_size
        polyhedron_data: list[tuple[int, np.ndarray, int]] = []

        for cell in cells:
            verts = np.array(cell["vertices"])
            grain_id = cell["id"]

            idx = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        shift = np.array([dx, dy, dz], dtype=float) * box_size
                        shifted = verts + shift
                        polyhedron_data.append((grain_id, shifted, idx))
                        idx += 1

        return diameters, polyhedron_data

    # ------------------------------------------------------------------
    # Voronoi neighbor extraction
    # ------------------------------------------------------------------

    def get_voronoi_neighbors(self, seeds: np.ndarray | None = None) -> list[list[int]]:
        """Return the adjacency list of Voronoi neighbors via pyvoro2."""
        if seeds is None:
            seeds = self.seeds
        if seeds is None:
            raise RuntimeError("No seeds available; call generate_seeds() first.")

        n = len(seeds)
        cells = self._pyvoro_cells(seeds)

        neighbors: list[set[int]] = [set() for _ in range(n)]
        for cell in cells:
            for face in cell["faces"]:
                adj = face["adjacent_cell"]
                if adj >= 0:
                    neighbors[cell["id"]].add(adj)
                    neighbors[adj].add(cell["id"])

        return [sorted(s) for s in neighbors]

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        verbose: bool = True,
        progress_callback: Callable[[int, int, float], None] | None = None,
        pause_callback: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    ) -> SeedResult:
        """Execute the full Module-1 pipeline.

        Parameters
        ----------
        verbose : bool
            Print progress and final statistics to stdout.
        progress_callback : callable or None
            Forwarded to ForceBiasedPacker for progress reporting.
        pause_callback : callable or None
            Forwarded to ForceBiasedPacker for pause/stop support.
        """
        # 1. Generate or load seeds
        self.generate_seeds()
        if verbose:
            print(f"=== Module 1: Grain Seed Initialization ===")
            print(f"  Box:        {self.box_start} -> {self.box_end}")
            print(f"  Volume:     {self.box_volume:.4f}")
            print(f"  N grains:   {self.n_grains}")
            print(f"  Dist:       {self.distribution}", end="")
            if self.distribution == "normal":
                print(f" (sigma_target = {self.std_dev:.4f})", end="")
            elif self.distribution == "bimodal":
                if self._bimodal_params:
                    f, m1, s1, m2, s2 = self._bimodal_params
                    print(f" (GMM: f1={f:.2f}, m1={m1:.1f}, s1={s1:.1f}, "
                          f"m2={m2:.1f}, s2={s2:.1f})", end="")
            elif self.distribution == "laminate":
                print(f"  |  direction = {self.laminate_direction},  "
                      f"in-plane dist = {self.laminate_in_plane_dist}", end="")
            print()

        # 2. Compute initial grain sizes, polyhedron data, and Voronoi neighbors
        diameters, polyhedron_data = self.compute_grain_cells()
        neighbors = self.get_voronoi_neighbors()
        if verbose:
            print(f"  Initial  -  mu_d = {np.mean(diameters):.4f},  "
                  f"sigma_d = {np.std(diameters):.4f},  "
                  f"min = {np.min(diameters):.4f},  max = {np.max(diameters):.4f}")

        # 3. Optimize if requested (Force-Biased Sphere Packing)
        needs_packing = (
            self.distribution == "normal"
            or self.distribution == "bimodal"
            or (
                self.distribution == "laminate"
                and self.laminate_in_plane_dist == "normal"
            )
        )
        if needs_packing:
            is_laminate = self.distribution == "laminate"
            if is_laminate:
                # 2D: target mean from plane area
                plane_area = float(
                    np.prod(self.box_size[[
                        i for i in range(3)
                        if i != {"x": 0, "y": 1, "z": 2}[self.laminate_direction]
                    ]])
                )
                target_mean = 2.0 * np.sqrt(plane_area / (np.pi * self.n_grains))
                lam_dir = self.laminate_direction
            else:
                # 3D: target mean from box volume
                target_mean = 2.0 * (
                    3.0 * self.box_volume / (4.0 * np.pi * self.n_grains)
                ) ** (1.0 / 3.0)
                lam_dir = None

            packer = ForceBiasedPacker(
                self.box_start,
                self.box_end,
                self.n_grains,
                target_mean=target_mean,
                target_std=self.std_dev if self.std_dev else 0.0,
                laminate_direction=lam_dir,
                rng=self.rng,
                bimodal_params=self._bimodal_params,
            )
            if verbose:
                dim_label = "2D" if is_laminate else "3D"
                print(f"\n  Starting Force-Biased Sphere Packing ({dim_label}, max 2000 iters) ...")
            self.seeds, self.target_radii = packer.run(
                self.seeds,
                progress_callback=progress_callback,
                pause_callback=pause_callback,
            )
            diameters, polyhedron_data = self.compute_grain_cells()
            neighbors = self.get_voronoi_neighbors()
            # For bimodal, use target-radii-derived diameters so the
            # Gaussian-mixture signal is not washed out by the Voronoi
            # re-tessellation (which has only weak correlation with radii).
            if self.distribution == "bimodal":
                diameters = 2.0 * self.target_radii
            if verbose:
                print("  FBSP finished")
                print(
                    f"  Optimised -  mu_d = {np.mean(diameters):.4f},  "
                    f"sigma_d = {np.std(diameters):.4f},  "
                    f"min = {np.min(diameters):.4f},  max = {np.max(diameters):.4f}"
                )

        # 4. Print seed table (compact)
        if verbose:
            print(f"\n  --- Seed coordinates (first 10) ---")
            print(f"  {'idx':>4s}  {'x':>10s}  {'y':>10s}  {'z':>10s}  {'d_eq':>10s}")
            for i in range(min(10, self.n_grains)):
                s = self.seeds[i]
                print(f"  {i:4d}  {s[0]:10.4f}  {s[1]:10.4f}  {s[2]:10.4f}  {diameters[i]:10.4f}")
            if self.n_grains > 10:
                print(f"  ... ({self.n_grains - 10} more)")

            print(f"\n=== Module 1 complete ===\n")

        return SeedResult(
            seeds=self.seeds.copy(),
            diameters=diameters.copy(),
            distribution=self.distribution,
            n_grains=self.n_grains,
            box_start=self.box_start.copy(),
            box_end=self.box_end.copy(),
            neighbors=neighbors,
            polyhedron_data=polyhedron_data,
            target_radii=self.target_radii,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_seeds(self, filepath: str) -> None:
        """Save seed coordinates and diameters as a compressed .npz."""
        if self.seeds is None:
            raise RuntimeError("No seeds to save; call run() first.")
        diameters = self.compute_grain_sizes()
        np.savez_compressed(
            filepath,
            seeds=self.seeds,
            diameters=diameters,
            distribution=self.distribution,
            n_grains=self.n_grains,
            box_start=self.box_start,
            box_end=self.box_end,
        )

    @staticmethod
    def load_seeds(filepath: str) -> dict:
        """Load a previously saved .npz and return its contents as a dict."""
        return dict(np.load(filepath))


# ---------------------------------------------------------------------------
# Quick-entry helper
# ---------------------------------------------------------------------------

def generate_grains(
    box_start: tuple[float, float, float],
    box_end: tuple[float, float, float],
    n_grains: int | None = None,
    avg_diameter: float | None = None,
    distribution: str = "random",
    std_dev: float | None = None,
    seed_positions: np.ndarray | None = None,
    seed_radii: np.ndarray | None = None,
    bimodal_params: tuple | None = None,
    random_seed: int | None = None,
    laminate_in_plane_dist: str = "random",
    laminate_direction: str = "z",
    verbose: bool = True,
    progress_callback: Callable[[int, int, float], None] | None = None,
    pause_callback: Callable[[np.ndarray, np.ndarray], bool] | None = None,
) -> SeedResult:
    """Convenience wrapper: create a generator, run the pipeline, return the result."""
    gen = GrainSeedGenerator(
        box_start=box_start,
        box_end=box_end,
        n_grains=n_grains,
        avg_diameter=avg_diameter,
        distribution=distribution,
        std_dev=std_dev,
        seed_positions=seed_positions,
        seed_radii=seed_radii,
        bimodal_params=bimodal_params,
        random_seed=random_seed,
        laminate_in_plane_dist=laminate_in_plane_dist,
        laminate_direction=laminate_direction,
    )
    return gen.run(
        verbose=verbose,
        progress_callback=progress_callback,
        pause_callback=pause_callback,
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke test
    result = generate_grains(
        box_start=(0, 0, 0),
        box_end=(100, 100, 100),
        n_grains=30,
        distribution="random",
        random_seed=42,
    )
    print(f"Generated {result.n_grains} seeds, "
          f"mean dia = {np.mean(result.diameters):.3f}")
