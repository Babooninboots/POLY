"""
Module 5: Polycrystalline Assembly.

Rotates, translates, and crops a pristine crystal into individual grains,
then assembles the final polycrystalline atomic structure.  Uses vectorized
nearest-seed distance testing for Voronoi-cell truncation.

Output: LAMMPS atomic data file + companion dump file with grain_id and
Euler angles (as specified in SPEC.md Module 6).
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Atomic masses (g/mol) -- lightweight built-in table; ASE preferred at runtime
# ---------------------------------------------------------------------------

_ATOMIC_MASSES: dict[str, float] = {
    "H": 1.00794,   "He": 4.002602, "Li": 6.941,    "Be": 9.012182,
    "B": 10.811,    "C": 12.0107,   "N": 14.0067,   "O": 15.9994,
    "F": 18.9984,   "Ne": 20.1797,  "Na": 22.9897,  "Mg": 24.3050,
    "Al": 26.9815,  "Si": 28.0855,  "P": 30.9738,   "S": 32.065,
    "Cl": 35.453,   "Ar": 39.948,   "K": 39.0983,   "Ca": 40.078,
    "Sc": 44.9559,  "Ti": 47.867,   "V": 50.9415,   "Cr": 51.9961,
    "Mn": 54.9380,  "Fe": 55.845,   "Co": 58.9332,  "Ni": 58.6934,
    "Cu": 63.546,   "Zn": 65.38,    "Ga": 69.723,   "Ge": 72.64,
    "As": 74.9216,  "Se": 78.96,    "Br": 79.904,   "Kr": 83.798,
    "Rb": 85.4678,  "Sr": 87.62,    "Y": 88.9059,   "Zr": 91.224,
    "Nb": 92.9064,  "Mo": 95.96,    "Tc": 98,       "Ru": 101.07,
    "Rh": 102.906,  "Pd": 106.42,   "Ag": 107.868,  "Cd": 112.411,
    "In": 114.818,  "Sn": 118.710,  "Sb": 121.760,  "Te": 127.60,
    "I": 126.904,   "Xe": 131.293,  "Cs": 132.905,  "Ba": 137.327,
    "La": 138.905,  "Ce": 140.116,  "Pr": 140.908,  "Nd": 144.242,
    "Pm": 145,      "Sm": 150.36,   "Eu": 151.964,  "Gd": 157.25,
    "Tb": 158.925,  "Dy": 162.500,  "Ho": 164.930,  "Er": 167.259,
    "Tm": 168.934,  "Yb": 173.054,  "Lu": 174.967,  "Hf": 178.49,
    "Ta": 180.948,  "W": 183.84,    "Re": 186.207,  "Os": 190.23,
    "Ir": 192.217,  "Pt": 195.084,  "Au": 196.967,  "Hg": 200.59,
    "Tl": 204.383,  "Pb": 207.2,    "Bi": 208.980,  "Po": 209,
    "At": 210,      "Rn": 222,      "Fr": 223,      "Ra": 226,
    "Ac": 227,      "Th": 232.038,  "Pa": 231.036,  "U": 238.029,
}


def _get_mass(symbol: str) -> float:
    """Return atomic mass for *symbol*, trying ASE first."""
    try:
        from ase.data import atomic_masses, atomic_numbers
        z = atomic_numbers[symbol]
        return float(atomic_masses[z])
    except Exception:
        return _ATOMIC_MASSES.get(symbol, 1.0)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AssemblyResult:
    """Container returned by PolycrystalAssembly.run()."""

    positions: np.ndarray             # (N_total, 3)
    types: np.ndarray                 # (N_total,) int  atom type IDs (1-based)
    grain_ids: np.ndarray             # (N_total,) int  grain index (0-based)
    euler_per_atom: np.ndarray        # (N_total, 3) zxz Euler angles (deg)
    symbols: list[str]                # unique element symbols
    type_to_symbol: dict[int, str]    # type_id -> symbol
    type_masses: dict[int, float]     # type_id -> mass (g/mol)
    n_grains: int
    n_atoms: int
    box_start: np.ndarray             # (3,)
    box_end: np.ndarray               # (3,)
    keep_counts: np.ndarray           # (N_grains,) atoms kept per grain


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------


class PolycrystalAssembly:
    """Assemble a polycrystal from seeds, a pristine crystal, and orientations.

    For each grain the pristine crystal is rotated, translated to the grain
    seed, and then trimmed by a vectorized nearest-seed test: every atom is
    assigned to whichever seed it is closest to, which is mathematically
    equivalent to Voronoi-cell truncation.

    Parameters
    ----------
    seeds : (N_grains, 3) ndarray
        Grain seed coordinates (from Module 1).
    crystal_atoms : ase.Atoms
        Pristine-crystal supercell with COM at origin (from Module 2).
    orientations : OrientationResult
        Output of Module 3 (holds rotation_matrices, euler_angles).
    box_start, box_end : array-like (3,)
        Simulation box bounds.
    """

    def __init__(
        self,
        seeds: np.ndarray,
        crystal_atoms,           # ase.Atoms
        orientations,            # OrientationResult
        box_start,
        box_end,
        target_radii: np.ndarray | None = None,
        grain_diagonals: np.ndarray | None = None,
        is_laminate: bool = False,
        hkl: tuple | None = None,
        stack_axis: str = "z",
        poly_data: list | None = None,
        is_columnar: bool = False,
        max_grain_z: float | None = None,
    ):
        self.seeds = np.asarray(seeds, dtype=float)
        self.n_grains = len(self.seeds)
        self.crystal = crystal_atoms
        self.orientations = orientations
        self.box_start = np.asarray(box_start, dtype=float)
        self.box_end = np.asarray(box_end, dtype=float)
        self.box_size = self.box_end - self.box_start
        self.target_radii = target_radii
        self.grain_diagonals = np.asarray(grain_diagonals, dtype=float) if grain_diagonals is not None else None
        self.is_laminate = is_laminate
        self.hkl = hkl
        self.stack_axis = stack_axis
        self._poly_data = poly_data
        self.is_columnar = is_columnar
        self._max_grain_z = max_grain_z

        if len(self.seeds) != len(self.orientations.rotation_matrices):
            raise ValueError(
                f"Seed count ({len(self.seeds)}) != orientation count "
                f"({len(self.orientations.rotation_matrices)})"
            )

        self._crystal_positions = self.crystal.get_positions()
        self._crystal_symbols = np.array(self.crystal.get_chemical_symbols())

        unique = sorted(set(self._crystal_symbols))
        self._symbol_to_type = {sym: i + 1 for i, sym in enumerate(unique)}
        self._type_to_symbol = {i + 1: sym for i, sym in enumerate(unique)}
        self._type_masses = {
            tid: _get_mass(sym) for tid, sym in self._type_to_symbol.items()
        }
        self._crystal_types = np.array(
            [self._symbol_to_type[s] for s in self._crystal_symbols], dtype=int
        )
        self._n_atoms_per_copy = len(self._crystal_positions)

        # Set after assemble()
        self._positions: np.ndarray | None = None
        self._types: np.ndarray | None = None
        self._grain_ids: np.ndarray | None = None
        self._euler_per_atom: np.ndarray | None = None
        self._n_total: int = 0

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble(
        self,
        batch_size: int = 50000,
        verbose: bool = True,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Run the polycrystalline assembly loop.

        Returns
        -------
        positions : (N_total, 3)
        types : (N_total,)
        grain_ids : (N_total,)
        euler_per_atom : (N_total, 3)
        keep_counts : (N_grains,)
        """
        if verbose:
            print(f"=== Module 5: Polycrystalline Assembly ===")
            print(f"  N grains:         {self.n_grains}")
            print(f"  Atoms / copy:     {self._n_atoms_per_copy:,}")

        rotation_matrices = self.orientations.rotation_matrices
        euler_grain = self.orientations.euler_angles

        if self.n_grains == 1:
            self.seeds[0] = (self.box_start + self.box_end) / 2.0

        # ----------------------------------------------------------------
        # Step 1: Build 27 PBC seed images + KD-tree.
        # ----------------------------------------------------------------
        grid = np.array(
            [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
            dtype=float,
        )
        n_img = 27
        shifts = grid * self.box_size[np.newaxis, :]  # (27, 3)

        seeds_all = (
            self.seeds[:, np.newaxis, :] + shifts[np.newaxis, :, :]
        )  # (N_grains, 27, 3)
        seeds_flat = seeds_all.reshape(-1, 3)
        grain_all = np.repeat(np.arange(self.n_grains), n_img)
        radii_all = (
            np.repeat(self.target_radii, n_img) if self.target_radii is not None else None
        )
        n_seeds = len(seeds_flat)
        seed_tree = cKDTree(seeds_flat)

        if verbose:
            print(f"  Seed images:      {n_seeds} ({self.n_grains} x 27)")

        # ----------------------------------------------------------------
        # Step 2: Select best periodic image per grain.
        # The image closest to the box center minimizes Voronoi cell
        # splitting after the final modulo wrap.
        # ----------------------------------------------------------------
        box_center = (self.box_start + self.box_end) / 2.0
        dists_to_center = np.sum((seeds_all - box_center) ** 2, axis=2)  # (N, 27)
        best_idx = np.argmin(dists_to_center, axis=1)                     # (N,)
        best_seeds = seeds_all[np.arange(self.n_grains), best_idx]        # (N, 3)
        best_seed_indices = np.arange(self.n_grains) * n_img + best_idx   # flat indices

        # ----------------------------------------------------------------
        # Step 3: Per-grain processing.
        #   Columnar (laminate / evenly-spaced):
        #     hkl-align crystal → master pillar → per-grain: crop cylinder
        #     → rotate Rz → translate → KD-tree.
        #   Standard:
        #     per-grain: pre-crop sphere → rotate → translate → KD-tree.
        # ----------------------------------------------------------------
        all_pos: list[np.ndarray] = []
        all_typ: list[np.ndarray] = []
        all_gid: list[np.ndarray] = []
        all_eul: list[np.ndarray] = []

        if (self.is_columnar and self.hkl is not None
                and self.grain_diagonals is not None
                and self._max_grain_z is not None):
            # ============================================================
            # Columnar pipeline
            # ============================================================

            # -- R_base: align crystal [hkl] with stack axis --
            n = np.array(self.hkl, dtype=float)
            n_norm = np.linalg.norm(n)
            z_crys = n / n_norm if n_norm > 0 else np.array([0.0, 0.0, 1.0])

            x_crys = None
            cand_list = [
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1, -1, 0], [1, 1, 0], [1, 0, -1], [0, 1, -1],
                [1, 1, -2], [1, -2, 1], [-2, 1, 1],
            ]
            for cand in cand_list:
                c = np.array(cand, dtype=float)
                c_norm = np.linalg.norm(c)
                if c_norm > 1e-6:
                    c /= c_norm
                    if abs(np.dot(z_crys, c)) < 1e-8:
                        x_crys = c
                        break
            if x_crys is None:
                arb = (np.array([1.0, 0.0, 0.0]) if abs(z_crys[0]) < 0.99
                       else np.array([0.0, 1.0, 0.0]))
                x_crys = np.cross(z_crys, arb)
                x_crys /= np.linalg.norm(x_crys)
            y_crys = np.cross(z_crys, x_crys)
            y_crys /= np.linalg.norm(y_crys)

            stack = self.stack_axis
            if stack == "x":
                perm = [2, 0, 1]
            elif stack == "y":
                perm = [1, 2, 0]
            else:
                perm = [0, 1, 2]
            R_base_mat = np.vstack([x_crys, y_crys, z_crys])[perm, :]
            R_base = Rotation.from_matrix(R_base_mat)

            if verbose:
                print(f"  Columnar pipeline: hkl={self.hkl}, stack={stack}")

            # -- Pre-rotate entire crystal --
            pre_rotated = R_base.apply(self._crystal_positions)

            # -- Trim to master pillar (diameter = 2*max_diag, height = 1.5*max_grain_z) --
            max_diag = float(self.grain_diagonals.max())
            pillar_radius = max_diag
            pillar_half_h = 0.75 * self._max_grain_z

            stack_idx = {"x": 0, "y": 1, "z": 2}[stack]
            ax0, ax1 = [(a, b) for a in (0, 1, 2) for b in (0, 1, 2)
                        if a < b and a != stack_idx and b != stack_idx][0]
            r_from_stack = np.sqrt(
                pre_rotated[:, ax0] ** 2 + pre_rotated[:, ax1] ** 2,
            )
            pillar_mask = (
                (r_from_stack <= pillar_radius)
                & (np.abs(pre_rotated[:, stack_idx]) <= pillar_half_h)
            )
            pos_pillar = pre_rotated[pillar_mask]
            typ_pillar = self._crystal_types[pillar_mask]
            r_pillar_xy = r_from_stack[pillar_mask]
            z_pillar_abs = np.abs(pos_pillar[:, stack_idx])

            if verbose:
                print(
                    f"  Master pillar:     r={pillar_radius:.1f} A, "
                    f"h={2 * pillar_half_h:.1f} A"
                )
                print(
                    f"  Pillar atoms:      {len(pos_pillar):,} "
                    f"/ {len(pre_rotated):,}"
                )

            # -- Per-grain loop --
            for g in range(self.n_grains):
                grain_mask = (
                    (r_pillar_xy <= self.grain_diagonals[g])
                    & (z_pillar_abs <= pillar_half_h)
                )
                n_crop = int(grain_mask.sum())
                if n_crop == 0:
                    if verbose:
                        print(
                            f"  Grain {g + 1}/{self.n_grains}: "
                            f"0 atoms (empty pre-crop)"
                        )
                    continue

                # In-plane rotation Rz = R_total * R_base^{-1}
                R_total = Rotation.from_matrix(rotation_matrices[g])
                Rz = R_total * R_base.inv()
                pos_rotated = Rz.apply(pos_pillar[grain_mask])

                # Translate to best seed (+ small shift to avoid boundaries)
                pos_crop = pos_rotated + best_seeds[g] + np.array([0.1, 0.1, 0.1])
                typ_crop = typ_pillar[grain_mask]

                # KD-tree query
                if radii_all is not None:
                    k_kd = min(30, n_seeds)
                    kd_dists, kd_indices = seed_tree.query(pos_crop, k=k_kd)
                    r_k = radii_all[kd_indices]
                    power_dists = kd_dists ** 2 - r_k ** 2
                    best_k = np.argmin(power_dists, axis=1)
                    closest = kd_indices[np.arange(n_crop), best_k]
                else:
                    _, closest = seed_tree.query(pos_crop, k=1)
                    closest = (closest.ravel() if closest.ndim > 1
                               else closest)

                keep_mask = closest == best_seed_indices[g]
                n_keep = int(keep_mask.sum())
                if n_keep == 0:
                    if verbose:
                        print(
                            f"  Grain {g + 1}/{self.n_grains}: "
                            f"0 atoms kept"
                        )
                    continue

                all_pos.append(pos_crop[keep_mask])
                all_typ.append(typ_crop[keep_mask])
                all_gid.append(np.full(n_keep, g, dtype=int))
                all_eul.append(np.tile(euler_grain[g], (n_keep, 1)))

                if verbose:
                    print(
                        f"  Grain {g + 1}/{self.n_grains}: "
                        f"{n_keep:,} atoms kept"
                    )

        else:
            # ============================================================
            # Standard pipeline
            # ============================================================

            # Per-grain spherical pre-crop radius (applied at origin).
            if self.grain_diagonals is not None:
                pre_crop_radii = self.grain_diagonals * 0.55
            else:
                pre_crop_radii = np.full(
                    self.n_grains, float(np.linalg.norm(self.box_size)) * 0.75,
                )

            if verbose:
                r_min, r_max = pre_crop_radii.min(), pre_crop_radii.max()
                print(
                    f"  Pre-crop radii:    ({r_min:.1f} .. {r_max:.1f}) A  "
                    f"({'grain diagonals' if self.grain_diagonals is not None else 'box diagonal fallback'})"
                )

            for g in range(self.n_grains):
                dists_from_origin = np.linalg.norm(
                    self._crystal_positions, axis=1,
                )
                crop_mask = dists_from_origin <= pre_crop_radii[g]
                n_crop = int(crop_mask.sum())
                if n_crop == 0:
                    if verbose:
                        print(
                            f"  Grain {g + 1}/{self.n_grains}: "
                            f"0 atoms (empty pre-crop)"
                        )
                    continue

                R = Rotation.from_matrix(rotation_matrices[g])
                pos_crop = (
                    R.apply(self._crystal_positions[crop_mask])
                    + best_seeds[g]
                )
                typ_crop = self._crystal_types[crop_mask]

                if radii_all is not None:
                    k_kd = min(30, n_seeds)
                    kd_dists, kd_indices = seed_tree.query(pos_crop, k=k_kd)
                    r_k = radii_all[kd_indices]
                    power_dists = kd_dists ** 2 - r_k ** 2
                    best_k = np.argmin(power_dists, axis=1)
                    closest = kd_indices[np.arange(n_crop), best_k]
                else:
                    _, closest = seed_tree.query(pos_crop, k=1)
                    closest = (closest.ravel() if closest.ndim > 1
                               else closest)

                keep_mask = closest == best_seed_indices[g]
                n_keep = int(keep_mask.sum())
                if n_keep == 0:
                    if verbose:
                        print(
                            f"  Grain {g + 1}/{self.n_grains}: "
                            f"0 atoms kept"
                        )
                    continue

                all_pos.append(pos_crop[keep_mask])
                all_typ.append(typ_crop[keep_mask])
                all_gid.append(np.full(n_keep, g, dtype=int))
                all_eul.append(np.tile(euler_grain[g], (n_keep, 1)))

                if verbose:
                    print(
                        f"  Grain {g + 1}/{self.n_grains}: "
                        f"{n_keep:,} atoms kept"
                    )

        if not all_pos:
            if verbose:
                print("  No atoms after KD-tree — returning empty result.")
            return (
                np.empty((0, 3)),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                np.empty((0, 3)),
                np.zeros(self.n_grains, dtype=int),
            )

        positions = np.concatenate(all_pos)
        types = np.concatenate(all_typ)
        grain_ids = np.concatenate(all_gid)
        euler_per_atom = np.concatenate(all_eul)

        if verbose:
            print(f"  After KD-tree:    {len(positions):,} atoms  (per-grain)")

        # ----------------------------------------------------------------
        # Step 5: Modulo wrap all positions into primary box.
        # ----------------------------------------------------------------
        positions = np.mod(
            positions - self.box_start, self.box_size
        ) + self.box_start

        keep_counts = np.zeros(self.n_grains, dtype=int)
        for g in range(self.n_grains):
            keep_counts[g] = int(np.sum(grain_ids == g))

        if verbose:
            total = len(positions)
            print(f"  Modulo wrap:      complete")
            print(f"  Total assembled:  {total:,} atoms")
            print(
                f"  Min/avg/max kept: "
                f"{keep_counts.min():,} / {keep_counts.mean():.0f} / "
                f"{keep_counts.max():,}"
            )
            print(f"=== Module 5 complete ===\n")

        return positions, types, grain_ids, euler_per_atom, keep_counts

    # ------------------------------------------------------------------
    # LAMMPS writers
    # ------------------------------------------------------------------

    @staticmethod
    def _lammps_data_header(
        n_atoms: int,
        n_types: int,
        box_start: np.ndarray,
        box_end: np.ndarray,
    ) -> str:
        xlo, ylo, zlo = box_start
        xhi, yhi, zhi = box_end
        return (
            f"LAMMPS data file via POLY\n\n"
            f"{n_atoms} atoms\n"
            f"{n_types} atom types\n\n"
            f"{xlo:.8f} {xhi:.8f} xlo xhi\n"
            f"{ylo:.8f} {yhi:.8f} ylo yhi\n"
            f"{zlo:.8f} {zhi:.8f} zlo zhi\n"
        )

    @staticmethod
    def _lammps_masses_block(type_masses: dict[int, float]) -> str:
        lines = ["Masses\n"]
        for tid in sorted(type_masses):
            lines.append(f"  {tid} {type_masses[tid]:.6f}")
        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _lammps_atoms_block(
        positions: np.ndarray,
        types: np.ndarray,
        start_id: int = 1,
    ) -> str:
        lines = ["Atoms\n"]
        for j in range(len(positions)):
            aid = start_id + j
            t = types[j]
            x, y, z = positions[j]
            lines.append(f"  {aid} {t} {x:.8f} {y:.8f} {z:.8f}")
        lines.append("")
        return "\n".join(lines) + "\n"

    def write_lammps_data(self, filepath: str) -> None:
        """Write the assembled structure as a LAMMPS atomic data file."""
        if self._positions is None:
            raise RuntimeError("No assembly data; call assemble() first.")

        header = self._lammps_data_header(
            self._n_total,
            len(self._type_to_symbol),
            self.box_start,
            self.box_end,
        )
        masses = self._lammps_masses_block(self._type_masses)
        atoms = self._lammps_atoms_block(self._positions, self._types)

        with open(filepath, "w") as fh:
            fh.write(header)
            fh.write("\n")
            fh.write(masses)
            fh.write("\n")
            fh.write(atoms)

    def write_lammps_dump(self, filepath: str) -> None:
        """Write a LAMMPS dump file with grain_id and Euler angles.

        Format:
          ITEM: ATOMS id type x y z grain_id euler_angle_1 euler_angle_2 euler_angle_3
        """
        if self._positions is None:
            raise RuntimeError("No assembly data; call assemble() first.")

        xlo, ylo, zlo = self.box_start
        xhi, yhi, zhi = self.box_end

        with open(filepath, "w") as fh:
            fh.write("ITEM: TIMESTEP\n0\n")
            fh.write(f"ITEM: NUMBER OF ATOMS\n{self._n_total}\n")
            fh.write("ITEM: BOX BOUNDS pp pp pp\n")
            fh.write(f"{xlo:.8f} {xhi:.8f}\n")
            fh.write(f"{ylo:.8f} {yhi:.8f}\n")
            fh.write(f"{zlo:.8f} {zhi:.8f}\n")
            fh.write(
                "ITEM: ATOMS id type x y z grain_id "
                "euler_angle_1 euler_angle_2 euler_angle_3\n"
            )

            for j in range(self._n_total):
                aid = j + 1
                t = self._types[j]
                x, y, z = self._positions[j]
                gid = self._grain_ids[j] + 1  # 1-based grain id
                e1, e2, e3 = self._euler_per_atom[j]
                fh.write(
                    f"{aid} {t} {x:.8f} {y:.8f} {z:.8f} "
                    f"{gid} {e1:.6f} {e2:.6f} {e3:.6f}\n"
                )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        data_path: str | None = None,
        dump_path: str | None = None,
        batch_size: int = 50000,
        verbose: bool = True,
    ) -> AssemblyResult:
        """Execute the full Module-5 pipeline: assemble + optional file output.

        Parameters
        ----------
        data_path : str, optional
            If given, write a LAMMPS data file to this path.
        dump_path : str, optional
            If given, write a LAMMPS dump file to this path.
        batch_size : int
            Atoms per batch during distance computation.
        verbose : bool
        """
        (
            positions,
            types,
            grain_ids,
            euler_per_atom,
            keep_counts,
        ) = self.assemble(batch_size=batch_size, verbose=verbose)

        self._positions = positions
        self._types = types
        self._grain_ids = grain_ids
        self._euler_per_atom = euler_per_atom
        self._n_total = len(positions)

        if data_path is not None:
            self.write_lammps_data(data_path)
            if verbose:
                print(f"  LAMMPS data file -> {data_path}")

        if dump_path is not None:
            self.write_lammps_dump(dump_path)
            if verbose:
                print(f"  LAMMPS dump file -> {dump_path}")

        return AssemblyResult(
            positions=positions,
            types=types,
            grain_ids=grain_ids,
            euler_per_atom=euler_per_atom,
            symbols=list(self._type_to_symbol.values()),
            type_to_symbol=dict(self._type_to_symbol),
            type_masses=dict(self._type_masses),
            n_grains=self.n_grains,
            n_atoms=self._n_total,
            box_start=self.box_start.copy(),
            box_end=self.box_end.copy(),
            keep_counts=keep_counts,
        )


# ---------------------------------------------------------------------------
# Quick-entry helper
# ---------------------------------------------------------------------------


def assemble_polycrystal(
    seeds: np.ndarray,
    crystal_atoms,
    orientations,
    box_start,
    box_end,
    target_radii: np.ndarray | None = None,
    grain_diagonals: np.ndarray | None = None,
    is_laminate: bool = False,
    hkl: tuple | None = None,
    stack_axis: str = "z",
    poly_data: list | None = None,
    is_columnar: bool = False,
    max_grain_z: float | None = None,
    data_path: str | None = None,
    dump_path: str | None = None,
    batch_size: int = 50000,
    verbose: bool = True,
) -> AssemblyResult:
    """Convenience wrapper: create assembler, run pipeline, return result."""
    assembler = PolycrystalAssembly(
        seeds=seeds,
        crystal_atoms=crystal_atoms,
        orientations=orientations,
        box_start=box_start,
        box_end=box_end,
        target_radii=target_radii,
        grain_diagonals=grain_diagonals,
        is_laminate=is_laminate,
        hkl=hkl,
        stack_axis=stack_axis,
        poly_data=poly_data,
        is_columnar=is_columnar,
        max_grain_z=max_grain_z,
    )
    return assembler.run(
        data_path=data_path,
        dump_path=dump_path,
        batch_size=batch_size,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from grain_seeds import generate_grains
    from pristine_crystal import generate_pristine_bravais
    from orientation import generate_orientations

    box_s, box_e = (0, 0, 0), (60, 60, 60)

    grains = generate_grains(
        box_start=box_s, box_end=box_e,
        n_grains=8, distribution="random",
        random_seed=1, verbose=False,
    )

    crystal = generate_pristine_bravais(
        "Al", "fcc", a=4.05,
        box_start=box_s, box_end=box_e,
        verbose=False,
    )

    ori = generate_orientations(
        "low_angle",
        n_grains=grains.n_grains,
        neighbors=grains.neighbors,
        random_seed=2, verbose=False,
    )

    result = assemble_polycrystal(
        seeds=grains.seeds,
        crystal_atoms=crystal.atoms,
        orientations=ori,
        box_start=box_s, box_end=box_e,
        verbose=True,
    )
