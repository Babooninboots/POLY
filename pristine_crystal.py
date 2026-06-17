"""
Module 2: Pristine Crystal Generation.

Builds a monolithic atomic lattice from a unit cell specification
(Bravais lattice, space group, CIF file, or custom cell).  The unit cell
is replicated along its lattice vectors until the supercell fully
encompasses the target simulation box, then shifted so its centre of
mass sits at the origin (0, 0, 0).

Requires: ASE (Atomic Simulation Environment)  --  ``pip install ase``
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

# ASE imports
from ase import Atoms
from ase.build import bulk as ase_bulk
from ase.spacegroup import crystal as ase_crystal
from ase.io import read as ase_read


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CrystalResult:
    """Container returned by PristineCrystal.run()."""

    atoms: Atoms                # ASE Atoms object for the supercell
    unit_cell: Atoms            # original unit cell
    box_start: np.ndarray       # (3,) target box lower bound
    box_end: np.ndarray         # (3,) target box upper bound
    box_size: np.ndarray        # (3,) box dimensions
    repeats: tuple[int, int, int]  # (nx, ny, nz) replication factors

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def chemical_formula(self) -> str:
        return str(self.atoms.get_chemical_formula())

    @property
    def supercell_shape(self) -> tuple[int, int, int, int]:
        """(n_total_atoms, nx, ny, nz)"""
        return (len(self.atoms), *self.repeats)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PristineCrystal:
    """Construct a pristine crystal supercell for the simulation box.

    Supports four entry points:

    - ``from_bravais``   — standard Bravais lattices (fcc, bcc, hcp, …)
    - ``from_spacegroup`` — arbitrary space groups with Wyckoff positions
    - ``from_cif``       — read a CIF file
    - ``from_custom``    — raw cell matrix + scaled positions + symbols

    Parameters
    ----------
    unit_cell : Atoms
        ASE Atoms object representing the unit cell.
    box_start, box_end : array-like (3,)
        Simulation box bounds (from Module 1 / user definition).
    """

    # -- lattice parameter shorthand table --------------------------------
    # Maps common element symbols to their ground-state crystal structure
    # and lattice constants (room-temperature experimental values, in Ang).
    # Users can override any value at call time.

    ELEMENT_DEFAULTS: dict[str, dict] = {
        "Al": {"structure": "fcc", "a": 4.0495},
        "Cu": {"structure": "fcc", "a": 3.6149},
        "Ni": {"structure": "fcc", "a": 3.5238},
        "Au": {"structure": "fcc", "a": 4.0782},
        "Ag": {"structure": "fcc", "a": 4.0853},
        "Pt": {"structure": "fcc", "a": 3.9242},
        "Pd": {"structure": "fcc", "a": 3.8907},
        "Pb": {"structure": "fcc", "a": 4.9505},
        "Fe": {"structure": "bcc", "a": 2.8665},
        "W":  {"structure": "bcc", "a": 3.1652},
        "Mo": {"structure": "bcc", "a": 3.1472},
        "Ta": {"structure": "bcc", "a": 3.3058},
        "Nb": {"structure": "bcc", "a": 3.3008},
        "V":  {"structure": "bcc", "a": 3.0240},
        "Cr": {"structure": "bcc", "a": 2.8846},
        "Na": {"structure": "bcc", "a": 4.2906},
        "K":  {"structure": "bcc", "a": 5.3210},
        "Mg": {"structure": "hcp", "a": 3.2094, "c": 5.2105},
        "Ti": {"structure": "hcp", "a": 2.9506, "c": 4.6788},
        "Zn": {"structure": "hcp", "a": 2.6649, "c": 4.9468},
        "Zr": {"structure": "hcp", "a": 3.2317, "c": 5.1476},
        "Co": {"structure": "hcp", "a": 2.5071, "c": 4.0695},
        "Si": {"structure": "diamond", "a": 5.4310},
        "Ge": {"structure": "diamond", "a": 5.6579},
        "C":  {"structure": "diamond", "a": 3.5668},
    }

    # -- intermetallic structure table ------------------------------------
    # Maps Strukturbericht symbols to crystallographic data.
    # Each entry holds the prototype, space group, Wyckoff sites
    # (unique representatives only), and prototype lattice parameters.
    # Users override elements via the *symbols* argument and lattice
    # constants via *a* / *c* / *cellpar*.

    INTERMETALLIC_DEFAULTS: dict[str, dict] = {
        "L1_2": {
            "prototype": "Cu3Au",
            "spacegroup": 221,
            "setting": 1,
            "wyckoff_symbols": ("Au", "Cu"),
            "wyckoff_basis": [(0, 0, 0), (0, 0.5, 0.5)],
            "cellpar": (3.75, 3.75, 3.75, 90, 90, 90),
            "description": "Ordered fcc derivative, Pm-3m #221.  "
                           "Prototype: Cu3Au.  4 atoms/cell.",
        },
        "L1_0": {
            "prototype": "CuAu",
            "spacegroup": 123,
            "setting": 1,
            "wyckoff_symbols": ("Au", "Au", "Cu"),
            "wyckoff_basis": [
                (0, 0, 0),            # 1a
                (0.5, 0.5, 0),        # 1c
                (0, 0.5, 0.5),        # 2e
            ],
            "cellpar": (2.80, 2.80, 3.70, 90, 90, 90),
            "description": "Tetragonal distortion of fcc, P4/mmm #123.  "
                           "Prototype: CuAu.  4 atoms/cell.",
        },
        "B2": {
            "prototype": "CsCl",
            "spacegroup": 221,
            "setting": 1,
            "wyckoff_symbols": ("Cs", "Cl"),
            "wyckoff_basis": [(0, 0, 0), (0.5, 0.5, 0.5)],
            "cellpar": (4.12, 4.12, 4.12, 90, 90, 90),
            "description": "Ordered bcc derivative, Pm-3m #221.  "
                           "Prototype: CsCl.  2 atoms/cell.",
        },
        "D0_3": {
            "prototype": "BiF3",
            "spacegroup": 225,
            "setting": 1,
            "wyckoff_symbols": ("Bi", "Fe", "Fe"),
            "wyckoff_basis": [
                (0, 0, 0),            # 4a
                (0.5, 0.5, 0.5),      # 4b
                (0.25, 0.25, 0.25),   # 8c
            ],
            "cellpar": (5.65, 5.65, 5.65, 90, 90, 90),
            "description": "Ordered bcc derivative, Fm-3m #225.  "
                           "Prototype: BiF3 / AlFe3.  16 atoms/cell.",
        },
        "L2_1": {
            "prototype": "AlCu2Mn",
            "spacegroup": 225,
            "setting": 1,
            "wyckoff_symbols": ("Al", "Mn", "Cu"),
            "wyckoff_basis": [
                (0, 0, 0),            # 4a
                (0.5, 0.5, 0.5),      # 4b
                (0.25, 0.25, 0.25),   # 8c
            ],
            "cellpar": (5.95, 5.95, 5.95, 90, 90, 90),
            "description": "Heusler alloy, Fm-3m #225.  "
                           "Prototype: AlCu2Mn.  16 atoms/cell.",
        },
        "D0_19": {
            "prototype": "Ni3Sn",
            "spacegroup": 194,
            "setting": 1,
            "wyckoff_symbols": ("Sn", "Ni"),
            "wyckoff_basis": [
                (1 / 3, 2 / 3, 0.25),    # 2c
                (5 / 6, 5 / 3, 0.25),    # 6h, x ~ 5/6
            ],
            "cellpar": (5.29, 5.29, 4.30, 90, 90, 120),
            "description": "Ordered hcp derivative, P6_3/mmc #194.  "
                           "Prototype: Ni3Sn.  8 atoms/cell.",
        },
        "A15": {
            "prototype": "Cr3Si",
            "spacegroup": 223,
            "setting": 1,
            "wyckoff_symbols": ("Si", "Cr"),
            "wyckoff_basis": [
                (0, 0, 0),            # 2a
                (0.25, 0, 0.5),       # 6c
            ],
            "cellpar": (4.56, 4.56, 4.56, 90, 90, 90),
            "description": "A15 / beta-W structure, Pm-3n #223.  "
                           "Prototype: Cr3Si.  8 atoms/cell.",
        },
        "C15": {
            "prototype": "MgCu2",
            "spacegroup": 227,
            "setting": 1,
            "wyckoff_symbols": ("Mg", "Cu"),
            "wyckoff_basis": [
                (0, 0, 0),                # 8a
                (0.625, 0.625, 0.625),    # 16d  (5/8)
            ],
            "cellpar": (7.04, 7.04, 7.04, 90, 90, 90),
            "description": "Cubic Laves phase, Fd-3m #227.  "
                           "Prototype: MgCu2.  24 atoms/cell.",
        },
    }

    def __init__(
        self,
        unit_cell: Atoms,
        box_start: ArrayLike = (0.0, 0.0, 0.0),
        box_end: ArrayLike = (100.0, 100.0, 100.0),
        coverage: float | ArrayLike | None = None,
    ):
        self.unit_cell = unit_cell.copy()
        self.box_start = np.asarray(box_start, dtype=float)
        self.box_end = np.asarray(box_end, dtype=float)
        self.box_size = self.box_end - self.box_start

        # Cartesian extent the supercell must cover in every direction.
        # Defaults to box_size for backward compatibility.  Set to a
        # larger value (e.g. 2 * max neighbour distance) to ensure the
        # rotated crystal fills every Voronoi cell.
        if coverage is None:
            self.coverage = self.box_size.copy()
        elif np.ndim(coverage) == 0:
            self.coverage = np.full(3, float(coverage), dtype=float)
        else:
            self.coverage = np.asarray(coverage, dtype=float)
            if self.coverage.shape != (3,):
                raise ValueError(
                    f"coverage must be None, a scalar, or (3,); "
                    f"got shape {self.coverage.shape}"
                )

        # Supercell (populated by build_supercell)
        self.atoms: Atoms | None = None
        self.repeats: tuple[int, int, int] | None = None

    # ------------------------------------------------------------------
    # Factory: Bravais lattice
    # ------------------------------------------------------------------

    @classmethod
    def from_bravais(
        cls,
        symbol: str,
        crystalstructure: str | None = None,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
        *,
        covera: float | None = None,
        u: float | None = None,
        alpha: float | None = None,
        orthorhombic: bool = False,
        cubic: bool = False,
        box_start: ArrayLike = (0.0, 0.0, 0.0),
        box_end: ArrayLike = (100.0, 100.0, 100.0),
        coverage: float | ArrayLike | None = None,
    ) -> "PristineCrystal":
        """Build from a standard Bravais lattice via ``ase.build.bulk``.

        If *crystalstructure* and lattice constants are omitted the module
        looks up the element in ``ELEMENT_DEFAULTS``.

        Parameters
        ----------
        symbol : str
            Chemical symbol (e.g. ``"Al"``, ``"NaCl"``, ``"MgO"``).
        crystalstructure : str, optional
            One of ``sc, fcc, bcc, bct, hcp, diamond, zincblende, rocksalt,
            cesiumchloride, fluorite, wurtzite``.
        a, b, c : float, optional
            Lattice constants in Angstrom.
        covera : float, optional
            c/a ratio (hcp only; default ~1.633).
        u : float, optional
            Internal coordinate (wurtzite only).
        alpha : float, optional
            Angle in degrees (rhombohedral only).
        orthorhombic : bool
            Build orthorhombic unit cell instead of primitive.
        cubic : bool
            Build cubic unit cell when possible.
        box_start, box_end : array-like (3,)
            Simulation box bounds.
        coverage : float or (3,) array-like, optional
            Cartesian extent the supercell must cover (default: box_size).
            Set to ``2 * max_neighbor_distance(seeds, neighbors, box_size)``
            to ensure full Voronoi-cell coverage after rotation.
        """
        # Auto-fill from element database
        if crystalstructure is None and symbol in cls.ELEMENT_DEFAULTS:
            info = cls.ELEMENT_DEFAULTS[symbol]
            crystalstructure = info["structure"]
            if a is None:
                a = info["a"]
            if c is None and "c" in info:
                c = info["c"]

        if crystalstructure is None:
            # Let ASE guess; it will fall back to fcc for single elements
            pass

        # Use conventional cells for better preview and solver stability.
        # sc/fcc/bcc/diamond are naturally cubic; bct needs a manual build.
        _cubic = cubic
        _is_bct = crystalstructure == "bct"
        if crystalstructure in ("sc", "fcc", "bcc", "diamond"):
            _cubic = True

        if _is_bct and a is not None and c is not None:
            atoms = Atoms(
                f"{symbol}2",
                positions=[[0, 0, 0], [a / 2, a / 2, c / 2]],
                cell=[[a, 0, 0], [0, a, 0], [0, 0, c]],
                pbc=True,
            )
        else:
            try:
                atoms = ase_bulk(
                    symbol,
                    crystalstructure=crystalstructure,
                    a=a,
                    b=b,
                    c=c,
                    covera=covera,
                    u=u,
                    alpha=alpha,
                    orthorhombic=orthorhombic,
                    cubic=_cubic,
                )
            except Exception:
                # ASE failed (likely a custom element name like "1").
                # Fall back to building with a placeholder element ('H')
                # and storing the custom name in atoms.info.
                atoms = ase_bulk(
                    "H",  # placeholder — same geometry, custom name below
                    crystalstructure=crystalstructure,
                    a=a, b=b, c=c,
                    covera=covera, u=u, alpha=alpha,
                    orthorhombic=orthorhombic, cubic=_cubic,
                )
                atoms.info["_custom_element"] = symbol
        return cls(atoms, box_start=box_start, box_end=box_end, coverage=coverage)

    # ------------------------------------------------------------------
    # Factory: intermetallic
    # ------------------------------------------------------------------

    @classmethod
    def from_intermetallic(
        cls,
        strukturbericht: str,
        symbols: Sequence[str] | None = None,
        a: float | None = None,
        c: float | None = None,
        cellpar: Sequence[float] | None = None,
        box_start: ArrayLike = (0.0, 0.0, 0.0),
        box_end: ArrayLike = (100.0, 100.0, 100.0),
        coverage: float | ArrayLike | None = None,
    ) -> "PristineCrystal":
        """Build an intermetallic structure from a Strukturbericht symbol.

        Uses ``ase.spacegroup.crystal`` with the Wyckoff positions stored
        in ``INTERMETALLIC_DEFAULTS``.

        Parameters
        ----------
        strukturbericht : str
            Strukturbericht designation (e.g. ``"L1_2"``, ``"B2"``,
            ``"D0_3"``, ``"A15"``, ``"C15"``).
        symbols : sequence of str, optional
            Element symbols for each unique Wyckoff site, in the order
            defined by the prototype.  If omitted the prototype elements
            are used.
        a : float, optional
            Lattice constant *a* in Angstrom.  Overrides the prototype
            value.  For cubic systems this is sufficient; for non-cubic
            systems *c* must also be supplied.
        c : float, optional
            Lattice constant *c* in Angstrom (non-cubic systems only).
        cellpar : sequence of float, optional
            Full set ``[a, b, c, alpha, beta, gamma]``.  Overrides *a*/*c*
            and the prototype defaults entirely.
        box_start, box_end : array-like (3,)
            Simulation box bounds.
        coverage : float or (3,) array-like, optional
            Cartesian extent the supercell must cover (default: box_size).
        """
        if strukturbericht not in cls.INTERMETALLIC_DEFAULTS:
            available = ", ".join(sorted(cls.INTERMETALLIC_DEFAULTS))
            raise ValueError(
                f"Unknown Strukturbericht symbol '{strukturbericht}'.  "
                f"Available: {available}"
            )

        info = cls.INTERMETALLIC_DEFAULTS[strukturbericht]

        # --- resolve elements ---
        wyckoff_symbols = list(info["wyckoff_symbols"])  # prototype
        if symbols is not None:
            symbols = list(symbols)
            if len(symbols) != len(wyckoff_symbols):
                raise ValueError(
                    f"'{strukturbericht}' expects {len(wyckoff_symbols)} "
                    f"element symbols (one per unique Wyckoff site), "
                    f"got {len(symbols)}"
                )
            wyckoff_symbols = symbols

        # --- resolve cell parameters ---
        if cellpar is not None:
            cp = list(cellpar)
        elif a is not None:
            proto = list(info["cellpar"])
            proto[0] = a
            proto[1] = a
            proto[2] = a if c is None else c
            cp = proto
        else:
            cp = list(info["cellpar"])

        atoms = ase_crystal(
            symbols=wyckoff_symbols,
            basis=info["wyckoff_basis"],
            spacegroup=info["spacegroup"],
            setting=info["setting"],
            cellpar=cp,
        )
        return cls(atoms, box_start=box_start, box_end=box_end, coverage=coverage)

    # ------------------------------------------------------------------
    # Factory: space group
    # ------------------------------------------------------------------

    @classmethod
    def from_spacegroup(
        cls,
        symbols: str | Sequence[str],
        basis: Sequence[Sequence[float]],
        spacegroup: int | str = 1,
        setting: int = 1,
        cellpar: Sequence[float] | None = None,
        cell: ArrayLike | None = None,
        size: tuple[int, int, int] = (1, 1, 1),
        ab_normal: Sequence[float] = (0, 0, 1),
        a_direction: Sequence[float] | None = None,
        primitive_cell: bool = False,
        box_start: ArrayLike = (0.0, 0.0, 0.0),
        box_end: ArrayLike = (100.0, 100.0, 100.0),
        coverage: float | ArrayLike | None = None,
        **kwargs,
    ) -> "PristineCrystal":
        """Build from space group data via ``ase.spacegroup.crystal``.

        Parameters
        ----------
        symbols : str or sequence of str
            Element symbols (e.g. ``"Al"`` or ``("Na", "Cl")``).
        basis : list of (3,) sequences
            Scaled (fractional) coordinates of the unique sites.
        spacegroup : int or str
            IT space group number or Hermann-Mauguin symbol (e.g. 225,
            ``"Fm-3m"``).
        setting : 1 or 2
            Space group setting.
        cellpar : [a, b, c, alpha, beta, gamma]
            Lattice parameters (Angstrom / degrees).
        cell : (3,3) array, optional
            Explicit cell matrix; overrides *cellpar*.
        size : (3,) tuple
            Conventional-cell repeats (kept at (1,1,1) — supercell
            expansion is handled later via *box_start*/*box_end*).
        ab_normal : (3,) sequence
            Normal vector of the a-b plane.
        a_direction : (3,) sequence, optional
            Orientation of the a vector.
        primitive_cell : bool
            Return the primitive cell instead of conventional.
        box_start, box_end : array-like (3,)
            Simulation box bounds.
        """
        atoms = ase_crystal(
            symbols=symbols,
            basis=basis,
            spacegroup=spacegroup,
            setting=setting,
            cell=cell,
            cellpar=list(cellpar) if cellpar is not None else None,
            size=size,
            ab_normal=ab_normal,
            a_direction=a_direction,
            primitive_cell=primitive_cell,
            **kwargs,
        )
        return cls(atoms, box_start=box_start, box_end=box_end, coverage=coverage)

    # ------------------------------------------------------------------
    # Factory: CIF file
    # ------------------------------------------------------------------

    @classmethod
    def from_cif(
        cls,
        cif_path: str,
        box_start: ArrayLike = (0.0, 0.0, 0.0),
        box_end: ArrayLike = (100.0, 100.0, 100.0),
        coverage: float | ArrayLike | None = None,
    ) -> "PristineCrystal":
        """Read a unit cell from a CIF file via ``ase.io.read``.

        Parameters
        ----------
        cif_path : str or Path
            Path to a ``.cif`` crystal information file.
        box_start, box_end : array-like (3,)
            Simulation box bounds.
        coverage : float or (3,) array-like, optional
            Cartesian extent the supercell must cover (default: box_size).
        """
        atoms = ase_read(cif_path)
        if not isinstance(atoms, Atoms):
            raise TypeError(
                f"CIF file must contain a single structure; "
                f"got {type(atoms).__name__}"
            )
        return cls(atoms, box_start=box_start, box_end=box_end, coverage=coverage)

    # ------------------------------------------------------------------
    # Factory: custom cell
    # ------------------------------------------------------------------

    @classmethod
    def from_custom(
        cls,
        cell: ArrayLike,
        scaled_positions: ArrayLike,
        symbols: str | Sequence[str],
        box_start: ArrayLike = (0.0, 0.0, 0.0),
        box_end: ArrayLike = (100.0, 100.0, 100.0),
        coverage: float | ArrayLike | None = None,
    ) -> "PristineCrystal":
        """Build from raw cell matrix, scaled positions, and symbols.

        Parameters
        ----------
        cell : (3,3) array-like
            Row-major cell vectors in Angstrom.
        scaled_positions : (N,3) array-like
            Fractional atomic coordinates.
        symbols : str or sequence of str
            Chemical symbols, length N.
        box_start, box_end : array-like (3,)
            Simulation box bounds.
        coverage : float or (3,) array-like, optional
            Cartesian extent the supercell must cover (default: box_size).
        """
        atoms = Atoms(
            symbols=symbols,
            scaled_positions=scaled_positions,
            cell=cell,
            pbc=True,
        )
        return cls(atoms, box_start=box_start, box_end=box_end, coverage=coverage)

    # ------------------------------------------------------------------
    # Supercell construction
    # ------------------------------------------------------------------

    def build_supercell(self, margin: int = 2) -> Atoms:
        r"""Replicate the unit cell until the supercell fills the target
        coverage extent.

        For a cell matrix *C* (rows = lattice vectors), one repeat along
        direction *j* contributes :math:`|C_{j,d}|` to the Cartesian extent
        in direction *d*.  The total Cartesian extent after *n_j* repeats is

        .. math::

            E_d = \sum_{j=0}^2 (n_j - 1)\,|C_{j,d}|

        We solve the 3x3 linear system :math:`A\,(n-1) \ge coverage` where
        :math:`A_{d,j} = |C_{j,d}|`, then add *margin* extra repeats per
        direction as a safety factor (rotation in Module 5, COM shift).

        Falls back to iterative refinement when the matrix is near-singular
        (e.g. a purely 2D cell).
        """
        cell = self.unit_cell.get_cell()   # (3, 3) row-major
        target = self.coverage             # (3,)

        # ---- linear-system estimate -----------------------------------
        A = np.abs(cell).T                 # A[d, j] = |C[j, d]|
        ones = np.ones(3)

        try:
            # Solve  A @ x = target   where  x = n - 1
            x = np.linalg.solve(A, target)
            if np.any(x < -0.5):  # suspicious solution
                raise np.linalg.LinAlgError
            n_est = np.maximum(1, np.ceil(x).astype(int) + 1)
        except np.linalg.LinAlgError:
            # Fallback: crude estimate from cell-vector norms
            cell_norms = np.linalg.norm(cell, axis=1)
            extent_per_repeat = np.maximum(cell_norms, 1e-12)
            n_est = np.array([
                max(1, int(np.ceil(target[i] / extent_per_repeat[i])) + 1)
                for i in range(3)
            ], dtype=int)

        # Apply margin
        n = tuple((n_est + margin).tolist())

        # ---- iterative refinement (rare) -------------------------------
        for _ in range(20):  # safety cap
            self.atoms = self.unit_cell.repeat(n)
            pos = self.atoms.get_positions()
            extent = pos.max(axis=0) - pos.min(axis=0)
            shortfall = target - extent
            if np.all(shortfall <= 0):
                break
            worst_dim = int(np.argmax(shortfall))
            contrib = np.abs(cell[:, worst_dim])  # per-lattice-vec contribution
            best_lat = int(np.argmax(contrib))
            n_list = list(n)
            n_list[best_lat] += 1
            n = tuple(n_list)

        self.repeats = n
        return self.atoms

    # ------------------------------------------------------------------
    # Centre-of-mass translation
    # ------------------------------------------------------------------

    def center_to_origin(self) -> Atoms:
        """Shift the supercell so its exact physical center of mass is at the origin."""
        if self.atoms is None:
            raise RuntimeError("No supercell built; call build_supercell() first.")

        # We MUST use the exact center of mass. Integer lattice shifting creates
        # an eccentric rotation origin for odd-numbered supercells, causing
        # the crystal to swing out of the simulation box bounds during rotation!
        com = self.atoms.get_center_of_mass()
        self.atoms.translate(-com)
        return self.atoms

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(self, margin: int = 2, verbose: bool = True) -> CrystalResult:
        """Execute the full Module-2 pipeline.

        1. Replicate the unit cell to fill the target coverage extent.
        2. Shift the centre of mass to the origin.
        """
        if verbose:
            print("=== Module 2: Pristine Crystal Generation ===")
            print(f"  Box:        {self.box_start} -> {self.box_end}")
            print(f"  Box size:   {self.box_size}")
            print(f"  Coverage:   {self.coverage}")
            print(f"  Unit cell:  {self.unit_cell.symbols}")
            print(f"  Unit atoms: {len(self.unit_cell)}")

        self.build_supercell(margin=margin)

        if verbose:
            print(f"  Repeats:    {self.repeats}")
            print(f"  Supercell:  {len(self.atoms)} atoms")

        self.center_to_origin()

        # Store unit-cell vectors for the GUI wireframe renderer
        self.atoms.info["_unit_cell"] = self.unit_cell.get_cell()[:]

        pos = self.atoms.get_positions()
        if verbose:
            print(f"  COM:        {self.atoms.get_center_of_mass()}  (should be origin)")
            print(f"  Bounds:     min {pos.min(axis=0)},  max {pos.max(axis=0)}")
            print(f"  Extent:     {pos.max(axis=0) - pos.min(axis=0)}")
            print(f"=== Module 2 complete ===\n")

        return CrystalResult(
            atoms=self.atoms.copy(),
            unit_cell=self.unit_cell.copy(),
            box_start=self.box_start.copy(),
            box_end=self.box_end.copy(),
            box_size=self.box_size.copy(),
            repeats=self.repeats,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Write the supercell to a file (format guessed from extension).

        Supported formats: ``.xyz``, ``.cif``, ``.vasp``, ``.lammps-data``,
        ``.extxyz``, etc.  See ``ase.io.write`` for the full list.
        """
        if self.atoms is None:
            raise RuntimeError("No supercell to save; call run() first.")
        from ase.io import write as ase_write
        ase_write(filepath, self.atoms)

    @staticmethod
    def load(filepath: str) -> Atoms:
        """Read an Atoms object from any ASE-supported format."""
        return ase_read(filepath)


# ---------------------------------------------------------------------------
# Quick-entry helpers
# ---------------------------------------------------------------------------

def generate_pristine_bravais(
    symbol: str,
    crystalstructure: str | None = None,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    *,
    box_start: ArrayLike = (0.0, 0.0, 0.0),
    box_end: ArrayLike = (100.0, 100.0, 100.0),
    coverage: float | ArrayLike | None = None,
    margin: int = 2,
    verbose: bool = True,
    **kwargs,
) -> CrystalResult:
    """Shortcut: build a Bravais-lattice pristine crystal."""
    pc = PristineCrystal.from_bravais(
        symbol=symbol,
        crystalstructure=crystalstructure,
        a=a, b=b, c=c,
        box_start=box_start,
        box_end=box_end,
        coverage=coverage,
        **kwargs,
    )
    return pc.run(margin=margin, verbose=verbose)


def generate_pristine_cif(
    cif_path: str,
    box_start: ArrayLike = (0.0, 0.0, 0.0),
    box_end: ArrayLike = (100.0, 100.0, 100.0),
    coverage: float | ArrayLike | None = None,
    margin: int = 2,
    verbose: bool = True,
) -> CrystalResult:
    """Shortcut: build a pristine crystal from a CIF file."""
    pc = PristineCrystal.from_cif(
        cif_path, box_start=box_start, box_end=box_end, coverage=coverage,
    )
    return pc.run(margin=margin, verbose=verbose)


def generate_pristine_intermetallic(
    strukturbericht: str,
    symbols: Sequence[str] | None = None,
    a: float | None = None,
    c: float | None = None,
    cellpar: Sequence[float] | None = None,
    box_start: ArrayLike = (0.0, 0.0, 0.0),
    box_end: ArrayLike = (100.0, 100.0, 100.0),
    coverage: float | ArrayLike | None = None,
    margin: int = 2,
    verbose: bool = True,
) -> CrystalResult:
    """Shortcut: build an intermetallic pristine crystal."""
    pc = PristineCrystal.from_intermetallic(
        strukturbericht=strukturbericht,
        symbols=symbols,
        a=a, c=c,
        cellpar=cellpar,
        box_start=box_start,
        box_end=box_end,
        coverage=coverage,
    )
    return pc.run(margin=margin, verbose=verbose)


# ---------------------------------------------------------------------------
# Coverage helper
# ---------------------------------------------------------------------------


def max_neighbor_distance(
    seeds: np.ndarray,
    neighbors: list[list[int]],
    box_size: ArrayLike,
) -> float:
    """Return the maximum PBC-aware distance between any two Voronoi neighbours.

    The required crystal coverage is ``2 * max_neighbor_distance(...)``:
    after COM centering the crystal extends coverage/2 in every direction
    from the seed, fully enclosing the farthest neighbour.

    Parameters
    ----------
    seeds : (N, 3) ndarray
        Grain seed coordinates (from Module 1).
    neighbors : list[list[int]]
        Voronoi adjacency per grain (``grains.neighbors``).
    box_size : (3,) array-like
        Simulation box dimensions (``box_end - box_start``).

    Returns
    -------
    d_max : float
        Largest PBC minimum-image distance between any two Voronoi neighbours.
    """
    box = np.asarray(box_size, dtype=float)
    d_max = 0.0
    for i, nbrs in enumerate(neighbors):
        si = seeds[i]
        for j in nbrs:
            if i < j:  # each pair once
                delta = si - seeds[j]
                delta -= box * np.round(delta / box)
                d_max = max(d_max, float(np.linalg.norm(delta)))
    if d_max == 0.0:
        # No neighbours (e.g. single grain): Voronoi cell spans the box.
        d_max = float(np.linalg.norm(box))
    return d_max


# ---------------------------------------------------------------------------
# d-spacing helper (used by laminate feature)
# ---------------------------------------------------------------------------

def calculate_d_hkl(unit_cell: Atoms, hkl: tuple[int, int, int]) -> float:
    """Interplanar spacing for plane *(hkl)* in *unit_cell* (Å).

    Uses the reciprocal-lattice formula:
       d = 2π / |G|,   where G = h·b₁ + k·b₂ + l·b₃
    and b₁, b₂, b₃ are the reciprocal basis vectors derived from the
    real-space cell matrix.
    """
    C = unit_cell.get_cell()          # 3×3 (row-major)
    B = 2.0 * np.pi * np.linalg.inv(C).T   # reciprocal basis (rows)
    h, k, l = hkl
    G = h * B[0] + k * B[1] + l * B[2]
    G_norm = float(np.linalg.norm(G))
    if G_norm == 0.0:
        return float("inf")
    return 2.0 * np.pi / G_norm


def get_hkl_spacing(
    crystal_source: int,
    params: dict,
    hkl: tuple[int, int, int],
) -> float:
    """Build a tiny PristineCrystal from *params* and return d-spacing.

    Uses the same routing as :func:`_generate_full_crystal` in ``gui_main.py``
    but with ``coverage=1.0`` so the supercell is a minimal single unit cell.
    """
    pc = _build_for_hkl(crystal_source, params)
    return calculate_d_hkl(pc.unit_cell, hkl)


def _build_for_hkl(crystal_source: int, params: dict) -> "PristineCrystal":
    """Minimal crystal construction for d-spacing calculation."""
    box_start = params.get("box_start", (0.0, 0.0, 0.0))
    box_end = params.get("box_end", (100.0, 100.0, 100.0))

    if crystal_source == 0:  # Bravais
        struct = params["single_structure"]
        kw = {}
        if struct in ("hcp",):
            kw["orthorhombic"] = True
        elif struct in ("sc", "fcc", "bcc", "diamond",
                         "rocksalt", "zincblende"):
            kw["cubic"] = True
        return PristineCrystal.from_bravais(
            symbol=params["single_element"].strip(),
            crystalstructure=struct,
            a=params["single_a"],
            c=params["single_c"],
            box_start=box_start,
            box_end=box_end,
            coverage=1.0,
            **kw,
        )

    if crystal_source == 1:  # Intermetallics
        inter_type = params["inter_type"]
        symbols = [s.strip() for s in params["inter_elements"].split(",") if s.strip()]
        if inter_type in ("rocksalt", "zincblende",
                            "cesiumchloride", "fluorite", "wurtzite"):
            formula = "".join(symbols) if symbols else params["inter_elements"]
            struct = inter_type
            kw = {"cubic": True} if struct not in ("hcp", "wurtzite") else {"orthorhombic": True}
            return PristineCrystal.from_bravais(
                symbol=formula, crystalstructure=struct,
                a=params["inter_a"], c=params["inter_c"],
                box_start=box_start, box_end=box_end,
                coverage=1.0, **kw,
            )
        return PristineCrystal.from_intermetallic(
            strukturbericht=inter_type, symbols=symbols,
            a=params["inter_a"], c=params["inter_c"],
            box_start=box_start, box_end=box_end,
            coverage=1.0,
        )

    if crystal_source == 2:  # Spacegroup
        raw = params["sg_basis"].strip()
        flat = [float(x) for x in raw.replace("(", "").replace(")", "").split(",")]
        basis = [(flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat), 3)]
        sg = params["sg_spacegroup"].strip()
        try:
            sg = int(sg)
        except ValueError:
            pass
        cellpar = [float(x.strip()) for x in params["sg_cellpar"].split(",")]
        return PristineCrystal.from_spacegroup(
            symbols=[s.strip() for s in params["sg_elements"].split(",") if s.strip()],
            basis=basis, spacegroup=sg, cellpar=cellpar,
            box_start=box_start, box_end=box_end,
            coverage=1.0,
        )

    # crystal_source == 3: CIF file
    return PristineCrystal.from_cif(
        cif_path=params["custom_file"],
        box_start=box_start, box_end=box_end,
        coverage=1.0,
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke test: fcc Aluminium
    result = generate_pristine_bravais(
        symbol="Al",
        crystalstructure="fcc",
        a=4.05,
        box_start=(0, 0, 0),
        box_end=(50, 50, 50),
        margin=2,
    )
    print(f"Built {result.n_atoms} atoms, formula = {result.chemical_formula}")
