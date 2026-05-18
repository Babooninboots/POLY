"""
Module 3: Crystallographic Orientation Assignment.

Computes rotation matrices and Euler angles for each grain using
``scipy.spatial.transform.Rotation``.  Supports six assignment modes:

  random                      -- random Euler angles for all grains
  z-axis alignment (hkl)      -- align (hkl) plane normal with Z, random in-plane
  low angle                   -- BFS traversal, relative misorientation < 10 deg
  high angle                  -- BFS traversal, relative misorientation > 20 deg
  custom misorientation       -- MC optimization to match target angle distribution
  custom orientation profile  -- direct assignment from user-provided map
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Crystal symmetry operators (proper rotations, determinant +1)
# ---------------------------------------------------------------------------


def _build_cubic_sym_ops() -> Rotation:
    """Return the 24 proper rotational symmetry operators for cubic crystals."""
    ops = []
    # Identity
    ops.append(np.eye(3))
    # 90, 180, 270 deg about x, y, z axes
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        for angle_deg in (90, 180, 270):
            ops.append(Rotation.from_rotvec(
                np.radians(angle_deg) * np.array(axis, dtype=float)
            ).as_matrix())
    # 180 deg about face diagonals: [110], [1-10], [101], [10-1], [011], [01-1]
    for diag in ([1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]):
        v = np.array(diag, dtype=float)
        v /= np.linalg.norm(v)
        ops.append(Rotation.from_rotvec(np.pi * v).as_matrix())
    # 120, 240 deg about body diagonals: [111], [1-1-1], [-11-1], [-1-11]
    for diag in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        v = np.array(diag, dtype=float)
        v /= np.linalg.norm(v)
        for angle_deg in (120, 240):
            ops.append(Rotation.from_rotvec(
                np.radians(angle_deg) * v
            ).as_matrix())
    return Rotation.from_matrix(np.stack(ops, axis=0))


def _build_hexagonal_sym_ops() -> Rotation:
    """Return the 12 proper rotational symmetry operators for hexagonal crystals."""
    ops = []
    # Identity
    ops.append(np.eye(3))
    # 60, 120, 180, 240, 300 deg about z-axis
    for angle_deg in (60, 120, 180, 240, 300):
        ops.append(Rotation.from_rotvec(
            np.radians(angle_deg) * np.array([0.0, 0.0, 1.0])
        ).as_matrix())
    # 180 deg about 6 directions in basal plane: at 0, 30, 60, 90, 120, 150 deg
    for theta_deg in (0, 30, 60, 90, 120, 150):
        theta = np.radians(theta_deg)
        v = np.array([np.cos(theta), np.sin(theta), 0.0])
        ops.append(Rotation.from_rotvec(np.pi * v).as_matrix())
    return Rotation.from_matrix(np.stack(ops, axis=0))


def _build_tetragonal_sym_ops() -> Rotation:
    """Return the 8 proper rotational symmetry operators for tetragonal crystals."""
    ops = []
    # Identity
    ops.append(np.eye(3))
    # 90, 180, 270 deg about z-axis
    for angle_deg in (90, 180, 270):
        ops.append(Rotation.from_rotvec(
            np.radians(angle_deg) * np.array([0.0, 0.0, 1.0])
        ).as_matrix())
    # 180 deg about x, y axes
    for axis in ([1, 0, 0], [0, 1, 0]):
        ops.append(Rotation.from_rotvec(np.pi * np.array(axis, dtype=float)).as_matrix())
    # 180 deg about face diagonals: [110], [1-10]
    for diag in ([1, 1, 0], [1, -1, 0]):
        v = np.array(diag, dtype=float)
        v /= np.linalg.norm(v)
        ops.append(Rotation.from_rotvec(np.pi * v).as_matrix())
    return Rotation.from_matrix(np.stack(ops, axis=0))


def _build_orthorhombic_sym_ops() -> Rotation:
    """Return the 4 proper rotational symmetry operators for orthorhombic crystals."""
    ops = []
    # Identity
    ops.append(np.eye(3))
    # 180 deg about x, y, z axes
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        ops.append(Rotation.from_rotvec(np.pi * np.array(axis, dtype=float)).as_matrix())
    return Rotation.from_matrix(np.stack(ops, axis=0))


def _build_triclinic_sym_ops() -> Rotation:
    """Return only the identity operator (triclinic, no symmetry)."""
    return Rotation.from_matrix(np.eye(3).reshape(1, 3, 3))


# Cached symmetry operators
_SYM_OPS_CACHE: dict[str, Rotation] = {
    "cubic":        _build_cubic_sym_ops(),
    "hexagonal":    _build_hexagonal_sym_ops(),
    "tetragonal":   _build_tetragonal_sym_ops(),
    "orthorhombic": _build_orthorhombic_sym_ops(),
    "triclinic":    _build_triclinic_sym_ops(),
}

# Map common structure names → crystal system
_STRUCTURE_SYMMETRY: dict[str, str] = {
    # Bravais
    "sc": "cubic", "fcc": "cubic", "bcc": "cubic", "diamond": "cubic",
    "hcp": "hexagonal", "bct": "tetragonal",
    # Intermetallics / compounds
    "rocksalt": "cubic", "zincblende": "cubic",
    "cesiumchloride": "cubic", "fluorite": "cubic",
    "L1_2": "cubic", "B2": "cubic", "D0_3": "cubic", "L2_1": "cubic",
    "A15": "cubic", "C15": "cubic",
    "L1_0": "tetragonal", "D0_19": "hexagonal",
    "wurtzite": "hexagonal",
}


def get_crystal_symmetry(crystal_structure: str | None = None,
                         spacegroup: int | None = None) -> str:
    """Determine the crystal system from structure name or spacegroup number.

    Returns one of ``"cubic"``, ``"hexagonal"``, ``"tetragonal"``,
    ``"orthorhombic"``, or ``"triclinic"``.  Defaults to ``"cubic"``.
    """
    if crystal_structure is not None:
        sym = _STRUCTURE_SYMMETRY.get(crystal_structure)
        if sym is not None:
            return sym
    if spacegroup is not None:
        if 195 <= spacegroup <= 230:
            return "cubic"
        if 168 <= spacegroup <= 194:
            return "hexagonal"
        if 75 <= spacegroup <= 142:
            return "tetragonal"
        if 16 <= spacegroup <= 74:
            return "orthorhombic"
    return "cubic"  # default


def get_sym_ops(crystal_system: str) -> Rotation:
    """Return the symmetry operators for *crystal_system* as a Rotation object."""
    if crystal_system not in _SYM_OPS_CACHE:
        raise ValueError(
            f"Unknown crystal system '{crystal_system}'. "
            f"Choose from: {list(_SYM_OPS_CACHE.keys())}"
        )
    return _SYM_OPS_CACHE[crystal_system]


def symmetry_from_atoms(atoms) -> str | None:
    """Detect the crystal system from an ASE Atoms object.

    Checks in order:
    1. ``atoms.info["crystal_system"]`` — stored by POLY's .crystal loader
    2. ``atoms.info["spacegroup"]`` — populated by ASE for CIF / spacegroup

    Returns ``None`` if no symmetry information is present.
    """
    if not hasattr(atoms, "info"):
        return None
    # POLY .crystal header: explicit crystal system string
    cs = atoms.info.get("crystal_system")
    if cs is not None:
        if cs in _SYM_OPS_CACHE:
            return cs
    # ASE spacegroup info (from CIF or spacegroup crystal)
    sg_info = atoms.info.get("spacegroup")
    if sg_info is not None:
        if isinstance(sg_info, int):
            sg = sg_info
        else:
            try:
                sg = int(str(sg_info).split()[0])
            except (ValueError, IndexError):
                return None
        return get_crystal_symmetry(spacegroup=sg)
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

def get_cubic_csl_angles(hkl: tuple[int, int, int], target_sigma: int) -> list[float]:
    """Calculate all valid symmetric tilt angles (radians) for a cubic CSL boundary."""
    N = sum(x**2 for x in hkl)
    if N == 0:
        return []

    angles = []
    for m in range(1, 100):
        for n in range(1, m + 1):
            if math.gcd(m, n) != 1:
                continue
            s = m**2 + N * (n**2)
            while s % 2 == 0 and s > 0:
                s //= 2
            if s == target_sigma:
                angles.append(2.0 * np.arctan((n / m) * np.sqrt(N)))

    if not angles:
        return []

    angles.sort()
    unique_angles = [angles[0]]
    for ang in angles[1:]:
        if ang - unique_angles[-1] > 1e-6:
            unique_angles.append(ang)
    return unique_angles


@dataclass
class OrientationResult:
    """Container returned by OrientationAssigner.run()."""

    euler_angles: np.ndarray         # (N, 3) zxz Euler angles (degrees)
    rotation_matrices: np.ndarray    # (N, 3, 3) rotation matrices
    rotations: Rotation              # scipy Rotation object for all grains
    mode: str
    neighbors: list[list[int]] | None = None
    misorientation_angles: np.ndarray | None = None  # per-edge misorientation (deg)
    mc_energy_history: list[float] | None = None      # custom misorientation MC
    mc_acceptance_rate: float | None = None


# ---------------------------------------------------------------------------
# Orientation assigner
# ---------------------------------------------------------------------------


class OrientationAssigner:
    """Assign crystallographic orientations to grains.

    Parameters
    ----------
    mode : str
        One of ``"random"``, ``"z_alignment"``, ``"low_angle"``,
        ``"high_angle"``, ``"custom_misorientation"``, ``"custom_profile"``.
    n_grains : int
        Number of grains.
    neighbors : list[list[int]], optional
        Voronoi adjacency list.  Required for ``low_angle``, ``high_angle``,
        and ``custom_misorientation``.
    random_seed : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        mode: str,
        n_grains: int,
        neighbors: list[list[int]] | None = None,
        random_seed: int | None = None,
        crystal_structure: str | None = None,
        crystal_system: str | None = None,
        spacegroup: int | None = None,
    ):
        if mode not in (
            "random",
            "z_alignment",
            "low_angle",
            "high_angle",
            "custom_misorientation",
            "custom_profile",
        ):
            raise ValueError(f"Unknown orientation mode: '{mode}'")

        self.mode = mode
        self.n_grains = n_grains
        self.neighbors = neighbors
        self.rng = np.random.default_rng(random_seed)

        # Determine crystal symmetry for misorientation calculation
        if crystal_system is not None:
            self.crystal_system = crystal_system
        elif crystal_structure is not None or spacegroup is not None:
            self.crystal_system = get_crystal_symmetry(crystal_structure, spacegroup)
        else:
            self.crystal_system = "cubic"
        self.sym_ops = get_sym_ops(self.crystal_system)

        # Validate prerequisites
        if mode in ("low_angle", "high_angle", "custom_misorientation"):
            if neighbors is None:
                raise ValueError(
                    f"Mode '{mode}' requires a Voronoi neighbor list."
                )

    # ------------------------------------------------------------------
    # Mode: random
    # ------------------------------------------------------------------

    def _assign_random(self) -> Rotation:
        return Rotation.random(self.n_grains, random_state=self.rng)

    # ------------------------------------------------------------------
    # Mode: z-axis alignment
    # ------------------------------------------------------------------

    def _assign_z_alignment(
        self, hkl: tuple[int, int, int], in_plane: str = "random",
        csl_angle_deg: float = 0.0,
    ) -> Rotation:
        """Align (hkl) plane normal with global Z, then assign in-plane rotation.

        *in_plane* modes:
          ``"random"``     -- uniform random angle per grain
          ``"low_angle"``  -- BFS traversal, twist difference < 10°
          ``"high_angle"`` -- BFS traversal, twist difference > 20°
        """
        n = np.array(hkl, dtype=float)
        n_norm = np.linalg.norm(n)
        z_crys = n / n_norm if n_norm > 0 else np.array([0.0, 0.0, 1.0])

        # Scan for a high-symmetry orthogonal vector
        x_crys = None
        candidates = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, -1, 0], [1, 1, 0], [1, 0, -1], [0, 1, -1],
            [1, 1, -2], [1, -2, 1], [-2, 1, 1]
        ]
        for cand in candidates:
            c = np.array(cand, dtype=float)
            c_norm = np.linalg.norm(c)
            if c_norm > 1e-6:
                c /= c_norm
                if abs(np.dot(z_crys, c)) < 1e-8:
                    x_crys = c
                    break

        # Fallback for completely arbitrary irrational tilt axes
        if x_crys is None:
            arb = np.array([1.0, 0.0, 0.0]) if abs(z_crys[0]) < 0.99 else np.array([0.0, 1.0, 0.0])
            x_crys = np.cross(z_crys, arb)
            x_crys /= np.linalg.norm(x_crys)

        y_crys = np.cross(z_crys, x_crys)
        y_crys /= np.linalg.norm(y_crys)

        # Build rotation matrix from row vectors to align crystal axes to global axes
        R_base = Rotation.from_matrix(np.vstack([x_crys, y_crys, z_crys]))

        # In-plane rotation per grain
        angles = np.zeros(self.n_grains)
        if in_plane == "random":
            angles = self.rng.uniform(0.0, 2.0 * np.pi, size=self.n_grains)
        elif in_plane == "symmetric_csl":
            best_theta = np.radians(csl_angle_deg)
            angles = np.zeros(self.n_grains)
            for i in range(self.n_grains):
                angles[i] = (best_theta / 2.0) if i % 2 == 0 else (-best_theta / 2.0)
        else:
            if self.neighbors is None:
                raise ValueError(
                    "Neighbors required for low/high angle in-plane rotation."
                )
            unvisited = set(range(self.n_grains))
            while unvisited:
                start = min(unvisited)
                unvisited.discard(start)
                angles[start] = self.rng.uniform(0.0, 2.0 * np.pi)
                queue = [start]
                while queue:
                    parent = queue.pop(0)
                    for child in self.neighbors[parent]:
                        if child not in unvisited:
                            continue
                        best_cand = angles[parent]
                        for _ in range(100):
                            if in_plane == "low_angle":
                                delta = self.rng.uniform(-10.0, 10.0)
                            else:  # high_angle
                                sign = self.rng.choice([-1, 1])
                                delta = sign * self.rng.uniform(20.0, 90.0)
                            cand = angles[parent] + np.radians(delta)
                            ok = True
                            for nbr in self.neighbors[child]:
                                if nbr not in unvisited and nbr != child:
                                    # Shortest angular difference
                                    diff = np.degrees(
                                        np.abs(
                                            (cand - angles[nbr] + np.pi) % (2 * np.pi)
                                            - np.pi
                                        )
                                    )
                                    if in_plane == "low_angle" and diff > 10.0:
                                        ok = False
                                    elif in_plane == "high_angle" and diff < 20.0:
                                        ok = False
                            if ok:
                                best_cand = cand
                                break
                        angles[child] = best_cand
                        unvisited.discard(child)
                        queue.append(child)

        z_axis = np.array([0.0, 0.0, 1.0])
        Rz = Rotation.from_rotvec(
            angles[:, None] * z_axis[None, :]
        )

        return Rz * R_base  # apply R_base first, then Rz

    # ------------------------------------------------------------------
    # Mode: low angle / high angle (BFS graph traversal)
    # ------------------------------------------------------------------

    def _misorientation_between(self, R1: Rotation, R2: Rotation) -> float:
        """Crystallographic misorientation angle (deg) between two rotations.

        Computes the minimum angle over all symmetrically equivalent relative
        rotations:  min_i |R1 · S_i · R2⁻¹|  where S_i are the symmetry
        operators for the crystal system (24 for cubic, 12 for hexagonal, etc.).
        """
        # ΔR = R1 · S_i · R2⁻¹  for each symmetry operator S_i
        # Broadcasting: R1 (N,), sym_ops (M,), R2_inv (N,) → result (M,)
        R2_inv = R2.inv()
        delta_all = R1 * self.sym_ops * R2_inv  # (M,)  or (M, N) for multi-grain
        return float(np.degrees(np.min(delta_all.magnitude())))

    def _assign_angle_bfs(
        self,
        max_angle: float | None = None,
        min_angle: float | None = None,
    ) -> Rotation:
        """BFS traversal assigning relative rotations within angle bounds.

        At each step the candidate rotation is checked against **all**
        already-assigned neighbors (not just the tree parent), so every
        Voronoi edge satisfies the constraint.

        Parameters
        ----------
        max_angle : float, optional
            Upper bound in degrees (for ``low_angle``).
        min_angle : float, optional
            Lower bound in degrees (for ``high_angle``).
        """
        n = self.n_grains
        neighbors = self.neighbors

        rotations: list[Rotation | None] = [None] * n
        unvisited = set(range(n))

        max_retries = 200

        while unvisited:
            start = min(unvisited)
            unvisited.discard(start)
            rotations[start] = Rotation.random(random_state=self.rng)
            queue = [start]

            while queue:
                parent = queue.pop(0)
                R_parent = rotations[parent]
                for child in neighbors[parent]:
                    if child not in unvisited:
                        continue
                    # Try to find a rotation that satisfies the constraint
                    # against all already-assigned neighbors
                    best_R = None
                    best_violation = float("inf")
                    for _ in range(max_retries):
                        R_rel = self._random_relative_rotation(
                            max_angle=max_angle, min_angle=min_angle
                        )
                        R_candidate = R_rel * R_parent
                        ok = True
                        worst = 0.0
                        for nbr in neighbors[child]:
                            if rotations[nbr] is not None:
                                miso = self._misorientation_between(
                                    R_candidate, rotations[nbr]
                                )
                                if max_angle is not None and miso > max_angle:
                                    ok = False
                                    worst = max(worst, miso - max_angle)
                                if min_angle is not None and miso < min_angle:
                                    ok = False
                                    worst = max(worst, min_angle - miso)
                        if ok:
                            best_R = R_candidate
                            break
                        if worst < best_violation:
                            best_violation = worst
                            best_R = R_candidate

                    unvisited.discard(child)
                    rotations[child] = best_R
                    queue.append(child)

        return Rotation.from_matrix(
            np.stack([r.as_matrix() for r in rotations], axis=0)
        )

    def _random_relative_rotation(
        self,
        max_angle: float | None = None,
        min_angle: float | None = None,
    ) -> Rotation:
        """Generate a random rotation with angle in the requested range."""
        axis = self.rng.normal(0.0, 1.0, 3)
        axis /= np.linalg.norm(axis)

        if max_angle is not None:
            angle_deg = self.rng.uniform(0.0, max_angle)
        elif min_angle is not None:
            angle_deg = self.rng.uniform(min_angle, 180.0)
        else:
            angle_deg = self.rng.uniform(0.0, 180.0)

        return Rotation.from_rotvec(np.radians(angle_deg) * axis)

    def _assign_low_angle(self) -> Rotation:
        return self._assign_angle_bfs(max_angle=10.0)

    def _assign_high_angle(self) -> Rotation:
        return self._assign_angle_bfs(min_angle=20.0)

    # ------------------------------------------------------------------
    # Mode: custom misorientation (MC optimization)
    # ------------------------------------------------------------------

    def _compute_pairwise_misorientation(self, rotations: Rotation) -> np.ndarray:
        """Crystallographic misorientation angle (deg) between each neighbour pair.

        Accounts for crystal symmetry by evaluating all symmetry-equivalent
        relative rotations and taking the minimum angle.
        """
        edges: list[float] = []

        for i, nbrs in enumerate(self.neighbors):
            R_i = rotations[i]
            R_i_inv = R_i.inv()
            for j in nbrs:
                if i < j:  # count each edge once
                    R_j = rotations[j]
                    delta_all = R_i * self.sym_ops * R_j.inv()
                    edges.append(float(np.degrees(np.min(delta_all.magnitude()))))

        return np.array(edges) if edges else np.zeros(0)

    def _misorientation_energy(
        self, rotations: Rotation, target_angles: np.ndarray
    ) -> float:
        """KS statistic between current and target misorientation distributions."""
        current = self._compute_pairwise_misorientation(rotations)
        if len(current) < 5 or len(target_angles) < 5:
            # Not enough data for KS test
            if len(current) == 0 or len(target_angles) == 0:
                return 1.0
            # Fallback: difference of means
            return float(
                min(1.0, np.abs(np.mean(current) - np.mean(target_angles)) / 180.0)
            )
        from scipy.stats import ks_2samp

        stat, _ = ks_2samp(current, target_angles)
        return float(stat)

    def _assign_custom_misorientation(
        self,
        target_angles: np.ndarray,
        max_steps: int = 2000,
        temp_start: float = 1.0,
        temp_end: float = 0.001,
        cooling_rate: float = 0.995,
        perturbation_std: float = 5.0,
        threshold: float = 0.05,
    ) -> tuple[Rotation, list[float], float]:
        """MC optimization to match a target misorientation-angle distribution.

        Parameters
        ----------
        target_angles : (M,) ndarray
            Sample of target misorientation angles (degrees).
        max_steps : int
            Maximum MC iterations.
        temp_start, temp_end, cooling_rate : float
            Simulated annealing schedule.
        perturbation_std : float
            Std dev of perturbation angle (degrees).
        threshold : float
            Early-stop KS energy threshold.

        Returns
        -------
        best_rotations : Rotation
        energy_history : list[float]
        acceptance_rate : float
        """
        n = self.n_grains

        rotations = Rotation.random(n, random_state=self.rng)
        current_energy = self._misorientation_energy(rotations, target_angles)

        best_rotations = Rotation.from_matrix(rotations.as_matrix())
        best_energy = current_energy

        T = temp_start
        energy_history = [current_energy]
        accepted = 0

        for step in range(max_steps):
            T = max(temp_end, T * cooling_rate)

            # Perturb one randomly chosen grain
            i = int(self.rng.integers(0, n))
            axis = self.rng.normal(0.0, 1.0, 3)
            axis /= np.linalg.norm(axis)
            angle = self.rng.normal(0.0, perturbation_std)
            R_perturb = Rotation.from_rotvec(np.radians(angle) * axis)

            mat = rotations.as_matrix()
            mat[i] = (R_perturb * rotations[i]).as_matrix()
            new_rotations = Rotation.from_matrix(mat)

            new_energy = self._misorientation_energy(new_rotations, target_angles)
            delta_e = new_energy - current_energy

            accept = delta_e <= 0 or (
                T > 0 and self.rng.random() < np.exp(-delta_e / T)
            )

            if accept:
                rotations = new_rotations
                current_energy = new_energy
                accepted += 1

                if current_energy < best_energy:
                    best_rotations = Rotation.from_matrix(mat)
                    best_energy = current_energy

            energy_history.append(current_energy)

            if current_energy < threshold:
                break

        acceptance_rate = accepted / (step + 1)
        return best_rotations, energy_history, acceptance_rate

    # ------------------------------------------------------------------
    # Mode: custom profile
    # ------------------------------------------------------------------

    def _assign_custom_profile(
        self, euler_map: dict[int, tuple[float, float, float]]
    ) -> Rotation:
        """Assign Euler angles from a user-provided grain->(phi1,theta,phi2) map.

        Grains not present in the map get identity (zero) rotation.
        """
        euler_deg = np.zeros((self.n_grains, 3), dtype=float)
        for i, angles in euler_map.items():
            euler_deg[int(i)] = np.asarray(angles, dtype=float)
        return Rotation.from_euler("zxz", euler_deg, degrees=True)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        hkl: tuple[int, int, int] | None = None,
        in_plane: str = "random",
        csl_sigma: int = 5,
        csl_angle_deg: float = 0.0,
        target_angles: np.ndarray | None = None,
        euler_map: dict[int, tuple[float, float, float]] | None = None,
        verbose: bool = True,
        **mc_kwargs,
    ) -> OrientationResult:
        """Execute orientation assignment.

        Parameters
        ----------
        hkl : tuple, required for ``z_alignment``
        in_plane : str, ``"random"``, ``"low_angle"`` or ``"high_angle"``
            In-plane rotation mode for ``z_alignment``.
        target_angles : ndarray, required for ``custom_misorientation``
        euler_map : dict, required for ``custom_profile``
        verbose : bool
        **mc_kwargs : forwarded to the custom-misorientation MC optimizer.
        """
        energy_history = None
        acceptance_rate = None

        if verbose:
            print(f"=== Module 3: Crystallographic Orientation Assignment ===")
            print(f"  Mode:        {self.mode}")
            print(f"  N grains:    {self.n_grains}")

        if self.mode == "random":
            rotations = self._assign_random()

        elif self.mode == "z_alignment":
            if hkl is None:
                raise ValueError("hkl is required for 'z_alignment' mode.")
            if verbose:
                print(f"  (hkl):       {hkl}")
                print(f"  in-plane:    {in_plane}")
            rotations = self._assign_z_alignment(hkl, in_plane, csl_angle_deg=csl_angle_deg)

        elif self.mode == "low_angle":
            rotations = self._assign_low_angle()
            if verbose:
                misos = self._compute_pairwise_misorientation(rotations)
                if len(misos) > 0:
                    print(
                        f"  Max pairwise misorientation: {np.max(misos):.2f} deg"
                    )

        elif self.mode == "high_angle":
            rotations = self._assign_high_angle()
            if verbose:
                misos = self._compute_pairwise_misorientation(rotations)
                if len(misos) > 0:
                    print(
                        f"  Min pairwise misorientation: {np.min(misos):.2f} deg"
                    )

        elif self.mode == "custom_misorientation":
            if target_angles is None:
                raise ValueError(
                    "target_angles is required for 'custom_misorientation' mode."
                )
            rotations, energy_history, acceptance_rate = (
                self._assign_custom_misorientation(
                    target_angles, **mc_kwargs
                )
            )
            if verbose:
                n_steps = len(energy_history) - 1
                print(
                    f"  MC finished: {n_steps} steps, "
                    f"accept rate = {acceptance_rate:.3f}"
                )
                print(f"  Final KS energy: {energy_history[-1]:.6f}")

        elif self.mode == "custom_profile":
            if euler_map is None:
                raise ValueError(
                    "euler_map is required for 'custom_profile' mode."
                )
            rotations = self._assign_custom_profile(euler_map)
            if verbose:
                print(f"  Assigned {len(euler_map)} explicit orientations")

        else:
            raise ValueError(f"Unhandled mode: {self.mode}")

        # Compute final misorientation for diagnostic / output
        misorientation_angles = None
        if self.neighbors is not None:
            misorientation_angles = self._compute_pairwise_misorientation(rotations)

        euler_angles = rotations.as_euler("zxz", degrees=True)
        rotation_matrices = rotations.as_matrix()

        if verbose:
            print(
                f"  Euler angles (zxz, deg)  -  "
                f"phi1: [{np.min(euler_angles[:, 0]):.1f}, {np.max(euler_angles[:, 0]):.1f}],  "
                f"theta: [{np.min(euler_angles[:, 1]):.1f}, {np.max(euler_angles[:, 1]):.1f}],  "
                f"phi2: [{np.min(euler_angles[:, 2]):.1f}, {np.max(euler_angles[:, 2]):.1f}]"
            )
            print(f"=== Module 3 complete ===\n")

        return OrientationResult(
            euler_angles=euler_angles,
            rotation_matrices=rotation_matrices,
            rotations=rotations,
            mode=self.mode,
            neighbors=self.neighbors,
            misorientation_angles=misorientation_angles,
            mc_energy_history=energy_history,
            mc_acceptance_rate=acceptance_rate,
        )


# ---------------------------------------------------------------------------
# Quick-entry helper
# ---------------------------------------------------------------------------


def generate_orientations(
    mode: str,
    n_grains: int,
    neighbors: list[list[int]] | None = None,
    random_seed: int | None = None,
    crystal_structure: str | None = None,
    crystal_system: str | None = None,
    spacegroup: int | None = None,
    verbose: bool = True,
    **kwargs,
) -> OrientationResult:
    """Convenience wrapper: create an assigner, run the pipeline, return the result."""
    assigner = OrientationAssigner(
        mode=mode,
        n_grains=n_grains,
        neighbors=neighbors,
        random_seed=random_seed,
        crystal_structure=crystal_structure,
        crystal_system=crystal_system,
        spacegroup=spacegroup,
    )
    return assigner.run(verbose=verbose, **kwargs)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke test -- requires neighbors from grain_seeds
    from grain_seeds import generate_grains

    grains = generate_grains(
        box_start=(0, 0, 0),
        box_end=(100, 100, 100),
        n_grains=12,
        distribution="random",
        random_seed=42,
        verbose=False,
    )

    for mode in ["random", "z_alignment", "low_angle", "high_angle"]:
        kwargs = {}
        if mode == "z_alignment":
            kwargs["hkl"] = (1, 1, 1)
        result = generate_orientations(
            mode=mode,
            n_grains=grains.n_grains,
            neighbors=grains.neighbors,
            random_seed=42,
            **kwargs,
        )
        print(
            f"{mode:20s}  euler[0] = ({result.euler_angles[0, 0]:6.1f}, "
            f"{result.euler_angles[0, 1]:6.1f}, {result.euler_angles[0, 2]:6.1f})"
        )
