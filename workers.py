"""
POLY background workers — seed generation, crystal generation, polycrystal assembly.
"""

import threading

import numpy as np

from PySide6.QtCore import QThread, Signal


# ---------------------------------------------------------------------------
# File load helpers
# ---------------------------------------------------------------------------

def _load_seed_file(path: str):
    """Load a ``.seed`` file, returning ``(positions, radii, box_start, box_end)``.

    Parses ``# box_start:`` / ``# box_end:`` header lines if present.
    Detects 3-column (x y z) vs 4-column (x y z radius) format.
    """
    box_start = None
    box_end = None
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("# box_start:"):
                parts = line.split()
                box_start = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            elif line.startswith("# box_end:"):
                parts = line.split()
                box_end = np.array([float(parts[2]), float(parts[3]), float(parts[4])])

    data = np.loadtxt(path)
    if data.ndim == 2 and data.shape[1] >= 4:
        positions = data[:, :3]
        radii = data[:, 3]
    else:
        positions = data
        radii = None
    return positions, radii, box_start, box_end


def _load_crystal_file(path: str):
    """Load a ``.crystal`` file (element x y z) directly as ASE Atoms."""
    from ase import Atoms

    symbols_list = []
    positions_list = []
    cell_rows = []
    crystal_system = None
    with open(path, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                lowered = line.lower()
                if lowered.startswith("# crystal_system:"):
                    crystal_system = line.split(":", 1)[1].strip()
                elif lowered.startswith("# cell_"):
                    parts = line.split()
                    cell_rows.append([float(parts[2]), float(parts[3]), float(parts[4])])
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            symbols_list.append(parts[0])
            positions_list.append([float(parts[1]), float(parts[2]), float(parts[3])])

    cell = np.array(cell_rows) if len(cell_rows) == 3 else np.eye(3) * 10.0
    a = float(np.linalg.norm(cell[0]))
    b = float(np.linalg.norm(cell[1]))
    c_val = float(np.linalg.norm(cell[2]))
    alpha = float(np.degrees(np.arccos(
        np.dot(cell[1], cell[2]) / (b * c_val + 1e-30)
    )))
    beta = float(np.degrees(np.arccos(
        np.dot(cell[0], cell[2]) / (a * c_val + 1e-30)
    )))
    gamma = float(np.degrees(np.arccos(
        np.dot(cell[0], cell[1]) / (a * b + 1e-30)
    )))

    # Handle custom element names (e.g. "1") that ASE doesn't recognise
    from ase.data import atomic_numbers as _ase_atomic_numbers
    _has_custom = any(s not in _ase_atomic_numbers for s in symbols_list)
    if _has_custom:
        _unique_custom = list(dict.fromkeys(symbols_list))  # preserve order
        atoms = Atoms(
            numbers=[0] * len(positions_list),
            positions=np.array(positions_list),
            cell=cell, pbc=True,
        )
        atoms.info["_custom_element"] = _unique_custom[0] if len(_unique_custom) == 1 else ",".join(_unique_custom)
    else:
        atoms = Atoms(
            symbols=symbols_list,
            positions=np.array(positions_list),
            cell=cell, pbc=True,
        )
    atoms.info["_cell_params"] = dict(
        a=a, b=b, c=c_val, alpha=alpha, beta=beta, gamma=gamma,
    )
    if crystal_system:
        atoms.info["crystal_system"] = crystal_system
    return atoms


# ---------------------------------------------------------------------------
# Seed generation + orientation assignment
# ---------------------------------------------------------------------------

class SeedGenerationWorker(QThread):
    """Runs ``generate_grains`` + ``generate_orientations`` off the main thread.

    Emits ``finished(SeedResult, OrientationResult)`` on success or
    ``error(str)`` on failure.
    """

    finished = Signal(object, object)
    progress = Signal(int, str)
    error = Signal(str)

    def __init__(
        self,
        box_start: tuple[float, float, float],
        box_end: tuple[float, float, float],
        n_grains: int | None,
        avg_diameter: float | None,
        distribution: str,
        std_dev: float | None,
        seed_positions_file: str | None,
        orientation_mode: str,
        hkl: tuple[int, int, int] | None,
        ori_custom_file: str | None,
        in_plane: str = "random",
        csl_sigma: int = 5,
        csl_angle_deg: float = 0.0,
        laminate_in_plane_dist: str = "random",
        laminate_direction: str = "z",
        bimodal_params: tuple | None = None,
        crystal_structure: str | None = None,
        spacegroup: int | None = None,
        crystal_system: str | None = None,
        run_seeds: bool = True,
        run_ori: bool = True,
        cached_seed_result=None,
        cached_ori_result=None,
        parent=None,
    ):
        super().__init__(parent)
        self._box_start = box_start
        self._box_end = box_end
        self._n_grains = n_grains
        self._avg_diameter = avg_diameter
        self._distribution = distribution
        self._std_dev = std_dev
        self._seed_positions_file = seed_positions_file
        self._orientation_mode = orientation_mode
        self._hkl = hkl
        self._ori_custom_file = ori_custom_file
        self._in_plane = in_plane
        self._csl_sigma = csl_sigma
        self._csl_angle_deg = csl_angle_deg
        self._laminate_in_plane_dist = laminate_in_plane_dist
        self._laminate_direction = laminate_direction
        self._bimodal_params = bimodal_params
        self._crystal_structure = crystal_structure
        self._spacegroup = spacegroup
        self._crystal_system = crystal_system
        self._run_seeds = run_seeds
        self._run_ori = run_ori
        self._cached_seed = cached_seed_result
        self._cached_ori = cached_ori_result
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()  # unblock if paused

    def run(self) -> None:
        try:
            from grain_seeds import generate_grains
            from orientation import generate_orientations

            # --- load custom seed positions (if any) ---
            seed_positions = None
            seed_radii = None
            file_box_start = None
            file_box_end = None
            if self._distribution == "customized" and self._seed_positions_file:
                seed_positions, seed_radii, file_box_start, file_box_end = _load_seed_file(
                    self._seed_positions_file,
                )

            # --- step 1: grain seeds ---
            if self._run_seeds:
                def _progress_cb(step: int, total: int, energy: float) -> None:
                    pct = int((step / total) * 100)
                    msg = f"Optimizing... Step {step}/{total} | Energy: {energy:.4f}"
                    self.progress.emit(pct, msg)

                def _pause_cb(current_seeds, current_diameters) -> bool:
                    self._current_seeds = current_seeds.copy()
                    self._current_diameters = current_diameters.copy()
                    self._pause_event.wait()
                    return self._stop_event.is_set()

                seed_result = generate_grains(
                    box_start=file_box_start if file_box_start is not None else self._box_start,
                    box_end=file_box_end if file_box_end is not None else self._box_end,
                    n_grains=self._n_grains,
                    avg_diameter=self._avg_diameter,
                    distribution=self._distribution,
                    std_dev=self._std_dev,
                    seed_positions=seed_positions,
                    seed_radii=seed_radii,
                    laminate_in_plane_dist=self._laminate_in_plane_dist,
                    laminate_direction=self._laminate_direction,
                    bimodal_params=self._bimodal_params,
                    progress_callback=_progress_cb,
                    pause_callback=_pause_cb,
                    verbose=False,
                )
            else:
                seed_result = self._cached_seed

            # --- step 2: orientations ---
            if self._run_ori:
                ori_kwargs = {}
                if self._orientation_mode == "z_alignment" and self._hkl:
                    ori_kwargs["hkl"] = self._hkl
                    ori_kwargs["in_plane"] = self._in_plane
                    ori_kwargs["csl_sigma"] = self._csl_sigma
                    ori_kwargs["csl_angle_deg"] = self._csl_angle_deg
                elif self._orientation_mode == "custom_misorientation" and self._ori_custom_file:
                    ori_kwargs["target_angles"] = np.loadtxt(self._ori_custom_file)
                elif self._orientation_mode == "custom_profile" and self._ori_custom_file:
                    data = np.loadtxt(self._ori_custom_file)
                    euler_map = {
                        int(row[0]): (float(row[1]), float(row[2]), float(row[3]))
                        for row in data
                    }
                    ori_kwargs["euler_map"] = euler_map

                ori_result = generate_orientations(
                    mode=self._orientation_mode,
                    n_grains=seed_result.n_grains,
                    neighbors=seed_result.neighbors,
                    crystal_structure=self._crystal_structure,
                    crystal_system=self._crystal_system,
                    spacegroup=self._spacegroup,
                    verbose=False,
                    **ori_kwargs,
                )
            else:
                ori_result = self._cached_ori

            self.finished.emit(seed_result, ori_result)

        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Background worker — crystal generation
# ---------------------------------------------------------------------------

class CrystalGenerationWorker(QThread):
    """Runs pristine-crystal generation off the main thread.

    Emits ``finished_crystal(ase.Atoms, phase_index)`` on success or
    ``error_crystal(str)`` on failure.
    """

    finished_crystal = Signal(object, int)  # (atoms, phase_index)
    error_crystal = Signal(str)

    def __init__(
        self,
        crystal_source: int,  # 0=bravais, 1=intermetallic, 2=spacegroup, 3=file
        single_structure: str,
        single_element: str,
        single_a: float,
        single_c: float,
        inter_type: str,
        inter_elements: str,
        inter_a: float,
        inter_c: float,
        sg_elements: str,
        sg_basis: str,
        sg_spacegroup: str,
        sg_cellpar: str,
        custom_file: str,
        box_start: tuple[float, float, float],
        box_end: tuple[float, float, float],
        phase_index: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self._crystal_source = crystal_source
        self._single_structure = single_structure
        self._single_element = single_element
        self._single_a = single_a
        self._single_c = single_c
        self._inter_type = inter_type
        self._inter_elements = inter_elements
        self._inter_a = inter_a
        self._inter_c = inter_c
        self._sg_elements = sg_elements
        self._sg_basis = sg_basis
        self._sg_spacegroup = sg_spacegroup
        self._sg_cellpar = sg_cellpar
        self._custom_file = custom_file
        self._box_start = box_start
        self._box_end = box_end
        self._phase_index = phase_index

    def run(self) -> None:
        try:
            from pristine_crystal import PristineCrystal

            if self._crystal_source == 0:  # Single Crystal
                struct = self._single_structure
                if struct in ("hcp",):
                    kw = {"orthorhombic": True}
                elif struct in ("sc", "fcc", "bcc", "diamond",
                                "rocksalt", "zincblende"):
                    kw = {"cubic": True}
                else:
                    kw = {}
                pc = PristineCrystal.from_bravais(
                    symbol=self._single_element.strip(),
                    crystalstructure=struct,
                    a=self._single_a,
                    c=self._single_c,
                    **kw,
                )
            elif self._crystal_source == 1:  # Intermetallics
                inter_type = self._inter_type
                symbols = [
                    s.strip()
                    for s in self._inter_elements.split(",")
                    if s.strip()
                ]
                if inter_type in ("rocksalt", "zincblende",
                                    "cesiumchloride", "fluorite", "wurtzite"):
                    formula = "".join(symbols) if symbols else self._inter_elements
                    struct = inter_type
                    if struct in ("hcp", "wurtzite"):
                        kw = {"orthorhombic": True}
                    else:
                        kw = {"cubic": True}
                    pc = PristineCrystal.from_bravais(
                        symbol=formula,
                        crystalstructure=struct,
                        a=self._inter_a,
                        c=self._inter_c,
                        **kw,
                    )
                else:
                    pc = PristineCrystal.from_intermetallic(
                        strukturbericht=inter_type,
                        symbols=symbols,
                        a=self._inter_a,
                        c=self._inter_c,
                    )
            elif self._crystal_source == 2:  # Spacegroup
                symbols = [
                    s.strip()
                    for s in self._sg_elements.split(",")
                    if s.strip()
                ]
                raw = self._sg_basis.strip()
                basis = []
                for part in raw.replace("(", "").replace(")", "").split(","):
                    basis.append(float(part))
                basis = [
                    (basis[i], basis[i + 1], basis[i + 2])
                    for i in range(0, len(basis), 3)
                ]
                sg = self._sg_spacegroup.strip()
                try:
                    sg = int(sg)
                except ValueError:
                    pass
                cellpar = [
                    float(x.strip())
                    for x in self._sg_cellpar.split(",")
                ]
                pc = PristineCrystal.from_spacegroup(
                    symbols=symbols,
                    basis=basis,
                    spacegroup=sg,
                    cellpar=cellpar,
                )
            else:  # Custom (CIF / .crystal), source == 3
                if self._custom_file.lower().endswith(".crystal"):
                    self.finished_crystal.emit(
                        _load_crystal_file(self._custom_file),
                        self._phase_index,
                    )
                    return
                pc = PristineCrystal.from_cif(
                    cif_path=self._custom_file,
                )

            preview_atoms = pc.unit_cell * (2, 2, 2)
            # Store the true unit-cell vectors so the viewport wireframe
            # draws a single conventional cell, not the 2x2x2 preview box.
            preview_atoms.info["_unit_cell"] = pc.unit_cell.get_cell()[:]
            self.finished_crystal.emit(preview_atoms, self._phase_index)

        except Exception as exc:
            self.error_crystal.emit(str(exc))


# ---------------------------------------------------------------------------
# Full crystal builder (used by build worker)
# ---------------------------------------------------------------------------

def _generate_full_crystal(crystal_source: int, params: dict, coverage=None,
                           z_cap=None):
    """Return a full supercell ASE Atoms object for the build step.

    *crystal_source*: 0=bravais, 1=intermetallic, 2=spacegroup, 3=file.
    *params*: dict with keys matching CrystalGenerationWorker constructor.
    *coverage*: Cartesian extent the supercell must span (Å).  Defaults to
    ``box_end - box_start``.  When building for the assembly step, pass
    ``2 * max_neighbor_distance(…)`` to generate only what the largest
    grain needs rather than the whole box.
    *z_cap*: if set, trim the crystal Z extent to ±z_cap/2 (used for
    laminate where columnar cells span the full stacking direction).
    """
    from pristine_crystal import PristineCrystal

    box_start = params["box_start"]
    box_end = params["box_end"]

    if crystal_source == 3 and params["custom_file"].lower().endswith(".crystal"):
        atoms = _load_crystal_file(params["custom_file"])
        pc = PristineCrystal(atoms, box_start=box_start, box_end=box_end,
                             coverage=coverage)
        result = pc.run(margin=0, verbose=False)
        return result.atoms

    if crystal_source == 0:  # Bravais
        struct = params["single_structure"]
        if struct in ("hcp",):
            kw = {"orthorhombic": True}
        elif struct in ("sc", "fcc", "bcc", "diamond", "rocksalt", "zincblende"):
            kw = {"cubic": True}
        else:
            kw = {}
        pc = PristineCrystal.from_bravais(
            symbol=params["single_element"].strip(),
            crystalstructure=struct,
            a=params["single_a"],
            c=params["single_c"],
            box_start=box_start,
            box_end=box_end,
            coverage=coverage,
            **kw,
        )
    elif crystal_source == 1:  # Intermetallics
        inter_type = params["inter_type"]
        symbols = [s.strip() for s in params["inter_elements"].split(",") if s.strip()]
        if inter_type in ("rocksalt", "zincblende",
                            "cesiumchloride", "fluorite", "wurtzite"):
            formula = "".join(symbols) if symbols else params["inter_elements"]
            struct = inter_type
            if struct in ("hcp", "wurtzite"):
                kw = {"orthorhombic": True}
            else:
                kw = {"cubic": True}
            pc = PristineCrystal.from_bravais(
                symbol=formula, crystalstructure=struct,
                a=params["inter_a"], c=params["inter_c"],
                box_start=box_start, box_end=box_end,
                coverage=coverage, **kw,
            )
        else:
            pc = PristineCrystal.from_intermetallic(
                strukturbericht=inter_type, symbols=symbols,
                a=params["inter_a"], c=params["inter_c"],
                box_start=box_start, box_end=box_end,
                coverage=coverage,
            )
    elif crystal_source == 2:  # Spacegroup
        symbols = [
            s.strip()
            for s in params["sg_elements"].split(",") if s.strip()
        ]
        raw = params["sg_basis"].strip()
        flat = [float(x) for x in raw.replace("(", "").replace(")", "").split(",")]
        basis = [(flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat), 3)]
        sg = params["sg_spacegroup"].strip()
        try:
            sg = int(sg)
        except ValueError:
            pass
        cellpar = [float(x.strip()) for x in params["sg_cellpar"].split(",")]
        pc = PristineCrystal.from_spacegroup(
            symbols=symbols, basis=basis, spacegroup=sg, cellpar=cellpar,
            box_start=box_start, box_end=box_end,
            coverage=coverage,
        )
    else:  # CIF file (source == 3)
        pc = PristineCrystal.from_cif(
            cif_path=params["custom_file"],
            box_start=box_start,
            box_end=box_end,
            coverage=coverage,
        )

    result = pc.run(margin=2, verbose=False)
    atoms = result.atoms
    if z_cap is not None:
        pos = atoms.get_positions()
        half_z = z_cap / 2.0
        mask = (pos[:, 2] >= -half_z) & (pos[:, 2] <= half_z)
        from ase import Atoms
        atoms = Atoms(
            symbols=np.array(atoms.get_chemical_symbols())[mask],
            positions=pos[mask],
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
        )
    return atoms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_diagonals_from_poly_data(poly_data, sr):
    """Compute per-grain diagonals from primary Voronoi cell image.

    Uses the primary image (index 13, shift [0,0,0]) — the ground-truth
    periodic Voronoi cell from pyvoro.

    Returns
    -------
    individual_diags : (N,) ndarray
    max_diag : float
    """
    n_grains = sr.n_grains

    lookup: dict[tuple[int, int], np.ndarray] = {}
    for g_id, verts, img_idx in poly_data:
        if len(verts) > 0:
            lookup[(g_id, img_idx)] = verts

    individual_diags = np.zeros(n_grains)
    for g in range(n_grains):
        verts = lookup[(g, 13)]  # primary image
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        extent = vmax - vmin

        individual_diags[g] = float(np.linalg.norm(extent))

    max_diag = float(individual_diags.max()) if n_grains > 0 else 0.0
    return individual_diags, max_diag


# ---------------------------------------------------------------------------
# Build worker
# ---------------------------------------------------------------------------

class PolycrystalBuildWorker(QThread):
    """Runs full crystal generation + polycrystal assembly off the main thread.

    Emits ``finished(AssemblyResult)`` on success, ``error(str)`` on failure.
    """

    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        seed_result,
        orientation_result,
        crystal_source: int,
        crystal_params: dict,
        data_path: str,
        dump_path: str,
        hkl: tuple | None = None,
        distribution: str = "random",
        crystal_atoms: dict | None = None,
        grain_phases: np.ndarray | None = None,
        phase_configs: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._seed_result = seed_result
        self._orientation_result = orientation_result
        self._crystal_source = crystal_source
        self._crystal_params = crystal_params
        self._data_path = data_path
        self._dump_path = dump_path
        self._hkl = hkl
        self._distribution = distribution
        self._crystal_atoms = crystal_atoms
        self._grain_phases = grain_phases
        self._phase_configs = phase_configs

    def run(self) -> None:
        try:
            from pc_assembly import assemble_polycrystal

            # Compute safe coverage and per-grain diagonals
            sr = self._seed_result
            poly_data = sr.polyhedron_data
            is_laminate = (self._distribution == "laminate")
            is_columnar = self._distribution in ("laminate", "even")

            individual_diags = None
            max_grain_z = None
            box_start_arr = np.array(sr.box_start, dtype=float)
            box_end_arr = np.array(sr.box_end, dtype=float)
            box_size = box_end_arr - box_start_arr
            box_diag = float(np.linalg.norm(box_size))
            coverage_floor = box_diag * 1.5

            if poly_data:
                individual_diags, max_diag = _compute_diagonals_from_poly_data(
                    poly_data, sr,
                )
                # Max grain Z extent (for columnar pillar height)
                lookup: dict[tuple[int, int], np.ndarray] = {}
                for g_id, verts, img_idx in poly_data:
                    if len(verts) > 0:
                        lookup[(g_id, img_idx)] = verts
                z_extents = []
                for g in range(sr.n_grains):
                    verts = lookup[(g, 13)]
                    z_extents.append(
                        float(verts[:, 2].max() - verts[:, 2].min())
                    )
                max_grain_z = float(max(z_extents)) if z_extents else float(box_size[2])

            # Coverage from max grain diagonal (principled)
            if poly_data and individual_diags is not None:
                safe_coverage = 2.0 * float(max_diag)
                print(f"[Build] Max grain diagonal = {max_diag:.1f} A")
                print(f"[Build] Safe coverage = 2 x max diagonal = {safe_coverage:.1f} A")
            else:
                safe_coverage = box_diag * 1.5
                if poly_data:
                    safe_coverage = float(max(max_diag * 2.0, safe_coverage))
                print(f"[Build] No grain diagonals; "
                      f"safe coverage = {safe_coverage:.1f} A")

            # Step A: full crystal supercell(s)
            if (self._phase_configs and len(self._phase_configs) > 1
                    and self._grain_phases is not None):
                # Multi-phase: generate per distinct phase, pass dict
                import json
                phase_full_crystals: dict[int, object] = {}
                generated_cache: dict[int, object] = {}
                for phase_idx, cfg in sorted(self._phase_configs.items()):
                    if phase_idx in self._crystal_atoms:
                        # Use preview crystal if available (same atoms, just
                        # re-supercell it)
                        pass
                    h = hash(json.dumps({
                        "src": cfg.get("crystal_source", 0),
                        "params": {k: v for k, v in
                                   cfg.get("crystal_params", {}).items()
                                   if k not in ("coverage", "box_start", "box_end")},
                    }, sort_keys=True))
                    if h in generated_cache:
                        phase_full_crystals[phase_idx] = generated_cache[h]
                    else:
                        fc = _generate_full_crystal(
                            cfg["crystal_source"], cfg["crystal_params"],
                            coverage=safe_coverage,
                        )
                        phase_full_crystals[phase_idx] = fc
                        generated_cache[h] = fc
                crystal_input = phase_full_crystals
            else:
                crystal_input = _generate_full_crystal(
                    self._crystal_source, self._crystal_params,
                    coverage=safe_coverage,
                )

            # Step B: polycrystal assembly
            result = assemble_polycrystal(
                seeds=sr.seeds,
                crystal_atoms=crystal_input,
                orientations=self._orientation_result,
                box_start=sr.box_start,
                box_end=sr.box_end,
                target_radii=sr.target_radii,
                grain_diagonals=individual_diags,
                grain_phases=self._grain_phases,
                is_laminate=is_laminate,
                hkl=self._hkl,
                poly_data=poly_data,
                is_columnar=is_columnar,
                max_grain_z=max_grain_z,
                data_path=self._data_path,
                dump_path=self._dump_path,
                verbose=True,
            )

            # Step C: emit
            self.finished.emit(result)

        except Exception as exc:
            self.error.emit(str(exc))
