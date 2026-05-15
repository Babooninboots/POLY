# SPEC.md: POLY — Interactive LAMMPS Polycrystalline Sample Generator

Author: Yang Zhang (张杨)

Affiliation: Stony Brook University (SUNY at Stony Brook)

## 1. Overview

POLY generates atomic structures of polycrystalline materials for LAMMPS
simulations. It constructs 3D granular structures from user-defined parameters
— box dimensions, grain size distributions, crystal structures, crystallographic
orientations — and outputs atomistic models in standard LAMMPS formats.

A PySide6 GUI provides interactive 3D preview, grain editing, and one-click
build. All backend modules can also be used headless via Python scripts.

---

## 2. Global Input Parameters

| Parameter                   | Options                                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| **Box Size**                | Start `(x0, y0, z0)` and end `(x1, y1, z1)`, auto-tuned for columnar/Z-aligned modes          |
| **Boundary Conditions**     | Periodic (PBC) along all three axes                                                           |
| **Grain Quantity**          | Fixed count OR average grain diameter                                                         |
| **Grain Size Distribution** | `random`, `normal`, `customized`, `laminate`, `even`                                          |
| **Crystal Structure**       | Bravais, Intermetallics, Spacegroup, or Custom (`.cif` / `.crystal`)                          |
| **Orientation**             | `random`, `z_alignment`, `low_angle`, `high_angle`, `custom_misorientation`, `custom_profile` |
| **Output**                  | LAMMPS data file (`.data`) + dump file (`.dump`)                                              |

---

## 3. Core Modules

### Module 1: Grain Seed Generation & Voronoi Tessellation  (`grain_seeds.py`)

**Objective:** Determine the spatial distribution and Voronoi cell boundaries of
individual grains.

1. **Seed Initialization**
   
   - *Random* / *Normal*: Place seeds uniformly, then enforce minimum separation.
   - *Laminate*: Seeds on a mid-plane with in-plane random or normal distribution.
   - *Evenly Spaced (1D)*: Seeds at equal intervals along one axis.
   - *Customized*: Read seed positions from a `.seed` file (3-col `x y z` or
     4-col `x y z radius`).

2. **Voronoi Tessellation**
   Uses `pyvoro2` with 27 PBC images (3×3×3 shift grid). When `target_radii`
   are available, runs Laguerre-Voronoi (power-distance) tessellation; otherwise
   standard Voronoi.

3. **Size Distribution Optimization (FBSP)**
   For `normal` distribution (and `laminate` + normal in-plane): Force-Biased
   Sphere Packing — a Monte Carlo relaxation that draws target diameters from a
   log-normal distribution and applies overlap-based repulsive forces with
   learning-rate decay. Runs up to 2000 iterations.

4. **Output**
   Returns a `SeedResult` dataclass: `seeds`, `diameters`, `target_radii` (for
   Laguerre-Voronoi), `neighbors` (Voronoi adjacency), `polyhedron_data` (27
   PBC images per grain), box bounds.

**Entry point:** `generate_grains(box_start, box_end, ...) -> SeedResult`

---

### Module 2: Pristine Crystal Generation  (`pristine_crystal.py`)

**Objective:** Build a monolithic supercell of the base crystal lattice,
centered at the origin.

1. **Crystal Sources**
   
   - **Bravais** (`sc`, `fcc`, `bcc`, `hcp`, `diamond`, `tetragonal`, `bct`)
     via `ase.build.bulk`.
   - **Intermetallics** (`L1_2`, `B2`, `D0_3`, `L2_1`, `D0_19`, `A15`, `C15`,
     `L1_0`, `rocksalt`, `zincblende`, etc.) via `ase.build.bulk`.
   - **Spacegroup** via `ase.spacegroup.crystal` with Wyckoff positions,
     spacegroup number, and cell parameters.
   - **Custom** — loads `.cif`, `.crystal`, or any ASE-readable file.

2. **Supercell Construction**
   Replicates the unit cell to cover `coverage` (default: 2× max box diagonal
   for assembly pre-crop headroom), plus a `margin` of extra replicas.

3. **Normalization**
   Shifts the supercell so its centre of mass is at the origin.

4. **Output**
   Returns a `CrystalResult` dataclass: `atoms` (ASE Atoms), `n_atoms`,
   `repeats` (multiplicity per axis), `unit_cell`.

**Entry points:**

- `generate_pristine_bravais(element, structure, a, c, ...) -> CrystalResult`
- `generate_pristine_intermetallic(prototype, symbols, a, ...) -> CrystalResult`
- `generate_pristine_cif(path, ...) -> CrystalResult`
- `PristineCrystal` class for programmatic use

---

### Module 3: Orientation Assignment  (`orientation.py`)

**Objective:** Assign ZXZ Euler angles and rotation matrices to each grain.

1. **`random`** — Uniform random Euler angles for all grains.

2. **`z_alignment`** — Align crystal direction `[hkl]` with the simulation
   Z-axis. In-plane rotation modes:
   
   - `random` — uniform random φ₂
   - `low_angle` / `high_angle` — BFS graph traversal with pairwise constraint
     checking
   - `symmetric_csl` — CSL misorientation with enumerated Σ boundaries via the
     Diophantine equation `m² + N_hkl·n² = Σ`

3. **`low_angle`** / **`high_angle`** — BFS graph traversal from a random seed
   grain. Neighbours are assigned rotations with pairwise misorientation below
   10° (low) or above 20° (high), computed via
   `min(tr(R_i·R_j^T))` angle.

4. **`custom_misorientation`** — Monte Carlo simulated annealing that matches
   the current misorientation distribution to a user-provided target
   distribution, using the KS-statistic as the energy function and the
   Metropolis criterion.

5. **`custom_profile`** — Load per-grain Euler angles from a `.euler` file
   (4 columns: `grain_id φ₁ Φ φ₂`).

**Output**
Returns an `OrientationResult` dataclass: `euler_angles` (N×3, degrees),
`rotation_matrices` (N×3×3), `rotations` (scipy Rotation), `misorientation_angles`,
`mc_energy_history`, `mc_acceptance_rate`.

**Entry point:** `generate_orientations(mode, n_grains, neighbors, ...) -> OrientationResult`

---

### Module 4: Interactive GUI  (`gui_main.py`, `gui_views.py`)

**Objective:** Provide interactive 3D preview, grain editing, and one-click build.

1. **Layout**
   
   - **Left dock** — "Global Settings" card with all simulation inputs
     (Box Size, Quantity, Distribution, Crystal, Orientation, Grain Editor,
     Terminal Messages), plus Pause/Stop/Progress controls.
   - **Left viewport** — 3D Voronoi polyhedra with Euler-angle RGB colouring,
     GPU clipping plane (slice slider), and point-picking for grain selection.
   - **Top-right viewport** — Pristine crystal unit cell, CPK-coloured atoms.
   - **Bottom-right** — Grain-size histogram and misorientation-angle histogram
     (pyqtgraph).
   - **Bottom bar** — Output folder, state name, action buttons.

2. **Action Buttons** (left to right)
   
   - **Save Pristine Crystal** — writes `{Name}.crystal`
   - **Save Seed State** — writes `{Name}.seed` (3 or 4 columns) and `{Name}.euler`
   - **Reroll** — clears caches, forces full regeneration
   - **Generate Initial Seeds** — runs seed + Voronoi + crystal generation
   - **Proceed with Build** — runs full polycrystal assembly

3. **Interactivity**
   
   - Click a grain in the Voronoi viewport to select it; editor populates with
     its position and Euler angles.
   - **Apply Edit** recomputes Voronoi cells and misorientations, then re-renders.
   - **Pause** during MC optimization renders the current intermediate state.
   - **Stop** requests early termination of the MC optimizer.
   - **Slice Depth** slider provides GPU-accelerated clipping through the
     polycrystal.

4. **Background Workers** (`workers.py`)
   
   - `SeedGenerationWorker` — QThread for seed + orientation generation.
   - `CrystalGenerationWorker` — QThread for supercell construction.
   - `PolycrystalBuildWorker` — QThread for assembly + file output.
   - Workers emit `progress`, `finished`, and `error` signals.

---

### Module 5: Polycrystalline Assembly  (`pc_assembly.py`)

**Objective:** Rotate, crop, translate, and assemble individual grains into the
final polycrystal.

1. **Seed Image Grid**
   Generate 27 PBC images (3×3×3 shift grid) for each grain seed. Build a
   global `cKDTree` over all seed images. Select the **best periodic image**
   per grain — the one closest to the simulation box centre — to minimise
   Voronoi cell splitting after modulo wrap.

2. **Assembly Pipelines**
   
   **Standard Pipeline** (random / normal / customized):
   
   - Per-grain spherical pre-crop at origin (radius = `grain_diag × 0.55`, or
     `0.75 × box_diag` fallback).
   - Rotate the pre-cropped crystal by the grain's rotation matrix.
   - Translate to the best seed position.
   - KD-tree query against the global seed-image tree (k=1, or k=min(30,N) for
     power-distance when `target_radii` are available).
   - Keep atoms whose nearest seed is the current grain's best image.
   
   **Columnar Pipeline** (laminate / evenly spaced):
   
   - Compute `R_base` to align crystal `[hkl]` with the stack axis.
   - Pre-rotate the entire crystal and trim to a **master pillar** (diameter =
     `2 × max(grain_diag)`, height = `1.5 × max_grain_z`).
   - Per-grain: crop to a cylinder of radius = `grain_diag[g]`, rotate in-plane
     via `Rz = R_total · R_base⁻¹`, translate to best seed, KD-tree query.
   - Stack-axis-aware pillar dimensions for non-Z stack directions.

3. **Modulo Wrap**
   All final positions are modulo-wrapped into the primary simulation box:
   `positions = (positions - box_start) % box_size + box_start`.

4. **Output**
   Returns an `AssemblyResult` dataclass: `positions`, `types` (1-based LAMMPS
   atom types), `grain_ids` (0-based), `euler_per_atom`, `symbols`,
   `type_to_symbol`, `type_masses`, `keep_counts` (atoms per grain).

**Entry point:** `assemble_polycrystal(seeds, crystal_atoms, orientations, ...) -> AssemblyResult`

---

### Module 6: Data Output  (`pc_assembly.py`)

**Objective:** Export the assembled structure in LAMMPS-readable formats.

1. **LAMMPS Data File** (`.data`)
   Header with atom count, type count, box bounds; Masses section; Atoms
   section (`id type x y z`).

2. **LAMMPS Dump File** (`.dump`)
   Custom dump format for OVITO/ParaView visualisation:
   `ITEM: ATOMS id type x y z grain_id euler_angle_1 euler_angle_2 euler_angle_3`

3. **Crystal File** (`.crystal`)
   XYZ-format crystal save with cell vectors in header comments, readable by
   the Custom crystal source.

4. **Seed File** (`.seed`)
   3-column (`x y z`) or 4-column (`x y z radius`) format. The 4-column variant
   preserves Laguerre-Voronoi radii for exact cell-shape reproduction on reload.

5. **Euler File** (`.euler`)
   4-column: `grain_id φ₁ Φ φ₂` (ZXZ convention, degrees).

---

## 4. File Inventory

| File                  | Role                                                                               |
| --------------------- | ---------------------------------------------------------------------------------- |
| `grain_seeds.py`      | Module 1 — seed generation, FBSP, Voronoi tessellation                             |
| `pristine_crystal.py` | Module 2 — supercell construction from Bravais / intermetallic / spacegroup / file |
| `orientation.py`      | Module 3 — orientation assignment, BFS, MC matching, CSL enumeration               |
| `gui_main.py`         | Module 4 — PySide6 application shell, signal wiring, build orchestration           |
| `gui_views.py`        | Module 4 — view widgets: dock, central viewports, file picker, terminal            |
| `pc_assembly.py`      | Modules 5 & 6 — assembly pipelines, KD-tree trimming, modulo wrap, LAMMPS I/O      |
| `workers.py`          | Background QThread workers bridging the GUI to backend modules                     |
| `user_manual.md`      | Comprehensive user guide                                                           |
| `gui_main.spec`       | PyInstaller spec for standalone Windows executable                                 |
| `examples/`           | Runnable scripts: `bicrystal.py`, `NaCl.py`                                        |
| `tests/`              | Test suites for all modules                                                        |

---

## 5. Dependencies

| Library                 | Role                                                                       |
| ----------------------- | -------------------------------------------------------------------------- |
| **NumPy**               | Array operations, linear algebra, RNG                                      |
| **SciPy**               | `cKDTree` spatial queries, `Rotation`, Delaunay triangulation, `ks_2samp`  |
| **PySide6**             | Qt6 GUI framework                                                          |
| **PyVista / pyvistaqt** | 3D visualisation, `QtInteractor` for embedding VTK in Qt                   |
| **pyvoro2**             | Periodic Voronoi tessellation (standard + Laguerre power-distance)         |
| **pyqtgraph**           | Histogram charts                                                           |
| **vtk**                 | GPU clipping planes, mesh manipulation                                     |
| **ASE**                 | Crystal structure database, spacegroup generation, atomic masses, file I/O |
