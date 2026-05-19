# POLY — Interactive LAMMPS Polycrystalline Sample Generator

Author: Yang Zhang (张杨)

Affiliation: Stony Brook University (SUNY at Stony Brook)

## Overview

POLY is a PySide6 GUI application that generates polycrystalline atomic structures
for LAMMPS molecular-dynamics simulations. It produces fully periodic 3D
polycrystals with user-defined grain-size distributions, crystal structures, and
crystallographic orientations.

**Backend pipeline** (runs in background threads):

1. **Seed generation** — grain positions via random, normal, bimodal (Gaussian
   mixture), customized, laminate, or evenly-spaced distributions, with
   Force-Biased Sphere Packing and Voronoi tessellation via pyvoro2.
2. **Pristine crystal** — supercell generation from Bravais lattices,
   intermetallics, spacegroup data, CIF files, or POLY `.crystal` files.
3. **Orientation assignment** — Euler angles (ZXZ convention) via random,
   low/high angle, Z-axis alignment, CSL, custom misorientation, or custom
   profile modes.
4. **Polycrystal assembly** — best periodic image selection, per-grain
   rotation/translation, KD-tree Voronoi trimming, and modulo wrap.
5. **LAMMPS output** — `.data` (atomic data file) and `.dump` files.

**Frontend** (GUI):

- Voronoi viewport (PyVista 3D) with interactive slicing and point-picking.
- Pristine crystal viewport with CPK-coloured atoms.
- Grain-size and misorientation histograms.
- Grain editor for manual seed/orientation tweaks.
- Background workers with progress bar, pause, and stop.

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Launch](#launch)
  - [Window Layout](#window-layout)
  - [Action Buttons (bottom bar, left to right)](#action-buttons-bottom-bar-left-to-right)
- [Step-by-Step Workflow](#step-by-step-workflow)
  - [1. Set the Simulation Box](#1-set-the-simulation-box)
  - [2. Choose Grain Quantity](#2-choose-grain-quantity)
  - [3. Select Grain Size Distribution](#3-select-grain-size-distribution)
  - [4. Configure Crystal Structure](#4-configure-crystal-structure)
  - [5. Choose Orientation Mode](#5-choose-orientation-mode)
  - [6. Generate Seeds](#6-generate-seeds)
  - [7. Review and Edit](#7-review-and-edit)
  - [8. Save Intermediate State (Optional)](#8-save-intermediate-state-optional)
  - [9. Configure Output](#9-configure-output)
  - [10. Build the Polycrystal](#10-build-the-polycrystal)
- [Columnar Structures (Laminate / Evenly Spaced)](#columnar-structures-laminate-evenly-spaced)
- [File Formats](#file-formats)
  - [`.seed` — Seed positions](#seed-seed-positions)
  - [`.euler` — Euler angles](#euler-euler-angles)
  - [`.crystal` — Pristine crystal](#crystal-pristine-crystal)
  - [`.data` — LAMMPS atomic data](#data-lammps-atomic-data)
  - [`.dump` — LAMMPS dump](#dump-lammps-dump)
- [Keyboard & Mouse](#keyboard-mouse)
- [Tips](#tips)
- [Algorithms](#algorithms)
  - [Seed Distribution Optimization](#seed-distribution-optimization)
  - [Misorientation Optimization](#misorientation-optimization)
  - [Grain Assembly (Polycrystal Build)](#grain-assembly-polycrystal-build)
- [Database](#database)
  - [Materials Science Context](#materials-science-context)
  - [External Python Libraries](#external-python-libraries)
- [Standalone Python Scripts](#standalone-python-scripts)
  - [Quick-Start: Minimal 5-Module Pipeline](#quick-start-minimal-5-module-pipeline)
  - [Module 1: `grain_seeds` — Seed Generation & Voronoi Tessellation](#module-1-grainseeds-seed-generation-voronoi-tessellation)
  - [Module 2: `pristine_crystal` — Crystal Supercell Generation](#module-2-pristinecrystal-crystal-supercell-generation)
  - [Module 3: `orientation` — Orientation Assignment](#module-3-orientation-orientation-assignment)
  - [Module 5: `pc_assembly` — Polycrystal Assembly](#module-5-pcassembly-polycrystal-assembly)
  - [Z-Axis Alignment + Bicrystal Example](#z-axis-alignment-bicrystal-example)
  - [Programmatic Build via `workers.py`](#programmatic-build-via-workerspy)
  - [Validation Pattern](#validation-pattern)


## Getting Started

### Launch

```bash
python gui_main.py
```

A splash screen appears while VTK and the UI load. The main window measures
1280×800 and can be freely resized.

### Window Layout

| Region                    | Contents                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Left dock**             | "Global Settings" — all simulation inputs. Double-click the title bar to detach/reattach or float the panel. |
| **Top-right viewport**    | Pristine crystal unit cell (CPK-coloured atoms + cell wireframe).                                            |
| **Bottom-right viewport** | Voronoi polyhedra with Euler-angle colouring, interactive slicing, and point-picking.                        |
| **Right charts**          | Grain-size histogram (top) and misorientation-angle histogram (bottom).                                      |
| **Bottom bar**            | Output folder, state name, and action buttons.                                                               |

### Action Buttons (bottom bar, left to right)

| Button                     | Colour | When Active             | Action                                                     |
| -------------------------- | ------ | ----------------------- | ---------------------------------------------------------- |
| **Save Pristine Crystal**  | Green  | After crystal generated | Writes `{Name}.crystal`                                    |
| **Save Seed State**        | Green  | After seeds generated   | Writes `{Name}.seed` and `{Name}.euler`                    |
| **Reroll**                 | Orange | Always                  | Clears all caches and re-runs seed + crystal generation    |
| **Generate Initial Seeds** | Blue   | Always                  | Runs seed generation + Voronoi tessellation + crystal      |
| **Proceed with Build**     | Red    | Always                  | Runs full polycrystal assembly, writes `.data` and `.dump` |

**MC controls** (bottom of left dock):

| Control          | Action                                                                                         |
| ---------------- | ---------------------------------------------------------------------------------------------- |
| **Pause**        | Suspends/resumes the Monte Carlo seed-placement optimizer; intermediate state is rendered live |
| **Stop**         | Requests early termination of the optimizer                                                    |
| **Progress bar** | Shows optimization progress (0–100%)                                                           |

---

## Step-by-Step Workflow

### 1. Set the Simulation Box

In **Box Size**, set `Min (start)` and `Max (end)` for X, Y, Z in Ångströms.
Default: `(0,0,0)` → `(100,100,100)`.

> For laminate and evenly-spaced distributions with Z-axis alignment,
> the Z dimension is auto-tuned to an integer multiple of the crystal repeat
> distance along the aligned direction.

### 2. Choose Grain Quantity

Toggle between **Number of Grains** (direct count, 1–10⁸) and
**Average Grain Diameter** (Å). Changing one auto-updates the other based on
box volume and a spherical-equivalent-diameter model.

### 3. Select Grain Size Distribution

| Distribution            | Description                                                               | Extra Fields                                 |
| ----------------------- | ------------------------------------------------------------------------- | -------------------------------------------- |
| **Random**              | Uniform random seed positions                                             | —                                            |
| **Normal Distribution** | Force-biased sphere packing targeting a log-normal size distribution      | StdDev (Å)                                   |
| **Bimodal**             | Gaussian mixture model: two normal distributions blended by number fraction | Fraction in mode 1, two means, two stds; count auto-calculated |
| **Customized**          | Load pre-defined seed positions from a `.seed` file                       | File picker                                  |
| **Laminate**            | 2D columnar grains on a mid-plane with in-plane Voronoi                   | In-plane distribution (`random` or `normal`) |
| **Evenly Spaced (1D)**  | Seeds evenly spaced along one axis, spanning the full perpendicular plane | Axis selection (`x`, `y`, or `z`)            |

> Choosing Laminate or Evenly Spaced automatically switches the orientation
> mode to Z-axis alignment.

### 4. Configure Crystal Structure

| Source             | Fields                                                                                                                 |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Bravais**        | Structure (`sc`, `fcc`, `bcc`, `hcp`, `diamond`, `tetragonal`, `bct`), Element, a (Å), c (Å)                           |
| **Intermetallics** | Prototype (`L1₂`, `B2`, `D0₃`, `L2₁`, `D0₁₉`, `A15`, `C15`, `L1₀`, rocksalt, zincblende, etc.), Elements, a (Å), c (Å) |
| **Spacegroup**     | Elements, Basis (fractional coordinates), Spacegroup number, Cell parameters (a,b,c,α,β,γ)                             |
| **Custom (File)**  | `.cif`, `.crystal`, or ASE-readable file                                                                               |

#### Multi-Phase Support

Click the green **ADD CRYSTAL** button to create additional crystal phases (Phase 1,
Phase 2, …). Use the **Phase** spinbox to switch between phases. Each phase stores
its own crystal source, structure, element, and lattice parameters independently.

For phases > 0, assign which grains use this crystal:
- **Fraction**: decimal fraction of total grains (e.g. `0.3` = 30%)
- **Grain List**: comma-separated grain IDs with formula support:
  - `2n` — all even-index grains (0, 2, 4, …)
  - `2n+1` — all odd-index grains (1, 3, 5, …)
  - `d < 20`, `d > 50`, `d <= X`, `d >= X` — filter by grain diameter
  - `1,3,5-8` — explicit IDs and ranges
  - Combine: `2n, 5, d<30`

Phase 0 (the matrix) automatically gets all remaining unassigned grains. Atom type
IDs are continuous across phases (Phase 0 types start at 1, Phase 1 types continue
from Phase 0's maximum type ID, etc.).

Saving pristine crystals in multi-phase mode writes `{name}.0.crystal`,
`{name}.1.crystal`, etc.

The `a` → `c` ratio is auto-filled for hexagonal structures (c/a ≈ 1.633).

### 5. Choose Orientation Mode

| Mode                      | Description                                                  | Extra Fields                                                                                            |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| **Random**                | Uniform random Euler angles                                  | —                                                                                                       |
| **Z-axis alignment**      | Align a crystal direction `[hkl]` with the simulation Z-axis | (hkl) indices, in-plane mode (`random`, `low_angle`, `high_angle`, `symmetric_csl`), Σ value, CSL angle |
| **Low angle**             | Small random misorientation from a base orientation          | —                                                                                                       |
| **High angle**            | Large random misorientation                                  | —                                                                                                       |
| **Custom Misorientation** | Load misorientation angles from file                         | File picker                                                                                             |
| **Custom Profile**        | Load per-grain Euler angles from a `.euler` file             | File picker                                                                                             |

**Symmetric CSL mode**: when `In-plane` is set to `symmetric_csl`, the
application computes all CSL misorientation angles for the given Σ and
`[hkl]` pair. Select one from the dropdown to split the misorientation
symmetrically (e.g., grain 0 gets +θ/2, grain 1 gets −θ/2 around Z).
Box dimensions are auto-tuned to exact integer multiples of the projected
crystal repeat distance in each perpendicular direction.

### 6. Generate Seeds

Click **Generate Initial Seeds**. A background thread:

- Places seeds via the chosen distribution.
- Runs Voronoi tessellation (Monte Carlo optimization for normal/laminar-normal
  distributions).
- Assigns orientations.
- Generates the pristine crystal supercell.

The Voronoi 3D viewport shows polyhedra coloured by Euler angles, and the
charts update with grain-size and misorientation distributions.

### 7. Review and Edit

- **Click a grain** in the Voronoi viewport to select it (highlight + wireframe
  cube). The **Selected Grain Editor** in the left dock populates with its
  position, Euler angles, and diameter.
- **Modify** values and click **Apply Edit** to recompute the Voronoi
  tessellation and misorientations in-place without regenerating seeds.
- **Slice the view** by dragging the Slice Depth slider or cycling the View
  button (Isometric → X → Y → Z).
- **Adjust perspective** with the Perspective slider.

### 8. Save Intermediate State (Optional)

- **Save Seed State** writes `{Name}.seed` (N×3 seed positions, or N×4 with
  Laguerre-Voronoi radii if FBSP was used) and `{Name}.euler` (N×4: grain_id φ₁ Φ φ₂).
- **Save Pristine Crystal** writes `{Name}.crystal` (XYZ format with cell
  vectors in the header).

### 9. Configure Output

Set the **Output Folder** (Browse…) and **State Name**. All output files are
written as `{Folder}/{Name}.ext`.

### 10. Build the Polycrystal

Click **Proceed with Build**. A background thread:

1. Generates the full pristine supercell at the required coverage.
2. Runs the polycrystal assembly pipeline (best periodic image, per-grain
   KD-tree trimming, modulo wrap).
3. Writes `{Name}.data` (LAMMPS atomic data) and `{Name}.dump`
   (per-atom grain_id + Euler angles).

The status bar shows the final atom and grain counts.

---

## Columnar Structures (Laminate / Evenly Spaced)

For laminate (2D columnar) and evenly-spaced (1D) distributions, the assembly
pipeline uses a specialized columnar algorithm:

1. The pristine crystal is generated as a cube of side `2 × max_grain_diag`.
2. The crystal is pre-rotated to align `[hkl]` with the box Z-axis (computed
   once, shared by all grains).
3. The aligned crystal is trimmed to a cylindrical pillar (diameter =
   `2 × max_grain_diag`, height = `1.5 × max_grain_z`).
4. Per grain: the pillar is cropped to the grain's radius, rotated in-plane
   (Rz = R_total × R_base⁻¹), translated to the seed position (with a 0.1 Å
   margin shift), and KD-tree-trimmed.
5. All atoms are modulo-wrapped into the primary simulation box.

---

## File Formats

### `.seed` — Seed positions

Three-column format (positions only):

```
# POLY seed state: {name}  |  {N} grains  |  distribution={dist}
# box_start: xlo ylo zlo
# box_end: xhi yhi zhi
x1 y1 z1
x2 y2 z2
...
```

Four-column format (positions + Laguerre-Voronoi radii, saved automatically when FBSP
optimization was used):

```
# POLY seed state: {name}  |  {N} grains  |  distribution={dist}
# columns: x y z radius
# box_start: xlo ylo zlo
# box_end: xhi yhi zhi
x1 y1 z1 r1
x2 y2 z2 r2
...
```

When a 4-column file is loaded with the **Customized** distribution, the radius column
is passed to the Laguerre-Voronoi power-distance tessellation, preserving the exact
cell shapes from the original FBSP run.

### `.euler` — Euler angles

```
# POLY orientation state: {name}  |  mode={mode}
# columns: grain_id phi1 Phi phi2 (zxz, degrees)
0 φ₁ Φ φ₂
1 φ₁ Φ φ₂
...
```

### `.crystal` — Pristine crystal

```
# POLY crystal: {name}  |  {N} atoms  |  formula={formula}
# crystal_system: {cubic|hexagonal|tetragonal|orthorhombic|triclinic}
# cell_1: ax ay az
# cell_2: bx by bz
# cell_3: cx cy cz
# element  x  y  z
Mg  x1 y1 z1
...
```

The `# crystal_system:` header specifies the crystal symmetry system for
correct misorientation calculation.  When saved from the GUI it is auto-detected
from the structure type or spacegroup.  If absent (legacy files), defaults to
`cubic`.

### `.data` — LAMMPS atomic data

Standard LAMMPS format with masses block. Atom types are 1-based, ordered by
element.

### `.dump` — LAMMPS dump

Per-atom fields: `id_POLY type x y z grain_id euler_angle_1 euler_angle_2 euler_angle_3`.
Grain IDs are 1-based.
Grain IDs are 1-based.

---

## Keyboard & Mouse

| Action                                  | Result                                           |
| --------------------------------------- | ------------------------------------------------ |
| **Left-click grain (Voronoi viewport)** | Select grain, populate editor                    |
| **Middle-drag (viewport)**              | Pan                                              |
| **Right-drag (viewport)**               | Rotate                                           |
| **Scroll (viewport)**                   | Zoom                                             |
| **Slice Depth slider**                  | GPU clipping plane through Voronoi polyhedra     |
| **View button**                         | Cycle Isometric → X-normal → Y-normal → Z-normal |

---

## Tips

- For **laminate** and **evenly spaced** distributions, always use
  **Z-axis alignment** orientation. The box Z dimension will be auto-tuned
  to an integer multiple of the `[hkl]` interplanar spacing.
- The **Reroll** button clears all caches and forces a full regeneration.
  Use it to try different random seeds (the random state advances on each
  generation).
- The **Pause** button during seed optimization renders the current Monte
  Carlo state so you can inspect convergence.
- If the build produces empty grains, try increasing the box dimensions
  or decreasing the grain count. Empty grains indicate Voronoi cells smaller
  than the lattice spacing.
- For reproducible state, save both `.seed` and `.euler` files, then reload
  with Customized + Custom Profile modes.

---

## Algorithms

### Seed Distribution Optimization

POLY supports five grain-seed placement strategies, the most sophisticated of
which is the **Force-Biased Sphere Packing** (FBSP) used for Normal
Distribution and Laminate/Normal in-plane distributions.

#### Random / Customized / Evenly Spaced

Seeds are placed directly without optimization:

- **Random**: uniform sampling within the box.
- **Customized**: loaded from a `.seed` file.
- **Evenly Spaced (1D)**: seeds placed at equal intervals along one axis,
  centered in the perpendicular plane.

#### Force-Biased Sphere Packing (FBSP)

Used when a target grain-size distribution is specified (Normal Distribution,
Bimodal, or Laminate with Normal in-plane distribution).

1. **Target sampling**: each grain is assigned a target diameter.  For Normal
   distribution: drawn from N(target_mean, target_std).  For Bimodal
   distribution: drawn from a Gaussian mixture φ·N(μ₁,σ₁) + (1−φ)·N(μ₂,σ₂),
   where φ is the user-specified number fraction in mode 1.  Clipped to
   ≥ 0.1 × mean (or 0.1 × min(μ₁,μ₂) for bimodal).
2. **Packing-fraction scaling**: diameters are scaled so the total sphere
   volume (3D, packing fraction ≈ 0.58) or area (2D, packing fraction ≈ 0.78)
   fills the domain, converting to equivalent Laguerre-Voronoi radii.
3. **Iterative relaxation** (up to 2000 steps):
   - A KD-tree (with PBC wrapping on active axes) finds all overlapping
     sphere pairs within 2 × max_radius.
   - For each overlapping pair (i, j), a repulsive force proportional to
     overlap × learning_rate is applied along the separation direction,
     weighted by r_j/(r_i+r_j) so smaller spheres are displaced more.
   - Learning rate decays as lr ← lr × 0.995 per step.
   - Converges when max_overlap < 0.01 Å or no overlaps remain.
4. **Output**: relaxed seed positions + target radii.  For bimodal, the final
   grain diameters are taken directly from the target radii (2 × r) to preserve
   the Gaussian-mixture signal; for normal, diameters are recomputed from
   Laguerre-Voronoi cell volumes.

The algorithm supports pause/resume for live inspection of convergence, and
early stop for user cancellation.  A yellow target-PDF curve is overlaid on
the grain-size histogram (single Gaussian for Normal, Gaussian mixture for
Bimodal).

#### Voronoi Tessellation

After seed placement, `pyvoro2` computes the 27-image periodic Voronoi
tessellation (3×3×3 replicas of the primary box). For power-distance
(Laguerre-Voronoi) tessellation, the target radii from FBSP are passed as
per-seed weights, ensuring each Voronoi cell closely matches the target grain
volume.

### Misorientation Optimization

#### BFS Graph Traversal (Low Angle / High Angle)

Used for both full-3D low/high-angle modes and the in-plane twist component
of Z-axis alignment.

1. A random starting grain is assigned a uniformly random rotation.
2. Breadth-first search over the Voronoi adjacency graph:
   - For each child grain, up to 200 candidate relative rotations are
     generated (random axis, angle drawn from the target range).
   - Each candidate is checked against **all** already-assigned neighbors.
   - For low-angle: all pairwise misorientations must be < 10°.
   - For high-angle: all pairwise misorientations must be > 20°.
   - The first valid candidate is accepted; if none pass, the least-violating
     candidate is used.
3. Misorientation is computed as the true 3D disorientation angle:
   `|R1 × R2⁻¹|` (the rotation angle of the misorientation quaternion).

#### Symmetric CSL (Coincidence Site Lattice)

For Z-axis alignment with `symmetric_csl` in-plane mode:

1. The function `get_cubic_csl_angles(hkl, Σ)` enumerates all (m, n) integer
   pairs where m² + (h²+k²+l²)n² = Σ (after removing factors of 2). The CSL
   misorientation angle is θ = 2 × arctan((n/m) × √N).
2. The selected angle is split symmetrically: grain 0 gets +θ/2, grain 1 gets
   −θ/2 around Z (for bicrystals), or alternating signs for multi-grain setups.
3. Box dimensions in the perpendicular directions are auto-tuned to integer
   multiples of the projected crystal repeat distance, ensuring the CSL boundary
   is fully periodic.

#### Monte Carlo Misorientation Matching (Custom Misorientation)

Used to match a user-supplied target misorientation-angle distribution:

1. **Initial state**: random rotations for all grains.
2. **Energy function**: the Kolmogorov-Smirnov (KS) statistic between the
   current pairwise misorientation-angle distribution and the target
   distribution. Falls back to difference-of-means for small sample sizes.
3. **MC loop** (up to 2000 steps, simulated annealing):
   - One randomly chosen grain is perturbed by a small random rotation
     (σ = 5° angular perturbation).
   - The KS energy of the new configuration is computed.
   - Accept/reject via Metropolis criterion with temperature T decaying as
     T ← max(0.001, T × 0.995).
   - Early stop when KS statistic < 0.05.
4. **Output**: best rotations found, energy history, and acceptance rate.

### Grain Assembly (Polycrystal Build)

The assembly pipeline converts per-grain orientations and a pristine crystal
supercell into the final atomic structure.

#### Standard Pipeline

1. **27-image PBC seed tree**: each of the N grains is replicated across a
   3×3×3 grid of periodic shifts (±box in each direction), yielding a flat
   array of 27N seed positions. A cKDTree is built for nearest-seed queries.
2. **Best periodic image**: for each grain, the one periodic image closest to
   the box center is selected, minimizing Voronoi-cell splitting after the
   final modulo wrap.
3. **Per-grain sequential processing**:
   - **Spherical pre-crop** at origin: pristine atoms within
     radius = 0.55 × grain_diagonal of the grain seed are kept.
     (Rotation preserves distance from origin, so cropping at origin is
     equivalent to cropping after rotation.)
   - **Rotate**: full rotation matrix applied to cropped atoms.
   - **Translate**: atoms shifted to the best-seed position.
   - **KD-tree query**: each atom queried against the 27N seed tree. For
     Voronoi trimming, atoms whose nearest seed is not the current grain's
     best image are discarded. For Laguerre-Voronoi, power distance
     (d² − r²) is used with k=30 nearest seeds.
4. **Modulo wrap**: all atoms are wrapped into the primary box:
   `pos = (pos − box_start) mod box_size + box_start`.

#### Columnar Pipeline (Laminate / Evenly Spaced)

Used when `is_columnar=True` and an `[hkl]` alignment direction is specified:

1. **R_base computation**: a rotation matrix is built that aligns the crystal
   `[hkl]` direction with the simulation Z-axis (stack axis). An orthonormal
   crystal frame (x_crys, y_crys, z_crys) is constructed by scanning
   low-index candidate directions perpendicular to z_crys.
2. **Master pillar**: the entire pristine crystal is pre-rotated by R_base,
   then trimmed to a cylinder: radius = max(grain_diagonals), height =
   1.5 × max_grain_z (the maximum Z-extent of any Voronoi cell).
3. **Per-grain processing**:
   - **Cylindrical pre-crop**: atoms within per-grain cylinder
     (radius = grain_diagonal[g], same height as master pillar).
   - **In-plane rotation**: Rz = R_total × R_base⁻¹, applied at origin.
   - **Translation**: shifted to best-seed position + 0.1 Å margin.
   - **KD-tree query**: same nearest-seed filtering as standard pipeline.
4. **Modulo wrap**: same as standard pipeline.

The columnar pipeline exploits the symmetry of columnar grains: the crystal
is pre-aligned once (R_base shared by all grains), and only the in-plane
rotation (Rz) varies per grain. The cylindrical pre-crop (radius-based) is
more memory-efficient than the spherical pre-crop for the high-aspect-ratio
cells typical of laminate structures.

---

## Database

### Materials Science Context

POLY draws on standard materials-science databases and conventions:

- **Crystal structures**: Bravais lattices (sc, fcc, bcc, hcp, diamond,
  tetragonal, bct) and intermetallic prototypes (L1₂/Ni₃Al, B2/CsCl,
  D0₃/Fe₃Al, L2₁/Heusler, D0₁₉/Mg₃Cd, A15/Cr₃Si, C15/MgCu₂, L1₀/CuAu,
  rocksalt/NaCl, zincblende/ZnS, cesiumchloride/CsCl, fluorite/CaF₂,
  wurtzite/ZnS-hex) are built-in via the `PristineCrystal` module.
- **CSL (Coincidence Site Lattice)**: CSL boundaries are enumerated for
  cubic crystals by the `get_cubic_csl_angles()` function, which solves
  the Diophantine equation m² + N·n² = Σ for integer (m, n) pairs, with
  N = h² + k² + l². This covers the standard Read-Shockley CSL theory for
  cubic grain boundaries.
- **Euler angles**: ZXZ convention (φ₁, Φ, φ₂) in degrees, consistent with
  the Bunge convention widely used in texture analysis and EBSD (Electron
  Back-Scatter Diffraction) data.
- **CPK colouring**: atoms are rendered using Jmol-inspired CPK (Corey-Pauling-Koltun)
  colours for element recognition.
- **Atomic masses**: built-in periodic-table masses (g/mol) with ASE
  (Atomic Simulation Environment) as the preferred runtime source when
  available.

### External Python Libraries

| Library                 | Role                                                                                                                                                                    |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NumPy**               | Array operations, linear algebra, random number generation                                                                                                              |
| **SciPy**               | KD-tree spatial queries (`scipy.spatial.cKDTree`), rotations (`scipy.spatial.transform.Rotation`), Delaunay triangulation, KS statistical test (`scipy.stats.ks_2samp`) |
| **PySide6**             | Qt6 GUI framework (main window, dock widgets, signals/slots, background threads via `QThread`)                                                                          |
| **PyVista / pyvistaqt** | 3D visualization (Voronoi polyhedra, crystal atoms, bounding box, clipping planes). pyvistaqt provides the `QtInteractor` for embedding VTK render windows in Qt        |
| **pyvoro2**             | Periodic Voronoi tessellation with Laguerre (power-distance) support — the core geometrical engine for grain-cell computation                                           |
| **pyqtgraph**           | Fast histogram charts (grain-size and misorientation distributions via `BarGraphItem`)                                                                                  |
| **vtk**                 | Low-level Visualization Toolkit — GPU clipping planes, mesh manipulation                                                                                                |
| **ASE** (optional)      | Atomic masses and chemical-symbol lookup; not required but preferred when available                                                                                     |

---

## Standalone Python Scripts

All backend modules can be used without the GUI — just import the convenience
functions and call them in sequence. The five modules correspond directly to
the numbered pipeline stages from SPEC.md.

### Quick-Start: Minimal 5-Module Pipeline

```python
import numpy as np
from grain_seeds import generate_grains
from pristine_crystal import generate_pristine_bravais
from orientation import generate_orientations
from pc_assembly import assemble_polycrystal

BOX_S = (0.0, 0.0, 0.0)
BOX_E = (100.0, 100.0, 100.0)

# 1. Seeds
grains = generate_grains(
    box_start=BOX_S, box_end=BOX_E, n_grains=20,
    distribution="normal", std_dev=5.0, random_seed=42,
)

# 2. Pristine crystal (origin-centered supercell)
crystal = generate_pristine_bravais(
    "Al", "fcc", a=4.05,
    box_start=BOX_S, box_end=BOX_E,
)

# 3. Orientations
ori = generate_orientations(
    "low_angle", n_grains=grains.n_grains,
    neighbors=grains.neighbors, random_seed=123,
)

# 4–5. Assemble + write LAMMPS files
result = assemble_polycrystal(
    seeds=grains.seeds,
    crystal_atoms=crystal.atoms,
    orientations=ori,
    box_start=BOX_S, box_end=BOX_E,
    data_path="output.data",
    dump_path="output.dump",
)

print(f"Assembled {result.n_atoms:,} atoms in {result.n_grains} grains")
```

### Module 1: `grain_seeds` — Seed Generation & Voronoi Tessellation

```python
from grain_seeds import generate_grains

grains = generate_grains(
    box_start=(0, 0, 0),
    box_end=(100, 100, 100),
    n_grains=30,                     # or avg_diameter=50.0
    distribution="random",           # random | normal | bimodal | customized | laminate | even
    std_dev=8.0,                     # required for normal / laminate+normal
    seed_positions=custom_array,     # required for customized
    seed_radii=radii_array,          # optional: Laguerre-Voronoi radii for customized
    bimodal_params=(0.3, 50, 5, 150, 15),  # (frac, m1, s1, m2, s2) required for bimodal
    random_seed=42,
    laminate_in_plane_dist="random", # random | normal (laminate only)
    laminate_direction="z",          # x | y | z (laminate & even)
    verbose=True,
    # MC callbacks (optional):
    progress_callback=lambda step, max_steps, overlap: print(f"{step}/{max_steps}"),
    pause_callback=None,             # lambda seeds, diams: should_pause -> bool
)
```

**Returns** `SeedResult` with fields:

| Field               | Type            | Description                                         |
| ------------------- | --------------- | --------------------------------------------------- |
| seeds               | (N, 3) ndarray  | Final seed coordinates                              |
| diameters           | (N,) ndarray    | Equivalent-sphere diameters (Å)                     |
| distribution        | str             | Distribution mode used                              |
| n_grains            | int             | Number of grains                                    |
| neighbors           | list[list[int]] | Voronoi adjacency per grain                         |
| polyhedron_data     | list[tuple]     | (grain_id, vertices, image_idx) — all 27 PBC images |
| target_radii        | (N,) ndarray    | Laguerre-Voronoi radii (FBSP only)                  |
| box_start / box_end | ndarray         | Box bounds                                          |

---

### Module 2: `pristine_crystal` — Crystal Supercell Generation

Three convenience functions for different crystal sources:

```python
from pristine_crystal import (
    generate_pristine_bravais,
    generate_pristine_intermetallic,
    generate_pristine_cif,
)

# Bravais lattice
crystal = generate_pristine_bravais(
    "Mg", "hcp", a=3.21, c=5.21,
    box_start=(0, 0, 0), box_end=(100, 100, 100),
    coverage=None,       # auto-computed; override for larger supercell
    margin=2,            # extra unit-cell replicas beyond coverage
    cubic=True,          # force cubic cell for sc/fcc/bcc/diamond/rocksalt/zincblende
    orthorhombic=True,   # force orthorhombic cell for hcp
)

# Intermetallic prototype
crystal = generate_pristine_intermetallic(
    "L1_2", symbols=["Ni", "Al"], a=3.57,
    box_start=(0, 0, 0), box_end=(100, 100, 100),
)

# CIF file
crystal = generate_pristine_cif(
    "structure.cif",
    box_start=(0, 0, 0), box_end=(100, 100, 100),
)
```

**Returns** `CrystalResult` with fields:

| Field       | Type        | Description                       |
| ----------- | ----------- | --------------------------------- |
| `atoms`     | `ase.Atoms` | Full supercell, COM at origin     |
| `n_atoms`   | `int`       | Total atom count                  |
| `repeats`   | `tuple`     | Unit-cell repeats along each axis |
| `unit_cell` | `ase.Atoms` | The underlying unit cell          |

For non-Bravais sources, use the `PristineCrystal` class directly:

```python
from pristine_crystal import PristineCrystal

pc = PristineCrystal.from_spacegroup(
    symbols=["Na", "Cl"], basis=[(0,0,0), (0.5,0.5,0.5)],
    spacegroup=225, cellpar=[5.64, 5.64, 5.64, 90, 90, 90],
    box_start=(0,0,0), box_end=(100,100,100),
)
result = pc.run(margin=2, verbose=True)
```

### Module 3: `orientation` — Orientation Assignment

```python
from orientation import generate_orientations

ori = generate_orientations(
    mode="z_alignment",          # random | z_alignment | low_angle | high_angle
                                  # | custom_misorientation | custom_profile
    n_grains=grains.n_grains,
    neighbors=grains.neighbors,  # required for low/high angle, custom misorientation
    random_seed=123,
    # mode-specific kwargs:
    hkl=(1, 1, 0),               # z_alignment: crystal direction to align with Z
    in_plane="symmetric_csl",    # z_alignment: random | low_angle | high_angle | symmetric_csl
    csl_sigma=5,                 # z_alignment+symmetric_csl: Σ value
    csl_angle_deg=36.87,         # z_alignment+symmetric_csl: CSL angle (degrees)
    target_angles=miso_array,    # custom_misorientation: target distribution
    euler_map={0: (45,90,0), 1: (45,90,35.25)},  # custom_profile: grain→euler
    # custom_misorientation MC parameters:
    max_steps=2000, temp_start=1.0, temp_end=0.001,
    cooling_rate=0.995, perturbation_std=5.0, threshold=0.05,
    verbose=True,
)
```

**Returns** `OrientationResult` with fields:

| Field                   | Type                               | Description                              |
| ----------------------- | ---------------------------------- | ---------------------------------------- |
| `euler_angles`          | `(N, 3) ndarray`                   | ZXZ Euler angles (degrees)               |
| `rotation_matrices`     | `(N, 3, 3) ndarray`                | Rotation matrices                        |
| `rotations`             | `scipy.spatial.transform.Rotation` | Combined Rotation object                 |
| `mode`                  | `str`                              | Orientation mode used                    |
| `misorientation_angles` | `(M,) ndarray`                     | Per-edge misorientation (degrees)        |
| `mc_energy_history`     | `list[float]`                      | Custom misorientation MC energy trace    |
| `mc_acceptance_rate`    | `float`                            | Custom misorientation MC acceptance rate |

### Module 5: `pc_assembly` — Polycrystal Assembly

```python
from pc_assembly import assemble_polycrystal

result = assemble_polycrystal(
    seeds=grains.seeds,
    crystal_atoms=crystal.atoms,
    orientations=ori,
    box_start=(0, 0, 0),
    box_end=(100, 100, 100),
    # Optional advanced parameters:
    target_radii=grains.target_radii,       # Laguerre-Voronoi radii
    grain_diagonals=individual_diags,       # (N,) per-grain diagonals from Voronoi
    is_laminate=False,
    is_columnar=False,
    max_grain_z=None,                       # float, for columnar pillar height
    hkl=(1, 1, 0),                          # for columnar pipeline
    stack_axis="z",                         # for columnar pipeline
    poly_data=grains.polyhedron_data,       # Voronoi vertices
    data_path="output.data",                # write LAMMPS data file
    dump_path="output.dump",                # write LAMMPS dump file
    batch_size=50000,
    verbose=True,
)
```

**Returns** `AssemblyResult` with fields:

| Field                   | Type                   | Description                  |
| ----------------------- | ---------------------- | ---------------------------- |
| `positions`             | `(N_total, 3) ndarray` | Final atom coordinates       |
| `types`                 | `(N_total,) ndarray`   | 1-based LAMMPS atom types    |
| `grain_ids`             | `(N_total,) ndarray`   | 0-based grain index per atom |
| `euler_per_atom`        | `(N_total, 3) ndarray` | Euler angles per atom (deg)  |
| `symbols`               | `list[str]`            | Unique element symbols       |
| `type_to_symbol`        | `dict[int, str]`       | type_id → symbol             |
| `type_masses`           | `dict[int, float]`     | type_id → mass (g/mol)       |
| `n_grains` / `n_atoms`  | `int`                  | Grain and atom counts        |
| `keep_counts`           | `(N_grains,) ndarray`  | Atoms kept per grain         |
| `box_start` / `box_end` | `ndarray`              | Box bounds                   |

### Z-Axis Alignment + Bicrystal Example

The `examples/` directory contains complete runnable scripts:

- **`bicrystal.py`** — Ni₃Al symmetric tilt bicrystal: two grains with [110]
  aligned to Z, ±35.25° in-plane twist, producing a Σ3 CSL boundary.
- **`NaCl.py`** — 40-grain NaCl polycrystal with normal grain-size
  distribution and low-angle boundaries.

### Programmatic Build via `workers.py`

For advanced use (custom coverage, grain-diagonal precomputation, columnar
pipeline), use `PolycrystalBuildWorker` or the underlying functions directly:

```python
from workers import PolycrystalBuildWorker

worker = PolycrystalBuildWorker(
    seed_result=grains,                # SeedResult from Module 1
    orientation_result=ori,            # OrientationResult from Module 3
    crystal_source=0,                  # 0=bravais, 1=intermetallic, 2=spacegroup, 3=file
    crystal_params=dict(
        single_element="Al",
        single_structure="fcc",
        single_a=4.05,
        single_c=4.05,
        box_start=(0, 0, 0),
        box_end=(100, 100, 100),
    ),
    data_path="output.data",
    dump_path="output.dump",
    hkl=(1, 1, 0),                    # for columnar/z-alignment
    distribution="laminate",          # triggers columnar pipeline for laminate/even
)
worker.finished.connect(lambda result: print(f"Done: {result.n_atoms} atoms"))
worker.error.connect(lambda msg: print(f"Error: {msg}"))
worker.run()  # blocking; use worker.start() for QThread async
```

The worker automatically:

1. Computes per-grain Voronoi diagonals and `max_grain_z` from polyhedron data.
2. Determines safe crystal coverage (`2 × max_grain_diag`).
3. Selects the standard or columnar assembly pipeline based on distribution type.
4. Writes LAMMPS output files.

### Validation Pattern

After assembly, validate the result:

```python
pos = result.positions
box_s = result.box_start
box_e = result.box_end

# Check all atoms inside box
in_box = ((pos[:, 0] >= box_s[0]) & (pos[:, 0] <= box_e[0])
        & (pos[:, 1] >= box_s[1]) & (pos[:, 1] <= box_e[1])
        & (pos[:, 2] >= box_s[2]) & (pos[:, 2] <= box_e[2]))

print(f"Atoms in box:    {in_box.sum():,} / {len(pos):,}")
print(f"Min/avg/max kept: {result.keep_counts.min():,} / "
      f"{result.keep_counts.mean():.0f} / {result.keep_counts.max():,}")

# Check for empty grains
empty = result.keep_counts == 0
if empty.any():
    print(f"WARNING: {empty.sum()} empty grains")
```
