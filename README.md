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
  mixture), customized, laminate, or evenly-spaced distributions, with Force-Biased
  Sphere Packing and Voronoi tessellation via pyvoro2.
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


## Getting Started

### Launch

```bash
python gui_main.py
```

A splash screen appears while VTK and the UI load. The main window measures
1280×800 and can be freely resized.

### Window Layout

| Region | Contents |
| --- | --- |
| **Left dock** | "Global Settings" — all simulation inputs. Double-click the title bar to detach/reattach or float the panel. |
| **Top-right viewport** | Pristine crystal unit cell (CPK-coloured atoms + cell wireframe). |
| **Bottom-right viewport** | Voronoi polyhedra with Euler-angle colouring, interactive slicing, and point-picking. |
| **Right charts** | Grain-size histogram (top) and misorientation-angle histogram (bottom). |
| **Bottom bar** | Output folder, state name, and action buttons. |

### Action Buttons (bottom bar, left to right)

| Button | Colour | When Active | Action |
| --- | --- | --- | --- |
| **Save Pristine Crystal** | Green | After crystal generated | Writes `{Name}.crystal` |
| **Save Seed State** | Green | After seeds generated | Writes `{Name}.seed` and `{Name}.euler` |
| **Reroll** | Orange | Always | Clears all caches and re-runs seed + crystal generation |
| **Generate Initial Seeds** | Blue | Always | Runs seed generation + Voronoi tessellation + crystal |
| **Proceed with Build** | Red | Always | Runs full polycrystal assembly, writes `.data` and `.dump` |

**MC controls** (bottom of left dock):

| Control | Action |
| --- | --- |
| **Pause** | Suspends/resumes the Monte Carlo seed-placement optimizer; intermediate state is rendered live |
| **Stop** | Requests early termination of the optimizer |
| **Progress bar** | Shows optimization progress (0–100%) |

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

| Distribution | Description | Extra Fields |
| --- | --- | --- |
| **Random** | Uniform random seed positions | —   |
| **Normal Distribution** | Force-biased sphere packing targeting a log-normal size distribution | StdDev (Å) |
| **Bimodal** | Gaussian mixture model: two normal distributions with number fraction | Fraction in mode 1, two means, two stds; count auto-calculated |
| **Customized** | Load pre-defined seed positions from a `.seed` file | File picker |
| **Laminate** | 2D columnar grains on a mid-plane with in-plane Voronoi | In-plane distribution (`random` or `normal`) |
| **Evenly Spaced (1D)** | Seeds evenly spaced along one axis, spanning the full perpendicular plane | Axis selection (`x`, `y`, or `z`) |

> Choosing Laminate or Evenly Spaced automatically switches the orientation
> mode to Z-axis alignment.

### 4. Configure Crystal Structure

| Source | Fields |
| --- | --- |
| **Bravais** | Structure (`sc`, `fcc`, `bcc`, `hcp`, `diamond`, `tetragonal`, `bct`), Element, a (Å), c (Å) |
| **Intermetallics** | Prototype (`L1₂`, `B2`, `D0₃`, `L2₁`, `D0₁₉`, `A15`, `C15`, `L1₀`, rocksalt, zincblende, etc.), Elements, a (Å), c (Å) |
| **Spacegroup** | Elements, Basis (fractional coordinates), Spacegroup number, Cell parameters (a,b,c,α,β,γ) |
| **Custom (File)** | `.cif`, `.crystal`, or ASE-readable file |

The `a` → `c` ratio is auto-filled for hexagonal structures (c/a ≈ 1.633).

### 5. Choose Orientation Mode

| Mode | Description | Extra Fields |
| --- | --- | --- |
| **Random** | Uniform random Euler angles | —   |
| **Z-axis alignment** | Align a crystal direction `[hkl]` with the simulation Z-axis | (hkl) indices, in-plane mode (`random`, `low_angle`, `high_angle`, `symmetric_csl`), Σ value, CSL angle |
| **Low angle** | Small random misorientation from a base orientation | —   |
| **High angle** | Large random misorientation | —   |
| **Custom Misorientation** | Load misorientation angles from file | File picker |
| **Custom Profile** | Load per-grain Euler angles from a `.euler` file | File picker |

**Symmetric CSL mode**: when `In-plane` is set to `symmetric_csl`, the
application computes all CSL misorientation angles for the given Σ and
`[hkl]` pair. Select one from the dropdown to split the misorientation
symmetrically (e.g., grain 0 gets +θ/2, grain 1 gets −θ/2 around Z).
Box dimensions are auto-tuned to exact integer multiples of the projected
crystal repeat distance in each perpendicular direction.

### 6. Generate Seeds

Click **Generate Initial Seeds**. A background thread:

- Places seeds via the chosen distribution.
- Runs Voronoi tessellation (FBSP Monte Carlo optimization for normal, bimodal,
  and laminar-normal distributions).
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
