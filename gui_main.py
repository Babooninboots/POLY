"""
POLY — Interactive GUI for LAMMPS Polycrystalline Generator.

Author: Yang Zhang (张杨)
Affiliation: Stony Brook University (SUNY at Stony Brook)

Run:
    python gui_main.py
"""

import sys

import numpy as np
import pyvista as pv

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QSplashScreen,
)

from orientation import get_cubic_csl_angles
from workers import (
    CrystalGenerationWorker,
    PolycrystalBuildWorker,
    SeedGenerationWorker,
)
from gui_views import CentralWidget, GlobalSettingsDock, TerminalEmitter


# ---------------------------------------------------------------------------
# CPK colour map (Jmol-inspired) + fallback
# ---------------------------------------------------------------------------

CPK_COLORS: dict[str, str] = {
    "H":  "#ffffff",  "He": "#d9ffff",
    "Li": "#cc80ff",  "Be": "#c2ff00",  "B":  "#ffb5b5",  "C":  "#303030",
    "N":  "#3050f8",  "O":  "#ff0d0d",  "F":  "#90e050",  "Ne": "#b3e3f5",
    "Na": "#ab5cf2",  "Mg": "#8aff00",  "Al": "#bfa6a6",  "Si": "#f0c8a0",
    "P":  "#ff8000",  "S":  "#ffff30",  "Cl": "#1ff01f",  "Ar": "#80d1e3",
    "K":  "#8f40d4",  "Ca": "#3dff00",  "Sc": "#e6e6e6",  "Ti": "#bfc2c7",
    "V":  "#a6a6ab",  "Cr": "#8a99c7",  "Mn": "#9c7ac7",  "Fe": "#e06633",
    "Co": "#f090a0",  "Ni": "#50d050",  "Cu": "#c88033",  "Zn": "#7d80b0",
    "Ga": "#c28f8f",  "Ge": "#668f8f",  "As": "#bd80e3",  "Se": "#ffa100",
    "Br": "#a62929",  "Kr": "#5cb8d1",  "Rb": "#702eb0",  "Sr": "#00ff00",
    "Y":  "#94ffff",  "Zr": "#94e0e0",  "Nb": "#73c2c9",  "Mo": "#54b5b5",
    "Tc": "#3b9e9e",  "Ru": "#248f8f",  "Rh": "#0a7d8c",  "Pd": "#006985",
    "Ag": "#c0c0c0",  "Cd": "#ffd98f",  "In": "#a67573",  "Sn": "#668080",
    "Sb": "#9e63b5",  "Te": "#d47a00",  "I":  "#940094",  "Xe": "#429eb0",
    "Cs": "#57178f",  "Ba": "#00c900",  "La": "#70d4ff",  "Ce": "#ffffc7",
    "Pr": "#d9ffc7",  "Nd": "#c7ffc7",  "Pm": "#a3ffc7",  "Sm": "#8fffc7",
    "Eu": "#61ffc7",  "Gd": "#45ffc7",  "Tb": "#30ffc7",  "Dy": "#1fffc7",
    "Ho": "#00ff9c",  "Er": "#00e675",  "Tm": "#00d452",  "Yb": "#00bf38",
    "Lu": "#00ab24",  "Hf": "#4dc2ff",  "Ta": "#4da6ff",  "W":  "#2194d6",
    "Re": "#267dab",  "Os": "#266696",  "Ir": "#175487",  "Pt": "#d0d0e0",
    "Au": "#ffd123",  "Hg": "#b8b8d0",  "Tl": "#a6544d",  "Pb": "#575961",
    "Bi": "#9e4fb5",
}

_FALLBACK_COLORS: list[str] = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
    "#469990", "#e6beff", "#9a6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]
_fallback_counter: int = 0


def _cpk_color(symbol: str) -> str:
    """Return a CPK colour for *symbol*, falling back to a cycling palette."""
    global _fallback_counter
    colour = CPK_COLORS.get(symbol)
    if colour is not None:
        return colour
    idx = _fallback_counter % len(_FALLBACK_COLORS)
    _fallback_counter += 1
    return _FALLBACK_COLORS[idx]


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """POLY application shell."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("POLY — Polycrystalline Generator")
        self.resize(1280, 800)

        # Dock
        self.settings_dock = GlobalSettingsDock()
        self.addDockWidget(Qt.LeftDockWidgetArea, self.settings_dock)

        # Central
        self.central = CentralWidget()
        self.setCentralWidget(self.central)

        # Wire dock buttons
        self.central.generate_clicked.connect(
            self._on_generate_seeds_clicked
        )
        self.central.reroll_clicked.connect(
            self._on_reroll_clicked
        )
        self.settings_dock.pause_btn.clicked.connect(
            self._on_pause_toggled
        )
        self.settings_dock.stop_btn.clicked.connect(
            self._on_stop_clicked
        )
        self.settings_dock.edit_apply_clicked.connect(
            self._on_apply_edit_clicked
        )
        self.settings_dock.edit_grain_id.valueChanged.connect(
            self._on_grain_id_changed
        )
        # Auto-tune box dimensions when distribution / crystal / orientation change
        dock = self.settings_dock
        dock.dist_combo.currentIndexChanged.connect(self._auto_tune_dimensions)
        dock.ori_combo.currentIndexChanged.connect(self._auto_tune_dimensions)
        dock.ori_hkl_edit.textChanged.connect(self._auto_tune_dimensions)
        dock.ori_in_plane_combo.currentTextChanged.connect(self._auto_tune_dimensions)
        dock.ori_csl_spin.valueChanged.connect(self._auto_tune_dimensions)
        dock.crystal_combo.currentIndexChanged.connect(self._auto_tune_dimensions)
        dock.crystal_type_combo.currentTextChanged.connect(self._auto_tune_dimensions)
        dock.crystal_elem_edit.textChanged.connect(self._auto_tune_dimensions)
        dock.crystal_shared_a.valueChanged.connect(self._auto_tune_dimensions)
        dock.crystal_shared_c.valueChanged.connect(self._auto_tune_dimensions)
        dock.crystal_sg_elements.textChanged.connect(self._auto_tune_dimensions)
        dock.crystal_sg_basis.textChanged.connect(self._auto_tune_dimensions)
        dock.crystal_sg_group.textChanged.connect(self._auto_tune_dimensions)
        dock.crystal_sg_cellpar.textChanged.connect(self._auto_tune_dimensions)
        dock.ori_csl_spin.valueChanged.connect(self._update_csl_angle)
        dock.ori_hkl_edit.textChanged.connect(self._update_csl_angle)
        dock.ori_csl_angle_combo.currentIndexChanged.connect(self._auto_tune_dimensions)
        self._laminate_z_tune_lock = False
        self._update_csl_angle()
        self._seed_worker: SeedGenerationWorker | None = None
        self._crystal_worker: CrystalGenerationWorker | None = None
        self._build_worker: PolycrystalBuildWorker | None = None
        self._pending_workers: int = 0
        self._seed_result = None
        self._orientation_result = None
        self._crystal_atoms = None
        self._last_seed_params = None
        self._last_ori_params = None
        self._last_crystal_params = None
        self._last_progress_msg = ""
        self._render_seeds = True
        self._render_crystal = True
        self._highlight_actor = None
        # crystal parameters (stored for build step)
        self._crystal_source: int = 0
        self._crystal_params: dict = {}
        self.central.save_clicked.connect(self._on_save)
        self.central.save_state_clicked.connect(self._on_save_seed_state)
        self.central.build_clicked.connect(self._on_build)

        import vtk
        self._clip_plane = vtk.vtkPlane()
        self.central.slice_slider.valueChanged.connect(self._update_clipping_plane)
        self.central.ortho_btn.clicked.connect(self._update_clipping_plane)

    # ------------------------------------------------------------------
    # Seed generation slot (Phase 2.5)
    # ------------------------------------------------------------------

    def _on_generate_seeds_clicked(self) -> None:
        """Extract GUI parameters and start background seed generation."""
        dock = self.settings_dock

        # --- box bounds ---
        box_start = (
            dock.box_min_x.value(),
            dock.box_min_y.value(),
            dock.box_min_z.value(),
        )
        box_end = (
            dock.box_max_x.value(),
            dock.box_max_y.value(),
            dock.box_max_z.value(),
        )

        # --- grain quantity ---
        if dock.qty_mode_combo.currentIndex() == 0:
            n_grains = dock.qty_spin.value()
            avg_diameter = None
        else:
            n_grains = None
            avg_diameter = dock.qty_diam_spin.value()

        # --- distribution ---
        dist_map = {0: "random", 1: "normal", 2: "customized", 3: "laminate", 4: "even"}
        distribution = dist_map[dock.dist_combo.currentIndex()]
        std_dev = dock.dist_stddev.value() if distribution in ("normal", "laminate") else None
        seed_file = (
            dock.dist_custom_picker.path
            if distribution == "customized"
            else None
        )

        # --- crystal source & params (needed early for laminate d_hkl) ---
        crystal_source = dock.crystal_combo.currentIndex()
        box_start_arr = np.array(box_start, dtype=float)
        box_end_arr = np.array(box_end, dtype=float)
        box_diag = float(np.linalg.norm(box_end_arr - box_start_arr))
        safe_coverage = box_diag * 2.0
        crystal_params = dict(
            single_structure=dock.crystal_type_combo.currentText().strip(),
            single_element=dock.crystal_elem_edit.text().strip(),
            single_a=dock.crystal_shared_a.value(),
            single_c=dock.crystal_shared_c.value(),
            inter_type=dock.crystal_type_combo.currentText().strip(),
            inter_elements=dock.crystal_elem_edit.text().strip(),
            inter_a=dock.crystal_shared_a.value(),
            inter_c=dock.crystal_shared_c.value(),
            sg_elements=dock.crystal_sg_elements.text().strip(),
            sg_basis=dock.crystal_sg_basis.text().strip(),
            sg_spacegroup=dock.crystal_sg_group.text().strip(),
            sg_cellpar=dock.crystal_sg_cellpar.text().strip(),
            custom_file=dock.crystal_custom_picker.path,
            box_start=box_start,
            box_end=box_end,
            coverage=safe_coverage,
        )

        laminate_in_plane_dist = "random"
        laminate_direction = "z"
        if distribution == "laminate":
            laminate_in_plane_dist = dock.dist_lam_inplane_dist_combo.currentText()
        elif distribution == "even":
            laminate_direction = dock.dist_lam_inplane_dist_combo.currentText()

        # Delegate box-dimension auto-tuning to the shared method so
        # Generate and live-update use the same algorithm.
        if distribution in ("laminate", "even"):
            self._auto_tune_dimensions()
            box_end = (
                dock.box_max_x.value(),
                dock.box_max_y.value(),
                dock.box_max_z.value(),
            )
            crystal_params["box_end"] = box_end

        # --- orientation mode ---
        ori_map = {
            0: "random",
            1: "z_alignment",
            2: "low_angle",
            3: "high_angle",
            4: "custom_misorientation",
            5: "custom_profile",
        }
        orientation_mode = ori_map[dock.ori_combo.currentIndex()]

        hkl = None
        ori_custom_file = None
        in_plane = "random"
        csl_sigma = 5
        csl_angle_deg = 0.0
        if orientation_mode == "z_alignment":
            text = dock.ori_hkl_edit.text().strip()
            hkl = tuple(int(x) for x in text.split())
            in_plane = dock.ori_in_plane_combo.currentText()
            csl_sigma = dock.ori_csl_spin.value()
            try:
                csl_angle_deg = float(dock.ori_csl_angle_combo.currentText().replace('°', ''))
            except ValueError:
                csl_angle_deg = 0.0
        elif orientation_mode in ("custom_misorientation", "custom_profile"):
            ori_custom_file = dock.ori_custom_picker.path

        # --- change detection ---
        current_seed_params = {
            "box_start": box_start, "box_end": box_end,
            "n_grains": n_grains, "avg_diameter": avg_diameter,
            "distribution": distribution, "std_dev": std_dev,
            "seed_positions_file": seed_file,
            "laminate_in_plane_dist": laminate_in_plane_dist,
            "laminate_direction": laminate_direction,
        }
        current_ori_params = {
            "orientation_mode": orientation_mode, "hkl": hkl,
            "ori_custom_file": ori_custom_file, "in_plane": in_plane,
            "csl_sigma": csl_sigma,
            "csl_angle_deg": csl_angle_deg,
        }
        current_crystal_params = crystal_params.copy()

        seeds_changed = self._seed_result is None or current_seed_params != self._last_seed_params
        ori_changed = self._orientation_result is None or current_ori_params != self._last_ori_params
        crystal_changed = self._crystal_atoms is None or current_crystal_params != self._last_crystal_params

        if not (seeds_changed or ori_changed or crystal_changed):
            # Nothing changed — nothing to do
            return

        run_seeds = seeds_changed
        run_ori = ori_changed or seeds_changed
        run_crystal = crystal_changed

        self._last_seed_params = current_seed_params
        self._last_ori_params = current_ori_params
        self._last_crystal_params = current_crystal_params

        # Track what ran so _render_all only updates what changed
        self._render_seeds = run_seeds or run_ori
        self._render_crystal = run_crystal

        # --- launch necessary workers ---
        workers_to_run = 0
        if run_seeds or run_ori:
            workers_to_run += 1
        if run_crystal:
            workers_to_run += 1

        self.central.generate_btn.setEnabled(False)
        self.central.reroll_btn.setEnabled(False)
        dock.pause_btn.setChecked(False)
        dock.pause_btn.setText("Pause")
        dock.pause_btn.setEnabled(True)
        dock.stop_btn.setEnabled(True)
        self.statusBar().showMessage("Updating polycrystal components…")
        self._pending_workers = workers_to_run

        if run_seeds or run_ori:
            self._seed_worker = SeedGenerationWorker(
                box_start=box_start,
                box_end=box_end,
                n_grains=n_grains,
                avg_diameter=avg_diameter,
                distribution=distribution,
                std_dev=std_dev,
                seed_positions_file=seed_file,
                orientation_mode=orientation_mode,
                hkl=hkl,
                ori_custom_file=ori_custom_file,
                in_plane=in_plane,
                csl_sigma=csl_sigma,
                csl_angle_deg=csl_angle_deg,
                laminate_in_plane_dist=laminate_in_plane_dist,
                laminate_direction=laminate_direction,
                run_seeds=run_seeds,
                run_ori=run_ori,
                cached_seed_result=self._seed_result,
                cached_ori_result=self._orientation_result,
            )
            self._seed_worker.finished.connect(self._on_generation_finished)
            self._seed_worker.progress.connect(self._on_seed_progress)
            self._seed_worker.error.connect(self._on_generation_error)
            self._seed_worker.start()
            dock.progress_bar.setValue(0)
            dock.progress_bar.setVisible(True)

        if run_crystal:
            self._crystal_source = crystal_source
            self._crystal_params = crystal_params
            worker_params = {k: v for k, v in crystal_params.items() if k != "coverage"}
            self._crystal_worker = CrystalGenerationWorker(
                crystal_source=crystal_source,
                **worker_params,
            )
            self._crystal_worker.finished_crystal.connect(self._on_crystal_finished)
            self._crystal_worker.error_crystal.connect(self._on_crystal_error)
            self._crystal_worker.start()

    def _on_reroll_clicked(self) -> None:
        """Force full recalculation of everything, ignoring caches."""
        self._last_seed_params = None
        self._last_ori_params = None
        self._last_crystal_params = None
        self._render_seeds = True
        self._render_crystal = True
        self._on_generate_seeds_clicked()

    def _on_pause_toggled(self, checked: bool) -> None:
        if self._seed_worker is not None:
            if checked:
                self.settings_dock.pause_btn.setText("Resume")
                self._seed_worker.pause()
                self.statusBar().showMessage(
                    f"{getattr(self, '_last_progress_msg', 'Optimizing…')}  ──  Paused"
                )
                # Render current MC state
                seeds = getattr(self._seed_worker, "_current_seeds", None)
                diameters = getattr(self._seed_worker, "_current_diameters", None)
                if seeds is not None and diameters is not None and self._seed_result is not None:
                    from grain_seeds import GrainSeedGenerator
                    gen = GrainSeedGenerator(
                        box_start=tuple(float(x) for x in self._seed_result.box_start),
                        box_end=tuple(float(x) for x in self._seed_result.box_end),
                        distribution="customized",
                        seed_positions=seeds,
                    )
                    gen.generate_seeds()
                    _, poly_data = gen.compute_grain_cells()
                    self._seed_result.seeds = seeds.copy()
                    self._seed_result.diameters = diameters.copy()
                    self._seed_result.polyhedron_data = poly_data
                    self._render_voronoi_viewport(
                        self._seed_result, self._orientation_result,
                    )
                    size_target, miso_target = self._chart_target_params()
                    self.central.update_charts(
                        self._seed_result.diameters,
                        self._orientation_result.misorientation_angles,
                        target_size_params=size_target,
                        target_miso_angles=miso_target,
                    )
                    # Also render crystal structure if already generated
                    if self._crystal_atoms is not None:
                        self._render_crystal_viewport(self._crystal_atoms)
                QApplication.processEvents()
            else:
                self.settings_dock.pause_btn.setText("Pause")
                self._seed_worker.resume()
                self.statusBar().showMessage(
                    getattr(self, "_last_progress_msg", "Optimizing…")
                )

    def _on_stop_clicked(self) -> None:
        """Request early stop of MC optimization."""
        if self._seed_worker is not None:
            self._seed_worker.stop()
            self.settings_dock.pause_btn.setChecked(False)
            self.settings_dock.pause_btn.setText("Pause")
            self.settings_dock.pause_btn.setEnabled(False)
            self.settings_dock.stop_btn.setEnabled(False)
            self.statusBar().showMessage("MC Optimization stopped…")

    def _worker_done(self) -> None:
        """Decrement pending counter; render when all workers finish."""
        self._pending_workers -= 1
        if self._pending_workers <= 0:
            self.settings_dock.progress_bar.setVisible(False)
            self.central.generate_btn.setEnabled(True)
            self.central.reroll_btn.setEnabled(True)
            self.settings_dock.pause_btn.setEnabled(False)
            self.settings_dock.stop_btn.setEnabled(False)
            self._render_all()

    def _render_all(self) -> None:
        """Render viewports and charts that have changed."""
        if self._render_seeds and self._seed_result is not None:
            self._render_voronoi_viewport(
                self._seed_result, self._orientation_result,
            )
            size_target, miso_target = self._chart_target_params()
            self.central.update_charts(
                self._seed_result.diameters,
                self._orientation_result.misorientation_angles,
                target_size_params=size_target,
                target_miso_angles=miso_target,
            )
        if self._render_crystal and self._crystal_atoms is not None:
            self._render_crystal_viewport(self._crystal_atoms)
        self._render_seeds = False
        self._render_crystal = False

    def _chart_target_params(self) -> tuple[dict | None, np.ndarray | None]:
        """Compute target-curve parameters for the grain-size and
        misorientation charts, based on the last-used seed/orientation params.

        Returns
        -------
        size_target : dict or None
            ``{'mean': float, 'std': float}`` when a normal grain-size
            distribution was requested, else ``None``.
        miso_target : ndarray or None
            Target misorientation angles loaded from the custom-misorientation
            file, if that mode was used.
        """
        size_target = None
        miso_target = None

        sp = self._last_seed_params
        if sp is not None:
            dist = sp["distribution"]
            std_dev = sp.get("std_dev")
            if dist == "normal" and std_dev is not None:
                box_start = np.asarray(sp["box_start"], dtype=float)
                box_end = np.asarray(sp["box_end"], dtype=float)
                box_size = box_end - box_start
                box_vol = float(np.prod(box_size))
                n = sp["n_grains"]
                mean = 2.0 * (3.0 * box_vol / (4.0 * np.pi * n)) ** (1.0 / 3.0)
                size_target = {"mean": mean, "std": std_dev}
            elif dist == "laminate" and sp.get("laminate_in_plane_dist") == "normal" and std_dev is not None:
                box_start = np.asarray(sp["box_start"], dtype=float)
                box_end = np.asarray(sp["box_end"], dtype=float)
                box_size = box_end - box_start
                lam_dir = sp.get("laminate_direction", "z")
                axis_map = {"x": 0, "y": 1, "z": 2}
                locked = axis_map[lam_dir]
                active = [i for i in range(3) if i != locked]
                plane_area = float(np.prod(box_size[active]))
                n = sp["n_grains"]
                mean = 2.0 * np.sqrt(plane_area / (np.pi * n))
                size_target = {"mean": mean, "std": std_dev}

        op = self._last_ori_params
        if op is not None:
            if op.get("orientation_mode") == "custom_misorientation":
                path = op.get("ori_custom_file")
                if path:
                    try:
                        miso_target = np.loadtxt(path)
                    except Exception:
                        miso_target = None

        return size_target, miso_target

    def _on_seed_progress(self, pct: int, msg: str) -> None:
        self.settings_dock.progress_bar.setValue(pct)
        self._last_progress_msg = msg
        self.statusBar().showMessage(msg)

    def _on_generation_finished(self, seed_result, orientation_result) -> None:
        """Store seed/orientation results; render deferred to _render_all."""
        self._seed_result = seed_result
        self._orientation_result = orientation_result

        # configure editor spinbox
        dock = self.settings_dock
        dock.edit_grain_id.blockSignals(True)
        dock.edit_grain_id.setRange(-1, seed_result.n_grains - 1)
        dock.edit_grain_id.setValue(-1)
        dock.edit_grain_id.setEnabled(True)
        dock.edit_grain_id.blockSignals(False)
        self.central.save_state_btn.setEnabled(True)

        # push box dimensions from seed result back to GUI spinboxes
        bs = seed_result.box_start
        be = seed_result.box_end
        dock.box_min_x.blockSignals(True)
        dock.box_min_y.blockSignals(True)
        dock.box_min_z.blockSignals(True)
        dock.box_max_x.blockSignals(True)
        dock.box_max_y.blockSignals(True)
        dock.box_max_z.blockSignals(True)
        dock.box_min_x.setValue(float(bs[0]))
        dock.box_min_y.setValue(float(bs[1]))
        dock.box_min_z.setValue(float(bs[2]))
        dock.box_max_x.setValue(float(be[0]))
        dock.box_max_y.setValue(float(be[1]))
        dock.box_max_z.setValue(float(be[2]))
        dock.box_min_x.blockSignals(False)
        dock.box_min_y.blockSignals(False)
        dock.box_min_z.blockSignals(False)
        dock.box_max_x.blockSignals(False)
        dock.box_max_y.blockSignals(False)
        dock.box_max_z.blockSignals(False)

        self._worker_done()
        self.statusBar().showMessage(
            f"Generated {seed_result.n_grains} grains.", 5000
        )

    def _render_voronoi_viewport(self, seed_result, orientation_result) -> None:
        """Render Voronoi polyhedra + seeds in the left viewport."""
        plotter = self.central.voronoi_plotter
        plotter.clear()
        plotter.enable_depth_peeling(number_of_peels=5)
        plotter.enable_anti_aliasing("msaa")

        seeds = seed_result.seeds
        eulers = orientation_result.euler_angles  # (N, 3)
        poly_data = seed_result.polyhedron_data

        # min-max normalise each Euler column → RGB channel
        colours = np.zeros_like(eulers)
        for i in range(3):
            col_min = float(eulers[:, i].min())
            col_max = float(eulers[:, i].max())
            if col_max > col_min:
                colours[:, i] = (eulers[:, i] - col_min) / (col_max - col_min)
            else:
                colours[:, i] = 0.5

        box_start = seed_result.box_start
        box_end = seed_result.box_end

        eps = 1e-4
        pad_bounds = (
            box_start[0] - eps, box_end[0] + eps,
            box_start[1] - eps, box_end[1] + eps,
            box_start[2] - eps, box_end[2] + eps,
        )

        box_diag = float(np.linalg.norm(box_end - box_start))

        from scipy.spatial import Delaunay as _Delaunay

        total_cells = len(poly_data)
        total_seeds = len(seeds)
        # Poly cells + seed spheres + bbox + 3 axes
        total_steps = total_cells + total_seeds + 4
        pb = self.central.voronoi_progress
        pb.setRange(0, total_steps)
        pb.setValue(0)
        pw = plotter.width()
        pb.setGeometry((pw - 300) // 2, 6, 300, 20)
        pb.setVisible(True)
        QApplication.processEvents()

        step = 0

        # render seed points as glyphs (single merged actor)
        seed_radius = 0.05 * box_diag / (max(seed_result.n_grains, 1) ** (1/3))
        seed_cloud = pv.PolyData(seeds)
        seed_cloud.point_data["GrainID"] = np.arange(len(seeds))
        sphere_geom = pv.Sphere(radius=seed_radius)
        self._merged_seeds = seed_cloud.glyph(
            geom=sphere_geom, orient=False, scale=False,
        )
        n_pts = self._merged_seeds.n_points
        rgba = np.zeros((n_pts, 4))
        rgba[:, 3] = 0.5
        self._merged_seeds.point_data["RGBA"] = rgba
        seed_actor = plotter.add_mesh(
            self._merged_seeds,
            scalars="RGBA",
            rgb=True,
            show_edges=False,
            name="merged_seeds",
        )
        seed_actor.mapper.AddClippingPlane(self._clip_plane)
        step += len(seeds)
        pb.setValue(step)
        QApplication.processEvents()

        blocks = pv.MultiBlock()
        for cell_idx, (parent_id, verts, _) in enumerate(poly_data):
            if len(verts) < 4:
                step += 1
                pb.setValue(step)
                continue

            v = verts

            try:
                tet = _Delaunay(v, qhull_options="QJ")
            except Exception:
                step += 1
                pb.setValue(step)
                continue

            ncells = tet.nsimplex
            cell_types = np.full(ncells, pv.CellType.TETRA, dtype=np.uint8)
            prefix = np.full((ncells, 1), 4, dtype=np.int64)
            cells = np.hstack([prefix, tet.simplices]).ravel()
            ugrid = pv.UnstructuredGrid(cells, cell_types, v)

            clipped = ugrid.clip_box(pad_bounds, invert=False)
            if clipped.n_cells == 0:
                step += 1
                pb.setValue(step)
                continue

            surf = clipped.extract_surface(algorithm="dataset_surface")
            if surf.n_points == 0:
                step += 1
                pb.setValue(step)
                continue

            # Shrink mesh to 99 % of local center for visual grain-boundary cracks
            center = surf.center
            surf.points = (surf.points - center) * 0.99 + center

            rgb = tuple(float(c) for c in colours[parent_id])
            surf.cell_data["RGB"] = np.tile(rgb, (surf.n_cells, 1))
            surf.cell_data["GrainID"] = np.full(surf.n_cells, parent_id)
            blocks.append(surf)

            step += 1
            pb.setValue(step)
            if step % 5 == 0:
                QApplication.processEvents()

        if len(blocks) > 0:
            self._merged_voronoi = blocks.combine()
            n_cells = self._merged_voronoi.n_cells
            rgb = self._merged_voronoi.cell_data["RGB"]
            alpha = np.full((n_cells, 1), 0.5)
            self._merged_voronoi.cell_data["RGBA"] = np.hstack([rgb, alpha])
            voronoi_actor = plotter.add_mesh(
                self._merged_voronoi,
                scalars="RGBA",
                rgb=True,
                show_edges=False,
                name="merged_grains",
            )
            voronoi_actor.mapper.AddClippingPlane(self._clip_plane)
        else:
            self._merged_voronoi = None

        # bounding box
        plotter.add_mesh(
            pv.Box(
                bounds=(
                    box_start[0], box_end[0],
                    box_start[1], box_end[1],
                    box_start[2], box_end[2],
                )
            ),
            color="white", style="wireframe", line_width=1,
            name="bbox",
        )
        step += 1
        pb.setValue(step)

        # Axes centered on box, 120 % of each box dimension
        box_center = (box_start + box_end) / 2.0
        box_dims = box_end - box_start
        half = box_dims * 0.6
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        for axis in range(3):
            start = box_center.copy()
            end = box_center.copy()
            start[axis] -= half[axis]
            end[axis] += half[axis]
            plotter.add_mesh(
                pv.Line(start, end), color=colors[axis], line_width=2,
                name=f"axis_{'xyz'[axis]}",
            )
            step += 1
            pb.setValue(step)

        pb.setVisible(False)
        self.central._box_center = box_center
        plotter.camera.focal_point = box_center
        if self.central._view_state == -1:
            plotter.view_isometric()
        else:
            self.central._apply_camera_view()
        plotter.reset_camera()
        self._update_clipping_plane()

    def _on_generation_error(self, error_msg: str) -> None:
        """Handle seed-generation failure and still count for completion."""
        self.statusBar().showMessage(f"Seed error: {error_msg}", 8000)
        print(f"[GUI] Seed generation error: {error_msg}")
        self._worker_done()

    def _on_crystal_finished(self, atoms) -> None:
        """Store crystal atoms; render deferred to _render_all."""
        self._crystal_atoms = atoms
        self.central.save_btn.setEnabled(True)

        # push cell params from loaded .crystal file back to GUI spinboxes
        cell_info = atoms.info.get("_cell_params") if hasattr(atoms, "info") else None
        if cell_info is not None:
            dock = self.settings_dock
            dock.crystal_shared_a.blockSignals(True)
            dock.crystal_shared_c.blockSignals(True)
            dock.crystal_shared_a.setValue(cell_info["a"])
            dock.crystal_shared_c.setValue(cell_info["c"])
            dock.crystal_shared_a.blockSignals(False)
            dock.crystal_shared_c.blockSignals(False)

        self._worker_done()
        self.statusBar().showMessage(
            f"Crystal: {len(atoms)} atoms.", 5000
        )

    def _on_crystal_error(self, error_msg: str) -> None:
        """Handle crystal-generation failure and still count for completion."""
        self.statusBar().showMessage(f"Crystal error: {error_msg}", 8000)
        print(f"[GUI] Crystal generation error: {error_msg}")
        self._crystal_atoms = None
        self._worker_done()

    def _render_crystal_viewport(self, atoms) -> None:
        """Render pristine crystal atoms in the top-right viewport."""
        plotter = self.central.crystal_plotter
        plotter.clear()

        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        # group atoms by element, render each as one PolyData
        unique_syms = sorted(set(symbols), key=symbols.index)
        for sym in unique_syms:
            mask = np.array([s == sym for s in symbols])
            pts = pv.PolyData(positions[mask])
            plotter.add_mesh(
                pts, color=_cpk_color(sym), point_size=14,
                render_points_as_spheres=True, style="points",
                name=f"crystal_{sym}",
            )

        # overlay 1×1×1 unit-cell wireframe box
        supercell_dims = np.diag(atoms.cell)          # 2×2×2 supercell size
        unit_dims = supercell_dims / 2.0              # single unit-cell size
        x, y, z = unit_dims
        corners = np.array([
            [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
            [0, 0, z], [x, 0, z], [x, y, z], [0, y, z],
        ])
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ])
        box_lines = pv.PolyData(corners, lines=np.hstack([np.full((len(edges), 1), 2), edges]).ravel())
        plotter.add_mesh(box_lines, color="black", line_width=2)

        plotter.add_axes()
        plotter.view_isometric()
        plotter.reset_camera()

    # ------------------------------------------------------------------
    # Point picking & grain editing
    # ------------------------------------------------------------------

    def _on_grain_id_changed(self, grain_id: int) -> None:
        """Populate editor fields from the selected grain and highlight it."""
        dock = self.settings_dock

        if self._seed_result is None or self._orientation_result is None:
            return
        if grain_id < 0 or grain_id >= self._seed_result.n_grains:
            # Blank selection — clear fields, remove highlight, re-render all
            dock.edit_x.blockSignals(True)
            dock.edit_y.blockSignals(True)
            dock.edit_z.blockSignals(True)
            dock.edit_alpha.blockSignals(True)
            dock.edit_beta.blockSignals(True)
            dock.edit_gamma.blockSignals(True)
            dock.edit_grain_diam.blockSignals(True)
            dock.edit_x.setValue(0.0)
            dock.edit_y.setValue(0.0)
            dock.edit_z.setValue(0.0)
            dock.edit_alpha.setValue(0.0)
            dock.edit_beta.setValue(0.0)
            dock.edit_gamma.setValue(0.0)
            dock.edit_grain_diam.setValue(0.0)
            dock.edit_x.blockSignals(False)
            dock.edit_y.blockSignals(False)
            dock.edit_z.blockSignals(False)
            dock.edit_alpha.blockSignals(False)
            dock.edit_beta.blockSignals(False)
            dock.edit_gamma.blockSignals(False)
            dock.edit_grain_diam.blockSignals(False)
            dock.edit_apply_btn.setEnabled(False)
            self._clear_highlight()
            return

        seeds = self._seed_result.seeds
        eulers = self._orientation_result.euler_angles

        dock.edit_grain_id.blockSignals(True)
        dock.edit_x.blockSignals(True)
        dock.edit_y.blockSignals(True)
        dock.edit_z.blockSignals(True)
        dock.edit_alpha.blockSignals(True)
        dock.edit_beta.blockSignals(True)
        dock.edit_gamma.blockSignals(True)
        dock.edit_grain_diam.blockSignals(True)

        dock.edit_grain_id.setValue(grain_id)
        dock.edit_x.setValue(float(seeds[grain_id, 0]))
        dock.edit_y.setValue(float(seeds[grain_id, 1]))
        dock.edit_z.setValue(float(seeds[grain_id, 2]))
        dock.edit_alpha.setValue(float(eulers[grain_id, 0]))
        dock.edit_beta.setValue(float(eulers[grain_id, 1]))
        dock.edit_gamma.setValue(float(eulers[grain_id, 2]))
        dock.edit_grain_diam.setValue(float(self._seed_result.diameters[grain_id]))

        dock.edit_grain_id.blockSignals(False)
        dock.edit_x.blockSignals(False)
        dock.edit_y.blockSignals(False)
        dock.edit_z.blockSignals(False)
        dock.edit_alpha.blockSignals(False)
        dock.edit_beta.blockSignals(False)
        dock.edit_gamma.blockSignals(False)
        dock.edit_grain_diam.blockSignals(False)
        dock.edit_apply_btn.setEnabled(True)

        self._update_highlight(seeds[grain_id], grain_id)

    def _auto_tune_dimensions(self) -> None:
        """Snap box dimensions to exact integer multiples of the true 3D repeat distance."""
        if self._laminate_z_tune_lock:
            return
        dock = self.settings_dock

        dist_map = {0: "random", 1: "normal", 2: "customized", 3: "laminate", 4: "even"}
        distribution = dist_map[dock.dist_combo.currentIndex()]

        ori_map = {0: "random", 1: "z_alignment", 2: "low_angle", 3: "high_angle",
                   4: "custom_misorientation", 5: "custom_profile"}
        orientation_mode = ori_map[dock.ori_combo.currentIndex()]
        in_plane = dock.ori_in_plane_combo.currentText()

        if distribution not in ("laminate", "even") and orientation_mode != "z_alignment":
            return

        box_start = (dock.box_min_x.value(), dock.box_min_y.value(), dock.box_min_z.value())
        box_end = (dock.box_max_x.value(), dock.box_max_y.value(), dock.box_max_z.value())

        crystal_source = dock.crystal_combo.currentIndex()
        crystal_params = dict(
            single_structure=dock.crystal_type_combo.currentText().strip(),
            single_element=dock.crystal_elem_edit.text().strip(),
            single_a=dock.crystal_shared_a.value(),
            single_c=dock.crystal_shared_c.value(),
            inter_type=dock.crystal_type_combo.currentText().strip(),
            inter_elements=dock.crystal_elem_edit.text().strip(),
            inter_a=dock.crystal_shared_a.value(),
            inter_c=dock.crystal_shared_c.value(),
            sg_elements=dock.crystal_sg_elements.text().strip(),
            sg_basis=dock.crystal_sg_basis.text().strip(),
            sg_spacegroup=dock.crystal_sg_group.text().strip(),
            sg_cellpar=dock.crystal_sg_cellpar.text().strip(),
            custom_file=dock.crystal_custom_picker.path,
        )

        self._laminate_z_tune_lock = True
        try:
            from pristine_crystal import _build_for_hkl
            try:
                pc = _build_for_hkl(crystal_source, crystal_params)
                C = pc.unit_cell.get_cell()
                pos = pc.unit_cell.get_positions()
            except Exception:
                return

            grid_range = np.arange(-30, 31)
            u, v_grid, w = np.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
            shifts = np.column_stack([u.ravel(), v_grid.ravel(), w.ravel()]) @ C
            points = shifts[:, np.newaxis, :] + pos[np.newaxis, :, :]
            points = points.reshape(-1, 3)
            points = points[np.linalg.norm(points, axis=1) > 1e-5]

            from scipy.spatial.transform import Rotation
            R_total = Rotation.identity()

            if orientation_mode == "z_alignment":
                hkl_text = dock.ori_hkl_edit.text().strip()
                try:
                    hkl = tuple(int(x) for x in hkl_text.split())
                    if len(hkl) != 3: hkl = (0, 0, 1)
                except Exception:
                    hkl = (0, 0, 1)

                n = np.array(hkl, dtype=float)
                n_norm = np.linalg.norm(n)
                z_crys = n / n_norm if n_norm > 0 else np.array([0.0, 0.0, 1.0])

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

                if x_crys is None:
                    arb = np.array([1.0, 0.0, 0.0]) if abs(z_crys[0]) < 0.99 else np.array([0.0, 1.0, 0.0])
                    x_crys = np.cross(z_crys, arb)
                    x_crys /= np.linalg.norm(x_crys)

                y_crys = np.cross(z_crys, x_crys)
                y_crys /= np.linalg.norm(y_crys)

                R_base = Rotation.from_matrix(np.vstack([x_crys, y_crys, z_crys]))

                if in_plane == "symmetric_csl":
                    try:
                        csl_angle_deg = float(dock.ori_csl_angle_combo.currentText().replace('°', ''))
                    except ValueError:
                        csl_angle_deg = 0.0
                    R_total = Rotation.from_rotvec(np.radians(csl_angle_deg / 2.0) * np.array([0.0, 0.0, 1.0])) * R_base
                else:
                    R_total = R_base

            rotated_points = points @ R_total.as_matrix().T

            def get_period(axis_idx):
                other1 = (axis_idx + 1) % 3
                other2 = (axis_idx + 2) % 3
                mask = (np.abs(rotated_points[:, other1]) <= 0.1) & \
                       (np.abs(rotated_points[:, other2]) <= 0.1) & \
                       (rotated_points[:, axis_idx] > 0.5)
                if np.any(mask):
                    return np.min(rotated_points[mask, axis_idx])
                return 0.0

            D_x = get_period(0)
            D_y = get_period(1)
            D_z = get_period(2)

            if distribution in ("laminate", "even"):
                stack_ax = 2
                if distribution == "even":
                    axis_str = dock.dist_lam_inplane_dist_combo.currentText()
                    stack_ax = {"x": 0, "y": 1, "z": 2}.get(axis_str, 2)

                D_stack = [D_x, D_y, D_z][stack_ax]
                if D_stack > 0:
                    box_L = box_end[stack_ax] - box_start[stack_ax]
                    if distribution == "laminate":
                        n_layers = max(1, int(round(box_L / D_stack)))
                        tuned_L = n_layers * D_stack
                    else:
                        n_grains = dock.qty_spin.value() if dock.qty_mode_combo.currentIndex() == 0 else 1
                        t_grain = box_L / max(1, n_grains)
                        n_planes = max(1, int(round(t_grain / D_stack)))
                        tuned_L = n_planes * D_stack * max(1, n_grains)

                    if stack_ax == 0: dock.box_max_x.setValue(box_start[0] + tuned_L)
                    elif stack_ax == 1: dock.box_max_y.setValue(box_start[1] + tuned_L)
                    elif stack_ax == 2: dock.box_max_z.setValue(box_start[2] + tuned_L)

            if orientation_mode == "z_alignment":
                stack_ax = -1
                if distribution in ("laminate", "even"):
                    axis_str = dock.dist_lam_inplane_dist_combo.currentText() if distribution == "even" else "z"
                    stack_ax = {"x": 0, "y": 1, "z": 2}.get(axis_str, 2)

                if stack_ax != 2 and D_z > 0:
                    box_z = box_end[2] - box_start[2]
                    n_z = max(1, int(round(box_z / D_z)))
                    dock.box_max_z.setValue(box_start[2] + n_z * D_z)

                if in_plane == "symmetric_csl":
                    if stack_ax != 0 and D_x > 0:
                        box_x = box_end[0] - box_start[0]
                        n_x = max(1, int(round(box_x / D_x)))
                        dock.box_max_x.setValue(box_start[0] + n_x * D_x)
                    if stack_ax != 1 and D_y > 0:
                        box_y = box_end[1] - box_start[1]
                        n_y = max(1, int(round(box_y / D_y)))
                        dock.box_max_y.setValue(box_start[1] + n_y * D_y)

        finally:
            self._laminate_z_tune_lock = False

    def _update_csl_angle(self, *args) -> None:
        """Calculate and display all misorientation angles for the given Sigma and [hkl]."""
        dock = self.settings_dock
        sigma = dock.ori_csl_spin.value()
        hkl_text = dock.ori_hkl_edit.text().strip()

        dock.ori_csl_angle_combo.blockSignals(True)
        dock.ori_csl_angle_combo.clear()

        try:
            hkl = tuple(int(x) for x in hkl_text.split())
            if len(hkl) != 3:
                raise ValueError
        except (ValueError, AttributeError):
            dock.ori_csl_angle_combo.addItem("Inv HKL")
            dock.ori_csl_angle_combo.blockSignals(False)
            return

        from orientation import get_cubic_csl_angles
        angles = get_cubic_csl_angles(hkl, sigma)

        if angles:
            for ang in angles:
                angle_deg = np.degrees(ang)
                dock.ori_csl_angle_combo.addItem(f"{angle_deg:.2f}°")
        else:
            dock.ori_csl_angle_combo.addItem("N/A")

        dock.ori_csl_angle_combo.blockSignals(False)
        self._auto_tune_dimensions()

    def _clear_highlight(self) -> None:
        """Remove highlight cube and reset all grain opacities to 0.5."""
        plotter = self.central.voronoi_plotter
        if self._highlight_actor is not None:
            plotter.remove_actor(self._highlight_actor)
            self._highlight_actor = None

        if getattr(self, "_merged_voronoi", None) is not None:
            self._merged_voronoi.cell_data["RGBA"][:, 3] = 0.5
            self._merged_voronoi.Modified()
        if getattr(self, "_merged_seeds", None) is not None:
            self._merged_seeds.point_data["RGBA"][:, 3] = 0.5
            self._merged_seeds.Modified()
        self.central.voronoi_plotter.render()

    def _update_highlight(self, center: np.ndarray, grain_id: int) -> None:
        """Draw a wireframe cube around the selected seed + dim/brighten grains."""
        plotter = self.central.voronoi_plotter
        if self._highlight_actor is not None:
            plotter.remove_actor(self._highlight_actor)
        box_size = np.asarray(self._seed_result.box_end) - np.asarray(self._seed_result.box_start)
        edge = float(np.mean(box_size)) * 0.04
        cube = pv.Cube(
            center=center, x_length=edge, y_length=edge, z_length=edge,
        )
        self._highlight_actor = plotter.add_mesh(
            cube, color="red", style="wireframe", line_width=3,
            name="grain_highlight",
        )
        self._highlight_grain_3d(grain_id)

    def _highlight_grain_3d(self, grain_id: int) -> None:
        """Set opacity 0.9 for selected grain faces/seeds, 0.05 for all others."""
        if getattr(self, "_merged_voronoi", None) is not None:
            grain_ids = self._merged_voronoi.cell_data["GrainID"]
            rgba = self._merged_voronoi.cell_data["RGBA"]
            rgba[:, 3] = 0.05
            rgba[grain_ids == grain_id, 3] = 0.9
            self._merged_voronoi.Modified()

        if getattr(self, "_merged_seeds", None) is not None:
            seed_ids = self._merged_seeds.point_data["GrainID"]
            seed_rgba = self._merged_seeds.point_data["RGBA"]
            seed_rgba[:, 3] = 0.05
            seed_rgba[seed_ids == grain_id, 3] = 0.9
            self._merged_seeds.Modified()

        self.central.voronoi_plotter.render()

    def _update_clipping_plane(self, *args) -> None:
        """Dynamically slice the polycrystal on the GPU."""
        if getattr(self, "_seed_result", None) is None:
            return

        pct = self.central.slice_slider.value() / 100.0
        state = self.central._view_state

        bs = self._seed_result.box_start
        be = self._seed_result.box_end

        # -1 (Isometric) defaults to Z-axis slice
        axis = state if state in (0, 1, 2) else 2

        # Normal vector points toward the camera (the part being peeled away)
        normal = [0.0, 0.0, 0.0]
        normal[axis] = -1.0 if axis in (0, 2) else 1.0

        # Origin moves from the front face to the back face
        origin = [0.0, 0.0, 0.0]
        origin[axis] = be[axis] - (be[axis] - bs[axis]) * pct if axis in (0, 2) else bs[axis] + (be[axis] - bs[axis]) * pct

        self._clip_plane.SetNormal(*normal)
        self._clip_plane.SetOrigin(*origin)

        self.central.voronoi_plotter.render()

    def _on_apply_edit_clicked(self) -> None:
        """Update the selected grain, recompute Voronoi & misorientations, re-render."""
        if self._seed_result is None or self._orientation_result is None:
            return

        dock = self.settings_dock
        gid = dock.edit_grain_id.value()
        if gid < 0:
            return

        new_x = dock.edit_x.value()
        new_y = dock.edit_y.value()
        new_z = dock.edit_z.value()
        new_a = dock.edit_alpha.value()
        new_b = dock.edit_beta.value()
        new_g = dock.edit_gamma.value()

        old_seed = self._seed_result.seeds[gid]
        old_euler = self._orientation_result.euler_angles[gid]

        seed_changed = (
            new_x != old_seed[0] or new_y != old_seed[1] or new_z != old_seed[2]
        )
        ori_changed = (
            new_a != old_euler[0] or new_b != old_euler[1] or new_g != old_euler[2]
        )

        if not (seed_changed or ori_changed):
            return

        # --- update master data ---
        self._seed_result.seeds[gid, 0] = new_x
        self._seed_result.seeds[gid, 1] = new_y
        self._seed_result.seeds[gid, 2] = new_z

        self._orientation_result.euler_angles[gid, 0] = new_a
        self._orientation_result.euler_angles[gid, 1] = new_b
        self._orientation_result.euler_angles[gid, 2] = new_g

        # --- recompute Voronoi (only if seed moved) ---
        if seed_changed:
            from grain_seeds import GrainSeedGenerator

            gen = GrainSeedGenerator(
                box_start=tuple(float(x) for x in self._seed_result.box_start),
                box_end=tuple(float(x) for x in self._seed_result.box_end),
                distribution="customized",
                seed_positions=self._seed_result.seeds,
            )
            gen.generate_seeds()
            diameters, poly_data = gen.compute_grain_cells()
            self._seed_result.diameters = diameters
            self._seed_result.polyhedron_data = poly_data
        else:
            diameters = self._seed_result.diameters

        # --- recompute misorientation (only if orientation changed) ---
        if ori_changed:
            from orientation import OrientationAssigner
            from scipy.spatial.transform import Rotation

            rot = Rotation.from_euler(
                "zxz", self._orientation_result.euler_angles, degrees=True,
            )
            assigner = OrientationAssigner(
                mode="custom_profile",
                n_grains=self._seed_result.n_grains,
                neighbors=self._seed_result.neighbors,
            )
            misos = assigner._compute_pairwise_misorientation(rot)
            self._orientation_result.misorientation_angles = misos
            self._orientation_result.rotation_matrices = rot.as_matrix()
        else:
            misos = self._orientation_result.misorientation_angles

        # --- re-render affected viewports ---
        if seed_changed:
            self._render_voronoi_viewport(
                self._seed_result, self._orientation_result,
            )
        if seed_changed or ori_changed:
            size_target, miso_target = self._chart_target_params()
            self.central.update_charts(
                self._seed_result.diameters, misos,
                target_size_params=size_target,
                target_miso_angles=miso_target,
            )

        # re-populate editor fields and re-highlight
        self._on_grain_id_changed(gid)

        self.statusBar().showMessage(
            f"Grain {gid} updated.", 3000
        )

    # ------------------------------------------------------------------
    # Other placeholder slots
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        """Save lattice point coordinates as {Name}.crystal."""
        if self._crystal_atoms is None:
            self.statusBar().showMessage("No crystal to save.", 3000)
            return

        path = self._output_path(".crystal")
        if path is None:
            return

        name = self.central.out_name_edit.text().strip()

        atoms = self._crystal_atoms
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        cell = atoms.get_cell()  # 3×3 row-major
        header = (f"# POLY crystal: {name}  |  "
                  f"{len(atoms)} atoms  |  "
                  f"formula={atoms.get_chemical_formula()}\n"
                  f"# cell_1: {cell[0,0]:.8f} {cell[0,1]:.8f} {cell[0,2]:.8f}\n"
                  f"# cell_2: {cell[1,0]:.8f} {cell[1,1]:.8f} {cell[1,2]:.8f}\n"
                  f"# cell_3: {cell[2,0]:.8f} {cell[2,1]:.8f} {cell[2,2]:.8f}\n"
                  f"# element  x  y  z")
        lines = [header]
        for sym, (x, y, z) in zip(symbols, positions):
            lines.append(f"{sym:>3s}  {x:.8f}  {y:.8f}  {z:.8f}")
        with open(path, "w") as fh:
            fh.write("\n".join(lines))

        folder = self.central.out_folder_edit.text()
        self.statusBar().showMessage(
            f"Saved {name}.crystal to {folder}", 8000,
        )

    def _output_path(self, ext: str) -> str | None:
        """Build ``folder/name.ext``; returns None and shows message if folder
        or name are missing."""
        folder = self.central.out_folder_edit.text().strip()
        name = self.central.out_name_edit.text().strip()
        if not folder:
            self.statusBar().showMessage("Select an output folder first.", 4000)
            return None
        if not name:
            self.statusBar().showMessage("Enter a state name first.", 4000)
            return None
        return f"{folder}/{name}{ext}"

    def _on_build(self) -> None:
        """Extract parameters, ask for output path, start build worker."""
        if self._seed_result is None or self._orientation_result is None:
            self.statusBar().showMessage(
                "Generate seeds and crystal first.", 4000,
            )
            return

        data_path = self._output_path(".data")
        dump_path = self._output_path(".dump")
        if data_path is None or dump_path is None:
            folder = self.central.out_folder_edit.text().strip()
            name = self.central.out_name_edit.text().strip()
            missing = []
            if not folder:
                missing.append("Output folder")
            if not name:
                missing.append("State name")
            QMessageBox.warning(
                self, "Missing Output Configuration",
                f"Please set the following before building:\n\n"
                + "\n".join(f"  • {m}" for m in missing),
            )
            return

        # disable UI
        self.central.build_btn.setEnabled(False)
        self.statusBar().showMessage(
            "Building full polycrystal (this may take a while)…",
        )

        dock = self.settings_dock
        hkl_text = dock.ori_hkl_edit.text().strip()
        try:
            hkl = tuple(int(x) for x in hkl_text.split())
            if len(hkl) != 3:
                hkl = None
        except Exception:
            hkl = None

        distribution = self._seed_result.distribution

        self._build_worker = PolycrystalBuildWorker(
            seed_result=self._seed_result,
            orientation_result=self._orientation_result,
            crystal_source=self._crystal_source,
            crystal_params=self._crystal_params,
            data_path=data_path,
            dump_path=dump_path,
            hkl=hkl,
            distribution=distribution,
        )
        self._build_worker.finished.connect(self._on_build_finished)
        self._build_worker.error.connect(self._on_build_error)
        self._build_worker.start()

    def _on_build_finished(self, assembly_result) -> None:
        """Handle successful polycrystal assembly."""
        self.central.build_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"Successfully assembled {assembly_result.n_atoms:,} atoms "
            f"across {assembly_result.n_grains} grains!",
            10000,
        )
        print(
            f"[GUI] Build complete: {assembly_result.n_atoms} atoms, "
            f"{assembly_result.n_grains} grains"
        )

    def _on_build_error(self, err_msg: str) -> None:
        """Handle build failure, restoring UI."""
        self.central.build_btn.setEnabled(True)
        self.statusBar().showMessage(f"Build error: {err_msg}", 10000)
        print(f"[GUI] Build error: {err_msg}")

    def _on_save_seed_state(self) -> None:
        """Save current seed positions and Euler angles to .seed / .euler files."""
        if self._seed_result is None or self._orientation_result is None:
            self.statusBar().showMessage("No seed state to save.", 3000)
            return

        seed_path = self._output_path(".seed")
        euler_path = self._output_path(".euler")
        if seed_path is None or euler_path is None:
            return

        name = self.central.out_name_edit.text().strip()

        bs = self._seed_result.box_start
        be = self._seed_result.box_end
        # .seed  — N×3 seed positions (compatible with Customized distribution)
        # If Laguerre-Voronoi radii are available, append as column 4.
        sr = self._seed_result
        radii = getattr(sr, "target_radii", None)
        if radii is not None and len(radii) == len(sr.seeds):
            combined = np.column_stack([sr.seeds, radii])
            np.savetxt(seed_path, combined, fmt="%.8f",
                       header=f"POLY seed state: {name}  |  "
                              f"{sr.n_grains} grains  |  "
                              f"distribution={sr.distribution}\n"
                              f"# columns: x y z radius\n"
                              f"# box_start: {bs[0]:.8f} {bs[1]:.8f} {bs[2]:.8f}\n"
                              f"# box_end: {be[0]:.8f} {be[1]:.8f} {be[2]:.8f}")
        else:
            np.savetxt(seed_path, sr.seeds, fmt="%.8f",
                       header=f"POLY seed state: {name}  |  "
                              f"{sr.n_grains} grains  |  "
                              f"distribution={sr.distribution}\n"
                              f"# box_start: {bs[0]:.8f} {bs[1]:.8f} {bs[2]:.8f}\n"
                              f"# box_end: {be[0]:.8f} {be[1]:.8f} {be[2]:.8f}")

        # .euler — N×4 [grain_id, phi1, Phi, phi2] (compatible with Custom Profile)
        eulers = self._orientation_result.euler_angles
        ids = np.arange(eulers.shape[0], dtype=int).reshape(-1, 1)
        euler_with_id = np.hstack([ids, eulers])
        np.savetxt(euler_path, euler_with_id, fmt=["%d", "%.6f", "%.6f", "%.6f"],
                   header=f"POLY orientation state: {name}  |  "
                          f"mode={self._orientation_result.mode}  |  "
                          f"columns: grain_id phi1 Phi phi2 (zxz, degrees)")

        folder = self.central.out_folder_edit.text()
        self.statusBar().showMessage(
            f"Saved {name}.seed and {name}.euler to {folder}", 8000,
        )

    def closeEvent(self, event) -> None:
        """Cleanly shut down QtInteractors before closing."""
        self.central.voronoi_plotter.close()
        self.central.crystal_plotter.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("POLY")
    app.setOrganizationName("POLY-Project")

    # Programmatic splash pixmap — no external image file needed
    splash_pm = QPixmap(480, 260)
    splash_pm.fill(QColor("#1a1a2e"))
    painter = QPainter(splash_pm)
    painter.setPen(QColor("#e0e0e0"))
    title_font = QFont("Segoe UI", 28, QFont.Bold)
    painter.setFont(title_font)
    painter.drawText(splash_pm.rect().adjusted(0, 10, 0, 0),
                     Qt.AlignHCenter | Qt.AlignTop, "POLY")
    sub_font = QFont("Segoe UI", 12)
    painter.setFont(sub_font)
    painter.setPen(QColor("#90a4ae"))
    painter.drawText(splash_pm.rect().adjusted(0, 65, 0, 0),
                     Qt.AlignHCenter | Qt.AlignTop,
                     "Polycrystalline LAMMPS Generator")
    painter.setPen(QColor("#78909c"))
    author_font = QFont("Segoe UI", 11)
    painter.setFont(author_font)
    painter.drawText(splash_pm.rect().adjusted(0, 110, 0, 0),
                     Qt.AlignHCenter | Qt.AlignTop,
                     "Author: Yang Zhang (张杨)")
    painter.drawText(splash_pm.rect().adjusted(0, 132, 0, 0),
                     Qt.AlignHCenter | Qt.AlignTop,
                     "Stony Brook University")
    painter.setPen(QColor("#546e7a"))
    painter.drawText(splash_pm.rect().adjusted(0, 195, 0, 0),
                     Qt.AlignHCenter | Qt.AlignBottom,
                     "Starting…")
    painter.end()

    splash = QSplashScreen(splash_pm)
    splash.show()
    app.processEvents()

    # Lazy imports — the splash is already visible before these load
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()

    window = MainWindow()
    window.show()
    splash.finish(window)

    # Redirect stdout/stderr to the dock's terminal window
    _term_out = TerminalEmitter(window.settings_dock.terminal)
    _term_err = TerminalEmitter(window.settings_dock.terminal)
    sys.stdout = _term_out
    sys.stderr = _term_err

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
