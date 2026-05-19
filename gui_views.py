"""
POLY GUI view widgets — file picker, settings dock, central viewports.
"""

import numpy as np

from PySide6.QtCore import Qt, QObject, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg
import pyvista as pv
from pyvistaqt import QtInteractor


# ---------------------------------------------------------------------------
# Thread-safe terminal emitter
# ---------------------------------------------------------------------------

class TerminalEmitter(QObject):
    """Signal-based writer that appends text to a QTextEdit from any thread."""

    text_received = Signal(str)

    def __init__(self, terminal: QTextEdit, parent: QObject | None = None):
        super().__init__(parent)
        self._terminal = terminal
        self.text_received.connect(self._append)

    def write(self, text: str) -> None:
        self.text_received.emit(text)

    def flush(self) -> None:
        pass  # required for file-like interface

    def _append(self, text: str) -> None:
        cursor = self._terminal.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._terminal.setTextCursor(cursor)
        self._terminal.insertPlainText(text)
        # Auto-scroll to bottom
        scrollbar = self._terminal.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


# ---------------------------------------------------------------------------
# Reusable file-picker helper
# ---------------------------------------------------------------------------

class FilePickerWidget(QWidget):
    """Horizontal row: read-only path display + Browse button.

    Emits ``file_selected(path)`` when the user picks a file.
    """

    file_selected = Signal(str)

    def __init__(
        self,
        filter_spec: str = "All Files (*.*)",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._filter = filter_spec

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("No file selected…")
        layout.addWidget(self.path_edit, 1)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        layout.addWidget(browse_btn)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", self._filter,
        )
        if path:
            self.path_edit.setText(path)
            self.file_selected.emit(path)

    @property
    def path(self) -> str:
        return self.path_edit.text()


# ---------------------------------------------------------------------------
# Dynamic QStackedWidget – collapses to the size of the current page
# ---------------------------------------------------------------------------

class DynamicStackedWidget(QStackedWidget):
    """A QStackedWidget that collapses to the size of its current page."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.currentChanged.connect(self._update_sizes)

    def _update_sizes(self, index: int) -> None:
        for i in range(self.count()):
            w = self.widget(i)
            if i == index:
                w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            else:
                w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.adjustSize()

    def addWidget(self, w: QWidget) -> int:
        idx = super().addWidget(w)
        if idx == self.currentIndex():
            w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        else:
            w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        return idx


# ---------------------------------------------------------------------------
# Helper: labelled DynamicStackedWidget for conditional visibility
# ---------------------------------------------------------------------------

def _make_stacked_pair(visible_widget: QWidget) -> DynamicStackedWidget:
    """Wrap *visible_widget* in a DynamicStackedWidget whose page-0 is empty.

    Switching to index 1 shows the widget; index 0 reclaims the space.
    """
    stack = DynamicStackedWidget()
    empty = QWidget()
    empty.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    stack.addWidget(empty)               # index 0 — hidden
    stack.addWidget(visible_widget)       # index 1 — shown
    stack.setCurrentIndex(0)
    return stack


# ---------------------------------------------------------------------------
# Left Dock — Global Settings
# ---------------------------------------------------------------------------

class GlobalSettingsDock(QDockWidget):
    """Dockable panel holding all global simulation inputs."""

    edit_apply_clicked = Signal()
    phase_changed = Signal(int)
    add_crystal_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Global Settings (double-click to detach/attach)", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        # Outer wrapper — button pinned above scroll area
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(6)
        self.setWidget(outer)

        # ---- Generate + Reroll + Pause buttons (pinned at top) ----
        # NOTE: generate_btn and reroll_btn moved to CentralWidget ctrl bar.

        # Scroll area for all other settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll, 1)

        container = QWidget()
        scroll.setWidget(container)
        self._main_layout = QVBoxLayout(container)

        # ---- MC status bar (pinned at bottom) ----
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(6)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setFixedWidth(70)
        self.pause_btn.setStyleSheet(
            "QPushButton { background-color: #F9A825; color: white; font-weight: bold; "
            "border: none; border-radius: 3px; padding: 2px 8px; }"
            "QPushButton:hover { background-color: #F57F17; }"
            "QPushButton:checked { background-color: #388E3C; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        bottom_bar.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setFixedWidth(50)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #D32F2F; color: white; font-weight: bold; "
            "border: none; border-radius: 3px; padding: 2px 8px; }"
            "QPushButton:hover { background-color: #C62828; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        bottom_bar.addWidget(self.stop_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.setVisible(False)
        bottom_bar.addWidget(self.progress_bar)

        bottom_bar.addStretch()

        outer_layout.addLayout(bottom_bar)
        self._main_layout.setSpacing(8)

        # ---- Box Size ----
        self._build_box_group()

        # ---- Grain Quantity ----
        self._build_quantity_group()

        # ---- Grain Size Distribution ----
        self._build_distribution_group()

        # ---- Crystal Structure ----
        self._build_crystal_group()

        # ---- Orientation ----
        self._build_orientation_group()

        # ---- Selected Grain Editor ----
        self._build_grain_editor()

        # ---- Terminal Messages ----
        self._build_terminal_group()

        self._main_layout.addStretch()

        # Size dock to fit content so the vertical scrollbar is hidden
        # under the default settings (no horizontal scrollbar needed).
        container.updateGeometry()
        sb_w = scroll.verticalScrollBar().sizeHint().width()
        scroll.setMinimumWidth(container.sizeHint().width() + sb_w + 4)

    # ------------------------------------------------------------------
    # 1. Box Size
    # ------------------------------------------------------------------

    def _build_box_group(self) -> None:
        group = QGroupBox("Box Size")
        form = QFormLayout(group)

        self.box_min_x = QDoubleSpinBox()
        self.box_min_y = QDoubleSpinBox()
        self.box_min_z = QDoubleSpinBox()
        self.box_max_x = QDoubleSpinBox()
        self.box_max_y = QDoubleSpinBox()
        self.box_max_z = QDoubleSpinBox()

        for sb in (
            self.box_min_x, self.box_min_y, self.box_min_z,
            self.box_max_x, self.box_max_y, self.box_max_z,
        ):
            sb.setRange(-1e6, 1e6)
            sb.setDecimals(4)

        self.box_min_x.setValue(0.0)
        self.box_min_y.setValue(0.0)
        self.box_min_z.setValue(0.0)
        self.box_max_x.setValue(100.0)
        self.box_max_y.setValue(100.0)
        self.box_max_z.setValue(100.0)

        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("X:"))
        min_row.addWidget(self.box_min_x)
        min_row.addWidget(QLabel("Y:"))
        min_row.addWidget(self.box_min_y)
        min_row.addWidget(QLabel("Z:"))
        min_row.addWidget(self.box_min_z)
        form.addRow("Min (start):", min_row)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("X:"))
        max_row.addWidget(self.box_max_x)
        max_row.addWidget(QLabel("Y:"))
        max_row.addWidget(self.box_max_y)
        max_row.addWidget(QLabel("Z:"))
        max_row.addWidget(self.box_max_z)
        form.addRow("Max (end):", max_row)

        self._main_layout.addWidget(group)

    # ------------------------------------------------------------------
    # 2. Grain Quantity (toggle: count vs diameter)
    # ------------------------------------------------------------------

    def _build_quantity_group(self) -> None:
        group = QGroupBox("Grain Quantity")
        form = QFormLayout(group)

        self.qty_mode_combo = QComboBox()
        self.qty_mode_combo.addItems(
            ["Number of Grains", "Average Grain Diameter"]
        )
        form.addRow("Mode:", self.qty_mode_combo)

        self.qty_spin = QSpinBox()
        self.qty_spin.setRange(1, 100_000_000)
        self.qty_spin.setValue(10)

        self.qty_diam_spin = QDoubleSpinBox()
        self.qty_diam_spin.setRange(0.1, 1e7)
        self.qty_diam_spin.setDecimals(4)
        self.qty_diam_spin.setValue(57.5882)

        qty_row = QHBoxLayout()
        qty_row.addWidget(QLabel("Count:"))
        qty_row.addWidget(self.qty_spin)
        qty_row.addWidget(QLabel("Diameter (Å):"))
        qty_row.addWidget(self.qty_diam_spin)
        form.addRow(qty_row)

        # Default: count active, diameter read-only
        self.qty_diam_spin.setEnabled(False)

        self.qty_mode_combo.currentIndexChanged.connect(self._on_qty_mode)
        self.qty_spin.valueChanged.connect(self._on_qty_count_changed)
        self.qty_diam_spin.valueChanged.connect(self._on_qty_diam_changed)
        self._main_layout.addWidget(group)

    def _box_volume(self) -> float:
        box_x = self.box_max_x.value() - self.box_min_x.value()
        box_y = self.box_max_y.value() - self.box_min_y.value()
        box_z = self.box_max_z.value() - self.box_min_z.value()
        return box_x * box_y * box_z

    def _on_qty_mode(self, idx: int) -> None:
        """Toggle active field; inactive field becomes read-only."""
        if idx == 0:  # Number of Grains
            self.qty_spin.setEnabled(True)
            self.qty_diam_spin.setEnabled(False)
            self._update_diameter_from_count()
        else:         # Average Grain Diameter
            self.qty_spin.setEnabled(False)
            self.qty_diam_spin.setEnabled(True)
            self._update_count_from_diameter()

    def _update_diameter_from_count(self) -> None:
        """Set diameter spinbox from current count and box volume."""
        V = self._box_volume()
        n = max(1, self.qty_spin.value())
        d = 2.0 * (3.0 * V / (4.0 * np.pi * n)) ** (1.0 / 3.0)
        self.qty_diam_spin.blockSignals(True)
        self.qty_diam_spin.setValue(round(d, 4))
        self.qty_diam_spin.blockSignals(False)

    def _update_count_from_diameter(self) -> None:
        """Set count spinbox from current diameter and box volume."""
        V = self._box_volume()
        d = max(0.001, self.qty_diam_spin.value())
        n = int(np.round(6.0 * V / (np.pi * d**3)))
        n = max(1, min(100_000_000, n))
        self.qty_spin.blockSignals(True)
        self.qty_spin.setValue(n)
        self.qty_spin.blockSignals(False)

    def _on_qty_count_changed(self, _value: int) -> None:
        """Auto-update diameter when count changes and count mode is active."""
        if self.qty_mode_combo.currentIndex() == 0:
            self._update_diameter_from_count()

    def _on_qty_diam_changed(self, _value: float) -> None:
        """Auto-update count when diameter changes and diameter mode is active."""
        if self.qty_mode_combo.currentIndex() == 1:
            self._update_count_from_diameter()

    # ------------------------------------------------------------------
    # 3. Grain Size Distribution
    # ------------------------------------------------------------------

    def _build_distribution_group(self) -> None:
        group = QGroupBox("Grain Size Distribution")
        form = QFormLayout(group)

        self.dist_combo = QComboBox()
        self.dist_combo.addItems(
            ["Random", "Normal Distribution", "Customized", "Laminate",
             "Evenly Spaced (1D)", "Bimodal"]
        )
        form.addRow("Distribution:", self.dist_combo)

        # Normal / laminate+normal → StdDev
        dist_normal = QWidget()
        nl = QHBoxLayout(dist_normal)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.addWidget(QLabel("StdDev:"))
        self.dist_stddev = QDoubleSpinBox()
        self.dist_stddev.setRange(0.01, 1e7)
        self.dist_stddev.setDecimals(4)
        self.dist_stddev.setValue(20.0)
        nl.addWidget(self.dist_stddev)
        self.dist_normal_stack = _make_stacked_pair(dist_normal)
        form.addRow(self.dist_normal_stack)

        # Customized → file picker
        self.dist_custom_picker = FilePickerWidget()
        self.dist_custom_stack = _make_stacked_pair(self.dist_custom_picker)
        form.addRow(self.dist_custom_stack)

        # Laminate → in-plane distribution
        lam_widget = QWidget()
        lam_layout = QFormLayout(lam_widget)
        lam_layout.setContentsMargins(0, 0, 0, 0)
        lam_layout.setSpacing(4)
        self.dist_lam_inplane_dist_combo = QComboBox()
        self.dist_lam_inplane_dist_combo.addItems(["random", "normal", "x", "y", "z"])
        lam_layout.addRow("In-plane Distribution:", self.dist_lam_inplane_dist_combo)
        self.dist_laminate_stack = _make_stacked_pair(lam_widget)
        form.addRow(self.dist_laminate_stack)

        # Bimodal → GMM params
        bim_widget = QWidget()
        bim_layout = QFormLayout(bim_widget)
        bim_layout.setContentsMargins(0, 0, 0, 0)
        bim_layout.setSpacing(3)
        # Row 1: fraction in mode 1
        frac_row = QHBoxLayout()
        frac_row.addWidget(QLabel("Fraction in mode 1:"))
        self.dist_bim_frac = QDoubleSpinBox()
        self.dist_bim_frac.setRange(0.01, 0.99)
        self.dist_bim_frac.setDecimals(2)
        self.dist_bim_frac.setSingleStep(0.05)
        self.dist_bim_frac.setValue(0.50)
        frac_row.addWidget(self.dist_bim_frac)
        bim_layout.addRow(frac_row)
        # Row 2: mode 1 mean + std
        m1_row = QHBoxLayout()
        m1_row.addWidget(QLabel("Mode 1 mean (Angstrom):"))
        self.dist_bim_m1 = QDoubleSpinBox()
        self.dist_bim_m1.setRange(0.1, 1e7)
        self.dist_bim_m1.setDecimals(2)
        self.dist_bim_m1.setValue(50.0)
        m1_row.addWidget(self.dist_bim_m1)
        m1_row.addWidget(QLabel("Std:"))
        self.dist_bim_s1 = QDoubleSpinBox()
        self.dist_bim_s1.setRange(0.01, 1e7)
        self.dist_bim_s1.setDecimals(2)
        self.dist_bim_s1.setValue(5.0)
        m1_row.addWidget(self.dist_bim_s1)
        bim_layout.addRow(m1_row)
        # Row 3: mode 2 mean + std
        m2_row = QHBoxLayout()
        m2_row.addWidget(QLabel("Mode 2 mean (Angstrom):"))
        self.dist_bim_m2 = QDoubleSpinBox()
        self.dist_bim_m2.setRange(0.1, 1e7)
        self.dist_bim_m2.setDecimals(2)
        self.dist_bim_m2.setValue(150.0)
        m2_row.addWidget(self.dist_bim_m2)
        m2_row.addWidget(QLabel("Std:"))
        self.dist_bim_s2 = QDoubleSpinBox()
        self.dist_bim_s2.setRange(0.01, 1e7)
        self.dist_bim_s2.setDecimals(2)
        self.dist_bim_s2.setValue(15.0)
        m2_row.addWidget(self.dist_bim_s2)
        bim_layout.addRow(m2_row)
        self.dist_bimodal_stack = _make_stacked_pair(bim_widget)
        form.addRow(self.dist_bimodal_stack)

        self.dist_combo.currentIndexChanged.connect(self._on_dist_mode)
        self._main_layout.addWidget(group)

    def _on_dist_mode(self, idx: int) -> None:
        self.dist_normal_stack.setCurrentIndex(1 if idx in (1, 3) else 0)
        self.dist_custom_stack.setCurrentIndex(1 if idx == 2 else 0)
        self.dist_laminate_stack.setCurrentIndex(1 if idx in (3, 4) else 0)
        self.dist_bimodal_stack.setCurrentIndex(1 if idx == 5 else 0)
        # Re-enable quantity inputs (may have been disabled by bimodal)
        if idx != 5:
            self.qty_spin.setReadOnly(False)
            self.qty_spin.setEnabled(True)
            self.qty_diam_spin.setReadOnly(False)
            self.qty_diam_spin.setEnabled(
                self.qty_mode_combo.currentIndex() == 1
            )

        if idx == 3:
            self.ori_combo.setCurrentIndex(1)  # laminate pairs with Z-axis alignment
            self.dist_stddev.setValue(20.0)
            # Restore laminate items
            self.dist_lam_inplane_dist_combo.clear()
            self.dist_lam_inplane_dist_combo.addItems(["random", "normal"])
        elif idx == 4:
            # Evenly Spaced (1D): repurpose combo for axis selection
            self.dist_lam_inplane_dist_combo.clear()
            self.dist_lam_inplane_dist_combo.addItems(["x", "y", "z"])
            self.ori_combo.setCurrentIndex(1)  # pairs with Z-axis alignment
            self.dist_stddev.setValue(0.0)
        elif idx == 5:
            self.dist_stddev.setValue(0.0)
            # Bimodal: disable quantity inputs (read-only), count is auto-calculated
            self.qty_spin.setReadOnly(True)
            self.qty_spin.setEnabled(False)
            self.qty_diam_spin.setReadOnly(True)
            self.qty_diam_spin.setEnabled(False)
        else:
            self.dist_stddev.setValue(20.0)

    def _on_phase_assign_mode_changed(self, idx: int) -> None:
        """Switch between fraction spinbox (idx=0) and grain list edit (idx=1)."""
        self.phase_fraction_spin.setVisible(idx == 0)
        self.phase_grain_list_edit.setVisible(idx == 1)

    # ------------------------------------------------------------------
    # 4. Crystal Structure
    # ------------------------------------------------------------------

    def _build_crystal_group(self) -> None:
        group = QGroupBox("Crystal Structure")
        outer = QVBoxLayout(group)

        # ---- Phase Index + ADD CRYSTAL header row ----
        phase_row = QHBoxLayout()
        phase_row.addWidget(QLabel("Phase:"))
        self.phase_index_spin = QSpinBox()
        self.phase_index_spin.setRange(0, 0)
        self.phase_index_spin.setValue(0)
        self.phase_index_spin.setPrefix("Phase ")
        self.phase_index_spin.valueChanged.connect(self.phase_changed.emit)
        phase_row.addWidget(self.phase_index_spin)
        phase_row.addStretch()
        self.add_crystal_btn = QPushButton("ADD CRYSTAL")
        self.add_crystal_btn.setStyleSheet(
            "QPushButton { background-color: #2E7D32; color: white; font-weight: bold; "
            "border: none; border-radius: 3px; padding: 3px 10px; }"
            "QPushButton:hover { background-color: #388E3C; }"
        )
        self.add_crystal_btn.clicked.connect(self.add_crystal_clicked.emit)
        phase_row.addWidget(self.add_crystal_btn)
        outer.addLayout(phase_row)

        # ---- Phase assignment row (hidden for phase 0) ----
        self.phase_assign_stack = DynamicStackedWidget()
        assign_hidden = QWidget()
        assign_widget = QWidget()
        assign_layout = QHBoxLayout(assign_widget)
        assign_layout.setContentsMargins(0, 0, 0, 0)
        assign_layout.setSpacing(4)
        assign_layout.addWidget(QLabel("Assign:"))
        self.phase_assign_mode = QComboBox()
        self.phase_assign_mode.addItems(["Fraction", "Grain List"])
        self.phase_assign_mode.currentIndexChanged.connect(
            self._on_phase_assign_mode_changed
        )
        assign_layout.addWidget(self.phase_assign_mode)
        self.phase_fraction_spin = QDoubleSpinBox()
        self.phase_fraction_spin.setRange(0.0, 1.0)
        self.phase_fraction_spin.setDecimals(2)
        self.phase_fraction_spin.setSingleStep(0.05)
        self.phase_fraction_spin.setValue(0.0)
        assign_layout.addWidget(self.phase_fraction_spin)
        self.phase_grain_list_edit = QLineEdit()
        self.phase_grain_list_edit.setPlaceholderText("e.g. 1,3,5-8, 2n, d<20")
        self.phase_grain_list_edit.setToolTip(
            "Grain ID formulas:\n"
            "  1,3,5-8  — explicit IDs and ranges\n"
            "  2n       — all even-index grains (0,2,4,…)\n"
            "  2n+1     — all odd-index grains (1,3,5,…)\n"
            "  d < 20   — grains with diameter < 20\n"
            "  d > 50   — grains with diameter > 50\n"
            "Combine with commas: 2n, 5, d<30"
        )
        self.phase_grain_list_edit.setVisible(False)
        assign_layout.addWidget(self.phase_grain_list_edit)
        assign_layout.addStretch()
        self.phase_assign_stack.addWidget(assign_hidden)  # index 0 = hidden
        self.phase_assign_stack.addWidget(assign_widget)  # index 1 = visible
        self.phase_assign_stack.setCurrentIndex(0)
        outer.addWidget(self.phase_assign_stack)

        # ---- Existing crystal form ----
        form = QFormLayout()
        outer.addLayout(form)

        self.crystal_combo = QComboBox()
        self.crystal_combo.addItems(
            ["Bravais", "Intermetallics", "Spacegroup", "Custom (File)"]
        )
        form.addRow("Source:", self.crystal_combo)

        # --- Shared Bravais / Intermetallics container ---
        shared_widget = QWidget()
        shared_layout = QVBoxLayout(shared_widget)
        shared_layout.setContentsMargins(0, 0, 0, 0)
        shared_layout.setSpacing(4)

        row1 = QHBoxLayout()
        self.crystal_type_label = QLabel("Structure:")
        row1.addWidget(self.crystal_type_label)
        self.crystal_type_combo = QComboBox()
        self.crystal_type_combo.setEditable(True)
        row1.addWidget(self.crystal_type_combo)
        self.crystal_elem_label = QLabel("Element:")
        row1.addWidget(self.crystal_elem_label)
        self.crystal_elem_edit = QLineEdit("Mg")
        row1.addWidget(self.crystal_elem_edit)
        shared_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("a (Å):"))
        self.crystal_shared_a = QDoubleSpinBox()
        self.crystal_shared_a.setRange(0.01, 1e4)
        self.crystal_shared_a.setDecimals(6)
        self.crystal_shared_a.setValue(3)
        row2.addWidget(self.crystal_shared_a)
        row2.addWidget(QLabel("c (Å):"))
        self.crystal_shared_c = QDoubleSpinBox()
        self.crystal_shared_c.setRange(0.01, 1e4)
        self.crystal_shared_c.setDecimals(6)
        self.crystal_shared_c.setValue(4.9)
        row2.addWidget(self.crystal_shared_c)
        shared_layout.addLayout(row2)

        self.crystal_shared_a.valueChanged.connect(
            lambda val: self.crystal_shared_c.setValue(val * 1.63299)
        )

        # --- Spacegroup sub-panel ---
        sg_widget = QWidget()
        sg_layout = QVBoxLayout(sg_widget)
        sg_layout.setContentsMargins(0, 0, 0, 0)
        sg_layout.setSpacing(4)

        sg_elem_row = QHBoxLayout()
        sg_elem_row.addWidget(QLabel("Elements:"))
        self.crystal_sg_elements = QLineEdit("Na,Cl")
        sg_elem_row.addWidget(self.crystal_sg_elements)
        sg_layout.addLayout(sg_elem_row)

        sg_basis_row = QHBoxLayout()
        sg_basis_row.addWidget(QLabel("Basis [(x1,y1,z1),...]:"))
        self.crystal_sg_basis = QLineEdit("(0,0,0),(0.5,0.5,0.5)")
        sg_basis_row.addWidget(self.crystal_sg_basis)
        sg_layout.addLayout(sg_basis_row)

        sg_group_row = QHBoxLayout()
        sg_group_row.addWidget(QLabel("Spacegroup:"))
        self.crystal_sg_group = QLineEdit("225")
        sg_group_row.addWidget(self.crystal_sg_group)
        sg_layout.addLayout(sg_group_row)

        sg_cell_row = QHBoxLayout()
        sg_cell_row.addWidget(QLabel("Cell (a,b,c,α,β,γ):"))
        self.crystal_sg_cellpar = QLineEdit("5.64,5.64,5.64,90,90,90")
        sg_cell_row.addWidget(self.crystal_sg_cellpar)
        sg_layout.addLayout(sg_cell_row)

        # --- Custom file sub-panel ---
        self.crystal_custom_picker = FilePickerWidget()

        # --- 3-page stack: 0=shared, 1=spacegroup, 2=custom file ---
        self.crystal_page_stack = DynamicStackedWidget()
        self.crystal_page_stack.addWidget(shared_widget)           # page 0
        self.crystal_page_stack.addWidget(sg_widget)               # page 1
        self.crystal_page_stack.addWidget(self.crystal_custom_picker)  # page 2
        form.addRow(self.crystal_page_stack)

        # Default: Bravais
        self.crystal_page_stack.setCurrentIndex(0)
        self._populate_crystal_type_combo(0)

        self.crystal_combo.currentIndexChanged.connect(self._on_crystal_mode)
        self._main_layout.addWidget(group)

    def _populate_crystal_type_combo(self, idx: int) -> None:
        """Fill the shared type combo for Bravais (idx==0) or
        Intermetallics (idx==1), relabel, and set a sensible default."""
        self.crystal_type_combo.clear()
        if idx == 0:  # Bravais
            self.crystal_type_label.setText("Structure:")
            self.crystal_elem_label.setText("Element:")
            self.crystal_type_combo.addItems([
                "sc", "fcc", "bcc", "hcp", "diamond", "bct",
            ])
            self.crystal_elem_edit.setText("Mg")
            self.crystal_shared_a.setValue(3.0)
            self.crystal_shared_c.setValue(4.9)
        else:  # Intermetallics
            self.crystal_type_label.setText("Prototype:")
            self.crystal_elem_label.setText("Elements:")
            self.crystal_type_combo.addItems([
                "L1_2", "B2", "D0_3", "L2_1",
                "D0_19", "A15", "C15", "L1_0",
                "rocksalt", "zincblende",
                "cesiumchloride", "fluorite", "wurtzite",
            ])
            self.crystal_elem_edit.setText("Ni,Al")
            self.crystal_shared_a.setValue(3.57)
            self.crystal_shared_c.setValue(3.57)

    def _on_crystal_mode(self, idx: int) -> None:
        if idx in (0, 1):  # Bravais or Intermetallics → shared page
            self.crystal_page_stack.setCurrentIndex(0)
            self._populate_crystal_type_combo(idx)
        elif idx == 2:     # Spacegroup
            self.crystal_page_stack.setCurrentIndex(1)
        else:              # Custom (File)
            self.crystal_page_stack.setCurrentIndex(2)

    # ------------------------------------------------------------------
    # 5. Orientation
    # ------------------------------------------------------------------

    def _build_orientation_group(self) -> None:
        group = QGroupBox("Orientation")
        form = QFormLayout(group)

        self.ori_combo = QComboBox()
        self.ori_combo.addItems([
            "Random",
            "Z-axis alignment",
            "Low angle",
            "High angle",
            "Custom Misorientation",
            "Custom Profile",
        ])
        form.addRow("Mode:", self.ori_combo)

        # Z-axis alignment → (hkl) line edit + in-plane combo
        z_widget = QWidget()
        z_row = QHBoxLayout(z_widget)
        z_row.setContentsMargins(0, 0, 0, 0)
        z_row.setSpacing(4)
        z_row.addWidget(QLabel("(hkl):"))
        self.ori_hkl_edit = QLineEdit("1 1 0")
        z_row.addWidget(self.ori_hkl_edit)
        z_row.addWidget(QLabel("In-plane:"))
        self.ori_in_plane_combo = QComboBox()
        self.ori_in_plane_combo.addItems(["random", "low_angle", "high_angle", "symmetric_csl"])
        z_row.addWidget(self.ori_in_plane_combo)
        z_row.addWidget(QLabel(" Σ:"))
        self.ori_csl_spin = QSpinBox()
        self.ori_csl_spin.setRange(3, 99)
        self.ori_csl_spin.setValue(5)
        self.ori_csl_spin.setSingleStep(2)
        self.ori_csl_spin.setVisible(False)
        z_row.addWidget(self.ori_csl_spin)
        z_row.addWidget(QLabel("  Angle:"))
        self.ori_csl_angle_combo = QComboBox()
        self.ori_csl_angle_combo.setMaximumWidth(80)
        z_row.addWidget(self.ori_csl_angle_combo)
        self.ori_in_plane_combo.currentTextChanged.connect(
            lambda t: self.ori_csl_spin.setVisible(t == "symmetric_csl")
        )
        self.ori_z_stack = _make_stacked_pair(z_widget)
        form.addRow(self.ori_z_stack)

        # Custom Misorientation / Custom Profile → file picker
        self.ori_custom_picker = FilePickerWidget()
        self.ori_custom_stack = _make_stacked_pair(self.ori_custom_picker)
        form.addRow(self.ori_custom_stack)

        self.ori_combo.currentIndexChanged.connect(self._on_ori_mode)
        self._main_layout.addWidget(group)

    def _on_ori_mode(self, idx: int) -> None:
        self.ori_z_stack.setCurrentIndex(1 if idx == 1 else 0)
        # Custom Misorientation (4) or Custom Profile (5)
        self.ori_custom_stack.setCurrentIndex(1 if idx in (4, 5) else 0)

    # ------------------------------------------------------------------
    # 6. Selected Grain Editor
    # ------------------------------------------------------------------

    def _build_grain_editor(self) -> None:
        group = QGroupBox("Selected Grain Editor")
        form = QFormLayout(group)

        id_diam_row = QHBoxLayout()
        self.edit_grain_id = QSpinBox()
        self.edit_grain_id.setRange(-1, -1)
        self.edit_grain_id.setSpecialValueText("--")
        self.edit_grain_id.setEnabled(False)
        id_diam_row.addWidget(self.edit_grain_id)
        id_diam_row.addWidget(QLabel("  Diameter:"))
        self.edit_grain_diam = QDoubleSpinBox()
        self.edit_grain_diam.setDecimals(4)
        self.edit_grain_diam.setRange(0.0, 1e7)
        self.edit_grain_diam.setReadOnly(True)
        self.edit_grain_diam.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.edit_grain_diam.setStyleSheet(
            "QDoubleSpinBox { background-color: #E0E0E0; }"
        )
        id_diam_row.addWidget(self.edit_grain_diam)
        id_diam_row.addWidget(QLabel("  LV Radius:"))
        self.edit_grain_radius = QDoubleSpinBox()
        self.edit_grain_radius.setDecimals(4)
        self.edit_grain_radius.setRange(0.0, 1e7)
        self.edit_grain_radius.setToolTip("Laguerre-Voronoi power-distance radius")
        id_diam_row.addWidget(self.edit_grain_radius)
        form.addRow("Grain ID:", id_diam_row)

        xyz_row = QHBoxLayout()
        self.edit_x = QDoubleSpinBox()
        self.edit_x.setRange(-1e6, 1e6)
        self.edit_x.setDecimals(4)
        xyz_row.addWidget(QLabel("X:"))
        xyz_row.addWidget(self.edit_x)
        self.edit_y = QDoubleSpinBox()
        self.edit_y.setRange(-1e6, 1e6)
        self.edit_y.setDecimals(4)
        xyz_row.addWidget(QLabel("Y:"))
        xyz_row.addWidget(self.edit_y)
        self.edit_z = QDoubleSpinBox()
        self.edit_z.setRange(-1e6, 1e6)
        self.edit_z.setDecimals(4)
        xyz_row.addWidget(QLabel("Z:"))
        xyz_row.addWidget(self.edit_z)
        form.addRow("Position:", xyz_row)

        euler_row = QHBoxLayout()
        self.edit_alpha = QDoubleSpinBox()
        self.edit_alpha.setRange(-180.0, 180.0)
        self.edit_alpha.setDecimals(2)
        euler_row.addWidget(QLabel("φ₁:"))
        euler_row.addWidget(self.edit_alpha)
        self.edit_beta = QDoubleSpinBox()
        self.edit_beta.setRange(0.0, 180.0)
        self.edit_beta.setDecimals(2)
        euler_row.addWidget(QLabel("Φ:"))
        euler_row.addWidget(self.edit_beta)
        self.edit_gamma = QDoubleSpinBox()
        self.edit_gamma.setRange(-180.0, 180.0)
        self.edit_gamma.setDecimals(2)
        euler_row.addWidget(QLabel("φ₂:"))
        euler_row.addWidget(self.edit_gamma)
        form.addRow("Euler (zxz):", euler_row)

        self.edit_apply_btn = QPushButton("Apply Edit")
        self.edit_apply_btn.setEnabled(False)
        self.edit_apply_btn.setStyleSheet(
            "QPushButton { background-color: #BDBDBD; color: #757575; "
            "font-weight: bold; border: none; border-radius: 3px; "
            "padding: 4px 12px; }"
        )
        self.edit_apply_btn.clicked.connect(self.edit_apply_clicked.emit)
        form.addRow(self.edit_apply_btn)

        self._edit_editor_group = group
        self._main_layout.addWidget(group)

    # ------------------------------------------------------------------
    # 7. Terminal Messages
    # ------------------------------------------------------------------

    def _build_terminal_group(self) -> None:
        group = QGroupBox("Terminal Messages")
        layout = QVBoxLayout(group)

        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setLineWrapMode(QTextEdit.NoWrap)
        self.terminal.setFont(QFont("Consolas", 9))
        self.terminal.setFixedHeight(120)
        self.terminal.setPlaceholderText("Terminal output...")
        self.terminal.setStyleSheet(
            "QTextEdit { background-color: #F5F5F5; color: #000000; "
            "border: 1px solid #BDBDBD; border-radius: 3px; }"
        )
        layout.addWidget(self.terminal)

        self._main_layout.addWidget(group)


# ---------------------------------------------------------------------------
# Central Widget — viewports
# ---------------------------------------------------------------------------

class CentralWidget(QWidget):
    """Holds three viewports inside QSplitters + a control bar."""

    save_clicked = Signal()
    save_state_clicked = Signal()
    build_clicked = Signal()
    generate_clicked = Signal()
    reroll_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # --- Horizontal splitter: Voronoi (left) || right side ---
        h_split = QSplitter(Qt.Horizontal)

        # Left: Voronoi 3D viewport + overlay controls
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        cam_ctrl = QHBoxLayout()
        self.ortho_btn = QPushButton("View: Isometric")
        self.ortho_btn.setStyleSheet("font-weight: bold; padding: 4px 8px; background-color: #E3F2FD;")
        self._view_state = -1  # -1=Iso, 0=X, 1=Y, 2=Z
        self._box_center = None
        self.ortho_btn.clicked.connect(self._on_ortho_clicked)
        cam_ctrl.addWidget(self.ortho_btn)

        cam_ctrl.addWidget(QLabel("  Slice Depth:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 100)
        self.slice_slider.setValue(0)
        self.slice_slider.setFixedWidth(400)
        cam_ctrl.addWidget(self.slice_slider)

        self.slice_box = QSpinBox()
        self.slice_box.setRange(0, 100)
        self.slice_box.setValue(0)
        self.slice_box.setButtonSymbols(QAbstractSpinBox.NoButtons)
        cam_ctrl.addWidget(self.slice_box)
        cam_ctrl.addStretch()

        self.slice_slider.valueChanged.connect(self.slice_box.setValue)
        self.slice_box.valueChanged.connect(self.slice_slider.setValue)

        left_layout.addLayout(cam_ctrl)

        persp_ctrl = QHBoxLayout()
        persp_ctrl.addWidget(QLabel("Perspective:"))
        self.persp_slider = QSlider(Qt.Horizontal)
        self.persp_slider.setRange(1, 100)
        self.persp_slider.setValue(30)
        self.persp_slider.setFixedWidth(400)
        persp_ctrl.addWidget(self.persp_slider)

        self.persp_box = QDoubleSpinBox()
        self.persp_box.setRange(1, 100)
        self.persp_box.setDecimals(1)
        self.persp_box.setValue(30.0)
        self.persp_box.setButtonSymbols(QAbstractSpinBox.NoButtons)
        persp_ctrl.addWidget(self.persp_box)
        persp_ctrl.addStretch()

        self.persp_slider.valueChanged.connect(lambda v: self.persp_box.setValue(v))
        self.persp_box.valueChanged.connect(lambda v: self.persp_slider.setValue(int(v)))
        self.persp_box.valueChanged.connect(self._on_persp_changed)

        left_layout.addLayout(persp_ctrl)

        self.voronoi_plotter = QtInteractor(self)
        self.voronoi_progress = QProgressBar(self.voronoi_plotter)
        self.voronoi_progress.setMaximumWidth(300)
        self.voronoi_progress.setVisible(False)
        self.voronoi_progress.setTextVisible(False)
        self.voronoi_progress.setStyleSheet(
            "QProgressBar { background-color: white; border: 1px solid #AAA; }"
        )
        left_layout.addWidget(self.voronoi_plotter, 1)

        h_split.addWidget(left_container)

        # Right: vertical splitter (crystal || charts)
        v_split = QSplitter(Qt.Vertical)

        # Top-right: Pristine Crystal 3D viewport
        self.crystal_plotter = QtInteractor(self)
        v_split.addWidget(self.crystal_plotter)

        # Bottom-right: two side-by-side pyqtgraph PlotWidgets
        charts_container = QWidget()
        charts_layout = QHBoxLayout(charts_container)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(4)

        self.grain_size_plot = pg.PlotWidget(title="Grain Size Distribution")
        self.grain_size_plot.setLabel("bottom", "Diameter (Å)")
        self.grain_size_plot.setLabel("left", "Count")
        charts_layout.addWidget(self.grain_size_plot)

        self.misorientation_plot = pg.PlotWidget(
            title="Misorientation Distribution"
        )
        self.misorientation_plot.setLabel("bottom", "Angle (deg)")
        self.misorientation_plot.setLabel("left", "Count")
        charts_layout.addWidget(self.misorientation_plot)

        v_split.addWidget(charts_container)

        h_split.addWidget(v_split)
        h_split.setSizes([600, 400])
        v_split.setSizes([250, 250])

        root.addWidget(h_split, 1)

        # --- Output path row ---
        out_row = QHBoxLayout()
        out_row.setSpacing(4)
        out_row.addWidget(QLabel("Folder:"))
        self.out_folder_edit = QLineEdit()
        self.out_folder_edit.setReadOnly(True)
        self.out_folder_edit.setPlaceholderText("Select output folder…")
        out_row.addWidget(self.out_folder_edit, 1)
        browse_folder_btn = QPushButton("Browse…")
        browse_folder_btn.clicked.connect(self._browse_output_folder)
        out_row.addWidget(browse_folder_btn)
        out_row.addSpacing(16)
        out_row.addWidget(QLabel("Name:"))
        self.out_name_edit = QLineEdit("polycrystal")
        self.out_name_edit.setMaximumWidth(180)
        out_row.addWidget(self.out_name_edit)
        root.addLayout(out_row)

        # --- Control panel ---
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        self.save_btn = QPushButton("Save Pristine Crystal")
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(
            "QPushButton { background-color: #388E3C; color: white; font-weight: bold; "
            "border: none; border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #2E7D32; }"
            "QPushButton:pressed { background-color: #1B5E20; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        self.build_btn = QPushButton("Proceed with Build")
        self.build_btn.setStyleSheet(
            "QPushButton { background-color: #D32F2F; color: white; font-weight: bold; "
            "border: none; border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #C62828; }"
            "QPushButton:pressed { background-color: #B71C1C; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )

        ctrl.addWidget(self.save_btn)

        self.save_state_btn = QPushButton("Save Seed State")
        self.save_state_btn.setStyleSheet(
            "QPushButton { background-color: #388E3C; color: white; font-weight: bold; "
            "border: none; border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #2E7D32; }"
            "QPushButton:pressed { background-color: #1B5E20; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        self.save_state_btn.setEnabled(False)
        ctrl.addWidget(self.save_state_btn)

        ctrl.addStretch()

        self.reroll_btn = QPushButton("Reroll")
        self.reroll_btn.setMinimumHeight(32)
        self.reroll_btn.setStyleSheet(
            "QPushButton { background-color: #F57C00; color: white; font-weight: bold; "
            "border: none; border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #E65100; }"
            "QPushButton:pressed { background-color: #BF360C; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        ctrl.addWidget(self.reroll_btn)

        self.generate_btn = QPushButton("Generate Initial Seeds")
        self.generate_btn.setMinimumHeight(32)
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #1976D2; color: white; font-weight: bold; "
            "border: none; border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #1565C0; }"
            "QPushButton:pressed { background-color: #0D47A1; }"
            "QPushButton:disabled { background-color: #B0BEC5; }"
        )
        ctrl.addWidget(self.generate_btn)

        ctrl.addWidget(self.build_btn)

        root.addLayout(ctrl)

        # Wire signals
        self.save_btn.clicked.connect(self.save_clicked.emit)
        self.save_state_btn.clicked.connect(self.save_state_clicked.emit)
        self.build_btn.clicked.connect(self.build_clicked.emit)
        self.generate_btn.clicked.connect(self.generate_clicked.emit)
        self.reroll_btn.clicked.connect(self.reroll_clicked.emit)

        # Defer initial renders so the UI paints its empty frames first
        QTimer.singleShot(0, self._init_renders)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def _update_charts(self) -> None:
        """Generate dummy histogram data and plot via BarGraphItem."""
        rng = np.random.default_rng()

        sizes = rng.normal(30, 8, size=500)
        sizes = sizes[(sizes > 5) & (sizes < 60)]
        counts, bins = np.histogram(sizes, bins=15)
        self.grain_size_plot.clear()
        self.grain_size_plot.addItem(
            pg.BarGraphItem(
                x0=bins[:-1], x1=bins[1:], height=counts,
                brush=(52, 152, 219, 180),
            )
        )
        self.grain_size_plot.setLabel("bottom", "Equivalent Diameter (Å)")
        self.grain_size_plot.setLabel("left", "Count")

        miso = rng.normal(20, 10, size=800)
        miso = miso[(miso > 0) & (miso < 62)]
        m_counts, m_bins = np.histogram(miso, bins=15)
        self.misorientation_plot.clear()
        self.misorientation_plot.addItem(
            pg.BarGraphItem(
                x0=m_bins[:-1], x1=m_bins[1:], height=m_counts,
                brush=(231, 76, 60, 180),
            )
        )
        self.misorientation_plot.setLabel("bottom", "Misorientation Angle (deg)")
        self.misorientation_plot.setLabel("left", "Count")

    def update_charts(
        self,
        diameters: np.ndarray,
        misorientation_angles: np.ndarray | None,
        target_size_params: dict | None = None,
        target_miso_angles: np.ndarray | None = None,
    ) -> None:
        """Plot **real** grain-size and misorientation histograms."""
        from scipy.stats import norm as scipy_norm

        self.grain_size_plot.clear()

        # --- Grain Size Distribution ---
        if len(diameters) < 1:
            self.grain_size_plot.setTitle("Grain Size Distribution (N/A)")
            size_counts, size_bins = np.array([]), np.array([0, 1])
        elif len(diameters) == 1 or np.ptp(diameters) < 1e-5:
            self.grain_size_plot.setTitle("Grain Size Distribution")
            val = diameters[0]
            size_counts = np.array([len(diameters)])
            size_bins = np.array([val - 0.5, val + 0.5])
        else:
            self.grain_size_plot.setTitle("Grain Size Distribution")
            n_bins = max(2, min(15, len(diameters)))
            try:
                size_counts, size_bins = np.histogram(diameters, bins=n_bins)
            except ValueError:
                self.grain_size_plot.setTitle("Grain Size Distribution (N/A)")
                size_counts, size_bins = np.array([]), np.array([0, 1])

        if len(size_counts) > 0:
            self.grain_size_plot.addItem(
                pg.BarGraphItem(
                    x0=size_bins[:-1], x1=size_bins[1:], height=size_counts,
                    brush=(52, 152, 219, 180),
                )
            )
            self.grain_size_plot.setLabel("bottom", "Equivalent Diameter (Å)")
            self.grain_size_plot.setLabel("left", "Count")

            # --- Grain Size target curve ---
            if target_size_params is not None and len(diameters) > 1 and np.ptp(diameters) >= 1e-5:
                x_vals = np.linspace(size_bins[0], size_bins[-1], 200)
                _type = target_size_params.get("type", "normal")
                if _type == "bimodal":
                    frac = target_size_params["frac"]
                    m1, s1 = target_size_params["m1"], target_size_params["s1"]
                    m2, s2 = target_size_params["m2"], target_size_params["s2"]
                    pdf_vals = (frac * scipy_norm.pdf(x_vals, loc=m1, scale=s1)
                                + (1 - frac) * scipy_norm.pdf(x_vals, loc=m2, scale=s2))
                else:
                    mean = target_size_params["mean"]
                    std = target_size_params["std"]
                    pdf_vals = scipy_norm.pdf(x_vals, loc=mean, scale=std)
                y_vals = pdf_vals * len(diameters) * (size_bins[1] - size_bins[0])
                self.grain_size_plot.plot(
                    x_vals, y_vals, pen=pg.mkPen(color="y", width=3),
                )

        # --- Misorientation Distribution ---
        self.misorientation_plot.clear()

        if misorientation_angles is None or len(misorientation_angles) == 0:
            self.misorientation_plot.setTitle("Misorientation Distribution (N/A for current mode)")
            self.misorientation_plot.setLabel("bottom", "")
            self.misorientation_plot.setLabel("left", "")
        elif len(misorientation_angles) == 1 or np.ptp(misorientation_angles) < 1e-5:
            self.misorientation_plot.setTitle("Misorientation Distribution")
            val = misorientation_angles[0]
            m_counts = np.array([len(misorientation_angles)])
            m_bins = np.array([val - 0.5, val + 0.5])

            self.misorientation_plot.addItem(
                pg.BarGraphItem(
                    x0=m_bins[:-1], x1=m_bins[1:], height=m_counts,
                    brush=(231, 76, 60, 180),
                )
            )
            self.misorientation_plot.setLabel("bottom", "Misorientation Angle (deg)")
            self.misorientation_plot.setLabel("left", "Count")
        else:
            self.misorientation_plot.setTitle("Misorientation Distribution")
            m_n_bins = max(2, min(15, len(misorientation_angles)))
            try:
                m_counts, m_bins = np.histogram(misorientation_angles, bins=m_n_bins)
            except ValueError:
                self.misorientation_plot.setTitle("Misorientation Distribution (N/A)")
                self.misorientation_plot.setLabel("bottom", "")
                self.misorientation_plot.setLabel("left", "")
                return

            self.misorientation_plot.addItem(
                pg.BarGraphItem(
                    x0=m_bins[:-1], x1=m_bins[1:], height=m_counts,
                    brush=(231, 76, 60, 180),
                )
            )
            self.misorientation_plot.setLabel("bottom", "Misorientation Angle (deg)")
            self.misorientation_plot.setLabel("left", "Count")

            # --- Misorientation target curve (custom_misorientation) ---
            if target_miso_angles is not None and len(target_miso_angles) > 0:
                t_counts, _ = np.histogram(target_miso_angles, bins=m_bins)
                t_counts = t_counts.astype(float)
                max_m_count = np.max(m_counts)
                t_counts *= max_m_count / (np.max(t_counts) + 1e-12)
                bin_centers = 0.5 * (m_bins[:-1] + m_bins[1:])
                self.misorientation_plot.plot(
                    bin_centers, t_counts,
                    pen=pg.mkPen(color="y", width=3),
                )

    # ------------------------------------------------------------------
    # Dummy renders
    # ------------------------------------------------------------------

    def render_dummy_voronoi(self) -> None:
        """Five random spheres with a bounding box and axes."""
        plotter = self.voronoi_plotter
        plotter.clear()

        rng = np.random.default_rng(42)
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

        for i in range(5):
            center = rng.uniform(0, 100, size=3)
            sphere = pv.Sphere(radius=rng.uniform(5, 15), center=center)
            plotter.add_mesh(
                sphere, color=colors[i], opacity=0.7,
                name=f"grain_{i}",
            )

        # Bounding box
        plotter.add_mesh(
            pv.Box(bounds=(0, 100, 0, 100, 0, 100)),
            color="white", style="wireframe", line_width=1,
            name="bbox",
        )
        # Axes centered on box, 120 % of each dimension
        box_center = np.array([50.0, 50.0, 50.0])
        half = np.array([60.0, 60.0, 60.0])
        colors_rgb = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        for axis in range(3):
            start = box_center.copy()
            end = box_center.copy()
            start[axis] -= half[axis]
            end[axis] += half[axis]
            plotter.add_mesh(
                pv.Line(start, end), color=colors_rgb[axis], line_width=2,
                name=f"axis_{'xyz'[axis]}",
            )
        plotter.camera.focal_point = box_center
        plotter.view_isometric()

    def render_dummy_crystal(self) -> None:
        """Simple cubic point cloud."""
        plotter = self.crystal_plotter
        plotter.clear()

        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        z = np.linspace(0, 10, 11)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        cloud = pv.PolyData(points)
        plotter.add_mesh(
            cloud, color="#f39c12", point_size=8,
            render_points_as_spheres=True, name="crystal",
        )
        plotter.show_grid(
            xtitle="X", ytitle="Y", ztitle="Z",
        )
        plotter.view_isometric()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _init_renders(self) -> None:
        """Deferred initial renders — called by single-shot timer so the
        UI paints its frames before OpenGL content blocks the event loop."""
        self.render_dummy_voronoi()
        self.render_dummy_crystal()
        self._update_charts()

    def _on_ortho_clicked(self) -> None:
        self._view_state = (self._view_state + 1) % 3
        views = ["Left (X)", "Front (Y)", "Top (Z)"]
        self.ortho_btn.setText(f"View: {views[self._view_state]} [Ortho]")

        self.voronoi_plotter.enable_parallel_projection()
        self._apply_camera_view()

    def _on_persp_changed(self, val: float) -> None:
        if val == 0.0:
            self.voronoi_plotter.enable_parallel_projection()
        else:
            self.voronoi_plotter.disable_parallel_projection()
            self.voronoi_plotter.camera.view_angle = max(1.0, val)
        self.voronoi_plotter.render()

    def _apply_camera_view(self) -> None:
        if self._view_state == 0:
            self.voronoi_plotter.view_yz()
        elif self._view_state == 1:
            self.voronoi_plotter.view_xz()
        elif self._view_state == 2:
            self.voronoi_plotter.view_xy()
        if self._box_center is not None:
            self.voronoi_plotter.camera.focal_point = self._box_center
        self.voronoi_plotter.reset_camera()
        self.voronoi_plotter.render()

    def _browse_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder",
        )
        if folder:
            self.out_folder_edit.setText(folder)
