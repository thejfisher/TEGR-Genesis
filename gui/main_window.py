"""TEGR Genesis main window (step 4: drag-and-drop setup grid + engine).

Layout
  Top bar   : 1 SETUP › 2 COMPUTE › 3 ANALYZE  ·  RUN  ·  hardware picker
  Left      : Toolbox (things to drag onto the grid — drag-and-drop in step 4)
  Center    : tabs  2D SETUP | 3D VIEW | TELEMETRY
  Right     : Properties (selected object) + Simulation settings
  Bottom    : status bar + progress bar
"""

from PyQt6.QtCore import Qt, QTimer, QSize, QMimeData
from PyQt6.QtGui import QAction, QPixmap, QPainter, QColor, QIcon, QPen, QBrush, QActionGroup
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QTabWidget, QDockWidget,
    QListWidget, QListWidgetItem, QGroupBox, QFormLayout, QComboBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QToolBar, QSizePolicy, QProgressBar, QButtonGroup,
    QMessageBox, QScrollArea, QCheckBox, QFileDialog, QApplication,
)
import os
import subprocess
import sys
from pathlib import Path

from core.entities import Entity
from gui.canvas_2d import Canvas2D, MIME
from gui.canvas_3d import Canvas3D
from gui.dashboard import Dashboard
from gui.entity_editor import EntityEditor
from gui.hardware_selector import HardwareSelector

ACCENT = "#00A2FF"

# name, kind, colour, shape, tooltip
TOOLBOX = [
    ("FIELD", None, None, None, None),
    ("Gravity Well", "well", "#00A2FF", "ring", "A fixed mass that bends the field"),
    ("Wave Defect", "defect", "#7FE0FF", "star", "A kink / pulse in the field"),
    ("PARTICLES", None, None, None, None),
    ("Static Anchor", "anchor", "#C8C8C8", "square", "Feeds the field, never moves"),
    ("Active Body", "active", "#FFB000", "dot", "Moves and feeds the field"),
    ("Tracer", "tracer", "#8A8A8A", "small", "Moves but does not feed the field"),
]

PHYSICS_RULES = [
    ("MS38 energy-conserving (default)", "ms38"),
    ("Legacy TEGR 2600", "legacy2600"),
]


class ToolboxList(QListWidget):
    """Toolbox list whose items can be dragged onto the 2D grid."""

    def mimeTypes(self):
        return [MIME]

    def mimeData(self, items):
        md = QMimeData()
        kind = items[0].data(Qt.ItemDataRole.UserRole) if items else None
        if kind:
            md.setData(MIME, kind.encode())
        return md


def _icon(colour, shape, size=28):
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(colour)
    m = size // 2
    if shape == "ring":
        p.setPen(QPen(c, 3)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(4, 4, size - 8, size - 8)
        p.setBrush(QBrush(c)); p.drawEllipse(m - 3, m - 3, 6, 6)
    elif shape == "star":
        p.setPen(QPen(c, 2))
        for dx, dy in ((0, 10), (10, 0), (7, 7), (7, -7)):
            p.drawLine(m - dx, m - dy, m + dx, m + dy)
    elif shape == "square":
        p.setPen(QPen(QColor("#FFFFFF"), 1)); p.setBrush(QBrush(c))
        p.drawRect(7, 7, size - 14, size - 14)
    elif shape == "dot":
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(c))
        p.drawEllipse(6, 6, size - 12, size - 12)
    else:  # small
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(c))
        p.drawEllipse(m - 4, m - 4, 8, 8)
    p.end()
    return QIcon(pm)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TEGR Genesis")
        self.resize(1500, 920)
        self.setDockNestingEnabled(True)
        self._hardware_checked = False
        self.entities = []      # things placed on the grid
        self.result = None      # last RunResult
        self.job = None         # running ComputeJob
        self.experiment_path = None   # where the experiment was last saved / opened
        self.experiment_name = "Untitled experiment"
        self.experiment_notes = ""

        self._build_central()
        self._build_toolbox()
        self._build_properties()
        self._build_topbar()
        self._build_menus()
        self._build_statusbar()

        self._on_rule_changed()
        self._set_stage(0)

    # ------------------------------------------------------------------ layout
    def _build_central(self):
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.canvas2d = Canvas2D()
        self.canvas3d = Canvas3D()
        self.dashboard = Dashboard()
        self.canvas3d.time_changed.connect(self.dashboard.set_cursor)
        
        from gui.pysindy_tab import PySindyTab
        self.pysindy_tab = PySindyTab()

        self.tabs.addTab(self.canvas2d, "2D SETUP")
        self.tabs.addTab(self.canvas3d, "3D VIEW")
        self.tabs.addTab(self.dashboard, "TELEMETRY")
        self.tabs.addTab(self.pysindy_tab, "PYSINDY")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

    def _build_toolbox(self):
        dock = QDockWidget("TOOLBOX", self)
        dock.setObjectName("toolboxDock")
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.toolbox = ToolboxList()
        self.toolbox.setObjectName("toolboxList")
        self.toolbox.setIconSize(QSize(28, 28))
        self.toolbox.setSpacing(2)
        self.toolbox.setDragEnabled(True)
        self.toolbox.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        for name, kind, colour, shape, tip in TOOLBOX:
            item = QListWidgetItem(name)
            if kind is None:  # section header
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                item.setForeground(QColor(ACCENT))
                f = item.font(); f.setBold(True); f.setPointSize(9); item.setFont(f)
            else:
                item.setIcon(_icon(colour, shape))
                item.setData(Qt.ItemDataRole.UserRole, kind)
                item.setToolTip(tip + "\nDrag onto the grid, or double-click to add.")
            self.toolbox.addItem(item)
        self.toolbox.itemDoubleClicked.connect(self._toolbox_double_clicked)
        hint = QLabel("Drag items onto the 2D grid\n(or double-click them).\n\n"
                      "Click to select · drag to move\n"
                      "Drag the arrow tip to set speed\n"
                      "Right-click for more · Delete key removes")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        box = QWidget(); lay = QVBoxLayout(box); lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.toolbox); lay.addWidget(hint)
        dock.setWidget(box)
        dock.setMinimumWidth(200)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.toolbox_dock = dock

    def _build_properties(self):
        dock = QDockWidget("PROPERTIES", self)
        dock.setObjectName("propertiesDock")
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        panel = QWidget(); lay = QVBoxLayout(panel); lay.setContentsMargins(6, 6, 6, 6)

        # Selected object
        sel = QGroupBox("Selected object")
        sel_lay = QVBoxLayout(sel)
        self.editor = EntityEditor()
        sel_lay.addWidget(self.editor)
        lay.addWidget(sel)

        # Simulation settings
        sim = QGroupBox("Simulation")
        form = QFormLayout(sim)
        self.rule_combo = QComboBox(); self.rule_combo.setObjectName("physicsRule")
        for text, key in PHYSICS_RULES:
            self.rule_combo.addItem(text, key)
        self.rule_combo.setToolTip("MS38: energy-conserving wave rule, sponge edges, no bulk drain.\n"
                                   "Legacy: old TEGR 2600 rule with decay, to match past results.")
        self.rule_combo.currentIndexChanged.connect(self._on_rule_changed)
        form.addRow("Physics rule", self.rule_combo)

        self.grid_spin = self._ispin("gridSize", 16, 256, 64, 16, "Cells per side of the 3D box")
        self.dx_spin = self._dspin("cellSize", 0.1, 50.0, 2.8125, 4, "Size of one cell")
        self.dt_spin = self._dspin("timeStep", 1e-5, 0.1, 0.005, 5, "Time per tick")
        self.ticks_spin = self._ispin("ticks", 10, 10_000_000, 16000, 1000, "How many ticks to compute")
        self.c_spin = self._dspin("waveSpeed", 0.1, 1000.0, 65.0, 2, "Wave speed where φ = 0 (c_base)")
        self.alpha_spin = self._dspin("alpha", 0.0, 100.0, 0.0, 3, "How strongly φ slows waves: c² = c_base²/(1+α|φ|)²\n0 = plain gravity. Raise it to get horizons.")
        self.sponge_spin = self._ispin("spongeWidth", 0, 64, 12, 1, "Soft edge that soaks up waves (cells)")
        self.source_spin = self._dspin("sourceStrength", 0.0, 1e6, 1.0, 4,
                                       "G: a mass m settles to a field φ = G·m/r")
        self.force_spin = self._dspin("fieldForce", 0.0, 1e6, 100.0, 3,
                                      "K: push on particles. MS38: F = K·m·∇φ  ·  Legacy: F = K·∇φ\n"
                                      "Newton's G in this model = K × G")
        self.warmup_spin = self._ispin("warmupTicks", 0, 10_000_000, 6000, 500,
                                       "Moving bodies are held still while the field settles")
        self.ramp_spin = self._ispin("rampTicks", 0, 10_000_000, 3000, 500,
                                     "Sources switch on smoothly over this many ticks")
        self.pauli_spin = self._dspin("pauliStrength", 0.0, 1e6, 0.0, 3,
                                      "Particle-particle push (TEGR 2600 Pauli). 0 = off")
        self.record_spin = self._ispin("recordEvery", 1, 100_000, 20, 10, "Save positions every N ticks")
        self.local_source_check = QCheckBox("Source uses local wave speed")
        self.local_source_check.setObjectName("sourceLocalSpeed")
        self.local_source_check.setChecked(False)
        self.local_source_check.setToolTip(
            "Off = original MS38 rule.\n"
            "On = each source is scaled by the local c/c_base, so horizon cores settle\n"
            "by themselves instead of piling up to the c2_min guard.\n"
            "Only matters when α > 0. Preserves 100% of gravitating mass at infinity.")
        form.addRow("Grid size", self.grid_spin)
        form.addRow("Cell size dx", self.dx_spin)
        form.addRow("Time step dt", self.dt_spin)
        form.addRow("Ticks", self.ticks_spin)
        form.addRow("Wave speed c", self.c_spin)
        form.addRow("Coupling α", self.alpha_spin)
        form.addRow("Sponge edge", self.sponge_spin)
        form.addRow("Source strength G", self.source_spin)
        form.addRow("Field force K", self.force_spin)
        form.addRow("Warm-up ticks", self.warmup_spin)
        form.addRow("Source ramp", self.ramp_spin)
        form.addRow("Pauli push", self.pauli_spin)
        form.addRow("Record every", self.record_spin)
        form.addRow(self.local_source_check)
        lay.addWidget(sim)

        # Legacy-only settings
        self.legacy_box = QGroupBox("Legacy TEGR 2600 only")
        lform = QFormLayout(self.legacy_box)
        self.decay_spin = self._dspin("decayBase", 0.5, 1.0, 0.999, 4, "Drain per tick (1.0 = none)")
        self.decay_gamma_spin = self._dspin("decayGamma", 0.0, 5.0, 0.5, 3, "How drain follows c²")
        self.legacy_source_spin = self._dspin("legacySource", 0.0, 1e6, 0.0, 5,
                                              "Raw TEGR 2600 source S per tick (S·m/m_max)")
        self.legacy_seed_check = QCheckBox("Seed bumps at start"); self.legacy_seed_check.setChecked(True)
        self.legacy_seed_check.setObjectName("legacySeed")
        self.legacy_periodic_check = QCheckBox("Wrap-around box"); self.legacy_periodic_check.setChecked(True)
        self.legacy_periodic_check.setObjectName("legacyPeriodic")
        lform.addRow("Decay", self.decay_spin)
        lform.addRow("Decay γ", self.decay_gamma_spin)
        lform.addRow("Source S", self.legacy_source_spin)
        lform.addRow(self.legacy_seed_check)
        lform.addRow(self.legacy_periodic_check)
        lay.addWidget(self.legacy_box)
        lay.addStretch(1)

        self.grid_spin.valueChanged.connect(self._on_box_changed)
        self.dx_spin.valueChanged.connect(self._on_box_changed)
        self.sponge_spin.valueChanged.connect(self._on_box_changed)
        self.force_spin.valueChanged.connect(self._sync_canvas)
        self.source_spin.valueChanged.connect(self._sync_canvas)

        # editor <-> grid
        self.canvas2d.set_entities(self.entities)
        self.canvas2d.selected.connect(self._on_selected)
        self.canvas2d.changed.connect(self._on_entities_changed)
        self.editor.changed.connect(self._on_editor_changed)
        self.editor.delete_requested.connect(self.canvas2d.delete_selected)
        self.editor.aim_requested.connect(self._aim)
        self.editor.orbit_requested.connect(self._orbit)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel)
        dock.setWidget(scroll)
        dock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.properties_dock = dock

    def _build_topbar(self):
        bar = QToolBar("Workflow")
        bar.setObjectName("workflowBar")
        bar.setMovable(False)
        bar.setFloatable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)

        logo = QLabel("TEGR GENESIS")
        logo.setObjectName("barTitle")
        bar.addWidget(logo)

        self.stage_group = QButtonGroup(self)
        self.stage_group.setExclusive(True)
        self.stage_buttons = []
        for i, text in enumerate(["1  SETUP", "2  COMPUTE", "3  ANALYZE"]):
            if i:
                arrow = QLabel("›"); arrow.setObjectName("stageArrow"); bar.addWidget(arrow)
            b = QPushButton(text)
            b.setObjectName(f"stage{i}")
            b.setProperty("stage", True)
            b.setCheckable(True)
            self.stage_group.addButton(b, i)
            self.stage_buttons.append(b)
            bar.addWidget(b)
        self.stage_group.idClicked.connect(self._set_stage)

        spacer = QWidget(); spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)

        self.run_button = QPushButton("▶  RUN")
        self.run_button.setObjectName("runButton")
        self.run_button.setToolTip("Compute the experiment (click again to stop)")
        self.run_button.clicked.connect(self._toggle_run)
        bar.addWidget(self.run_button)

        self.hardware = HardwareSelector()
        self.hardware.device_changed.connect(self._on_device_changed)
        hw_wrap = QWidget(); hl = QHBoxLayout(hw_wrap); hl.setContentsMargins(16, 0, 8, 0)
        hl.addWidget(self.hardware)
        bar.addWidget(hw_wrap)

    def _build_menus(self):
        bar = self.menuBar()
        m = bar.addMenu("&File")
        self.act_new = m.addAction("New Experiment"); self.act_new.setShortcut("Ctrl+N")
        self.act_open = m.addAction("Open Experiment (.toml)…"); self.act_open.setShortcut("Ctrl+O")
        self.act_save = m.addAction("Save Experiment"); self.act_save.setShortcut("Ctrl+S")
        self.act_save_as = m.addAction("Save Experiment As…"); self.act_save_as.setShortcut("Ctrl+Shift+S")
        self.act_new.triggered.connect(self.new_experiment)
        self.act_open.triggered.connect(self.open_experiment)
        self.act_save.triggered.connect(lambda: self.save_experiment(ask=False))
        self.act_save_as.triggered.connect(lambda: self.save_experiment(ask=True))
        m.addSeparator()
        m.addAction("Load demo: orbits around a well").triggered.connect(self.load_demo)
        m.addSeparator()
        quit_action = m.addAction("Quit"); quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        m = bar.addMenu("&Edit")
        a = m.addAction("Delete selected  (Delete key on the grid)")
        a.triggered.connect(self.canvas2d.delete_selected)
        a = m.addAction("Duplicate selected"); a.setShortcut("Ctrl+D")
        a.triggered.connect(self.canvas2d.duplicate_selected)
        m.addSeparator()
        a = m.addAction("Aim selected at nearest body"); a.triggered.connect(self._aim)
        a = m.addAction("Circular orbit for selected"); a.triggered.connect(self._orbit)
        m.addSeparator()
        a = m.addAction("Clear the grid…"); a.triggered.connect(self.new_experiment)

        m = bar.addMenu("&Run")
        a = m.addAction("Compute / Stop"); a.setShortcut("F5"); a.triggered.connect(self._toggle_run)

        m = bar.addMenu("&Export")
        for text, what in (("Graphs for papers (PNG + PDF, white background)…", "graphs"),
                           ("Raw data for PySINDy (CSV + NPY)…", "raw"),
                           ("PySINDy results…", "sindy")):
            a = m.addAction(text); a.triggered.connect(lambda _=False, w=what: self.export(w))
        m.addSeparator()
        a = m.addAction("Export everything…"); a.setShortcut("Ctrl+E")
        a.triggered.connect(lambda: self.export("all"))
        m.addAction("Open the exports folder").triggered.connect(lambda: self._open_folder(self._exports_dir()))

        m = bar.addMenu("&View")
        m.addAction(self.toolbox_dock.toggleViewAction())
        m.addAction(self.properties_dock.toggleViewAction())
        m.addSeparator()
        group = QActionGroup(self)
        for i, name in enumerate(["2D Setup", "3D View", "Telemetry", "PySINDy"]):
            a = QAction(name, self); a.setShortcut(f"Ctrl+{i + 1}")
            a.triggered.connect(lambda _=False, idx=i: self.tabs.setCurrentIndex(idx))
            group.addAction(a); m.addAction(a)

        m = bar.addMenu("&Help")
        m.addAction("Physics rules…").triggered.connect(self._show_physics)
        m.addAction("About TEGR Genesis").triggered.connect(self._show_about)

    def _build_statusbar(self):
        sb = self.statusBar()
        self.progress = QProgressBar()
        self.progress.setObjectName("computeProgress")
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        sb.addPermanentWidget(self.progress)
        sb.showMessage("Ready")

    # ------------------------------------------------------------- helpers
    def _ispin(self, name, lo, hi, val, step, tip):
        s = QSpinBox(); s.setObjectName(name); s.setRange(lo, hi); s.setSingleStep(step)
        s.setValue(val); s.setToolTip(tip); s.setGroupSeparatorShown(True)
        return s

    def _dspin(self, name, lo, hi, val, decimals, tip):
        s = QDoubleSpinBox(); s.setObjectName(name); s.setDecimals(decimals)
        s.setRange(lo, hi); s.setValue(val); s.setToolTip(tip)
        s.setSingleStep(10 ** -min(decimals, 3))
        return s

    def settings(self):
        """Current simulation settings as a plain dict (names match core.engine.SimSettings)."""
        return {
            "physics_rule": self.rule_combo.currentData(),
            "grid_resolution": self.grid_spin.value(),
            "dx": self.dx_spin.value(),
            "dt": self.dt_spin.value(),
            "total_ticks": self.ticks_spin.value(),
            "c_base": self.c_spin.value(),
            "alpha": self.alpha_spin.value(),
            "sponge_width": self.sponge_spin.value(),
            "source_strength": self.source_spin.value(),
            "field_force": self.force_spin.value(),
            "warmup_ticks": self.warmup_spin.value(),
            "source_ramp_ticks": self.ramp_spin.value(),
            "pauli_strength": self.pauli_spin.value(),
            "record_every": self.record_spin.value(),
            "source_local_speed": self.local_source_check.isChecked(),
            "decay_base": self.decay_spin.value(),
            "decay_gamma": self.decay_gamma_spin.value(),
            "legacy_source": self.legacy_source_spin.value(),
            "legacy_seed": self.legacy_seed_check.isChecked(),
            "legacy_periodic": self.legacy_periodic_check.isChecked(),
            "hardware": self.hardware.choice_key(),
        }

    def apply_settings(self, d):
        """Fill the panel from a settings dict (used by the demo and, later, TOML load)."""
        widgets = {
            "grid_resolution": self.grid_spin, "dx": self.dx_spin, "dt": self.dt_spin,
            "total_ticks": self.ticks_spin, "c_base": self.c_spin, "alpha": self.alpha_spin,
            "sponge_width": self.sponge_spin, "source_strength": self.source_spin,
            "field_force": self.force_spin, "warmup_ticks": self.warmup_spin,
            "source_ramp_ticks": self.ramp_spin, "pauli_strength": self.pauli_spin,
            "record_every": self.record_spin, "decay_base": self.decay_spin,
            "decay_gamma": self.decay_gamma_spin, "legacy_source": self.legacy_source_spin,
        }
        for k, w in widgets.items():
            if k in d:
                w.setValue(type(w.value())(d[k]))
        if "legacy_seed" in d:
            self.legacy_seed_check.setChecked(bool(d["legacy_seed"]))
        if "legacy_periodic" in d:
            self.legacy_periodic_check.setChecked(bool(d["legacy_periodic"]))
        self.local_source_check.setChecked(bool(d.get("source_local_speed", False)))
        if "physics_rule" in d:
            i = self.rule_combo.findData(d["physics_rule"])
            if i >= 0:
                self.rule_combo.setCurrentIndex(i)
        if "hardware" in d:
            self.hardware.set_choice(d["hardware"])

    # ------------------------------------------------------------- compute
    def load_demo(self):
        """Two tracers around one gravity well (same settings as the engine's orbit check)."""
        from core.engine import SimSettings, circular_speed
        d = dict(physics_rule="ms38", grid_resolution=64, dx=2.8125, dt=0.005, total_ticks=18000,
                 c_base=65.0, alpha=0.0, sponge_width=12, source_strength=1.0, field_force=100.0,
                 warmup_ticks=10000, source_ramp_ticks=4000, pauli_strength=0.0, record_every=20)
        self.apply_settings(d)
        s = SimSettings.from_dict(d)
        M = 30.0
        v1 = circular_speed(s, M, 30.0)
        v2 = circular_speed(s, M, 45.0)
        self.entities[:] = [
            Entity("well", 0, 0, 0, mass=M, label="Central well"),
            Entity("tracer", 30.0, 0, 0, vy=v1, label="Circular orbit r=30"),
            Entity("tracer", -45.0, 0, 0, vy=-0.85 * v2, label="Elliptical orbit r=45"),
        ]
        self.result = None
        self.canvas2d.clear_paths()
        self.canvas2d.set_entities(self.entities)
        self.canvas2d.select(None)
        self.canvas3d.show_setup(self.entities)
        self.dashboard.clear()
        self.pysindy_tab.set_result(None)
        self._set_stage(0)
        self.statusBar().showMessage("Demo loaded: one gravity well and two tracers. Press RUN.", 8000)

    def new_experiment(self):
        if self.job is not None:
            return
        if self.entities:
            ans = QMessageBox.question(self, "Clear the grid?",
                                       "Remove everything from the grid and start again?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        self.entities.clear()
        self.result = None
        self.experiment_path = None
        self.experiment_name = "Untitled experiment"
        self.experiment_notes = ""
        self._update_title()
        self.canvas2d.clear_paths()
        self.canvas2d.set_entities(self.entities)
        self.canvas2d.select(None)
        self.canvas3d.show_setup(self.entities)
        self.dashboard.clear()
        self.pysindy_tab.set_result(None)
        self._set_stage(0)
        self.statusBar().showMessage("New experiment: drag items onto the grid.", 6000)

    # ------------------------------------------------------------- save / load (TOML)
    def _experiments_dir(self):
        from core.state_manager import EXPERIMENTS_DIR
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        return EXPERIMENTS_DIR

    def _update_title(self):
        self.setWindowTitle(f"TEGR Genesis — {self.experiment_name}")

    def save_experiment(self, ask=False):
        if self.job is not None:
            return False
        from core.state_manager import save_experiment
        path = self.experiment_path
        if ask or path is None:
            start = str(path or (self._experiments_dir() / f"{self.experiment_name.replace(' ', '_')}.toml"))
            fn, _ = QFileDialog.getSaveFileName(self, "Save experiment", start, "TEGR experiment (*.toml)")
            if not fn:
                return False
            path = Path(fn)
            self.experiment_name = path.stem.replace("_", " ")
        try:
            path = save_experiment(path, self.settings(), self.entities, self.experiment_name, self.experiment_notes)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save", str(exc))
            return False
        self.experiment_path = path
        self._update_title()
        self.statusBar().showMessage(f"Saved: {path}", 8000)
        return True

    def open_experiment(self, path=None):
        if self.job is not None:
            return False
        from core.state_manager import load_experiment
        if not path:
            fn, _ = QFileDialog.getOpenFileName(self, "Open experiment", str(self._experiments_dir()),
                                                "TEGR experiment (*.toml)")
            if not fn:
                return False
            path = fn
        try:
            settings, entities, meta = load_experiment(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open", str(exc))
            return False
        self.apply_settings(settings)
        self.entities[:] = entities
        self.result = None
        self.experiment_path = Path(path)
        self.experiment_name = str(meta.get("name") or Path(path).stem)
        self.experiment_notes = str(meta.get("notes", ""))
        self._update_title()
        self.canvas2d.clear_paths()
        self.canvas2d.set_entities(self.entities)
        self.canvas2d.select(None)
        self.canvas3d.show_setup(self.entities)
        self.dashboard.clear()
        self.pysindy_tab.set_result(None)
        self._set_stage(0)
        self.statusBar().showMessage(f"Opened {Path(path).name}: {len(entities)} items. Press RUN.", 8000)
        return True

    # ------------------------------------------------------------- export
    def _exports_dir(self):
        from core.exporter import EXPORTS_DIR
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        return EXPORTS_DIR

    def export(self, what):
        if self.job is not None:
            return
        if self.result is None:
            QMessageBox.information(self, "Nothing to export yet",
                                    "Press RUN first. Exports are made from the last finished run.")
            return
        from core import exporter
        start = exporter.default_folder(self.experiment_name)
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export: choose a name for the new folder", str(start), "Folder (*)")
        if not fn:
            return
        folder = Path(fn)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Exporting…")
        QApplication.processEvents()
        try:
            if what == "graphs":
                files = exporter.export_graphs(self.result, folder)
            elif what == "raw":
                files = exporter.export_raw(self.result, folder)
            elif what == "sindy":
                files, _ = exporter.export_sindy(self.result, folder)
            else:
                files = exporter.export_all(self.result, folder)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Export failed", f"{type(exc).__name__}: {exc}")
            return
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Exported {len(files)} files to {folder}", 10000)
        box = QMessageBox(self)
        box.setWindowTitle("Export done")
        box.setText(f"Saved {len(files)} files in:\n{folder}")
        box.setDetailedText("\n".join(Path(f).name for f in files))
        open_btn = box.addButton("Open folder", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is open_btn:
            self._open_folder(folder)

    def _open_folder(self, folder):
        folder = str(folder)
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            QMessageBox.information(self, "Folder", f"{folder}\n\n(could not open it: {exc})")

    # ------------------------------------------------------------- editing glue
    def _toolbox_double_clicked(self, item):
        kind = item.data(Qt.ItemDataRole.UserRole)
        if kind and self.job is None:
            self.tabs.setCurrentIndex(0)
            self.canvas2d.add_entity_free_spot(kind)

    def _sync_canvas(self):
        self.canvas2d.field_force = self.force_spin.value()
        self.canvas2d.source_strength = self.source_spin.value()
        self.canvas2d.legacy = self.rule_combo.currentData() == "legacy2600"

    def _on_selected(self, e):
        if e is not self.editor.entity:
            self.editor.set_entity(e, legacy=self.canvas2d.legacy)
        else:
            self.editor.refresh()

    def _on_entities_changed(self):
        self.result = None
        self.canvas3d.show_setup(self.entities)
        n = len(self.entities)
        self.statusBar().showMessage(f"{n} item{'s' if n != 1 else ''} on the grid", 3000)

    def _on_editor_changed(self):
        self.canvas2d.clear_paths()
        self.canvas2d.refresh()
        self.result = None
        self.canvas3d.show_setup(self.entities)

    def _aim(self):
        if not self.canvas2d.aim_at_nearest():
            self.statusBar().showMessage("Select a moving body (Active Body or Tracer) with something to aim at.", 5000)

    def _orbit(self):
        if not self.canvas2d.circular_orbit():
            self.statusBar().showMessage("Select a moving body near a Gravity Well or Static Anchor.", 5000)

    def _toggle_run(self):
        if self.job is not None:
            self.job.stop()
            self.statusBar().showMessage("Stopping…")
            return
        if not self.entities:
            QMessageBox.information(self, "Nothing to run",
                                    "Put something on the grid first.\n\n"
                                    "Tip: File → Load demo shows how it works.")
            return
        from core.engine import SimSettings
        from gui.compute_worker import ComputeJob
        s = SimSettings.from_dict(self.settings())
        problems = s.check()
        if problems:
            QMessageBox.warning(self, "Check the settings", "\n\n".join(problems))
            return
        if self.hardware.device is None or not self.hardware._info:
            self.hardware.detect()
        self.job = ComputeJob(s, list(self.entities), self.hardware.device, self)
        self.job.progress.connect(self._on_progress)
        self.job.finished.connect(self._on_finished)
        self.job.failed.connect(self._on_failed)
        self.run_button.setText("■  STOP")
        self.canvas2d.editable = False
        self.toolbox.setEnabled(False)
        self.editor.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.stage_buttons[1].setChecked(True)
        self.statusBar().showMessage(f"Computing on {self.hardware.device.upper()}…")
        self.job.start()

    def _end_job(self):
        self.job = None
        self.run_button.setText("▶  RUN")
        self.progress.setVisible(False)
        self.canvas2d.editable = True
        self.toolbox.setEnabled(True)
        self.editor.setEnabled(True)

    def _on_progress(self, frac, msg):
        self.progress.setValue(int(frac * 100))
        self.progress.setFormat(f"%p%  ·  {msg}")

    def _on_finished(self, res):
        self._end_job()
        self.result = res
        self.canvas2d.show_paths(res)
        tel = res.telemetry
        self.dashboard.set_data(res.t, energy=tel["total_energy"], peak_phi=tel["peak_phi"],
                                horizon=tel["horizon_radius"], field_energy=tel.get("field_energy"),
                                particle_kinetic=tel.get("particle_kinetic"))
        self.pysindy_tab.set_result(res)
        self.tabs.setCurrentIndex(1)          # jump to the 3D view and replay the run
        self.canvas3d.show_result(res, autoplay=True)
        self.stage_buttons[2].setChecked(True)
        msg = f"{res.message}  ({res.wall_seconds:.1f} s on {res.device.upper()})"
        self.statusBar().showMessage(msg)
        if res.status == "diverged":
            QMessageBox.warning(self, "The field blew up", res.message)

    def _on_failed(self, msg):
        self._end_job()
        self.statusBar().showMessage("Compute failed")
        QMessageBox.critical(self, "Compute failed", msg)

    def closeEvent(self, event):
        if self.job is not None:
            self.job.stop()
            self.job.thread.quit()
            self.job.thread.wait(3000)
        super().closeEvent(event)

    # -------------------------------------------------------------- events
    def showEvent(self, event):
        super().showEvent(event)
        if not self._hardware_checked:
            self._hardware_checked = True
            self.statusBar().showMessage("Checking hardware…")
            QTimer.singleShot(150, self.hardware.detect)

    def _set_stage(self, idx):
        self.stage_buttons[idx].setChecked(True)
        if idx == 0:
            self.tabs.setCurrentIndex(0)
            self.statusBar().showMessage("Setup: place items on the grid")
        elif idx == 1:
            if self.job is None:
                self.statusBar().showMessage("Compute: press RUN to start", 6000)
        else:
            if self.tabs.currentIndex() == 0:
                self.tabs.setCurrentIndex(1)
            self.statusBar().showMessage("Analyze: view paths and graphs")

    def _on_tab_changed(self, idx):
        self.stage_buttons[0 if idx == 0 else 2].setChecked(True)

    def _on_rule_changed(self):
        legacy = self.rule_combo.currentData() == "legacy2600"
        self.legacy_box.setEnabled(legacy)
        self.sponge_spin.setEnabled(not legacy)
        self.source_spin.setEnabled(not legacy)
        self.local_source_check.setEnabled(not legacy)
        self._sync_canvas()
        if self.editor.entity is not None:
            self.editor.set_entity(self.editor.entity, legacy=legacy)
        self._on_box_changed()

    def _on_box_changed(self):
        g, dx = self.grid_spin.value(), self.dx_spin.value()
        legacy = self.rule_combo.currentData() == "legacy2600"
        self.canvas2d.set_box(g, dx, 0 if legacy else self.sponge_spin.value())
        self.canvas3d.set_box(g, dx)

    def _on_device_changed(self, device, note):
        msg = f"Hardware: {device.upper()}"
        if note:
            msg += f"  ({note})"
        self.statusBar().showMessage(msg, 6000)

    def _coming(self, step):
        self.statusBar().showMessage(f"Coming in step {step}", 4000)

    def _show_physics(self):
        QMessageBox.information(
            self, "Physics rules",
            "<b>Field (MS38, default)</b><br>"
            "φ<sub>tt</sub> = ∇·(c²∇φ) − ½ (c²)′ |∇φ|²<br>"
            "c²(φ) = c<sub>base</sub>² / (1 + α|φ|)²<br>"
            "No drain inside the box. Sponge edges soak up waves (true 1/r gravity).<br>"
            "Optional: 'Source uses local wave speed' scales each source by c/c<sub>base</sub> "
            "(off by default; only matters when α > 0; preserves total mass at infinity).<br><br>"
            "<b>Legacy TEGR 2600</b><br>"
            "φ<sub>tt</sub> = c²(φ) ∇²φ, times a decay each tick.<br><br>"
            "<b>Particles</b><br>"
            "Pushed by the field slope (∇φ). Active bodies and anchors feed the field; tracers don't.")

    def _show_about(self):
        QMessageBox.about(self, "About TEGR Genesis",
                          "<b>TEGR Genesis</b><br>Setup → Compute → Analyze<br>"
                          "PyQt6 · PyQtGraph · PyTorch · PySINDy")
