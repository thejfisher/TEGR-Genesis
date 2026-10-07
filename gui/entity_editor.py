"""'Selected object' editor shown in the Properties panel."""

import math

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QDoubleSpinBox, QPushButton,
)

from core.entities import NICE_NAMES, MOVING_KINDS

FIELDS = [  # attribute, label, decimals, min, max, tooltip, kinds that use it
    ("x", "x", 2, -1e4, 1e4, "Position across", None),
    ("y", "y", 2, -1e4, 1e4, "Position up/down", None),
    ("mass", "Mass", 3, 0.0, 1e7, "Source size and inertia",
     ("well", "anchor", "active", "tracer")),
    ("vx", "Speed x", 3, -1e4, 1e4, "Starting speed across", MOVING_KINDS),
    ("vy", "Speed y", 3, -1e4, 1e4, "Starting speed up/down", MOVING_KINDS),
    ("amplitude", "Height", 3, -1e4, 1e4, "Peak height of the bump", ("defect",)),
    ("width", "Width", 3, 0.1, 1e4, "Bump width (grid units)", ("defect",)),
    ("phase", "Phase θ", 3, -100.0, 100.0, "Particle phase (used by the Pauli push)",
     ("anchor", "active", "tracer")),
]


class EntityEditor(QWidget):
    changed = pyqtSignal()
    delete_requested = pyqtSignal()
    aim_requested = pyqtSignal()
    orbit_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entity = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.empty = QLabel("Nothing selected.\nDrag an item from the Toolbox onto the grid,\nor click one that is already there.")
        self.empty.setObjectName("hintLabel")
        outer.addWidget(self.empty)

        self.body = QWidget()
        form = QFormLayout(self.body)
        form.setContentsMargins(0, 0, 0, 0)
        self.kind_label = QLabel()
        self.kind_label.setObjectName("entityKind")
        form.addRow("Type", self.kind_label)
        self.name_edit = QLineEdit(); self.name_edit.setObjectName("entityName")
        self.name_edit.editingFinished.connect(self._name_changed)
        form.addRow("Name", self.name_edit)

        self.spins = {}
        self.rows = {}
        for attr, label, dec, lo, hi, tip, kinds in FIELDS:
            s = QDoubleSpinBox(); s.setObjectName(f"entity_{attr}")
            s.setDecimals(dec); s.setRange(lo, hi); s.setToolTip(tip)
            s.setSingleStep(1.0 if attr in ("x", "y", "vx", "vy", "mass", "width") else 0.1)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(lambda v, a=attr: self._value_changed(a, v))
            lab = QLabel(label)
            form.addRow(lab, s)
            self.spins[attr] = s
            self.rows[attr] = (lab, s, kinds)

        self.speed_label = QLabel(); self.speed_label.setObjectName("hintLabel")
        form.addRow("", self.speed_label)

        btns = QVBoxLayout()
        self.aim_btn = QPushButton("Aim at nearest body"); self.aim_btn.setObjectName("aimButton")
        self.aim_btn.setToolTip("Point the speed arrow at the nearest item (collision course)")
        self.orbit_btn = QPushButton("Circular orbit"); self.orbit_btn.setObjectName("orbitButton")
        self.orbit_btn.setToolTip("Speed for a circular orbit around the nearest well or anchor")
        row = QHBoxLayout(); row.addWidget(self.aim_btn); row.addWidget(self.orbit_btn)
        btns.addLayout(row)
        self.del_btn = QPushButton("Delete"); self.del_btn.setObjectName("deleteButton")
        btns.addWidget(self.del_btn)
        form.addRow(btns)
        self.aim_btn.clicked.connect(self.aim_requested)
        self.orbit_btn.clicked.connect(self.orbit_requested)
        self.del_btn.clicked.connect(self.delete_requested)

        outer.addWidget(self.body)
        self.set_entity(None)

    def set_entity(self, e, legacy=False):
        self.entity = e
        self.empty.setVisible(e is None)
        self.body.setVisible(e is not None)
        if e is None:
            return
        self.kind_label.setText(NICE_NAMES[e.kind])
        self.name_edit.setText(e.label)
        for attr, (lab, spin, kinds) in self.rows.items():
            show = kinds is None or e.kind in kinds
            lab.setVisible(show); spin.setVisible(show)
        moving = e.kind in MOVING_KINDS
        self.aim_btn.setVisible(moving)
        self.orbit_btn.setVisible(moving)
        self.orbit_btn.setEnabled(not legacy)
        self.speed_label.setVisible(moving)
        self.refresh()

    def refresh(self):
        e = self.entity
        if e is None:
            return
        for attr, spin in self.spins.items():
            spin.blockSignals(True)
            spin.setValue(float(getattr(e, attr)))
            spin.blockSignals(False)
        if e.kind in MOVING_KINDS:
            v = math.hypot(e.vx, e.vy)
            ang = math.degrees(math.atan2(e.vy, e.vx)) if v else 0.0
            self.speed_label.setText(f"speed {v:.2f}  ·  direction {ang:.0f}°")

    def _value_changed(self, attr, v):
        if self.entity is None:
            return
        setattr(self.entity, attr, float(v))
        self.refresh()
        self.changed.emit()

    def _name_changed(self):
        if self.entity is not None and self.name_edit.text().strip():
            self.entity.label = self.name_edit.text().strip()
            self.changed.emit()
