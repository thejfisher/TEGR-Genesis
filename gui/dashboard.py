"""Telemetry dashboard (step 5).

Three linked graphs: energy, peak |phi|, horizon radius r_h.
  - auto-fits to the data after each run (double-click a graph to fit again)
  - the energy graph also shows the field and particle parts (thin lines)
  - a dashed time cursor follows the 3D replay
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout

ACCENT = "#00A2FF"
SILVER = "#A8A8A8"

GRAPHS = [
    ("energy", "Total energy", "E"),
    ("peak_phi", "Peak |φ|", "|φ| max"),
    ("horizon", "Horizon radius r_h", "r_h"),
]
EXTRA = {  # extra thin curves on the energy graph
    "field_energy": ("energy", "#C8C8C8", "field"),
    "particle_kinetic": ("energy", "#FFB000", "particles"),
}


class _FitViewBox(pg.ViewBox):
    """Double-click to fit the graph to its data again."""

    def mouseDoubleClickEvent(self, ev):
        self.enableAutoRange()
        ev.accept()


class Dashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.win = pg.GraphicsLayoutWidget()
        self.win.setObjectName("telemetry")
        self.win.setBackground("#000000")
        layout.addWidget(self.win)

        self.plots, self.curves, self.cursors, self.notes = {}, {}, {}, {}
        first = None
        for row, (key, title, ylabel) in enumerate(GRAPHS):
            p = self.win.addPlot(row=row, col=0, viewBox=_FitViewBox())
            p.setTitle(title, color=ACCENT, size="10pt")
            p.showGrid(x=True, y=True, alpha=0.18)
            p.setLabel("left", ylabel)
            p.setMenuEnabled(False)
            for side in ("left", "bottom"):
                ax = p.getAxis(side)
                ax.setPen(pg.mkPen(SILVER))
                ax.setTextPen(pg.mkPen(SILVER))
                ax.enableAutoSIPrefix(False)
            if first is None:
                first = p
            else:
                p.setXLink(first)
            self.plots[key] = p
            self.curves[key] = p.plot(pen=pg.mkPen(ACCENT, width=2), name="total" if key == "energy" else None)
            cur = pg.InfiniteLine(angle=90, movable=False,
                                  pen=pg.mkPen("#FFFFFF", width=1, style=Qt.PenStyle.DashLine))
            cur.setVisible(False)
            p.addItem(cur, ignoreBounds=True)
            self.cursors[key] = cur
            note = pg.TextItem("", color=SILVER, anchor=(0.5, 0.5))
            note.setVisible(False)
            p.addItem(note, ignoreBounds=True)
            self.notes[key] = note
        self.plots[GRAPHS[-1][0]].setLabel("bottom", "time")

        legend = self.plots["energy"].addLegend(offset=(-10, 10), labelTextColor=SILVER)
        legend.setBrush(pg.mkBrush(0, 0, 0, 160))
        legend.addItem(self.curves["energy"], "total")
        for key, (target, colour, name) in EXTRA.items():
            self.curves[key] = self.plots[target].plot(pen=pg.mkPen(colour, width=1), name=name)

    def set_data(self, t, **series):
        """Fill graphs after a run, e.g. set_data(t, energy=E, peak_phi=P, horizon=R, field_energy=F)."""
        t = np.asarray(t, float)
        for key, values in series.items():
            if key in self.curves and values is not None:
                self.curves[key].setData(t, np.asarray(values, float))
        for key, p in self.plots.items():
            p.enableAutoRange()
            note = self.notes[key]
            vals = series.get(key)
            if key == "horizon" and vals is not None and len(t) and not np.any(np.asarray(vals) > 0):
                p.setYRange(0, 1, padding=0)
                p.disableAutoRange(axis="y")
                note.setText("no horizon formed (α|φ| never reached the threshold)")
                note.setPos(float(t[0] + t[-1]) / 2, 0.5)
                note.setVisible(True)
            else:
                note.setVisible(False)

    def set_cursor(self, t):
        for cur in self.cursors.values():
            cur.setPos(t)
            cur.setVisible(True)

    def clear(self):
        for c in self.curves.values():
            c.setData([], [])
        for cur in self.cursors.values():
            cur.setVisible(False)
        for n in self.notes.values():
            n.setVisible(False)
