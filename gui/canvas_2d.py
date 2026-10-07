"""2D setup grid: the z = 0 plane of the 3D simulation box, in grid units.

Editing (Setup)
  - Drop items from the Toolbox (drag, or double-click a Toolbox item)
  - Click an item to select it; drag it to move it
  - Moving bodies show a speed arrow; drag the arrow tip to set speed and direction
  - Right-click an item: aim at nearest body, circular orbit, duplicate, delete
  - Delete key removes the selected item
After a run it also shows the paths the bodies took.
"""

import math

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QMenu

from core.entities import Entity, MOVING_KINDS

ACCENT = "#00A2FF"
SILVER = "#A8A8A8"
MIME = "application/x-tegr-kind"
ARROW_TIME = 2.0        # arrow length = speed x this many time units
HIT_PX = 14             # how close (pixels) a click must be to grab something
ZERO_HANDLE = 8.0       # grid units: where the speed handle sits when a body is not moving

STYLE = {  # kind -> (symbol, size, colour)
    "well":   ("o", 18, "#00A2FF"),
    "defect": ("star", 20, "#7FE0FF"),
    "anchor": ("s", 13, "#C8C8C8"),
    "active": ("o", 13, "#FFB000"),
    "tracer": ("o", 8, "#9A9A9A"),
}

DEFAULTS = {  # kind -> starting values for a newly dropped item
    "well":   dict(mass=30.0),
    "defect": dict(amplitude=2.0, width=6.0),
    "anchor": dict(mass=10.0),
    "active": dict(mass=1.0),
    "tracer": dict(mass=1.0),
}


class _EditViewBox(pg.ViewBox):
    """ViewBox that lets the canvas grab clicks/drags before the normal pan/zoom."""

    canvas = None

    def mouseClickEvent(self, ev):
        if self.canvas is not None and self.canvas._on_click(ev):
            ev.accept()
            return
        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if self.canvas is not None and self.canvas._on_drag(ev):
            ev.accept()
            return
        super().mouseDragEvent(ev, axis)


class _DropPlot(pg.PlotWidget):
    """PlotWidget that accepts items dragged from the Toolbox."""

    dropped = pyqtSignal(str, float, float)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasFormat(MIME):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        self.dragEnterEvent(ev)

    def dropEvent(self, ev):
        if not ev.mimeData().hasFormat(MIME):
            ev.ignore()
            return
        kind = bytes(ev.mimeData().data(MIME)).decode()
        p = self.plotItem.vb.mapSceneToView(self.mapToScene(ev.position().toPoint()))
        self.dropped.emit(kind, p.x(), p.y())
        ev.acceptProposedAction()


class Canvas2D(QWidget):
    changed = pyqtSignal()              # something was added, removed, moved or edited
    selected = pyqtSignal(object)       # Entity or None

    def __init__(self, grid_res=64, dx=2.8125, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vb = _EditViewBox()
        self.vb.canvas = self
        self.plot = _DropPlot(background="#000000", viewBox=self.vb)
        self.plot.setObjectName("canvas2D")
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setMenuEnabled(False)
        for side in ("left", "bottom"):
            ax = self.plot.getAxis(side)
            ax.setPen(pg.mkPen(SILVER))
            ax.setTextPen(pg.mkPen(SILVER))
        self.plot.setLabel("bottom", "x")
        self.plot.setLabel("left", "y")
        self.plot.setTitle("SETUP GRID  ·  z = 0 plane  ·  drag items in from the Toolbox", color=ACCENT, size="11pt")
        self.plot.dropped.connect(self.add_entity)
        layout.addWidget(self.plot)

        self.entities = []          # shared list (owned by the main window)
        self.current = None         # selected Entity
        self.editable = True
        self.field_force = 1.0      # used by "circular orbit" (K)
        self.source_strength = 1.0  # used by "circular orbit" (G)
        self.legacy = False
        self._half = grid_res * dx / 2.0
        self._dx = dx
        self._box = None
        self._sponge = None
        self._marks = []
        self._paths = []
        self._drag = None           # (mode, entity) while dragging
        self.set_box(grid_res, dx)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------ box
    def set_box(self, grid_res, dx, sponge_cells=0):
        """Draw the simulation box outline (and the sponge edge) and fit the view to it."""
        half = grid_res * dx / 2.0
        self._half, self._dx = half, dx
        for item in (self._box, self._sponge):
            if item is not None:
                self.plot.removeItem(item)
        xs = [-half, half, half, -half, -half]
        ys = [-half, -half, half, half, -half]
        self._box = self.plot.plot(xs, ys, pen=pg.mkPen(ACCENT, width=2))
        self._sponge = None
        if sponge_cells > 0:
            inner = half - sponge_cells * dx
            if inner > 0:
                self._sponge = self.plot.plot([-inner, inner, inner, -inner, -inner],
                                              [-inner, -inner, inner, inner, -inner],
                                              pen=pg.mkPen("#3A5A70", width=1, style=Qt.PenStyle.DashLine))
        self.plot.setRange(xRange=(-half, half), yRange=(-half, half), padding=0.05)

    def _clamp(self, x, y):
        lim = self._half - 2.0 * self._dx
        return max(-lim, min(lim, x)), max(-lim, min(lim, y))

    # ------------------------------------------------------------- editing
    def set_entities(self, entities):
        self.entities = entities
        if self.current not in entities:
            self.current = None
        self.refresh()

    def add_entity(self, kind, x=0.0, y=0.0):
        if not self.editable or kind not in DEFAULTS:
            return None
        x, y = self._clamp(x, y)
        e = Entity(kind, x, y, 0.0, **DEFAULTS[kind])
        self.entities.append(e)
        self.clear_paths()
        self.select(e)
        self.changed.emit()
        return e

    def add_entity_free_spot(self, kind):
        """Add near the centre, stepping outwards until the spot is free (for double-click)."""
        step = 4 * self._dx
        for ring in range(0, 12):
            for k in range(max(1, ring * 6)):
                a = 2 * math.pi * k / max(1, ring * 6)
                x, y = ring * step * math.cos(a), ring * step * math.sin(a)
                if all(math.hypot(e.x - x, e.y - y) > step * 0.9 for e in self.entities):
                    return self.add_entity(kind, x, y)
        return self.add_entity(kind, 0.0, 0.0)

    def delete_selected(self):
        if self.current is not None and self.current in self.entities:
            self.entities.remove(self.current)
            self.current = None
            self.clear_paths()
            self.refresh()
            self.selected.emit(None)
            self.changed.emit()

    def duplicate_selected(self):
        e = self.current
        if e is None:
            return
        d = e.to_dict()
        d["label"] = ""
        n = Entity.from_dict(d)
        n.x, n.y = self._clamp(e.x + 4 * self._dx, e.y + 4 * self._dx)
        self.entities.append(n)
        self.clear_paths()
        self.select(n)
        self.changed.emit()

    def select(self, e):
        self.current = e
        self.refresh()
        self.selected.emit(e)

    # ---------------------------------------------------------- helpers for collisions/orbits
    def _nearest(self, e, kinds):
        best, bd = None, float("inf")
        for o in self.entities:
            if o is e or o.kind not in kinds:
                continue
            d = math.hypot(o.x - e.x, o.y - e.y)
            if d < bd:
                best, bd = o, d
        return best

    def aim_at_nearest(self, e=None):
        """Point the body's speed at the nearest other item (collision course)."""
        e = e or self.current
        if e is None or e.kind not in MOVING_KINDS:
            return False
        target = self._nearest(e, ("well", "anchor", "active", "tracer"))
        if target is None:
            return False
        dx, dy = target.x - e.x, target.y - e.y
        d = math.hypot(dx, dy)
        if d == 0:
            return False
        speed = math.hypot(e.vx, e.vy) or 10.0
        e.vx, e.vy = speed * dx / d, speed * dy / d
        self._after_edit()
        return True

    def circular_orbit(self, e=None):
        """Give the body the speed for a circular orbit around the nearest well or anchor."""
        e = e or self.current
        if e is None or e.kind not in MOVING_KINDS:
            return False
        c = self._nearest(e, ("well", "anchor"))
        if c is None:
            return False
        rx, ry = e.x - c.x, e.y - c.y
        r = math.hypot(rx, ry)
        if r == 0:
            return False
        gm = self.field_force * self.source_strength * c.mass
        v = math.sqrt(max(gm, 0.0) / r)
        e.vx, e.vy = -v * ry / r, v * rx / r     # anticlockwise
        self._after_edit()
        return True

    def _after_edit(self):
        self.clear_paths()
        self.refresh()
        self.selected.emit(self.current)
        self.changed.emit()

    # ------------------------------------------------------------- drawing
    def _tip(self, e):
        if e.vx == 0 and e.vy == 0:
            return e.x + ZERO_HANDLE, e.y
        return e.x + e.vx * ARROW_TIME, e.y + e.vy * ARROW_TIME

    def refresh(self):
        for m in self._marks:
            self.plot.removeItem(m)
        self._marks = []
        add = self._marks.append

        # speed arrows (behind the markers)
        for e in self.entities:
            if e.kind not in MOVING_KINDS:
                continue
            tx, ty = self._tip(e)
            col = STYLE[e.kind][2]
            still = e.vx == 0 and e.vy == 0
            if still and e is not self.current:
                continue
            pen = pg.mkPen(col, width=2, style=Qt.PenStyle.DotLine if still else Qt.PenStyle.SolidLine)
            line = self.plot.plot([e.x, tx], [e.y, ty], pen=pen)
            line.setZValue(5); add(line)
            handle = pg.ScatterPlotItem([tx], [ty], symbol="t1" if not still else "o", size=11,
                                        brush=pg.mkBrush(col if not still else "#000000"),
                                        pen=pg.mkPen(col, width=2))
            if not still:
                ang = math.degrees(math.atan2(e.vy, e.vx))
                arrow = pg.ArrowItem(pos=(tx, ty), angle=180 - ang, headLen=14, tipAngle=35,
                                     brush=pg.mkBrush(col), pen=pg.mkPen(col))
                arrow.setZValue(6); add(arrow); self.plot.addItem(arrow)
            else:
                handle.setZValue(6); add(handle); self.plot.addItem(handle)

        # markers
        for kind, (sym, size, col) in STYLE.items():
            pts = [e for e in self.entities if e.kind == kind]
            if not pts:
                continue
            item = pg.ScatterPlotItem([e.x for e in pts], [e.y for e in pts], symbol=sym, size=size,
                                      brush=pg.mkBrush(col), pen=pg.mkPen("#FFFFFF", width=1))
            item.setZValue(10)
            self.plot.addItem(item); add(item)

        # selection ring + label
        e = self.current
        if e is not None and e in self.entities:
            ring = pg.ScatterPlotItem([e.x], [e.y], symbol="o", size=STYLE[e.kind][1] + 14,
                                      brush=pg.mkBrush(None), pen=pg.mkPen(ACCENT, width=2))
            ring.setZValue(9); self.plot.addItem(ring); add(ring)
            label = pg.TextItem(e.label, color=ACCENT, anchor=(0, 1.2))
            label.setPos(e.x, e.y); label.setZValue(12)
            self.plot.addItem(label); add(label)

    def show_entities(self, entities):
        """Kept for older callers: same as set_entities."""
        self.set_entities(entities)

    def show_paths(self, result):
        self.clear_paths()
        if result is None or result.pos.shape[1] == 0:
            return
        for j, kind in enumerate(result.kinds):
            if kind not in ("active", "tracer"):
                continue
            ok = result.alive[:, j]
            xy = result.pos[ok, j, :2]
            if len(xy) < 2:
                continue
            col = STYLE[kind][2]
            line = self.plot.plot(xy[:, 0], xy[:, 1], pen=pg.mkPen(col, width=2))
            end = pg.ScatterPlotItem([xy[-1, 0]], [xy[-1, 1]], symbol="o", size=8,
                                     brush=pg.mkBrush(col), pen=pg.mkPen("#FFFFFF"))
            end.setZValue(11)
            self.plot.addItem(end)
            self._paths += [line, end]

    def clear_paths(self):
        for p in self._paths:
            self.plot.removeItem(p)
        self._paths = []

    # ------------------------------------------------------------- mouse
    def _px(self, x, y):
        return self.vb.mapViewToScene(QPointF(x, y))

    def _hit(self, scene_pos):
        """Return (mode, entity): mode 'speed' for an arrow tip, 'move' for an item, or (None, None)."""
        def near(x, y):
            p = self._px(x, y)
            return math.hypot(p.x() - scene_pos.x(), p.y() - scene_pos.y()) <= HIT_PX
        order = sorted(self.entities, key=lambda e: e is not self.current)  # selected first
        for e in order:
            if e.kind in MOVING_KINDS and ((e.vx or e.vy) or e is self.current):
                if near(*self._tip(e)):
                    return "speed", e
        for e in order:
            if near(e.x, e.y):
                return "move", e
        return None, None

    def _on_click(self, ev):
        if not self.editable:
            return False
        mode, e = self._hit(ev.scenePos())
        if ev.button() == Qt.MouseButton.LeftButton:
            self.select(e)
            self.setFocus()
            return True
        if ev.button() == Qt.MouseButton.RightButton and e is not None:
            self.select(e)
            self._menu(e, ev.screenPos())
            return True
        return False

    def _on_drag(self, ev):
        if not self.editable or ev.button() != Qt.MouseButton.LeftButton:
            return False
        if ev.isStart():
            mode, e = self._hit(ev.buttonDownScenePos())
            if e is None:
                self._drag = None
                return False       # let the normal pan happen
            self._drag = (mode, e)
            if e is not self.current:
                self.select(e)
        if self._drag is None:
            return False
        mode, e = self._drag
        p = self.vb.mapSceneToView(ev.scenePos())
        x, y = self._clamp(p.x(), p.y())
        if mode == "move":
            e.x, e.y = x, y
        else:
            dx, dy = x - e.x, y - e.y
            if math.hypot(dx, dy) < 1.5 * self._dx:
                e.vx = e.vy = 0.0
            else:
                e.vx, e.vy = dx / ARROW_TIME, dy / ARROW_TIME
        self.clear_paths()
        self.refresh()
        self.selected.emit(e)
        if ev.isFinish():
            self._drag = None
            self.changed.emit()
        return True

    def _menu(self, e, screen_pos):
        m = QMenu(self)
        if e.kind in MOVING_KINDS:
            a = m.addAction("Aim at nearest body (collision course)")
            a.triggered.connect(lambda: self.aim_at_nearest(e))
            a = m.addAction("Circular orbit around nearest well / anchor")
            a.setEnabled(not self.legacy)
            a.triggered.connect(lambda: self.circular_orbit(e))
            a = m.addAction("Stop (speed = 0)")
            a.triggered.connect(lambda: (setattr(e, "vx", 0.0), setattr(e, "vy", 0.0), self._after_edit()))
            m.addSeparator()
        m.addAction("Duplicate").triggered.connect(self.duplicate_selected)
        m.addAction("Delete").triggered.connect(self.delete_selected)
        m.exec(screen_pos.toPoint() if hasattr(screen_pos, "toPoint") else screen_pos)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.editable:
            self.delete_selected()
            return
        if ev.key() == Qt.Key.Key_Escape:
            self.select(None)
            return
        super().keyPressEvent(ev)
