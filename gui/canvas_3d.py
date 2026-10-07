"""3D results view (step 5). Built on the TEGR Collider 3D viewer (GLViewWidget + line paths + points).

What it shows
  Before a run : a 3D preview of the setup (items + speed arrows) on the X/Y/Z reference grid.
  After a run  : - the field on the z = 0 plane as a height sheet (a gravity well is a dent;
                   white-hot cells are inside the horizon, where alpha|phi| >= threshold)
                 - the paths the moving bodies took (true 3D positions)
                 - where every body is at the replay time
                 - a replay bar: Play / Pause, time slider, show/hide options, reset camera
"""

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QVector3D, QFont
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
                             QCheckBox)

from core.entities import MOVING_KINDS

ACCENT = (0.0, 0.635, 1.0)

# kind -> (rgba, point size in pixels)   (same colours as the 2D grid)
KIND_STYLE = {
    "well":   ((0.0, 0.635, 1.0, 1.0), 18),
    "defect": ((0.5, 0.88, 1.0, 1.0), 14),
    "anchor": ((0.78, 0.78, 0.78, 1.0), 13),
    "active": ((1.0, 0.69, 0.0, 1.0), 12),
    "tracer": ((0.75, 0.75, 0.75, 1.0), 8),
}

# colour ramp for the field sheet: |phi| / max  ->  rgb
SHEET_STOPS = np.array([
    [0.00, 0.05, 0.08, 0.12],
    [0.30, 0.00, 0.30, 0.55],
    [0.65, 0.00, 0.635, 1.00],
    [1.00, 0.85, 0.95, 1.00],
])
HORIZON_RGB = (1.0, 1.0, 1.0)
SHEET_DEPTH = 0.22          # deepest dent = this fraction of the box size
REPLAY_SECONDS = 8.0        # Play runs through the whole result in about this long
FPS = 30


def _ramp(u):
    u = np.clip(u, 0.0, 1.0)
    out = np.empty(u.shape + (4,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(u, SHEET_STOPS[:, 0], SHEET_STOPS[:, c + 1])
    out[..., 3] = 1.0
    return out


class Canvas3D(QWidget):
    time_changed = pyqtSignal(float)    # replay time (for the telemetry cursor)

    def __init__(self, grid_res=64, dx=2.8125, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.result = None
        self._grid_items = []
        self._scene_items = []      # setup preview or result items
        self._sheet = None
        self._paths = []            # (GLLinePlotItem, xyz array, last_alive_index)
        self._movers = None
        self._fixed = None
        self._idx = 0
        self._size = grid_res * dx

        self.title = QLabel("3D VIEW")
        self.title.setObjectName("view3dTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("color:#00A2FF; font-weight:bold; padding:2px;")
        layout.addWidget(self.title)

        try:
            self.view = gl.GLViewWidget()
        except Exception as exc:  # no OpenGL on this machine
            self.view = None
            layout.addWidget(QLabel(f"3D view unavailable (OpenGL problem):\n{exc}"))
            return
        self.view.setObjectName("canvas3D")
        self.view.setBackgroundColor("k")
        layout.addWidget(self.view, 1)
        layout.addWidget(self._build_controls())

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / FPS))
        self.timer.timeout.connect(self._tick)

        self.set_box(grid_res, dx)
        self._set_controls_enabled(False)

    # ------------------------------------------------------------------ controls
    def _build_controls(self):
        bar = QWidget(); bar.setObjectName("replayBar")
        h = QHBoxLayout(bar); h.setContentsMargins(6, 2, 6, 4)
        self.play_btn = QPushButton("▶  Play"); self.play_btn.setObjectName("replayPlay")
        self.play_btn.setToolTip("Replay the run")
        self.play_btn.setFixedWidth(96)
        self.play_btn.clicked.connect(self.toggle_play)
        self.repeat_btn = QPushButton("↻  Repeat"); self.repeat_btn.setObjectName("replayRepeat")
        self.repeat_btn.setCheckable(True)
        self.repeat_btn.setToolTip("On: the replay starts again from the beginning when it reaches the end")
        self.repeat_btn.setFixedWidth(104)
        self.repeat_btn.toggled.connect(self._on_repeat_toggled)
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setObjectName("replaySlider")
        self.slider.setToolTip("Drag to move through time")
        self.slider.valueChanged.connect(self._on_slider)
        self.time_label = QLabel("t = 0"); self.time_label.setObjectName("replayTime")
        self.time_label.setMinimumWidth(110)
        self.sheet_check = QCheckBox("Field sheet"); self.sheet_check.setObjectName("showSheet")
        self.sheet_check.setChecked(True)
        self.sheet_check.setToolTip("The field φ on the z = 0 plane, drawn as a height sheet.\n"
                                    "Dents = gravity wells. White = inside the horizon.")
        self.sheet_check.toggled.connect(self._update_visibility)
        self.paths_check = QCheckBox("Whole paths"); self.paths_check.setObjectName("wholePaths")
        self.paths_check.setToolTip("On: show each full path.  Off: show the path only up to the replay time.")
        self.paths_check.toggled.connect(lambda _=False: self._show_index(self._idx))
        reset = QPushButton("Reset camera"); reset.setObjectName("resetCamera")
        reset.clicked.connect(self.reset_camera)
        for w in (self.play_btn, self.repeat_btn, self.slider, self.time_label, self.sheet_check,
                  self.paths_check, reset):
            h.addWidget(w, 1 if w is self.slider else 0)
        return bar

    def _set_controls_enabled(self, on):
        for w in (self.play_btn, self.repeat_btn, self.slider, self.sheet_check, self.paths_check):
            w.setEnabled(on)

    # ------------------------------------------------------------------ box + grid
    def set_box(self, grid_res, dx):
        if self.view is None:
            return
        for item in self._grid_items:
            self.view.removeItem(item)
        self._grid_items = []

        size = grid_res * dx
        half = size / 2.0
        spacing = size / 8.0
        self._size = size

        # Three reference planes: XY (floor), XZ and YZ (back walls)
        floor = gl.GLGridItem(color=(0, 162, 255, 90))
        floor.setSize(size, size)
        floor.setSpacing(spacing, spacing)
        floor.translate(0, 0, -half)

        wall_xz = gl.GLGridItem(color=(168, 168, 168, 50))
        wall_xz.setSize(size, size)
        wall_xz.setSpacing(spacing, spacing)
        wall_xz.rotate(90, 1, 0, 0)
        wall_xz.translate(0, half, 0)

        wall_yz = gl.GLGridItem(color=(168, 168, 168, 50))
        wall_yz.setSize(size, size)
        wall_yz.setSpacing(spacing, spacing)
        wall_yz.rotate(90, 0, 1, 0)
        wall_yz.translate(-half, 0, 0)

        # X / Y / Z axes through the centre, with labels
        axes = []
        font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        for vec, name, col in (((1, 0, 0), "X", (255, 90, 90, 255)),
                               ((0, 1, 0), "Y", (120, 230, 120, 255)),
                               ((0, 0, 1), "Z", (0, 162, 255, 255))):
            end = np.array(vec, float) * half
            line = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], end]), color=tuple(c / 255 for c in col),
                                     width=2, antialias=True)
            axes.append(line)
            try:
                axes.append(gl.GLTextItem(pos=end * 1.06, text=name, color=col, font=font))
            except Exception:
                pass

        for item in [floor, wall_xz, wall_yz] + axes:
            self.view.addItem(item)
            self._grid_items.append(item)
        self.reset_camera()

    def reset_camera(self):
        if self.view is None:
            return
        self.view.opts["center"] = QVector3D(0, 0, 0)
        self.view.setCameraPosition(distance=self._size * 1.9, elevation=24, azimuth=-55)

    # ------------------------------------------------------------------ scene helpers
    def _clear_scene(self):
        self.timer.stop() if self.view is not None else None
        if self.view is None:
            return
        for item in self._scene_items:
            self.view.removeItem(item)
        self._scene_items = []
        self._sheet = None
        self._paths = []
        self._movers = None
        self._fixed = None

    def _add(self, item):
        self.view.addItem(item)
        self._scene_items.append(item)
        return item

    # ------------------------------------------------------------------ setup preview
    def show_setup(self, entities):
        """3D preview of what is on the 2D grid (before a run)."""
        if self.view is None:
            return
        self.result = None
        self._clear_scene()
        self._set_controls_enabled(False)
        self.play_btn.setText("▶  Play")
        self.time_label.setText("t = 0")
        self.title.setText("3D VIEW  ·  setup preview  ·  press RUN to compute")
        if not entities:
            self.title.setText("3D VIEW  ·  empty  ·  drag items onto the 2D grid, then press RUN")
            return
        pos = np.array([[e.x, e.y, e.z] for e in entities], float)
        col = np.array([KIND_STYLE[e.kind][0] for e in entities], float)
        size = np.array([KIND_STYLE[e.kind][1] for e in entities], float)
        self._add(gl.GLScatterPlotItem(pos=pos, color=col, size=size, pxMode=True))
        # speed arrows (2 time units long, same as the 2D grid)
        segs, cols = [], []
        for e in entities:
            if e.kind in MOVING_KINDS and (e.vx or e.vy or e.vz):
                segs += [[e.x, e.y, e.z], [e.x + 2 * e.vx, e.y + 2 * e.vy, e.z + 2 * e.vz]]
                c = KIND_STYLE[e.kind][0]
                cols += [c, c]
        if segs:
            self._add(gl.GLLinePlotItem(pos=np.array(segs), color=np.array(cols), mode="lines",
                                        width=2, antialias=True))

    # ------------------------------------------------------------------ results
    def show_result(self, res, autoplay=True):
        """Show a finished run: field sheet, paths, markers, replay bar."""
        if self.view is None:
            return
        self._clear_scene()
        self.result = res
        s = res.settings
        G, dx = int(s["grid_resolution"]), float(s["dx"])
        half = G * dx / 2.0
        self._half = half
        self.title.setText(f"3D VIEW  ·  results  ·  {res.message}")

        # --- field sheet
        frames = res.field_frames
        if frames is not None and len(frames):
            xs = -half + (np.arange(G) + 0.5) * dx
            peak = float(np.abs(frames).max())
            self._sheet_peak = peak if peak > 0 else 1.0
            # height is asinh-compressed so the gentle 1/r tail shows, not only the spike at the centre
            self._sheet_soft = self._sheet_peak / 30.0
            self._sheet_norm = float(np.arcsinh(self._sheet_peak / self._sheet_soft))
            self._sheet_scale = SHEET_DEPTH * G * dx if peak > 0 else 0.0
            alpha = float(s.get("alpha", 0.0))
            thr = float(s.get("horizon_threshold", 1.0))
            self._horizon_phi = (thr / alpha) if alpha > 0 else None
            z0, c0 = self._sheet_data(frames[-1])
            self._sheet = self._add(gl.GLSurfacePlotItem(x=xs, y=xs, z=z0, colors=c0, smooth=False,
                                                         computeNormals=False, glOptions="opaque"))

        # --- paths of moving bodies
        T = len(res.t)
        for j, kind in enumerate(res.kinds):
            if kind not in MOVING_KINDS:
                continue
            alive = res.alive[:, j]
            last = int(np.nonzero(alive)[0][-1]) if alive.any() else 0
            xyz = res.pos[:, j, :].astype(np.float32)
            rgba = KIND_STYLE[kind][0]
            line = self._add(gl.GLLinePlotItem(pos=xyz[: last + 1], color=rgba, width=2,
                                               antialias=True, mode="line_strip"))
            self._paths.append((line, xyz, last))

        # --- body markers (fixed and moving)
        fixed = [j for j, k in enumerate(res.kinds) if k not in MOVING_KINDS]
        if fixed:
            self._fixed = self._add(gl.GLScatterPlotItem(
                pos=res.pos[0, fixed, :], color=np.array([KIND_STYLE[res.kinds[j]][0] for j in fixed]),
                size=np.array([KIND_STYLE[res.kinds[j]][1] for j in fixed], float), pxMode=True))
        self._mover_ids = [j for j, k in enumerate(res.kinds) if k in MOVING_KINDS]
        if self._mover_ids:
            self._movers = self._add(gl.GLScatterPlotItem(
                pos=res.pos[-1, self._mover_ids, :],
                color=np.array([KIND_STYLE[res.kinds[j]][0] for j in self._mover_ids]),
                size=np.array([KIND_STYLE[res.kinds[j]][1] + 3 for j in self._mover_ids], float),
                pxMode=True))
        # draw markers on top of the sheet
        for it in (self._fixed, self._movers):
            if it is not None:
                it.setGLOptions("translucent")

        self._set_controls_enabled(T > 1)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, T - 1))
        self.slider.blockSignals(False)
        self._update_visibility()
        if autoplay and T > 1:
            self.slider.setValue(0)
            self._show_index(0)
            self.play()
        else:
            self.slider.setValue(T - 1)
            self._show_index(T - 1)

    def _sheet_data(self, frame):
        u = np.arcsinh(frame / self._sheet_soft) / self._sheet_norm      # -1 .. 1
        z = (-self._sheet_scale * u).astype(np.float32)
        colors = _ramp(np.abs(u))
        if self._horizon_phi is not None:
            inside = np.abs(frame) >= self._horizon_phi
            colors[inside, :3] = HORIZON_RGB
        return z, colors.reshape(-1, 4)    # pyqtgraph 0.14 wants one colour per vertex, flat

    def _frame_for_time(self, t):
        ft = self.result.frame_t
        if ft is None or len(ft) == 0:
            return None
        k = int(np.searchsorted(ft, t + 1e-12, side="right") - 1)
        return max(0, min(k, len(ft) - 1))

    def _show_index(self, i):
        res = self.result
        if res is None or self.view is None or len(res.t) == 0:
            return
        i = max(0, min(int(i), len(res.t) - 1))
        self._idx = i
        t = float(res.t[i])
        self.time_label.setText(f"t = {t:,.2f}")
        whole = self.paths_check.isChecked()
        for line, xyz, last in self._paths:
            n = last + 1 if whole else min(i, last) + 1
            line.setData(pos=xyz[: max(n, 1)])
        if self._movers is not None:
            self._movers.setData(pos=res.pos[i, self._mover_ids, :])
        if self._sheet is not None:
            k = self._frame_for_time(t)
            if k is not None and k != getattr(self, "_sheet_k", None):
                self._sheet_k = k
                z, c = self._sheet_data(res.field_frames[k])
                self._sheet.setData(z=z, colors=c)
        self.time_changed.emit(t)

    def _update_visibility(self):
        if self._sheet is not None:
            self._sheet.setVisible(self.sheet_check.isChecked())

    # ------------------------------------------------------------------ replay
    def play(self):
        if self.result is None or len(self.result.t) < 2:
            return
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)
        self._step = max(1, int(round(len(self.result.t) / (REPLAY_SECONDS * FPS))))
        self.play_btn.setText("❚❚  Pause")
        self.timer.start()

    def pause(self):
        self.timer.stop()
        self.play_btn.setText("▶  Play")

    def toggle_play(self):
        if self.timer.isActive():
            self.pause()
        else:
            self.play()

    def _tick(self):
        v = self.slider.value() + self._step
        if v >= self.slider.maximum():
            self.slider.setValue(self.slider.maximum())
            if self.repeat_btn.isChecked():
                self.slider.setValue(0)         # loop: start again
                return
            self.pause()
            return
        self.slider.setValue(v)

    def _on_repeat_toggled(self, on):
        self.repeat_btn.setStyleSheet("background:#00A2FF; color:#000000;" if on else "")
        if on and not self.timer.isActive():
            self.play()

    def _on_slider(self, v):
        self._sheet_k = getattr(self, "_sheet_k", None)
        self._show_index(v)
