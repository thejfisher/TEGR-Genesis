"""TEGR Genesis physics engine (PyTorch: CUDA / MPS / CPU).

Not real-time: Setup -> Compute (this file) -> Analyze.

FIELD RULES  (source of truth: TEGR 2600 engine.py + MS38 field_harness.py, confirmed by user 2026-10-07)

  MS38 energy-conserving (default)
      phi_tt = div(c^2 grad phi) - 1/2 (c^2)'(phi) |grad phi|^2
      from L = 1/2 phi_t^2 - 1/2 c^2(phi) |grad phi|^2
      discrete (exact copy of field_harness.py, form="lagrangian"):
        phi_i'' = sum_j [ (c2_i + c2_j)/2 (phi_j - phi_i) - 1/4 (c^2)'_i (phi_j - phi_i)^2 ] / dx^2
      no bulk drain; graded sponge edge layer (MS38 Test 2: width 12, strength 0.35); phi = 0 outside.

  Legacy TEGR 2600
      phi_next = decay * (2 phi - phi_prev + c^2 dt^2/dx^2 Lap(phi))       (engine.py L386-405)
      decay = decay_base * (c^2/c_base^2)^gamma, clamped [0.5, 1]           (engine.py L211-215)
      periodic box (2600 default), seed bumps at start, raw source S*m/m_max per tick.

  Both
      c^2(phi) = c_base^2 / (1 + alpha|phi|)^2, clamped to [c2_min (=1.0), CFL max]  (engine.py L203-209)

PARTICLES
  MS38    : F = K * m * grad(phi)   (equivalence principle; as in MS38 wake_orbit.py, which held a bound orbit)
  Legacy  : F = K * grad(phi)       (TEGR 2600 torsion force, engine.py L468-470)
  Optional Pauli push (2600 form): chi * cos(theta_i - theta_j) * r_hat / r^n
  Relativistic momentum: p += F dt ; gamma = sqrt(1 + p^2/(m c)^2) ; v = p/(gamma m) ; x += v dt
  Phase clock: dtheta/dt = m / gamma
  Force timing (MS38, time_centred_force = True, v0.1a 2026-10-18): the force on a body at x^n is read
  from phi^n, the field at the SAME time level (kick-drift leapfrog). Before v0.1a it was read from
  phi^(n+1), one tick ahead. A source's own well moves with it, so one tick ahead the body sat v*dt
  behind its own well centre and was pushed forward by it (self-push ~ dt / dx^3: grew on finer grids,
  halved with dt). Legacy 2600 keeps the old ordering (it reproduces TEGR 2600).
  Body cloud (MS38, body_kernel, v0.1a): how an ACTIVE body (moves and feeds the field) spreads its
  source and reads the force. Tracers, wells and anchors always use CIC (their results are unchanged).
    cic   : 2 x 2 x 2 cells (stock). With the timing fix, a lone body still speeds up at ~0.4 c, and
            more on finer grids (+0.3 / +1.6 / +10 % per time unit at dx 5.6 / 2.8 / 1.4): the
            one-cell-sharp well gets shaken as the body crosses cell faces.
    tsc   : 3 x 3 x 3 cells. About 10x smaller, but still grows on finer grids.
    gauss : Gaussian of fixed physical width body_width (a finite body size, same on every grid).
            Lone body speed steady to ~0.01 % per time unit at 0.1-0.4 c on dx 2.8 and 1.4.
    Deposit and force read use the same cloud for each body.

SOURCES (MS38)
  Each source of mass m adds q = 4 pi c_base^2 dt^2 G m / dx^3 to phi per tick (cloud-in-cell),
  so the settled far field is phi = G m / r (checked by the self-test). Smooth half-cosine switch-on.
  Newton's constant in this model is G_N = K * G.
  Optional (source_local_speed = True, updated 2026-10-17): each cloud-in-cell share is multiplied by
  c(phi_cell) / c_base, so the source ticks at the local wave speed (conserved mass coupling).
  Off by default. With alpha = 0 the factor is exactly 1 (identical to the stock rule).
  Reason (TEGR but Round v2, Phase I): this linearizes the static equation in u = int c dphi,
  stopping horizon core runaway without hiding far-field mass (full mass visible at infinity)
  and recovering linear horizon growth r_h ~ M.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict, fields
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from core.entities import Entity, FIXED_KINDS, SOURCE_KINDS, MOVING_KINDS, PAULI_KINDS
from core.hardware import resolve_device

SQRT3 = math.sqrt(3.0)


# ============================================================================ settings
@dataclass
class SimSettings:
    physics_rule: str = "ms38"          # "ms38" or "legacy2600"
    grid_resolution: int = 64
    dx: float = 2.8125
    dt: float = 0.005
    total_ticks: int = 16000
    c_base: float = 65.0
    alpha: float = 0.0                  # 0 = plain gravity; raise it for horizons (user choice 2026-10-07)
    c2_min: float = 1.0
    sponge_width: int = 12
    sponge_strength: float = 0.35
    source_strength: float = 1.0        # G : settled field phi = G m / r
    source_local_speed: bool = False    # MS38 only: scale each source by local c^2/c_base^2 (see SOURCES above)
    field_force: float = 100.0          # K : F = K m grad(phi)   (legacy: F = K grad(phi))
    source_ramp_ticks: int = 3000
    warmup_ticks: int = 6000            # moving bodies are held still while the field settles
    time_centred_force: bool = True     # MS38 only: force from phi^n at x^n (False = pre-v0.1a, one tick ahead)
    body_kernel: str = "tsc"            # MS38 only: cloud for ACTIVE bodies: "cic" (2^3 cells), "tsc" (3^3), "gauss"
                                        #   default tsc (user choice 2026-10-18); "cic" = pre-v0.1a
    body_width: float = 2.8125          # gauss only: Gaussian width sigma in grid units (physical, not cells)
    pauli_strength: float = 0.0
    pauli_power: float = 3.0
    pauli_reach: float = 0.0            # Yukawa/Debye screening length (0 = infinite range; >0 = exp(-r/reach))
    obstacle_anchors: bool = False      # if True, anchor entities act as rigid field obstacles (phi = 0)
    decay_base: float = 0.999           # legacy only
    decay_gamma: float = 0.5            # legacy only
    legacy_periodic: bool = True        # legacy only
    legacy_source: float = 0.0          # legacy only: raw 2600 source S per tick (S * m / m_max)
    legacy_seed: bool = True            # legacy only: seed Gaussian bumps at start (2600 default)
    record_every: int = 20
    max_field_frames: int = 200
    divergence_limit: float = 0.0       # 0 = auto: 1000 for legacy (TEGR 2600 guard), 1e6 for MS38
    horizon_threshold: float = 1.0      # horizon = where alpha|phi| >= this (1.0 -> wave speed halved)
    hardware: str = "auto"

    @property
    def box_size(self):
        return self.grid_resolution * self.dx

    @property
    def is_legacy(self):
        return self.physics_rule == "legacy2600"

    @property
    def blowup_limit(self):
        if self.divergence_limit > 0:
            return self.divergence_limit
        return 1000.0 if self.is_legacy else 1.0e6

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "torsion_coupling" in d and "field_force" not in d:
            d["field_force"] = d.pop("torsion_coupling")
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def check(self):
        """Return a list of plain-language problems (empty = OK)."""
        errs = []
        if self.physics_rule not in ("ms38", "legacy2600"):
            errs.append(f"Unknown physics rule {self.physics_rule!r}")
        if self.grid_resolution < 8:
            errs.append("Grid size must be at least 8")
        if self.dt <= 0 or self.dx <= 0:
            errs.append("Time step and cell size must be positive")
        if self.total_ticks < 1:
            errs.append("Ticks must be at least 1")
        courant = self.c_base * self.dt / self.dx
        if courant > 1.0 / SQRT3:
            errs.append(f"Unstable: wave speed x dt / dx = {courant:.3f} must be below {1 / SQRT3:.3f}. "
                        f"Lower dt or wave speed, or raise cell size.")
        if not self.is_legacy and self.sponge_width * 2 >= self.grid_resolution:
            errs.append("Sponge edge is too wide for this grid")
        if self.body_kernel not in ("cic", "tsc", "gauss"):
            errs.append(f"Unknown body cloud {self.body_kernel!r} (use cic, tsc or gauss)")
        if self.body_kernel == "gauss" and self.body_width <= 0:
            errs.append("Body width must be positive for the Gaussian cloud")
        return errs


# ============================================================================ results
@dataclass
class RunResult:
    settings: dict
    entities: list                       # list of entity dicts (same order as particle arrays)
    kinds: list                          # kind per particle row
    t: np.ndarray                        # (T,)
    pos: np.ndarray                      # (T, N, 3)
    vel: np.ndarray                      # (T, N, 3)
    alive: np.ndarray                    # (T, N) bool (False after leaving the box)
    telemetry: dict                      # name -> (T,) array
    frame_t: np.ndarray                  # (F,)
    field_frames: np.ndarray             # (F, G, G) z = 0 slice of phi
    final_phi: np.ndarray                # (G, G, G)
    gamma: np.ndarray = None             # (T, N)
    phase: np.ndarray = None             # (T, N)
    status: str = "completed"
    message: str = ""
    device: str = "cpu"
    wall_seconds: float = 0.0
    ticks_done: int = 0


# ============================================================================ field
class FieldSolver:
    """3D field update. Mirrors MS38 field_harness.py so results can be checked against it."""

    def __init__(self, s: SimSettings, device, dtype=torch.float32):
        self.s = s
        self.device = device
        self.dtype = dtype
        G = s.grid_resolution
        self.G, self.dx, self.dt = G, s.dx, s.dt
        self.cb2 = s.c_base ** 2
        self.alpha = s.alpha
        self.form = "engine" if s.is_legacy else "lagrangian"
        self.periodic = bool(s.legacy_periodic) if s.is_legacy else False
        self.c2_min = s.c2_min
        self.c2_max = (s.dx / (s.dt * SQRT3)) ** 2

        shape = (1, 1, G, G, G)
        self.phi = torch.zeros(shape, device=device, dtype=dtype)
        self.phi_prev = torch.zeros_like(self.phi)

        k = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=dtype)
        k[0, 0, 1, 1, 1] = -6.0
        for idx in [(1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1), (0, 1, 1), (2, 1, 1)]:
            k[(0, 0) + idx] = 1.0
        self.lap_kernel = k

        if s.is_legacy:
            self.base_decay = None
        else:
            self.base_decay = self._sponge(1.0, s.sponge_width, s.sponge_strength) if s.sponge_width > 0 else None
        self.c2 = None
        self.decay = None
        self.obstacle_mask = None
        self._update_impedance()

    # -- setup
    def _sponge(self, bulk, W, s_max):
        """Exact copy of field_harness._sponge: graded quadratic drop inside a W-cell edge layer."""
        G = self.G
        i = torch.arange(G, device=self.device, dtype=self.dtype)
        depth = torch.clamp(W - torch.minimum(i, G - 1 - i), min=0) / W
        d = torch.maximum(torch.maximum(depth[:, None, None], depth[None, :, None]), depth[None, None, :])
        return (bulk * (1.0 - s_max * d ** 2)).reshape(1, 1, G, G, G)

    def _pad(self, t):
        if self.periodic:
            return F.pad(t, (1, 1, 1, 1, 1, 1), mode="circular")
        return F.pad(t, (1, 1, 1, 1, 1, 1), mode="constant", value=0.0)

    def _neighbors(self, t):
        p = self._pad(t)
        return [p[:, :, 2:, 1:-1, 1:-1], p[:, :, :-2, 1:-1, 1:-1],
                p[:, :, 1:-1, 2:, 1:-1], p[:, :, 1:-1, :-2, 1:-1],
                p[:, :, 1:-1, 1:-1, 2:], p[:, :, 1:-1, 1:-1, :-2]]

    # -- physics
    def c2_of(self, phi):
        raw = self.cb2 / (1.0 + self.alpha * phi.abs()).square()
        return raw, raw.clamp(min=self.c2_min, max=self.c2_max)

    def _update_impedance(self):
        raw, self.c2 = self.c2_of(self.phi)
        self._raw = raw
        if self.s.is_legacy:
            d = self.s.decay_base * (self.c2 / self.cb2).pow(self.s.decay_gamma)
            self.decay = d.clamp(min=0.5, max=1.0)
        else:
            self.decay = self.base_decay  # None = no drain at all

    def _accel_term(self):
        """Spatial term times dt^2 (what gets added to 2 phi - phi_prev)."""
        dt2_dx2 = self.dt * self.dt / (self.dx * self.dx)
        if self.form == "engine":
            lap = F.conv3d(self._pad(self.phi), self.lap_kernel, padding=0)
            return lap * (self.c2 * dt2_dx2)
        phi, c2, raw = self.phi, self.c2, self._raw
        if self.alpha != 0.0:
            dc2 = -2.0 * self.alpha * torch.sign(phi) * self.cb2 / (1.0 + self.alpha * phi.abs()) ** 3
            clamped = (raw < self.c2_min) | (raw > self.c2_max)
            dc2 = torch.where(clamped, torch.zeros_like(dc2), dc2)
        else:
            dc2 = None
        acc = torch.zeros_like(phi)
        for pj, c2j in zip(self._neighbors(phi), self._neighbors(c2)):
            diff = pj - phi
            acc = acc + 0.5 * (c2 + c2j) * diff
            if dc2 is not None:
                acc = acc - 0.25 * dc2 * diff * diff
        return acc * dt2_dx2

    def step(self):
        self._update_impedance()
        nxt = torch.mul(self.phi, 2.0)
        nxt.sub_(self.phi_prev)
        nxt.add_(self._accel_term())
        if self.decay is not None:
            nxt.mul_(self.decay)
        if self.obstacle_mask is not None:
            nxt.mul_(self.obstacle_mask)
        self.phi_prev, self.phi = self.phi, nxt

    def energy(self):
        """Field energy (kinetic, potential) — same formula as field_harness.energy()."""
        phi_t = (self.phi - self.phi_prev) / self.dt
        kin = 0.5 * (phi_t ** 2).sum()
        _, c2 = self.c2_of(self.phi)
        p = self._pad(self.phi)
        c2p = self._pad(c2)
        pot = torch.zeros((), device=self.device, dtype=self.dtype)
        for s in [(slice(2, None), slice(1, -1), slice(1, -1)),
                  (slice(1, -1), slice(2, None), slice(1, -1)),
                  (slice(1, -1), slice(1, -1), slice(2, None))]:
            d = p[(0, 0) + s] - self.phi[0, 0]
            c2f = 0.5 * (c2[0, 0] + c2p[(0, 0) + s])
            pot = pot + 0.5 * (c2f * d * d).sum() / (self.dx * self.dx)
        return float(kin), float(pot)


# ============================================================================ grid sampling
class Sampler:
    """Cloud-in-cell deposit and trilinear sampling on cell-centred coordinates.

    Cell i has its centre at x = -L/2 + (i + 0.5) dx. Deposit and sampling use the same map,
    so a source and the force it creates are always in the same place.
    """

    def __init__(self, G, dx, device, dtype):
        self.G, self.dx = G, dx
        self.half = G * dx / 2.0
        self.device, self.dtype = device, dtype
        o = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                          [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], device=device)
        self.corners = o  # (8, 3)

    def _cell(self, pos):
        u = (pos + self.half) / self.dx - 0.5
        i0 = torch.floor(u)
        w1 = u - i0
        i0 = i0.long().clamp(1, self.G - 3)
        return i0, w1

    def _weights(self, w1):
        w0 = 1.0 - w1
        o = self.corners.to(w1.dtype)                       # (8, 3)
        w = (w0[:, None, :] * (1 - o)[None] + w1[:, None, :] * o[None]).prod(dim=2)  # (N, 8)
        return w

    def deposit(self, phi, pos, amount, weight_fn=None):
        """Add `amount` (N,) at positions (N,3) into phi using cloud-in-cell weights.

        weight_fn (optional): maps the phi values at the 8 corner cells (N, 8) to extra weights (N, 8).
        Used by source_local_speed to multiply each share by c(phi_cell) / c_base.
        """
        i0, w1 = self._cell(pos)
        w = self._weights(w1) * amount[:, None]             # (N, 8)
        idx = i0[:, None, :] + self.corners[None]           # (N, 8, 3)
        view = phi[0, 0]
        if weight_fn is not None:
            w = w * weight_fn(view[idx[..., 0], idx[..., 1], idx[..., 2]])
        view.index_put_((idx[..., 0].reshape(-1), idx[..., 1].reshape(-1), idx[..., 2].reshape(-1)),
                        w.reshape(-1).to(view.dtype), accumulate=True)

    def value(self, phi, pos):
        i0, w1 = self._cell(pos)
        w = self._weights(w1)
        idx = i0[:, None, :] + self.corners[None]
        p = phi[0, 0]
        v = p[idx[..., 0], idx[..., 1], idx[..., 2]]        # (N, 8)
        return (w * v).sum(dim=1)

    def gradient(self, phi, pos):
        """Trilinear interpolation of the central-difference gradient. Returns (N, 3)."""
        i0, w1 = self._cell(pos)
        w = self._weights(w1)                               # (N, 8)
        idx = i0[:, None, :] + self.corners[None]           # (N, 8, 3)
        p = phi[0, 0]
        ix, iy, iz = idx[..., 0], idx[..., 1], idx[..., 2]
        gx = p[ix + 1, iy, iz] - p[ix - 1, iy, iz]
        gy = p[ix, iy + 1, iz] - p[ix, iy - 1, iz]
        gz = p[ix, iy, iz + 1] - p[ix, iy, iz - 1]
        g = torch.stack([gx, gy, gz], dim=-1) / (2.0 * self.dx)   # (N, 8, 3)
        return (w[..., None] * g).sum(dim=1)

    # -- Smooth clouds for moving bodies. Same weights for deposit and gather, so a body never pushes
    #    itself with the part of the field it has just deposited (as with CIC), but the cloud is
    #    smoother, so the grid "shakes" the source much less as it crosses cell faces.
    #      tsc   : triangular-shaped cloud, 3 x 3 x 3 cells (width tied to the cell size)
    #      gauss : Gaussian of fixed PHYSICAL width sigma (same body size on every grid), cut at 3 sigma,
    #              weights normalised per axis so the deposited amount is exact.
    def _offsets(self, R):
        key = f"_off{R}"
        if not hasattr(self, key):
            r = torch.arange(-R, R + 1, device=self.device)
            ox, oy, oz = torch.meshgrid(r, r, r, indexing="ij")
            setattr(self, key, torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], dim=1))
        return getattr(self, key)

    def _stencil(self, pos, kernel, sigma=0.0):
        u = (pos + self.half) / self.dx - 0.5
        ic = torch.round(u)
        if kernel == "tsc":
            R = 1
            d = u - ic                                       # (N, 3) in [-0.5, 0.5]
            w1d = torch.stack([0.5 * (0.5 - d) ** 2, 0.75 - d ** 2, 0.5 * (0.5 + d) ** 2], dim=-1)
        elif kernel == "gauss":
            s = max(float(sigma), 1e-6) / self.dx            # width in cells
            R = max(1, int(math.ceil(3.0 * s)))
            r = torch.arange(-R, R + 1, device=pos.device, dtype=pos.dtype)
            d = (ic[..., None] + r) - u[..., None]           # (N, 3, 2R+1) cell offsets from the body
            w1d = torch.exp(-0.5 * (d / s) ** 2)
            w1d = w1d / w1d.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"unknown body kernel {kernel!r}")
        ic = ic.long().clamp(R + 1, self.G - R - 2)
        o = self._offsets(R)
        w = w1d[:, 0, o[:, 0] + R] * w1d[:, 1, o[:, 1] + R] * w1d[:, 2, o[:, 2] + R]   # (N, P)
        idx = ic[:, None, :] + o[None]                                                   # (N, P, 3)
        return idx, w

    def deposit_smooth(self, phi, pos, amount, kernel, sigma=0.0, weight_fn=None):
        idx, w = self._stencil(pos, kernel, sigma)
        w = w * amount[:, None]
        view = phi[0, 0]
        if weight_fn is not None:
            w = w * weight_fn(view[idx[..., 0], idx[..., 1], idx[..., 2]])
        view.index_put_((idx[..., 0].reshape(-1), idx[..., 1].reshape(-1), idx[..., 2].reshape(-1)),
                        w.reshape(-1).to(view.dtype), accumulate=True)

    def gradient_smooth(self, phi, pos, kernel, sigma=0.0):
        idx, w = self._stencil(pos, kernel, sigma)
        p = phi[0, 0]
        ix, iy, iz = idx[..., 0], idx[..., 1], idx[..., 2]
        gx = p[ix + 1, iy, iz] - p[ix - 1, iy, iz]
        gy = p[ix, iy + 1, iz] - p[ix, iy - 1, iz]
        gz = p[ix, iy, iz + 1] - p[ix, iy, iz - 1]
        g = torch.stack([gx, gy, gz], dim=-1) / (2.0 * self.dx)
        return (w[..., None] * g).sum(dim=1)


# ============================================================================ helpers
def source_per_tick(s: SimSettings, mass):
    """Per-tick deposit that gives a settled far field phi = G m / r (MS38 sources)."""
    return 4.0 * math.pi * s.c_base ** 2 * s.dt ** 2 * s.source_strength * mass / s.dx ** 3


def circular_speed(s: SimSettings, central_mass, r):
    """Newtonian circular speed around a well, using G_N = K * G (ignores the c^2(phi) bending)."""
    gm = s.field_force * s.source_strength * central_mass
    return math.sqrt(max(gm, 0.0) / r) if r > 0 else 0.0


def gaussian_bump(G, dx, centre, amplitude, width, device, dtype):
    half = G * dx / 2.0
    x = -half + (torch.arange(G, device=device, dtype=dtype) + 0.5) * dx
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    r2 = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2
    return (amplitude * torch.exp(-r2 / (2.0 * width ** 2))).reshape(1, 1, G, G, G)


class EngineCancelled(Exception):
    pass


# ============================================================================ engine
class Engine:
    def __init__(self, settings: SimSettings, device: Optional[str] = None):
        self.s = settings
        if device is None:
            device, self.device_note = resolve_device(settings.hardware)
        else:
            self.device_note = ""
        self.device = torch.device(device)
        self.dtype = torch.float32

    @torch.no_grad()
    def run(self, entities: list[Entity],
            progress: Optional[Callable[[float, str], bool]] = None) -> RunResult:
        s = self.s
        errs = s.check()
        if errs:
            raise ValueError("\n".join(errs))
        t_wall = time.time()
        dev, dt_ = self.device, self.dtype
        G, dx, dt, T = s.grid_resolution, s.dx, s.dt, s.total_ticks
        C = s.c_base
        half = G * dx / 2.0
        edge = half - 1.5 * dx

        field_ = FieldSolver(s, dev, dt_)
        samp = Sampler(G, dx, dev, dt_)

        defects = [e for e in entities if e.kind == "defect"]
        bodies = [e for e in entities if e.kind != "defect"]
        N = len(bodies)

        # --- starting field: wave defects (zero starting velocity)
        for e in defects:
            bump = gaussian_bump(G, dx, (e.x, e.y, e.z), e.amplitude, max(e.width, 0.5 * dx), dev, dt_)
            field_.phi += bump
            field_.phi_prev += bump

        # --- obstacle mask for field barriers (anchors)
        if s.obstacle_anchors:
            anchors = [e for e in entities if e.kind == "anchor"]
            if anchors:
                obs = torch.ones((1, 1, G, G, G), device=dev, dtype=dt_)
                for a in anchors:
                    ix = int((a.x + half) / dx)
                    iy = int((a.y + half) / dx)
                    iz = int((a.z + half) / dx)
                    if 0 <= ix < G and 0 <= iy < G and 0 <= iz < G:
                        obs[0, 0, max(0, ix-1):min(G, ix+2), max(0, iy-1):min(G, iy+2), max(0, iz-1):min(G, iz+2)] = 0.0
                field_.obstacle_mask = obs
                field_.phi *= obs
                field_.phi_prev *= obs

        # --- body arrays
        kinds = [e.kind for e in bodies]
        def mask(kset):
            return torch.tensor([k in kset for k in kinds], device=dev, dtype=torch.bool)
        if N:
            pos = torch.tensor([[e.x, e.y, e.z] for e in bodies], device=dev, dtype=dt_)
            mass = torch.tensor([max(e.mass, 1e-6) for e in bodies], device=dev, dtype=dt_)
            v0 = torch.tensor([[e.vx, e.vy, e.vz] for e in bodies], device=dev, dtype=dt_)
            speed = v0.norm(dim=1, keepdim=True).clamp(max=0.999 * C)
            v0 = torch.where(v0.norm(dim=1, keepdim=True) > 0,
                             v0 / v0.norm(dim=1, keepdim=True).clamp(min=1e-12) * speed, v0)
            gam0 = 1.0 / torch.sqrt(1.0 - (speed / C) ** 2)
            mom = gam0 * mass[:, None] * v0
            theta = torch.tensor([e.phase for e in bodies], device=dev, dtype=dt_)
            moving = mask(MOVING_KINDS)
            is_src = mask(SOURCE_KINDS)
            in_pauli = mask(PAULI_KINDS)
            alive = torch.ones(N, device=dev, dtype=torch.bool)
            if s.is_legacy:
                q = s.legacy_source * mass / mass.max().clamp(min=1e-8)
            else:
                q = torch.tensor([source_per_tick(s, float(m)) for m in mass.tolist()], device=dev, dtype=dt_)
            q = q * is_src.to(dt_)
            pauli_pair = (in_pauli[:, None] & in_pauli[None, :]).to(dt_)
            # legacy: 2600 seeds a Gaussian bump (amplitude m/m_max, width 3 cells) at every body
            if s.is_legacy and s.legacy_seed:
                mmax = float(mass.max())
                for e in bodies:
                    bump = gaussian_bump(G, dx, (e.x, e.y, e.z), max(e.mass, 0.0) / (mmax + 1e-8), 3 * dx, dev, dt_)
                    field_.phi += bump
                field_.phi_prev.copy_(field_.phi)
        else:
            pos = torch.zeros((0, 3), device=dev, dtype=dt_)

        # --- recording
        rec_every = max(1, int(s.record_every))
        n_rec = T // rec_every + 2
        frame_every = rec_every * max(1, math.ceil(n_rec / max(1, s.max_field_frames)))
        L = {k: [] for k in ("t", "field_kinetic", "field_potential", "field_energy",
                             "particle_kinetic", "total_energy", "peak_phi", "horizon_radius")}
        P, V, A, Gam, Thetas = [], [], [], [], []
        frames, frame_t = [], []
        zc = G // 2
        cell_vol = dx ** 3

        def gamma_of(m):
            return torch.sqrt(1.0 + (m ** 2).sum(dim=1) / (mass * C) ** 2)

        def record(tick):
            tt = tick * dt
            kin, pot = field_.energy()
            phi = field_.phi
            peak = float(phi.abs().max())
            if s.alpha > 0:
                n_h = int((s.alpha * phi.abs() >= s.horizon_threshold).sum())
                r_h = (3.0 * n_h * cell_vol / (4.0 * math.pi)) ** (1.0 / 3.0) if n_h else 0.0
            else:
                r_h = 0.0
            if N:
                g = gamma_of(mom)
                ke = float((((g - 1.0) * mass * C * C) * moving.to(dt_)).sum())
                P.append(pos.cpu().numpy().astype(np.float64))
                V.append((mom / (g[:, None] * mass[:, None])).cpu().numpy().astype(np.float64))
                A.append(alive.cpu().numpy().copy())
                Gam.append(g.cpu().numpy().astype(np.float64))
                Thetas.append(theta.cpu().numpy().astype(np.float64))
            else:
                ke = 0.0
            L["t"].append(tt)
            L["field_kinetic"].append(kin)
            L["field_potential"].append(pot)
            L["field_energy"].append(kin + pot)
            L["particle_kinetic"].append(ke)
            L["total_energy"].append(kin + pot + ke)
            L["peak_phi"].append(peak)
            L["horizon_radius"].append(r_h)
            if tick % frame_every == 0 or tick == T:
                sl = phi[0, 0, :, :, zc - 1:zc + 1].mean(dim=2) if G % 2 == 0 else phi[0, 0, :, :, zc]
                frames.append(sl.cpu().numpy().astype(np.float32))
                frame_t.append(tt)
            return peak

        status, message = "completed", ""
        record(0)
        prog_every = max(1, T // 200)
        tick_done = 0
        K = s.field_force
        ramp_n = max(0, int(s.source_ramp_ticks))
        pauli_exp = (s.pauli_power + 1.0) / 2.0
        if s.source_local_speed and not s.is_legacy and s.alpha != 0.0:
            def src_weight(phi_cells):
                return torch.sqrt(field_.c2_of(phi_cells)[1]) / s.c_base
        else:
            src_weight = None
        centred = bool(s.time_centred_force) and not s.is_legacy
        smooth = N > 0 and s.body_kernel != "cic" and not s.is_legacy
        kern, sig = s.body_kernel, s.body_width
        if smooth:
            mov_rows = torch.nonzero(moving & is_src).flatten()     # active bodies only
            fix_rows = torch.nonzero(~(moving & is_src)).flatten()

        try:
            for tick in range(1, T + 1):
                # 1. field update
                field_.step()

                if N:
                    # 2. sources (into the new field), smooth switch-on
                    ramp = 1.0 if (ramp_n == 0 or tick >= ramp_n) else 0.5 * (1.0 - math.cos(math.pi * tick / ramp_n))
                    amt = q * alive.to(dt_) * ramp
                    if smooth:
                        if len(fix_rows):
                            samp.deposit(field_.phi, pos[fix_rows], amt[fix_rows], src_weight)
                        if len(mov_rows):
                            samp.deposit_smooth(field_.phi, pos[mov_rows], amt[mov_rows], kern, sig, src_weight)
                    else:
                        samp.deposit(field_.phi, pos, amt, src_weight)

                    # 3. particle motion (held still during warm-up)
                    if tick > s.warmup_ticks:
                        # after step(): phi_prev = phi^n (field at the bodies' time level), phi = phi^(n+1)
                        phi_f = field_.phi_prev if centred else field_.phi
                        grad = samp.gradient(phi_f, pos)
                        if smooth and len(mov_rows):
                            grad[mov_rows] = samp.gradient_smooth(phi_f, pos[mov_rows], kern, sig)
                        force = K * grad * (1.0 if s.is_legacy else mass[:, None])
                        if s.pauli_strength != 0.0 and N > 1:
                            diff = pos[:, None, :] - pos[None, :, :]
                            dist_sq = (diff ** 2).sum(dim=2) + 1e-6
                            coup = torch.cos(theta[:, None] - theta[None, :]) * pauli_pair
                            p_term = (s.pauli_strength * coup[..., None] * diff
                                      / dist_sq[..., None] ** pauli_exp)
                            if s.pauli_reach > 0.0:
                                dist = torch.sqrt(dist_sq)
                                p_term = p_term * torch.exp(-dist / s.pauli_reach)[..., None]
                            force = force + p_term.sum(dim=1)
                        go = (moving & alive).to(dt_)[:, None]
                        mom = mom + force * dt * go
                        g = gamma_of(mom)
                        vel = mom / (g[:, None] * mass[:, None])
                        pos = pos + vel * dt * go
                        out = (pos.abs() > edge).any(dim=1) & moving
                        alive = alive & ~out
                    g = gamma_of(mom)
                    theta = theta + (mass / g) * dt

                tick_done = tick
                if tick % rec_every == 0 or tick == T:
                    peak = record(tick)
                    if not math.isfinite(peak) or peak > s.blowup_limit:
                        status = "diverged"
                        message = (f"The field blew up at tick {tick} (peak |phi| = {peak:.3g}). "
                                   f"Try smaller masses, smaller alpha or a smaller time step.")
                        break
                if progress is not None and tick % prog_every == 0:
                    if progress(tick / T, f"tick {tick:,} / {T:,}") is False:
                        raise EngineCancelled()
        except EngineCancelled:
            status, message = "cancelled", f"Stopped by user at tick {tick_done:,}."

        if status == "completed":
            message = f"Finished {T:,} ticks."
        res = RunResult(
            settings=s.to_dict(),
            entities=[e.to_dict() for e in bodies] + [e.to_dict() for e in defects],
            kinds=kinds,
            t=np.asarray(L["t"]),
            pos=np.stack(P) if P else np.zeros((len(L["t"]), 0, 3)),
            vel=np.stack(V) if V else np.zeros((len(L["t"]), 0, 3)),
            alive=np.stack(A) if A else np.zeros((len(L["t"]), 0), dtype=bool),
            gamma=np.stack(Gam) if Gam else np.zeros((len(L["t"]), 0)),
            phase=np.stack(Thetas) if Thetas else np.zeros((len(L["t"]), 0)),
            telemetry={k: np.asarray(v) for k, v in L.items() if k != "t"},
            frame_t=np.asarray(frame_t),
            field_frames=np.stack(frames) if frames else np.zeros((0, G, G), np.float32),
            final_phi=field_.phi[0, 0].cpu().numpy().astype(np.float32),
            status=status,
            message=message,
            device=str(self.device),
            wall_seconds=time.time() - t_wall,
            ticks_done=tick_done,
        )
        return res
