"""Engine validation. Run:  python tests/validate_engine.py   (prints PASS / FAIL for each check)

1-3  Field update matches the user's own MS38 harness (Z:/TryTri/Manuscripts/38_Yang-Mills/field_harness.py)
     for: MS38 periodic, MS38 + sponge walls, Legacy TEGR 2600.
4    MS38 rule conserves energy (no sources, no sponge).
5    A gravity well settles to phi = G m / r (so the "source strength G" setting means what it says).
6    A tracer launched at circular speed stays in orbit around a well (wake-only gravity).
7    CPU and GPU give the same answer.
8    Legacy run with moving bodies finishes without blowing up.
9    Optional 'source uses local wave speed': identical to stock at alpha = 0, and with alpha = 0.1
     the horizon core settles to the same value for c2_min = 1 and 0.01 (not set by the guard).
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HARNESS_DIR = Path(r"Z:/TryTri/Manuscripts/38_Yang-Mills")

from core.engine import Engine, SimSettings, FieldSolver, circular_speed  # noqa: E402
from core.entities import Entity  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def noise(G, amp, seed=1234):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return amp * torch.randn((1, 1, G, G, G), generator=gen)


# ---------------------------------------------------------------- 1-3 harness match
def harness_match():
    if not (HARNESS_DIR / "field_harness.py").exists():
        print("[SKIP] harness not found at", HARNESS_DIR)
        return
    sys.path.insert(0, str(HARNESS_DIR))
    from field_harness import FieldHarness

    G, ticks = 32, 300
    cases = [
        ("MS38 periodic", dict(physics_rule="ms38", alpha=0.1, sponge_width=0), True,
         dict(form="lagrangian", periodic=True, e_alpha=0.1, e_decay_base=1.0, e_decay_gamma=0.0)),
        ("MS38 + sponge walls", dict(physics_rule="ms38", alpha=0.1, sponge_width=6, sponge_strength=0.35), False,
         dict(form="lagrangian", periodic=False, e_alpha=0.1, e_decay_base=1.0, e_decay_gamma=0.0,
              sponge_width=6, sponge_strength=0.35)),
        ("Legacy TEGR 2600", dict(physics_rule="legacy2600", alpha=0.1, decay_base=0.999, decay_gamma=0.5,
                                  legacy_periodic=True), True,
         dict(form="engine", periodic=True, e_alpha=0.1, e_decay_base=0.999, e_decay_gamma=0.5)),
    ]
    for name, sk, periodic, hk in cases:
        s = SimSettings(grid_resolution=G, dt=0.001, **sk)
        mine = FieldSolver(s, torch.device(DEV))
        if s.physics_rule == "ms38" and periodic:
            mine.periodic = True
        h = FieldHarness(G=G, dx=s.dx, dt=s.dt, c=s.c_base, decay=1.0, emergent=True,
                         device=torch.device(DEV), **hk)
        n0 = noise(G, 0.3)
        h.set_field(n0, n0)
        mine.phi = n0.to(DEV).clone(); mine.phi_prev = n0.to(DEV).clone()
        for _ in range(ticks):
            h.step(); mine.step()
        diff = (h.phi - mine.phi).abs().max().item()
        scale = h.phi.abs().max().item()
        report(f"Harness match — {name}", diff <= 1e-5 * max(1.0, scale),
               f"max difference {diff:.2e} after {ticks} ticks (field size {scale:.3f})")


# ---------------------------------------------------------------- 4 energy
def energy_conservation():
    for alpha in (0.0, 0.1):
        s = SimSettings(grid_resolution=32, dt=0.001, alpha=alpha, sponge_width=0)
        f = FieldSolver(s, torch.device(DEV))
        f.periodic = True
        n0 = noise(32, 1e-3).to(DEV)
        f.phi = n0.clone(); f.phi_prev = n0.clone()
        k0, p0 = f.energy(); E0 = k0 + p0
        Es = []
        for t in range(2000):
            f.step()
            if t % 100 == 99:
                k, p = f.energy(); Es.append((k + p) / E0)
        drift = max(abs(e - 1.0) for e in Es)
        report(f"Energy conserved (alpha={alpha})", drift < 5e-3, f"largest change {drift * 100:.3f}% over 2000 ticks")


# ---------------------------------------------------------------- 5 point source -> G m / r
def point_source():
    s = SimSettings(grid_resolution=64, dt=0.005, alpha=0.0, total_ticks=24000, source_ramp_ticks=4000,
                    warmup_ticks=0, source_strength=1.0, record_every=1000)
    M = 1.0
    res = Engine(s, DEV).run([Entity("well", 0, 0, 0, mass=M)])
    phi = res.final_phi.astype(np.float64)
    G, dx = s.grid_resolution, s.dx
    c = G // 2
    # cells along +x from the centre; centre sits between cells c-1 and c
    rows = []
    for i in range(c + 2, c + 14):
        r = (i + 0.5 - G / 2) * dx
        v = phi[i, c - 1:c + 1, c - 1:c + 1].mean()   # average of the 4 cells straddling the y=z=0 line
        rows.append((r, v))
    r = np.array([a for a, _ in rows]); v = np.array([b for _, b in rows])
    A = np.vstack([1.0 / r, np.ones_like(r)]).T
    (coef, const), *_ = np.linalg.lstsq(A, v, rcond=None)
    ratio = coef / (s.source_strength * M)
    report("Well settles to phi = G m / r", abs(ratio - 1.0) < 0.05,
           f"fitted G m = {coef:.4f} (expected {s.source_strength * M:.4f}, ratio {ratio:.3f}); "
           f"{res.wall_seconds:.1f}s on {res.device}")


# ---------------------------------------------------------------- 6 orbit
def orbit():
    s = SimSettings(grid_resolution=64, dt=0.005, alpha=0.0, total_ticks=18000, source_ramp_ticks=4000,
                    warmup_ticks=10000, source_strength=1.0, field_force=100.0, record_every=20)
    M, r0 = 30.0, 30.0
    v = circular_speed(s, M, r0)
    ents = [Entity("well", 0, 0, 0, mass=M), Entity("tracer", r0, 0, 0, vy=v, mass=1.0)]
    res = Engine(s, DEV).run(ents)
    free = res.t > s.warmup_ticks * s.dt
    if free.sum() < 2:
        report("Tracer orbits a well (wake only)", False, f"run stopped early: {res.message}")
        return
    xy = res.pos[free, 1, :2]
    r = np.hypot(xy[:, 0], xy[:, 1])
    ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    orbits = (ang[-1] - ang[0]) / (2 * math.pi)
    ok = res.status == "completed" and r.min() > 0.85 * r0 and r.max() < 1.15 * r0 and orbits > 0.9
    report("Tracer orbits a well (wake only)", ok,
           f"{orbits:.2f} orbits, r between {r.min():.2f} and {r.max():.2f} (started {r0}), v = {v:.2f}; "
           f"{res.wall_seconds:.1f}s")


# ---------------------------------------------------------------- 7 CPU vs GPU
def cpu_vs_gpu():
    if DEV == "cpu":
        print("[SKIP] no GPU for CPU-vs-GPU check")
        return
    s = SimSettings(grid_resolution=32, dt=0.001, field_force=1.0, total_ticks=400, warmup_ticks=0,
                    source_ramp_ticks=100, record_every=50)
    ents = [Entity("well", 0, 0, 0, mass=50.0), Entity("active", 15, 0, 0, vy=5, mass=2.0),
            Entity("defect", -10, 10, 0, amplitude=0.5, width=5)]
    a = Engine(s, "cpu").run(ents)
    b = Engine(s, "cuda").run(ents)
    d = np.abs(a.pos - b.pos).max()
    e = np.abs(a.final_phi - b.final_phi).max() / max(1e-9, np.abs(a.final_phi).max())
    report("CPU and GPU agree", d < 1e-3 and e < 1e-4, f"position difference {d:.2e}, field difference {e:.2e} (relative)")


# ---------------------------------------------------------------- 8 legacy smoke
def legacy_smoke():
    s = SimSettings(physics_rule="legacy2600", grid_resolution=48, dt=0.001, field_force=1.0, total_ticks=3000,
                    warmup_ticks=0, legacy_source=0.001, record_every=50)
    ents = [Entity("anchor", 0, 0, 0, mass=1.0), Entity("active", 20, 0, 0, vy=3, mass=0.5),
            Entity("tracer", -20, 5, 0, vx=2)]
    res = Engine(s, DEV).run(ents)
    ok = res.status == "completed" and np.isfinite(res.pos).all()
    report("Legacy 2600 run finishes", ok, f"{res.message} peak |phi| = {res.telemetry['peak_phi'].max():.3f}")


# ---------------------------------------------------------------- 9 local-speed source (opt-in)
def local_speed_source():
    dx = 2.8125
    well = [Entity("well", 0.5 * dx, 0.5 * dx, 0.5 * dx, mass=10.0)]
    # (a) alpha = 0: the option must change nothing at all
    s_off = SimSettings(alpha=0.0, total_ticks=3000, record_every=500)
    s_on = SimSettings(alpha=0.0, total_ticks=3000, record_every=500, source_local_speed=True)
    d = np.abs(Engine(s_off, DEV).run(well).final_phi - Engine(s_on, DEV).run(well).final_phi).max()
    report("Local-speed source: no change at alpha = 0", d == 0.0, f"max field difference {d:.2e}")
    # (b) alpha = 0.1: core must settle below the guard and not depend on it
    peaks, notes = [], []
    for fl in (1.0, 0.01):
        s = SimSettings(alpha=0.1, c2_min=fl, total_ticks=16000, record_every=100, source_local_speed=True)
        res = Engine(s, DEV).run(well)
        raw = s.c_base ** 2 / (1.0 + s.alpha * np.abs(res.final_phi.astype(np.float64))) ** 2
        peaks.append(float(res.telemetry["peak_phi"][-1]) if res.status == "completed" else float("inf"))
        notes.append(f"c2_min={fl:g}: {res.status}, peak {peaks[-1]:.4g}, cells at floor {int((raw < fl).sum())}")
    spread = abs(peaks[0] - peaks[1]) / max(peaks) if all(np.isfinite(peaks)) else float("inf")
    report("Local-speed source: horizon core not set by c2_min", spread < 0.01,
           f"{'; '.join(notes)}; spread {spread * 100:.3f}%")


if __name__ == "__main__":
    t0 = time.time()
    print(f"Device: {DEV}\n")
    harness_match()
    energy_conservation()
    point_source()
    orbit()
    cpu_vs_gpu()
    legacy_smoke()
    local_speed_source()
    print(f"\n{sum(results)}/{len(results)} checks passed in {time.time() - t0:.0f}s")
    sys.exit(0 if all(results) else 1)
