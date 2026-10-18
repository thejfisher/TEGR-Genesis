"""Galaxy rotation curve test - TEGR Genesis (stock 7-point engine; engine.py is NOT edited).

QUESTION
  Does the Genesis field pull on the outer part of a disk galaxy harder than its visible mass
  predicts? That extra pull is the "dark matter" signature (flat rotation curves).

PREDICTION, WRITTEN BEFORE RUNNING (from the engine's own equations)
  alpha = 0 : the settled field obeys plain Poisson, so the rotation curve must equal the
              Newtonian curve of the visible mass. Any excess is a box / sponge effect, not physics.
  alpha > 0 with the source scaled by c/c_base (Fabbri rule):
              u = integral of c dphi obeys Poisson, so phi = (exp(alpha*phiN) - 1) / alpha and
              grad phi = exp(alpha*phiN) * grad phiN.  The pull is boosted by exp(alpha*phiN).
              phiN fades like 1/r, so the boost is strongest in the INNER disk and fades outward.
              So we do NOT expect flat outer curves from a still (static) field.

RUNS (all galaxies are identical: fixed wells, same random seed)
  A0      alpha = 0,   G = 128 (box 360 units)   visible-mass baseline
  A0_box  alpha = 0,   G = 96  (box 270 units)   box-size control
  A1      alpha = 0.1, source x c/c_base, c2_min = 0.01, G = 128

TRACERS ride along in every run. They feel the field but do not feed it. Each is launched at the
predicted circular speed. If the prediction is right, they stay on circles.

Outputs: results/galaxy_rotation.json and results/galaxy_rotation.png
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
GENESIS = os.path.abspath(os.path.join(HERE, "..", ".."))
if GENESIS not in sys.path:
    sys.path.insert(0, GENESIS)

from core.engine import Engine, SimSettings   # noqa: E402
from core.entities import Entity             # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- galaxy
SEED = 7
BULGE_N, BULGE_M, BULGE_A, BULGE_RMAX = 30, 60.0, 5.0, 15.0          # Plummer bulge
DISK_N, DISK_M, DISK_RD, DISK_RMIN, DISK_RMAX = 300, 140.0, 18.0, 3.0, 75.0   # exponential disk
ALPHA = 0.1
TICKS = 16000
R_MEASURE = np.arange(6.0, 121.0, 3.0)      # physical units, in the disk plane
N_ANG = 72
TRACER_R = [15.0, 25.0, 35.0, 50.0, 65.0, 80.0, 95.0, 110.0]
GOLDEN = math.radians(137.507764)


def build_galaxy():
    rng = np.random.default_rng(SEED)
    pts, ms = [], []
    n = 0
    while n < BULGE_N:                       # Plummer sphere, clipped
        u = rng.uniform(0.01, 0.99)
        r = BULGE_A / math.sqrt(u ** (-2.0 / 3.0) - 1.0)
        if r > BULGE_RMAX:
            continue
        v = rng.normal(size=3)
        pts.append(v / np.linalg.norm(v) * r)
        ms.append(BULGE_M / BULGE_N)
        n += 1
    n = 0
    while n < DISK_N:                        # surface density ~ exp(-R/Rd)  ->  R ~ Gamma(2, Rd)
        R = rng.gamma(2.0, DISK_RD)
        if not (DISK_RMIN <= R <= DISK_RMAX):
            continue
        th = rng.uniform(0.0, 2.0 * math.pi)
        pts.append(np.array([R * math.cos(th), R * math.sin(th), 0.0]))
        ms.append(DISK_M / DISK_N)
        n += 1
    return np.array(pts), np.array(ms)


def ring_points(r, n_ang=N_ANG):
    th = np.linspace(0.0, 2.0 * math.pi, n_ang, endpoint=False)
    return np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)], axis=1)


def newton(pts, ms, G_src, eps, P):
    """Visible-mass Newtonian potential and gradient at points P (softened by eps)."""
    d = P[:, None, :] - pts[None]
    d2 = (d ** 2).sum(-1) + eps ** 2
    phi = G_src * (ms[None] / np.sqrt(d2)).sum(1)
    grad = -G_src * (ms[None, :, None] * d / d2[..., None] ** 1.5).sum(1)
    return phi, grad


def newton_curve(pts, ms, s):
    """Ring-averaged phiN and v^2 = -K r <grad phiN . r_hat>, plus the alpha-boosted prediction."""
    K, Gs, eps = s.field_force, s.source_strength, s.dx
    v2N, v2A = [], []
    for r in R_MEASURE:
        P = ring_points(r)
        phi, grad = newton(pts, ms, Gs, eps, P)
        gr = (grad * P / r).sum(1)
        v2N.append(-K * r * gr.mean())
        v2A.append(-K * r * (gr * np.exp(ALPHA * phi)).mean())
    return np.sqrt(np.maximum(v2N, 0)), np.sqrt(np.maximum(v2A, 0))


def grid_curve(phi3, s, alpha_boost=0.0):
    """Rotation curve measured from the engine's settled field (z = 0 plane, ring average).

    alpha_boost > 0: also return the prediction v^2 * exp(alpha * phi) built from THIS field
    (used to predict A1 from the A0 field, which already contains the same box effects).
    """
    G, dx, K = s.grid_resolution, s.dx, s.field_force
    L = G * dx
    phi3 = phi3.astype(np.float64)
    gx, gy, _ = np.gradient(phi3, dx)
    v, vb = [], []
    for r in R_MEASURE:
        P = ring_points(r)
        idx = (P.T + L / 2.0) / dx - 0.5
        ph = map_coordinates(phi3, idx, order=3, mode="nearest")
        ax = map_coordinates(gx, idx, order=3, mode="nearest")
        ay = map_coordinates(gy, idx, order=3, mode="nearest")
        gr = (ax * P[:, 0] + ay * P[:, 1]) / r
        v.append(-K * r * gr.mean())
        vb.append(-K * r * (gr * np.exp(alpha_boost * ph)).mean())
    return np.sqrt(np.maximum(v, 0)), np.sqrt(np.maximum(vb, 0))


def outer_slope(v, rmin=80.0, rmax=115.0):
    """d log v / d log r beyond the disk edge. Keplerian = -0.5, flat = 0."""
    m = (R_MEASURE >= rmin) & (R_MEASURE <= rmax) & (v > 0)
    return float(np.polyfit(np.log(R_MEASURE[m]), np.log(v[m]), 1)[0])


def make_entities(pts, ms, tracer_v, rmax_tracer):
    ents = [Entity("well", x=p[0], y=p[1], z=p[2], mass=m) for p, m in zip(pts, ms)]
    for k, (r, vt) in enumerate(zip(TRACER_R, tracer_v)):
        if r > rmax_tracer:
            continue
        th = k * GOLDEN
        ents.append(Entity("tracer", x=r * math.cos(th), y=r * math.sin(th), z=0.0,
                           vx=-vt * math.sin(th), vy=vt * math.cos(th), mass=1.0, label=f"tracer r={r:g}"))
    return ents


def tracer_report(res, s):
    rows = []
    t = res.t
    go = t > s.warmup_ticks * s.dt
    for j, k in enumerate(res.kinds):
        if k != "tracer":
            continue
        p = res.pos[:, j, :]
        rr = np.hypot(p[:, 0], p[:, 1])
        a = res.alive[:, j] & go
        r0 = float(rr[0])
        if a.sum() < 2:
            rows.append({"r0": r0, "alive_frac": 0.0})
            continue
        rows.append({
            "r0": r0,
            "r_min_ratio": float(rr[a].min() / r0),
            "r_max_ratio": float(rr[a].max() / r0),
            "alive_frac": float(a.sum() / go.sum()),
            "t": t[go].tolist(), "r_over_r0": (rr[go] / r0).tolist(),
        })
    return rows


def run(name, s, ents):
    t0 = time.time()
    print(f"\n--- {name}: G={s.grid_resolution}, alpha={s.alpha}, c/c_base source={s.source_local_speed}, "
          f"{len(ents)} entities ---", flush=True)
    res = Engine(s).run(ents)
    ff, ft = res.field_frames, res.frame_t
    j = int(np.argmin(np.abs(ft - (ft[-1] - 2000 * s.dt))))
    drift = float(np.abs(ff[-1] - ff[j]).max() / max(np.abs(ff[-1]).max(), 1e-12))
    phi = res.final_phi.astype(np.float64)
    raw = s.c_base ** 2 / (1.0 + s.alpha * np.abs(phi)) ** 2
    info = {"status": res.status, "message": res.message, "device": res.device,
            "wall_seconds": round(time.time() - t0, 1), "drift_last_2000_ticks": drift,
            "peak_phi": float(np.abs(phi).max()),
            "cells_at_floor": int((raw < s.c2_min).sum()) if s.alpha > 0 else 0}
    print(f"    {res.status} on {res.device} in {info['wall_seconds']} s | settled drift {100 * drift:.3f}% | "
          f"peak phi {info['peak_phi']:.3g} | cells at speed floor {info['cells_at_floor']}", flush=True)
    return res, info


def main():
    pts, ms = build_galaxy()
    print(f"Galaxy: bulge {BULGE_M:g} + disk {DISK_M:g} = {ms.sum():g} total mass in {len(ms)} fixed wells")

    common = dict(total_ticks=TICKS, record_every=100, max_field_frames=80)
    s0 = SimSettings(grid_resolution=128, alpha=0.0, **common)
    sb = SimSettings(grid_resolution=96, alpha=0.0, **common)
    s1 = SimSettings(grid_resolution=128, alpha=ALPHA, source_local_speed=True, c2_min=0.01, **common)

    vN, vA_pred_newton = newton_curve(pts, ms, s0)
    tv0 = np.interp(TRACER_R, R_MEASURE, vN)
    tv1 = np.interp(TRACER_R, R_MEASURE, vA_pred_newton)

    interior = lambda s: (s.grid_resolution / 2 - s.sponge_width) * s.dx - 5.0
    runs = {}
    for name, s, tv in (("A0", s0, tv0), ("A0_box", sb, tv0), ("A1", s1, tv1)):
        res, info = run(name, s, make_entities(pts, ms, tv, interior(s)))
        v_meas, v_boost = grid_curve(res.final_phi, s, alpha_boost=ALPHA)
        runs[name] = {"settings": s.to_dict(), **info, "v_measured": v_meas.tolist(),
                      "v_boost_from_this_field": v_boost.tolist(),
                      "tracers": tracer_report(res, s)}

    vA0 = np.array(runs["A0"]["v_measured"])
    vB = np.array(runs["A0_box"]["v_measured"])
    vA1 = np.array(runs["A1"]["v_measured"])

    # A1 tracers again, launched at the speed the engine measured (the infinite-space prediction
    # ignores that the finite box lowers the field level, so it launches them ~4% too fast)
    res, info = run("A1_retrace", s1, make_entities(pts, ms, np.interp(TRACER_R, R_MEASURE, vA1), interior(s1)))
    runs["A1_retrace"] = {"settings": s1.to_dict(), **info, "tracers": tracer_report(res, s1)}
    vA1_pred = np.array(runs["A0"]["v_boost_from_this_field"])   # prediction for A1 built from A0 field

    def band(a, b, lo, hi):
        m = (R_MEASURE >= lo) & (R_MEASURE <= hi)
        q = a[m] / b[m]
        return float(q.min()), float(q.max())

    verdict = {
        "A0_over_newton_r10_100": band(vA0, vN, 10, 100),
        "A0box_over_A0_r10_90": band(vB, vA0, 10, 90),
        "A1_over_newton_r10_100": band(vA1, vN, 10, 100),
        "A1_over_prediction_r10_100": band(vA1, vA1_pred, 10, 100),
        "outer_slope_newton": outer_slope(vN),
        "outer_slope_A0": outer_slope(vA0),
        "outer_slope_A1": outer_slope(vA1),
    }
    summary = {"galaxy": {"bulge_mass": BULGE_M, "disk_mass": DISK_M, "n_wells": int(len(ms)),
                          "disk_scale_length": DISK_RD, "disk_edge": DISK_RMAX, "seed": SEED},
               "r": R_MEASURE.tolist(), "v_newton_visible": vN.tolist(),
               "v_alpha_prediction_direct_sum": vA_pred_newton.tolist(),
               "v_A1_prediction_from_A0_field": vA1_pred.tolist(),
               "runs": runs, "verdict": verdict}
    with open(os.path.join(OUT, "galaxy_rotation.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("\n=== RESULTS ===")
    print(f"A0 (alpha=0) / visible-mass Newton, r=10-100:   {verdict['A0_over_newton_r10_100'][0]:.3f} - "
          f"{verdict['A0_over_newton_r10_100'][1]:.3f}   (1.000 = no extra pull)")
    print(f"Box control G=96 / G=128, r=10-90:              {verdict['A0box_over_A0_r10_90'][0]:.3f} - "
          f"{verdict['A0box_over_A0_r10_90'][1]:.3f}")
    print(f"A1 (alpha=0.1, Fabbri source) / Newton:         {verdict['A1_over_newton_r10_100'][0]:.3f} - "
          f"{verdict['A1_over_newton_r10_100'][1]:.3f}")
    print(f"A1 / prediction exp(alpha*phi) from A0 field:   {verdict['A1_over_prediction_r10_100'][0]:.3f} - "
          f"{verdict['A1_over_prediction_r10_100'][1]:.3f}")
    print(f"Outer slope d(log v)/d(log r), r=80-115 (Kepler -0.5, flat 0): Newton {verdict['outer_slope_newton']:+.3f} | "
          f"A0 {verdict['outer_slope_A0']:+.3f} | A1 {verdict['outer_slope_A1']:+.3f}")
    for name in runs:
        for tr in runs[name]["tracers"]:
            if tr.get("alive_frac", 0) > 0:
                print(f"  {name} tracer r0={tr['r0']:6.1f}: r stays within {tr['r_min_ratio']:.3f} - "
                      f"{tr['r_max_ratio']:.3f} of start (alive {100 * tr['alive_frac']:.0f}%)")

    # ------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(R_MEASURE, vN, "k-", lw=2, label="Newton, visible mass only")
    ax[0].plot(R_MEASURE, vA0, "o", ms=4, color="tab:blue", label="engine, alpha=0 (G=128)")
    ax[0].plot(R_MEASURE, vB, "x", ms=4, color="tab:gray", label="engine, alpha=0 (G=96 box control)")
    ax[0].plot(R_MEASURE, vA1_pred, "--", color="tab:red", label="prediction: alpha=0.1, exp(alpha phi) boost")
    ax[0].plot(R_MEASURE, vA1, "s", ms=4, color="tab:red", label="engine, alpha=0.1 + Fabbri source")
    ax[0].axvline(DISK_RMAX, color="0.6", ls=":", lw=1)
    ax[0].text(DISK_RMAX + 1, ax[0].get_ylim()[1] * 0.92, "disk edge", fontsize=8, color="0.4")
    ax[0].set_xlabel("radius (sim units)"); ax[0].set_ylabel("circular speed")
    ax[0].set_title("Rotation curve"); ax[0].legend(fontsize=7)
    ax[1].plot(R_MEASURE, vA0 / vN, "o-", ms=3, color="tab:blue", label="alpha=0 / Newton")
    ax[1].plot(R_MEASURE, vA1 / vN, "s-", ms=3, color="tab:red", label="alpha=0.1 / Newton")
    ax[1].plot(R_MEASURE, vA1 / vA1_pred, "^-", ms=3, color="tab:green", label="alpha=0.1 / its prediction")
    ax[1].axhline(1.0, color="k", ls=":")
    ax[1].set_xlabel("radius"); ax[1].set_title("Speed ratio (1 = no extra pull)"); ax[1].legend(fontsize=7)
    for name, col in (("A0", "tab:blue"), ("A1_retrace", "tab:red")):
        for tr in runs[name]["tracers"]:
            if tr.get("alive_frac", 0) > 0:
                ax[2].plot(tr["t"], tr["r_over_r0"], color=col, lw=0.8)
    ax[2].axhline(1.0, color="k", ls=":")
    ax[2].set_xlabel("time"); ax[2].set_ylabel("r / r_start")
    ax[2].set_title("Tracer orbits (blue alpha=0, red alpha=0.1 re-launched)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "galaxy_rotation.png"), dpi=170)
    print("\nSaved results/galaxy_rotation.png and results/galaxy_rotation.json")


if __name__ == "__main__":
    main()
