"""Grid check: is the self-push of a moving source real physics or a side effect of the grid?

Follow-up to diagnose.py, which found that a single star that FEEDS the field while it orbits gains
spin and spirals out, while a tracer (feels the field, never feeds it) on the same orbit does not.
The push grew with the star's mass. engine.py is NOT edited. alpha = 0 (plain gravity) only.

TWO TESTS, each repeated on four grids that cover the SAME physical box (360 units):
  coarse   G = 64,  dx = 5.625,   dt = 0.005
  base     G = 128, dx = 2.8125,  dt = 0.005     (the grid used so far)
  fine     G = 256, dx = 1.40625, dt = 0.005
  half_dt  G = 128, dx = 2.8125,  dt = 0.0025    (same grid, half the time step)

  1. FREE STAR: one active star (mass 4.67), no centre, nothing else in the box. Held still while its own
     well settles, then released in a straight line at 0.1, 0.2 or 0.4 of the wave speed along the
     x axis, and at 0.2 along the diagonal (to see if grid direction matters).
  2. ORBIT: the diagnose.py case. Centre well (mass 60) + one active star (mass 4.67) at r = 35,
     with a tracer on the same orbit on the opposite side as the control.

PREDICTION, WRITTEN BEFORE RUNNING
  Real physics (a wave field with a finite speed, no friction):
    - Free star: a source moving in a straight line at constant speed feels NO net force from its own
      field (the field just travels with it). Expect a small one-off slowdown at release (a few % at
      most: the star has to drag its own well along, which acts like a little extra inertia), then a
      constant speed. No sideways drift.
    - Orbit: an orbiting source radiates waves, which carry energy away, so it should spiral slowly
      INWARD (like binary pulsars). It should not gain spin.
  Grid side effect:
    - The push (or drag) changes size when dx or dt changes, and may depend on direction
      (x axis vs diagonal). The free star speeds up or slows down steadily even with nothing else around.
  So: if the result changes from grid to grid, it is the grid, not the physics.

Outputs: results/grid_check.json and results/grid_check.png
"""
import json
import math
import os

import numpy as np

import spinning_disk as sd
from spinning_disk import gr, Entity, SimSettings, curve_from_slice, relativistic_speed

OUT = sd.OUT
BOX = 360.0
C = 65.0
M_STAR = 4.67
M_CORE = 60.0
R_ORBIT = 35.0
WARM_T = 30.0          # physical time held still (6000 ticks at dt = 0.005)
FREE_T = 7.5           # physical time of free flight after release (0.4c path stays inside |x| < 100)
ORBIT_T = 30.0         # physical time of orbiting after release

GRIDS = {
    "coarse":  dict(grid_resolution=64,  dx=BOX / 64,  dt=0.005),
    "base":    dict(grid_resolution=128, dx=BOX / 128, dt=0.005),
    "fine":    dict(grid_resolution=256, dx=BOX / 256, dt=0.005),
    "half_dt": dict(grid_resolution=128, dx=BOX / 128, dt=0.0025),
}
FREE_CASES = [("x 0.1c", 0.1, 0.0), ("x 0.2c", 0.2, 0.0), ("x 0.4c", 0.4, 0.0), ("diag 0.2c", 0.2, 45.0)]


def settings(g, after_t):
    dt = g["dt"]
    warm = int(round(WARM_T / dt))
    return SimSettings(grid_resolution=g["grid_resolution"], dx=g["dx"], dt=dt, alpha=0.0,
                       sponge_width=g["grid_resolution"] * 12 // 128,   # same physical edge layer (33.75 units)
                       warmup_ticks=warm, total_ticks=warm + int(round(after_t / dt)),
                       source_ramp_ticks=int(round(15.0 / dt)),
                       record_every=max(1, int(round(0.05 / dt))), max_field_frames=4)


def after_release(res, s, j):
    t = res.t
    rel = int(np.searchsorted(t, s.warmup_ticks * s.dt - 1e-9))
    return t[rel:] - t[rel], res.pos[rel:, j], res.vel[rel:, j], res.gamma[rel:, j], res.alive[rel:, j]


def free_star(name, s, frac, ang_deg):
    v = frac * C
    a = math.radians(ang_deg)
    ux, uy = math.cos(a), math.sin(a)
    half_path = 0.5 * v * FREE_T
    ent = Entity("active", x=-half_path * ux, y=-half_path * uy, z=0.0, vx=v * ux, vy=v * uy,
                 mass=M_STAR, label="free")
    res, _ = gr.run(name, s, [ent])
    t, p, vel, g, ok = after_release(res, s, 0)
    t, vel = t[ok], vel[ok]
    along = vel[:, 0] * ux + vel[:, 1] * uy
    side = -vel[:, 0] * uy + vel[:, 1] * ux
    late = t >= 2.0
    slope = float(np.polyfit(t[late], along[late], 1)[0]) if late.sum() > 3 else float("nan")
    i1 = int(np.searchsorted(t, 1.0))
    return {"t": t.tolist(), "v_over_v0": (along / v).tolist(),
            "release_change_pct": float(100 * (along[i1] / v - 1.0)),
            "steady_accel": slope,                         # units of speed per time unit
            "steady_accel_pct_per_t": float(100 * slope / v),
            "end_change_pct": float(100 * (along[-1] / v - 1.0)),
            "max_side_over_v0": float(np.abs(side).max() / v)}


def orbit(name, s):
    res_core, _ = gr.run(f"{name} pre_core", s, [Entity("well", 0.0, 0.0, 0.0, mass=M_CORE)])
    w = curve_from_slice(res_core.field_frames[-1], s, np.array([R_ORBIT]))
    v = float(relativistic_speed(w)[0])
    ents = [Entity("well", 0.0, 0.0, 0.0, mass=M_CORE)]
    for kind, th in (("active", 0.0), ("tracer", math.pi)):
        ents.append(Entity(kind, x=R_ORBIT * math.cos(th), y=R_ORBIT * math.sin(th), z=0.0,
                           vx=-v * math.sin(th), vy=v * math.cos(th), mass=M_STAR, label=kind))
    res, _ = gr.run(f"{name} orbit", s, ents)
    out = {"v_launch": v}
    for j, kind in ((1, "active"), (2, "tracer")):
        t, p, vel, g, ok = after_release(res, s, j)
        rr = np.hypot(p[:, 0], p[:, 1]) / R_ORBIT
        L = g * (p[:, 0] * vel[:, 1] - p[:, 1] * vel[:, 0])
        out[kind] = {"t": t.tolist(), "r_over_r0": rr.tolist(),
                     "r_min": float(rr[ok].min()), "r_max": float(rr[ok].max()), "alive_end": bool(ok[-1]),
                     "L_change_pct": float(100 * (L[ok][-1] / L[0] - 1.0))}
    return out


def main():
    out = {"free": {}, "orbit": {}}
    for gname, g in GRIDS.items():
        s_free = settings(g, FREE_T)
        out["free"][gname] = {}
        for label, frac, ang in FREE_CASES:
            r = free_star(f"{gname} free {label}", s_free, frac, ang)
            out["free"][gname][label] = r
            print(f"    {gname:<8} free {label:<10} release {r['release_change_pct']:+.2f}% | steady "
                  f"{r['steady_accel_pct_per_t']:+.3f}% of speed per time unit | end {r['end_change_pct']:+.2f}% | "
                  f"max sideways {100 * r['max_side_over_v0']:.2f}%", flush=True)
        o = orbit(gname, settings(g, ORBIT_T))
        out["orbit"][gname] = o
        for kind in ("active", "tracer"):
            k = o[kind]
            print(f"    {gname:<8} orbit {kind:<7} r within {k['r_min']:.3f} - {k['r_max']:.3f} | spin change "
                  f"{k['L_change_pct']:+.1f}% | alive at end {k['alive_end']}", flush=True)
        with open(os.path.join(OUT, "grid_check.json"), "w") as f:
            json.dump(out, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"coarse": "tab:blue", "base": "tab:red", "fine": "tab:green", "half_dt": "tab:orange"}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for gname, d in out["free"].items():
        for label, ls in (("x 0.2c", "-"), ("diag 0.2c", "--")):
            ax[0].plot(d[label]["t"], d[label]["v_over_v0"], color=cols[gname], ls=ls, label=f"{gname} {label}")
        ax[1].plot([0.1, 0.2, 0.4], [d[l]["steady_accel_pct_per_t"] for l in ("x 0.1c", "x 0.2c", "x 0.4c")],
                   "o-", color=cols[gname], label=gname)
    ax[0].axhline(1.0, color="k", ls=":")
    ax[0].set_title("Free star (no centre): speed / launch speed"); ax[0].set_xlabel("time since release")
    ax[0].legend(fontsize=7)
    ax[1].axhline(0.0, color="k", ls=":")
    ax[1].set_title("Free star: steady speed-up (+) or slow-down (-), % per time unit")
    ax[1].set_xlabel("launch speed / wave speed"); ax[1].legend(fontsize=7)
    for gname, o in out["orbit"].items():
        ax[2].plot(o["active"]["t"], o["active"]["r_over_r0"], color=cols[gname], label=f"{gname} active")
        ax[2].plot(o["tracer"]["t"], o["tracer"]["r_over_r0"], color=cols[gname], ls=":", label=f"{gname} tracer")
    ax[2].axhline(1.0, color="k", ls=":")
    ax[2].set_title("Orbit at r = 35, star mass 4.67 (solid = feeds field, dotted = tracer)")
    ax[2].set_xlabel("time since release"); ax[2].set_ylabel("r / r_start"); ax[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "grid_check.png"), dpi=160)
    print("\nSaved results/grid_check.png and results/grid_check.json")


if __name__ == "__main__":
    main()
