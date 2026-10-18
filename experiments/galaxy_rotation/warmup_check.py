"""Warm-up check for the galaxy test (follow-up to galaxy_rotation.py).

Question: the alpha = 0.1 test stars drifted outward even at the measured circular speed.
Guess: the alpha = 0.1 field is still building up when the stars are released (tick 6000),
so the pull at release is weaker than the final pull they were launched for.

This script checks that guess directly, then repeats the orbit test with a long warm-up.
  1. Rotation curve vs time, read from the z = 0 field slices the engine saves.
     -> how much weaker is the pull at tick 6000 than at the end?
  2. Same galaxy, alpha = 0.1, warm-up 6000 vs 20000 ticks (10000 ticks of orbit after release).
     Stars launched at the measured final alpha = 0.1 speed (from results/galaxy_rotation.json).
  3. alpha = 0 with the long warm-up too, as a control.

Outputs: results/warmup_check.json and results/warmup_check.png
"""
import json
import math
import os

import numpy as np
from scipy.ndimage import map_coordinates

import galaxy_rotation as gr
from core.engine import Engine, SimSettings   # path set up by galaxy_rotation

OUT = gr.OUT
ORBIT_TICKS = 10000
R_CHECK = np.array(gr.TRACER_R)


def curve_from_slice(sl, s, R):
    """Ring-averaged circular speed from a z = 0 slice of phi (G, G)."""
    G, dx, K = s.grid_resolution, s.dx, s.field_force
    L = G * dx
    sl = sl.astype(np.float64)
    gx, gy = np.gradient(sl, dx)
    out = []
    for r in R:
        P = gr.ring_points(r)
        idx = (P[:, :2].T + L / 2.0) / dx - 0.5
        ax = map_coordinates(gx, idx, order=3, mode="nearest")
        ay = map_coordinates(gy, idx, order=3, mode="nearest")
        g = (ax * P[:, 0] + ay * P[:, 1]) / r
        out.append(math.sqrt(max(-K * r * g.mean(), 0.0)))
    return np.array(out)


def main():
    ref = json.load(open(os.path.join(OUT, "galaxy_rotation.json")))
    r_ref = np.array(ref["r"])
    vA1 = np.interp(R_CHECK, r_ref, ref["runs"]["A1"]["v_measured"])
    vA0 = np.interp(R_CHECK, r_ref, ref["runs"]["A0"]["v_measured"])
    pts, ms = gr.build_galaxy()

    plan = [("A1_warm6000", 0.1, 6000, vA1), ("A1_warm20000", 0.1, 20000, vA1), ("A0_warm20000", 0.0, 20000, vA0)]
    runs = {}
    for name, alpha, warm, tv in plan:
        kw = dict(grid_resolution=128, alpha=alpha, warmup_ticks=warm, total_ticks=warm + ORBIT_TICKS,
                  record_every=100, max_field_frames=400)
        if alpha > 0:
            kw.update(source_local_speed=True, c2_min=0.01)
        s = SimSettings(**kw)
        res, info = gr.run(name, s, gr.make_entities(pts, ms, tv, (64 - s.sponge_width) * s.dx - 5.0))
        ft = res.frame_t
        curves = np.array([curve_from_slice(f, s, R_CHECK) for f in res.field_frames])
        final = curves[-1]
        i_rel = int(np.argmin(np.abs(ft - warm * s.dt)))
        runs[name] = {**info, "alpha": alpha, "warmup_ticks": warm,
                      "frame_t": ft.tolist(), "v_vs_time": curves.tolist(),
                      "v_at_release_over_final": (curves[i_rel] / final).tolist(),
                      "tracers": gr.tracer_report(res, s)}
        print(f"    pull at release / final pull (as speed), r = {R_CHECK.astype(int).tolist()}:")
        print("     ", " ".join(f"{x:.3f}" for x in curves[i_rel] / final))

    with open(os.path.join(OUT, "warmup_check.json"), "w") as f:
        json.dump({"r_check": R_CHECK.tolist(), "orbit_ticks": ORBIT_TICKS, "runs": runs}, f, indent=1)

    print("\n=== ORBITS (r stays within ... of its start; 1.000 - 1.000 = perfect circle) ===")
    for name in runs:
        print(f"{name}:")
        for tr in runs[name]["tracers"]:
            if tr.get("alive_frac", 0) > 0:
                print(f"   r0 = {tr['r0']:6.1f}: {tr['r_min_ratio']:.3f} - {tr['r_max_ratio']:.3f}"
                      f"   (alive {100 * tr['alive_frac']:.0f}%)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for name, a in (("A1_warm20000", ax[0]),):
        ft = np.array(runs[name]["frame_t"])
        cv = np.array(runs[name]["v_vs_time"])
        for k, r in enumerate(R_CHECK):
            a.plot(ft, cv[:, k] / cv[-1, k], label=f"r = {r:g}")
        a.axvline(6000 * 0.005, color="k", ls=":", lw=1)
        a.text(6000 * 0.005, 0.05, " old release", fontsize=8)
        a.axvline(20000 * 0.005, color="k", ls="--", lw=1)
        a.text(20000 * 0.005, 0.05, " new release", fontsize=8)
        a.set_ylim(0, 1.15)
        a.set_title("alpha=0.1: pull building up (speed / final)")
        a.set_xlabel("time"); a.legend(fontsize=7)
    for a, name, col in ((ax[1], "A1_warm6000", "tab:red"), (ax[2], "A1_warm20000", "tab:red"),
                         (ax[3], "A0_warm20000", "tab:blue")):
        for tr in runs[name]["tracers"]:
            if tr.get("alive_frac", 0) > 0:
                t = np.array(tr["t"])
                a.plot(t - t[0], tr["r_over_r0"], color=col, lw=0.8)
        a.axhline(1.0, color="k", ls=":")
        a.set_ylim(0, 3.2)
        a.set_title(f"{name} orbits"); a.set_xlabel("time since release"); a.set_ylabel("r / r_start")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "warmup_check.png"), dpi=160)
    print("\nSaved results/warmup_check.png and results/warmup_check.json")


if __name__ == "__main__":
    main()
