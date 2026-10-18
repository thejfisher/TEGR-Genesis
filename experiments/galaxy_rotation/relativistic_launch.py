"""Relativistic launch check for the galaxy test (follow-up to warmup_check.py).

warmup_check.py showed the field is fully built long before the stars are released, so the
warm-up was NOT why the alpha = 0.1 stars drifted outward.

New guess: the engine moves bodies relativistically (p = gamma m v, wave speed c_base = 65).
A circular orbit then needs  K |grad phi| = gamma v^2 / r,  not  v^2 / r.
We launched at the Newtonian speed w = sqrt(K r |grad phi|), which is slightly too fast,
so every star starts at its closest point and swings outward. The error grows with speed:
  v = 15  -> gamma = 1.03 ;  v = 47 (alpha = 0.1 inner disk) -> gamma = 1.45.

Test: launch at the relativistic circular speed  v^2 / sqrt(1 - v^2/c^2) = w^2  and compare
with the Newtonian launch. Same galaxy, same field, warm-up 6000 (enough, per warmup_check).
Outputs: results/relativistic_launch.json and results/relativistic_launch.png
"""
import json
import math
import os

import numpy as np

import galaxy_rotation as gr
from core.engine import SimSettings   # path set up by galaxy_rotation

OUT = gr.OUT
C = 65.0


def relativistic_speed(w):
    """Solve v^2 / sqrt(1 - v^2/C^2) = w^2 for v."""
    w4 = np.asarray(w, float) ** 4
    x = 0.5 * (-w4 / C ** 2 + np.sqrt(w4 ** 2 / C ** 4 + 4.0 * w4))
    return np.sqrt(x)


def main():
    ref = json.load(open(os.path.join(OUT, "galaxy_rotation.json")))
    r_ref = np.array(ref["r"])
    R = np.array(gr.TRACER_R)
    pts, ms = gr.build_galaxy()
    runs = {}
    for tag, alpha, key in (("A0", 0.0, "A0"), ("A1", 0.1, "A1")):
        w = np.interp(R, r_ref, ref["runs"][key]["v_measured"])
        for launch, tv in (("newton", w), ("relativistic", relativistic_speed(w))):
            kw = dict(grid_resolution=128, alpha=alpha, total_ticks=16000, record_every=100, max_field_frames=40)
            if alpha > 0:
                kw.update(source_local_speed=True, c2_min=0.01)
            s = SimSettings(**kw)
            name = f"{tag}_{launch}"
            res, info = gr.run(name, s, gr.make_entities(pts, ms, tv, (64 - s.sponge_width) * s.dx - 5.0))
            runs[name] = {**info, "launch_speed": tv.tolist(), "newton_speed": w.tolist(),
                          "gamma_at_launch": (1.0 / np.sqrt(1.0 - (tv / C) ** 2)).tolist(),
                          "tracers": gr.tracer_report(res, s)}

    with open(os.path.join(OUT, "relativistic_launch.json"), "w") as f:
        json.dump({"r": R.tolist(), "runs": runs}, f, indent=1)

    print("\n=== ORBITS: r stays within (min - max) of its start. Perfect circle = 1.000 - 1.000 ===")
    print(f"{'r0':>6} | {'A0 Newton launch':>17} | {'A0 relativistic':>17} | {'A1 Newton launch':>17} | "
          f"{'A1 relativistic':>17} | gamma (A1)")
    for k, r in enumerate(R):
        cells = []
        for name in ("A0_newton", "A0_relativistic", "A1_newton", "A1_relativistic"):
            tr = [t for t in runs[name]["tracers"] if abs(t["r0"] - r) < 0.5]
            cells.append(f"{tr[0]['r_min_ratio']:.3f} - {tr[0]['r_max_ratio']:.3f}" if tr and tr[0].get("alive_frac", 0) > 0
                         else "-")
        print(f"{r:6.0f} | " + " | ".join(f"{c:>17}" for c in cells) +
              f" | {runs['A1_newton']['gamma_at_launch'][k]:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), sharey=True)
    for a, name, col in zip(ax, ("A0_newton", "A0_relativistic", "A1_newton", "A1_relativistic"),
                            ("tab:blue", "tab:blue", "tab:red", "tab:red")):
        for tr in runs[name]["tracers"]:
            if tr.get("alive_frac", 0) > 0:
                t = np.array(tr["t"])
                a.plot(t - t[0], tr["r_over_r0"], color=col, lw=0.8)
        a.axhline(1.0, color="k", ls=":")
        a.set_ylim(0, 3.2)
        a.set_title(name.replace("_", ", ") + " launch")
        a.set_xlabel("time since release")
    ax[0].set_ylabel("r / r_start")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "relativistic_launch.png"), dpi=160)
    print("\nSaved results/relativistic_launch.png and results/relativistic_launch.json")


if __name__ == "__main__":
    main()
