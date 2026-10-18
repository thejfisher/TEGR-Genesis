"""Diagnosis: why did the spinning disk fly apart? (follow-up to spinning_disk.py, alpha = 0 only)

In spinning_disk.py about half the active disk stars left the box (alpha = 0) and most did at alpha = 0.1,
while the no-wake tracer control kept every star. Before reading anything as a wake, find the cause.

Two candidate causes, tested separately:
  A. A single moving source misbehaves on its own (self-force / own wake).
     Test: ONE light active star at a time (far apart from each other) orbiting the centre, next to
     tracers (no feeding) on the same orbits. Then the same with 10x heavier stars.
     If active stars drift and tracers don't, a moving source pushes or drags itself.
  B. The cold, heavy disk is gravitationally unstable (breaks into clumps that fling stars out).
     Test: the same disk, 10x lighter (mass 14 instead of 140), so the centre dominates.
     Also record WHEN the full disk loses its stars (all at once = launch problem; gradual = instability).

Outputs: results/diagnose.json and results/diagnose.png
"""
import json
import math
import os

import numpy as np

import spinning_disk as sd
from spinning_disk import gr, Entity, SimSettings, curve_from_slice, relativistic_speed

OUT = sd.OUT
SINGLE_R = [20.0, 35.0, 50.0]


def timeline(res, rows, s):
    t = res.t
    rel = int(np.searchsorted(t, sd.WARM * s.dt))
    alive = res.alive[rel:, rows].mean(1)
    rr = np.hypot(res.pos[rel:, rows, 0], res.pos[rel:, rows, 1])
    r0 = rr[0]
    med = np.median(rr / r0, axis=1)
    return (t[rel:] - t[rel]).tolist(), alive.tolist(), med.tolist()


def orbit_stats(res, j, s):
    t = res.t
    rel = int(np.searchsorted(t, sd.WARM * s.dt))
    p, v, g = res.pos[rel:, j], res.vel[rel:, j], res.gamma[rel:, j]
    ok = res.alive[rel:, j]
    rr = np.hypot(p[:, 0], p[:, 1]) / math.hypot(*p[0, :2])
    L = g * (p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0])
    return {"r_min": float(rr[ok].min()), "r_max": float(rr[ok].max()), "alive_end": bool(ok[-1]),
            "L_change_pct": float(100 * (L[ok][-1] / L[0] - 1.0)),
            "t": (t[rel:] - t[rel]).tolist(), "r_over_r0": rr.tolist()}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Spinning disk diagnosis with v0.1a engine support")
    parser.add_argument("--tag", default="v01a", help="Tag for output files, e.g. v01a")
    parser.add_argument("--kernel", choices=["cic", "tsc", "gauss"], default="tsc", help="Body cloud kernel")
    parser.add_argument("--width", type=float, default=2.8125, help="Width for Gaussian kernel")
    args = parser.parse_args()

    sd.BODY = dict(body_kernel=args.kernel, body_width=args.width)

    s = sd.settings(0.0)
    res_core, _ = gr.run("pre_core", s, [sd.core()])
    w = curve_from_slice(res_core.field_frames[-1], s, sd.R_CURVE)
    v_core = lambda r: float(np.interp(r, sd.R_CURVE, relativistic_speed(w)))
    out = {"singles": {}}

    # ---- A. single active stars vs tracers on the same orbits
    for mass in (0.467, 4.67):
        ents = [sd.core()]
        for k, r in enumerate(SINGLE_R):
            for kind, off in (("active", 0.0), ("tracer", math.pi)):
                th = k * 2.0944 + off
                v = v_core(r)
                ents.append(Entity(kind, x=r * math.cos(th), y=r * math.sin(th), z=0.0,
                                   vx=-v * math.sin(th), vy=v * math.cos(th), mass=mass, label=f"{kind} r={r:g}"))
        res, _ = gr.run(f"singles m={mass}", s, ents)
        block = {}
        for j, e in enumerate(res.entities[:len(res.kinds)]):
            if e["kind"] in ("active", "tracer"):
                st = orbit_stats(res, j, s)
                block[e["label"]] = st
                print(f"    m={mass:<5} {e['label']:<14} r stays within {st['r_min']:.3f} - {st['r_max']:.3f}, "
                      f"spin change {st['L_change_pct']:+.2f}%, alive at end {st['alive_end']}", flush=True)
        out["singles"][str(mass)] = block

    # ---- B. full disk vs 10x lighter disk (prograde), with a timeline
    disk = sd.build_disk()
    rows = list(range(1, 1 + sd.DISK_N))
    out["disks"] = {}
    for label, dm in (("full (mass 140)", sd.DISK_M), ("light (mass 14)", sd.DISK_M / 10.0)):
        old = sd.DISK_M
        sd.DISK_M = dm
        try:
            res_pre, _ = gr.run(f"pre {label}", s, [sd.core()] + sd.disk_entities(disk, None, "well", [1] * sd.DISK_N))
            vf, _ = sd.speed_fn(res_pre, s)
            res, _ = gr.run(f"disk {label}", s, [sd.core()] + sd.disk_entities(disk, vf, "active", [1] * sd.DISK_N))
        finally:
            sd.DISK_M = old
        t, alive, med = timeline(res, rows, s)
        out["disks"][label] = {"t": t, "alive": alive, "median_r_over_r0": med}
        i10 = int(np.searchsorted(t, 10.0))
        print(f"    {label}: stars left at t=10: {100 * alive[i10]:.0f}%, at end: {100 * alive[-1]:.0f}%", flush=True)

    tag_str = f"_{args.tag}" if args.tag else ""
    json_path = os.path.join(OUT, f"diagnose{tag_str}.json")
    png_path = os.path.join(OUT, f"diagnose{tag_str}.png")

    with open(json_path, "w") as f:
        json.dump(out, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for a, mass in zip(ax[:2], ("0.467", "4.67")):
        for lab, st in out["singles"][mass].items():
            a.plot(st["t"], st["r_over_r0"], color="tab:red" if lab.startswith("active") else "tab:gray",
                   ls="-" if lab.startswith("active") else "--", label=lab)
        a.axhline(1.0, color="k", ls=":")
        a.set_title(f"One star at a time, mass {mass} (red = feeds field, gray = tracer)")
        a.set_xlabel("time since release"); a.set_ylabel("r / r_start"); a.legend(fontsize=7)
    for lab, d in out["disks"].items():
        ax[2].plot(d["t"], d["alive"], label=f"{lab}: stars still in box")
    ax[2].set_ylim(0, 1.05)
    ax[2].set_title("Disk stars still in the box"); ax[2].set_xlabel("time since release"); ax[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    print(f"\nSaved {png_path} and {json_path}")


if __name__ == "__main__":
    main()
