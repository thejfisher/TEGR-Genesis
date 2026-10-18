"""Spinning disk: look for the wake of moving sources (follow-up to galaxy_rotation).

Stock 7-point Genesis engine; engine.py is NOT edited.

IDEA
  In galaxy_rotation the disk was made of fixed wells, so the field was still. Here the disk stars are
  ACTIVE bodies: they orbit AND feed the field as they move. Because the field is a wave with finite
  speed, a moving source leaves a wake behind it. We look for three things the wake could do:
    1. Drag: the wake pulls back on the stars, so the disk loses spin (angular momentum).
    2. Extra pull outside the disk (the "dark matter" question): is the outer rotation curve
       stronger after the disk starts spinning than when it was still?
    3. A twist: do test stars orbiting WITH the disk behave differently from ones orbiting AGAINST it?

SETUP (kept simple so each effect can be traced)
  - Bulge = ONE fixed well (mass 60) exactly at the centre. It is round, so it cannot twist the disk.
    (In galaxy_rotation the bulge was 30 lumps, which could exchange spin with the disk.)
  - Disk = 300 active bodies (total mass 140), exponential, radius 8 to 75, in the z = 0 plane.
  - Stars are held still for 6000 ticks while the field settles (that still field is the baseline),
    then released on circular orbits at the RELATIVISTIC circular speed (see relativistic_launch.py).

RUNS, for alpha = 0 (plain gravity) and alpha = 0.1 (Fabbri c/c_base source)
  pre_full   still disk (fixed wells): gives the launch speeds for the disk
  pre_core   central well only: gives launch speeds for the no-wake control
  P          active disk, all stars orbit the same way (prograde)
  C          active disk, half the stars orbit each way (counter-rotating), same speeds
  T          no-wake control: the disk stars are tracers (feel the field, never feed it) orbiting the
             central well alone. Any spin they lose is numerical (grid), not wake.

PREDICTION, WRITTEN BEFORE RUNNING
  alpha = 0: stars move at about 0.2-0.35 of the wave speed. A wake exists but should be weak.
             Expect small spin loss beyond the T control, and little or no extra pull outside the disk.
  alpha = 0.1: near the centre the LOCAL wave speed drops (to about 65/3 at radius 5), while inner stars
             move at 40+. They can outrun the local wave speed, like a boat outrunning its own waves.
             If so, expect a much stronger wake and spin loss in the inner disk (a Cherenkov-like drag).
  We do not expect a flat outer rotation curve: a lumpy disk of moving masses still has the same total mass.

Outputs: results/spinning_disk.json and results/spinning_disk.png
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.abspath(os.path.join(HERE, "..", "galaxy_rotation"))
if GAL not in sys.path:
    sys.path.insert(0, GAL)

import galaxy_rotation as gr                       # noqa: E402  (also puts the engine on the path)
from warmup_check import curve_from_slice          # noqa: E402
from relativistic_launch import relativistic_speed  # noqa: E402
from core.engine import SimSettings                # noqa: E402
from core.entities import Entity                   # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

SEED = 7
CORE_M = 60.0
DISK_N, DISK_M, DISK_RD, DISK_RMIN, DISK_RMAX = 300, 140.0, 18.0, 8.0, 75.0
WARM, ORBIT = 6000, 12000
R_CURVE = np.arange(6.0, 121.0, 1.5)
R_OUTER = np.arange(85.0, 116.0, 3.0)
R_PAIRS = [85.0, 100.0, 115.0]
R_BINS = [(8, 25), (25, 45), (45, 75)]


def build_disk():
    rng = np.random.default_rng(SEED)
    out = []
    while len(out) < DISK_N:
        R = rng.gamma(2.0, DISK_RD)
        if DISK_RMIN <= R <= DISK_RMAX:
            out.append((R, rng.uniform(0.0, 2.0 * math.pi)))
    return np.array(out)


def settings(alpha):
    kw = dict(grid_resolution=128, alpha=alpha, warmup_ticks=WARM, total_ticks=WARM + ORBIT,
              record_every=100, max_field_frames=200)
    if alpha > 0:
        kw.update(source_local_speed=True, c2_min=0.01)
    return SimSettings(**kw)


def core():
    return Entity("well", 0.0, 0.0, 0.0, mass=CORE_M, label="core")


def disk_entities(disk, speed_of_r, kind, senses):
    ents = []
    m = DISK_M / DISK_N
    for (R, th), sense in zip(disk, senses):
        v = sense * float(speed_of_r(R)) if kind != "well" else 0.0
        ents.append(Entity(kind, x=R * math.cos(th), y=R * math.sin(th), z=0.0,
                           vx=-v * math.sin(th), vy=v * math.cos(th), mass=m))
    return ents


def pair_tracers(speed_of_r):
    ents = []
    for k, r in enumerate(R_PAIRS):
        for sense, tag in ((1, "with"), (-1, "against")):
            th = k * gr.GOLDEN + (0.0 if sense > 0 else math.pi)
            v = sense * float(speed_of_r(r))
            ents.append(Entity("tracer", x=r * math.cos(th), y=r * math.sin(th), z=0.0,
                               vx=-v * math.sin(th), vy=v * math.cos(th), mass=1.0, label=f"pair {tag} r={r:g}"))
    return ents


def speed_fn(res, s):
    """Relativistic circular speed vs radius from the settled z = 0 slice."""
    w = curve_from_slice(res.field_frames[-1], s, R_CURVE)
    v = relativistic_speed(w)
    return (lambda r: np.interp(r, R_CURVE, v)), w


def spin(res, rows, release):
    """Angular momentum L_z = gamma m (x vy - y vx), summed over rows; only bodies alive at the end."""
    alive_end = res.alive[-1, rows]
    rows = np.asarray(rows)[alive_end]
    m = np.array([res.entities[j]["mass"] for j in rows])
    p, v, g = res.pos[:, rows, :], res.vel[:, rows, :], res.gamma[:, rows]
    Lz = g * m[None] * (p[..., 0] * v[..., 1] - p[..., 1] * v[..., 0])
    return Lz[release:], rows, float(alive_end.mean())


def analyse(name, res, s, disk_rows, senses, pre_slice):
    t = res.t
    release = int(np.searchsorted(t, WARM * s.dt))
    out = {"status": res.status, "device": res.device, "wall_seconds": round(res.wall_seconds, 1)}
    if disk_rows:
        Lz, rows, alive = spin(res, disk_rows, release)
        sense_alive = np.array(senses)[[disk_rows.index(j) for j in rows]]
        r0 = np.hypot(res.pos[release, rows, 0], res.pos[release, rows, 1])
        out["disk_alive_frac"] = alive
        out["t"] = (t[release:] - t[release]).tolist()
        for sign, tag in ((1, "with"), (-1, "against")):
            sel = sense_alive == sign
            if not sel.any():
                continue
            tot = Lz[:, sel].sum(1)
            out[f"L_{tag}_over_start"] = (tot / tot[0]).tolist()
            out[f"L_{tag}_change_pct"] = float(100.0 * (tot[-1] / tot[0] - 1.0))
            for lo, hi in R_BINS:
                b = sel & (r0 >= lo) & (r0 < hi)
                if b.sum() >= 5:
                    tb = Lz[:, b].sum(1)
                    out[f"L_{tag}_r{lo}_{hi}_change_pct"] = float(100.0 * (tb[-1] / tb[0] - 1.0))
        rr = np.hypot(res.pos[-1, rows, 0], res.pos[-1, rows, 1])
        out["radius_spread_end"] = float(np.std(rr / r0))
    # outer pull after release vs the still baseline (second half of the orbit time)
    ft = res.frame_t
    late = ft >= ft[-1] - 0.5 * ORBIT * s.dt
    base = curve_from_slice(pre_slice, s, R_OUTER)
    late_curve = np.mean([curve_from_slice(f, s, R_OUTER) for f in res.field_frames[late]], axis=0)
    out["outer_pull_after_over_before"] = (late_curve / base).tolist()
    # with / against test-star pairs outside the disk
    pairs = {}
    for j, e in enumerate(res.entities[:len(res.kinds)]):
        if e["label"].startswith("pair"):
            p = res.pos[release:, j, :]
            rr = np.hypot(p[:, 0], p[:, 1]) / np.hypot(*p[0, :2])
            ok = res.alive[release:, j]
            pairs[e["label"]] = [float(rr[ok].min()), float(rr[ok].max())] if ok.any() else None
    out["pairs"] = pairs
    # wake map: final in-plane field minus the still field
    rel_frame = int(np.argmin(np.abs(ft - WARM * s.dt)))
    out["_wake"] = res.field_frames[-1] - res.field_frames[rel_frame]
    out["_late_mean_minus_still"] = res.field_frames[late].mean(0) - res.field_frames[rel_frame]
    return out


def main():
    disk = build_disk()
    rng = np.random.default_rng(SEED + 1)
    mixed = list(rng.permutation([1] * (DISK_N // 2) + [-1] * (DISK_N - DISK_N // 2)))
    summary = {"setup": {"core_mass": CORE_M, "disk_mass": DISK_M, "disk_n": DISK_N, "disk_r": [DISK_RMIN, DISK_RMAX],
                         "warmup_ticks": WARM, "orbit_ticks": ORBIT}, "alpha": {}}
    wake_maps = {}
    for alpha in (0.0, 0.1):
        s = settings(alpha)
        print(f"\n######## alpha = {alpha} ########", flush=True)
        res_full, _ = gr.run("pre_full (still disk)", s, [core()] + disk_entities(disk, None, "well", [1] * DISK_N))
        v_full, w_full = speed_fn(res_full, s)
        res_core, _ = gr.run("pre_core (centre only)", s, [core()])
        v_core, _ = speed_fn(res_core, s)
        still_slice = res_full.field_frames[-1]
        block = {"r_curve": R_CURVE.tolist(), "v_still_newtonian": w_full.tolist(),
                 "v_launch_disk": v_full(R_CURVE).tolist(), "v_launch_core_only": v_core(R_CURVE).tolist(),
                 "max_v_over_c": float(v_full(R_CURVE).max() / s.c_base), "runs": {}}
        rows = list(range(1, 1 + DISK_N))
        plan = [("P", "active", [1] * DISK_N, v_full),
                ("C", "active", mixed, v_full),
                ("T", "tracer", [1] * DISK_N, v_core)]
        for tag, kind, senses, vf in plan:
            ents = [core()] + disk_entities(disk, vf, kind, senses) + pair_tracers(v_full if tag != "T" else v_core)
            res, info = gr.run(f"{tag} ({'prograde' if tag == 'P' else 'counter-rotating' if tag == 'C' else 'no-wake control'})",
                               s, ents)
            pre = still_slice if tag != "T" else res_core.field_frames[-1]
            a = analyse(tag, res, s, rows, senses, pre)
            wake_maps[(alpha, tag)] = (a.pop("_wake"), a.pop("_late_mean_minus_still"))
            a.update({k: v for k, v in info.items() if k not in a})
            block["runs"][tag] = a
            msg = []
            for k in ("L_with_change_pct", "L_against_change_pct"):
                if k in a:
                    msg.append(f"{k.split('_')[1]} {a[k]:+.2f}%")
            print(f"    disk spin change: {', '.join(msg)} | alive {100 * a.get('disk_alive_frac', 0):.0f}% | "
                  f"outer pull after/before {min(a['outer_pull_after_over_before']):.4f}-"
                  f"{max(a['outer_pull_after_over_before']):.4f}", flush=True)
        summary["alpha"][str(alpha)] = block
        with open(os.path.join(OUT, "spinning_disk.json"), "w") as f:
            json.dump(summary, f, indent=1)

    # ------------------------------------------------------------ plain-language summary
    print("\n=== RESULTS ===")
    for alpha in ("0.0", "0.1"):
        b = summary["alpha"][alpha]
        print(f"\nalpha = {alpha}  (fastest disk star = {b['max_v_over_c']:.2f} of the wave speed)")
        for tag in ("P", "C", "T"):
            a = b["runs"][tag]
            parts = []
            for side in ("with", "against"):
                if f"L_{side}_change_pct" in a:
                    bins = " ".join(f"[{lo}-{hi}: {a.get(f'L_{side}_r{lo}_{hi}_change_pct', float('nan')):+.1f}%]"
                                    for lo, hi in R_BINS)
                    parts.append(f"spin {side}: {a[f'L_{side}_change_pct']:+.2f}% {bins}")
            print(f"  {tag}: " + " | ".join(parts) + f" | alive {100 * a['disk_alive_frac']:.0f}% | "
                  f"radius spread {a['radius_spread_end']:.3f}")
            op = a["outer_pull_after_over_before"]
            print(f"      outer pull after/before (r 85-115): {min(op):.4f} - {max(op):.4f}")
            for lab, rng_ in a["pairs"].items():
                if rng_:
                    print(f"      {lab}: r stays within {rng_[0]:.3f} - {rng_[1]:.3f}")

    # ------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    for row, alpha in enumerate(("0.0", "0.1")):
        b = summary["alpha"][alpha]
        a0 = ax[row, 0]
        for tag, col in (("P", "tab:red"), ("C", "tab:purple"), ("T", "tab:gray")):
            a = b["runs"][tag]
            t = np.array(a["t"])
            a0.plot(t, a["L_with_over_start"], color=col, label=f"{tag}: stars orbiting with")
            if "L_against_over_start" in a:
                a0.plot(t, a["L_against_over_start"], color=col, ls="--", label=f"{tag}: stars orbiting against")
        a0.axhline(1.0, color="k", ls=":")
        a0.set_title(f"alpha={alpha}: disk spin / starting spin")
        a0.set_xlabel("time since release"); a0.legend(fontsize=7)
        a1 = ax[row, 1]
        for tag, col in (("P", "tab:red"), ("C", "tab:purple"), ("T", "tab:gray")):
            a1.plot(R_OUTER, b["runs"][tag]["outer_pull_after_over_before"], "o-", ms=3, color=col, label=tag)
        a1.axhline(1.0, color="k", ls=":")
        a1.set_title(f"alpha={alpha}: outer pull after release / still (1 = no change)")
        a1.set_xlabel("radius"); a1.legend(fontsize=7)
        wk = wake_maps[(float(alpha), "P")][1]
        G, dx = wk.shape[0], 2.8125
        half = G * dx / 2
        lim = float(np.percentile(np.abs(wk), 99.5)) or 1e-9
        im = ax[row, 2].imshow(wk.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                               extent=[-half, half, -half, half])
        ax[row, 2].set_xlim(-110, 110); ax[row, 2].set_ylim(-110, 110)
        ax[row, 2].set_title(f"alpha={alpha}: P wake = field (late average) minus still field")
        plt.colorbar(im, ax=ax[row, 2], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "spinning_disk.png"), dpi=150)
    print("\nSaved results/spinning_disk.png and results/spinning_disk.json")


if __name__ == "__main__":
    main()
