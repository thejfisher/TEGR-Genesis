"""Stable Galaxy Experiment (TEGR Genesis v0.1a).

Tests collective wave-medium effects (acoustic drag, wake vortex, and outer pull)
in a dynamically STABLE galaxy where the central core dominates the mass
(Core = 150.0 vs Disk = 20.0, ratio 7.5 : 1).

In this regime:
- Stars are gravitationally bound on stable circular orbits.
- Violent Newtonian Toomre fragmentation and star ejections are suppressed.
- Any spin loss or outer field modification is directly compared against an
  instantaneous Newtonian N-body control with the exact same star size.

Runs both alpha = 0.0 (plain wave gravity) and alpha = 0.1 (Fabbri wave speed).

Outputs:
  results/stable_galaxy.json
  results/stable_galaxy.png
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.abspath(os.path.join(HERE, "..", "galaxy_rotation"))
if GAL not in sys.path:
    sys.path.insert(0, GAL)

import galaxy_rotation as gr
from warmup_check import curve_from_slice
from relativistic_launch import relativistic_speed
from core.engine import Engine, SimSettings
from core.entities import Entity

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 7
CORE_M = 150.0
DISK_N = 300
DISK_M = 20.0
DISK_RD = 18.0
DISK_RMIN = 8.0
DISK_RMAX = 75.0
WARM = 6000
ORBIT = 12000
C_BASE = 65.0
SIGMA = 2.8125
K_FORCE = 100.0
G_SRC = 1.0
G_NEWTON = K_FORCE * G_SRC

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


def settings(alpha):
    kw = dict(grid_resolution=128, alpha=alpha, warmup_ticks=WARM, total_ticks=WARM + ORBIT,
              record_every=100, max_field_frames=200, body_kernel="gauss", body_width=SIGMA)
    if alpha > 0:
        kw.update(source_local_speed=True, c2_min=0.01)
    return SimSettings(**kw)


def speed_fn(res, s):
    w = curve_from_slice(res.field_frames[-1], s, R_CURVE)
    v = relativistic_speed(w)
    return (lambda r: np.interp(r, R_CURVE, v)), w


def spin(res, rows, release):
    alive_end = res.alive[-1, rows]
    rows = np.asarray(rows)[alive_end]
    m = np.array([res.entities[j]["mass"] for j in rows])
    p, v, g = res.pos[:, rows, :], res.vel[:, rows, :], res.gamma[:, rows]
    Lz = g * m[None] * (p[..., 0] * v[..., 1] - p[..., 1] * v[..., 0])
    return Lz[release:], rows, float(alive_end.mean())


def analyse_wave(name, res, s, disk_rows, senses, pre_slice):
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

    ft = res.frame_t
    late = ft >= ft[-1] - 0.5 * ORBIT * s.dt
    base = curve_from_slice(pre_slice, s, R_OUTER)
    late_curve = np.mean([curve_from_slice(f, s, R_OUTER) for f in res.field_frames[late]], axis=0)
    out["outer_pull_after_over_before"] = (late_curve / base).tolist()

    pairs = {}
    for j, e in enumerate(res.entities[:len(res.kinds)]):
        if e["label"].startswith("pair"):
            p = res.pos[release:, j, :]
            rr = np.hypot(p[:, 0], p[:, 1]) / np.hypot(*p[0, :2])
            ok = res.alive[release:, j]
            pairs[e["label"]] = [float(rr[ok].min()), float(rr[ok].max())] if ok.any() else None
    out["pairs"] = pairs

    rel_frame = int(np.argmin(np.abs(ft - WARM * s.dt)))
    out["_wake"] = res.field_frames[-1] - res.field_frames[rel_frame]
    out["_late_mean_minus_still"] = res.field_frames[late].mean(0) - res.field_frames[rel_frame]
    return out


def run_newton(disk, plan_tag, kind, senses, speed_fn, sigma=SIGMA):
    ents = [core()] + disk_entities(disk, speed_fn, kind, senses) + pair_tracers(speed_fn)
    N = len(ents)
    m = torch.tensor([e.mass for e in ents], device=dev, dtype=torch.float32)
    is_well = torch.tensor([e.kind == "well" for e in ents], device=dev, dtype=torch.bool)
    is_tracer = torch.tensor([e.kind == "tracer" for e in ents], device=dev, dtype=torch.bool)
    is_active = torch.tensor([e.kind == "active" for e in ents], device=dev, dtype=torch.bool)

    pos = torch.tensor([[e.x, e.y, e.z] for e in ents], device=dev, dtype=torch.float32)
    v0 = torch.tensor([[e.vx, e.vy, e.vz] for e in ents], device=dev, dtype=torch.float32)

    dt = 0.005
    ticks = ORBIT
    rec_every = 100

    speed = v0.norm(dim=1, keepdim=True).clamp(max=0.999 * C_BASE)
    v0 = torch.where(v0.norm(dim=1, keepdim=True) > 0,
                     v0 / v0.norm(dim=1, keepdim=True).clamp(min=1e-12) * speed, v0)
    gam0 = 1.0 / torch.sqrt(1.0 - (speed / C_BASE) ** 2)
    mom = gam0 * m[:, None] * v0

    alive = torch.ones(N, device=dev, dtype=torch.bool)
    half = 128 * sigma / 2.0
    edge = half - 1.5 * sigma

    P_hist, V_hist, A_hist, Gam_hist, T_hist = [], [], [], [], []

    def gamma_of(p_vec):
        return torch.sqrt(1.0 + (p_vec ** 2).sum(dim=1) / (m * C_BASE) ** 2)

    t_wall = time.time()
    for tick in range(ticks + 1):
        tt = tick * dt
        g = gamma_of(mom)
        vel = mom / (g[:, None] * m[:, None].clamp(min=1e-8))

        if tick % rec_every == 0 or tick == ticks:
            P_hist.append(pos.cpu().numpy().copy())
            V_hist.append(vel.cpu().numpy().copy())
            A_hist.append(alive.cpu().numpy().copy())
            Gam_hist.append(g.cpu().numpy().copy())
            T_hist.append(tt)

        if tick == ticks:
            break

        diff = pos[None, :, :] - pos[:, None, :]
        dist_sq = (diff ** 2).sum(dim=-1)
        dist = torch.sqrt(dist_sq + 1e-12)

        s_eff = torch.full((N, N), math.sqrt(2.0) * sigma, device=dev)
        s_eff[0, :] = sigma
        s_eff[:, 0] = sigma
        u = dist / (s_eff * math.sqrt(2.0))
        erf_u = torch.erf(u)
        factor = (erf_u - (2.0 / math.sqrt(math.pi)) * u * torch.exp(-u ** 2)) / (dist ** 3)
        inv_r3 = torch.where(dist > 1e-6, factor, torch.zeros_like(factor))
        inv_r3.fill_diagonal_(0.0)

        src_mask = (alive & (is_well | is_active)).float()
        force_pairs = G_NEWTON * m[:, None, None] * m[None, :, None] * diff * inv_r3[:, :, None] * src_mask[None, :, None]
        force = force_pairs.sum(dim=1)

        mov_mask = (~is_well & alive).float()[:, None]
        mom = mom + force * dt * mov_mask
        g = gamma_of(mom)
        vel = mom / (g[:, None] * m[:, None].clamp(min=1e-8))
        pos = pos + vel * dt * mov_mask

        out = (pos.abs() > edge).any(dim=1) & (~is_well)
        alive = alive & ~out

    wall_sec = round(time.time() - t_wall, 2)
    P_hist = np.array(P_hist)
    V_hist = np.array(V_hist)
    A_hist = np.array(A_hist)
    Gam_hist = np.array(Gam_hist)
    T_hist = np.array(T_hist)

    disk_rows = np.arange(1, 1 + DISK_N)
    alive_end = A_hist[-1, disk_rows]
    surv_rows = disk_rows[alive_end]
    surv_mass = m[surv_rows].cpu().numpy()

    p = P_hist[:, surv_rows, :]
    v = V_hist[:, surv_rows, :]
    g = Gam_hist[:, surv_rows]

    Lz = g * surv_mass[None, :] * (p[..., 0] * v[..., 1] - p[..., 1] * v[..., 0])
    sense_surv = np.array(senses)[alive_end]

    r0 = np.hypot(P_hist[0, surv_rows, 0], P_hist[0, surv_rows, 1])
    r_end = np.hypot(P_hist[-1, surv_rows, 0], P_hist[-1, surv_rows, 1])

    out_data = {
        "status": "completed",
        "wall_seconds": wall_sec,
        "disk_alive_frac": float(alive_end.mean()),
        "radius_spread_end": float(np.std(r_end / r0)),
        "t": T_hist.tolist(),
    }
    for sign, tag in ((1, "with"), (-1, "against")):
        sel = sense_surv == sign
        if sel.any():
            tot = Lz[:, sel].sum(axis=1)
            out_data[f"L_{tag}_over_start"] = (tot / tot[0]).tolist()
            out_data[f"L_{tag}_change_pct"] = float(100.0 * (tot[-1] / tot[0] - 1.0))
            for lo, hi in R_BINS:
                b = sel & (r0 >= lo) & (r0 < hi)
                if b.sum() >= 5:
                    tb = Lz[:, b].sum(axis=1)
                    out_data[f"L_{tag}_r{lo}_{hi}_change_pct"] = float(100.0 * (tb[-1] / tb[0] - 1.0))

    return out_data


def main():
    print("=================================================================")
    print("          TEGR GENESIS: STABLE GALAXY EXPERIMENT                 ")
    print(f"       Core Mass = {CORE_M:g}  |  Disk Mass = {DISK_M:g} (ratio {CORE_M/DISK_M:.1f}:1)     ")
    print("=================================================================\n")

    disk = build_disk()
    rng = np.random.default_rng(SEED + 1)
    mixed = list(rng.permutation([1] * (DISK_N // 2) + [-1] * (DISK_N - DISK_N // 2)))

    summary = {
        "setup": {
            "core_mass": CORE_M, "disk_mass": DISK_M, "disk_n": DISK_N,
            "disk_r": [DISK_RMIN, DISK_RMAX], "warmup_ticks": WARM, "orbit_ticks": ORBIT,
            "body_kernel": "gauss", "body_width": SIGMA,
        },
        "alpha": {},
        "newton": {},
    }

    wake_maps = {}

    for alpha in (0.0, 0.1):
        s = settings(alpha)
        print(f"\n######## RUNNING GENESIS WAVE: alpha = {alpha} ########", flush=True)

        res_full, _ = gr.run("pre_full (still disk)", s, [core()] + disk_entities(disk, None, "well", [1] * DISK_N))
        v_full, w_full = speed_fn(res_full, s)

        res_core, _ = gr.run("pre_core (centre only)", s, [core()])
        v_core, _ = speed_fn(res_core, s)

        still_slice = res_full.field_frames[-1]

        block = {
            "r_curve": R_CURVE.tolist(),
            "v_still_newtonian": w_full.tolist(),
            "v_launch_disk": v_full(R_CURVE).tolist(),
            "v_launch_core_only": v_core(R_CURVE).tolist(),
            "max_v_over_c": float(v_full(R_CURVE).max() / s.c_base),
            "runs": {},
        }

        rows = list(range(1, 1 + DISK_N))
        plan = [("P", "active", [1] * DISK_N, v_full),
                ("C", "active", mixed, v_full),
                ("T", "tracer", [1] * DISK_N, v_core)]

        for tag, kind, senses, vf in plan:
            ents = [core()] + disk_entities(disk, vf, kind, senses) + pair_tracers(v_full if tag != "T" else v_core)
            res, info = gr.run(f"{tag} ({'prograde' if tag == 'P' else 'counter' if tag == 'C' else 'tracer'})", s, ents)
            pre = still_slice if tag != "T" else res_core.field_frames[-1]
            a = analyse_wave(tag, res, s, rows, senses, pre)
            wake_maps[(alpha, tag)] = (a.pop("_wake"), a.pop("_late_mean_minus_still"))
            a.update({k: v for k, v in info.items() if k not in a})
            block["runs"][tag] = a

            msg = []
            for k in ("L_with_change_pct", "L_against_change_pct"):
                if k in a:
                    msg.append(f"{k.split('_')[1]} {a[k]:+.2f}%")
            print(f"    disk spin change: {', '.join(msg)} | alive {100 * a.get('disk_alive_frac', 0):.1f}% | "
                  f"spread {a['radius_spread_end']:.3f} | outer pull {min(a['outer_pull_after_over_before']):.4f}-"
                  f"{max(a['outer_pull_after_over_before']):.4f}", flush=True)

        summary["alpha"][str(alpha)] = block

        # Also run Newtonian control using this alpha's launch speeds
        if alpha == 0.0:
            print("\n######## RUNNING NEWTONIAN CONTROL (c = inf) ########", flush=True)
            newt_block = {}
            for tag, kind, senses, vf in plan:
                n_res = run_newton(disk, tag, kind, senses, vf, sigma=SIGMA)
                newt_block[tag] = n_res
                msg = []
                for k in ("L_with_change_pct", "L_against_change_pct"):
                    if k in n_res:
                        msg.append(f"{k.split('_')[1]} {n_res[k]:+.2f}%")
                print(f"    Newton {tag}: spin {', '.join(msg)} | alive {100 * n_res['disk_alive_frac']:.1f}% | "
                      f"spread {n_res['radius_spread_end']:.3f} | {n_res['wall_seconds']}s", flush=True)
            summary["newton"] = newt_block

        with open(os.path.join(OUT, "stable_galaxy.json"), "w") as f:
            json.dump(summary, f, indent=1)

    # ------------------------------------------------------------ Printout
    print("\n" + "=" * 80)
    print("                    STABLE GALAXY RESULTS SUMMARY")
    print("=" * 80)
    nw = summary["newton"]
    for alpha_str in ("0.0", "0.1"):
        wv = summary["alpha"][alpha_str]["runs"]
        print(f"\n--- alpha = {alpha_str} ---")
        print(f"  Run P (prograde)      : Wave spin = {wv['P']['L_with_change_pct']:+6.2f}%, alive = {100*wv['P']['disk_alive_frac']:.1f}%, spread = {wv['P']['radius_spread_end']:.3f}")
        if alpha_str == "0.0":
            print(f"       (Newton control) : Newton spin = {nw['P']['L_with_change_pct']:+6.2f}%, alive = {100*nw['P']['disk_alive_frac']:.1f}%, spread = {nw['P']['radius_spread_end']:.3f}")
            print(f"       Difference       : {wv['P']['L_with_change_pct'] - nw['P']['L_with_change_pct']:+6.2f}% spin drag from wave field")
        print(f"  Run C (counter-rot)   : Wave spin with = {wv['C']['L_with_change_pct']:+6.2f}%, against = {wv['C']['L_against_change_pct']:+6.2f}%, alive = {100*wv['C']['disk_alive_frac']:.1f}%")
        if alpha_str == "0.0":
            print(f"       (Newton control) : Newton with = {nw['C']['L_with_change_pct']:+6.2f}%, against = {nw['C']['L_against_change_pct']:+6.2f}%")
        print(f"  Run T (tracers)       : Wave spin = {wv['T']['L_with_change_pct']:+6.2f}%, alive = {100*wv['T']['disk_alive_frac']:.1f}%")
        op = wv["P"]["outer_pull_after_over_before"]
        print(f"  Outer Pull (P, r=85-115) : {min(op):.4f} - {max(op):.4f} of still field")

    # ------------------------------------------------------------ Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # Row 0: alpha = 0.0
    # Panel 0,0: Spin evolution (Wave vs Newton)
    t_wv = np.array(summary["alpha"]["0.0"]["runs"]["P"]["t"])
    ax[0, 0].plot(t_wv, summary["alpha"]["0.0"]["runs"]["P"]["L_with_over_start"], "tab:red", lw=2, label="Wave P (all with)")
    ax[0, 0].plot(t_wv, summary["newton"]["P"]["L_with_over_start"], "k--", lw=1.8, label="Newton P (Instant)")
    ax[0, 0].plot(t_wv, summary["alpha"]["0.0"]["runs"]["C"]["L_with_over_start"], "tab:purple", lw=1.8, label="Wave C (with)")
    ax[0, 0].plot(t_wv, summary["newton"]["C"]["L_with_over_start"], "k:", lw=1.5, label="Newton C (with)")
    ax[0, 0].plot(t_wv, summary["alpha"]["0.0"]["runs"]["T"]["L_with_over_start"], "tab:gray", lw=1.5, label="Wave T (control)")
    ax[0, 0].axhline(1.0, color="0.4", ls=":")
    ax[0, 0].set_title("alpha=0.0: Disk Spin Over Time (Wave vs Newton)")
    ax[0, 0].set_xlabel("Time since release")
    ax[0, 0].set_ylabel("Spin L_z / L_0")
    ax[0, 0].legend(fontsize=7, loc="lower left")
    ax[0, 0].grid(True, alpha=0.3)

    # Panel 0,1: Outer pull ratio
    for tag, col in (("P", "tab:red"), ("C", "tab:purple"), ("T", "tab:gray")):
        ax[0, 1].plot(R_OUTER, summary["alpha"]["0.0"]["runs"][tag]["outer_pull_after_over_before"], "o-", ms=4, color=col, label=f"Wave {tag}")
    ax[0, 1].axhline(1.0, color="k", ls=":")
    ax[0, 1].set_title("alpha=0.0: Outer Pull After Release / Still (1 = no change)")
    ax[0, 1].set_xlabel("Radius r")
    ax[0, 1].set_ylabel("Pull ratio")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(True, alpha=0.3)

    # Panel 0,2: Spatial wake map alpha = 0.0
    wk0 = wake_maps[(0.0, "P")][1]
    G, dx = wk0.shape[0], 2.8125
    half = G * dx / 2
    lim0 = float(np.percentile(np.abs(wk0), 99.5)) or 1e-9
    im0 = ax[0, 2].imshow(wk0.T, origin="lower", cmap="RdBu_r", vmin=-lim0, vmax=lim0, extent=[-half, half, -half, half])
    ax[0, 2].set_xlim(-110, 110); ax[0, 2].set_ylim(-110, 110)
    ax[0, 2].set_title("alpha=0.0: P Wake = Field (Late Mean) - Still Field")
    plt.colorbar(im0, ax=ax[0, 2], fraction=0.046)

    # Row 1: alpha = 0.1
    # Panel 1,0: Spin evolution alpha = 0.1
    t_wv1 = np.array(summary["alpha"]["0.1"]["runs"]["P"]["t"])
    ax[1, 0].plot(t_wv1, summary["alpha"]["0.1"]["runs"]["P"]["L_with_over_start"], "tab:red", lw=2, label="Wave P (with)")
    ax[1, 0].plot(t_wv1, summary["alpha"]["0.1"]["runs"]["C"]["L_with_over_start"], "tab:purple", lw=1.8, label="Wave C (with)")
    ax[1, 0].plot(t_wv1, summary["alpha"]["0.1"]["runs"]["C"]["L_against_over_start"], "tab:purple", ls="--", lw=1.8, label="Wave C (against)")
    ax[1, 0].plot(t_wv1, summary["alpha"]["0.1"]["runs"]["T"]["L_with_over_start"], "tab:gray", lw=1.5, label="Wave T (control)")
    ax[1, 0].axhline(1.0, color="0.4", ls=":")
    ax[1, 0].set_title("alpha=0.1: Disk Spin Over Time (Wave Medium)")
    ax[1, 0].set_xlabel("Time since release")
    ax[1, 0].set_ylabel("Spin L_z / L_0")
    ax[1, 0].legend(fontsize=7, loc="lower left")
    ax[1, 0].grid(True, alpha=0.3)

    # Panel 1,1: Outer pull ratio alpha = 0.1
    for tag, col in (("P", "tab:red"), ("C", "tab:purple"), ("T", "tab:gray")):
        ax[1, 1].plot(R_OUTER, summary["alpha"]["0.1"]["runs"][tag]["outer_pull_after_over_before"], "o-", ms=4, color=col, label=f"Wave {tag}")
    ax[1, 1].axhline(1.0, color="k", ls=":")
    ax[1, 1].set_title("alpha=0.1: Outer Pull After Release / Still (1 = no change)")
    ax[1, 1].set_xlabel("Radius r")
    ax[1, 1].set_ylabel("Pull ratio")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(True, alpha=0.3)

    # Panel 1,2: Spatial wake map alpha = 0.1
    wk1 = wake_maps[(0.1, "P")][1]
    lim1 = float(np.percentile(np.abs(wk1), 99.5)) or 1e-9
    im1 = ax[1, 2].imshow(wk1.T, origin="lower", cmap="RdBu_r", vmin=-lim1, vmax=lim1, extent=[-half, half, -half, half])
    ax[1, 2].set_xlim(-110, 110); ax[1, 2].set_ylim(-110, 110)
    ax[1, 2].set_title("alpha=0.1: P Wake = Field (Late Mean) - Still Field")
    plt.colorbar(im1, ax=ax[1, 2], fraction=0.046)

    plt.tight_layout()
    fig_path = os.path.join(OUT, "stable_galaxy.png")
    plt.savefig(fig_path, dpi=160)
    print(f"\nSaved {fig_path} and results/stable_galaxy.json\n")


if __name__ == "__main__":
    main()
