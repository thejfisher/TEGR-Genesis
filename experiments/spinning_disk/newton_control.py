"""Newtonian Control for the Spinning Disk Experiment.

Tests whether the ~45% spin loss of the active disk in TEGR Genesis (alpha = 0)
is caused by retarded wave wake drag, or by standard Newtonian self-gravitational
relaxation / clumping of a heavy, cold stellar disk (mass 140 vs core 60).

Runs the EXACT same disk (same random seed 7, same 300 stars, same launch speeds)
under instantaneous Newtonian gravity (infinite wave speed) using:
  1. Gaussian softening (matching the Genesis Gaussian body cloud sigma = 2.8125)
  2. Plummer softening (epsilon = 2.8125)

Outputs:
  results/newton_control.json
  results/newton_control.png
"""
import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spinning_disk as sd

OUT = sd.OUT
os.makedirs(OUT, exist_ok=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMA = 2.8125
C_LIGHT = 65.0
K_FORCE = 100.0
G_SRC = 1.0
G_NEWTON = K_FORCE * G_SRC  # 100.0


def load_wave_results():
    """Load the Genesis wave simulation results (Gaussian and TSC)."""
    results = {}
    for tag in ("gauss", "tsc"):
        p = os.path.join(OUT, f"spinning_disk_v01a_{tag}.json")
        if os.path.exists(p):
            with open(p) as f:
                results[tag] = json.load(f)
    return results


def run_newton(disk, mixed_senses, plan_tag, kind, senses, speed_fn, softening="gauss", sigma=SIGMA):
    ents = [sd.core()] + sd.disk_entities(disk, speed_fn, kind, senses) + sd.pair_tracers(speed_fn)
    N = len(ents)
    m = torch.tensor([e.mass for e in ents], device=dev, dtype=torch.float32)
    is_well = torch.tensor([e.kind == "well" for e in ents], device=dev, dtype=torch.bool)
    is_tracer = torch.tensor([e.kind == "tracer" for e in ents], device=dev, dtype=torch.bool)
    is_active = torch.tensor([e.kind == "active" for e in ents], device=dev, dtype=torch.bool)

    pos = torch.tensor([[e.x, e.y, e.z] for e in ents], device=dev, dtype=torch.float32)
    v0 = torch.tensor([[e.vx, e.vy, e.vz] for e in ents], device=dev, dtype=torch.float32)

    dt = 0.005
    ticks = sd.ORBIT
    rec_every = 100

    # Relativistic momentum update identical to Genesis engine
    speed = v0.norm(dim=1, keepdim=True).clamp(max=0.999 * C_LIGHT)
    v0 = torch.where(v0.norm(dim=1, keepdim=True) > 0,
                     v0 / v0.norm(dim=1, keepdim=True).clamp(min=1e-12) * speed, v0)
    gam0 = 1.0 / torch.sqrt(1.0 - (speed / C_LIGHT) ** 2)
    mom = gam0 * m[:, None] * v0

    alive = torch.ones(N, device=dev, dtype=torch.bool)
    half = 128 * sigma / 2.0
    edge = half - 1.5 * sigma

    P_hist, V_hist, A_hist, Gam_hist, T_hist = [], [], [], [], []

    def gamma_of(p_vec):
        return torch.sqrt(1.0 + (p_vec ** 2).sum(dim=1) / (m * C_LIGHT) ** 2)

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

        # Instantaneous pairwise Newtonian gravity
        diff = pos[None, :, :] - pos[:, None, :]  # (N, N, 3): pos[j] - pos[i]
        dist_sq = (diff ** 2).sum(dim=-1)
        dist = torch.sqrt(dist_sq + 1e-12)

        if softening == "plummer":
            inv_r3 = 1.0 / (dist_sq + sigma ** 2) ** 1.5
        elif softening == "gauss":
            s_eff = torch.full((N, N), math.sqrt(2.0) * sigma, device=dev)
            s_eff[0, :] = sigma
            s_eff[:, 0] = sigma
            u = dist / (s_eff * math.sqrt(2.0))
            erf_u = torch.erf(u)
            factor = (erf_u - (2.0 / math.sqrt(math.pi)) * u * torch.exp(-u ** 2)) / (dist ** 3)
            inv_r3 = torch.where(dist > 1e-6, factor, torch.zeros_like(factor))
        else:
            raise ValueError(f"Unknown softening: {softening}")

        inv_r3.fill_diagonal_(0.0)

        # Active bodies and the central well exert gravity; tracers and dead particles do not
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

    disk_rows = np.arange(1, 1 + sd.DISK_N)
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
            for lo, hi in sd.R_BINS:
                b = sel & (r0 >= lo) & (r0 < hi)
                if b.sum() >= 5:
                    tb = Lz[:, b].sum(axis=1)
                    out_data[f"L_{tag}_r{lo}_{hi}_change_pct"] = float(100.0 * (tb[-1] / tb[0] - 1.0))

    return out_data


def main():
    print("=================================================================")
    print("      NEWTONIAN CONTROL FOR TEGR GENESIS SPINNING DISK           ")
    print("   Comparing Instant Action-at-a-Distance vs Wave Wake Gravity   ")
    print("=================================================================\n")

    wave_results = load_wave_results()
    if not wave_results or "gauss" not in wave_results:
        print("[ERROR] Could not find results/spinning_disk_v01a_gauss.json. Run spinning_disk.py first.")
        return

    gauss_a0 = wave_results["gauss"]["alpha"]["0.0"]
    r_curve = np.array(gauss_a0["r_curve"])
    v_launch_disk = np.array(gauss_a0["v_launch_disk"])
    v_launch_core = np.array(gauss_a0["v_launch_core_only"])

    speed_disk = lambda r: np.interp(r, r_curve, v_launch_disk)
    speed_core = lambda r: np.interp(r, r_curve, v_launch_core)

    disk = sd.build_disk()
    rng = np.random.default_rng(sd.SEED + 1)
    mixed = list(rng.permutation([1] * (sd.DISK_N // 2) + [-1] * (sd.DISK_N - sd.DISK_N // 2)))

    plans = [
        ("P", "active", [1] * sd.DISK_N, speed_disk),
        ("C", "active", mixed, speed_disk),
        ("T", "tracer", [1] * sd.DISK_N, speed_core),
    ]

    out_summary = {"softening": {}}

    for soft in ("gauss", "plummer"):
        print(f"\n--- Running Instant Newtonian Gravity ({soft} softening, sigma={SIGMA}) ---")
        out_summary["softening"][soft] = {}
        for tag, kind, senses, sf in plans:
            res = run_newton(disk, mixed, tag, kind, senses, sf, softening=soft, sigma=SIGMA)
            out_summary["softening"][soft][tag] = res
            msg = []
            for k in ("L_with_change_pct", "L_against_change_pct"):
                if k in res:
                    msg.append(f"{k.split('_')[1]} {res[k]:+.2f}%")
            print(f"  {tag}: spin {', '.join(msg)} | alive {100 * res['disk_alive_frac']:.1f}% | "
                  f"spread {res['radius_spread_end']:.3f} | {res['wall_seconds']}s")

    # Save JSON summary
    with open(os.path.join(OUT, "newton_control.json"), "w") as f:
        json.dump(out_summary, f, indent=1)
    print("\nSaved results/newton_control.json")

    # ---------------------------------------------------------------- Comparison Printout
    newt_g = out_summary["softening"]["gauss"]
    wave_g = gauss_a0["runs"]

    print("\n" + "=" * 78)
    print("                  HEAD-TO-HEAD COMPARISON (alpha = 0)")
    print("=" * 78)
    print(f"{'Run':<6} | {'Instant Newton (Gaussian)':<24} | {'Wave Gravity (Gaussian v0.1a)':<26} | {'Difference (Wake?)':<18}")
    print("-" * 78)

    # Case P
    p_nw_spin = newt_g["P"]["L_with_change_pct"]
    p_wv_spin = wave_g["P"]["L_with_change_pct"]
    p_diff = p_wv_spin - p_nw_spin
    print(f"{'P':<6} | spin: {p_nw_spin:+6.1f}%, alive: {100*newt_g['P']['disk_alive_frac']:.0f}%  | "
          f"spin: {p_wv_spin:+6.1f}%, alive: {100*wave_g['P']['disk_alive_frac']:.0f}%    | "
          f"{p_diff:+6.1f}% spin loss")

    # Case C
    c_nw_w = newt_g["C"]["L_with_change_pct"]
    c_nw_a = newt_g["C"]["L_against_change_pct"]
    c_wv_w = wave_g["C"]["L_with_change_pct"]
    c_wv_a = wave_g["C"]["L_against_change_pct"]
    print(f"{'C (w)':<6} | spin: {c_nw_w:+6.1f}%, alive: {100*newt_g['C']['disk_alive_frac']:.0f}%  | "
          f"spin: {c_wv_w:+6.1f}%, alive: {100*wave_g['C']['disk_alive_frac']:.0f}%    | "
          f"{c_wv_w - c_nw_w:+6.1f}%")
    print(f"{'C (ag)':<6} | spin: {c_nw_a:+6.1f}%, alive: {100*newt_g['C']['disk_alive_frac']:.0f}%  | "
          f"spin: {c_wv_a:+6.1f}%, alive: {100*wave_g['C']['disk_alive_frac']:.0f}%    | "
          f"{c_wv_a - c_nw_a:+6.1f}%")

    # Case T
    t_nw_spin = newt_g["T"]["L_with_change_pct"]
    t_wv_spin = wave_g["T"]["L_with_change_pct"]
    print(f"{'T':<6} | spin: {t_nw_spin:+6.2f}%, alive: {100*newt_g['T']['disk_alive_frac']:.0f}%  | "
          f"spin: {t_wv_spin:+6.2f}%, alive: {100*wave_g['T']['disk_alive_frac']:.0f}%    | "
          f"{t_wv_spin - t_nw_spin:+6.2f}%")
    print("=" * 78)

    # ---------------------------------------------------------------- Visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

    # Panel 1: Time evolution L(t)/L0 for Newton vs Wave
    t_arr = np.array(newt_g["P"]["t"])
    ax[0].plot(t_arr, newt_g["P"]["L_with_over_start"], "k--", lw=2, label="Newton P (Instant)")
    ax[0].plot(t_arr, wave_g["P"]["L_with_over_start"], "tab:red", lw=2, label="Genesis Wave P (c=65)")

    ax[0].plot(t_arr, newt_g["C"]["L_with_over_start"], "k:", lw=1.8, label="Newton C with")
    ax[0].plot(t_arr, wave_g["C"]["L_with_over_start"], "tab:purple", lw=1.8, label="Genesis Wave C with")

    ax[0].plot(t_arr, newt_g["T"]["L_with_over_start"], "k-.", lw=1.5, label="Newton T (control)")
    ax[0].plot(t_arr, wave_g["T"]["L_with_over_start"], "tab:gray", lw=1.5, label="Genesis Wave T (control)")

    ax[0].axhline(1.0, color="0.4", ls=":")
    ax[0].set_title("Disk Spin Over Time: Newton (Instant) vs Genesis (Wave)")
    ax[0].set_xlabel("Time since release")
    ax[0].set_ylabel("Spin / Starting Spin (L_z / L_0)")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].grid(True, alpha=0.3)

    # Panel 2: Radial bins breakdown
    labels = ["Inner (8-25)", "Middle (25-45)", "Outer (45-75)"]
    x_idx = np.arange(len(labels))
    width = 0.35

    nw_p_bins = [newt_g["P"].get(f"L_with_r{lo}_{hi}_change_pct", 0) for lo, hi in sd.R_BINS]
    wv_p_bins = [wave_g["P"].get(f"L_with_r{lo}_{hi}_change_pct", 0) for lo, hi in sd.R_BINS]

    ax[1].bar(x_idx - width/2, nw_p_bins, width, label="Newton P (Instant)", color="0.3")
    ax[1].bar(x_idx + width/2, wv_p_bins, width, label="Genesis Wave P (c=65)", color="tab:red")

    ax[1].set_xticks(x_idx)
    ax[1].set_xticklabels(labels)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("Spin Change by Radial Zone (Case P)")
    ax[1].set_ylabel("Spin Change (%)")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3, axis="y")

    # Panel 3: Direct Summary Comparison
    categories = ["P (all with)", "C (with)", "C (against)", "T (control)"]
    x3 = np.arange(len(categories))
    w3 = 0.25

    # Old buggy Genesis engine:
    old_p_spin = +9.1
    old_c_with = +2.8
    old_c_ag = -15.0
    old_t = 0.0
    old_vals = [old_p_spin, old_c_with, old_c_ag, old_t]

    wave_vals = [wave_g["P"]["L_with_change_pct"], wave_g["C"]["L_with_change_pct"],
                 wave_g["C"]["L_against_change_pct"], wave_g["T"]["L_with_change_pct"]]
    newt_vals = [newt_g["P"]["L_with_change_pct"], newt_g["C"]["L_with_change_pct"],
                 newt_g["C"]["L_against_change_pct"], newt_g["T"]["L_with_change_pct"]]

    ax[2].bar(x3 - w3, old_vals, w3, label="Old Engine (Buggy)", color="tab:orange", alpha=0.8)
    ax[2].bar(x3, wave_vals, w3, label="Fixed Engine (Wave c=65)", color="tab:red", alpha=0.9)
    ax[2].bar(x3 + w3, newt_vals, w3, label="Newton Control (Instant c=inf)", color="tab:blue", alpha=0.9)

    ax[2].set_xticks(x3)
    ax[2].set_xticklabels(categories, fontsize=8)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_title("Spin Loss: Bug vs Wave vs Instant Gravity")
    ax[2].set_ylabel("Net Spin Change (%)")
    ax[2].legend(fontsize=8)
    ax[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    png_path = os.path.join(OUT, "newton_control.png")
    plt.savefig(png_path, dpi=160)
    print(f"Saved {png_path}\n")


if __name__ == "__main__":
    main()
