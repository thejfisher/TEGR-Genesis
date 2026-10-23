"""TEGR Genesis — Double-Slit Quantum Interference Runner.

Loads the self-contained experiment preset 'experiments/double_slit.toml',
simulates wave-barrier interaction, slit diffraction, and particle guidance
on the discrete 3D lattice, and plots the detector screen interference fringes.

Usage:
    python run_double_slit.py [path_to_toml]
"""

import datetime
import math
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core.engine import Engine, SimSettings
from core.state_manager import load_experiment
from core.exporter import sindy_fit, export_sindy


def run_double_slit(toml_path=None):
    if toml_path is None:
        toml_path = os.path.join(HERE, "experiments", "double_slit.toml")

    print("\n" + "=" * 74)
    print("      TEGR Genesis -- Double-Slit Quantum Interference Benchmark")
    print("=" * 74)
    print(f" Loading Preset: {toml_path}")

    settings_dict, entities, meta = load_experiment(toml_path)
    settings = SimSettings.from_dict(settings_dict)

    n_anchors = sum(1 for e in entities if e.kind == "anchor")
    n_defects = sum(1 for e in entities if e.kind == "defect")
    n_active = sum(1 for e in entities if e.kind == "active")

    print(f" Experiment Name: {meta.get('name', 'Double Slit')}")
    print(f" Description    : {meta.get('notes', '')}")
    print(f" Grid Resolution: {settings.grid_resolution}^3 cells (box: {settings.box_size:.1f})")
    print(f" Entities Loaded: {len(entities)} ({n_anchors} wall anchors, {n_defects} wave seeds, {n_active} beam particles)")
    print("=" * 74 + "\n")

    # Run simulation
    print("[1/3] Running 3D discrete lattice simulation...", flush=True)
    t0 = time.time()
    engine = Engine(settings)
    res = engine.run(entities)
    wall_s = time.time() - t0
    print(f"      Simulation completed on {res.device} in {wall_s:.1f} s ({len(res.t)} recorded steps).", flush=True)

    # Analyze beam particle trajectories
    print("[2/3] Analyzing slit diffraction and particle trajectories...", flush=True)
    bodies = [e for e in entities if e.kind != "defect"]
    active_indices = [i for i, b in enumerate(bodies) if b.kind == "active"]

    t = res.t
    final_pos = res.pos[-1, active_indices, :]
    
    passed_mask = final_pos[:, 0] > 5.0
    reflected_mask = final_pos[:, 0] < -15.0
    in_flight_mask = ~passed_mask & ~reflected_mask

    n_passed = int(np.sum(passed_mask))
    n_reflected = int(np.sum(reflected_mask))
    n_in_flight = int(np.sum(in_flight_mask))
    total_beam = len(active_indices)

    y_screen = final_pos[passed_mask, 1].tolist()

    out_dir = os.path.join(HERE, "results", "double_slit")
    os.makedirs(out_dir, exist_ok=True)

    # Generate 1D Histogram Plot
    print("[3/3] Generating interference fringe analysis & histogram plot...", flush=True)
    plot_path = os.path.join(out_dir, "double_slit_histogram.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left panel: Top-down 2D Trajectories (x vs y)
        ax1.set_facecolor("#1e1e1e")
        fig.patch.set_facecolor("#121212")

        # Plot wall barrier
        wall_anchors_y = [e.y for e in entities if e.kind == "anchor"]
        ax1.scatter([0.0] * len(wall_anchors_y), wall_anchors_y, color="#888888", s=40, label="Wall Anchors", zorder=3)
        # Indicate slits
        ax1.axhspan(3.0, 9.0, color="#00e5ff", alpha=0.15, label="Upper Slit")
        ax1.axhspan(-9.0, -3.0, color="#00e5ff", alpha=0.15, label="Lower Slit")

        # Plot trajectories of beam particles
        for idx in active_indices:
            traj_x = res.pos[:, idx, 0]
            traj_y = res.pos[:, idx, 1]
            color = "#00e5ff" if traj_x[-1] > 5.0 else "#ff5252"
            ax1.plot(traj_x, traj_y, color=color, alpha=0.5, linewidth=1.2)

        ax1.set_title("2D Particle Trajectories & Barrier Scattering", color="white", fontsize=12, fontweight="bold")
        ax1.set_xlabel("X (Propagation Axis)", color="white")
        ax1.set_ylabel("Y (Transverse Position)", color="white")
        ax1.tick_params(colors="white")
        for spine in ax1.spines.values():
            spine.set_color("#444444")
        ax1.grid(color="#333333", linestyle="--", alpha=0.5)
        ax1.legend(facecolor="#1e1e1e", edgecolor="#444444", labelcolor="white", loc="upper left")

        # Right panel: 1D Y-Screen Hit Distribution
        ax2.set_facecolor("#1e1e1e")
        if y_screen:
            bins = np.linspace(-35, 35, 35)
            ax2.hist(y_screen, bins=bins, color="#00e5ff", edgecolor="white", alpha=0.75, linewidth=0.8)
            ax2.axvline(0.0, color="#ffb000", linestyle="--", linewidth=1.5, label="Central Axis (y=0)")
        ax2.set_title(f"Detector Screen Distribution (Hits: {n_passed}/{total_beam})", color="white", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Screen Y Position", color="white")
        ax2.set_ylabel("Particle Count", color="white")
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_color("#444444")
        ax2.grid(color="#333333", linestyle="--", alpha=0.5)
        ax2.legend(facecolor="#1e1e1e", edgecolor="#444444", labelcolor="white")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"      Fringe plot saved: {plot_path}")
    except Exception as exc:
        print(f"      [Warning] Could not generate plot: {exc}")

    # Text report
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "=" * 80,
        "   TEGR Genesis -- Double-Slit Quantum Interference Report",
        "=" * 80,
        f" Date & Time        : {timestamp}",
        f" Compute Device     : {res.device}",
        f" Grid Resolution    : {settings.grid_resolution}^3 (dx={settings.dx})",
        f" Total Ticks        : {settings.total_ticks} (dt={settings.dt})",
        f" Execution Time     : {wall_s:.2f} s",
        "",
        "--- BEAM TRANSMISSION METRICS ---",
        f" Total Beam Particles Launched : {total_beam}",
        f" Transmitted Through Slits     : {n_passed} ({n_passed/total_beam*100:.1f}%)",
        f" Reflected / Blocked by Wall   : {n_reflected} ({n_reflected/total_beam*100:.1f}%)",
        f" In-Flight / Interaction Zone  : {n_in_flight} ({n_in_flight/total_beam*100:.1f}%)",
        "",
        "--- DIFFRACTION & FRINGE GEOMETRY ---",
        f" Slit Separation (Center-Center): ~12.0 grid units (slits at y = +/- 6.0)",
        f" Central Barrier Width          : 6.0 grid units (anchors y in [-3.0, 3.0])",
    ]
    if y_screen:
        report_lines.extend([
            f" Transmitted Y Range            : [{min(y_screen):.2f}, {max(y_screen):.2f}]",
            f" Transverse Spread (std dev)    : {float(np.std(y_screen)):.2f}",
        ])
    report_lines.extend([
        "",
        f" Histogram Plot     : {plot_path}",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)
    report_file = os.path.join(out_dir, "double_slit_results.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text + "\n")
    return res


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_double_slit(path)
