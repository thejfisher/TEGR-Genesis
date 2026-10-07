"""Exports for papers and for PySINDy (step 6).

USER RULE: every saved picture has a WHITE background with dark lines and text (paper-ready).
The app stays black on screen; pictures here are re-drawn with matplotlib in a light "paper" style.

    export_graphs(res, folder)  -> telemetry, 2D paths, field slice, 3D paths   (PNG 300 dpi + PDF)
    export_raw(res, folder)     -> CSV (long + wide), NPY, NPZ, experiment.toml, README.txt
    export_sindy(res, folder)   -> sindy_results.txt + sindy_fit.png/.pdf
    export_all(res, folder)
"""

from __future__ import annotations

import csv
import datetime as _dt
import math
from pathlib import Path

import numpy as np
import matplotlib
from matplotlib.figure import Figure

from core.entities import MOVING_KINDS
from core.state_manager import APP_ROOT, save_experiment

EXPORTS_DIR = APP_ROOT / "exports"

PAPER = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.edgecolor": "white", "axes.edgecolor": "#222222", "axes.labelcolor": "#111111",
    "xtick.color": "#222222", "ytick.color": "#222222", "text.color": "#111111",
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#CCCCCC",
    "lines.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42,
}
DPI = 300
KIND_PAPER = {  # kind -> (marker, size, colour)
    "well": ("o", 70, "#0057A8"),
    "defect": ("*", 90, "#2E9BD6"),
    "anchor": ("s", 45, "#555555"),
    "active": ("o", 30, "#D17A00"),
    "tracer": ("o", 18, "#333333"),
}


def default_folder(name="experiment"):
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (name or "experiment")).strip() or "experiment"
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return EXPORTS_DIR / f"{safe.replace(' ', '_')}_{stamp}"


# ----------------------------------------------------------------------------- helpers
def _labels(res):
    n = len(res.kinds)
    return [str(d.get("label", f"body {j}")) for j, d in enumerate(res.entities[:n])]


def _save(fig, folder, stem):
    out = []
    for ext in ("png", "pdf"):
        p = Path(folder) / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
        out.append(p)
    return out


def _warmup_time(res):
    s = res.settings
    return float(s.get("warmup_ticks", 0)) * float(s.get("dt", 0))


def _path_colours(n):
    cmap = matplotlib.colormaps["tab10"]
    return [cmap(i % 10) for i in range(n)]


def _box(res):
    s = res.settings
    G, dx = int(s["grid_resolution"]), float(s["dx"])
    half = G * dx / 2.0
    sponge = 0 if s.get("physics_rule") == "legacy2600" else int(s.get("sponge_width", 0))
    return G, dx, half, sponge


# ----------------------------------------------------------------------------- graphs
def fig_telemetry(res):
    t = np.asarray(res.t)
    tel = res.telemetry
    tw = _warmup_time(res)
    with matplotlib.rc_context(PAPER):
        fig = Figure(figsize=(6.5, 7.0))
        axs = fig.subplots(3, 1, sharex=True)
        ax = axs[0]
        ax.plot(t, tel["total_energy"], color="#0057A8", label="total")
        if "field_energy" in tel:
            ax.plot(t, tel["field_energy"], color="#777777", lw=1.0, ls="--", label="field")
        if "particle_kinetic" in tel:
            ax.plot(t, tel["particle_kinetic"], color="#D17A00", lw=1.0, label="particles (kinetic)")
        ax.set_ylabel("Energy E")
        ax.set_title("Total energy")
        ax.legend(loc="best", fontsize=8)
        axs[1].plot(t, tel["peak_phi"], color="#0057A8")
        axs[1].set_ylabel("max |φ|")
        axs[1].set_title("Peak field |φ|")
        rh = np.asarray(tel["horizon_radius"])
        axs[2].plot(t, rh, color="#0057A8")
        axs[2].set_ylabel("r_h (grid units)")
        axs[2].set_title("Horizon radius r_h")
        if not np.any(rh > 0):
            axs[2].set_ylim(0, 1)
            axs[2].text(0.5, 0.5, "no horizon formed", transform=axs[2].transAxes,
                        ha="center", va="center", color="#666666")
        axs[2].set_xlabel("time")
        for a in axs:
            if tw > 0 and len(t) and tw < t[-1]:
                a.axvspan(t[0], tw, color="#EEEEEE", zorder=0, lw=0)
        if tw > 0 and len(t):
            axs[0].text(min(tw, t[-1]) / 2, 0.04, "warm-up\n(bodies held)", transform=axs[0].get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7, color="#777777")
        fig.tight_layout()
    return fig


def fig_paths_2d(res):
    G, dx, half, sponge = _box(res)
    labels = _labels(res)
    with matplotlib.rc_context(PAPER):
        fig = Figure(figsize=(6.0, 6.0))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color="#222222", lw=1.2)
        if sponge > 0 and half - sponge * dx > 0:
            h2 = half - sponge * dx
            ax.plot([-h2, h2, h2, -h2, -h2], [-h2, -h2, h2, h2, -h2], color="#999999", lw=0.8, ls="--",
                    label="sponge edge")
        cols = _path_colours(len(res.kinds))
        for j, kind in enumerate(res.kinds):
            if kind in MOVING_KINDS:
                ok = res.alive[:, j]
                last = int(np.nonzero(ok)[0][-1]) if ok.any() else 0
                xy = res.pos[: last + 1, j, :2]
                ax.plot(xy[:, 0], xy[:, 1], color=cols[j], lw=1.2, label=labels[j])
                ax.plot(xy[0, 0], xy[0, 1], "o", mfc="white", mec=cols[j], ms=5)
                ax.plot(xy[-1, 0], xy[-1, 1], "o", color=cols[j], ms=5)
            else:
                mk, sz, c = KIND_PAPER[kind]
                ax.scatter([res.pos[0, j, 0]], [res.pos[0, j, 1]], marker=mk, s=sz, color=c,
                           edgecolors="black", linewidths=0.6, zorder=5, label=labels[j])
        for d in res.entities[len(res.kinds):]:
            if d.get("kind") == "defect":
                mk, sz, c = KIND_PAPER["defect"]
                ax.scatter([d["x"]], [d["y"]], marker=mk, s=sz, color=c, edgecolors="black",
                           linewidths=0.5, zorder=5, label=d.get("label", "defect"))
        ax.set_aspect("equal")
        ax.set_xlim(-half * 1.03, half * 1.03)
        ax.set_ylim(-half * 1.03, half * 1.03)
        ax.set_xlabel("x (grid units)")
        ax.set_ylabel("y (grid units)")
        ax.set_title("Paths on the z = 0 plane  (open circle = start, dot = end)")
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
    return fig


def fig_field_slice(res):
    G, dx, half, _ = _box(res)
    if res.field_frames is None or not len(res.field_frames):
        return None
    frame = np.asarray(res.field_frames[-1], float)
    with matplotlib.rc_context(PAPER):
        fig = Figure(figsize=(6.4, 5.4))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(frame.T, origin="lower", extent=[-half, half, -half, half], cmap="viridis",
                       interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("φ on z = 0 (end of run)")
        for j, kind in enumerate(res.kinds):
            if kind in MOVING_KINDS:
                ok = res.alive[:, j]
                last = int(np.nonzero(ok)[0][-1]) if ok.any() else 0
                xy = res.pos[: last + 1, j, :2]
                ax.plot(xy[:, 0], xy[:, 1], color="white", lw=0.8)
        ax.set_xlabel("x (grid units)")
        ax.set_ylabel("y (grid units)")
        ax.set_title(f"Field φ at t = {float(res.frame_t[-1]):.2f}  (white lines = paths)")
        ax.grid(False)
        fig.tight_layout()
    return fig


def fig_paths_3d(res):
    G, dx, half, _ = _box(res)
    labels = _labels(res)
    with matplotlib.rc_context(PAPER):
        fig = Figure(figsize=(6.4, 6.0))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((1, 1, 1, 1))
        if res.field_frames is not None and len(res.field_frames):
            xs = -half + (np.arange(G) + 0.5) * dx
            X, Y = np.meshgrid(xs, xs, indexing="ij")
            ax.contourf(X, Y, np.asarray(res.field_frames[-1], float), zdir="z", offset=-half,
                        levels=14, cmap="Blues", alpha=0.8)
        cols = _path_colours(len(res.kinds))
        for j, kind in enumerate(res.kinds):
            if kind in MOVING_KINDS:
                ok = res.alive[:, j]
                last = int(np.nonzero(ok)[0][-1]) if ok.any() else 0
                p = res.pos[: last + 1, j, :]
                ax.plot(p[:, 0], p[:, 1], p[:, 2], color=cols[j], lw=1.2, label=labels[j])
                ax.scatter([p[-1, 0]], [p[-1, 1]], [p[-1, 2]], color=cols[j], s=12)
            else:
                mk, sz, c = KIND_PAPER[kind]
                ax.scatter([res.pos[0, j, 0]], [res.pos[0, j, 1]], [res.pos[0, j, 2]], marker=mk, s=sz,
                           color=c, edgecolors="black", linewidths=0.5, label=labels[j])
        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(-half, half)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=24, azim=-55)
        ax.set_title("3D paths  (floor: field φ on z = 0, end of run)")
        ax.legend(loc="upper left", fontsize=7)
    return fig


def export_graphs(res, folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    files = []
    for stem, maker in (("telemetry", fig_telemetry), ("paths_2d", fig_paths_2d),
                        ("field_slice", fig_field_slice), ("paths_3d", fig_paths_3d)):
        fig = maker(res)
        if fig is not None:
            files += _save(fig, folder, stem)
    return files


# ----------------------------------------------------------------------------- raw data
def export_raw(res, folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    t = np.asarray(res.t)
    N = len(res.kinds)
    labels = _labels(res)
    files = []

    # long format: one row per body per time
    p = folder / "trajectories_long.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "body", "label", "kind", "x", "y", "z", "vx", "vy", "vz", "alive"])
        for i in range(len(t)):
            for j in range(N):
                w.writerow([f"{t[i]:.6g}", j, labels[j], res.kinds[j],
                            *(f"{v:.9g}" for v in res.pos[i, j]), *(f"{v:.9g}" for v in res.vel[i, j]),
                            int(res.alive[i, j])])
    files.append(p)

    # wide format: one row per time (what PySINDy wants)
    p = folder / "trajectories_wide.csv"
    cols = ["t"]
    for j in range(N):
        cols += [f"{c}_{j}" for c in ("x", "y", "z", "vx", "vy", "vz")]
    wide = np.column_stack([t] + [np.column_stack([res.pos[:, j, :], res.vel[:, j, :]]) for j in range(N)]) \
        if N else t[:, None]
    np.savetxt(p, wide, delimiter=",", header=",".join(cols), comments="", fmt="%.9g")
    files.append(p)

    # telemetry
    p = folder / "telemetry.csv"
    keys = list(res.telemetry.keys())
    np.savetxt(p, np.column_stack([t] + [np.asarray(res.telemetry[k]) for k in keys]), delimiter=",",
               header=",".join(["t"] + keys), comments="", fmt="%.9g")
    files.append(p)

    # numpy
    for name, arr in (("t", t), ("positions", res.pos), ("velocities", res.vel), ("alive", res.alive)):
        q = folder / f"{name}.npy"
        np.save(q, arr)
        files.append(q)
    extra = {}
    if res.final_phi is not None and res.final_phi.size <= 128 ** 3:
        extra["final_phi"] = res.final_phi
    q = folder / "run_all.npz"
    np.savez_compressed(q, t=t, positions=res.pos, velocities=res.vel, alive=res.alive,
                        kinds=np.array(res.kinds), labels=np.array(labels),
                        frame_t=res.frame_t, field_frames_z0=res.field_frames,
                        **{f"telemetry_{k}": np.asarray(v) for k, v in res.telemetry.items()}, **extra)
    files.append(q)

    # the experiment itself (so the data can be re-made)
    files.append(save_experiment(folder / "experiment.toml", res.settings, res.entities,
                                 name="Exported run", notes=res.message))

    readme = folder / "README.txt"
    readme.write_text(
        "TEGR Genesis export\n"
        f"Made: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"Run: {res.message}  ({res.ticks_done} ticks, {res.wall_seconds:.1f} s on {res.device})\n\n"
        "Bodies (index: label, kind):\n" +
        "".join(f"  {j}: {labels[j]}, {res.kinds[j]}\n" for j in range(N)) +
        "\nFiles\n"
        "  trajectories_wide.csv  one row per time: t, x_j, y_j, z_j, vx_j, vy_j, vz_j   (PySINDy-ready)\n"
        "  trajectories_long.csv  one row per body per time (alive = 0 after leaving the box)\n"
        "  telemetry.csv          energy parts, peak |phi|, horizon radius over time\n"
        "  t.npy, positions.npy (T,N,3), velocities.npy (T,N,3), alive.npy (T,N)\n"
        "  run_all.npz            everything above + field slices (z = 0) [+ final 3D field if small]\n"
        "  experiment.toml        settings + items; open it in TEGR Genesis to re-run\n\n"
        f"Moving bodies are held still until t = {_warmup_time(res):g} (warm-up).\n",
        encoding="utf-8")
    files.append(readme)
    return files


# ----------------------------------------------------------------------------- PySINDy
def sindy_fit(res, threshold=0.05):
    """Fit each moving body's motion with PySINDy. Returns (report text, list of per-body fit dicts)."""
    import pysindy as ps

    s = res.settings
    t = np.asarray(res.t, float)
    labels = _labels(res)
    tw = _warmup_time(res)
    legacy = s.get("physics_rule") == "legacy2600"
    K, Gs = float(s.get("field_force", 1.0)), float(s.get("source_strength", 1.0))
    src = [j for j, k in enumerate(res.kinds) if k in ("well", "anchor")]
    masses = [float(res.entities[j].get("mass", 0.0)) for j in range(len(res.kinds))]

    lines = [
        "TEGR Genesis — PySINDy fit of the body motion",
        f"Made {_dt.datetime.now().isoformat(timespec='seconds')}  ·  run: {res.message}",
        "",
        "What was fitted",
        "  State per moving body: x, y, vx, vy  (z ignored; bodies start on the z = 0 plane).",
        "  Derivatives: x' = vx, y' = vy from the record; vx', vy' = numerical slope of the recorded speed.",
        "  Only times after the warm-up, while the body is inside the box.",
        "  Candidate terms: 1, x, y, vx, vy, and for each fixed well/anchor k the gravity-shaped terms",
        "    gx_k = (X_k - x)/r_k^3 ,  gy_k = (Y_k - y)/r_k^3",
        "  NOTE: these candidate terms are a choice of the ANALYSIS only. The simulation itself never uses",
        "  them: bodies only feel the slope of the field φ.",
        f"  Optimizer: STLSQ, threshold {threshold} (on normalized columns).",
        "",
        "Two fits per body",
        "  A. Gravity-only: vx', vy' using ONLY the gravity-shaped terms. A direct test of Newton's law.",
        "  B. Open search: all candidate terms; PySINDy picks which to keep.",
        "  If the terms look alike along the path (e.g. on a circle x and (X-x)/r^3 move together),",
        "  fit B cannot tell them apart and may invent extra terms. The report warns when that happens.",
        "",
    ]
    fits = []
    for j, kind in enumerate(res.kinds):
        if kind not in MOVING_KINDS:
            continue
        m = (t > tw + 1e-9) & res.alive[:, j]
        idx = np.nonzero(m)[0]
        if len(idx) < 30:
            lines.append(f"[{labels[j]}] skipped: only {len(idx)} usable samples.")
            continue
        tt = t[idx]
        x, y = res.pos[idx, j, 0], res.pos[idx, j, 1]
        vx, vy = res.vel[idx, j, 0], res.vel[idx, j, 1]
        ax_, ay_ = np.gradient(vx, tt), np.gradient(vy, tt)
        X = np.column_stack([x, y, vx, vy])
        Xd = np.column_stack([vx, vy, ax_, ay_])
        U, unames = [], []
        for k in src:
            dxk, dyk = res.pos[0, k, 0] - x, res.pos[0, k, 1] - y
            r3 = np.maximum(np.hypot(dxk, dyk), 1e-6) ** 3
            U += [dxk / r3, dyk / r3]
            unames += [f"gx{k}", f"gy{k}"]
        U = np.column_stack(U) if U else None
        names = ["x", "y", "vx", "vy"] + unames
        model = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold, normalize_columns=True),
                         feature_library=ps.PolynomialLibrary(degree=1))
        model.fit(X, t=tt, x_dot=Xd, u=U, feature_names=names)
        eqs = model.equations(precision=4)
        try:
            score = float(model.score(X, t=tt, x_dot=Xd, u=U))
        except Exception:
            score = float("nan")
        pred = model.predict(X, u=U)
        lines.append(f"[{labels[j]}]  ({kind}, {len(idx)} samples)")

        # --- A. gravity-only fit (PySINDy STLSQ on just the gravity terms)
        if U is not None:
            gA = ps.STLSQ(threshold=0.0)
            gA.fit(U, Xd[:, 2:4])
            cA = np.asarray(gA.coef_)            # (2, n_u)
            predA = U @ cA.T
            ss = ((Xd[:, 2:4] - Xd[:, 2:4].mean(axis=0)) ** 2).sum()
            r2A = 1.0 - ((Xd[:, 2:4] - predA) ** 2).sum() / ss if ss > 0 else float("nan")
            lines.append(f"  A. Gravity-only fit  (R² = {r2A:.4f})")
            for k_i, k in enumerate(src):
                fx, fy = cA[0, 2 * k_i], cA[1, 2 * k_i + 1]
                txt = f"     source {labels[k]}: strength {fx:.4g} (x) / {fy:.4g} (y)"
                if not legacy:
                    expect = K * Gs * masses[k]
                    txt += f"   Newton K·G·M = {expect:.4g}   ratio {0.5 * (fx + fy) / expect:.3f}"
                lines.append(txt)

        # --- B. open search
        lines.append(f"  B. Open search  (R² = {score:.4f})")
        for lhs, e in zip(("x'", "y'", "vx'", "vy'"), eqs):
            lines.append(f"     {lhs:4s} = {e}")
        Th = np.column_stack([np.ones(len(idx)), X] + ([U] if U is not None else []))
        Th = Th / np.maximum(np.linalg.norm(Th, axis=0), 1e-30)
        cond = float(np.linalg.cond(Th))
        n_terms = int(np.count_nonzero(np.asarray(model.coefficients())[2:4]))
        gain = score - r2A if U is not None else float("nan")
        if U is not None:
            lines.append(f"     Fit B uses {n_terms} terms for vx', vy' and improves R² by {gain:+.4f} over fit A.")
        if U is not None and gain < 0.01 and n_terms > 2 * len(src):
            lines.append("     WARNING: the extra terms buy almost no accuracy, so they are most likely overfitting")
            lines.append("     (PySINDy trading look-alike terms against each other). Trust fit A for this body.")
        elif cond > 1e3:
            lines.append(f"     WARNING: candidate terms are near-duplicates on this path (condition number {cond:.2g}).")
            lines.append("     Fit B cannot separate them, so its extra terms are not trustworthy. Use fit A.")
        lines.append("")
        fits.append(dict(body=j, label=labels[j], t=tt, actual=Xd, predicted=pred, score=score, equations=eqs,
                         predicted_A=(predA if U is not None else None)))
    if not fits:
        lines.append("No moving bodies with enough data after the warm-up.")
    return "\n".join(lines), fits


def fig_sindy(fits):
    if not fits:
        return None
    n = min(len(fits), 6)
    with matplotlib.rc_context(PAPER):
        fig = Figure(figsize=(7.0, 2.0 * n + 0.6))
        axs = np.atleast_2d(fig.subplots(n, 2, squeeze=False))
        for r, f in enumerate(fits[:n]):
            for c, (col, name) in enumerate(((2, "vx' (acceleration x)"), (3, "vy' (acceleration y)"))):
                a = axs[r, c]
                a.plot(f["t"], f["actual"][:, col], color="#222222", lw=1.2, label="simulation")
                if f.get("predicted_A") is not None:
                    a.plot(f["t"], f["predicted_A"][:, col - 2], color="#0057A8", lw=1.0, ls="--",
                           label="fit A (gravity-only)")
                a.plot(f["t"], f["predicted"][:, col], color="#D17A00", lw=0.9, ls=":", label="fit B (open search)")
                a.set_title(f"{f['label']}: {name}", fontsize=8)
                if r == n - 1:
                    a.set_xlabel("time")
                if r == 0 and c == 0:
                    a.legend(fontsize=7)
        fig.tight_layout()
    return fig


def export_sindy(res, folder, threshold=0.05):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    text, fits = sindy_fit(res, threshold)
    p = folder / "sindy_results.txt"
    p.write_text(text, encoding="utf-8")
    files = [p]
    fig = fig_sindy(fits)
    if fig is not None:
        files += _save(fig, folder, "sindy_fit")
    return files, text


def export_all(res, folder):
    files = export_graphs(res, folder)
    files += export_raw(res, folder)
    f2, _ = export_sindy(res, folder)
    return files + f2
