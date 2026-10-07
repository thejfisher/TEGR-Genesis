# TEGR Genesis — Handoff Notes

> **IMPORTANT:** The user is dyslexic and does not code. The agent writes, runs, and checks all files.
> Never ask the user to copy, paste, or edit code. Keep explanations short and plain.

## Locations
| What | Path |
|---|---|
| New app (TEGR Genesis) | `Z:\TEGR Genesis` |
| Splash GIF | `Z:\TEGR Genesis\assets\TEGR_retro_logo_intro.gif` |
| TEGR 2600 (TOML presets, user's favorite engine) | `Z:\TryTri\TEGR 2600` |
| TEGR Collider (has 3D view to keep) | `Z:\rerun` (also `Z:\TEGR Collider`) |
| Newest physics work (MS38) | `Z:\TryTri\Manuscripts\38_Yang-Mills` (`field_harness.py`, `MS38_Notes.md`) |
| MS37 work | `Z:\TryTri\Manuscripts\37_Copi et al` |

## What the app must do
- Flat 2D grid: drag-and-drop gravity wells, wave defects, particles
- Particle types: static anchors, active bodies, passive tracers (N-body)
- Set particles on collision courses
- Not real-time: Setup → Compute → Analyze (plot paths after the run)
- 3D view with X, Y, Z reference grid (keep Collider's 3D view)
- Telemetry graphs (energy, peak |φ|, horizon radius r_h)
- Export: graphs (PNG/PDF), raw data for PySINDy (CSV/NPY), PySINDy results
  - **USER RULE (2026-10-07): every saved picture (grids, paths, graphs, 3D) must have a WHITE background,
    dark lines/text, ready for papers.** The app stays black on screen; only exports switch to a light "paper" style.
- Save/load experiments as self-contained TOML files
- Hardware picker: Auto / CUDA / Mac (MPS) / CPU, with fallback to CPU
- Windows, Linux, Mac. Desktop app, not web.
- Look: SEGA Genesis — black, silver, accent blue `#00A2FF`
- Splash: play the GIF exactly once, then open the main window

## Tech stack
PyQt6 + PyQtGraph (OpenGL 2D/3D) · PyTorch (CUDA/MPS/CPU) · TOML · PySINDy
(Checked 2026-10-07: Python 3.14, PyQt6 6.11, pyqtgraph 0.14, torch 2.11+cu128 with CUDA, pysindy 2.1.
`tomli_w` is NOT installed yet — needed for TOML saving in step 6.)

## Build steps
1. ✅ App shell + theme + one-play splash (done 2026-10-07, tested: plays 45 frames once, ~2.9 s, then main window)
2. ✅ Main window layout (done 2026-10-07, tested with screenshots; RTX 5070 detected, MPS→CPU fallback works)
3. ✅ Master equation confirmed + PyTorch engine built (done 2026-10-07)
   - Decision: new MS38 rules are the default; include a "Legacy TEGR 2600" switch (old form + decay) to match past results.
   - Decision: physics runs in 3D; the user places things on a flat 2D grid (z = 0 plane). Keeps true 1/r gravity.
   - `python tests/validate_engine.py` — all 9 checks pass: bit-identical to MS38 `field_harness.py` (MS38 periodic,
     MS38+sponge, Legacy); energy drift 0.19%/2000 ticks; well settles to φ = G·m/r (fit 0.981); tracer holds a
     circular orbit (2.09 orbits, r 29.96–30.52); CPU = GPU; Legacy run OK.
   - RUN works in the GUI (background thread, STOP button, progress bar). File → Load demo = 1 well + 2 tracers.
4. ✅ Drag-and-drop 2D grid (done 2026-10-07, tested: drop from Toolbox, click/drag move, arrow-tip speed drag,
   aim at nearest, circular orbit, duplicate, delete, right-click menu, editor panel). Defaults: dt 0.005, ticks 16000,
   K 100, warm-up 6000, ramp 3000, α 0.1, G 1, sponge 12. Validation re-run: 9/9 pass.
5. ✅ 3D results view + telemetry polish (done 2026-10-07, tested with screenshots)
   - 3D: setup preview before a run; after RUN it jumps to 3D and replays: field sheet (z=0 slice of φ, asinh-compressed
     height, white = inside horizon), 3D paths, body markers, Play/Pause, time slider, "Whole paths", reset camera.
   - Telemetry: auto-fit (double-click to refit), no SI prefix, field + particle energy lines, "no horizon" note,
     dashed time cursor follows the 3D replay.
   - Finding: with α = 0.1 and a mass-30 well the horizon runs away (c² hits c2_min, source keeps pumping,
     peak φ ≈ 5e4, r_h grows to fill the box). Real model behaviour (MS38 Test 4 "trap"), not an app bug.
     **User decision: new experiments start at α = 0 (plain gravity); the user raises α when they want horizons.**
   - User tried steps 4–5, added a Repeat button for the replay (loops the play-through).
6. ✅ TOML save/load + export (done 2026-10-07, tested)
   - File: New / Open / Save / Save As (.toml, default folder `experiments/`). Own TOML writer (no tomli_w needed);
     reading via built-in tomllib. Window title shows the experiment name. TEGR 2600 presets are refused politely
     (different format) — an importer could be added later.
   - Export menu (default folder `exports/<name>_<time>/`, "Open folder" button after):
     graphs (telemetry, 2D paths, field slice, 3D paths; PNG 300 dpi + PDF; WHITE paper style via matplotlib),
     raw data (trajectories_long/wide.csv, telemetry.csv, t/positions/velocities/alive .npy, run_all.npz,
     experiment.toml, README.txt), PySINDy results (sindy_results.txt + sindy_fit.png/pdf), Export everything (Ctrl+E).
   - PySINDy report: fit A = gravity-only terms (direct Newton test; demo gives K·G·M ratio 0.998 / 0.999),
     fit B = open search; warns when B's extra terms are overfitting / near-duplicate terms.
     Report states the candidate terms are an ANALYSIS choice; the simulation never uses them.

## To-do list (user requests, later)
- [x] **PySINDy output tab** (user 2026-10-07): a 4th tab next to TELEMETRY that runs PySINDy on the run data and
  shows the found equations + fit plots. Implemented via `gui/pysindy_tab.py`.
- [ ] **Warm-up / Ramp to 0 Option**
  Add a quick toggle or make it clearer how to set Warm-up and Source Ramp to 0 for a fully "hands-off" initial start.

## "Are we cheating?" — what sits outside the two field equations (told the user 2026-10-07)
Particle rule F = K·m·∇φ (MS38 wake_orbit) · source rule (calibrated to φ = G·m/r) · "Circular orbit" button uses
Newton once for a STARTING speed only · warm-up holds bodies still (set 0 to disable) · source ramp · sponge edges ·
wave-speed floor c² ≥ c2_min = 1 (numerical, not in the equation; matters in horizons) · relativistic speed limit ·
3D sheet height is asinh-compressed (display only). No hidden forces, no fitting in the simulation.
User chose to keep warm-up and ramp as they are.

## Master equation — findings (2026-10-07)
Source of truth: TEGR 2600 `engine.py`, updated by MS38 (`field_harness.py`, `MS38_Notes.md` §9–11).

**Field**
- Wave rule (energy-conserving / Lagrangian form, MS38 Test 3):
  `φ_tt = ∇·(c²∇φ) − ½ (c²)'(φ) |∇φ|²`, from `L = ½φ_t² − ½c²(φ)|∇φ|²`
  - Discrete: `φ_i'' = Σ_j [ ½(c²_i + c²_j)(φ_j − φ_i) − ¼(c²)'_i (φ_j − φ_i)² ] / dx²` (6 neighbours)
- Speed rule (emergent impedance): `c²(φ) = c_base² / (1 + α|φ|)²`, clamped to `[c2_min (=1.0), CFL max]`
- No bulk drain (decay = 1.0). Open boundary = sponge edge layer (W = 12 cells). Gives 1/r gravity (MS38 Test 2, Presets 49–52).
- Old 2600 form for legacy runs: `φ_tt = c²(φ)∇²φ`, times `decay` each tick.

**Particles**
- MS38 mode: `F = K·m·∇φ` (equivalence principle; as in MS38 `Z:\rerun\wake_orbit.py`, which held a bound orbit).
- Legacy mode: `F = K·∇φ` (TEGR 2600 torsion force). Optional Pauli push (2600 form) in both.
- Relativistic momentum (γ); phase clock dθ/dt = m/γ.
- Not ported from 2600 (yet): impedance drag, vacuum damping, pilot wave, RAE, Kuramoto, topologies E1/E3.

**Sources / units (MS38)**: per tick q = 4π c² dt² G m / dx³ (cloud-in-cell) so the settled field is φ = G·m/r.
Newton's constant in the model G_N = K·G. Half-cosine source ramp; warm-up holds moving bodies still while the field settles.
Horizon radius r_h = volume-equivalent radius of cells where α|φ| ≥ 1 (wave speed halved). Blow-up guard: 1000 (legacy), 1e6 (MS38).

**Watch-outs from MS38:** φ = 0 is unstable under the speed rule (slow clumping); strong sources can trap φ near a horizon (Test 4).

## Code files so far
- `Start TEGR Genesis.bat` — Windows double-click launcher (pythonw, no console; falls back to python)
- `Start TEGR Genesis (Mac).command` / `Start TEGR Genesis (Linux).sh` — need `chmod +x` once on that computer
- `main.py` — entry point (theme → splash → main window)
- `gui/splash.py` — one-play splash (skips if GIF missing; 20 s safety)
- `gui/main_window.py` — full layout: workflow bar (Setup/Compute/Analyze, RUN/STOP, hardware), Toolbox dock (left), tabs 2D/3D/Telemetry (center), Properties + Simulation settings dock (right). `settings()` / `apply_settings(d)` use `SimSettings` field names. `self.entities` = list of `Entity`; `self.result` = last `RunResult`.
- `gui/canvas_2d.py` — editable z=0 plane. Signals `changed`, `selected(obj)`. `set_entities`, `add_entity(kind,x,y)`,
  `add_entity_free_spot`, `delete_selected`, `duplicate_selected`, `select`, `aim_at_nearest`, `circular_orbit`,
  `show_paths(result)`, `clear_paths`, `set_box(grid, dx, sponge)`, `editable` flag. MIME `application/x-tegr-kind`.
- `gui/entity_editor.py` — `EntityEditor` panel for the selected item (type, name, x, y, mass, speed, amplitude…)
- `gui/canvas_3d.py` — Collider-style GLViewWidget: XY/XZ/YZ grids + labelled axes; `show_setup(entities)`,
  `show_result(res, autoplay)`, replay bar; signal `time_changed(t)`. Surface colours must be flat (N,4) in pyqtgraph 0.14.
- `gui/dashboard.py` — 3 linked graphs; `set_data(t, energy=, peak_phi=, horizon=, field_energy=, particle_kinetic=)`,
  `set_cursor(t)`, `clear()`
- `gui/hardware_selector.py` — combo widget; torch imported only after window shows
- `gui/compute_worker.py` — `ComputeJob` runs `Engine.run` in a QThread (progress, stop)
- `core/engine.py` — `SimSettings`, `FieldSolver` (mirrors field_harness.py), `Sampler` (CiC), `Engine.run(entities, progress) -> RunResult`
- `core/entities.py` — `Entity` (well / defect / anchor / active / tracer), `to_dict` / `from_dict`
- `core/hardware.py` — `available_devices()`, `resolve_device(choice)` (no Qt)
- `core/state_manager.py` — `save_experiment(path, settings, entities, name, notes)`, `load_experiment(path)`,
  `toml_dumps()`; `EXPERIMENTS_DIR`
- `core/exporter.py` — `export_graphs/raw/sindy/all(res, folder)`, `sindy_fit(res)`, paper style `PAPER`; `EXPORTS_DIR`
- `tests/validate_engine.py` — engine validation (run after any engine change)
- `assets/genesis_theme.qss` — Genesis theme
