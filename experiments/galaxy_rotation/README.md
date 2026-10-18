# Galaxy rotation curve test (2026-10-18)

Script: `galaxy_rotation.py`. Stock 7-point Genesis engine; `engine.py` is not edited.
Results: `results/galaxy_rotation.png`, `results/galaxy_rotation.json`.

## Question
Does the Genesis field pull on the outer disk harder than the visible mass predicts?
That extra pull would be the "dark matter" signature (a flat rotation curve).

## Setup
- 30 fixed wells make a bulge (mass 60) and 300 make a flat disk (mass 140). The disk ends at radius 75.
- Every run uses the same galaxy (same random seed).
- The rotation curve is measured from the settled field in the disk plane, averaged around each ring.

| Run | What changes |
|---|---|
| A0 | α = 0 (plain gravity), box G = 128 |
| A0_box | α = 0, smaller box G = 96 (checks whether the box walls bend the result) |
| A1 | α = 0.1 with the Fabbri c/c_base source, c2_min = 0.01, G = 128 |

## Prediction (written before running)
- **α = 0:** the field is plain Poisson, so the curve should match the visible mass.
- **α > 0 with the c/c_base source:** the pull should be boosted by exp(α·φ_N). The boost is strongest
  near the centre and fades outward, so the outer curve should **not** go flat.

## Results
| Check | Result |
|---|---|
| A0 vs visible-mass Newton, r = 10–100 | 1.005–1.078 (mostly within 1–3%) |
| Box control, G = 96 vs G = 128, r = 10–90 | 1.000–1.029 |
| A1 vs Newton | 2.4× at r = 6, 1.07× at r = 100 |
| A1 vs its prediction exp(α·φ) | 0.995–0.999 from r = 15 outward |
| Outer slope d(log v)/d(log r), r = 80–115 (Kepler −0.5, flat 0) | Newton −0.64, A0 −0.58, A1 −0.71 |
| Cells at speed floor (A1) | 0 |

## What it means
1. **Plain gravity matches the visible mass.** The few-% excess near the box edge grows as the box
   shrinks, so it comes from the walls, not from new physics.
2. **The Fabbri rule behaves exactly as the theory says** (within 0.5%). It adds real extra pull,
   but mostly in the inner galaxy. Far out it fades, so the outer curve falls *faster* than Newton, not flatter.
3. **A still galaxy in this engine shows no dark-matter-like flat curve.**

## Follow-up 1: was the warm-up too short? No (`warmup_check.py`)
- The pull is fully built by about tick 3000. At the old release (tick 6000) it is within 1% of final.
- A 20,000-tick warm-up gave the same orbits. So the warm-up was not the problem.

## Follow-up 2: relativistic launch speed. Yes, for the outer disk (`relativistic_launch.py`)
- The engine moves bodies relativistically (p = γmv, wave speed 65). A circular orbit needs
  K|∇φ| = γv²/r, not v²/r. We had launched at the Newtonian speed, which is slightly too fast.
- Relaunching at the relativistic circular speed (r stays within … of its start):

| r0 | α = 0, old launch | α = 0, relativistic | α = 0.1, old launch | α = 0.1, relativistic | γ (α = 0.1) |
|---|---|---|---|---|---|
| 80 | 0.94–1.22 | 0.88–1.04 | 1.00–1.44 | 0.83–1.15 | 1.045 |
| 95 | 0.96–1.18 | 0.96–1.10 | 0.95–1.23 | 0.95–1.12 | 1.033 |
| 110 | 1.00–1.10 | 0.99–1.05 | 1.00–1.11 | 1.00–1.04 | 1.027 |

- The outward-only drift is gone. Outside the disk, orbits now confirm the measured curve.
- Side lesson: orbital speeds here are a sizeable fraction of the wave speed (up to ~0.7 in the
  α = 0.1 inner disk), so launches must always use the relativistic formula.

## Known limits
- Inside the disk (r < 70) test stars still swing widely with either launch, for both α.
  Likely cause (not yet tested): they orbit in the same plane as 300 point-like masses and get kicked on close passes.
  A smoother disk (many more, lighter wells) is needed before the inner orbits can check the curve.
- The rotation curve itself comes from the still field, so these orbit issues do not change it.
- Only a still disk was tested. A spinning disk of moving bodies (wake / drag effects) has not been tested.

