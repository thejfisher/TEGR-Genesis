# TEGR Genesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23380953.svg)](https://doi.org/10.5281/zenodo.23380953)

A desktop application for simulating and analyzing topological gravity (TEGR).
Instead of hard-coding gravity equations, TEGR Genesis simulates a flat field. Mass creates waves, and particles slide down the local slope of those waves. Gravity emerges naturally.

## 🚀 Quick Start (Zero-Setup Launcher)

You do **not** need to manually install complex packages or open a terminal. 

1. Ensure [Python 3.10+](https://www.python.org/downloads/) is installed on your computer.
2. Launch the app for your operating system:
   * **Windows:** Double-click [`Start TEGR Genesis.bat`](Start%20TEGR%20Genesis.bat).
   * **Mac:** Double-click [`Start TEGR Genesis (Mac).command`](Start%20TEGR%20Genesis%20%28Mac%29.command).
   * **Linux:** Run [`./Start TEGR Genesis (Linux).sh`](Start%20TEGR%20Genesis%20%28Linux%29.sh).

**What happens on first launch:**
The built-in smart bootstrap scans your hardware peripherals (detecting NVIDIA CUDA GPUs, Apple Silicon M-series chips, or CPUs) and automatically installs any missing scientific libraries (`PyQt6`, `PyTorch`, `PySINDy`, `pyqtgraph`, `matplotlib`, `numpy`) tailored specifically to your hardware. 

On subsequent launches, it verifies your environment in milliseconds and opens TEGR Genesis immediately.

---

## 🌌 Outer Rim Gravitational Boost Benchmark (Standalone)

Want to verify the emergent **+16.5% to +28.0% outer-rim pull boost** ($R^2 = 0.9968$, no dark matter) without running the full GUI?

* **Windows (1-click):** Double-click [`Run Outer Rim Boost.bat`](Run%20Outer%20Rim%20Boost.bat).
* **Linux:** Run [`./Run Outer Rim Boost (Linux).sh`](Run%20Outer%20Rim%20Boost%20%28Linux%29.sh).
* **Mac:** Double-click [`Run Outer Rim Boost (Mac).command`](Run%20Outer%20Rim%20Boost%20%28Mac%29.command).
* **Command Line:** `python run_outer_rim_boost.py`
* **Full Benchmark Guide:** See [`README_OUTER_RIM_BOOST.md`](README_OUTER_RIM_BOOST.md) for requirements, physics details, and output file documentation.

---

## ⚛️ Hydrogen 1s Bound State Benchmark (Duda de Broglie Clock Preset)

Looking to test a bound atomic electron with an internal de Broglie phase clock ($\dot{\theta} = m/\gamma$) and Pauli push on a discrete 3D lattice?

* **In the GUI:** Launch TEGR Genesis, go to **File → Open Experiment (`Ctrl+O`)**, and select [`experiments/duda_hydrogen.toml`](experiments/duda_hydrogen.toml).
* **Windows (1-click):** Double-click [`Run Duda Hydrogen.bat`](Run%20Duda%20Hydrogen.bat).
* **Linux:** Run [`./Run Duda Hydrogen (Linux).sh`](Run%20Duda%20Hydrogen%20%28Linux%29.sh).
* **Mac:** Double-click [`Run Duda Hydrogen (Mac).command`](Run%20Duda%20Hydrogen%20%28Mac%29.command).
* **Command Line:** `python run_duda_hydrogen.py`
* **Key Findings:** Stable non-collapsing orbit, natural de Broglie clock phase closure ($1.07 \times 2\pi$ per orbit), and PySINDy inverse-square Coulomb recovery ($R^2 = 0.9997$).

---

## 🎯 Double-Slit Quantum Interference Benchmark (Tonomura Lattice Preset)

Looking to test discrete lattice wave diffraction, slit scattering, and phase-guided particle trajectories through a dual-slit barrier?

* **In the GUI:** Launch TEGR Genesis, go to **File → Open Experiment (`Ctrl+O`)**, and select [`experiments/double_slit.toml`](experiments/double_slit.toml).
* **Windows (1-click):** Double-click [`Run Double Slit.bat`](Run%20Double%20Slit.bat).
* **Linux:** Run [`./Run Double Slit (Linux).sh`](Run%20Double%20Slit%20%28Linux%29.sh).
* **Mac:** Double-click [`Run Double Slit (Mac).command`](Run%20Double%20Slit%20%28Mac%29.command).
* **Command Line:** `python run_double_slit.py`
* **Key Findings:** Coherent wave pulse diffraction across two 5-unit slits ($y = \pm 6.0$), 48% transmission with wave-guided transverse deflection, generating detector screen interference fringe distributions.

---

## 🛠️ Manual Installation (Optional)

If you prefer to manage your own virtual environment manually:
```bash
# 1. Core GUI and analysis libraries
pip install PyQt6 pyqtgraph numpy matplotlib pysindy

# 2. PyTorch physics engine (choose your hardware):
# NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Mac (Apple Silicon / Intel):
pip install torch
# Standard CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Run
python main.py
```

---

## 🕹️ How to Use the App

The app is broken down into a simple 3-step workflow:

### Step 1: SETUP
*   Use the **Toolbox** on the left to drag and drop items onto the flat 2D grid (the `z=0` plane).
*   **Gravity Wells & Wave Defects:** These bend the field.
*   **Active Bodies & Tracers:** These move through the field.
*   Click any object on the grid to change its properties (like mass, speed, and direction) in the **Properties** panel on the right.
*   You can also adjust global simulation settings (like wave speed and time steps) on the right side.

### Step 2: COMPUTE
*   Click the **▶ RUN** button at the top to start the simulation.
*   The physics engine will calculate how the field waves ripple and how the particles ride those waves over time.
*   The hardware picker at the top will automatically try to use your GPU (CUDA on Windows/Linux, MPS on Mac) for faster processing, but it will fall back to your CPU if necessary.

### Step 3: ANALYZE
Once the run is complete, the app automatically switches to the analysis tabs:
*   **3D VIEW:** Replay the entire run in 3D. You will see the physical "sheet" of the field (where the waves and dips are) and the paths the bodies took over time.
*   **TELEMETRY:** View energy conservation graphs, the peak field strength, and the horizon radius as the simulation progressed.
*   **PYSINDY:** Click "Run PySINDy Analysis" to have an AI (PySINDy) mathematically analyze the paths of your bodies. 
    *   **Fit A** checks if the emergent motion matches traditional Newtonian gravity. 
    *   **Fit B** attempts an open search to find any equation that fits, and will warn you if it thinks it is hallucinating fake math.

### Exporting for Papers
You can export all of your results at any time by going to **File → Export everything...**
All exported images (PNG and PDF) will be rendered with a white background and dark lines, making them perfect for academic papers. The raw data (CSV, NPY) will also be saved so you can analyze it further in other tools.
