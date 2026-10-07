# TEGR Genesis

A desktop application for simulating and analyzing topological gravity (TEGR).
Instead of hard-coding gravity equations, TEGR Genesis simulates a flat field. Mass creates waves, and particles slide down the local slope of those waves. Gravity emerges naturally.

## 📥 Required Downloads

To run TEGR Genesis, you need Python and the following packages:

**1. Python**
*   [Python 3.12 or newer](https://www.python.org/downloads/) (Tested on Python 3.14)

**2. Core Libraries**
Run the following command in your terminal to install the necessary libraries:
```bash
pip install PyQt6 pyqtgraph numpy matplotlib pysindy
```

**3. PyTorch (For the Physics Engine)**
You must install PyTorch to run the simulation. The exact command depends on your computer's hardware. Go to the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) to find the right command for your system, or use one of the common ones below:

*   **NVIDIA GPU (Windows/Linux) - Highly Recommended:**
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```
*   **Mac (Apple Silicon or Intel):**
    ```bash
    pip install torch
    ```
*   **CPU only (slowest):**
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

---

## 🚀 How to Launch

*   **Windows:** Double-click the `Start TEGR Genesis.bat` file.
*   **Mac:** Double-click `Start TEGR Genesis (Mac).command`. (You may need to run `chmod +x "Start TEGR Genesis (Mac).command"` in the terminal first).
*   **Linux:** Run `./Start TEGR Genesis (Linux).sh`.

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
