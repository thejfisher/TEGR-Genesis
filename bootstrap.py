"""TEGR Genesis — Smart Bootstrap & Hardware Peripheral Scanner.

Automatically checks system peripherals (CPU, NVIDIA CUDA, Apple Silicon MPS)
and verifies that all required scientific and GUI libraries are installed.
If any package is missing, it automatically installs the optimal version
for the detected hardware before launching the application.
"""

import sys
import os
import platform
import subprocess
import shutil

CORE_PACKAGES = [
    ("PyQt6", "PyQt6"),
    ("pyqtgraph", "pyqtgraph"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("pysindy", "pysindy"),
]


def log(msg):
    print(f"[TEGR Genesis] {msg}", flush=True)


def has_nvidia_gpu():
    """Detect if an NVIDIA GPU is available on the system."""
    return shutil.which("nvidia-smi") is not None


def check_dependencies():
    missing_core = []
    for module_name, pip_name in CORE_PACKAGES:
        try:
            __import__(module_name)
        except ImportError:
            missing_core.append(pip_name)

    torch_missing = False
    needs_cuda_torch = False

    try:
        import torch
    except ImportError:
        torch_missing = True
        if has_nvidia_gpu() and platform.system() in ("Windows", "Linux"):
            needs_cuda_torch = True

    return missing_core, torch_missing, needs_cuda_torch


def install_packages(packages, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    log(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def setup_environment():
    missing_core, torch_missing, needs_cuda_torch = check_dependencies()

    if not missing_core and not torch_missing:
        # All packages present! Start immediately with 0 delay.
        return True

    print("\n" + "=" * 58)
    print("        TEGR Genesis — First-Time System Setup")
    print("=" * 58)
    log(f"Operating System : {platform.system()} ({platform.machine()})")

    # Peripheral scanning
    if has_nvidia_gpu():
        log("Peripheral Scan  : NVIDIA GPU detected.")
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        log("Peripheral Scan  : Apple Silicon (M-series) detected.")
    else:
        log("Peripheral Scan  : Standard CPU / Integrated Graphics detected.")

    # Install core libraries if missing
    if missing_core:
        log(f"Installing missing libraries: {', '.join(missing_core)}...")
        try:
            install_packages(missing_core)
            log("Core libraries installed successfully.")
        except Exception as e:
            log(f"Error installing core libraries: {e}")
            return False

    # Install PyTorch tailored to detected hardware
    if torch_missing:
        log("Installing PyTorch physics engine tailored to your hardware...")
        try:
            if needs_cuda_torch:
                log("Selecting PyTorch with CUDA acceleration for your NVIDIA GPU...")
                install_packages(["torch"], extra_args=["--index-url", "https://download.pytorch.org/whl/cu121"])
            elif platform.system() in ("Windows", "Linux") and not has_nvidia_gpu():
                log("Selecting CPU PyTorch...")
                install_packages(["torch"], extra_args=["--index-url", "https://download.pytorch.org/whl/cpu"])
            else:
                # macOS / standard
                log("Selecting standard PyTorch...")
                install_packages(["torch"])
            log("PyTorch installed successfully.")
        except Exception as e:
            log(f"Error installing PyTorch: {e}")
            return False

    print("=" * 58)
    log("Setup complete! Launching TEGR Genesis...\n")
    return True


def main():
    if not setup_environment():
        print("\nSetup failed. Please check your internet connection or install requirements manually.")
        input("Press Enter to exit...")
        sys.exit(1)

    from main import main as run_genesis
    run_genesis()


if __name__ == "__main__":
    main()
