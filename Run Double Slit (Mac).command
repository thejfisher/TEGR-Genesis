#!/usr/bin/env bash
# TEGR Genesis -- Double-Slit Quantum Interference Benchmark Launcher (macOS)
set -e
cd "$(dirname "$0")"

echo "========================================================================"
echo "      TEGR Genesis -- Double-Slit Quantum Interference Benchmark"
echo "========================================================================"
echo ""

if command -v python3 &>/dev/null; then
    python3 run_double_slit.py
elif command -v python &>/dev/null; then
    python run_double_slit.py
else
    echo "Python 3 was not found. Please install Python 3.10+."
    exit 1
fi
