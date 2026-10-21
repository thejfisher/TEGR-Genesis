#!/usr/bin/env bash
# TEGR Genesis — Duda Hydrogen Bound State Benchmark Launcher (macOS)
set -e
cd "$(dirname "$0")"

echo "========================================================================"
echo "      TEGR Genesis — Duda Hydrogen Bound State Benchmark"
echo "========================================================================"
echo ""

if command -v python3 &>/dev/null; then
    python3 run_duda_hydrogen.py
elif command -v python &>/dev/null; then
    python run_duda_hydrogen.py
else
    echo "Python 3 was not found. Please install Python 3.10+."
    exit 1
fi
