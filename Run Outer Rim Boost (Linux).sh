#!/usr/bin/env bash
# TEGR Genesis — Outer Rim Boost Benchmark Launcher (Linux)
set -e
cd "$(dirname "$0")"

echo "========================================================================"
echo "      TEGR Genesis — Outer Rim Gravitational Boost Benchmark"
echo "========================================================================"
echo ""

if command -v python3 &>/dev/null; then
    python3 run_outer_rim_boost.py
elif command -v python &>/dev/null; then
    python run_outer_rim_boost.py
else
    echo "Python 3 was not found. Please install Python 3.10+."
    exit 1
fi
