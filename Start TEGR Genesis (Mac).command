#!/bin/bash
# TEGR Genesis - double-click to open (Mac)
cd "$(dirname "$0")"

if command -v python3 &>/dev/null; then
    python3 bootstrap.py
elif command -v python &>/dev/null; then
    python bootstrap.py
else
    echo "Python 3 was not found on your system."
    echo "Please install Python from https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
fi
