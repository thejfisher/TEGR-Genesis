#!/bin/bash
# TEGR Genesis - double-click to open (Linux)
cd "$(dirname "$0")"

if command -v python3 &>/dev/null; then
    python3 bootstrap.py
elif command -v python &>/dev/null; then
    python bootstrap.py
else
    echo "Python 3 was not found on your system."
    echo "Please install Python (e.g. sudo apt install python3 python3-pip)"
    read -p "Press Enter to exit..."
fi
