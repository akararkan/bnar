#!/bin/zsh
# ──────────────────────────────────────────────────────────
#  FourierLab — macOS double-click launcher
#  Double-click this file in Finder to start the application.
# ──────────────────────────────────────────────────────────

# Move to the folder that contains this script (and main.py)
cd "$(dirname "$0")"

# Activate the virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Install / upgrade dependencies silently if needed
pip install --quiet PySide6 matplotlib numpy

# Launch the application
python main.py

