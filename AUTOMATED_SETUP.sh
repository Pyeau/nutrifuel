#!/bin/bash

# NutriFuel Render Deployment - Linux/macOS Automated Setup
# Runs the Python automation script

echo ""
echo "========================================"
echo "   NutriFuel Render - Automated Setup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[-] Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Run the automation script
python3 AUTOMATED_SETUP.py

if [ $? -eq 0 ]; then
    echo ""
    echo "[+] Setup complete!"
    echo "[*] Next: Read START_HERE.txt or YOUR_ACTION_ITEMS.md"
    echo ""
else
    echo ""
    echo "[-] Setup failed. Check the output above."
    echo ""
    exit 1
fi
