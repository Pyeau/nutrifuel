@echo off
REM NutriFuel Render Deployment - Windows Automated Setup
REM Runs the Python automation script

echo.
echo ========================================
echo   NutriFuel Render - Automated Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [-] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Run the automation script
python AUTOMATED_SETUP.py

if %errorlevel% equ 0 (
    echo.
    echo [+] Setup complete!
    echo [*] Next: Read START_HERE.txt or YOUR_ACTION_ITEMS.md
    echo.
) else (
    echo.
    echo [-] Setup failed. Check the output above.
    echo.
)

pause
