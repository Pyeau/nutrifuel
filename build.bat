@echo off
REM NutriFuel Deployment Build Script (Windows)
REM This script prepares the project for deployment to Render

echo.
echo ========================================
echo   NutriFuel Deployment Build Script
echo ========================================
echo.

REM Check if git is initialized
if not exist .git (
    echo [*] Initializing Git repository...
    git init
    git config user.email "deployment@nutrifuel.com"
    git config user.name "NutriFuel Deployment"
)

REM Check dependencies
echo [+] Checking dependencies...

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [-] Python not found. Please install Python 3.11+
    exit /b 1
)

where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [-] Node.js not found. Please install Node 18+
    exit /b 1
)

where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo [-] npm not found. Please install npm 9+
    exit /b 1
)

echo [+] All dependencies found
echo.

REM Create virtual environment for backend
echo [*] Setting up Python environment...
if not exist venv (
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install Python dependencies
echo [*] Installing Python dependencies...
pip install -r requirements.txt

REM Install frontend dependencies
echo.
echo [*] Setting up Node.js environment...
cd fyp\frontend
call npm install
cd ..\..

echo.
echo [+] Build completed successfully!
echo.
echo [*] Next steps:
echo     1. Set up a GitHub repository:
echo        git remote add origin ^<your-repo-url^>
echo        git add .
echo        git commit -m "Initial commit"
echo        git push -u origin main
echo.
echo     2. Go to https://render.com and connect your GitHub
echo.
echo     3. Upload files using Render Shell or Git LFS
echo.
echo     See DEPLOYMENT_GUIDE.md for detailed instructions
echo.
pause
