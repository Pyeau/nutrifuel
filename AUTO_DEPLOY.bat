@echo off
REM NutriFuel Deployment - Complete Automation Script
REM This script automates all steps for Render deployment

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║   NutriFuel Render Deployment - Complete Automation            ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Navigate to project root
cd /d "C:\Users\Haikal\Desktop\fyp\devoplement"

echo [1/5] Checking Git...
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [-] Git not found. Please install Git.
    pause
    exit /b 1
)
echo [+] Git found

echo.
echo [2/5] Initialize Git Repository...
git init
git config user.email "deployment@nutrifuel.com"
git config user.name "NutriFuel Deployment"
echo [+] Git initialized

echo.
echo [3/5] Stage All Files...
git add .
echo [+] Files staged

echo.
echo [4/5] Create Initial Commit...
git commit -m "NutriFuel - Production Ready for Render Deployment"
echo [+] Initial commit created

echo.
echo [5/5] Verify Setup...
git status
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║   ✅ LOCAL SETUP COMPLETE!                                    ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

echo Next Steps:
echo.
echo 1. CREATE GITHUB REPOSITORY
echo    → Go to: https://github.com/new
echo    → Name: "nutrifuel"
echo    → Click "Create Repository"
echo.
echo 2. PUSH TO GITHUB (run these commands)
echo    $ git remote add origin https://github.com/YOUR_USERNAME/nutrifuel.git
echo    $ git branch -M main
echo    $ git push -u origin main
echo.
echo 3. DEPLOY ON RENDER
echo    → Go to: https://render.com/dashboard
echo    → Click: "New +" ^→ "Blueprint"
echo    → Select: nutrifuel repo
echo    → Click: "Create Blueprint"
echo    → Wait: 5-10 minutes
echo.
echo 4. UPLOAD MODEL FILES
echo    → Backend Service ^→ Shell
echo    → Upload to /data:
echo      • improved_food_database.csv
echo      • meal_plan_model.joblib
echo.
echo 5. TEST YOUR DEPLOYMENT
echo    → Backend: https://nutrifuel-backend.onrender.com/health
echo    → Frontend: https://nutrifuel-frontend.onrender.com
echo.
echo ═══════════════════════════════════════════════════════════════
echo Time to deploy: ~30 minutes
echo Documentation: START_HERE.txt or YOUR_ACTION_ITEMS.md
echo ═══════════════════════════════════════════════════════════════
echo.

pause
