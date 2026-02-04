# Quick Command Reference

## Git Commands

```bash
# Navigate to project
cd c:\Users\Haikal\Desktop\fyp\devoplement

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "NutriFuel - Render Deploy Ready"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/nutrifuel.git

# Change branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

## Render Deployment

### Via Web Dashboard
1. Go to: https://render.com/dashboard
2. Click: "New +" → "Blueprint"
3. Select: Your nutrifuel GitHub repository
4. Click: "Create Blueprint"
5. Wait: 5-10 minutes for deployment

### Via Render CLI (Optional)
```bash
npm install -g @render-com/cli
render login
render deploy
render logs --service nutrifuel-backend
```

## Upload Models to Backend

### Via Shell
```bash
# SSH into backend service
# Command found in: Service Dashboard → Shell

# Create data directory
mkdir -p /data

# Upload files via web interface
# or use scp if you have SSH access
```

### Via SCP
```bash
# Get SSH command from Render dashboard
# Then run:
scp improved_food_database.csv user@your-service.onrender.com:/data/
scp meal_plan_model.joblib user@your-service.onrender.com:/data/
```

## Test Deployment

```bash
# Test backend health
curl https://nutrifuel-backend.onrender.com/health

# Expected response:
# {"status":"ok","database":"connected","model":"loaded"}

# View backend logs
# Go to: Backend Service → Logs

# View frontend logs
# Go to: Frontend Service → Logs

# SSH into backend
# Go to: Backend Service → Shell

# SSH into frontend
# Go to: Frontend Service → Shell
```

## Troubleshooting

```bash
# Check if models are loaded
# SSH into backend and run:
ls -la /data/

# Check Python path
python3 -c "import sys; print(sys.executable)"

# Test API directly
curl https://nutrifuel-backend.onrender.com/api/predict \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"age":25,"gender":"Male","height":170,"weight":70,"fat":15,"bpm":60,"goal":"Build Muscle","activity":"Moderate Training (3–5 days/week)","calorieStrategy":"Match TDEE","manualCalories":2000}'

# Check service status
# Go to: Render Dashboard → Service → Metrics
```

## Local Development

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies
cd fyp/frontend
npm install
cd ../..

# Run backend locally
cd fyp/BAckend
python App.py

# Run frontend locally (in another terminal)
cd fyp/frontend
npm start

# Run both (if npm scripts configured)
npm start
npm run backend
```

## Environment Variables

### Set on Render Dashboard

**Backend Service:**
```
FLASK_ENV=production
PORT=5000
FRONTEND_URL=https://nutrifuel-frontend.onrender.com
```

**Frontend Service:**
```
REACT_APP_API_URL=https://nutrifuel-backend.onrender.com
```

### Local .env Files

**Root/.env:**
```
FLASK_ENV=development
FLASK_APP=fyp/BAckend/App.py
PORT=5000
FRONTEND_URL=http://localhost:3000
REACT_APP_API_URL=http://127.0.0.1:5000
```

**frontend/.env.local:**
```
REACT_APP_API_URL=http://127.0.0.1:5000
```

## Build & Deploy Locally

### Windows
```bash
# Run build script
build.bat

# This will:
# 1. Check dependencies
# 2. Create virtual environment
# 3. Install Python packages
# 4. Install Node packages
```

### Linux/macOS
```bash
# Run build script
chmod +x build.sh
./build.sh

# This will:
# 1. Check dependencies
# 2. Create virtual environment
# 3. Install Python packages
# 4. Install Node packages
```

## Useful Links

```
GitHub:            https://github.com
Render Dashboard:  https://render.com/dashboard
Render Docs:       https://render.com/docs
Flask Docs:        https://flask.palletsprojects.com
React Docs:        https://react.dev
```

## Important Files Locations

```
Project Root:      c:\Users\Haikal\Desktop\fyp\devoplement
Backend Code:      fyp\BAckend\App.py
Frontend Code:     fyp\frontend\src\App.jsx
Data Files:        fyp\BAckend\ (local) or /data/ (Render)
Configuration:     render.yaml, Procfile, requirements.txt
Documentation:     All .md and .txt files in root
```

## Common Issues Quick Fixes

| Problem | Solution |
|---------|----------|
| Models not loading | Check `/data/` directory exists, files uploaded |
| CORS errors | Already configured in App.py |
| 503 service error | Wait 2-3 minutes, service may be starting |
| Build failed | Check GitHub repo is public, render.yaml valid |
| Slow response | Free tier may have cold starts (50+ sec) |
| API not responding | Check backend logs, verify health endpoint |

---

**Quick Deploy Checklist:**
```
[ ] Read START_HERE.txt (2 min)
[ ] Read YOUR_ACTION_ITEMS.md (3 min)
[ ] Initialize git (2 min)
[ ] Push to GitHub (5 min)
[ ] Deploy on Render (10 min)
[ ] Upload models (5 min)
[ ] Test health endpoint (1 min)
[ ] Visit frontend URL (1 min)
[ ] Generate meal plan (1 min)
[ ] Test food exchange (1 min)
```

**Total Time: ~31 minutes**

---

*Generated: February 4, 2026*
*Version: 1.0.0*
