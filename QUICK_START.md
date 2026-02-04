# 🚀 NutriFuel Render Deployment - Quick Start

## 5-Minute Deploy Guide

### Step 1: Prepare GitHub (2 min)
```bash
cd c:\Users\Haikal\Desktop\fyp\devoplement

# Initialize git
git init
git add .
git commit -m "NutriFuel - Ready for Render"

# Create repo at https://github.com/new
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/nutrifuel.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy to Render (1 min)
1. Go to https://render.com/dashboard
2. Click "New +" → "Blueprint"
3. Select your nutrifuel GitHub repo
4. Click "Create Blueprint"
5. Wait for both services to deploy (5-10 minutes)

### Step 3: Upload Model Files (1 min)
1. Backend Service → Shell
2. Create /data directory:
   ```bash
   mkdir -p /data
   ```
3. Upload files via Render Shell
4. Or use Git LFS if files are small

### Step 4: Test (1 min)
```bash
# Test backend
curl https://nutrifuel-backend.onrender.com/health

# Visit frontend
https://nutrifuel-frontend.onrender.com
```

---

## What Got Created For You

✅ **requirements.txt** - All Python dependencies
✅ **render.yaml** - Deployment configuration for both services
✅ **Procfile** - Process types
✅ **.env.example** - Environment variable template
✅ **.gitignore** - Git ignore rules
✅ **DEPLOYMENT_GUIDE.md** - Detailed deployment guide
✅ **DEPLOYMENT_CHECKLIST.md** - Step-by-step checklist
✅ **App.py** - Updated for production (environment variables, CORS, logging)
✅ **App.jsx** - Updated to use environment variables for API URL

---

## File Structure

```
devoplement/
├── requirements.txt              ✅ Python deps
├── render.yaml                   ✅ Render config
├── Procfile                      ✅ Process config
├── .env.example                  ✅ Env template
├── .gitignore                    ✅ Git ignore
├── package.json                  ✅ Root config
├── build.sh                      ✅ Linux build script
├── build.bat                     ✅ Windows build script
├── DEPLOYMENT_GUIDE.md           ✅ Full guide
├── DEPLOYMENT_CHECKLIST.md       ✅ Checklist
│
├── fyp/
│   ├── BAckend/App.py            ✅ Updated
│   ├── improved_food_database.csv
│   ├── meal_plan_model.joblib
│   └── ... (other model files)
│
└── fyp/frontend/src/App.jsx      ✅ Updated
```

---

## Critical Files to Upload

These MUST be in /data directory on Render:

```
required:
  - improved_food_database.csv
  - meal_plan_model.joblib
  
optional (if used):
  - goal_model.joblib
  - food_kmeans_model.joblib
  - food_scaler.joblib
  - regressor_features.joblib
  - regressor_metrics.joblib
```

---

## Environment Variables (Already Set in render.yaml)

### Backend
```
FLASK_ENV=production
PORT=5000
FRONTEND_URL=https://nutrifuel-frontend.onrender.com
```

### Frontend
```
REACT_APP_API_URL=https://nutrifuel-backend.onrender.com
```

---

## Expected Result

✅ Backend running at: `https://nutrifuel-backend.onrender.com`
✅ Frontend running at: `https://nutrifuel-frontend.onrender.com`
✅ API responding at: `https://nutrifuel-backend.onrender.com/api/predict`
✅ Health check at: `https://nutrifuel-backend.onrender.com/health`

---

## Troubleshooting

### Deployment Fails
→ Check GitHub repository is public
→ Check render.yaml syntax
→ Check Python 3.11+, Node 18+ requirements

### Models Not Loading
→ SSH into backend service
→ Check: `ls -la /data/`
→ Verify filenames match exactly

### CORS Errors
→ Already configured in App.py
→ Check frontend URL in environment variables

### Slow Performance
→ Free tier: Cold starts take 50+ seconds
→ Use paid tier for < 10 second starts

### 503 Errors
→ Service may be starting
→ Wait 2-3 minutes and refresh
→ Check logs in Render dashboard

---

## Quick Commands

```bash
# Local development
npm install                        # Install all deps
npm run frontend                   # Start frontend only
npm run backend                    # Start backend only

# Git commands
git status                         # Check changes
git add .                         # Stage all
git commit -m "message"           # Commit
git push                          # Push to GitHub

# Render CLI (optional)
render login                       # Login to Render
render deploy                      # Deploy
render logs                        # View logs
```

---

## Next Steps

1. **Push to GitHub** (see Step 1 above)
2. **Deploy to Render** (see Step 2 above)
3. **Upload Model Files** (see Step 3 above)
4. **Test Everything** (see Step 4 above)
5. **Share Your App**: `https://nutrifuel-frontend.onrender.com`

---

## Support

- 📖 Full Guide: See `DEPLOYMENT_GUIDE.md`
- ✅ Checklist: See `DEPLOYMENT_CHECKLIST.md`
- 🔗 Render Docs: https://render.com/docs
- 💬 Community: https://render.com/community

---

**You're All Set! 🎉**

Everything is configured and ready to deploy.

Just push to GitHub and deploy on Render.

Good luck! 🚀

---
*Last Updated: February 4, 2026*
