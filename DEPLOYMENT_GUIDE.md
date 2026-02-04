# NutriFuel - Render Deployment Guide

## Overview
This guide will help you deploy the NutriFuel application (Backend Flask API + Frontend React App) to Render.

## Prerequisites
- GitHub account with the project repository
- Render.com account (https://render.com)
- Git installed locally

## Deployment Steps

### Step 1: Push to GitHub
```bash
cd c:\Users\Haikal\Desktop\fyp\devoplement
git init
git add .
git commit -m "Initial commit - NutriFuel app"
git remote add origin https://github.com/YOUR_USERNAME/nutrifuel.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy Using render.yaml

The `render.yaml` file in the root directory defines both services:

**Option A: Deploy with render.yaml (Recommended)**

1. Go to https://dashboard.render.com
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Select the branch (main)
5. Render will automatically read `render.yaml` and deploy both services

**Option B: Manual Deployment**

#### Deploy Backend (Flask API)
1. In Render Dashboard, click "New +" → "Web Service"
2. Connect GitHub repository
3. Set configuration:
   - **Name**: nutrifuel-backend
   - **Environment**: Python 3.11
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 3 --timeout 120 fyp.BAckend.App:app`
   - **Plan**: Starter (free tier)

4. Add Environment Variables:
   ```
   FLASK_ENV=production
   PORT=5000
   ```

5. Click "Create Web Service"

#### Deploy Frontend (React App)
1. In Render Dashboard, click "New +" → "Web Service"
2. Connect GitHub repository
3. Set configuration:
   - **Name**: nutrifuel-frontend
   - **Environment**: Node 18
   - **Build Command**: `cd fyp/frontend && npm install && npm run build`
   - **Start Command**: `cd fyp/frontend && npx serve -s build -l 3000`
   - **Plan**: Starter (free tier)

4. Add Environment Variables:
   ```
   REACT_APP_API_URL=https://nutrifuel-backend.onrender.com
   ```

5. Click "Create Web Service"

### Step 3: Configure Data & Models

Since Render services are ephemeral, you need to store models/data persistently:

**Option A: Use Render Disks (Recommended for small projects)**
- Add a disk to the backend service
- Mount at `/data`
- Size: 5GB (includes free tier limits)

**Option B: Use Git LFS for large files**
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.joblib"
git lfs track "*.csv"

git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

**Option C: Use Cloud Storage (AWS S3, Google Cloud Storage)**
- Upload models to S3
- Modify App.py to fetch models on startup
- Add AWS credentials to environment variables

### Step 4: Copy Data Files to Backend

After deployment, you need to upload the model and data files to your Render service:

1. SSH into your backend service:
```bash
# Get the SSH command from Render Dashboard
# Backend Service → Settings → SSH command
```

2. Upload files:
```bash
# From your local machine
scp -r improved_food_database.csv user@your-backend-service.onrender.com:/data/
scp -r meal_plan_model.joblib user@your-backend-service.onrender.com:/data/
```

Or use Render Shell:
1. Go to Backend Service → Shell
2. Upload files directly through the web interface

### Step 5: Verify Deployment

**Test Backend:**
```bash
curl https://nutrifuel-backend.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "message": "Server is running",
  "database": "connected",
  "model": "loaded"
}
```

**Test Frontend:**
Visit: `https://nutrifuel-frontend.onrender.com`

**Test API Connection:**
1. Open frontend
2. Fill in user profile
3. Click "Generate Meal Plan"
4. Should see results loading from backend

## Environment Variables Configuration

### Backend (.env)
```
FLASK_ENV=production
FLASK_APP=fyp/BAckend/App.py
PORT=5000
FRONTEND_URL=https://nutrifuel-frontend.onrender.com
```

### Frontend (.env.local in frontend folder)
```
REACT_APP_API_URL=https://nutrifuel-backend.onrender.com
```

## Troubleshooting

### Models Not Loading
- Check Render logs: Service → Logs
- Verify files are in correct location
- SSH into service and check: `ls -la /data/`

### CORS Errors
- Check backend CORS configuration (already set in App.py)
- Verify frontend API URL matches backend domain

### Build Failures
- Check Python version compatibility (Python 3.11+)
- Verify `requirements.txt` has all dependencies
- Check Node version for frontend (18+)

### Performance Issues
- Free tier: Cold start may take 50+ seconds
- Upgrade plan if needed
- Add more workers in gunicorn command

## File Structure for Deployment

```
devoplement/
├── requirements.txt          # Python dependencies
├── Procfile                  # Heroku/Render config
├── render.yaml              # Blueprint config (NEW)
├── .env.example             # Example env vars (NEW)
├── .gitignore               # Git ignore file (NEW)
├── fyp/
│   ├── BAckend/
│   │   ├── App.py          # Updated for production
│   │   ├── improved_food_database.csv
│   │   └── meal_plan_model.joblib
│   └── frontend/
│       ├── package.json
│       ├── src/
│       │   └── App.jsx     # Updated with env vars
│       └── public/
```

## Cost Estimation (Free Tier)

- **Backend Web Service**: Free (0.50 CPU, 0.5 GB RAM)
- **Frontend Web Service**: Free (0.50 CPU, 0.5 GB RAM)
- **Disk Storage**: Free (5 GB)
- **Total**: ~$0 (with limitations)

**Paid Tier** (Recommended for production):
- Backend: $7/month
- Frontend: $7/month
- Disk: $0.50/GB/month

## Next Steps After Deployment

1. Monitor logs for errors
2. Test all features thoroughly
3. Set up monitoring/alerts
4. Add custom domain (if needed)
5. Set up backups for models/data
6. Plan for scaling

## Support

- Render Docs: https://render.com/docs
- Flask Deployment: https://flask.palletsprojects.com/deployment/
- React Deployment: https://create-react-app.dev/deployment/

---
**Last Updated**: February 4, 2026
**App Version**: 1.0.0
