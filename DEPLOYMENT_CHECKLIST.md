# Render Deployment Checklist

## Pre-Deployment Checklist

### Code Preparation
- [ ] All code committed to GitHub
- [ ] Environment variables set in `.env.example`
- [ ] requirements.txt updated with all Python dependencies
- [ ] package.json in frontend folder has all npm dependencies
- [ ] No hardcoded API URLs (using environment variables)
- [ ] CORS configuration set properly in App.py
- [ ] All models/data files included in repository or planned for upload

### Files Created
- [ ] requirements.txt
- [ ] render.yaml
- [ ] Procfile
- [ ] .env.example
- [ ] .gitignore
- [ ] DEPLOYMENT_GUIDE.md
- [ ] package.json (root)
- [ ] build.sh / build.bat

### GitHub Setup
- [ ] Repository created on GitHub
- [ ] All files pushed to main branch
- [ ] Repository is public (for Render access)

## Deployment Steps

### Phase 1: Create Render Account
- [ ] Sign up at https://render.com
- [ ] Connect GitHub account

### Phase 2: Deploy Services
- [ ] Deploy Backend (Flask API)
  - [ ] Service name: nutrifuel-backend
  - [ ] Build command set correctly
  - [ ] Start command set correctly
  - [ ] Environment variables configured
- [ ] Deploy Frontend (React App)
  - [ ] Service name: nutrifuel-frontend
  - [ ] Build command set correctly
  - [ ] Start command set correctly
  - [ ] Environment variables configured (REACT_APP_API_URL)

### Phase 3: Upload Data & Models
- [ ] Create Render Disk (if using disk storage)
- [ ] Mount disk at /data path
- [ ] Upload improved_food_database.csv
- [ ] Upload meal_plan_model.joblib
- [ ] Upload other required model files:
  - [ ] goal_model.joblib
  - [ ] food_kmeans_model.joblib
  - [ ] meal_plan_model.joblib
  - [ ] food_scaler.joblib
  - [ ] regressor_features.joblib
  - [ ] regressor_metrics.joblib

### Phase 4: Testing
- [ ] Test health endpoint: GET https://nutrifuel-backend.onrender.com/health
- [ ] Response shows database: "connected"
- [ ] Response shows model: "loaded"
- [ ] Test frontend: Visit https://nutrifuel-frontend.onrender.com
- [ ] Frontend loads successfully
- [ ] Can fill in profile
- [ ] Can click "Generate Meal Plan"
- [ ] Receives results from backend
- [ ] Can search food exchange list
- [ ] Can replace foods

### Phase 5: Production Verification
- [ ] No 503 errors in logs
- [ ] No CORS errors in browser console
- [ ] Database loads in < 2 seconds
- [ ] Model prediction completes in < 5 seconds
- [ ] Frontend builds successfully
- [ ] All API endpoints respond

## Environment Variables Required

### Backend (nutrifuel-backend)
```
FLASK_ENV=production
PORT=5000
FRONTEND_URL=https://nutrifuel-frontend.onrender.com
```

### Frontend (nutrifuel-frontend)
```
REACT_APP_API_URL=https://nutrifuel-backend.onrender.com
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Models not loading | SSH into backend, verify files in /data directory |
| CORS errors | Check backend CORS config, verify frontend URL |
| 503 errors on startup | Models may be loading, wait 2-3 minutes |
| Slow responses | Upgrade from free tier or optimize queries |
| Build failures | Check logs, verify Node/Python versions |
| API not responding | Check backend logs, verify port 5000 |

## Performance Expectations

### Free Tier
- Cold start: 50-100 seconds first request
- Memory: 0.5 GB per service
- CPU: 0.5 vCPU
- Requests/second: ~10
- Expected latency: 200-500ms

### Paid Tier ($7/month each)
- Cold start: 5-10 seconds
- Memory: 2+ GB per service
- CPU: 1+ vCPU
- Requests/second: ~50+
- Expected latency: 50-100ms

## File Locations During Deployment

```
/workspace/
├── requirements.txt       (Backend dependencies)
├── Procfile              (Process config)
├── render.yaml           (Render config)
├── package.json          (Root config)
├── .env.example          (Example env vars)
├── .gitignore            (Git ignore rules)
├── DEPLOYMENT_GUIDE.md   (This file)
├── build.sh              (Build script)
├── build.bat             (Build script - Windows)
│
├── fyp/
│   ├── BAckend/
│   │   ├── App.py                      (Flask app - updated)
│   │   ├── improved_food_database.csv  (Food data)
│   │   └── meal_plan_model.joblib     (ML model)
│   │
│   └── frontend/
│       ├── package.json                (Frontend dependencies)
│       ├── src/
│       │   ├── App.jsx                (React app - updated)
│       │   └── index.js
│       └── public/
│           └── index.html
```

## Post-Deployment Maintenance

### Weekly
- [ ] Check logs for errors
- [ ] Monitor response times
- [ ] Verify all endpoints working

### Monthly
- [ ] Update dependencies
- [ ] Review error rates
- [ ] Test disaster recovery

### Quarterly
- [ ] Performance optimization
- [ ] Security audit
- [ ] Scale if needed

## Useful Render Commands

### Via Web Dashboard
- View logs: Service → Logs
- SSH access: Service → Shell
- Restart service: Service → Settings → Restart
- View metrics: Service → Metrics

### Via Render CLI (optional)
```bash
# Install Render CLI
npm install -g @render-com/cli

# Login
render login

# Deploy
render deploy

# View logs
render logs --service nutrifuel-backend
```

## Support & Resources

- Render Status: https://status.render.com
- Render Docs: https://render.com/docs
- Community: https://render.com/community
- Email Support: support@render.com

---

**Status**: ✅ Ready to Deploy
**Last Updated**: February 4, 2026
**Version**: 1.0.0
