# Render Environment Variables Configuration

## Create these files:

### 1. Root .env file (for local development)
```
# Backend
FLASK_ENV=development
FLASK_APP=fyp/BAckend/App.py
PORT=5000
FRONTEND_URL=http://localhost:3000

# Frontend
REACT_APP_API_URL=http://127.0.0.1:5000
```

### 2. frontend/.env.local (for local React development)
```
REACT_APP_API_URL=http://127.0.0.1:5000
```

### 3. Render Environment Variables (Set in Dashboard)

#### For Backend Service (nutrifuel-backend)
```
FLASK_ENV=production
PORT=5000
FRONTEND_URL=https://nutrifuel-frontend.onrender.com
```

#### For Frontend Service (nutrifuel-frontend)
```
REACT_APP_API_URL=https://nutrifuel-backend.onrender.com
```

## How to Set on Render Dashboard

1. Go to Service Dashboard
2. Click "Environment" tab
3. Add each variable:
   - Key: Variable name
   - Value: Variable value
4. Save and redeploy

## Important Notes

⚠️  Never commit .env files with real credentials
⚠️  Use .env.example for template only
⚠️  Render variables override local .env files
⚠️  Changes require service restart/redeploy

## Variable Meanings

| Variable | Purpose | Example |
|----------|---------|---------|
| FLASK_ENV | Flask environment mode | production / development |
| PORT | Backend port | 5000 |
| FRONTEND_URL | Frontend service URL | https://nutrifuel-frontend.onrender.com |
| REACT_APP_API_URL | Backend API URL for React | https://nutrifuel-backend.onrender.com |

---
Generated: February 4, 2026
