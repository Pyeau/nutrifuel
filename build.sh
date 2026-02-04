#!/bin/bash

# NutriFuel Deployment Build Script
# This script prepares the project for deployment to Render

echo "🚀 NutriFuel Deployment Build Script"
echo "======================================"

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📝 Initializing Git repository..."
    git init
    git config user.email "deployment@nutrifuel.com"
    git config user.name "NutriFuel Deployment"
fi

# Check dependencies
echo "✅ Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node 18+"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install npm 9+"
    exit 1
fi

echo "✅ All dependencies found"

# Create virtual environment for backend
echo ""
echo "📦 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

# Install Python dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Install frontend dependencies
echo ""
echo "📦 Setting up Node.js environment..."
cd fyp/frontend
npm install
cd ../..

echo ""
echo "✅ Build completed successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Set up a GitHub repository:"
echo "   git remote add origin <your-repo-url>"
echo "   git add ."
echo "   git commit -m 'Initial commit'"
echo "   git push -u origin main"
echo ""
echo "2. Go to https://render.com and connect your GitHub"
echo ""
echo "3. Upload files using Render Shell or Git LFS"
echo ""
echo "📖 See DEPLOYMENT_GUIDE.md for detailed instructions"
