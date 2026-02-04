"""
Simple wrapper to import the Flask app from fyp.BAckend.App
This allows Render to use 'app:app' as the gunicorn entry point
"""
from fyp.BAckend.App import app

if __name__ == "__main__":
    app.run()
