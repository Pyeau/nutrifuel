import requests
import json

BASE_URL = "http://127.0.0.1:5000/api/predict"

# Extreme Endurance Profile
data = {
    'age': 25, 
    'weight': 55,       # Very light
    'height': 175,      # Tall -> Low BMI
    'gender': 'Male', 
    'activity': 'High Training (6–7 days/week)', 
    'fat': 6,           # Very low body fat (Marathon runner level)
    'bpm': 45,          # Low resting HR
    'goal': 'Not Sure - Let AI Suggest'
}

print(f"🧪 Testing Strict Endurance Profile: {data}")

try:
    r = requests.post(BASE_URL, json=data)
    res = r.json()
    print(f"\n✅ Prediction: {res.get('finalGoal')}")
    print(f"📊 Confidence: {res.get('confidence_score')}")
except Exception as e:
    print(f"❌ Failed: {e}")
