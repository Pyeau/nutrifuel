import requests
import json

BASE_URL = "http://127.0.0.1:5000/api/predict"

test_cases = [
    {
        "name": "Endurance Profile",
        "data": {
            "age": 25, "weight": 60, "height": 160, "gender": "Male",
            "activity": "High Training (6–7 days/week)", "fat": 15,
            "goal": "Not Sure - Let AI Suggest"
        }
    },
    {
         "name": "Build Muscle Profile",
         "data": {
             "age": 25, "weight": 70, "height": 175, "gender": "Male",
             "activity": "High Training (6–7 days/week)", "fat": 12,
             "goal": "Not Sure - Let AI Suggest"
         }
    },
    {
        "name": "Weight Loss Profile",
        "data": {
            "age": 35, "weight": 95, "height": 170, "gender": "Male",
            "activity": "Light Training (1–3 days/week)", "fat": 30,
             "goal": "Not Sure - Let AI Suggest"
        }
    }
]

for test in test_cases:
    print(f"\n🧪 Testing: {test['name']}")
    try:
        response = requests.post(BASE_URL, json=test['data'])
        if response.status_code == 200:
            res_json = response.json()
            print(f"   ✅ Prediction: {res_json.get('finalGoal')}")
            print(f"   📊 Confidence: {res_json.get('confidence')}") # Fixed key from confidence_score
            print(f"   🥗 First Meal Cluster: {res_json['meals'][0]['cluster'] if res_json['meals'] else 'No meals'}")
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   ❌ Connection Failed: {e}")
