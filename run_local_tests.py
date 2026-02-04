import os
import importlib.util

# Load the App module by path
BASE = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(BASE, 'fyp', 'BAckend', 'App.py')

spec = importlib.util.spec_from_file_location('backend_app', app_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = mod.app

test_cases = [
    ("Endurance Profile", {"age":25, "weight":60, "height":160, "gender":"Male", "activity":"High Training (6–7 days/week)", "fat":15, "goal":"Not Sure - Let AI Suggest"}),
    ("Build Muscle Profile", {"age":25, "weight":70, "height":175, "gender":"Male", "activity":"High Training (6–7 days/week)", "fat":12, "goal":"Not Sure - Let AI Suggest"}),
    ("Weight Loss Profile", {"age":35, "weight":95, "height":170, "gender":"Male", "activity":"Light Training (1–3 days/week)", "fat":30, "goal":"Not Sure - Let AI Suggest"})
]

with app.test_client() as client:
    for name, payload in test_cases:
        print(f"\n🧪 Testing: {name}")
        res = client.post('/api/predict', json=payload)
        try:
            data = res.get_json()
            if data is None:
                print(f"   ❌ No JSON returned (status {res.status_code})")
            else:
                print(f"   ✅ Prediction: {data.get('finalGoal')}")
                print(f"   📊 Confidence: {data.get('confidence')}")
                print(f"   🥗 First Meal Cluster: {data['meals'][0]['cluster'] if data.get('meals') else 'No meals'}")
        except Exception as e:
            print(f"   ❌ Error parsing response: {e}")
