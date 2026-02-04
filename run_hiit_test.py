import os
import importlib.util

BASE = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(BASE, 'fyp', 'BAckend', 'App.py')

spec = importlib.util.spec_from_file_location('backend_app', app_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = mod.app

hiit = ("HIIT Profile", {"age":28, "weight":68, "height":172, "gender":"Male", "activity":"High Training (6–7 days/week)", "fat":14, "goal":"Not Sure - Let AI Suggest"})

with app.test_client() as client:
    name, payload = hiit
    print(f"\n🧪 Testing: {name}")
    res = client.post('/api/predict', json=payload)
    data = res.get_json()
    if data:
        print(f"   ✅ Prediction: {data.get('finalGoal')}")
        print(f"   📊 Confidence: {data.get('confidence')}")
        print(f"   🥗 First Meal Cluster: {data['meals'][0]['cluster'] if data.get('meals') else 'No meals'}")
    else:
        print(f"   ❌ No JSON returned (status {res.status_code})")
