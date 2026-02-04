import os
import importlib.util
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(BASE, 'fyp', 'BAckend', 'App.py')

spec = importlib.util.spec_from_file_location('backend_app', app_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = mod.app

ages = [18,25,35,45,55]
weights = [50,60,70,80,90,100]
heights = [150,160,170,180,190]
fats = [8,15,25,35]
activities = [
    "Light Training (1–3 days/week)",
    "Moderate Training (3–5 days/week)",
    "High Training (6–7 days/week)"
]
genders = ["Male","Female"]

counts = defaultdict(int)
conf_sums = defaultdict(float)
samples = defaultdict(list)

total = 0
with app.test_client() as client:
    for age in ages:
        for weight in weights:
            for height in heights:
                for fat in fats:
                    for activity in activities:
                        for gender in genders:
                            payload = {
                                "age": age,
                                "weight": weight,
                                "height": height,
                                "gender": gender,
                                "activity": activity,
                                "fat": fat,
                                "goal": "Not Sure - Let AI Suggest"
                            }
                            res = client.post('/api/predict', json=payload)
                            data = res.get_json()
                            if not data:
                                continue
                            label = data.get('finalGoal', 'Unknown')
                            conf = data.get('confidence', 0)
                            counts[label] += 1
                            conf_sums[label] += conf
                            if len(samples[label]) < 5:
                                samples[label].append({"payload": payload, "confidence": conf})
                            total += 1

print(f"Total profiles evaluated: {total}")
print("\nCounts per predicted cluster:")
for k, v in sorted(counts.items(), key=lambda x: -x[1]):
    avg_conf = conf_sums[k] / v if v else 0
    print(f"- {k}: {v} profiles (avg confidence {avg_conf:.2f})")

print("\nExample profiles per cluster (up to 5 each):")
for k, arr in samples.items():
    print(f"\n== {k} ==")
    for ex in arr:
        p = ex['payload']
        print(f"{p} -> conf {ex['confidence']}")
