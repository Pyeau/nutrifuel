import os
import importlib.util
import csv

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

out_path = os.path.join(BASE, 'profile_sweep_results.csv')
fields = [
    'age','weight','height','gender','activity','fat','goal',
    'predicted_plan','confidence','total_meals_calories'
]

rows_written = 0
with app.test_client() as client, open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

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
                            label = data.get('finalGoal', '')
                            conf = data.get('confidence', 0)
                            total_cal = None
                            try:
                                total_cal = sum(m.get('cal', 0) for m in data.get('meals', []))
                            except Exception:
                                total_cal = ''

                            writer.writerow({
                                'age': age,
                                'weight': weight,
                                'height': height,
                                'gender': gender,
                                'activity': activity,
                                'fat': fat,
                                'goal': 'Not Sure - Let AI Suggest',
                                'predicted_plan': label,
                                'confidence': conf,
                                'total_meals_calories': total_cal
                            })
                            rows_written += 1

print(f"Wrote {rows_written} rows to {out_path}")
