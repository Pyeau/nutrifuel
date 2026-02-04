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

# Stats collectors
count = 0
meal_type_counts = defaultdict(int)
meal_type_prot_sum = defaultdict(float)
overall_prot_sum = 0.0
per_profile_tot_prot = []

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
                            if not data: continue
                            if data.get('finalGoal') != 'Build Muscle':
                                continue
                            count += 1
                            profile_prot_total = 0.0
                            for m in data.get('meals', []):
                                prot = float(m.get('prot', 0) if m.get('prot') is not None else 0)
                                meal_type = m.get('type', 'Unknown')
                                meal_type_counts[meal_type] += 1
                                meal_type_prot_sum[meal_type] += prot
                                profile_prot_total += prot
                                overall_prot_sum += prot
                            per_profile_tot_prot.append(profile_prot_total)

# Report
print(f"Build Muscle profiles found: {count}")
if count == 0:
    print("No Build Muscle predictions in sweep.")
else:
    print("\nAverage protein per meal type:")
    for mt, cnt in meal_type_counts.items():
        avg = meal_type_prot_sum[mt] / cnt if cnt else 0
        print(f"- {mt}: {avg:.2f} g (count {cnt})")
    avg_profile_prot = sum(per_profile_tot_prot)/len(per_profile_tot_prot) if per_profile_tot_prot else 0
    print(f"\nAverage total protein per profile (sum of meals): {avg_profile_prot:.2f} g")
    print(f"Overall average protein per meal (all types): {overall_prot_sum / sum(meal_type_counts.values()):.2f} g")
