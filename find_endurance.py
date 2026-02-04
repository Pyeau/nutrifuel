import requests

def find_endurance():
    print("🔎 Searching for Endurance profile...")
    # Try different combinations
    for weight in range(50, 90, 5):
        for height in range(160, 195, 5):
            for activity in ['Light Training (1–3 days/week)', 'Moderate Training (3–5 days/week)', 'High Training (6–7 days/week)']:
                data = {
                    'age': 25, 
                    'weight': weight, 
                    'height': height, 
                    'gender': 'Male', 
                    'activity': activity, 
                    'fat': 12, 
                    'bpm': 55, 
                    'goal': 'Not Sure - Let AI Suggest'
                }
                try:
                    r = requests.post('http://127.0.0.1:5000/api/predict', json=data)
                    res = r.json()
                    goal = res.get('finalGoal')
                    if goal == "Endurance":
                        print(f"✅ FOUND! W:{weight}kg, H:{height}cm, Act:{activity} -> {goal}")
                        return
                except:
                    pass
    print("❌ No endurance profile found in sample range.")

if __name__ == "__main__":
    find_endurance()
