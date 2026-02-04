import requests

def test(name, weight, height, activity):
    data = {
        'age': 25, 
        'weight': weight, 
        'height': height, 
        'gender': 'Male', 
        'activity': activity, 
        'fat': 18, 
        'bpm': 65, 
        'goal': 'Not Sure - Let AI Suggest'
    }
    try:
        r = requests.post('http://127.0.0.1:5000/api/predict', json=data)
        res = r.json()
        print(f"[{name}] Profile: {weight}kg, {height}cm, {activity}")
        print(f"      -> Suggestion: {res.get('finalGoal')} (Confidence: {res.get('confidence')})")
    except Exception as e:
        print(f"[{name}] Failed: {e}")

print("--- Testing Model Diversity ---")
test("Average Jo", 75, 175, 'Moderate Training (3–5 days/week)')
test("Weight Loss Candidate", 110, 180, 'Light Training (1–3 days/week)')
test("Underweight/Lean", 60, 185, 'High Training (6–7 days/week)')
test("Athletic", 80, 180, 'High Training (6–7 days/week)')
test("Endurance Profile", 62, 180, 'High Training (6–7 days/week)')
