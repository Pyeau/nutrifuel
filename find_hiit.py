import requests
import time

def find_hiit():
    print("🔎 Searching for HIIT profile...")
    # HIIT usually implies high intensity but maybe not as much output as muscle building?
    # Or specific heart rate zones?
    
    # Try different combinations
    for weight in range(60, 90, 5):
        for height in range(160, 190, 5):
            for activity in ['Moderate Training (3–5 days/week)', 'High Training (6–7 days/week)']:
                for age in [20, 25, 30]:
                    data = {
                        'age': age, 
                        'weight': weight, 
                        'height': height, 
                        'gender': 'Male', 
                        'activity': activity, 
                        'fat': 15, # Average/Athletic fat
                        'bpm': 60, # Average/Athletic bpm
                        'goal': 'Not Sure - Let AI Suggest'
                    }
                    try:
                        r = requests.post('http://127.0.0.1:5000/api/predict', json=data)
                        res = r.json()
                        goal = res.get('finalGoal')
                        if goal == "HIIT":
                            print(f"✅ FOUND HIIT! {data}")
                            return
                    except:
                        pass
    print("❌ No HIIT profile found in sample range.")

if __name__ == "__main__":
    find_hiit()
