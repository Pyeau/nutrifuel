import pandas as pd
import joblib
import traceback

MODEL_PATH = 'meal_plan_model.joblib'

try:
    print(f"📂 Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded.")
    
    print("🔍 scanning for Endurance...")
    
    found = False
    for wt in range(50, 90, 2):
        for ht_cm in range(160, 195, 2):
            ht_m = ht_cm / 100
            bmi = wt / (ht_m ** 2)
            
            # Skip unrealistic BMIs
            if bmi < 15 or bmi > 35: continue

            for activity_level in [3]: # Force High Activity for realistic Endurance input
                input_data = {
                    'Age': 25, 
                    'Weight (kg)': wt, 
                    'Height (m)': ht_m, 
                    'BMI': bmi,
                    'Fat_Percentage': 12.0, # Endurance ideal
                    'Avg_BPM': 55.0, 
                    'Resting_BPM': 55, 
                    'Experience_Level': activity_level, # App maps Act->Exp now
                    'Gender': 'Male', 
                    'Goals': 'Unknown', 
                    'Calories_Burned': 2500, 
                    'Workout_Frequency (days/week)': activity_level, 
                    'protein_per_kg': 1.2 # Neutral value
                }
                
                df_in = pd.DataFrame([input_data])
                pred = model.predict(df_in)[0]
                
                if pred == "Endurance":
                    print(f"\n✅ FOUND ENDURANCE!")
                    print(f"Weight: {wt}kg")
                    print(f"Height: {ht_cm}cm")
                    print(f"Activity Level: {activity_level} (3=High)")
                    print(f"BMI: {bmi:.2f}")
                    found = True
                    break
            if found: break
        if found: break
    
    if not found:
        print("❌ Search failed.")

except Exception as e:
    print("\n❌ ERROR")
    traceback.print_exc()
