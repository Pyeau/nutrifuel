import pandas as pd
import numpy as np
import random

def generate_synthetic_data(num_samples=15000):
    print(f"🚀 Generating {num_samples} synthetic athlete records...")
    
    data = []
    # 5 targets
    targets = ["Weight Loss", "Build Muscle", "HIIT", "Endurance", "Balanced"]
    samples_per_class = num_samples // len(targets)
    
    for target in targets:
        for _ in range(samples_per_class):
            gender = random.choice(["Male", "Female"])
            age = random.randint(18, 55)
            height_m = round(random.uniform(1.55, 1.95), 2)
            
            # Logic-based data generation to ensure the model can LEARN
            if target == "Weight Loss":
                bmi = random.uniform(28, 35)
                fat_pct = random.uniform(25, 40)
                wt = round(bmi * (height_m ** 2), 1)
                activity = random.randint(1, 3)
                exp = random.randint(1, 2)
            elif target == "Build Muscle":
                bmi = random.uniform(20, 25)
                fat_pct = random.uniform(10, 18)
                wt = round(bmi * (height_m ** 2), 1)
                activity = random.randint(3, 5)
                exp = 3
            elif target == "Endurance":
                bmi = random.uniform(18, 22)
                fat_pct = random.uniform(8, 15)
                wt = round(bmi * (height_m ** 2), 1)
                activity = random.randint(5, 7)
                exp = random.randint(2, 3)
            elif target == "HIIT":
                bmi = random.uniform(22, 26)
                fat_pct = random.uniform(15, 22)
                wt = round(bmi * (height_m ** 2), 1)
                activity = random.randint(4, 6)
                exp = random.randint(2, 3)
            else: # Balanced
                bmi = random.uniform(21, 24)
                fat_pct = random.uniform(18, 25)
                wt = round(bmi * (height_m ** 2), 1)
                activity = random.randint(2, 4)
                exp = random.randint(1, 2)
            
            avg_bpm = 60 + (activity * 10) + random.randint(-5, 5)
            rest_bpm = 60 + random.randint(-10, 10)
            cal_burned = (activity * 400) + (wt * 2)
            prot_kg = 1.2 + (0.2 if target == "Build Muscle" else 0)
            
            data.append({
                'Age': age + random.randint(-1, 1), # Subtle noise
                'Gender': gender,
                'Weight (kg)': wt + random.uniform(-0.5, 0.5), # Subtle noise
                'Height (m)': height_m,
                'BMI': round(bmi + random.uniform(-0.5, 0.5), 2), # Introduce overlap
                'Fat_Percentage': round(fat_pct + random.uniform(-2, 2), 1), # Introduce overlap
                'Avg_BPM': avg_bpm + random.randint(-10, 10), # High noise
                'Resting_BPM': rest_bpm + random.randint(-5, 5),
                'Experience_Level': exp,
                'Calories_Burned': round(cal_burned + random.uniform(-100, 100), 1),
                'Workout_Frequency (days/week)': activity,
                'protein_per_kg': round(prot_kg + random.uniform(-0.1, 0.1), 2),
                'Goals': target,
                'Meal_Plan': target,
                'Max_BPM': 220 - age,
                'Session_Duration (hours)': 1.0
            })
            
    df = pd.DataFrame(data)
    
    # ✅ INTRODUCE CATEGORY OVERLAP (Noise)
    # Randomly flip 15% of labels to create realistic misclassifications
    flip_mask = np.random.rand(len(df)) < 0.15
    df.loc[flip_mask, 'Meal_Plan'] = df.loc[flip_mask, 'Meal_Plan'].apply(lambda x: random.choice([t for t in targets if t != x]))
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('Final_data.csv', index=False)
    print("✅ Realistic Noisy 'Final_data.csv' generated and saved.")

if __name__ == "__main__":
    generate_synthetic_data()
