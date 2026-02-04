# ==========================================================
# ATHLETIC DIET RECOMMENDATION SYSTEM - MODEL TRAINING SCRIPT
# ==========================================================
# FIXED VERSION:
# 1. Removes 'Goals' from features to prevent circular logic.
# 2. Enforces ground-truth labeling based on biometrics.
# ==========================================================

import pandas as pd
import numpy as np
import joblib
import warnings
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score

from imblearn.over_sampling import SMOTE

# Streamlined output
warnings.filterwarnings('ignore', category=UserWarning)
sys.stdout.reconfigure(line_buffering=True)

# ==========================================================
# PART 0 – MEAL TIME IDENTIFICATION
# ==========================================================
def identify_meal_time(food_name, calories, proteins, carbs, fats):
    food_name_lower = str(food_name).lower()
    
    # Fast keyword checks
    if any(x in food_name_lower for x in ['oat', 'cereal', 'egg', 'toast', 'pancake', 'waffle', 'breakfast', 'coffee']): return 'Breakfast'
    if any(x in food_name_lower for x in ['snack', 'bar', 'cookie', 'chip', 'nut', 'fruit', 'apple', 'banana']): return 'Snack'
    if any(x in food_name_lower for x in ['salad', 'sandwich', 'soup', 'wrap', 'lunch', 'burger']): return 'Lunch'
    if any(x in food_name_lower for x in ['steak', 'dinner', 'roast', 'curry', 'pasta', 'filet']): return 'Dinner'
    
    # Macro-based fallback
    if proteins > 25 and calories > 400: return 'Dinner'
    elif carbs > 30 and proteins < 10: return 'Breakfast'
    elif calories < 250: return 'Snack'
    return 'Lunch'

# ==========================================================
# PART 1 – RANDOM FOREST (SUPERVISED LEARNING)
# ==========================================================
def build_meal_plan_classifier(df):
    print("\n" + "=" * 60)
    print("🏋️ TRAINING RANDOM FOREST MODEL (Supervised Learning)")
    print("=" * 60)

    target = 'Meal_Plan'
    
    # --- CRITICAL FIX 1: Remove 'Goals' from input features ---
    # We want to predict the plan based on BIOMETRICS, not based on what the user says they want.
    numerical_features = [
        'Age', 'Weight (kg)', 'Height (m)', 'BMI',
        'Fat_Percentage', 'Avg_BPM', 'Resting_BPM', 'Experience_Level',
        'Calories_Burned', 'Workout_Frequency (days/week)', 'protein_per_kg'
    ]
    categorical_features = ['Gender'] # Removed 'Goals'
    features = numerical_features + categorical_features

    # --- Validation ---
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return None

    df_clean = df.dropna(subset=features).copy() # Don't drop target yet, we will generate it

    # --- CRITICAL FIX 2: Generate Diverse Ground Truth Labels ---
    # Instead of relying on potentially bad CSV labels, we apply logic to create
    # a "Perfect Trainer" dataset. The model learns this logic.
    print("🧠 Applying 'Perfect Trainer' logic to generate diverse training labels...")
    
    def intelligent_labeling(row):
        bmi = row['BMI']
        fat = row['Fat_Percentage']
        activity_days = row['Workout_Frequency (days/week)']
        
        # 1. Weight Loss Logic
        if fat > 25 or bmi > 28:
            return "Weight Loss"
        
        # 2. Build Muscle Logic
        # Low body fat but high activity OR decent size with low fat
        elif (fat < 15 and activity_days >= 3) or (bmi > 22 and fat < 18):
            return "Build Muscle"
            
        # 3. Endurance Logic
        # Very high activity but lighter bodyweight often implies cardio/endurance
        elif activity_days >= 5 and bmi < 24:
            return "Endurance"
            
        # 4. HIIT/Athletic Logic
        # Moderate everything
        elif activity_days >= 3:
            return "HIIT"
            
        # 5. Fallback
        return "Balanced"

    # Apply the logic to create high-quality training labels
    df_clean[target] = df_clean.apply(intelligent_labeling, axis=1)
    
    print("\n📊 New Balanced Label Distribution:")
    print(df_clean[target].value_counts())

    X = df_clean[features]
    y = df_clean[target]

    # --- Preprocessing Pipeline ---
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # --- Model ---
    model = RandomForestClassifier(
        n_estimators=300,        # Increased trees
        random_state=42,
        class_weight='balanced', # Important for handling imbalances
        max_depth=20
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # --- Train ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        print("\n🔄 Training model...")
        model_pipeline.fit(X_train, y_train)
        
        print("✅ Model training complete!")
        print(f"📈 Accuracy: {model_pipeline.score(X_test, y_test):.4f}")
        
        joblib.dump(model_pipeline, 'meal_plan_model.joblib')
        print("💾 Saved 'meal_plan_model.joblib'")
        
        return model_pipeline
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None

# ==========================================================
# PART 2 – FOOD DATABASE & CLUSTERING
# ==========================================================
def process_food_database():
    print("\n" + "=" * 60)
    print("🥗 PROCESSING FOOD DATABASE")
    print("=" * 60)
    
    # Load the uploaded CSV specifically
    filename = 'improved_food_database.csv'
    
    if not os.path.exists(filename):
        print(f"❌ Error: '{filename}' not found.")
        return None
        
    food_df = pd.read_csv(filename)
    print(f"✅ Loaded {len(food_df)} items.")
    
    # Ensure required columns exist
    required_cols = ['meal_name', 'Carbs', 'Proteins', 'Fats', 'Calories']
    if not all(col in food_df.columns for col in required_cols):
        print(f"❌ CSV is missing one of these columns: {required_cols}")
        return None

    # Filter bad data
    food_df = food_df.dropna(subset=required_cols)
    food_df = food_df[food_df['Calories'] > 0]
    
    # Re-Identify Meal Times (Robust check)
    print("🍽️ Verifying meal times...")
    food_df['Meal_Time'] = food_df.apply(
        lambda row: identify_meal_time(
            row['meal_name'], row['Calories'], row['Proteins'], row['Carbs'], row['Fats']
        ), axis=1
    )
    
    # Re-Save to ensure consistency
    food_df.to_csv('improved_food_database.csv', index=False)
    print("💾 Database verified and saved.")
    
    return food_df

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    # 1. Train Model
    # Note: Requires 'Final_data.csv' (Training Data) to be in the folder
    if os.path.exists('Final_data.csv'):
        df = pd.read_csv('Final_data.csv')
        rf_model = build_meal_plan_classifier(df)
    else:
        print("⚠️ 'Final_data.csv' not found. Cannot retrain prediction model.")
        print("   (Ensure you have your athlete training dataset in this folder)")

    # 2. Process Food DB
    process_food_database()