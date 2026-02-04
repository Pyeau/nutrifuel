# ==========================================================
# ATHLETIC DIET RECOMMENDATION SYSTEM - MODEL TRAINING SCRIPT
# ==========================================================

# --- [CRITICAL FIXES FOR PYTHON 3.14 / SKLEARN 1.6+] ----------
import sklearn.utils.validation
if not hasattr(sklearn.utils.validation, "_is_pandas_df"):
    def _is_pandas_df(x):
        return hasattr(x, "dtypes") and hasattr(x, "columns")
    sklearn.utils.validation._is_pandas_df = _is_pandas_df

import sklearn.ensemble
_OriginalAdaBoost = sklearn.ensemble.AdaBoostClassifier
class SafeAdaBoostClassifier(_OriginalAdaBoost):
    def __init__(self, *args, **kwargs):
        if 'algorithm' in kwargs: del kwargs['algorithm']
        super().__init__(*args, **kwargs)
sklearn.ensemble.AdaBoostClassifier = SafeAdaBoostClassifier
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
import warnings
import sys
import os

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report, silhouette_score, precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore', category=UserWarning)
sys.stdout.reconfigure(line_buffering=True)

# ==========================================================
# PART 0 – LOAD ATHLETE DATA
# ==========================================================
def load_and_explore_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"✅ '{filepath}' loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"❌ Error: '{filepath}' not found.")
        return None

# ==========================================================
# PART 0A – MEAL TIME IDENTIFICATION
# ==========================================================
def identify_meal_time(food_name, calories, proteins, carbs, fats):
    food_name_lower = str(food_name).lower()
    
    BREAKFAST = ['oatmeal', 'cereal', 'pancake', 'waffle', 'toast', 'bagel', 'egg', 'bacon', 'sausage', 'granola', 'yogurt', 'smoothie', 'coffee']
    LUNCH = ['sandwich', 'wrap', 'salad', 'soup', 'burger', 'pizza', 'pasta', 'noodle', 'rice', 'taco', 'burrito', 'panini']
    DINNER = ['steak', 'chicken breast', 'salmon', 'roast', 'grilled', 'stew', 'curry', 'casserole', 'filet', 'chop', 'dinner']
    SNACK = ['bar', 'chip', 'cracker', 'nut', 'seed', 'popcorn', 'cookie', 'candy', 'chocolate', 'fruit', 'apple', 'banana', 'snack']

    for k in BREAKFAST:
        if k in food_name_lower: return 'Breakfast'
    for k in SNACK:
        if k in food_name_lower: return 'Snack'
    for k in LUNCH:
        if k in food_name_lower: return 'Lunch'
    for k in DINNER:
        if k in food_name_lower: return 'Dinner'
    
    if proteins > 25 and calories > 350: return 'Dinner'
    elif proteins > 15 and 200 < calories <= 400: return 'Lunch'
    elif carbs > 30 and proteins < 10: return 'Breakfast'
    elif calories < 250: return 'Snack'
    return 'Lunch'

# ==========================================================
# PART 1 – RANDOM FOREST (SUPERVISED)
# ==========================================================
def build_meal_plan_classifier(df):
    print("\n" + "=" * 60)
    print("🏋️ TRAINING RANDOM FOREST MODEL")
    print("=" * 60)

    target = 'Meal_Plan'
    numerical = ['Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Fat_Percentage', 'Avg_BPM', 'Resting_BPM', 'Experience_Level', 'Calories_Burned', 'Workout_Frequency (days/week)', 'protein_per_kg']
    categorical = ['Gender', 'Goals']
    features = numerical + categorical

    df_clean = df.dropna(subset=[target] + features).copy()
    
    def reassign_meal_plan(row):
        goal = str(row['Goals'])
        fat = row['Fat_Percentage']
        if fat > 25: return "Weight Loss"
        if "Muscle" in goal or row['protein_per_kg'] > 1.8: return "Build Muscle"
        if "Endurance" in goal or row['Calories_Burned'] > 2800: return "Endurance"
        if "HIIT" in goal: return "HIIT"
        return "Balanced"
    
    df_clean['Meal_Plan'] = df_clean.apply(reassign_meal_plan, axis=1)
    
    X = df_clean[features]
    y = df_clean[target]

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical)
    ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    try:
        X_train_proc = preprocessor.fit_transform(X_train)
        smote = SMOTE(random_state=42, k_neighbors=1) 
        X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
        model_pipeline.named_steps['model'].fit(X_train_res, y_train_res)
        print("✅ SMOTE applied successfully.")
    except Exception as e:
        print(f"⚠️ SMOTE failed. Training on original data. ({e})")
        model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    joblib.dump(model_pipeline, 'meal_plan_model.joblib')
    return model_pipeline

# ==========================================================
# PART 2 – K-MEANS CLUSTERING (Fixed 1-to-1 Mapping)
# ==========================================================
def map_cluster_to_meal_plan(food_df, n_clusters):
    print("\n🧠 [1-TO-1 MAPPING ACTIVE] Mapping clusters to unique labels...")
    
    # 1. Get Cluster Centers
    cluster_stats = food_df.groupby('Food_Cluster')[['Proteins', 'Carbs', 'Fats']].mean().reset_index()
    totals = cluster_stats[['Proteins', 'Carbs', 'Fats']].sum(axis=1).replace(0, 1)
    cluster_stats['p'] = cluster_stats['Proteins'] / totals
    cluster_stats['c'] = cluster_stats['Carbs'] / totals
    cluster_stats['f'] = cluster_stats['Fats'] / totals

    # 2. Define Ideal Targets
    targets = {
        "Build Muscle": {'p': 0.40, 'c': 0.35, 'f': 0.25},
        "Weight Loss":  {'p': 0.50, 'c': 0.20, 'f': 0.30},
        "Endurance":    {'p': 0.20, 'c': 0.60, 'f': 0.20},
        "HIIT":         {'p': 0.30, 'c': 0.45, 'f': 0.25},
        "Balanced":     {'p': 0.25, 'c': 0.45, 'f': 0.30}
    }
    
    target_names = list(targets.keys())
    
    # 3. Create Distance Matrix (Cluster vs Target)
    distances = np.zeros((n_clusters, len(target_names)))
    
    for i, row in cluster_stats.iterrows():
        c_id = int(row['Food_Cluster'])
        for j, name in enumerate(target_names):
            t = targets[name]
            # Manhattan Distance
            dist = abs(row['p'] - t['p']) + abs(row['c'] - t['c']) + abs(row['f'] - t['f'])
            distances[c_id, j] = dist

    # 4. Greedy Assignment (Find best global match, lock it, repeat)
    mapping = {}
    assigned_clusters = set()
    assigned_targets = set()
    
    print("\n  [Cluster] -> [Unique Label Assigned]")
    
    # We loop exactly 5 times to assign 5 clusters to 5 unique labels
    for _ in range(n_clusters):
        min_dist = float('inf')
        best_c = -1
        best_t_idx = -1
        
        # Search for the best available pair
        for c in range(n_clusters):
            if c in assigned_clusters: continue
            for t_idx in range(len(target_names)):
                if t_idx in assigned_targets: continue
                
                if distances[c, t_idx] < min_dist:
                    min_dist = distances[c, t_idx]
                    best_c = c
                    best_t_idx = t_idx
        
        # Lock the best match found
        if best_c != -1:
            final_name = target_names[best_t_idx]
            mapping[best_c] = final_name
            assigned_clusters.add(best_c)
            assigned_targets.add(best_t_idx)
            print(f"  Cluster {best_c} -> {final_name}")

    return mapping

def build_food_clusters(food_df):
    print("\n" + "=" * 60)
    print("🍽️ CLUSTERING FOODS")
    print("=" * 60)

    food_df = food_df[(food_df['Calories'] > 50) & (food_df['Calories'] < 900)].copy()
    
    totals = food_df[['Proteins', 'Carbs', 'Fats']].sum(axis=1).replace(0, 1)
    food_df['p_ratio'] = food_df['Proteins'] / totals
    food_df['c_ratio'] = food_df['Carbs'] / totals
    food_df['f_ratio'] = food_df['Fats'] / totals
    
    features = ['p_ratio', 'c_ratio', 'f_ratio']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(food_df[features])
    
    # Force 5 Clusters for 5 Meal Plans
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    food_df['Food_Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Use the new 1-to-1 Mapping Function
    mapping = map_cluster_to_meal_plan(food_df, 5)
    
    food_df['Meal_Plan'] = food_df['Food_Cluster'].map(mapping)
    
    joblib.dump(kmeans, 'food_kmeans_model.joblib')
    return food_df

# ==========================================================
# PART 3 – FOOD DATABASE LOADING
# ==========================================================
def load_and_process_food_database():
    print("\n🥗 Processing Food Database...")
    try:
        df = pd.read_csv('cleaned_nutrition_dataset_per100g.csv')
    except:
        try:
            df = pd.read_csv('daily_food_nutrition_dataset.csv')
        except:
            print("⚠️ CSVs not found. Using Dummy Data.")
            df = pd.DataFrame({
                'food': ['Chicken', 'Rice', 'Oats', 'Steak', 'Salad', 'Fish', 'Pasta', 'Egg', 'Nuts', 'Yogurt'],
                'Calories': [165, 130, 389, 271, 50, 200, 150, 140, 600, 100],
                'Proteins': [31, 2, 16, 26, 1, 25, 5, 12, 20, 10],
                'Carbs': [0, 28, 66, 0, 10, 0, 30, 1, 15, 15],
                'Fats': [3, 0, 6, 19, 0, 10, 2, 10, 50, 5]
            })

    col_map = {
        'food': 'meal_name', 'Food_Item': 'meal_name', 'name': 'meal_name',
        'Carbohydrates (g per 100g)': 'Carbs', 'Carbohydrates (g)': 'Carbs', 'carbs': 'Carbs',
        'Protein (g per 100g)': 'Proteins', 'Protein (g)': 'Proteins', 'protein': 'Proteins',
        'Fat (g per 100g)': 'Fats', 'Fat (g)': 'Fats', 'fat': 'Fats',
        'Calories (kcal per 100g)': 'Calories', 'Calories (kcal)': 'Calories', 'calories': 'Calories'
    }
    df = df.rename(columns=col_map)
    required = ['meal_name', 'Carbs', 'Proteins', 'Fats', 'Calories']
    existing = [c for c in required if c in df.columns]
    df = df[existing].dropna()
    
    df['Meal_Time'] = df.apply(lambda x: identify_meal_time(x['meal_name'], x['Calories'], x['Proteins'], x['Carbs'], x['Fats']), axis=1)
    df = build_food_clusters(df)
    
    df.to_csv('improved_food_database.csv', index=False)
    print("✅ improved_food_database.csv saved.")
    return df

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    print("🚀 STARTING TRAINING...")
    
    df_athlete = load_and_explore_data('Final_data.csv')
    if df_athlete is not None:
        build_meal_plan_classifier(df_athlete)
    
    load_and_process_food_database()
    
    print("\n✅ DONE. Run 'streamlit run appp.py'")