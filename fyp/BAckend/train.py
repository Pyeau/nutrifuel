# ==========================================================
# ATHLETIC DIET RECOMMENDATION SYSTEM - MODEL TRAINING SCRIPT
# ==========================================================
# Hybrid Architecture:
# - Random Forest (Supervised Learning)
# - K-Means (Unsupervised Learning)
# - Meal Time Classification (Breakfast/Lunch/Dinner/Snack)
# - Integrated for Personalized Food Recommendations
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
import sys
import ast
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    silhouette_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# Ensure you have installed: pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Streamlined output
warnings.filterwarnings('ignore', category=UserWarning)
sys.stdout.reconfigure(line_buffering=True)


# ==========================================================
# PART 0 – LOAD ATHLETE DATA
# ==========================================================
def load_and_explore_data(filepath):
    """Load the athlete dataset for Random Forest training"""
    try:
        if not os.path.exists(filepath):
            print(f"❌ Error: '{filepath}' not found. Please ensure your training data CSV is in the folder.")
            return None
        df = pd.read_csv(filepath)
        print(f"✅ '{filepath}' loaded successfully.")
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


# ==========================================================
# PART 0A – MEAL TIME IDENTIFICATION SYSTEM
# ==========================================================
def identify_meal_time(food_name, calories, proteins, carbs, fats):
    """
    Identify when a food is typically eaten (Breakfast, Lunch, Dinner, Snack).
    Based on research: USDA NHANES, International Breakfast Research Initiative, 
    and Dietary Guidelines for meal classification.
    
    Returns: meal_time (string)
    """
    food_name_lower = str(food_name).lower()
    
    # Define keyword dictionaries for meal time classification
    BREAKFAST_KEYWORDS = [
        'oatmeal', 'cereal', 'pancake', 'waffle', 'toast', 'bagel', 'muffin',
        'breakfast', 'scrambled', 'omelet', 'omelette', 'egg', 'bacon',
        'sausage', 'granola', 'yogurt', 'smoothie', 'coffee', 'juice',
        'croissant', 'porridge', 'french toast', 'crepe', 'hash brown'
    ]
    
    LUNCH_KEYWORDS = [
        'sandwich', 'wrap', 'salad', 'soup', 'burger', 'pizza', 'pasta',
        'noodle', 'rice bowl', 'burrito', 'taco', 'sub', 'panini',
        'quesadilla', 'pita', 'flatbread', 'lunch', 'deli'
    ]
    
    DINNER_KEYWORDS = [
        'steak', 'chicken breast', 'salmon', 'fish fillet', 'roast', 'grilled',
        'baked chicken', 'baked fish', 'stir fry', 'curry', 'casserole', 
        'roasted', 'braised', 'pot roast', 'dinner', 'ribeye', 'pork chop',
        'lamb chop', 'filet', 'tenderloin', 'prime rib'
    ]
    
    SNACK_KEYWORDS = [
        'bar', 'chip', 'cracker', 'nut', 'seed', 'trail mix', 'popcorn',
        'pretzel', 'cookie', 'brownie', 'cake', 'candy', 'chocolate',
        'fruit', 'berry', 'apple', 'banana', 'orange', 'grape', 'snack',
        'protein bar', 'energy bar', 'granola bar'
    ]
    
    # Check for meal time keywords in order of specificity
    for keyword in BREAKFAST_KEYWORDS:
        if keyword in food_name_lower: return 'Breakfast'
    
    for keyword in SNACK_KEYWORDS:
        if keyword in food_name_lower: return 'Snack'
    
    for keyword in LUNCH_KEYWORDS:
        if keyword in food_name_lower: return 'Lunch'
    
    for keyword in DINNER_KEYWORDS:
        if keyword in food_name_lower: return 'Dinner'
    
    # If no keyword match, use nutritional profile
    if proteins > 25 and calories > 300: return 'Dinner'
    elif proteins > 15 and 200 < calories <= 400: return 'Lunch'
    elif carbs > 30 and proteins < 10: return 'Breakfast'
    elif calories < 200: return 'Snack'
    elif 250 < calories <= 450: return 'Lunch'
    elif calories > 450: return 'Dinner'
    
    return 'Lunch'


# ==========================================================
# PART 1 – RANDOM FOREST (SUPERVISED LEARNING)
# ==========================================================
def build_meal_plan_classifier(df):
    """
    Train Random Forest with SMOTE integrated to handle class imbalance.
    Predicts 'Meal_Plan' from athlete attributes.
    """
    print("\n" + "=" * 60)
    print("🏋️ TRAINING RANDOM FOREST MODEL (Supervised Learning)")
    print("=" * 60)

    target = 'Meal_Plan'
    numerical_features = [
        'Age', 'Weight (kg)', 'Height (m)', 'BMI',
        'Fat_Percentage', 'Avg_BPM', 'Resting_BPM', 'Experience_Level',
        'Calories_Burned', 'Workout_Frequency (days/week)', 'protein_per_kg'
    ]
    categorical_features = ['Gender', 'Goals']
    features = numerical_features + categorical_features

    # --- Validation ---
    missing = [c for c in [target] + features if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return None

    df_clean = df.dropna(subset=[target] + features)
    if df_clean.empty:
        print("❌ No valid data for training after removing missing rows.")
        return None

    # Map old macro-based meal plans to new goal-based meal plans
    macro_to_goal_mapping = {
        "High Protein": "Build Muscle",
        "High Protein High Carb": "Build Muscle",
        "High Protein High Fat": "Build Muscle",
        "High Protein High Carb High Fat": "Build Muscle",
        "High Carb": "Endurance",
        "High Carb High Fat": "Balanced",
        "High Fat": "Weight Loss",
        "Other": "Balanced"
    }
    # Only map if the values exist; if dataset already has correct labels, skip/adjust
    if any(x in macro_to_goal_mapping for x in df_clean['Meal_Plan'].unique()):
        df_clean['Meal_Plan'] = df_clean['Meal_Plan'].map(macro_to_goal_mapping).fillna(df_clean['Meal_Plan'])
    
    # Create more diverse training labels based on user profiles logic
    df_clean['fat_bmi_ratio'] = df_clean['Fat_Percentage'] / (df_clean['BMI'] + 1)
    df_clean['activity_score'] = (df_clean['Avg_BPM'] - df_clean['Resting_BPM']) * df_clean['Experience_Level']
    
    # Reassign meal plans based on physical metrics (simulation/heuristic for training)
    def reassign_meal_plan(row):
        bmi = row['BMI']
        fat = row['Fat_Percentage']
        
        # Simple heuristic logic for Ground Truth training
        if fat > 25 or (bmi > 29):
            return "Weight Loss"
        elif row['Goals'] == 'Build Muscle' or (bmi < 25 and row['Workout_Frequency (days/week)'] >= 4):
            return "Build Muscle"
        elif row['Goals'] == 'Endurance':
            return "Endurance"
        elif row['Goals'] == 'HIIT':
            return "HIIT"
        else:
            return "Balanced"
    
    # Overwrite labels with logic if dataset labels are weak, otherwise stick to data
    # df_clean['Meal_Plan'] = df_clean.apply(reassign_meal_plan, axis=1)
    
    print("\n📊 Meal Plan Distribution:")
    dist = df_clean['Meal_Plan'].value_counts()
    print(dist)
    
    # Standardized goal mapping for Goals column
    goal_mapping = {
        "Build Muscle": "Build Muscle",
        "Endurance": "Endurance",
        "HIIT": "HIIT",
        "Weight Loss": "Weight Loss",
        "General": "Balanced",
        "Balanced": "Balanced"
    }
    df_clean['Goals'] = df_clean['Goals'].replace(goal_mapping)

    X = df_clean[features]
    y = df_clean[target]

    # --- Preprocessing ---
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

    # --- Random Forest Classifier ---
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        max_depth=15
    )

    # --- Full Pipeline ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    if len(y.unique()) <= 1:
        print("❌ Not enough unique target values to train the model.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n🔄 Training Random Forest model...")
    model_pipeline.fit(X_train, y_train)
    print("✅ Model training complete!")
    
    # --- Evaluation ---
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📈 Accuracy: {accuracy:.4f}")
    
    # --- Save Model ---
    joblib.dump(model_pipeline, 'meal_plan_model.joblib')
    print("\n💾 Model saved as 'meal_plan_model.joblib'")
    
    return model_pipeline


# ==========================================================
# PART 2 – K-MEANS CLUSTERING (UNSUPERVISED LEARNING)
# ==========================================================
def map_cluster_to_meal_plan(food_df, n_clusters, target_labels=None):
    """
    Simplified mapping: Assigns each cluster to the label it is mathematically closest to.
    """
    print("\n🧠 Mapping clusters to meal plans...")
    
    # 1. Calculate the average macros for each cluster
    cluster_stats = (
        food_df.groupby('Food_Cluster')[['Proteins', 'Carbs', 'Fats']]
        .mean()
        .reset_index()
    )
    
    # Calculate ratios
    totals = cluster_stats[['Proteins', 'Carbs', 'Fats']].sum(axis=1).replace(0, 1e-6)
    cluster_stats['p_ratio'] = cluster_stats['Proteins'] / totals
    cluster_stats['c_ratio'] = cluster_stats['Carbs'] / totals
    cluster_stats['f_ratio'] = cluster_stats['Fats'] / totals

    # 2. Define "Ideal" profiles
    ideal_profiles = {
        "Weight Loss":  {"p": 0.50, "c": 0.20, "f": 0.30},
        "Build Muscle": {"p": 0.40, "c": 0.40, "f": 0.20},
        "Endurance":    {"p": 0.20, "c": 0.60, "f": 0.20},
        "HIIT":         {"p": 0.30, "c": 0.40, "f": 0.30},
        "Balanced":     {"p": 0.25, "c": 0.45, "f": 0.30}
    }

    if target_labels is None:
        target_labels = list(ideal_profiles.keys())

    mapping = {}

    for i, row in cluster_stats.iterrows():
        cluster_id = int(row['Food_Cluster'])
        p, c, f = row['p_ratio'], row['c_ratio'], row['f_ratio']
        
        best_label = target_labels[0]
        min_dist = float('inf')
        
        for label in target_labels:
            prof = ideal_profiles.get(label, {'p': 0.33, 'c': 0.33, 'f': 0.33})
            # Manhattan Distance
            dist = (abs(p - prof['p']) + abs(c - prof['c']) + abs(f - prof['f']))
            
            if dist < min_dist:
                min_dist = dist
                best_label = label
        
        mapping[cluster_id] = best_label
        print(f"  -> Cluster {cluster_id} ({p:.2f}P/{c:.2f}C/{f:.2f}F) mapped to '{best_label}'")

    return mapping

def build_food_clusters(food_df):
    """
    Apply K-Means to dataset of foods by clustering on nutritional ratios.
    """
    print("\n" + "=" * 60)
    print("🍽️ CLUSTERING FOODS WITH K-MEANS")
    print("=" * 60)

    base_features = ['Proteins', 'Carbs', 'Fats', 'Calories']
    food_df = food_df.dropna(subset=base_features).copy()

    # Pre-filtering for quality
    food_df = food_df[
        (food_df['Proteins'] >= 5) &
        (food_df['Calories'] >= 50)
    ].copy()
    
    if len(food_df) < 50:
        print("❌ Not enough data to cluster.")
        return None

    print("Creating macronutrient ratio features...")
    totals = food_df[['Proteins', 'Carbs', 'Fats']].sum(axis=1).replace(0, 1e-6)
    
    food_df['p_ratio'] = food_df['Proteins'] / totals
    food_df['c_ratio'] = food_df['Carbs'] / totals
    food_df['f_ratio'] = food_df['Fats'] / totals
    
    cluster_features = ['p_ratio', 'c_ratio', 'f_ratio']
    
    # Sampling for speed if dataset is huge
    if len(food_df) > 20000:
        food_df_sample = food_df.sample(n=20000, random_state=42)
    else:
        food_df_sample = food_df.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(food_df_sample[cluster_features])

    k = 5
    print(f"\n✓ Using k={k} clusters")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    food_df_sample['Food_Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Predict for all
    X_all_scaled = scaler.transform(food_df[cluster_features])
    food_df['Food_Cluster'] = kmeans.predict(X_all_scaled)

    mapping = map_cluster_to_meal_plan(food_df, k)

    joblib.dump(kmeans, 'food_kmeans_model.joblib')
    joblib.dump(scaler, 'food_scaler.joblib')
    
    # Map clusters to meaningful names
    food_df['Meal_Plan'] = food_df['Food_Cluster'].map(mapping).fillna('Balanced')
    return food_df


# ==========================================================
# PART 3 – FOOD DATABASE PREPROCESSING
# ==========================================================
def load_and_process_food_database():
    """Load, process, and combine food datasets."""
    print("\n" + "=" * 60)
    print("🥗 LOADING AND PROCESSING FOOD DATABASE")
    print("=" * 60)
    
    common_cols = ['meal_name', 'Carbs', 'Proteins', 'Fats', 'Calories']
    dfs_to_concat = []

    # Attempt to load various source files
    files = [
        ('cleaned_nutrition_dataset_per100g.csv', {
            'food': 'meal_name', 
            'Carbohydrates (g per 100g)': 'Carbs', 
            'Protein (g per 100g)': 'Proteins', 
            'Fat (g per 100g)': 'Fats', 
            'Calories (kcal per 100g)': 'Calories'
        }),
        ('parsed_recipe_nutrition.csv', {
            'name': 'meal_name', 'Proteins_g': 'Proteins', 'Fats_g': 'Fats', 'Carbs_g': 'Carbs'
        }),
        ('daily_food_nutrition_dataset.csv', {
            'Food_Item': 'meal_name', 'Protein (g)': 'Proteins', 'Carbohydrates (g)': 'Carbs', 
            'Fat (g)': 'Fats', 'Calories (kcal)': 'Calories'
        })
    ]

    for filename, rename_map in files:
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df = df.rename(columns=rename_map)
                # Keep only valid columns
                valid_cols = [c for c in common_cols if c in df.columns]
                df = df[valid_cols]
                # Add missing cols as 0 if needed, but better to drop
                if set(common_cols).issubset(df.columns):
                    dfs_to_concat.append(df)
                    print(f"✅ Loaded {len(df)} items from '{filename}'")
            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")

    if not dfs_to_concat:
        print("❌ No food data files found. Please ensure CSV files are in the directory.")
        return None
        
    food_df = pd.concat(dfs_to_concat, ignore_index=True)
    food_df = food_df.drop_duplicates(subset=['meal_name'])
    
    # Clean numeric columns
    numeric_cols = ['Calories', 'Proteins', 'Fats', 'Carbs']
    for col in numeric_cols:
        food_df[col] = pd.to_numeric(food_df[col], errors='coerce')
    food_df = food_df.dropna(subset=numeric_cols)

    # Filter raw/unsuitable items
    food_df = food_df[~food_df['meal_name'].str.contains(r'\braw\b', case=False, regex=True, na=False)]
    
    print(f"\n📊 Total processed food items: {len(food_df)}")

    # Identify meal times
    print("\n🍽️ Identifying meal times...")
    food_df['Meal_Time'] = food_df.apply(
        lambda row: identify_meal_time(
            row['meal_name'], row['Calories'], row['Proteins'], row['Carbs'], row['Fats']
        ), axis=1
    )
    
    # Cluster
    food_df = build_food_clusters(food_df)
    
    if food_df is not None:
        food_df.to_csv('improved_food_database.csv', index=False)
        print("\n💾 Saved final database as 'improved_food_database.csv'")
        
    return food_df


# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ATHLETIC DIET RECOMMENDATION SYSTEM - TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load athlete data and train classifier
    # Ensure 'Final_data.csv' exists or provide path
    if os.path.exists('Final_data.csv'):
        df = load_and_explore_data('Final_data.csv')
        if df is not None:
            rf_model = build_meal_plan_classifier(df)
    else:
        print("⚠️ 'Final_data.csv' not found. Skipping supervised model training.")

    # 2. Process food database
    food_df = load_and_process_food_database()

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print("Run 'python app.py' to start the server with these new models.")