import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# CONFIGURATION
DATA_PATH = 'Final_data.csv'
OUTPUT_DIR = 'fyp_evaluation_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_comparison():
    print("🚀 Starting Split Comparison...")
    if not os.path.exists(DATA_PATH):
        print("❌ Data not found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Standardize Goal column
    goal_mapping = {
        "Build Muscle": "Build Muscle", "Endurance": "Endurance", "HIIT": "HIIT",
        "Weight Loss": "Weight Loss", "General": "Balanced", "Balanced": "Balanced"
    }
    df['Goals'] = df['Goals'].replace(goal_mapping)
    
    target = 'Meal_Plan'
    numerical_features = [
        'Age', 'Weight (kg)', 'Height (m)', 'BMI',
        'Fat_Percentage', 'Avg_BPM', 'Resting_BPM', 'Experience_Level',
        'Calories_Burned', 'Workout_Frequency (days/week)', 'protein_per_kg'
    ]
    categorical_features = ['Gender', 'Goals']
    features = numerical_features + categorical_features

    X = df[features]
    y = df[target]

    # Preprocessing
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

    splits = [0.2, 0.3, 0.4] 
    results = []

    for test_size in splits:
        split_name = f"{int((1-test_size)*100)}/{int(test_size*100)}"
        print(f"🔄 Testing Split: {split_name}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Split Ratio': split_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'split_comparison_results.csv'), index=False)
    
    plt.figure(figsize=(10, 6))
    melted_df = res_df.melt(id_vars="Split Ratio", var_name="Metric", value_name="Score")
    sns.barplot(data=melted_df, x="Split Ratio", y="Score", hue="Metric", palette="viridis")
    plt.ylim(0.9, 1.01) # High accuracy expected
    plt.title("Model Performance vs Data Split Ratio (Balanced)")
    plt.savefig(os.path.join(OUTPUT_DIR, 'split_comparison.png'))
    plt.close()
    print("✅ Split Comparison Complete.")

if __name__ == "__main__":
    run_comparison()
