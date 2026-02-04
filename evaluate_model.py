import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# CONFIGURATION
DATA_PATH = 'Final_data.csv'
MODEL_PATH = 'meal_plan_model.joblib'
OUTPUT_DIR = 'fyp_evaluation_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def evaluate():
    print(f"🚀 Starting Evaluation...")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found.")
        return
    model = joblib.load(MODEL_PATH)
    
    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        print("❌ Data not found.")
        return
    df = pd.read_csv(DATA_PATH)
    
    # Standardize Goal column for the model
    goal_mapping = {
        "Build Muscle": "Build Muscle",
        "Endurance": "Endurance",
        "HIIT": "HIIT",
        "Weight Loss": "Weight Loss",
        "General": "Balanced",
        "Balanced": "Balanced"
    }
    df['Goals'] = df['Goals'].replace(goal_mapping)
    
    # Features (Must match train.py)
    numerical_features = [
        'Age', 'Weight (kg)', 'Height (m)', 'BMI',
        'Fat_Percentage', 'Avg_BPM', 'Resting_BPM', 'Experience_Level',
        'Calories_Burned', 'Workout_Frequency (days/week)', 'protein_per_kg'
    ]
    categorical_features = ['Gender', 'Goals']
    features = numerical_features + categorical_features
    
    X = df[features]
    y_true = df['Meal_Plan']
    
    # 3. Predict
    print("🔮 Running predictions...")
    y_pred = model.predict(X)
    
    # 4. Confusion Matrix
    print("📊 Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(y_true.unique())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Meal Plan Prediction (Balanced)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # 5. Classification Report
    print("📝 Generating Classification Report...")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(OUTPUT_DIR, 'classification_report.csv'))
    
    # 6. Feature Importance
    print("🔝 Generating Feature Importance...")
    try:
        rf_model = model.named_steps['model']
        importances = rf_model.feature_importances_
        preprocessor = model.named_steps['preprocessor']
        cat_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        all_names = np.r_[numerical_features, cat_names]
        
        feat_imp = pd.DataFrame({'Feature': all_names, 'Importance': importances})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(15)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
        plt.close()
    except Exception as e:
        print(f"⚠️ Feature importance failed: {e}")

    print(f"✅ Evaluation Complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate()
