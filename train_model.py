# ==========================================================
# ATHLETIC DIET RECOMMENDATION SYSTEM - MODEL TRAINING SCRIPT
# ==========================================================
# Hybrid Architecture:
# - Random Forest Regressor (Predicts Protein Ratio) - NEW
# - K-Means (Unsupervised Learning)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
import sys
import ast

# ADDED REGRESSOR IMPORTS AND METRICS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, 
    r2_score,
    silhouette_score,
)
from sklearn.ensemble import RandomForestClassifier # Kept from original, but now unused for training
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support # Kept from original, but now unused for RF evaluation
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.optimize import linear_sum_assignment # Kept from original

# Streamlined output
warnings.filterwarnings('ignore', category=UserWarning)
sys.stdout.reconfigure(line_buffering=True)


# ==========================================================
# PART 0 – LOAD ATHLETE DATA
# ==========================================================
def load_and_explore_data(filepath):
    """Load the athlete dataset and calculate the regression target."""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ '{filepath}' loaded successfully.")
        
        # --- CRITICAL: Add the Ideal_Protein_Ratio target for Regression ---
        protein_ratio_map = {
            'High Protein': 1.8,
            'High Protein High Carb': 1.7,
            'High Protein High Carb High Fat': 2.0,
            'High Protein High Fat': 1.6,
            'High Carb': 1.3,
            'High Carb High Fat': 1.1,
            'High Fat': 1.0,
            'Other': 1.0,
            'Balanced': 1.5,
        }
        # Create the new numerical target column
        df['Ideal_Protein_Ratio'] = df['Meal_Plan'].map(protein_ratio_map).fillna(1.0)
        print("✅ Calculated 'Ideal_Protein_Ratio' for Regression Target.")
        # -------------------------------------------------------------------

        return df
    except FileNotFoundError:
        print(f"❌ Error: '{filepath}' not found.")
        return None


# ==========================================================
# PART 1 – RANDOM FOREST REGRESSOR (Replaces Classifier)
# ==========================================================
def build_protein_regressor(df):
    """
    Train Random Forest Regressor to predict Ideal Protein Ratio (g/kg).
    """
    print("\n" + "=" * 60)
    print("🧠 TRAINING RANDOM FOREST REGRESSOR (Predicting Protein Ratio)")
    print("=" * 60)

    target = 'Ideal_Protein_Ratio'
    
    numerical_features = [
        'Age', 'Weight (kg)', 'Height (m)', 'BMI',
        'Fat_Percentage', 'Avg_BPM', 'Resting_BPM', 'Experience_Level'
    ]
    categorical_features = ['Gender', 'Goals']
    features = numerical_features + categorical_features

    # --- Validation ---
    if target not in df.columns:
        print(f"❌ ERROR: The target column '{target}' is missing. Cannot train Regressor.")
        return None

    df_clean = df.dropna(subset=[target] + features)
    if df_clean.empty:
        print("❌ No valid data for training after removing missing rows.")
        return None

    X = df_clean[features]
    y = df_clean[target] 

    print(f"\n📊 Samples: {len(df_clean)}")
    print(f"Mean Ideal Protein Ratio: {y.mean():.2f} g/kg")

    # --- Preprocessing ---
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)])

    # --- Random Forest Regressor (MODEL CHANGE) ---
    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    # Note: SMOTE is not generally used for Regression, so we use a standard Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Train Model ---
    print("\n🔄 Training Random Forest Regressor...")
    model_pipeline.fit(X_train, y_train)
    print("✅ Model training complete!")

    # --- Evaluation (Using Regression Metrics) ---
    y_pred = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 MODEL PERFORMANCE METRICS (Regression):")
    print("=" * 60)
    print(f"✓ Mean Squared Error (MSE): {mse:.4f} (Lower is better)")
    print(f"✓ R-squared (R2 Score):     {r2:.4f} (Closer to 1.0 is better)")
    print("=" * 60)

    # --- Save Model ---
    joblib.dump(model_pipeline, 'protein_regressor_model.joblib')
    joblib.dump(X.columns.tolist(), 'regressor_features.joblib')
    
    # Save metrics for the app sidebar
    metrics = {'mse': mse, 'r2': r2}
    joblib.dump(metrics, 'regressor_metrics.joblib')

    return model_pipeline


# ==========================================================
# PART 2 – K-MEANS CLUSTERING (Existing Logic)
# ==========================================================
def perform_elbow_method(scaled_data):
    """Generates the Elbow Method plot."""
    print("\n📊 Performing Elbow Method analysis...")
    wcss = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.savefig('elbow_method_plot.png')
    print("✅ Elbow plot saved as 'elbow_method_plot.png'")

def map_cluster_to_meal_plan(food_df, n_clusters, target_labels=None):
    """Map K-Means clusters to meal plans using Tuned Ideal Profiles.

    If `target_labels` is provided and its length equals `n_clusters`, this
    function will compute a one-to-one assignment between clusters and labels
    using the Hungarian algorithm to avoid mapping multiple clusters to the
    same meal-plan label.
    """
    print("\n🧠 Mapping clusters to meal plans...")
    cluster_stats = (
        food_df.groupby('Food_Cluster')[['Proteins', 'Carbs', 'Fats']]
        .mean()
        .reset_index()
    )
    totals = cluster_stats[['Proteins', 'Carbs', 'Fats']].sum(axis=1).replace(0, 1e-6)
    cluster_stats['p_ratio'] = cluster_stats['Proteins'] / totals
    cluster_stats['c_ratio'] = cluster_stats['Carbs'] / totals
    cluster_stats['f_ratio'] = cluster_stats['Fats'] / totals

    # Default ideal profiles (can be overridden by target_labels)
    ideal_profiles = {
        "Weight Loss":  {"p": 0.60, "c": 0.10, "f": 0.30},
        "Build Muscle": {"p": 0.40, "c": 0.30, "f": 0.30},
        "Endurance":    {"p": 0.15, "c": 0.65, "f": 0.20},
        "HIIT":         {"p": 0.30, "c": 0.45, "f": 0.25},
        "Balanced":     {"p": 0.15, "c": 0.25, "f": 0.60}
    }

    # If user provided a target label ordering, use it to build the profiles list
    if target_labels is not None:
        labels = list(target_labels)
    else:
        labels = list(ideal_profiles.keys())

    # Build profile vectors for the chosen labels
    profiles = [ideal_profiles.get(lbl, {'p': 0.33, 'c': 0.33, 'f': 0.34}) for lbl in labels]

    mapping = {}

    # If number of clusters matches number of labels, compute one-to-one assignment
    unique_clusters = sorted(cluster_stats['Food_Cluster'].unique())
    n_clusters_found = len(unique_clusters)
    n_labels = len(labels)

    if n_clusters_found == n_labels:
        # Create cost matrix (clusters x labels) using L1 distance between ratios
        cost = []
        for _, row in cluster_stats.sort_values('Food_Cluster').iterrows():
            vec = [row['p_ratio'], row['c_ratio'], row['f_ratio']]
            row_cost = [abs(vec[0] - prof['p']) + abs(vec[1] - prof['c']) + abs(vec[2] - prof['f']) for prof in profiles]
            cost.append(row_cost)

        import numpy as _np
        from scipy.optimize import linear_sum_assignment
        cost = _np.array(cost)
        row_idx, col_idx = linear_sum_assignment(cost)

        for r, c in zip(row_idx, col_idx):
            cluster_id = int(sorted(cluster_stats['Food_Cluster'])[r])
            mapping[cluster_id] = labels[c]
    else:
        # Fallback: map each cluster to the nearest label (may cause many-to-one)
        for i, row in cluster_stats.iterrows():
            cluster_id = int(row['Food_Cluster'])
            p, c, f = row['p_ratio'], row['c_ratio'], row['f_ratio']
            best_label = labels[0]
            min_dist = float('inf')
            for lbl, prof in zip(labels, profiles):
                dist = (abs(p - prof['p']) + abs(c - prof['c']) + abs(f - prof['f']))
                if dist < min_dist:
                    min_dist = dist
                    best_label = lbl
            mapping[cluster_id] = best_label

    return mapping

def build_food_clusters(food_df):
    """Apply K-Means clustering to the food database."""
    print("\n" + "=" * 60)
    print("🍽️ CLUSTERING FOODS WITH K-MEANS")
    print("=" * 60)

    base_features = ['Proteins', 'Carbs', 'Fats', 'Calories']
    food_df = food_df.dropna(subset=base_features).copy()

    # Filtering logic 
    food_df = food_df[
        (food_df['Proteins'] > 0) | (food_df['Carbs'] > 0) | (food_df['Fats'] > 0)
    ]
    
    totals = food_df[['Proteins', 'Carbs', 'Fats']].sum(axis=1).replace(0, 1e-6)
    food_df['p_ratio'] = food_df['Proteins'] / totals
    food_df['c_ratio'] = food_df['Carbs'] / totals
    food_df['f_ratio'] = food_df['Fats'] / totals
    
    cluster_features = ['p_ratio', 'c_ratio', 'f_ratio']
    
    if len(food_df) > 20000:
        food_df_sample = food_df.sample(n=20000, random_state=42)
    else:
        food_df_sample = food_df.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(food_df_sample[cluster_features])
    perform_elbow_method(X_scaled)

    # Define the target meal-plan labels we want represented one-to-one by clusters
    target_labels = ["Weight Loss", "Build Muscle", "Endurance", "HIIT", "Balanced"]
    k = len(target_labels)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    food_df_sample['Food_Cluster'] = kmeans.fit_predict(X_scaled)
    X_all_scaled = scaler.transform(food_df[cluster_features])
    food_df['Food_Cluster'] = kmeans.predict(X_all_scaled)

    silhouette_avg = silhouette_score(X_scaled, food_df_sample['Food_Cluster'])
    mapping = map_cluster_to_meal_plan(food_df, k, target_labels=target_labels)

    joblib.dump(kmeans, 'food_kmeans_model.joblib')
    joblib.dump(scaler, 'food_scaler.joblib')
    joblib.dump(mapping, 'cluster_mealplan_mapping.joblib')
    
    clustering_metrics = { 'optimal_k': k, 'silhouette_score': silhouette_avg }
    joblib.dump(clustering_metrics, 'clustering_metrics.joblib')

    food_df['Meal_Plan'] = food_df['Food_Cluster'].map(mapping).fillna('Balanced')
    return food_df

# ==========================================================
# PART 3 – FOOD DATABASE PREPROCESSING 
# ==========================================================
def load_and_process_food_database():
    """
    Load and process food database (assumes two files are merged).
    Uses a robust method to guarantee columns exist before concatenation.
    """
    print("\n" + "=" * 60)
    print("🥗 LOADING FOOD DATABASE (Data Fusion)")
    print("=" * 60)
    
    required_cols = ['meal_name', 'Proteins', 'Carbs', 'Fats', 'Calories']
    dataframes_to_merge = []
    
    rename_map = {
        'food': 'meal_name', 'recipe_name': 'meal_name',
        'Carbohydrates (g per 100g)': 'Carbs', 'Protein (g per 100g)': 'Proteins',
        'Fat (g per 100g)': 'Fats', 'Calories (kcal per 100g)': 'Calories'
    }

    def safe_load_food_file(names, name_type):
        for name in names:
            try:
                df = pd.read_csv(name)
                print(f"✅ Loaded {name_type} data from '{name}'.")
                
                df = df.rename(columns=rename_map).copy()
                
                # --- Ensure all required columns exist, fill with NaN ---
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = np.nan 
                # -------------------------------------------------------------

                # Select only the required columns and return
                return df[required_cols]
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"❌ Error processing {name}: {e}")
                continue
        print(f"❌ Warning: Could not load {name_type} data from any expected path.")
        # Return empty DF with guaranteed headers to prevent later KeyError
        return pd.DataFrame(columns=required_cols) 

    # 1. Load Ingredient Data
    dataframes_to_merge.append(safe_load_food_file(
        ['Nutritional_Breakdown_of_Foods.csv', 'cleaned_nutrition_dataset_per100g.csv'], 'ingredient'
    ))

    # 2. Load Recipe Data
    dataframes_to_merge.append(safe_load_food_file(
        ['Parsed_Recipe_Nutrition.csv', 'parsed_recipe_nutrition.csv'], 'recipe'
    ))

    # Combine into Unified Food Database
    non_empty_dfs = [df for df in dataframes_to_merge if not df.empty]

    if not non_empty_dfs:
        print("❌ ERROR: No food data files were found or loaded. Returning empty database.")
        return pd.DataFrame()

    # Use join='outer' to combine robustly
    food_df = pd.concat(non_empty_dfs, ignore_index=True, join='outer')
    
    # Final cleanup: drop rows where core nutrient data is missing
    food_df = food_df.dropna(subset=['meal_name', 'Proteins', 'Carbs', 'Fats', 'Calories'], how='any')
    food_df = food_df.drop_duplicates(subset=['meal_name'])
    
    print(f"✅ Unified Food Database created. Total items: {len(food_df)}")
    return food_df


# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ATHLETIC DIET RECOMMENDATION SYSTEM - TRAINING PIPELINE")
    print("=" * 60)

    # 1. Train Random Forest Regressor on athlete profiles
    df = load_and_explore_data('Final_data.csv')
    if df is not None:
        # CORRECT CALL: Use the new Regressor function
        rf_model = build_protein_regressor(df) 
    else:
        rf_model = None
        print("⚠️ Skipped Regressor training (no data).")

    # 2. Process food database and apply K-Means clustering
    food_df = load_and_process_food_database()
    if food_df is not None and not food_df.empty:
        food_df = build_food_clusters(food_df)
        # Save final database 
        food_df.to_csv('improved_food_database.csv', index=False)
        print("\n💾 Saved as 'improved_food_database.csv'")

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
