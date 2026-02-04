
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
import os

# Configuration
DATA_PATH = 'improved_food_database.csv'
OUTPUT_DIR = 'fyp_evaluation_results'
MODEL_PATH = 'food_kmeans_model.joblib'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def evaluate_kmeans():
    print("🔄 Loading Food Data...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Features used for clustering (as per train.py)
    features = ['Calories', 'Proteins', 'Carbs', 'Fats']
    
    # Clean data (drop NaNs in features)
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features]
    
    # Scale Data
    print("🔄 Scaling Data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Elbow Method Analysis
    print("📊 Generating Elbow Plot...")
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_elbow_plot.png'))
    plt.close()
    print("✅ Elbow Plot saved.")

    # 2. Evaluate Current Model (K=5)
    print("📊 Evaluating Current Model (K=5)...")
    # We can load the saved model or retrain. Retraining ensures we match the X_scaled used here.
    # To be perfectly safe and consistent with the elbow plot, let's instantiate a fresh one with K=5
    # (Checking validation of the choice K=5)
    
    model_k5 = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = model_k5.fit_predict(X_scaled)
    
    # Silhouette Score
    score = silhouette_score(X_scaled, cluster_labels)
    print(f"✅ Silhouette Score for K=5: {score:.4f}")
    
    # Save Score to file
    with open(os.path.join(OUTPUT_DIR, 'kmeans_metrics.txt'), 'w') as f:
        f.write(f"Silhouette Score (K=5): {score:.4f}\n")
    
    # 3. Visualizing Clusters using PCA (2D)
    print("📊 Generating PCA Cluster Plot...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = cluster_labels
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette='viridis', alpha=0.6)
    plt.title(f'Food Clusters Visualization (PCA) - Silhouette: {score:.2f}')
    plt.xlabel('Principal Component 1 (Variance Explained)')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_pca_plot.png'))
    plt.close()
    print("✅ PCA Plot saved.")
    
    print("🎉 K-Means Evaluation Complete!")

if __name__ == "__main__":
    evaluate_kmeans()
