import pandas as pd
import numpy as np

# ===============================
# 1️⃣ LOAD DATA
# ===============================
file_path = "cleaned_nutrition_dataset_per100g.csv"  # change path if needed
df = pd.read_csv(file_path)

print("✅ Original dataset loaded:", df.shape)
print("Columns:", list(df.columns))
print()

# ===============================
# 2️⃣ DEFINE NUTRIENT COLUMNS
# ===============================
nutrient_columns = {
    "Protein (g per 100g)": "Protein_Level",
    "Fat (g per 100g)": "Fat_Level",
    "Carbohydrates (g per 100g)": "Carb_Level",
    "Sugar (g per 100g)": "Sugar_Level",
    "Calories (kcal per 100g)": "Calorie_Level"
}

# ===============================
# 3️⃣ CREATE 5-LEVEL CATEGORIES
# ===============================
labels_5 = ["Very Low", "Low", "Moderate", "High", "Very High"]

for col, new_col in nutrient_columns.items():
    if col in df.columns:
        try:
            df[new_col] = pd.qcut(df[col], q=5, labels=labels_5, duplicates='drop')
        except Exception as e:
            print(f"⚠️ Could not bin {col}: {e}")
    else:
        print(f"⚠️ Missing column: {col}")

print("\n✅ Added detailed nutrient categories!")
print(df.head())

# ===============================
# 4️⃣ DEFINE GOAL MATCH LOGIC
# ===============================
def assign_goal(row):
    protein = row.get("Protein_Level", "")
    carb = row.get("Carb_Level", "")
    fat = row.get("Fat_Level", "")
    calorie = row.get("Calorie_Level", "")

    # HIIT goal → High protein, low carb
    if (protein in ["High", "Very High"]) and (carb in ["Very Low", "Low"]):
        return "HIIT"
    # Endurance → High carb, moderate protein
    elif (carb in ["High", "Very High"]) and (protein in ["Moderate", "High"]):
        return "Endurance"
    # Weight Loss → Low calorie, low fat
    elif (calorie in ["Very Low", "Low"]) and (fat in ["Very Low", "Low"]):
        return "Weight Loss"
    # Weight Gain → High calorie, high protein
    elif (calorie in ["High", "Very High"]) and (protein in ["High", "Very High"]):
        return "Weight Gain"
    # Balanced → Everything moderate
    elif (protein == "Moderate") and (carb == "Moderate") and (fat == "Moderate"):
        return "Balanced"
    else:
        return "General"

df["Goal_Match"] = df.apply(assign_goal, axis=1)

print("\n✅ Goal_Match column added!")
print(df["Goal_Match"].value_counts())

# ===============================
# 5️⃣ SAVE IMPROVED DATASET
# ===============================
output_path = "improved_food_dataset.csv"
df.to_csv(output_path, index=False)
print(f"\n🎉 Improved dataset saved as: {output_path}")
print(f"Total rows: {len(df)}")
print("\nPreview of improved dataset:")
print(df.head(10))
