import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def categorize_foods(df):
    """
    Enhanced food categorization system that properly handles combined datasets.
    """
    print("\n🔍 Analyzing nutritional distribution...")
    
    # Calculate key statistics
    stats = {}
    for col in ['Proteins', 'Carbs', 'Fats']:
        q75 = df[col].quantile(0.75)
        q50 = df[col].quantile(0.50)
        q25 = df[col].quantile(0.25)
        stats[col] = {'75th': q75, '50th': q50, '25th': q25}
        print(f"\n{col}:")
        print(f"25th percentile: {q25:.1f}g")
        print(f"Median: {q50:.1f}g")
        print(f"75th percentile: {q75:.1f}g")

    # Define stricter category criteria using dynamic thresholds
    categories = {
        "High Protein High Fat": {
            "criteria": lambda row: (
                row['Proteins'] > stats['Proteins']['75th'] * 0.9 and  # Very high protein
                row['Fats'] > stats['Fats']['75th'] * 0.9 and         # Very high fat
                row['Carbs'] < stats['Carbs']['25th'] * 1.2 and       # Low carbs
                row['Proteins'] + row['Fats'] > row['Carbs'] * 2      # Protein+Fat dominates
            ),
            "goal": "Build Muscle"
        },
        "High Carb": {
            "criteria": lambda row: (
                row['Carbs'] > stats['Carbs']['75th'] and            # Very high carbs
                row['Carbs'] > (row['Proteins'] + row['Fats']) * 2 and # Carbs dominates
                row['Fats'] < stats['Fats']['50th'] * 0.8            # Low fat
            ),
            "goal": "Endurance"
        },
        "High Protein High Carb": {
            "criteria": lambda row: (
                row['Proteins'] > stats['Proteins']['75th'] * 0.8 and # High protein
                row['Carbs'] > stats['Carbs']['75th'] * 0.8 and      # High carbs
                row['Fats'] < stats['Fats']['25th'] * 1.5 and        # Low fat
                row['Proteins'] + row['Carbs'] > row['Fats'] * 3     # Protein+Carbs dominates
            ),
            "goal": "HIIT"
        },
        "Balanced": {
            "criteria": lambda row: (
                all(row[macro] > stats[macro]['50th'] * 0.8 for macro in ['Proteins', 'Carbs', 'Fats']) and
                all(row[macro] < stats[macro]['75th'] * 1.2 for macro in ['Proteins', 'Carbs', 'Fats']) and
                max(row['Proteins'], row['Carbs'], row['Fats']) / 
                min(row['Proteins'], row['Carbs'], row['Fats']) < 3  # No macro dominates too much
            ),
            "goal": "General"
        }
    }

    # First pass: analyze potential matches
    matches_per_category = {cat: 0 for cat in categories.keys()}
    all_matches = []
    
    for idx, row in df.iterrows():
        row_matches = []
        for cat, details in categories.items():
            if details["criteria"](row):
                row_matches.append(cat)
                matches_per_category[cat] += 1
        all_matches.append(row_matches)

    print("\n📊 Initial category matches:")
    for cat, count in matches_per_category.items():
        print(f"{cat:20s}: {count:6d} ({count/len(df)*100:.1f}%)")

    # Second pass: selective assignment
    max_per_category = len(df) * 0.15  # Cap each category at 15% of total foods
    category_counts = {cat: 0 for cat in categories.keys()}
    df['Meal_Plan'] = 'Other'  # Default category

    for idx, row_matches in enumerate(all_matches):
        if row_matches:
            # Score each matching category
            scores = {}
            row = df.iloc[idx]
            
            for cat in row_matches:
                if category_counts[cat] >= max_per_category:
                    continue  # Skip if category is full
                    
                # Base score from how well it matches category criteria
                base_score = 0
                if cat == "High Protein High Fat":
                    base_score = (row['Proteins'] / stats['Proteins']['75th'] + 
                                row['Fats'] / stats['Fats']['75th']) / 2
                elif cat == "High Carb":
                    base_score = row['Carbs'] / stats['Carbs']['75th']
                elif cat == "High Protein High Carb":
                    base_score = (row['Proteins'] / stats['Proteins']['75th'] + 
                                row['Carbs'] / stats['Carbs']['75th']) / 2
                elif cat == "Balanced":
                    ratios = [row[m] / stats[m]['50th'] for m in ['Proteins', 'Carbs', 'Fats']]
                    base_score = 1 - (max(ratios) - min(ratios)) / 2
                
                # Adjust score based on current category distribution
                distribution_factor = 1.0 - (category_counts[cat] / max_per_category)
                scores[cat] = base_score * (1 + distribution_factor)
            
            if scores:  # Only assign if there are valid categories available
                best_cat = max(scores.items(), key=lambda x: x[1])[0]
                df.iloc[idx, df.columns.get_loc('Meal_Plan')] = best_cat
                category_counts[best_cat] += 1

    print("\n📊 Final category distribution:")
    for cat, count in category_counts.items():
        print(f"{cat:20s}: {count:6d} ({count/len(df)*100:.1f}%)")
        print(f"Goal: {categories[cat]['goal']}")

    return df