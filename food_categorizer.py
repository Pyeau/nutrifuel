import pandas as pd
import numpy as np

def categorize_foods(df):
    """
    Categorize foods into meal plan categories based on their nutritional content
    while maintaining a balanced distribution.
    """
    # Calculate percentiles for thresholds
    p75_protein = df['Proteins'].quantile(0.75)
    p75_carb = df['Carbs'].quantile(0.75)
    p75_fat = df['Fats'].quantile(0.75)
    
    print(f"\nNutritional Thresholds (75th percentile):")
    print(f"Protein: {p75_protein:.1f}g")
    print(f"Carbs: {p75_carb:.1f}g")
    print(f"Fats: {p75_fat:.1f}g")
    
    # Define category criteria
    categories = {
        "High Protein High Fat": {
            "criteria": lambda p, c, f: p > p75_protein * 0.7 and f > p75_fat * 0.7,
            "goal": "Build Muscle"
        },
        "High Carb": {
            "criteria": lambda p, c, f: c > p75_carb * 0.7 and c/(p+f+0.1) > 1.2,
            "goal": "Endurance"
        },
        "High Protein High Carb": {
            "criteria": lambda p, c, f: p > p75_protein * 0.6 and c > p75_carb * 0.6,
            "goal": "HIIT"
        },
        "Balanced": {
            "criteria": lambda p, c, f: all(x > 0.3 * y for x, y in [
                (p, p75_protein), (c, p75_carb), (f, p75_fat)
            ]),
            "goal": "General"
        }
    }
    
    # Initialize results
    df['Meal_Plan'] = "Uncategorized"
    category_counts = {cat: 0 for cat in categories.keys()}
    
    # First pass: Count potential matches
    potential_matches = {cat: 0 for cat in categories.keys()}
    for idx, row in df.iterrows():
        for cat, details in categories.items():
            if details["criteria"](row['Proteins'], row['Carbs'], row['Fats']):
                potential_matches[cat] += 1
    
    print("\nPotential matches per category:")
    for cat, count in potential_matches.items():
        print(f"{cat:20s}: {count:6d} foods")
    
    # Second pass: Assign categories with distribution balancing
    target_per_category = len(df) / len(categories)
    
    for idx, row in df.iterrows():
        matches = []
        scores = {}
        
        # Check which categories this food could belong to
        for cat, details in categories.items():
            if details["criteria"](row['Proteins'], row['Carbs'], row['Fats']):
                matches.append(cat)
                
                # Base score: how well it fits the category
                base_score = 1.0
                
                # Distribution factor: prefer under-represented categories
                distribution_factor = 1.0 - (category_counts[cat] / target_per_category)
                if distribution_factor < 0:
                    distribution_factor = 0
                    
                scores[cat] = base_score * (1 + distribution_factor)
        
        # Assign to best matching category
        if matches:
            best_cat = max(scores.items(), key=lambda x: x[1])[0]
            df.at[idx, 'Meal_Plan'] = best_cat
            category_counts[best_cat] += 1
        else:
            # If no matches, assign to least represented category that it's closest to
            min_cat = min(category_counts.items(), key=lambda x: x[1])[0]
            df.at[idx, 'Meal_Plan'] = min_cat
            category_counts[min_cat] += 1
    
    print("\nFinal distribution:")
    for cat, count in category_counts.items():
        print(f"{cat:20s}: {count:6d} foods ({count/len(df)*100:.1f}%)")
    
    return df