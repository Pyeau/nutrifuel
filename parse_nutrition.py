import pandas as pd
import numpy as np
import ast  # For safely parsing the string list
import sys

def parse_nutrition_string(nutrition_str):
    """
    Safely parse the nutrition string from RAW_recipes.csv.
    Expected format: [calories, total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), total carbohydrate (PDV)]
    """
    try:
        # Use ast.literal_eval for safe parsing
        nutr_list = ast.literal_eval(nutrition_str)
        if isinstance(nutr_list, list) and len(nutr_list) == 7:
            # Return the values we care about
            return pd.Series({
                'Calories': nutr_list[0],
                'Fats_PDV': nutr_list[1],
                'Proteins_PDV': nutr_list[4],
                'Carbs_PDV': nutr_list[6]  # <-- ADDED
            })
    except (ValueError, SyntaxError, TypeError):
        pass
    # Return NaNs if parsing fails
    return pd.Series({
        'Calories': np.nan, 
        'Fats_PDV': np.nan,
        'Proteins_PDV': np.nan,
        'Carbs_PDV': np.nan  # <-- ADDED
    })

def main():
    """
    Main function to load, process, and save the new dataset.
    """
    
    # --- Configuration ---
    # This is the filename from your upload
    input_filename = 'RAW_recipes.csv' 
    output_filename = 'parsed_recipe_nutrition.csv'
    
    # Using FDA Daily Value (DV) references to convert PDV to Grams
    DV_PROTEIN_G = 50
    DV_FAT_G = 78
    DV_CARBS_G = 275  # <-- ADDED
    
    print(f"🚀 Starting nutrition parsing for '{input_filename}'...")

    # --- Load Data ---
    try:
        df = pd.read_csv(input_filename)
        print(f"✅ Loaded {len(df)} recipes.")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: '{input_filename}'")
        print("Please make sure the file is in the same directory as this script.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Could not load file. {e}")
        sys.exit(1)

    # --- Parse Nutrition Column ---
    print("🔄 Parsing 'nutrition' column... (This may take a moment)")
    
    # Apply the parsing function to the 'nutrition' column
    parsed_nutrition = df['nutrition'].apply(parse_nutrition_string)
    
    # Combine the parsed data with the original dataframe
    df = pd.concat([df, parsed_nutrition], axis=1)

    # --- Convert PDV to Grams ---
    print("⚖️ Converting PDV to grams...")
    
    df['Proteins_g'] = (df['Proteins_PDV'] / 100) * DV_PROTEIN_G
    df['Fats_g'] = (df['Fats_PDV'] / 100) * DV_FAT_G
    df['Carbs_g'] = (df['Carbs_PDV'] / 100) * DV_CARBS_G  # <-- ADDED
    
    # Round to one decimal place for cleanliness
    df['Proteins_g'] = df['Proteins_g'].round(1)
    df['Fats_g'] = df['Fats_g'].round(1)
    df['Calories'] = df['Calories'].round(1)
    df['Carbs_g'] = df['Carbs_g'].round(1)  # <-- ADDED

    # --- Create Final Dataset ---
    # Select only the columns you want
    final_columns = ['name', 'Calories', 'Proteins_g', 'Fats_g', 'Carbs_g']  # <-- EDITED
    final_df = df[final_columns].copy()
    
    # Optional: Drop any rows that failed parsing
    final_df = final_df.dropna().reset_index(drop=True)

    # --- NEW: Add Low/Medium/High Bins ---
    print("📊 Creating 'Low/Medium/High' bins...")
    labels = ["Very Low", "Low", "Moderate", "High", "Very High"]
    
    # Define columns to bin and their new names
    cols_to_bin = {
        'Calories': 'Calories_Level',
        'Proteins_g': 'Proteins_Level',
        'Fats_g': 'Fats_Level',
        'Carbs_g': 'Carbs_Level'  # <-- ADDED
    }

    for col, new_col_name in cols_to_bin.items():
        if col in final_df.columns:
            try:
                # Try quantile-based binning first (creates bins with equal numbers of items)
                final_df[new_col_name] = pd.qcut(final_df[col], q=5, labels=labels, duplicates='drop')
            except ValueError:
                try:
                    # Fallback to equal-width binning if qcut fails (e.g., too many identical values)
                    final_df[new_col_name] = pd.cut(final_df[col], bins=5, labels=labels, include_lowest=True)
                except Exception as e:
                    # Handle other potential errors
                    print(f"⚠️  Could not create bins for {col}: {e}")
                    final_df[new_col_name] = pd.NA
        
    # --- Save New CSV File ---
    try:
        final_df.to_csv(output_filename, index=False)
        print("\n" + "="*50)
        print(f"🎉 SUCCESS! 🎉")
        print(f"💾 New dataset saved as: '{output_filename}'")
        print(f"Total processed recipes: {len(final_df)}")
        print("\n--- Sample Data ---")
        print(final_df.head())
        print("="*50)
        
    except Exception as e:
        print(f"❌ ERROR: Could not save new file. {e}")

# Run the script
if __name__ == "__main__":
    main()