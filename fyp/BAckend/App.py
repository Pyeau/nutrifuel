from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes and origins to allow React frontend to connect
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5000",
            os.getenv("FRONTEND_URL", "http://localhost:3000"),
            "https://nutrifuel-frontend.onrender.com"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# --- LOAD DATA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Get parent directories - from App.py location, go up 2 levels to devoplement folder
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

print(f"📂 BASE_DIR: {BASE_DIR}")
print(f"📂 DATA_DIR: {DATA_DIR}")

def load_resources():
    try:
        # Try loading from the data directory
        food_db_path = os.path.join(DATA_DIR, 'improved_food_database.csv')
        model_path = os.path.join(DATA_DIR, 'meal_plan_model.joblib')
        
        # Fallback: check current directory if not found
        if not os.path.exists(food_db_path):
            food_db_path = os.path.join(DATA_DIR, 'fyp', 'BAckend', 'improved_food_database.csv')
        if not os.path.exists(model_path):
            model_path = os.path.join(DATA_DIR, 'fyp', 'BAckend', 'meal_plan_model.joblib')
        
        # Additional fallback
        if not os.path.exists(food_db_path):
            food_db_path = 'improved_food_database.csv'
        if not os.path.exists(model_path):
            model_path = 'meal_plan_model.joblib'

        print(f"📂 Loading food database from: {food_db_path}")
        print(f"📂 Loading model from: {model_path}")
        
        if not os.path.exists(food_db_path):
            print(f"⚠️  Food database not found at {food_db_path}")
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}")
            
        food_db = pd.read_csv(food_db_path)
        classifier_model = joblib.load(model_path)
        print("✅ Database & Models Loaded Successfully!")
        return food_db, classifier_model
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        import traceback
        traceback.print_exc()
        return None, None

food_db, classifier_model = load_resources()

# --- HELPER FUNCTIONS (Integrated from appp.py) ---

def calculate_bmr(age, weight, height, gender):
    """Harris-Benedict Equation"""
    if gender == "Male":
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

def calculate_tdee(bmr, activity_level):
    """Total Daily Energy Expenditure"""
    activity_multipliers = {
        "Light Training (1–3 days/week)": 1.375,
        "Moderate Training (3–5 days/week)": 1.55,
        "High Training (6–7 days/week)": 1.725
    }
    return bmr * activity_multipliers.get(activity_level, 1.55)

def map_goal_to_model_format(goal):
    """Map user UI goal to the specific keys used by the model/system"""
    mapping = {
        "Build Muscle": "Build Muscle", "Endurance": "Endurance", "HIIT": "HIIT", 
        "Weight Loss": "Weight Loss", "General / Balanced": "Balanced",
        "Balanced": "Balanced"
    }
    return mapping.get(goal, "Balanced")

def predict_goal_for_user(age, bmi, training_intensity):
    """
    Safe fallback prediction if the ML model fails.
    Uses balanced logic to suggest a starting point.
    """
    if bmi > 28: return "Weight Loss"
    if training_intensity == "High Training (6–7 days/week)": return "Endurance"
    return "Balanced"

# --- CORE RECOMMENDATION ENGINE (Ported from appp.py) ---

def get_recommendations_by_cluster(food_db, predicted_plan, exclude_names=None, calorie_target=2000):
    """
    Selects 3 meals. If calorie target is not met, finds a 'Snack' 
    and pairs it with one of the meals to close the gap.
    """
    if food_db is None or food_db.empty: return []

    if exclude_names is None: exclude_names = []
    
    # 1. Targets Setup
    meal_calories = {"Breakfast": calorie_target * 0.30, "Lunch": calorie_target * 0.40, "Dinner": calorie_target * 0.30}
    
    # Nutrition Requirements
    nutrition_requirements = {
        "Weight Loss":  {'Proteins': {'min': 25}, 'Carbs': {'min': 0}},
        "Build Muscle": {'Proteins': {'min': 30}, 'Carbs': {'min': 10}},
        "Endurance":    {'Proteins': {'min': 5}, 'Carbs': {'min': 40}},
        "HIIT":         {'Proteins': {'min': 15}, 'Carbs': {'min': 30}},
        "Balanced":     {'Proteins': {'min': 10}, 'Carbs': {'min': 20}}
    }
    plan_reqs = nutrition_requirements.get(predicted_plan, nutrition_requirements["Balanced"])

    # 2. Filter Database
    cluster_filtered = food_db[food_db['Meal_Plan'] == predicted_plan].copy()
    if cluster_filtered.empty: cluster_filtered = food_db.copy()
    
    cluster_filtered['Calories'] = pd.to_numeric(cluster_filtered['Calories'], errors='coerce')
    
    # Exclusions
    base_pool = cluster_filtered[~cluster_filtered['meal_name'].isin(exclude_names)]
    if base_pool.empty: base_pool = cluster_filtered

    # Helper: Find best food candidates
    def get_candidates(pool, target_cal, meal_time_filter=None):
        # Base filter by macro requirements
        filtered = pool[
            (pool['Proteins'] >= plan_reqs['Proteins']['min']) &
            (pool['Carbs'] >= plan_reqs['Carbs']['min'])
        ]
        if filtered.empty:
            filtered = pool.copy() # Fallback

        # For Build Muscle, prefer foods where protein is dominant over carbs/fats
        if predicted_plan == 'Build Muscle':
            prot_dominant = filtered[(filtered['Proteins'] >= filtered['Carbs']) & (filtered['Proteins'] >= filtered['Fats'])]
            if not prot_dominant.empty:
                filtered = prot_dominant

        # Filter by Meal Time if specified
        if meal_time_filter:
            time_match = filtered[filtered['Meal_Time'] == meal_time_filter]
            if time_match.empty:
                time_match = filtered[filtered['Meal_Time'] == 'Snack']
            if not time_match.empty:
                filtered = time_match

        filtered = filtered.dropna(subset=['Calories'])
        # Compute closeness and prefer high-protein options for Build Muscle
        filtered = filtered.assign(diff=(filtered['Calories'] - target_cal).abs())

        if predicted_plan == 'Build Muscle':
            # Sort by proteins desc, then by calorie closeness
            filtered = filtered.sort_values(by=['Proteins', 'diff'], ascending=[False, True])
            candidates = filtered.head(50).to_dict('records')
        else:
            # Default: sort by calorie closeness
            candidates = filtered.nsmallest(50, 'diff').to_dict('records')

        random.shuffle(candidates)
        return candidates

    # 3. Select Main 3 Meals
    meals = []
    selected_names = set(exclude_names)
    
    for m_type in ["Breakfast", "Lunch", "Dinner"]:
        candidates = get_candidates(base_pool, meal_calories[m_type], meal_time_filter=m_type)
        
        # Pick one that hasn't been used
        selection = next((c for c in candidates if c['meal_name'] not in selected_names), None)
        
        # If all used, pick random
        if not selection and candidates: selection = random.choice(candidates)
        
        if selection:
            selected_names.add(selection['meal_name'])
            meals.append({
                "Meal": m_type,
                "Food": selection['meal_name'],
                "Source_Meal_Time": selection.get('Meal_Time', 'General'),
                "Target_Cal_Num": meal_calories[m_type],
                "Cal_Num": selection['Calories'],
                "Prot_Num": selection['Proteins'],
                "Carb_Num": selection['Carbs'],
                "Fat_Num": selection['Fats'],
                "Cluster_Name": predicted_plan
            })

    # 4. GAP FILLING: Check Total vs Target
    current_total = sum(m['Cal_Num'] for m in meals)
    deficit = calorie_target - current_total
    
    # Tolerance: If we are under by more than 150 calories, add a snack
    if deficit > 150:
        # Find a snack that fits the deficit
        snack_pool = base_pool[base_pool['Meal_Time'] == 'Snack']
        if not snack_pool.empty:
            snack_pool = snack_pool.assign(diff=(snack_pool['Calories'] - deficit).abs())
            # Find best snack matches
            snack_candidates = snack_pool.nsmallest(20, 'diff').to_dict('records')
            
            # Find one not already used
            best_snack = next((s for s in snack_candidates if s['meal_name'] not in selected_names), None)
            
            if best_snack:
                # Find which meal has the largest individual deficit
                meals.sort(key=lambda x: x['Target_Cal_Num'] - x['Cal_Num'], reverse=True)
                target_meal = meals[0] # This meal needs calories the most
                
                # Update the meal with the snack
                target_meal['Food'] += f" + {best_snack['meal_name']}"
                target_meal['Source_Meal_Time'] += " + Snack"
                target_meal['Cal_Num'] += best_snack['Calories']
                target_meal['Prot_Num'] += best_snack['Proteins']
                target_meal['Carb_Num'] += best_snack['Carbs']
                target_meal['Fat_Num'] += best_snack['Fats']
                
                # Re-sort meals back to Breakfast-Lunch-Dinner order
                order = {"Breakfast": 0, "Lunch": 1, "Dinner": 2}
                meals.sort(key=lambda x: order.get(x['Meal'], 3))

    return meals

# --- API ENDPOINTS ---

@app.route('/', methods=['GET'])
def index():
    """Root endpoint - health check"""
    return jsonify({
        "status": "ok",
        "message": "NutriFuel API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/api/predict",
            "replace-food": "/api/replace-food"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    db_status = "connected" if food_db is not None else "disconnected"
    model_status = "loaded" if classifier_model is not None else "not_loaded"
    return jsonify({
        "status": "ok",
        "message": "Server is running",
        "database": db_status,
        "model": model_status
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if food_db is None or classifier_model is None:
        return jsonify({"error": "Server unavailable. Data/Models not loaded."}), 503

    try:
        data = request.json
        print("📥 Received Data:", data)
        
        # 1. Parsing Input
        age = int(data.get('age', 25))
        weight = float(data.get('weight', 70))
        height = float(data.get('height', 170)) # CM
        gender = data.get('gender', 'Male')
        activity = data.get('activity', 'Moderate Training (3–5 days/week)')
        fat_percentage = float(data.get('fat', 18.0))
        bpm = float(data.get('bpm', 60))
        goal_input = data.get('goal', 'Not Sure - Let AI Suggest')
        calorie_strategy = data.get('calorieStrategy', 'Match TDEE')
        manual_calorie = data.get('manualCalories', 2000)

        # 2. Calculations
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        bmr = calculate_bmr(age, weight, height, gender)
        tdee = calculate_tdee(bmr, activity)
        
        # Map activity to levels (Scaled 1-7 to match Training Data)
        exp_level_map = {
            "Light Training (1–3 days/week)": 2, 
            "Moderate Training (3–5 days/week)": 4, 
            "High Training (6–7 days/week)": 6
        }
        exp_level = exp_level_map.get(activity, 4)

        # 3. Determine Goal (AI vs Manual)
        final_goal = "Balanced"
        confidence_score = 0
        is_ai_predicted = False
        
        if goal_input == "Not Sure - Let AI Suggest":
            # AI Heuristic Check first
            heuristic_goal = predict_goal_for_user(age, bmi, activity)
            
            try:
                # Prepare Model Input
                # Note: Model expects specific columns in specific order
                input_df = pd.DataFrame([{
                    'Age': age, 
                    'Weight (kg)': weight, 
                    'Height (m)': height_m, 
                    'BMI': bmi,
                    'Fat_Percentage': fat_percentage,
                    'Avg_BPM': bpm, # Approximating Avg as Resting for now or user input
                    'Resting_BPM': 65, # Default if not provided
                    'Experience_Level': exp_level, 
                    'Gender': gender, 
                    'Goals': 'Unknown', # Use 'Unknown' to avoid biasing the model with a placeholder
                    'Calories_Burned': tdee, 
                    'Workout_Frequency (days/week)': exp_level, 
                    'protein_per_kg': 1 # Set to neutral/low value to avoid forcing 'Build Muscle' prediction
                }])
                
                # Predict
                prediction = classifier_model.predict(input_df)[0]
                proba = classifier_model.predict_proba(input_df)[0].max()
                
                final_goal = prediction
                confidence_score = proba
                is_ai_predicted = True
            except Exception as e:
                print(f"⚠️ Model prediction failed: {e}. Using heuristic.")
                final_goal = heuristic_goal
                confidence_score = 0.5
        else:
            # User Selected Goal
            final_goal = map_goal_to_model_format(goal_input)
            confidence_score = 1.0

        # 4. Determine Calorie Target
        target_calories = tdee
        if calorie_strategy == "TDEE - 200 kcal":
            target_calories = max(1200, tdee - 200)
        elif calorie_strategy == "TDEE - 300 kcal":
            target_calories = max(1200, tdee - 300)
        elif calorie_strategy == "Manual target":
            target_calories = float(manual_calorie)

        # 5. Generate Recommendations
        meals_raw = get_recommendations_by_cluster(
            food_db, 
            final_goal, 
            exclude_names=[], 
            calorie_target=target_calories
        )

        # MAP KEYS TO FRONTEND EXPECTATIONS
        meals_formatted = []
        for m in meals_raw:
            meals_formatted.append({
                "type": m['Meal'],          # Frontend expects 'type'
                "name": m['Food'],          # Frontend expects 'name'
                "cal": int(m['Cal_Num']),   # Frontend expects 'cal'
                "prot": float(m['Prot_Num']), # Frontend expects 'prot'
                "carbs": float(m['Carb_Num']),# Frontend expects 'carbs'
                "fats": float(m['Fat_Num']),  # Frontend expects 'fats'
                "cluster": m['Cluster_Name']
            })
        
        # 6. Get Full Food List for this Cluster (for "Swap/Add" feature)
        cluster_foods_df = food_db[food_db['Meal_Plan'] == final_goal].fillna(0)
        
        # Optimize: Convert to list of dicts directly
        cluster_list = []
        # We'll take a sample or all? Let's take up to 200 random items to avoid payload too large
        # If user wants *all*, we might need pagination, but let's give a generous subset for now or all if < 500
        if len(cluster_foods_df) > 500:
             cluster_foods_df = cluster_foods_df.sample(500)
             
        for _, row in cluster_foods_df.iterrows():
            cluster_list.append({
                "name": row['meal_name'],
                "type": row.get('Meal_Time', 'Snack'), # Default to Snack if missing
                "cal": int(row['Calories']),
                "prot": float(row['Proteins']),
                "carbs": float(row['Carbs']),
                "fats": float(row['Fats'])
            })

        # 7. Response
        response = {
            "bmi": round(bmi, 2),
            "bmr": round(bmr, 2),
            "tdee": round(tdee, 2),
            "targetCals": round(target_calories, 2),
            "finalGoal": final_goal,
            "confidence": round(confidence_score, 2),
            "isAiPredicted": is_ai_predicted,
            "meals": meals_formatted,
            "clusterFoods": cluster_list,
            "status": "success",
            "db_status": "connected" if food_db is not None else "disconnected"
        }
        print(f"✅ Generated plan with {len(cluster_list)} extra exchange options.")
        return jsonify(response)

    except Exception as e:
        print(f"❌ Server Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- FOOD REPLACEMENT ENDPOINT ---
@app.route('/api/replace-food', methods=['POST'])
def replace_food():
    """
    Replace a food in the meal plan with an alternative from the exchange list.
    Expected request body:
    {
        "mealIndex": 0,
        "newFoodName": "Oatmeal",
        "currentMeals": [...],  # Current meals array
        "goal": "Build Muscle"
    }
    """
    try:
        data = request.json
        meal_index = data.get('mealIndex')
        new_food_name = data.get('newFoodName')
        current_meals = data.get('currentMeals', [])
        goal = data.get('goal', 'Balanced')
        
        if meal_index is None or not new_food_name or food_db is None:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Find the new food in the database
        new_food = food_db[food_db['meal_name'].str.lower() == new_food_name.lower()]
        
        if new_food.empty:
            return jsonify({"error": f"Food '{new_food_name}' not found in database"}), 404
        
        new_food_row = new_food.iloc[0]
        
        # Create the replacement food object
        replacement_food = {
            "type": current_meals[meal_index]["type"] if meal_index < len(current_meals) else "Snack",
            "name": new_food_row['meal_name'],
            "cal": int(new_food_row['Calories']),
            "prot": float(new_food_row['Proteins']),
            "carbs": float(new_food_row['Carbs']),
            "fats": float(new_food_row['Fats']),
            "cluster": goal
        }
        
        # Calculate new totals
        updated_meals = current_meals.copy()
        updated_meals[meal_index] = replacement_food
        
        total_calories = sum(meal['cal'] for meal in updated_meals)
        total_protein = sum(meal['prot'] for meal in updated_meals)
        total_carbs = sum(meal['carbs'] for meal in updated_meals)
        total_fats = sum(meal['fats'] for meal in updated_meals)
        
        response = {
            "success": True,
            "replacedFood": replacement_food,
            "updatedMeals": updated_meals,
            "totals": {
                "calories": round(total_calories, 2),
                "protein": round(total_protein, 2),
                "carbs": round(total_carbs, 2),
                "fats": round(total_fats, 2)
            },
            "status": "success"
        }
        
        print(f"✅ Replaced meal at index {meal_index} with {new_food_row['meal_name']}")
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error replacing food: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    print(f"🚀 Starting Flask Server on port {port}...")
    print(f"🔧 Debug mode: {debug}")
    app.run(debug=debug, host='0.0.0.0', port=port, use_reloader=False)