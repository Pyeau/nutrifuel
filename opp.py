import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Athletic Diet Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for Professional Design ---
st.markdown("""
<style>
/* Base Resets and Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* {box-sizing: border-box; font-family: 'Inter', sans-serif;}
header, footer {visibility: hidden;}

/* Main App Background - Modern gradient */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    color: #1e293b;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid rgba(30,41,59,0.1);
    padding: 1.5rem 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

/* Headers */
h1 {
    color: #0f172a;
    font-weight: 800;
    font-size: 2.5rem !important;
    margin: 1rem 0 0.5rem 0;
    text-align: center;
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2, h3 {
    color: #0f172a;
    font-weight: 700;
    margin: 1rem 0;
}

h4 {
    color: #1e40af;
    font-weight: 600;
}

/* Main Card */
.main-card {
    background: #ffffff;
    border: 1px solid rgba(30,41,59,0.1);
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 10px 30px rgba(30,41,59,0.08);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.main-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 40px rgba(30,41,59,0.12);
}

/* Form Elements */
.stTextInput > div, .stSelectbox > div, .stNumberInput > div {
    background-color: #f8fafc !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    transition: all 0.2s ease !important;
}

.stTextInput > div:focus, .stSelectbox > div:focus, .stNumberInput > div:focus,
.stTextInput > div:hover, .stSelectbox > div:hover, .stNumberInput > div:hover {
    border-color: #1e40af !important;
    box-shadow: 0 0 0 3px rgba(30,64,175,0.1) !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 3rem;
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(30,64,175,0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(30,64,175,0.4);
}

/* Meal Card Styling */
.meal-header {
    font-size: 1.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.food-name {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 1rem;
}

.macro-box {
    background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 2px solid #cbd5e1;
    transition: all 0.2s ease;
}

.macro-box:hover {
    border-color: #1e40af;
    box-shadow: 0 4px 12px rgba(30,64,175,0.15);
}

.macro-value {
    font-weight: 800;
    font-size: 1.3rem;
    color: #0f172a;
    margin: 0.5rem 0;
}

.macro-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
}

/* Alerts */
.stAlert {
    background-color: #f0f4f8 !important;
    color: #0f172a !important;
    border: 2px solid #bfdbfe !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(30,64,175,0.1) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #f8fafc !important;
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    color: #0f172a !important;
}

.streamlit-expanderHeader:hover {
    background-color: #f0f4f8 !important;
    border-color: #1e40af !important;
}
</style>
""", unsafe_allow_html=True)


# --- 3. Load Models and Data ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    # Try new filename first, then fall back to old filename
    new_file = os.path.join(BASE_DIR, 'daily_nutrition_clustered.csv')
    old_file = os.path.join(BASE_DIR, 'improved_food_database.csv')
    
    try:
        if os.path.exists(new_file):
            df = pd.read_csv(new_file)
            st.sidebar.success("✅ Loaded: daily_nutrition_clustered.csv")
            return df
        elif os.path.exists(old_file):
            df = pd.read_csv(old_file)
            st.sidebar.success("✅ Loaded: improved_food_database.csv")
            return df
        else:
            st.error(f"❌ Food database not found. Please run train.py first.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading food database: {e}")
        return None

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, 'meal_plan_model.joblib')
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"❌ Meal Plan Classifier Model not found. Please run train.py first.")
        return None

@st.cache_resource
def load_clustering_models():
    """Load K-Means model and cluster mapping from train.py"""
    try:
        kmeans_path = os.path.join(BASE_DIR, 'food_kmeans_model.joblib')
        mapping_path = os.path.join(BASE_DIR, 'cluster_mealplan_mapping.joblib')
        kmeans = joblib.load(kmeans_path)
        mapping = joblib.load(mapping_path)
        return kmeans, mapping
    except FileNotFoundError:
        st.warning("⚠️ K-Means models not found. Ensure train.py ran successfully.")
        return None, None

@st.cache_resource
def load_metrics():
    """Load Classification metrics for sidebar"""
    try:
        metrics_path = os.path.join(BASE_DIR, 'model_performance_metrics.joblib')
        return joblib.load(metrics_path)
    except FileNotFoundError:
        return {'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None}

# Load everything
food_db = load_data()
classifier_model = load_model()
kmeans_model, cluster_mapping = load_clustering_models()
classifier_metrics = load_metrics()

if food_db is None or classifier_model is None:
    st.stop()

# Check if Meal_Type column exists (from new dataset)
has_meal_type = 'Meal_Type' in food_db.columns
if has_meal_type:
    st.sidebar.info("✨ Meal Type detection enabled!")


# --- 4. Helper Functions ---
def calculate_bmr(age, weight_kg, height_cm, gender):
    """Harris-Benedict Equation"""
    if gender == "Male":
        return 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age)

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
        "Weight Loss": "Weight Loss", "General / Balanced": "Balanced"
    }
    return mapping.get(goal, "Balanced")


def predict_goal_for_user(age, bmi, training_intensity):
    """
    Predict the best performance goal based on user's BMI, age, and training frequency.
    """
    intensity_map = {
        "Light Training (1–3 days/week)": 1,
        "Moderate Training (3–5 days/week)": 2,
        "High Training (6–7 days/week)": 3
    }
    intensity = intensity_map.get(training_intensity, 2)
    
    # BMI Category
    if bmi < 18.5:
        bmi_cat = "underweight"
    elif bmi < 25:
        bmi_cat = "normal"
    elif bmi < 30:
        bmi_cat = "overweight"
    else:
        bmi_cat = "obese"
    
    # Prediction Logic
    if bmi_cat in ["overweight", "obese"]:
        return "Weight Loss"
    elif bmi_cat in ["underweight", "normal"] and intensity == 3:
        return "Build Muscle"
    elif bmi_cat == "normal" and intensity == 1:
        return "General / Balanced"
    elif intensity == 3:
        return "HIIT"
    else:
        return "General / Balanced"


# --- RECOMMENDATION ENGINE ---
import random

def get_recommendations_by_cluster(food_db, predicted_plan, exclude_names=None, protein_target=None, calorie_target=None):
    """
    Select meals based on cluster with meal type awareness if available.
    """
    if food_db is None or food_db.empty: return []

    if exclude_names is None: exclude_names = []
    used_foods = set(exclude_names)

    bmr = st.session_state.get('bmr', 2000)
    if calorie_target is None:
        calorie_target = st.session_state.get('tdee', bmr)

    meal_calories = {"Breakfast": calorie_target * 0.30, "Lunch": calorie_target * 0.40, "Dinner": calorie_target * 0.30}

    nutrition_requirements = {
        "Weight Loss":  {'Proteins': {'min': 25, 'target': 40}, 'Carbs': {'min': 0, 'target': 15}, 'Fats': {'min': 5, 'target': 20}},
        "Build Muscle": {'Proteins': {'min': 20, 'target': 35}, 'Carbs': {'min': 20, 'target': 30}, 'Fats': {'min': 15, 'target': 25}},
        "Endurance":    {'Proteins': {'min': 5, 'target': 15}, 'Carbs': {'min': 40, 'target': 60}, 'Fats': {'min': 5, 'target': 15}},
        "HIIT":         {'Proteins': {'min': 15, 'target': 25}, 'Carbs': {'min': 30, 'target': 45}, 'Fats': {'min': 10, 'target': 20}},
        "Balanced":     {'Proteins': {'min': 10, 'target': 20}, 'Carbs': {'min': 20, 'target': 40}, 'Fats': {'min': 20, 'target': 35}}
    }
    plan_reqs = nutrition_requirements.get(predicted_plan, nutrition_requirements["Balanced"])
    
    if protein_target is not None and protein_target > 0:
        protein_per_meal_avg = protein_target / 3
        plan_reqs['Proteins'] = {
            'min': protein_per_meal_avg * 0.75,
            'target': protein_per_meal_avg * 1.2
        }
    
    cluster_filtered = food_db[food_db['Meal_Plan'] == predicted_plan]
    if cluster_filtered.empty: 
        cluster_filtered = food_db

    cluster_filtered = cluster_filtered.copy()
    cluster_filtered['Calories'] = pd.to_numeric(cluster_filtered['Calories'], errors='coerce')

    # Filter out excluded foods
    available_foods = cluster_filtered[~cluster_filtered['meal_name'].isin(exclude_names)] if exclude_names else cluster_filtered
    if available_foods.empty:
        available_foods = cluster_filtered

    refined_foods = available_foods[
        (available_foods['Proteins'] >= plan_reqs['Proteins']['min']) &
        (available_foods['Carbs'] >= plan_reqs['Carbs']['min'])
    ]
    if refined_foods.empty:
        refined_foods = available_foods

    # Check if Meal_Type column exists
    has_meal_type = 'Meal_Type' in refined_foods.columns

    def top_candidates_for_target(df, meal_target, meal_type=None, top_k=100):
        df = df.dropna(subset=['Calories'])
        if df.empty: return []
        
        # If meal type exists and specified, filter by it
        if has_meal_type and meal_type:
            # Try to match meal type, but if no matches, use all foods
            meal_type_foods = df[df['Meal_Type'] == meal_type]
            if not meal_type_foods.empty:
                df = meal_type_foods
        
        df = df.assign(diff=(df['Calories'] - meal_target).abs())
        candidates = df.nsmallest(top_k, 'diff').to_dict('records')
        random.shuffle(candidates)
        return candidates

    meal_types = ["Breakfast", "Lunch", "Dinner"]
    candidates = {}
    
    # Get candidates with meal type awareness
    for mt in meal_types:
        candidates[mt] = top_candidates_for_target(refined_foods, meal_calories[mt], meal_type=mt, top_k=100)

    # STEP 1: Try a normal 3-meal plan
    meals = []
    total_calories = 0
    selected_food_names = set(exclude_names) if exclude_names else set()
    
    for meal_type in meal_types:
        target_calories = meal_calories[meal_type]
        cand_list = candidates.get(meal_type, [])
        if not cand_list:
            continue
        
        available_candidates = [c for c in cand_list if c.get('meal_name') not in selected_food_names]
        if not available_candidates:
            available_candidates = cand_list
        
        selection_pool_size = min(30, len(available_candidates))
        selected = random.choice(available_candidates[:selection_pool_size])
        
        if selected is None:
            continue
        
        selected_food_names.add(selected.get('meal_name'))
        
        # Get meal type if available
        detected_meal_type = selected.get('Meal_Type', 'General') if has_meal_type else 'General'
        
        meals.append({
            "Meal": meal_type, 
            "Food": selected.get('meal_name', 'Unknown'),
            "Meal_Type": detected_meal_type,
            "Target Calories": f"{target_calories:.0f}", 
            "Actual Calories": f"{selected.get('Calories', 0):.0f}",
            "Proteins": f"{selected.get('Proteins', 0):.1f}g", 
            "Carbs": f"{selected.get('Carbs', 0):.1f}g", 
            "Fats": f"{selected.get('Fats', 0):.1f}g", 
            "Calories": f"{selected.get('Calories', 0):.0f} kcal",
            "Macro Score": "--", 
            "Cluster_Name": predicted_plan
        })
        total_calories += (selected.get('Calories') or 0)

    # STEP 2: Check if total calories match the target
    calorie_tolerance = max(0.08 * calorie_target, 150)
    if abs(total_calories - calorie_target) <= calorie_tolerance:
        return meals
    
    # STEP 3: Try combining main meal with snack if needed
    meal_order = meal_types.copy()
    random.shuffle(meal_order)
    
    # Get snack foods from the same meal plan
    snack_foods = []
    if has_meal_type:
        snack_foods = refined_foods[
            (refined_foods['Meal_Type'] == 'Snack') & 
            (~refined_foods['meal_name'].isin(selected_food_names))
        ].to_dict('records')
        random.shuffle(snack_foods)
    
    for meal_to_replace_type in meal_order:
        meal_to_replace_idx = meal_types.index(meal_to_replace_type)
        other_meals = [m for i, m in enumerate(meals) if i != meal_to_replace_idx]
        other_calories = sum(float(str(m['Calories']).split()[0]) for m in other_meals)
        
        remaining_target = calorie_target - other_calories
        if remaining_target < 100:
            continue
        
        # Get main meal candidates
        main_meal_cands = [c for c in candidates.get(meal_to_replace_type, [])[:100] 
                          if c.get('meal_name') not in selected_food_names]
        random.shuffle(main_meal_cands)
        
        if not main_meal_cands:
            continue
        
        best_combo = None
        best_combo_diff = float('inf')
        combo_tolerance = max(0.10 * remaining_target, 50)
        
        # Try pairing main meal with snacks
        if snack_foods and has_meal_type:
            check_limit_main = min(len(main_meal_cands), 30)
            check_limit_snack = min(len(snack_foods), 30)
            
            for i in range(check_limit_main):
                main_food = main_meal_cands[i]
                for j in range(check_limit_snack):
                    snack_food = snack_foods[j]
                    
                    if main_food['meal_name'] == snack_food['meal_name']:
                        continue
                    
                    total = (main_food.get('Calories') or 0) + (snack_food.get('Calories') or 0)
                    diff = abs(total - remaining_target)
                    
                    if diff < best_combo_diff:
                        best_combo_diff = diff
                        best_combo = (main_food, snack_food)
                    
                    if best_combo_diff <= combo_tolerance:
                        break
                
                if best_combo_diff <= combo_tolerance:
                    break
        else:
            # Fallback: pair any two foods from the meal type if no snacks available
            n = len(main_meal_cands)
            if n >= 2:
                check_limit = min(n, 50)
                for i in range(check_limit):
                    for j in range(i+1, check_limit):
                        if main_meal_cands[i]['meal_name'] == main_meal_cands[j]['meal_name']:
                            continue
                        total = (main_meal_cands[i].get('Calories') or 0) + (main_meal_cands[j].get('Calories') or 0)
                        diff = abs(total - remaining_target)
                        if diff < best_combo_diff:
                            best_combo_diff = diff
                            best_combo = (main_meal_cands[i], main_meal_cands[j])
                        if best_combo_diff <= combo_tolerance:
                            break
                    if best_combo_diff <= combo_tolerance:
                        break
        
        if best_combo is not None:
            f1, f2 = best_combo
            cal_sum = (f1.get('Calories') or 0) + (f2.get('Calories') or 0)
            prot_sum = (f1.get('Proteins') or 0) + (f2.get('Proteins') or 0)
            carb_sum = (f1.get('Carbs') or 0) + (f2.get('Carbs') or 0)
            fat_sum = (f1.get('Fats') or 0) + (f2.get('Fats') or 0)
            
            # Get meal types for both foods
            type1 = f1.get('Meal_Type', 'General') if has_meal_type else 'General'
            type2 = f2.get('Meal_Type', 'General') if has_meal_type else 'General'
            
            # Format the combined type display
            if type2 == 'Snack':
                combined_type = f"{type1} + Snack"
            elif type1 == 'Snack':
                combined_type = f"Snack + {type2}"
            elif type1 != type2:
                combined_type = f"{type1} + {type2}"
            else:
                combined_type = type1
            
            result = []
            for i, m in enumerate(meals):
                if i == meal_to_replace_idx:
                    result.append({
                        "Meal": meal_to_replace_type,
                        "Food": f"{f1.get('meal_name', 'Food1')} + {f2.get('meal_name', 'Food2')}",
                        "Meal_Type": combined_type,
                        "Target Calories": f"{remaining_target:.0f}",
                        "Actual Calories": f"{cal_sum:.0f}",
                        "Proteins": f"{prot_sum:.1f}g", 
                        "Carbs": f"{carb_sum:.1f}g", 
                        "Fats": f"{fat_sum:.1f}g", 
                        "Calories": f"{cal_sum:.0f} kcal",
                        "Macro Score": "--", 
                        "Cluster_Name": predicted_plan
                    })
                else:
                    result.append(m)
            return result
    
    return meals


# --- 6. Main Page ---
st.title("💪 Athletic Diet AI")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 18px; font-weight: 500; margin-bottom: 2rem;'>AI-Powered Personalized Nutrition for Peak Performance</div>", unsafe_allow_html=True)

# --- Input Form ---
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("<h3 style='color: #0f172a; margin-bottom: 1.5rem;'>👤 Athlete Profile</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    age = st.number_input("Age", 15, 80, 25)
    gender = st.selectbox("Gender", ("Male", "Female"))
    height_cm = st.number_input("Height (cm)", 140, 220, 170)
    weight_kg = st.number_input("Weight (kg)", 40, 150, 70)
with col2:
    goals = ["Not Sure - Let AI Suggest", "Build Muscle", "Endurance", "HIIT", "Weight Loss", "General / Balanced"]
    goal = st.selectbox("Performance Goal", goals)
    activity = st.selectbox("Training Frequency", (
        "Light Training (1–3 days/week)", "Moderate Training (3–5 days/week)", "High Training (6–7 days/week)"
    ), index=1)

    calorie_choices = ["Match TDEE", "TDEE - 200 kcal", "TDEE - 300 kcal", "Manual target"]
    calorie_strategy = st.selectbox("Calorie Strategy", calorie_choices)
    manual_calorie = None
    if calorie_strategy == "Manual target":
        manual_calorie = st.number_input("Manual Calorie Target (kcal)", 1000, 5000, 2000, step=50)

    if goal == "Not Sure - Let AI Suggest":
        bmi = weight_kg / ((height_cm / 100) ** 2)
        suggested_goal = predict_goal_for_user(age, bmi, activity)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 12px; margin-bottom: 20px; 
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);">
            <p style="color: white; font-size: 1rem; margin: 0; font-weight: 500;">
                🎯 AI Recommendation
            </p>
            <p style="color: #f0f4ff; font-size: 1.1rem; margin: 10px 0 0 0; font-weight: 600;">
                Based on your profile, we recommend: <strong>{suggested_goal}</strong>
            </p>
            <p style="color: #e8ecff; font-size: 0.95rem; margin: 8px 0 0 0; line-height: 1.4;">
                • BMI Category: {'Normal' if 18.5 <= bmi < 25 else 'Overweight' if 25 <= bmi < 30 else 'Obese' if bmi >= 30 else 'Underweight'}<br>
                • Training Level: {activity}<br>
                • Age Group: {age} years<br>
                This goal is optimized for your fitness profile.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        goal = suggested_goal
        st.session_state['use_rf_prediction'] = True
    else:
        st.session_state['use_rf_prediction'] = False

    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("GENERATE PLAN", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# --- 7. Logic Execution ---
if btn:
    with st.spinner("Analyzing metabolic needs..."):
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        bmr = calculate_bmr(age, weight_kg, height_cm, gender)
        tdee = calculate_tdee(bmr, activity)
        exp_level = {"Light Training (1–3 days/week)": 2, "Moderate Training (3–5 days/week)": 4, "High Training (6–7 days/week)": 6}[activity]

        use_rf_prediction = st.session_state.get('use_rf_prediction', False)
        
        if use_rf_prediction:
            user_data = {
                'Age': age, 'Weight (kg)': weight_kg, 'Height (m)': height_m, 'BMI': bmi,
                'Fat_Percentage': 18.0, 'Avg_BPM': 130, 'Resting_BPM': 65, 
                'Experience_Level': exp_level, 'Gender': gender, 'Goals': 'Balanced',
                'Calories_Burned': tdee, 'Workout_Frequency (days/week)': exp_level,
                'protein_per_kg': 1.6
            }
        else:
            user_data = {
                'Age': age, 'Weight (kg)': weight_kg, 'Height (m)': height_m, 'BMI': bmi,
                'Fat_Percentage': 18.0, 'Avg_BPM': 130, 'Resting_BPM': 65, 
                'Experience_Level': exp_level, 'Gender': gender, 'Goals': map_goal_to_model_format(goal),
                'Calories_Burned': tdee, 'Workout_Frequency (days/week)': exp_level,
                'protein_per_kg': 1.6
            }

        user_df_input = pd.DataFrame([user_data])
        
        try:
            if use_rf_prediction:
                predicted_meal_plan = classifier_model.predict(user_df_input)[0]
                proba_array = classifier_model.predict_proba(user_df_input)[0]
                confidence_proba = proba_array.max()
                
                classifier_classes = classifier_model.named_steps['model'].classes_
                proba_dict = dict(zip(classifier_classes, proba_array))
                st.session_state['classifier_proba'] = proba_dict
                
                st.info("✨ Using Random Forest Classifier to predict your optimal meal plan!")
            else:
                predicted_meal_plan = map_goal_to_model_format(goal)
                proba_array = classifier_model.predict_proba(user_df_input)[0]
                confidence_proba = proba_array.max()
                
                classifier_classes = classifier_model.named_steps['model'].classes_
                proba_dict = dict(zip(classifier_classes, proba_array))
                st.session_state['classifier_proba'] = proba_dict
        except Exception as e:
            st.warning(f"Could not predict with classifier: {e}. Using user-selected goal.")
            predicted_meal_plan = map_goal_to_model_format(goal)
            confidence_proba = 0.0
            st.session_state['classifier_proba'] = {}
        
        final_plan = predicted_meal_plan
        
        training_level_map = {
            "Light Training (1–3 days/week)": {"name": "Light", "intensity": 1},
            "Moderate Training (3–5 days/week)": {"name": "Moderate", "intensity": 2},
            "High Training (6–7 days/week)": {"name": "High", "intensity": 3}
        }
        training_profile = training_level_map.get(activity, training_level_map["Moderate Training (3–5 days/week)"])
        st.session_state['training_intensity'] = training_profile['name']
        st.session_state['training_intensity_level'] = training_profile['intensity'] 

        if calorie_strategy == "Match TDEE":
            calorie_target = tdee
        elif calorie_strategy == "TDEE - 200 kcal":
            calorie_target = max(1200, tdee - 200)
        elif calorie_strategy == "TDEE - 300 kcal":
            calorie_target = max(1200, tdee - 300)
        elif calorie_strategy == "Manual target" and manual_calorie is not None:
            calorie_target = manual_calorie
        else:
            calorie_target = tdee

        meals = get_recommendations_by_cluster(food_db, final_plan, protein_target=None, calorie_target=calorie_target)
        
        st.session_state.update({
            'bmr': bmr, 'tdee': tdee, 'bmi': bmi, 'predicted_plan': final_plan, 
            'classifier_confidence': confidence_proba,
            'meal_plan': pd.DataFrame(meals),
            'calorie_target': calorie_target
        })


# --- 8. Display Results ---
if st.session_state.get('predicted_plan'):
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    st.subheader(f"Recommended Strategy: {st.session_state['predicted_plan']}")
    
    st.markdown("---")
    st.markdown("<h3 style='color: #0f172a; margin-bottom: 1.5rem;'>📊 Your Analysis</h3>", unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3 = st.columns(3, gap="large")
    with metric_col1:
        bmi_value = st.session_state['bmi']
        if bmi_value < 18.5:
            bmi_category = "Underweight"
            bmi_color = "#3b82f6"
        elif bmi_value < 25:
            bmi_category = "Normal Weight"
            bmi_color = "#10b981"
        elif bmi_value < 30:
            bmi_category = "Overweight"
            bmi_color = "#f59e0b"
        else:
            bmi_category = "Obese"
            bmi_color = "#ef4444"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%); 
                    border: 2px solid #1e40af; border-radius: 12px; padding: 1.5rem; text-align: center;'>
            <p style='color: #64748b; margin: 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase;'>BMI</p>
            <h2 style='color: #1e40af; margin: 0.5rem 0; font-size: 2.5rem;'>{bmi_value:.1f}</h2>
            <p style='color: {bmi_color}; margin: 0.5rem 0 0 0; font-weight: 700; font-size: 1rem;'>{bmi_category}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%); 
                    border: 2px solid #1e40af; border-radius: 12px; padding: 1.5rem; text-align: center;'>
            <p style='color: #64748b; margin: 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase;'>BMR</p>
            <h2 style='color: #1e40af; margin: 0.5rem 0; font-size: 2rem;'>{st.session_state['bmr']:.0f}</h2>
            <p style='color: #64748b; margin: 0; font-size: 0.8rem;'>kcal/day</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%); 
                    border: 2px solid #1e40af; border-radius: 12px; padding: 1.5rem; text-align: center;'>
            <p style='color: #64748b; margin: 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase;'>TDEE</p>
            <h2 style='color: #1e40af; margin: 0.5rem 0; font-size: 2rem;'>{st.session_state['tdee']:.0f}</h2>
            <p style='color: #64748b; margin: 0; font-size: 0.8rem;'>kcal/day</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    training_intensity = st.session_state.get('training_intensity', 'Unknown')
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                border-left: 4px solid #f59e0b; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0;'>
        <p style='color: #92400e; margin: 0; font-weight: 600;'>⚡ <strong>Training Intensity Analysis</strong></p>
        <p style='color: #92400e; margin: 0.5rem 0 0 0;'>Your <strong>{training_intensity} Training</strong> frequency is optimized for this plan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #f0fdf4 0%, #e7f5e4 100%); 
                border-left: 4px solid #22c55e; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0;'>
        <p style='color: #166534; margin: 0; font-weight: 600;'>🎯 <strong>Meal Plan Prediction</strong></p>
        <p style='color: #166534; margin: 0.5rem 0 0 0;'>AI predicted: <strong>{st.session_state['predicted_plan']}</strong> 
        (Confidence: <strong>{st.session_state.get("classifier_confidence", 0)*100:.1f}%</strong>)</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📊 View Classifier Probability Breakdown"):
        st.markdown("**How the AI categorized your profile:**")
        
        if 'classifier_proba' in st.session_state:
            proba_dict = st.session_state['classifier_proba']
            
            col1, col2 = st.columns([2, 1])
            with col1:
                for meal_plan, prob in sorted(proba_dict.items(), key=lambda x: x[1], reverse=True):
                    color = "#10b981" if meal_plan == st.session_state['predicted_plan'] else "#cbd5e1"
                    st.write(f"**{meal_plan}**: {prob*100:.1f}%")
                    st.progress(prob, text=f"{prob*100:.1f}%")
            with col2:
                st.markdown("**Classes:**")
                st.markdown("- 🏋️ Build Muscle")
                st.markdown("- ⚖️ Balanced")
                st.markdown("- 🏃 Endurance")
                st.markdown("- ⚡ HIIT")
                st.markdown("- 📉 Weight Loss")

    if not st.session_state['meal_plan'].empty:
        st.markdown("<h3 style='color: #0f172a; margin: 2rem 0 1rem 0;'>🍽️ Your Daily Meal Plan</h3>", unsafe_allow_html=True)
        
        plan = st.session_state['predicted_plan']
        focus_text = ""
        if plan == "Endurance": focus_text = "⚡ **Focus:** High Carbohydrates for sustained energy."
        elif plan == "Weight Loss": focus_text = "🔥 **Focus:** High Protein, Low Calorie for fat loss."
        elif plan == "Build Muscle": focus_text = "💪 **Focus:** High protein with balanced macros."
        elif plan == "HIIT": focus_text = "🏃 **Focus:** Moderate Carbs & Protein for explosive energy."
        elif plan == "Balanced": focus_text = "⚖️ **Focus:** Balanced macro distribution."
            
        st.info(focus_text)
        
        meal_emojis = {"Breakfast": "🍳", "Lunch": "🥗", "Dinner": "🍽️"}
        
        for index, row in st.session_state['meal_plan'].iterrows():
            with st.container():
                # Show meal type badge if available
                meal_type_badge = ""
                if 'Meal_Type' in row and pd.notna(row['Meal_Type']) and row['Meal_Type'] != 'General':
                    meal_type_badge = f"<span style='background: #e0f2fe; color: #0369a1; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem;'>📋 {row['Meal_Type']}</span>"
                
                st.markdown(f"""
                <div style="background-color: white; padding: 1.5rem; border-radius: 12px; 
                            border: 1px solid rgba(60,64,67,0.12); box-shadow: 0 2px 6px rgba(60,64,67,0.05); margin-bottom: 1rem;">
                    <div class="meal-header">{meal_emojis.get(row['Meal'], '🍱')} {row['Meal']}{meal_type_badge}</div>
                    <div class="food-name">{row['Food']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                m1, m2, m3, m4 = st.columns(4)
                
                with m1: st.markdown(f"""<div class="macro-box"><div class="macro-label">Protein</div><div class="macro-value" style="color: #d93025;">{row['Proteins']}</div></div>""", unsafe_allow_html=True)
                with m2: st.markdown(f"""<div class="macro-box"><div class="macro-label">Carbs</div><div class="macro-value" style="color: #1e8e3e;">{row['Carbs']}</div></div>""", unsafe_allow_html=True)
                with m3: st.markdown(f"""<div class="macro-box"><div class="macro-label">Fats</div><div class="macro-value" style="color: #f9ab00;">{row['Fats']}</div></div>""", unsafe_allow_html=True)
                with m4: st.markdown(f"""<div class="macro-box" style="background-color: #e8f0fe; border-color: #d2e3fc;"><div class="macro-label">Calories</div><div class="macro-value" style="color: #1967d2;">{row['Calories']}</div></div>""", unsafe_allow_html=True)
                
                st.write("") 

        df = st.session_state['meal_plan']
        total_cals = sum(float(str(x).split()[0]) for x in df['Calories'])
        
        st.markdown("---")
        st.caption(f"📊 **Daily Summary:** Total Intake: **{total_cals:.0f} kcal** | Target: **{st.session_state['tdee']:.0f} kcal**")
        
    else:
        st.error("Could not generate meals. Please check database connection.")

    if st.button("🔄 Regenerate Meals", type="primary", use_container_width=True):
        current_foods = st.session_state['meal_plan']['Food'].tolist()
        exclude_foods = []
        for food_str in current_foods:
            if ' + ' in food_str:
                parts = food_str.split(' + ')
                exclude_foods.extend([p.strip() for p in parts])
            else:
                exclude_foods.append(food_str)
        
        calorie_target = st.session_state.get('calorie_target', st.session_state.get('tdee', st.session_state.get('bmr', 2000)))
        new_meals = get_recommendations_by_cluster(
            food_db, st.session_state['predicted_plan'], exclude_names=exclude_foods, 
            protein_target=None, calorie_target=calorie_target
        )
        st.session_state['meal_plan'] = pd.DataFrame(new_meals)
        st.rerun()

    with st.expander("📋 View All Available Foods for This Meal Plan"):
        plan = st.session_state['predicted_plan']
        plan_foods = food_db[food_db['Meal_Plan'] == plan]
        
        if plan_foods.empty:
            st.info(f"No foods found for {plan} meal plan.")
        else:
            st.markdown(f"**Total Available Foods for {plan}: {len(plan_foods)}**")
            
            # Build columns list - always include Meal_Plan
            display_cols = ['meal_name', 'Meal_Plan']
            col_names = ['Food Name', 'Meal Plan']
            
            # Add Meal_Type if available
            if has_meal_type:
                display_cols.append('Meal_Type')
                col_names.append('Meal Type')
            
            # Add nutritional columns
            display_cols.extend(['Calories', 'Proteins', 'Carbs', 'Fats'])
            col_names.extend(['Calories (kcal)', 'Protein (g)', 'Carbs (g)', 'Fats (g)'])
            
            display_df = plan_foods[display_cols].copy()
            display_df.columns = col_names
            
            # Format numerical columns
            display_df['Calories (kcal)'] = display_df['Calories (kcal)'].apply(lambda x: f"{float(x):.0f}" if pd.notna(x) else "N/A")
            display_df['Protein (g)'] = display_df['Protein (g)'].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "N/A")
            display_df['Carbs (g)'] = display_df['Carbs (g)'].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "N/A")
            display_df['Fats (g)'] = display_df['Fats (g)'].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "N/A")
            
            # Add filter by meal type if available
            if has_meal_type:
                st.markdown("**Filter by Meal Type:**")
                meal_type_filter = st.multiselect(
                    "Select meal types to display",
                    options=['All'] + sorted(plan_foods['Meal_Type'].unique().tolist()),
                    default=['All']
                )
                
                if 'All' not in meal_type_filter and meal_type_filter:
                    display_df = display_df[display_df['Meal Type'].isin(meal_type_filter)]
                    st.markdown(f"*Showing {len(display_df)} foods*")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)