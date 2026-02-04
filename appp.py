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

# --- 2. Custom CSS for Professional Design (Removed for brevity, assuming standard styling remains) ---
# --- Standard CSS and layout elements are assumed to be here ---

# --- 2. Custom CSS for Athlete Dashboard Design ---
st.markdown("""
<style>
/* Import Fonts: Oswald (Sporty/Bold) and Open Sans (Clean) */
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap');

* {box-sizing: border-box;}

/* Main Background - Deep Gunmetal/Black */
.stApp {
    background-color: #0b0e11;
    color: #e2e8f0;
}

header, footer {visibility: hidden;}

/* --- HERO SECTION --- */
.hero-container {
    padding: 2rem 1rem;
    margin-bottom: 2rem;
    background: linear-gradient(180deg, rgba(11,14,17,0) 0%, #0b0e11 100%);
}

.hero-title {
    font-family: 'Oswald', sans-serif;
    font-size: 4rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    background: linear-gradient(90deg, #ffffff 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}

.hero-subtitle {
    font-family: 'Open Sans', sans-serif;
    color: #3b82f6; /* Performance Blue */
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 4px;
    margin-bottom: 0.5rem;
}

/* --- DASHBOARD FORM --- */
.profile-card {
    background-color: #151921; /* Lighter dark for card */
    border: 1px solid #2d3748;
    border-top: 4px solid #3b82f6; /* Blue Accent Top */
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

.form-header {
    font-family: 'Oswald', sans-serif;
    color: #ffffff;
    font-size: 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #2d3748;
    padding-bottom: 10px;
}

/* Input Fields - Dashboard Look */
.stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div {
    background-color: #0b0e11 !important;
    color: #ffffff !important;
    border: 1px solid #2d3748 !important;
    border-radius: 4px !important;
}

/* Labels */
.stTextInput label, .stNumberInput label, .stSelectbox label {
    color: #94a3b8 !important; /* Muted blue-grey */
    font-family: 'Oswald', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.9rem !important;
}

/* Focus State */
.stTextInput > div > div:focus-within, .stNumberInput > div > div:focus-within, .stSelectbox > div > div:focus-within {
    border-color: #3b82f6 !important; /* Blue glow */
    box-shadow: 0 0 0 1px #3b82f6 !important;
}

/* --- BUTTON STYLING --- */
.stButton > button {
    background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
    color: #ffffff !important;
    font-family: 'Oswald', sans-serif;
    font-weight: 600;
    font-size: 1.2rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-radius: 4px !important;
    border: none !important;
    padding: 1rem 2rem;
    transition: all 0.2s ease;
    clip-path: polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px); /* Techy shape */
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(37, 99, 235, 0.4);
}

/* --- RESULTS AREA --- */
.main-card {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 2rem;
    color: #0f172a;
    margin-top: 2rem;
}

.main-card h1, .main-card h2, .main-card h3 {
    color: #0f172a !important;
}
</style>
""", unsafe_allow_html=True)


# --- 3. Load Models and Data ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    file_path = os.path.join(BASE_DIR, 'improved_food_database.csv')
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"❌ 'improved_food_database.csv' not found. Please run train.py first.")
        return None

@st.cache_resource
def load_model():
    # Load the Random Forest Classifier Model for Meal Plan Prediction
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
        return {'accuracy': None, 'r2': None}

# Load everything
food_db = load_data()
classifier_model = load_model()
kmeans_model, cluster_mapping = load_clustering_models()
classifier_metrics = load_metrics()

if food_db is None or classifier_model is None:
    st.stop()


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
    
    Logic:
    - Underweight/Normal BMI + High Training -> Build Muscle
    - Overweight/Obese + Any Training -> Weight Loss
    - Normal BMI + Light Training -> General/Balanced
    - High Training + Normal BMI -> HIIT or Build Muscle
    - Any BMI + High Training -> Endurance or HIIT
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


# --- RECOMMENDATION ENGINE (Updated with Meal Time Logic) ---
# --- RECOMMENDATION ENGINE (Updated with Snack Pairing Logic) ---
import random

def get_recommendations_by_cluster(food_db, predicted_plan, exclude_names=None, protein_target=None, calorie_target=None):
    """
    Selects 3 meals. If calorie target is not met, finds a 'Snack' 
    and pairs it with one of the meals to close the gap.
    """
    if food_db is None or food_db.empty: return []

    if exclude_names is None: exclude_names = []
    
    # Defaults
    bmr = st.session_state.get('bmr', 2000)
    if calorie_target is None:
        calorie_target = st.session_state.get('tdee', bmr)

    # 1. Targets Setup
    meal_calories = {"Breakfast": calorie_target * 0.30, "Lunch": calorie_target * 0.40, "Dinner": calorie_target * 0.30}
    
    # Nutrition Requirements
    nutrition_requirements = {
        "Weight Loss":  {'Proteins': {'min': 25}, 'Carbs': {'min': 0}},
        "Build Muscle": {'Proteins': {'min': 20}, 'Carbs': {'min': 20}},
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
        # Filter by macro requirements
        filtered = pool[
            (pool['Proteins'] >= plan_reqs['Proteins']['min']) &
            (pool['Carbs'] >= plan_reqs['Carbs']['min'])
        ]
        if filtered.empty: filtered = pool # Fallback
        
        # Filter by Meal Time if specified
        if meal_time_filter:
            # Try specific time (e.g. Breakfast)
            time_match = filtered[filtered['Meal_Time'] == meal_time_filter]
            # Fallback to Snack if specific time empty
            if time_match.empty:
                time_match = filtered[filtered['Meal_Time'] == 'Snack']
            # Fallback to All if still empty
            if not time_match.empty:
                filtered = time_match

        filtered = filtered.dropna(subset=['Calories'])
        # Sort by calorie closeness
        filtered = filtered.assign(diff=(filtered['Calories'] - target_cal).abs())
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
            # We store raw numbers for calculation, format strings later if needed
            meals.append({
                "Meal": m_type,
                "Food": selection['meal_name'],
                "Source_Meal_Time": selection.get('Meal_Time', 'General'),
                "Target_Cal_Num": meal_calories[m_type], # Store for gap calculation
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
                # (Target - Actual)
                meals.sort(key=lambda x: x['Target_Cal_Num'] - x['Cal_Num'], reverse=True)
                target_meal = meals[0] # This meal needs calories the most
                
                # Update the meal with the snack
                target_meal['Food'] += f" + {best_snack['meal_name']}"
                target_meal['Source_Meal_Time'] += " + Snack" # Update badge
                
                # Sum Macros
                target_meal['Cal_Num'] += best_snack['Calories']
                target_meal['Prot_Num'] += best_snack['Proteins']
                target_meal['Carb_Num'] += best_snack['Carbs']
                target_meal['Fat_Num'] += best_snack['Fats']
                
                # Re-sort meals back to Breakfast-Lunch-Dinner order for display
                order = {"Breakfast": 0, "Lunch": 1, "Dinner": 2}
                meals.sort(key=lambda x: order.get(x['Meal'], 3))

    # 5. Final Formatting (Convert numbers to strings for UI)
    final_output = []
    for m in meals:
        final_output.append({
            "Meal": m['Meal'],
            "Food": m['Food'],
            "Source_Meal_Time": m['Source_Meal_Time'],
            "Target Calories": f"{m['Target_Cal_Num']:.0f}",
            "Actual Calories": f"{m['Cal_Num']:.0f}",
            "Proteins": f"{m['Prot_Num']:.1f}g",
            "Carbs": f"{m['Carb_Num']:.1f}g",
            "Fats": f"{m['Fat_Num']:.1f}g",
            "Calories": f"{m['Cal_Num']:.0f} kcal",
            "Cluster_Name": m['Cluster_Name']
        })

    return final_output
    
    


# --- 6. Main Page ---

# 1. HERO SECTION (Top of Page)
# Using a wide banner image that implies fitness AND food

st.markdown("""
<style>
/* 1. REMOVE DEFAULT STREAMLIT PADDING so image touches the top */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
}



.hero-image {
    width: 100%;
    height: 500px;          /* Taller height for better impact */
    object-fit: cover;      /* Ensure image covers the area */
    
    
    /* FADE TO BLACK GRADIENT */
    -webkit-mask-image: linear-gradient(to bottom, black 50%, transparent 100%);
    mask-image: linear-gradient(to bottom, black 50%, transparent 100%);
}
</style>


<div style="width:100vw; margin-left:-50vw; left:50%; position:relative;">
<img src="https://images.unsplash.com/photo-1534438327276-14e5300c3a48"
style="width:100%; height:500px; object-fit:cover;
-webkit-mask-image: linear-gradient(to bottom, black 60%, transparent 100%);">
</div>
""", unsafe_allow_html=True)


# 2. INPUT FORM (Below Hero)
# --- 6. Main Page ---

# We create 3 columns: Left (Spacer), Middle (Content), Right (Spacer)
# The ratio [1, 2, 1] means the middle takes up 50% of the screen width.
spacer_left, col_center, spacer_right = st.columns([1, 2, 1])

with col_center:
    
    # Title (Pushed up to overlap the image)
    st.markdown("""
    <div style="text-align: center; margin-top: -120px; position: relative; z-index: 2;">
        <div style="font-family: 'Open Sans', sans-serif; color: #3b82f6; font-weight: 700; letter-spacing: 4px; margin-bottom: 0.5rem; text-shadow: 0 2px 4px rgba(0,0,0,0.8);">
            ADVANCED ATHLETIC NUTRITION
        </div>
        <h1 style="font-family: 'Oswald', sans-serif; font-size: 4rem; margin: 0; color: white; text-transform: uppercase; text-shadow: 0 4px 10px rgba(0,0,0,1);">
            Optimize Your Fuel
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # The Dashboard Card (Centered)
    st.markdown('<div class="profile-card" style="margin-top: 2rem;">', unsafe_allow_html=True)
    st.markdown('<div class="form-header">⚙️ ATHLETE DATA</div>', unsafe_allow_html=True)

    # Row 1: Age, Gender, Height
    c1, c2, c3 = st.columns(3)
    with c1: age = st.number_input("Age", 15, 80, 25)
    with c2: gender = st.selectbox("Gender", ("Male", "Female"))
    with c3: height_cm = st.number_input("Height (cm)", 140, 220, 170)

    # Row 2: Weight, Fat, BPM
    c4, c5, c6 = st.columns(3)
    with c4: weight_kg = st.number_input("Weight (kg)", 40, 150, 70)
    with c5: fat_percentage = st.number_input("Body Fat %", 5.0, 50.0, 18.0, step=0.1)
    with c6: avg_bpm = st.number_input("Resting BPM", 40, 120, 60, step=1)

    st.markdown("---") # Divider

    # Row 3: Goals & Activity
    goals = ["Not Sure - Let AI Suggest", "Build Muscle", "Endurance", "HIIT", "Weight Loss", "General / Balanced"]
    goal = st.selectbox("Performance Goal", goals)
    
    activity = st.selectbox("Training Frequency", (
        "Light Training (1–3 days/week)", "Moderate Training (3–5 days/week)", "High Training (6–7 days/week)"
    ), index=1)

    # Row 4: Calories
    calorie_choices = ["Match TDEE", "TDEE - 200 kcal", "TDEE - 300 kcal", "Manual target"]
    calorie_strategy = st.selectbox("Calorie Protocol", calorie_choices)

    if calorie_strategy == "Manual target":
        manual_calorie = st.number_input("Target Calories", 1000, 5000, 2000, step=50)
    else:
        manual_calorie = None

    # AI Suggestion Logic
    if goal == "Not Sure - Let AI Suggest":
        bmi_calc = weight_kg / ((height_cm / 100) ** 2)
        suggested_goal = predict_goal_for_user(age, bmi_calc, activity)
        
        st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid #3b82f6; padding: 10px; border-radius: 4px; margin-top: 15px; text-align: center;">
            <small style="color: #3b82f6; font-weight: bold;">AI RECOMMENDS:</small> <span style="color: white; font-weight: bold;">{suggested_goal.upper()}</span>
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
        # 1. Calculations
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        bmr = calculate_bmr(age, weight_kg, height_cm, gender)
        tdee = calculate_tdee(bmr, activity)
        exp_level = {"Light Training (1–3 days/week)": 2, "Moderate Training (3–5 days/week)": 4, "High Training (6–7 days/week)": 6}[activity]

        # Check if user selected "Not Sure - Let AI Suggest"
        use_rf_prediction = st.session_state.get('use_rf_prediction', False)
        
        # 2. Define User Data (Input for Classifier) - with all required features
        if use_rf_prediction:
            # For AI prediction, pass neutral goal to let classifier decide
            user_data = {
                'Age': age, 'Weight (kg)': weight_kg, 'Height (m)': height_m, 'BMI': bmi,
                'Fat_Percentage': fat_percentage,'Avg_BPM': avg_bpm,  'Resting_BPM': 65, 
                'Experience_Level': exp_level, 'Gender': gender, 'Goals': 'Balanced',
                'Calories_Burned': tdee, 'Workout_Frequency (days/week)': exp_level,
                'protein_per_kg': 1.6
            }
        else:
            # For user-selected goal
            user_data = {
                'Age': age, 'Weight (kg)': weight_kg, 'Height (m)': height_m, 'BMI': bmi,
                'Fat_Percentage': fat_percentage, 'Avg_BPM': 130, 'Resting_BPM': 65, 
                'Experience_Level': exp_level, 'Gender': gender, 'Goals': map_goal_to_model_format(goal),
                'Calories_Burned': tdee, 'Workout_Frequency (days/week)': exp_level,
                'protein_per_kg': 1.6
            }

        # 3. PREDICTION: Use Random Forest Classifier to predict Meal Plan
        user_df_input = pd.DataFrame([user_data])
        
        try:
            if use_rf_prediction:
                # Use Random Forest Classifier for prediction
                predicted_meal_plan = classifier_model.predict(user_df_input)[0]
                proba_array = classifier_model.predict_proba(user_df_input)[0]
                confidence_proba = proba_array.max()
                
                # Create probability dictionary for display
                classifier_classes = classifier_model.named_steps['model'].classes_
                proba_dict = dict(zip(classifier_classes, proba_array))
                st.session_state['classifier_proba'] = proba_dict
                
                st.info("✨ Using Random Forest Classifier to predict your optimal meal plan!")
            else:
                # Use user-selected goal
                predicted_meal_plan = map_goal_to_model_format(goal)
                proba_array = classifier_model.predict_proba(user_df_input)[0]
                confidence_proba = proba_array.max()
                
                # Create probability dictionary
                classifier_classes = classifier_model.named_steps['model'].classes_
                proba_dict = dict(zip(classifier_classes, proba_array))
                st.session_state['classifier_proba'] = proba_dict
        except Exception as e:
            st.warning(f"Could not predict with classifier: {e}. Using user-selected goal.")
            predicted_meal_plan = map_goal_to_model_format(goal)
            confidence_proba = 0.0
            st.session_state['classifier_proba'] = {}
        
        # Use predicted meal plan from classifier
        final_plan = predicted_meal_plan
        
        # Store training profile for display
        training_level_map = {
            "Light Training (1–3 days/week)": {"name": "Light", "intensity": 1},
            "Moderate Training (3–5 days/week)": {"name": "Moderate", "intensity": 2},
            "High Training (6–7 days/week)": {"name": "High", "intensity": 3}
        }
        training_profile = training_level_map.get(activity, training_level_map["Moderate Training (3–5 days/week)"])
        st.session_state['training_intensity'] = training_profile['name']
        st.session_state['training_intensity_level'] = training_profile['intensity'] 

        # Decide calorie target based on chosen strategy
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

        # Generate Meals (pass calorie_target)
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
    
    # User Metrics with updated color and styling
    st.markdown("---")
    st.markdown("<h3 style='color: #0f172a; margin-bottom: 1.5rem;'>📊 Your Analysis</h3>", unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3 = st.columns(3, gap="large")
    with metric_col1:
        # Determine BMI category and color
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
    
    # Display Training Impact Info
    training_intensity = st.session_state.get('training_intensity', 'Unknown')
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                border-left: 4px solid #f59e0b; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0;'>
        <p style='color: #92400e; margin: 0; font-weight: 600;'>⚡ <strong>Training Intensity Analysis</strong></p>
        <p style='color: #92400e; margin: 0.5rem 0 0 0;'>Your <strong>{training_intensity} Training</strong> frequency is optimized for this plan. Higher frequency training requires more carbs & protein for recovery.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display Protein Prediction
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #f0fdf4 0%, #e7f5e4 100%); 
                border-left: 4px solid #22c55e; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0;'>
        <p style='color: #166534; margin: 0; font-weight: 600;'>🎯 <strong>Meal Plan Prediction</strong></p>
        <p style='color: #166534; margin: 0.5rem 0 0 0;'>AI predicted: <strong>{st.session_state['predicted_plan']}</strong> 
        (Confidence: <strong>{st.session_state.get("classifier_confidence", 0)*100:.1f}%</strong>)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add Expander to show classifier prediction breakdown
    with st.expander("📊 View Classifier Probability Breakdown"):
        st.markdown("**How the AI categorized your profile:**")
        
        # Get classifier probabilities if available
        if 'classifier_proba' in st.session_state:
            proba_dict = st.session_state['classifier_proba']
            
            # Create a visual bar chart of probabilities
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
    
    # Add expander showing confidence factors
    with st.expander("🎯 Confidence Factors - How to Improve"):
        st.markdown("""
        **Your classifier confidence depends on how clearly your profile matches one category:**
        
        **Key Factors (in order of importance):**
        
        1. **Fat Percentage** (Most Important)
           - High fat (>25%) → Strong "Weight Loss" signal
           - Low fat (<15%) → Strong "Build Muscle" signal
        
        2. **BMI (Body Mass Index)**
           - Overweight/Obese (>27) → "Weight Loss"
           - Athletic (<23) → "Build Muscle" or "HIIT"
        
        3. **Experience Level**
           - Beginners (1-2) → "Endurance"
           - Advanced (5-6) → "Build Muscle" or "HIIT"
        
        4. **Training Frequency**
           - High (6-7 days/week) → "Build Muscle" or "HIIT"
           - Low (1-3 days/week) → "Balanced" or "Endurance"
        
        5. **Age**
           - 40+ years → "Endurance"
           - Young + Athletic → "Build Muscle"
        
        **Tips to Get Higher Confidence:**
        - ✅ Make sure your profile factors align (e.g., low fat + high training = strong signal)
        - ✅ More extreme values (very fit or needs weight loss) = higher confidence
        - ✅ Mixed signals (medium values across all factors) = lower confidence
        """)

    # Meal Plan Display (Card View)
    # Meal Plan Display (Card View)
    # Meal Plan Display (Card View)
    if not st.session_state['meal_plan'].empty:
        st.markdown("<h3 style='color: #0f172a; margin: 2rem 0 1rem 0;'>🍽️ Your Daily Meal Plan</h3>", unsafe_allow_html=True)
        
        plan = st.session_state['predicted_plan']
        
        # --- ✅ FIX: Define focus_text here before using it ---
        focus_text = ""
        if plan == "Endurance": focus_text = "⚡ **Focus:** High Carbohydrates for sustained energy."
        elif plan == "Weight Loss": focus_text = "🔥 **Focus:** High Protein, Low Calorie density for fat loss."
        elif plan == "Build Muscle": focus_text = "💪 **Focus:** Hyper-personalized protein intake with balanced fats/carbs."
        elif plan == "HIIT": focus_text = "🏃 **Focus:** Moderate Carbs & Protein for explosive energy."
        elif plan == "Balanced": focus_text = "⚖️ **Focus:** Personalized protein goals with even macro distribution."
        
        # Now it is safe to use
        st.info(focus_text)
        
        meal_emojis = {"Breakfast": "🍳", "Lunch": "🥗", "Dinner": "🍽️"}
        
        for index, row in st.session_state['meal_plan'].iterrows():
            with st.container():
                # Define color for the source tag based on time
                tag_color = "#e2e8f0" # gray default
                text_color = "#475569"
                # Use .get() to avoid errors if 'Source_Meal_Time' is missing in old data
                source_time = row.get('Source_Meal_Time', 'General')
                
                if source_time == 'Breakfast': 
                    tag_color = "#fef3c7"; text_color = "#d97706" # yellow
                elif source_time == 'Lunch': 
                    tag_color = "#dcfce7"; text_color = "#166534" # green
                elif source_time == 'Dinner': 
                    tag_color = "#dbeafe"; text_color = "#1e40af" # blue
                elif source_time == 'Snack':
                    tag_color = "#f3e8ff"; text_color = "#7e22ce" # purple

                st.markdown(f"""
                <div style="background-color: white; padding: 1.5rem; border-radius: 12px; 
                            border: 1px solid rgba(60,64,67,0.12); box-shadow: 0 2px 6px rgba(60,64,67,0.05); margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div class="meal-header" style="margin-bottom: 0;">{meal_emojis.get(row['Meal'], '🍱')} {row['Meal']}</div>
                        <span style="background-color: {tag_color}; color: {text_color}; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
                            {source_time}
                        </span>
                    </div>
                    <div class="food-name" style="margin-bottom: 0.2rem;">{row['Food']}</div>
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
        # Extract individual food names from combined meals (e.g., "Food1 + Food2")
        exclude_foods = []
        for food_str in current_foods:
            if ' + ' in food_str:
                # Split combined foods
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

    # Show all available meals for this meal plan
    # Show all available meals for this meal plan
    with st.expander("📋 View All Available Foods for This Meal Plan"):
        plan = st.session_state['predicted_plan']
        plan_foods = food_db[food_db['Meal_Plan'] == plan]
        
        if plan_foods.empty:
            st.info(f"No foods found for {plan} meal plan.")
        else:
            st.markdown(f"**Total Available Foods: {len(plan_foods)}**")
            
            # --- ✅ FIX: Add Meal Time Column Logic ---
            # Check if 'Meal_Time' exists (from training script) or 'Meal_Type'
            time_col = 'Meal_Time' if 'Meal_Time' in plan_foods.columns else 'Meal_Type' if 'Meal_Type' in plan_foods.columns else None
            
            # define columns to display
            cols_to_show = ['meal_name']
            col_headers = ['Food Name']
            
            # If we found the time column, add it to the list
            if time_col:
                cols_to_show.append(time_col)
                col_headers.append('Meal Time')

            # Add the rest of the nutrition columns
            cols_to_show.extend(['Calories', 'Proteins', 'Carbs', 'Fats'])
            col_headers.extend(['Calories (kcal)', 'Protein (g)', 'Carbs (g)', 'Fats (g)'])
            
            # Create the dataframe for display
            display_df = plan_foods[cols_to_show].copy()
            display_df.columns = col_headers
            
            # Format the numbers to look nice
            display_df['Calories (kcal)'] = display_df['Calories (kcal)'].apply(lambda x: f"{float(x):.0f}" if pd.notna(x) else "N/A")
            display_df['Protein (g)'] = display_df['Protein (g)'].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "N/A")
            display_df['Carbs (g)'] = display_df['Carbs (g)'].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "N/A")
            display_df['Fats (g)'] = display_df['Fats (g)'].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "N/A")
            
            # --- ✅ Bonus: Add a Filter Dropdown ---
            if time_col:
                # Create a list of unique times (e.g., Breakfast, Lunch...)
                categories = ['Show All'] + sorted(plan_foods[time_col].unique().tolist())
                selected_cat = st.selectbox("Filter list by:", categories)
                
                if selected_cat != 'Show All':
                    display_df = display_df[display_df['Meal Time'] == selected_cat]

            st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)


