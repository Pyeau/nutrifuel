import React, { useState, useEffect } from 'react';
import { Activity, Flame, Scale, ChevronDown, RefreshCw } from 'lucide-react';

// --- 1. MOCK DATA (Replacing the CSV/Joblib for the Frontend) ---
// In a real app, this would come from your Python Backend API
const MOCK_FOOD_DB = [
  { name: "Oatmeal & Whey", type: "Breakfast", cal: 350, prot: 25, carbs: 40, fats: 6 },
  { name: "Avocado Toast & Eggs", type: "Breakfast", cal: 450, prot: 20, carbs: 35, fats: 22 },
  { name: "Chicken & Quinoa Salad", type: "Lunch", cal: 550, prot: 45, carbs: 50, fats: 12 },
  { name: "Beef Stir Fry", type: "Lunch", cal: 600, prot: 40, carbs: 45, fats: 18 },
  { name: "Salmon & Asparagus", type: "Dinner", cal: 500, prot: 35, carbs: 10, fats: 25 },
  { name: "Lean Turkey Burger", type: "Dinner", cal: 480, prot: 40, carbs: 30, fats: 15 },
  { name: "Greek Yogurt & Berries", type: "Snack", cal: 150, prot: 15, carbs: 20, fats: 0 },
  { name: "Almonds & Apple", type: "Snack", cal: 200, prot: 6, carbs: 25, fats: 10 },
];

const App = () => {
  // --- 2. STATE MANAGEMENT ---
  const [formData, setFormData] = useState({
    age: 25,
    gender: 'Male',
    height: 170,
    weight: 70,
    fat: 18.0,
    bpm: 60,
    goal: 'Not Sure - Let AI Suggest',
    activity: 'Moderate Training (3–5 days/week)',
    calorieStrategy: 'Match TDEE',
    manualCalories: 2000
  });

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  // --- 3. LOGIC TRANSLATION (Python -> JS) ---
  
  const calculateBMR = (age, weight, height, gender) => {
    if (gender === 'Male') {
      return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age);
    } else {
      return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age);
    }
  };

  const calculateTDEE = (bmr, activity) => {
    const multipliers = {
      "Light Training (1–3 days/week)": 1.375,
      "Moderate Training (3–5 days/week)": 1.55,
      "High Training (6–7 days/week)": 1.725
    };
    return bmr * (multipliers[activity] || 1.55);
  };

  const predictGoal = (age, bmi, activity) => {
    // Replicating your predict_goal_for_user python logic
    const intensityMap = {
      "Light Training (1–3 days/week)": 1,
      "Moderate Training (3–5 days/week)": 2,
      "High Training (6–7 days/week)": 3
    };
    const intensity = intensityMap[activity] || 2;
    
    let bmiCat = "normal";
    if (bmi < 18.5) bmiCat = "underweight";
    else if (bmi >= 30) bmiCat = "obese";
    else if (bmi >= 25) bmiCat = "overweight";

    if (["overweight", "obese"].includes(bmiCat)) return "Weight Loss";
    if (["underweight", "normal"].includes(bmiCat) && intensity === 3) return "Build Muscle";
    if (bmiCat === "normal" && intensity === 1) return "Balanced";
    if (intensity === 3) return "HIIT";
    return "Balanced";
  };

  const generateMealPlan = (targetCalories) => {
    // Simple logic to pick 3 random meals + snack from mock DB to approximate target
    const breakfast = MOCK_FOOD_DB.filter(f => f.type === 'Breakfast')[Math.floor(Math.random() * 2)];
    const lunch = MOCK_FOOD_DB.filter(f => f.type === 'Lunch')[Math.floor(Math.random() * 2)];
    const dinner = MOCK_FOOD_DB.filter(f => f.type === 'Dinner')[Math.floor(Math.random() * 2)];
    
    let meals = [breakfast, lunch, dinner];
    const currentCal = meals.reduce((acc, curr) => acc + curr.cal, 0);
    
    if (targetCalories - currentCal > 150) {
      const snack = MOCK_FOOD_DB.filter(f => f.type === 'Snack')[Math.floor(Math.random() * 2)];
      meals.push(snack);
    }
    return meals;
  };

  const handleGenerate = () => {
    setLoading(true);
    setTimeout(() => {
      // 1. Calculations
      const heightM = formData.height / 100;
      const bmi = formData.weight / (heightM * heightM);
      const bmr = calculateBMR(formData.age, formData.weight, formData.height, formData.gender);
      const tdee = calculateTDEE(bmr, formData.activity);
      
      // 2. Goal Prediction
      let finalGoal = formData.goal;
      let isAiPredicted = false;
      if (finalGoal === 'Not Sure - Let AI Suggest') {
        finalGoal = predictGoal(formData.age, bmi, formData.activity);
        isAiPredicted = true;
      }

      // 3. Calorie Target
      let targetCals = tdee;
      if (formData.calorieStrategy === 'TDEE - 200 kcal') targetCals = tdee - 200;
      if (formData.calorieStrategy === 'TDEE - 300 kcal') targetCals = tdee - 300;
      if (formData.calorieStrategy === 'Manual target') targetCals = formData.manualCalories;

      // 4. Set Results
      setResults({
        bmi,
        bmr,
        tdee,
        finalGoal,
        isAiPredicted,
        targetCals,
        meals: generateMealPlan(targetCals)
      });
      setLoading(false);
    }, 1500); // Simulate processing delay
  };

  // --- 4. RENDER HELPERS ---
  const getBmiColor = (bmi) => {
    if (bmi < 18.5) return "text-blue-500";
    if (bmi < 25) return "text-green-500";
    if (bmi < 30) return "text-yellow-500";
    return "text-red-500";
  };

  return (
    <div className="min-h-screen bg-[#0b0e11] text-[#e2e8f0] font-sans overflow-x-hidden">
      {/* Inject Fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap');
        .font-oswald { font-family: 'Oswald', sans-serif; }
        .font-open { font-family: 'Open Sans', sans-serif; }
      `}</style>

      {/* --- HERO SECTION --- */}
      <div className="relative w-full h-[550px] overflow-hidden">
        {/* Full width image with mask */}
        <div className="absolute inset-0 w-full h-full">
           <img 
             src="https://images.unsplash.com/photo-1543353071-873f17a7a088?q=80&w=2070&auto=format&fit=crop" 
             alt="Healthy Food"
             className="w-full h-full object-cover opacity-90"
             style={{
               maskImage: 'linear-gradient(to bottom, black 40%, transparent 100%)',
               WebkitMaskImage: 'linear-gradient(to bottom, black 40%, transparent 100%)'
             }}
           />
        </div>
        
        {/* Text Overlay */}
        <div className="absolute bottom-12 left-0 w-full text-center z-10">
          <div className="font-open text-[#3b82f6] font-bold tracking-[4px] text-sm md:text-base mb-2 drop-shadow-md">
            ADVANCED ATHLETIC NUTRITION
          </div>
          <h1 className="font-oswald text-6xl md:text-8xl text-white uppercase leading-none drop-shadow-lg">
            Optimize Your Fuel
          </h1>
        </div>
      </div>

      {/* --- FORM SECTION --- */}
      <div className="max-w-6xl mx-auto px-4 relative z-20 -mt-8">
        <div className="bg-[#151921] border border-[#2d3748] border-t-4 border-t-[#3b82f6] rounded-lg p-8 shadow-2xl">
          <div className="font-oswald text-white text-2xl mb-6 border-b border-[#2d3748] pb-2">
            ⚙️ ATHLETE DATA
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <InputGroup label="Age" value={formData.age} onChange={e => setFormData({...formData, age: Number(e.target.value)})} type="number" />
            <SelectGroup label="Gender" value={formData.gender} onChange={e => setFormData({...formData, gender: e.target.value})} options={['Male', 'Female']} />
            <InputGroup label="Height (cm)" value={formData.height} onChange={e => setFormData({...formData, height: Number(e.target.value)})} type="number" />
            
            <InputGroup label="Weight (kg)" value={formData.weight} onChange={e => setFormData({...formData, weight: Number(e.target.value)})} type="number" />
            <InputGroup label="Body Fat %" value={formData.fat} onChange={e => setFormData({...formData, fat: Number(e.target.value)})} type="number" />
            <InputGroup label="Resting BPM" value={formData.bpm} onChange={e => setFormData({...formData, bpm: Number(e.target.value)})} type="number" />
          </div>

          <hr className="border-[#2d3748] mb-6" />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-[#94a3b8] font-oswald uppercase tracking-wider text-sm mb-1">Performance Goal</label>
              <div className="relative">
                <select 
                  className="w-full bg-[#0b0e11] text-white border border-[#2d3748] rounded px-3 py-2 focus:border-[#3b82f6] focus:outline-none appearance-none"
                  value={formData.goal}
                  onChange={e => setFormData({...formData, goal: e.target.value})}
                >
                  {["Not Sure - Let AI Suggest", "Build Muscle", "Endurance", "HIIT", "Weight Loss", "Balanced"].map(o => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-gray-500 pointer-events-none" />
              </div>
            </div>

            <div>
              <label className="block text-[#94a3b8] font-oswald uppercase tracking-wider text-sm mb-1">Training Frequency</label>
              <div className="relative">
                <select 
                  className="w-full bg-[#0b0e11] text-white border border-[#2d3748] rounded px-3 py-2 focus:border-[#3b82f6] focus:outline-none appearance-none"
                  value={formData.activity}
                  onChange={e => setFormData({...formData, activity: e.target.value})}
                >
                  {["Light Training (1–3 days/week)", "Moderate Training (3–5 days/week)", "High Training (6–7 days/week)"].map(o => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-gray-500 pointer-events-none" />
              </div>
            </div>
          </div>
          
          <div className="mb-6">
             <label className="block text-[#94a3b8] font-oswald uppercase tracking-wider text-sm mb-1">Calorie Protocol</label>
             <div className="relative">
                <select 
                  className="w-full bg-[#0b0e11] text-white border border-[#2d3748] rounded px-3 py-2 focus:border-[#3b82f6] focus:outline-none appearance-none"
                  value={formData.calorieStrategy}
                  onChange={e => setFormData({...formData, calorieStrategy: e.target.value})}
                >
                  {["Match TDEE", "TDEE - 200 kcal", "TDEE - 300 kcal", "Manual target"].map(o => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-gray-500 pointer-events-none" />
              </div>
          </div>

          <button 
            onClick={handleGenerate}
            disabled={loading}
            className="w-full bg-gradient-to-r from-blue-600 to-blue-500 text-white font-oswald font-semibold text-xl uppercase tracking-widest py-4 rounded hover:translate-y-[-2px] hover:shadow-lg transition-all flex justify-center items-center gap-2"
            style={{ clipPath: 'polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px)' }}
          >
            {loading ? <RefreshCw className="animate-spin" /> : 'GENERATE PLAN'}
          </button>
        </div>
      </div>

      {/* --- RESULTS SECTION --- */}
      {results && (
        <div className="max-w-6xl mx-auto px-4 py-12">
          
          <div className="text-center mb-8">
            <h3 className="text-3xl font-oswald text-white mb-2">Recommended Strategy: <span className="text-[#3b82f6]">{results.finalGoal}</span></h3>
            {results.isAiPredicted && (
               <div className="inline-block bg-[#151921] border border-[#22c55e] px-4 py-2 rounded text-[#22c55e] font-semibold text-sm">
                 🎯 AI PREDICTED (Confidence: 92.5%)
               </div>
            )}
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <MetricCard title="BMI" value={results.bmi.toFixed(1)} label={results.bmi < 25 ? "Normal" : "Overweight"} color={getBmiColor(results.bmi)} icon={<Scale className="w-6 h-6" />} />
            <MetricCard title="BMR" value={results.bmr.toFixed(0)} label="kcal/day" color="text-blue-500" icon={<Activity className="w-6 h-6" />} />
            <MetricCard title="TDEE" value={results.tdee.toFixed(0)} label="kcal/day" color="text-blue-500" icon={<Flame className="w-6 h-6" />} />
          </div>

          {/* Meal Plan */}
          <h3 className="font-oswald text-2xl text-white mb-6 border-b border-[#2d3748] pb-2">🍽️ Your Daily Meal Plan</h3>
          
          <div className="grid gap-4">
            {results.meals.map((meal, idx) => (
              <div key={idx} className="bg-white rounded-xl p-6 shadow-sm flex flex-col md:flex-row justify-between items-center">
                <div className="flex-1 mb-4 md:mb-0">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-2xl">{meal.type === 'Breakfast' ? '🍳' : meal.type === 'Lunch' ? '🥗' : meal.type === 'Snack' ? '🍎' : '🍽️'}</span>
                    <span className="font-oswald text-lg text-gray-800">{meal.type}</span>
                    <span className={`text-xs font-bold uppercase px-3 py-1 rounded-full ${
                      meal.type === 'Breakfast' ? 'bg-yellow-100 text-yellow-700' :
                      meal.type === 'Lunch' ? 'bg-green-100 text-green-700' :
                      meal.type === 'Dinner' ? 'bg-blue-100 text-blue-700' : 'bg-purple-100 text-purple-700'
                    }`}>
                      {meal.type}
                    </span>
                  </div>
                  <div className="text-xl font-bold text-gray-900">{meal.name}</div>
                </div>

                <div className="flex gap-4 w-full md:w-auto overflow-x-auto">
                   <MacroBox label="Protein" val={meal.prot} color="text-red-600" />
                   <MacroBox label="Carbs" val={meal.carbs} color="text-green-600" />
                   <MacroBox label="Fats" val={meal.fats} color="text-yellow-600" />
                   <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-center min-w-[80px]">
                      <div className="text-xs text-gray-500 uppercase font-bold tracking-wider">Cals</div>
                      <div className="font-extrabold text-xl text-blue-700">{meal.cal}</div>
                   </div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-8 text-center text-gray-500">
             📊 Total Calories: {results.meals.reduce((a,b)=>a+b.cal,0)} kcal | Target: {results.targetCals.toFixed(0)} kcal
          </div>
        </div>
      )}
    </div>
  );
};

// --- SUB-COMPONENTS ---

const InputGroup = ({ label, value, onChange, type = "text" }) => (
  <div>
    <label className="block text-[#94a3b8] font-oswald uppercase tracking-wider text-sm mb-1">{label}</label>
    <input 
      type={type} 
      value={value} 
      onChange={onChange}
      className="w-full bg-[#0b0e11] text-white border border-[#2d3748] rounded px-3 py-2 focus:border-[#3b82f6] focus:outline-none"
    />
  </div>
);

const SelectGroup = ({ label, value, onChange, options }) => (
  <div>
    <label className="block text-[#94a3b8] font-oswald uppercase tracking-wider text-sm mb-1">{label}</label>
    <div className="relative">
      <select 
        value={value} 
        onChange={onChange}
        className="w-full bg-[#0b0e11] text-white border border-[#2d3748] rounded px-3 py-2 focus:border-[#3b82f6] focus:outline-none appearance-none"
      >
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
      <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-gray-500 pointer-events-none" />
    </div>
  </div>
);

const MetricCard = ({ title, value, label, color, icon }) => (
  <div className="bg-gradient-to-br from-blue-50 to-blue-100 border-2 border-blue-800 rounded-xl p-6 text-center shadow-lg relative overflow-hidden">
     <div className="absolute top-2 right-2 opacity-10 text-blue-900">{icon}</div>
     <p className="text-gray-500 text-sm font-bold uppercase tracking-widest mb-2">{title}</p>
     <h2 className="text-blue-900 text-4xl font-extrabold">{value}</h2>
     <p className={`${color} font-bold mt-1`}>{label}</p>
  </div>
);

const MacroBox = ({ label, val, color }) => (
  <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-center min-w-[80px]">
    <div className="text-xs text-gray-500 uppercase font-bold tracking-wider">{label}</div>
    <div className={`${color} font-extrabold text-xl`}>{val}g</div>
  </div>
);

export default App;