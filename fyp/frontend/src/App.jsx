import React, { useState } from 'react';
import {
  Activity, Flame, Scale, ChevronDown, RefreshCw, Utensils, Zap, Target,
  Award, Dumbbell, Leaf, Timer, Sparkles, Heart, Apple, ChefHat, Droplet,
  Sun, Moon, Calendar, User, ArrowRight, Info, TrendingUp, CheckCircle2, Coffee, Replace
} from 'lucide-react';

const App = () => {
  // --- STATE MANAGEMENT ---
  const [formData, setFormData] = useState({
    age: 25,
    gender: 'Male',
    height: 170,
    weight: 70,
    fat: 15.0,
    bpm: 60,
    goal: 'Build Muscle',
    activity: 'Moderate Training (3–5 days/week)',
    calorieStrategy: 'Match TDEE',
    manualCalories: 2000
  });

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [replacingMealIndex, setReplacingMealIndex] = useState(null);
  const [replacementLoading, setReplacementLoading] = useState(false);

  const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:5000';

  // --- API HANDLER (Connected to Flask) ---
  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      // Connect to your local Python Flask server
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResults(data);

      // Scroll to results section on success
      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);

    } catch (err) {
      console.error("Connection Error:", err);
      setError("Could not connect to the Flask backend. Ensure 'app.py' is running on port 5000.");
    } finally {
      setLoading(false);
    }
  };

  // --- FOOD REPLACEMENT HANDLER ---
  const handleReplaceFood = async (mealIndex, newFoodName) => {
    if (!results || replacementLoading) return;

    setReplacementLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/replace-food`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mealIndex: mealIndex,
          newFoodName: newFoodName,
          currentMeals: results.meals,
          goal: results.finalGoal
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        // Update results with the new meals and totals
        setResults({
          ...results,
          meals: data.updatedMeals,
          targetCals: data.totals.calories
        });
        setReplacingMealIndex(null);
        alert(`✅ Successfully replaced with ${newFoodName}!`);
      }
    } catch (err) {
      console.error("Replacement Error:", err);
      alert(`❌ Could not replace food: ${err.message}`);
    } finally {
      setReplacementLoading(false);
    }
  };

  const getBmiColor = (bmi) => {
    if (bmi < 18.5) return "text-blue-500";
    if (bmi < 25) return "text-emerald-600";
    if (bmi < 30) return "text-orange-500";
    return "text-red-500";
  };

  const getBmiStatus = (bmi) => {
    if (bmi < 18.5) return "Underweight";
    if (bmi < 25) return "Healthy Weight";
    if (bmi < 30) return "Overweight";
    return "Obese";
  };

  // --- HELPER: Meal Card Styling (Fresh Food Theme) ---
  const getMealStyle = (type) => {
    switch (type) {
      case 'Breakfast': return {
        bg: 'bg-orange-50',
        border: 'border-orange-100',
        iconColor: 'bg-orange-100 text-orange-600',
        icon: <Sun className="w-8 h-8" />,
        label: 'text-orange-800'
      };
      case 'Lunch': return {
        bg: 'bg-emerald-50',
        border: 'border-emerald-100',
        iconColor: 'bg-emerald-100 text-emerald-600',
        icon: <Leaf className="w-8 h-8" />,
        label: 'text-emerald-800'
      };
      case 'Dinner': return {
        bg: 'bg-stone-50',
        border: 'border-stone-200',
        iconColor: 'bg-stone-200 text-stone-600',
        icon: <Moon className="w-8 h-8" />,
        label: 'text-stone-800'
      };
      default: return {
        bg: 'bg-yellow-50',
        border: 'border-yellow-100',
        iconColor: 'bg-yellow-100 text-yellow-600',
        icon: <Coffee className="w-8 h-8" />,
        label: 'text-yellow-800'
      };
    }
  };

  return (
    <div className="min-h-screen bg-[#FDFBF7] text-slate-800 font-sans selection:bg-orange-200 selection:text-orange-900 overflow-x-hidden">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        * { font-family: 'Plus Jakarta Sans', sans-serif; }
        
        /* --- ANIMATION KEYFRAMES --- */
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(40px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes popIn {
          0% { opacity: 0; transform: scale(0.5); }
          70% { transform: scale(1.05); }
          100% { opacity: 1; transform: scale(1); }
        }

        @keyframes float {
          0% { transform: translateY(0px); }
          50% { transform: translateY(-15px); }
          100% { transform: translateY(0px); }
        }

        @keyframes wiggle {
            0%, 100% { transform: rotate(0deg); }
            25% { transform: rotate(-10deg); }
            75% { transform: rotate(10deg); }
        }

        @keyframes shimmer {
          0% { transform: translateX(-150%) skewX(-15deg); }
          100% { transform: translateX(150%) skewX(-15deg); }
        }

        @keyframes gradient-text {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* --- ANIMATION CLASSES --- */
        .animate-fade-in { animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
        .animate-slide-up { animation: slideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards; opacity: 0; }
        .animate-pop-in { animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; opacity: 0; }
        .animate-float { animation: float 6s ease-in-out infinite; }
        .animate-wiggle:hover { animation: wiggle 0.5s ease-in-out; }
        
        .animate-gradient-text {
            background-size: 200% auto;
            animation: gradient-text 4s linear infinite;
        }

        /* --- UI COMPONENTS --- */
        .food-card {
          background: #ffffff;
          border-radius: 24px;
          border: 1px solid #e7e5e4;
          box-shadow: 0 10px 40px -10px rgba(0,0,0,0.05);
          transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .food-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.1);
        }

        .food-input {
            background-color: #ffffff;
            border: 2px solid #f5f5f4;
            color: #1c1917;
            border-radius: 12px;
            padding-left: 3rem; /* Space for icon */
            transition: all 0.2s;
        }
        .food-input:focus {
            border-color: #f97316;
            background-color: #fff7ed;
            outline: none;
            box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.1);
        }

        .shimmer-wrapper {
            position: relative;
            overflow: hidden;
        }
        .shimmer-wrapper::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(to right, transparent, rgba(255,255,255,0.4), transparent);
            transform: translateX(-150%);
        }
        .shimmer-wrapper:hover::after {
            animation: shimmer 1s;
        }
      `}</style>

      {/* --- BACKGROUND ACCENTS --- */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute top-[-5%] right-[-5%] w-[600px] h-[600px] bg-orange-200/40 rounded-full blur-[100px] animate-float opacity-60"></div>
        <div className="absolute bottom-[10%] left-[-10%] w-[500px] h-[500px] bg-emerald-200/40 rounded-full blur-[100px] animate-float opacity-60" style={{ animationDelay: '2s' }}></div>
        {/* Floating Icons Background */}
        <div className="absolute top-20 left-20 opacity-10 animate-float" style={{ animationDelay: '1s' }}><Apple size={64} /></div>
        <div className="absolute top-40 right-40 opacity-10 animate-float" style={{ animationDelay: '2s' }}><Coffee size={48} /></div>
        <div className="absolute bottom-40 left-1/3 opacity-10 animate-float" style={{ animationDelay: '4s' }}><Dumbbell size={56} /></div>
      </div>

      {/* --- HEADER --- */}
      <header className="sticky top-0 z-50 bg-[#FDFBF7]/80 backdrop-blur-md border-b border-stone-100">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 group cursor-pointer">
              <div className="bg-orange-500 p-2.5 rounded-xl shadow-lg shadow-orange-500/20 animate-pop-in group-hover:rotate-12 transition-transform duration-300">
                <ChefHat className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-extrabold tracking-tight text-slate-900 group-hover:text-orange-600 transition-colors">Nutri<span className="text-orange-500">Fuel</span></span>
            </div>
            <div className="hidden md:flex items-center gap-8">
              <NavIcon label="Recipes" icon={<Utensils size={14} />} />
              <NavIcon label="Tracking" icon={<TrendingUp size={14} />} />
              <NavIcon label="Community" icon={<User size={14} />} />
              <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-50 rounded-full border border-emerald-100 cursor-pointer hover:bg-emerald-100 transition-colors">
                <Leaf className="w-3 h-3 text-emerald-600 animate-wiggle" />
                <span className="text-xs font-bold text-emerald-700">Eco-Friendly</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* --- HERO SECTION --- */}
      <div className="relative z-10 pt-24 pb-20">
        <div className="max-w-7xl mx-auto px-6 text-center relative">

          <div className="absolute top-0 right-10 w-48 h-48 opacity-90 pointer-events-none hidden lg:block animate-float">
            <div className="relative w-full h-full">
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                <Apple className="w-24 h-24 text-red-500 fill-red-50 drop-shadow-lg" />
              </div>
              <div className="absolute top-0 right-0 animate-bounce" style={{ animationDuration: '3s' }}>
                <Leaf className="w-12 h-12 text-green-500 fill-green-50 drop-shadow-md" />
              </div>
              <div className="absolute bottom-4 left-4 animate-pulse">
                <Zap className="w-8 h-8 text-yellow-400 fill-yellow-100" />
              </div>
            </div>
          </div>

          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-stone-200 shadow-sm mb-8 hover:scale-105 transition-transform cursor-default animate-pop-in">
            <span className="flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="text-xs font-bold text-slate-600 tracking-wide uppercase flex items-center gap-1">
              <Sparkles size={12} className="text-yellow-500" />
              AI-Powered Nutrition System
            </span>
          </div>

          <h1 className="text-5xl md:text-8xl font-black mb-6 leading-[0.95] tracking-tight text-slate-900 animate-slide-up" style={{ animationDelay: '0.1s' }}>
            Eat Smart.<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-500 via-red-500 to-purple-600 animate-gradient-text">
              Train Harder.
            </span>
          </h1>

          <p className="text-lg md:text-xl text-slate-500 mb-12 leading-relaxed max-w-2xl mx-auto font-medium animate-slide-up" style={{ animationDelay: '0.2s' }}>
            Stop guessing your macros. Get a precision meal plan tailored to your biology, training load, and performance goals in seconds.
          </p>

          <div className="flex flex-wrap justify-center gap-4 text-sm font-bold animate-slide-up" style={{ animationDelay: '0.3s' }}>
            <button onClick={() => document.getElementById('athlete-profile-section')?.scrollIntoView({ behavior: 'smooth' })} className="bg-slate-900 text-white pl-8 pr-6 py-4 rounded-full transition-all duration-300 shadow-xl hover:bg-slate-800 hover:shadow-slate-900/20 hover:-translate-y-1 shimmer-wrapper overflow-hidden relative group flex items-center gap-2">
              Start Calibrating <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
            </button>
            <button className="bg-white text-slate-700 px-8 py-4 rounded-full border border-stone-200 transition-colors duration-300 hover:border-orange-200 hover:text-orange-600 flex items-center gap-2">
              <Info size={16} /> Learn the Science
            </button>
          </div>
        </div>
      </div>

      {/* --- HOW IT WORKS --- */}
      <div className="max-w-7xl mx-auto px-6 pb-20 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <FeatureStep
            icon={<User className="text-blue-500" />}
            title="1. Profile"
            desc="Input your biometrics and training intensity."
            delay="0.4s"
          />
          <FeatureStep
            icon={<Zap className="text-yellow-500" />}
            title="2. Analyze"
            desc="Our AI calculates your exact metabolic needs."
            delay="0.5s"
          />
          <FeatureStep
            icon={<CheckCircle2 className="text-emerald-500" />}
            title="3. Execute"
            desc="Receive a daily macro-perfect meal plan."
            delay="0.6s"
          />
        </div>
      </div>

      {/* --- INPUT SECTION --- */}
      <div id="athlete-profile-section" className="max-w-5xl mx-auto px-6 pb-24 relative z-20">
        <div className="food-card p-8 md:p-12 relative overflow-hidden animate-slide-up" style={{ animationDelay: '0.4s' }}>

          {/* Decorative Corner */}
          <div className="absolute top-0 right-0 w-48 h-48 bg-orange-50 rounded-bl-[150px] -mr-12 -mt-12 z-0"></div>

          <div className="flex items-center gap-4 mb-10 pb-6 border-b border-stone-100 relative z-10">
            <div className="bg-slate-100 p-3 rounded-xl">
              <Activity className="w-6 h-6 text-slate-700" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-slate-900">Athlete Profile</h2>
              <p className="text-sm text-slate-500">Configure your parameters for maximum accuracy</p>
            </div>
          </div>

          <div className="space-y-12 relative z-10">
            {/* Biometrics Grid */}
            <div>
              <h3 className="text-xs font-extrabold text-orange-600 uppercase tracking-widest mb-6 flex items-center gap-2">
                <Scale className="w-4 h-4" />
                Body Metrics
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <InputGroup label="Age" value={formData.age} onChange={v => setFormData({ ...formData, age: v })} unit="yrs" icon={Calendar} />
                <InputGroup label="Height" value={formData.height} onChange={v => setFormData({ ...formData, height: v })} unit="cm" icon={Target} />
                <InputGroup label="Weight" value={formData.weight} onChange={v => setFormData({ ...formData, weight: v })} unit="kg" icon={Scale} />
                <InputGroup label="Body Fat" value={formData.fat} onChange={v => setFormData({ ...formData, fat: v })} unit="%" step={0.1} icon={Activity} />
                <InputGroup label="Average HR" value={formData.bpm} onChange={v => setFormData({ ...formData, bpm: v })} unit="bpm" icon={Heart} />
                <SelectGroup label="Gender" value={formData.gender} onChange={v => setFormData({ ...formData, gender: v })} options={['Male', 'Female']} icon={User} />
              </div>
            </div>

            {/* Training & Strategy Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
              <div>
                <h3 className="text-xs font-extrabold text-emerald-600 uppercase tracking-widest mb-6 flex items-center gap-2">
                  <Dumbbell className="w-4 h-4" />
                  Training Goals
                </h3>
                <div className="space-y-6">
                  <SelectGroup
                    label="Primary Goal"
                    value={formData.goal}
                    onChange={v => setFormData({ ...formData, goal: v })}
                    options={["Not Sure - Let AI Suggest", "Build Muscle", "Endurance", "HIIT", "Weight Loss", "Balanced"]}
                    icon={Target}
                  />
                  <SelectGroup
                    label="Training Frequency"
                    value={formData.activity}
                    onChange={v => setFormData({ ...formData, activity: v })}
                    options={["Light (1–3 days/week)", "Moderate (3–5 days/week)", "Elite (6–7 days/week)"]}
                    icon={Activity}
                  />
                </div>
              </div>
              <div>
                <h3 className="text-xs font-extrabold text-blue-600 uppercase tracking-widest mb-6 flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Fuel Strategy
                </h3>
                <div className="space-y-6">
                  <SelectGroup
                    label="Calorie Target"
                    value={formData.calorieStrategy}
                    onChange={v => setFormData({ ...formData, calorieStrategy: v })}
                    options={["Match TDEE", "TDEE - 200 kcal", "TDEE - 300 kcal"]}
                    icon={Flame}
                  />
                  <div className="p-5 rounded-2xl bg-orange-50 border border-orange-100 flex gap-4 transition-transform hover:scale-[1.02] cursor-help">
                    <div className="bg-white p-2 rounded-full shadow-sm h-fit">
                      <ChefHat className="w-5 h-5 text-orange-500" />
                    </div>
                    <div>
                      <p className="text-xs text-orange-800 leading-relaxed font-medium">
                        <strong>Chef's Tip:</strong> "Match TDEE" is perfect for performance maintenance. To cut weight without losing power, try a slight deficit of 200kcal.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {error && (
            <div className="mt-8 p-4 bg-red-50 border border-red-100 rounded-xl flex items-center gap-3 animate-pop-in">
              <div className="w-2 h-2 rounded-full bg-red-500"></div>
              <p className="text-red-600 text-sm font-bold">{error}</p>
            </div>
          )}

          <div className="mt-12">
            <button
              onClick={handleGenerate}
              disabled={loading}
              className="w-full bg-gradient-to-r from-orange-500 to-red-500 text-white font-extrabold text-lg py-5 px-6 rounded-2xl hover:shadow-2xl hover:shadow-orange-500/30 hover:scale-[1.01] transition-all duration-300 flex justify-center items-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed shimmer-wrapper"
            >
              {loading ? (
                <>
                  <RefreshCw className="animate-spin w-5 h-5" />
                  Calculating Macros...
                </>
              ) : (
                <>
                  <Flame className="w-5 h-5 fill-white" />
                  Generate Meal Plan
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* --- RESULTS SECTION --- */}
      {results && (
        <div id="results-section" className="max-w-7xl mx-auto px-6 pb-24 animate-fade-in scroll-mt-24 relative z-10">

          {/* Strategy Header */}
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-900 mb-6 animate-pop-in shadow-lg">
              {results.isAiPredicted && <Zap className="w-3 h-3 text-yellow-400 fill-current animate-pulse" />}
              <span className="text-white font-bold text-xs tracking-wide uppercase">
                {results.isAiPredicted ? 'AI Optimized Plan' : 'Manual Plan'}
              </span>
            </div>
            <h2 className="text-4xl md:text-6xl font-black text-slate-900 mb-4 animate-slide-up tracking-tight">
              Your Fuel Map
            </h2>
            <p className="text-xl font-bold text-slate-500 animate-slide-up" style={{ animationDelay: '0.1s' }}>
              Objective: <span className="text-orange-600">{results.finalGoal}</span>
            </p>
          </div>

          {/* Metrics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16 animate-slide-up" style={{ animationDelay: '0.2s' }}>
            <MetricCard
              title="BMI Index"
              value={results.bmi.toFixed(1)}
              sub={getBmiStatus(results.bmi)}
              subColor={getBmiColor(results.bmi)}
              icon={<Scale className="w-6 h-6" />}
              gradient="from-slate-700 to-slate-900"
              delay="0.3s"
            />
            <MetricCard
              title="Resting Burn (BMR)"
              value={results.bmr.toFixed(0)}
              sub="kcal / day"
              subColor="text-slate-400"
              icon={<Timer className="w-6 h-6" />}
              gradient="from-slate-700 to-slate-900"
              delay="0.4s"
            />
            <MetricCard
              title="Daily Target (TDEE)"
              value={results.tdee.toFixed(0)}
              sub="kcal / day"
              subColor="text-slate-400"
              icon={<Flame className="w-6 h-6 fill-white" />}
              gradient="from-slate-700 to-slate-900"
              delay="0.5s"
            />
          </div>

          {/* Meal Plan */}
          <div className="grid grid-cols-1 gap-6 max-w-5xl mx-auto">
            <div className="flex items-end justify-between mb-4 px-2 animate-slide-up" style={{ animationDelay: '0.4s' }}>
              <div>
                <div className="flex items-center gap-3">
                  <h3 className="text-2xl font-black text-slate-900">Daily Intake</h3>
                  <button
                    onClick={handleGenerate}
                    disabled={loading}
                    className="group p-2 rounded-full bg-slate-100 hover:bg-orange-100 text-slate-500 hover:text-orange-600 transition-all duration-300"
                    title="Regenerate meals based on current profile"
                  >
                    <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : 'group-hover:rotate-180 transition-transform duration-700'}`} />
                  </button>
                </div>
                <p className="text-slate-500 font-medium">Optimal macro distribution</p>
              </div>
              <div className="text-right">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Total Intake</span>
                <div className="text-3xl font-black text-slate-900">
                  {results.meals.reduce((acc, meal) => acc + meal.cal, 0).toFixed(0)}
                  <span className="text-sm font-bold text-slate-400">kcal</span>
                </div>
              </div>
            </div>

            {results.meals.map((meal, index) => {
              const style = getMealStyle(meal.type);
              const delay = `${0.5 + (index * 0.1)}s`;
              return (
                <div key={index} className={`relative overflow-hidden ${style.bg} border ${style.border} rounded-3xl p-6 md:p-8 transition-all duration-300 hover:shadow-lg group animate-slide-up`} style={{ animationDelay: delay }}>
                  <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-8">

                    <div className="flex items-center gap-6">
                      <div className={`w-16 h-16 rounded-2xl ${style.iconColor} bg-white flex items-center justify-center shadow-sm group-hover:scale-110 transition-transform duration-300`}>
                        {style.icon}
                      </div>
                      <div>
                        <div className={`text-xs font-extrabold uppercase tracking-widest mb-1 ${style.label}`}>{meal.type}</div>
                        <div className="text-xl md:text-2xl font-bold text-slate-900 group-hover:text-orange-600 transition-colors">{meal.name}</div>
                      </div>
                    </div>

                    {/* Macros */}
                    <div className="flex items-center gap-3 w-full md:w-auto mt-4 md:mt-0 pt-6 md:pt-0 border-t md:border-t-0 border-stone-200 md:border-none">
                      <MacroPill label="Protein" value={meal.prot.toFixed(1)} color="bg-red-100 text-red-700" />
                      <MacroPill label="Carbs" value={meal.carbs.toFixed(1)} color="bg-yellow-100 text-yellow-700" />
                      <MacroPill label="Fats" value={meal.fats.toFixed(1)} color="bg-emerald-100 text-emerald-700" />

                      <div className="ml-4 pl-6 border-l-2 border-stone-200">
                        <div className="text-xl font-black text-slate-900">{meal.cal} <span className="text-xs font-bold text-slate-400">kcal</span></div>
                      </div>
                    </div>

                  </div>
                </div>
              )
            })}
          </div>

          {/* --- CLUSTER FOODS LIST (Exchange Options) --- */}
          {results.clusterFoods && results.clusterFoods.length > 0 && (
            <div className="max-w-5xl mx-auto mt-16 animate-slide-up" style={{ animationDelay: '0.6s' }}>
              <div className="bg-white rounded-3xl border border-stone-200 overflow-hidden shadow-sm">
                <details className="group">
                  <summary className="flex items-center justify-between p-6 md:p-8 cursor-pointer bg-stone-50 hover:bg-stone-100 transition-colors list-none">
                    <div className="flex items-center gap-4">
                      <div className="bg-white p-3 rounded-xl border border-stone-200 text-orange-500 shadow-sm">
                        <Utensils className="w-6 h-6" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-slate-900">Food Exchange List</h3>
                        <p className="text-sm text-slate-500 font-medium">View all {results.clusterFoods.length} approved foods for your <strong>{results.finalGoal}</strong> plan</p>
                      </div>
                    </div>
                    <div className="transform transition-transform duration-300 group-open:rotate-180 text-slate-400">
                      <ChevronDown className="w-6 h-6" />
                    </div>
                  </summary>

                  <div className="p-6 md:p-8 border-top border-stone-100 max-h-[600px] overflow-y-auto custom-scrollbar">

                    {/* Search Input */}
                    <div className="mb-6 relative">
                      <div className="absolute left-4 top-3.5 text-slate-400">
                        <RefreshCw className="w-5 h-5" />
                      </div>
                      <input
                        type="text"
                        placeholder="Search for foods (e.g. 'Oats', 'Chicken')..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-12 pr-4 py-3 bg-stone-50 border border-stone-200 rounded-xl focus:border-orange-400 focus:bg-white focus:outline-none transition-all font-medium text-slate-700"
                      />
                    </div>

                    {replacingMealIndex !== null && (
                      <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-xl flex items-center justify-between">
                        <span className="text-sm text-blue-700 font-semibold">Select a food to replace <strong>{results.meals[replacingMealIndex]?.name}</strong> ({results.meals[replacingMealIndex]?.type})</span>
                        <button
                          onClick={() => setReplacingMealIndex(null)}
                          className="text-sm font-bold text-blue-600 hover:text-blue-800 underline"
                        >
                          Cancel
                        </button>
                      </div>
                    )}

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {results.clusterFoods
                        .filter(f => f.name.toLowerCase().includes(searchQuery.toLowerCase()))
                        .map((food, idx) => (
                          <div key={idx} className={`relative p-4 rounded-2xl border transition-all flex flex-col justify-between group/card bg-white overflow-hidden ${
                            replacingMealIndex !== null 
                              ? 'border-stone-200 hover:border-orange-400 hover:bg-orange-50 cursor-pointer' 
                              : 'border-stone-100 hover:border-orange-200 hover:bg-orange-50/50'
                          }`}>
                            <div className="overflow-hidden z-10">
                              <p className="font-bold text-slate-800 text-sm truncate pr-2 capitalize" title={food.name}>{food.name}</p>
                              <p className="text-xs text-slate-400 font-bold uppercase tracking-wider">{food.type}</p>
                            </div>

                            {/* Normal State: Calories & Macros */}
                            <div className="flex items-end justify-between mt-4 pt-4 border-t border-stone-100 gap-3">
                              <div className="flex gap-3">
                                <div className="flex flex-col items-center bg-red-50 px-3 py-2 rounded-lg">
                                  <span className="font-bold text-red-600 uppercase text-xs">Pro</span>
                                  <span className="font-black text-slate-900 text-lg">{food.prot?.toFixed(0) || 0}g</span>
                                </div>
                                <div className="flex flex-col items-center bg-yellow-50 px-3 py-2 rounded-lg">
                                  <span className="font-bold text-yellow-600 uppercase text-xs">Carb</span>
                                  <span className="font-black text-slate-900 text-lg">{food.carbs?.toFixed(0) || 0}g</span>
                                </div>
                                <div className="flex flex-col items-center bg-emerald-50 px-3 py-2 rounded-lg">
                                  <span className="font-bold text-emerald-600 uppercase text-xs">Fat</span>
                                  <span className="font-black text-slate-900 text-lg">{food.fats?.toFixed(0) || 0}g</span>
                                </div>
                              </div>
                              <div className="text-right bg-slate-100 px-3 py-2 rounded-lg">
                                <span className="block font-black text-slate-900 text-lg">{food.cal}</span>
                                <span className="text-xs font-bold text-slate-600 uppercase">kcal</span>
                              </div>
                            </div>

                            {/* Replacement Mode: Show Replace Buttons */}
                            {replacingMealIndex !== null && (
                              <button
                                onClick={() => handleReplaceFood(replacingMealIndex, food.name)}
                                disabled={replacementLoading}
                                className="mt-3 w-full py-2 px-3 bg-orange-500 hover:bg-orange-600 disabled:bg-slate-300 text-white font-bold text-sm rounded-lg transition-colors flex items-center justify-center gap-2"
                              >
                                <Replace className="w-4 h-4" />
                                Replace
                              </button>
                            )}

                            {/* Normal Mode: Show Replace Meal Buttons */}
                            {replacingMealIndex === null && (
                              <div className="mt-3 flex gap-2">
                                {results.meals.map((meal, mealIdx) => (
                                  <button
                                    key={mealIdx}
                                    onClick={() => setReplacingMealIndex(mealIdx)}
                                    title={`Replace this food with ${food.name}`}
                                    className="flex-1 py-1.5 px-2 bg-slate-100 hover:bg-slate-200 text-slate-700 font-bold text-xs rounded-lg transition-colors flex items-center justify-center gap-1"
                                  >
                                    <Replace className="w-3 h-3" />
                                    {meal.type.slice(0, 3)}
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        ))}
                    </div>
                    {results.clusterFoods.filter(f => f.name.toLowerCase().includes(searchQuery.toLowerCase())).length === 0 && (
                      <div className="text-center py-10 text-slate-400 font-medium">
                        No foods found matching "{searchQuery}"
                      </div>
                    )}
                    <div className="mt-6 pt-6 border-t border-stone-100 text-center text-xs text-slate-400 font-medium">
                      Showing top recommended items from our database
                    </div>
                  </div>
                </details>
              </div>
            </div>
          )}

        </div>
      )}



      {/* Footer */}
      <footer className="border-t border-stone-200 py-16 mt-20 relative z-10 bg-white">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <div className="flex justify-center items-center gap-2 mb-6">
            <div className="bg-slate-900 p-1.5 rounded-lg">
              <Utensils className="w-4 h-4 text-white" />
            </div>
            <span className="font-extrabold text-slate-900 text-lg">NutriFuel</span>
          </div>
          <p className="text-slate-500 text-sm font-medium mb-8 max-w-sm mx-auto">
            Science-backed nutrition planning for the modern athlete. <br />Eat well, train hard.
          </p>
          <div className="flex justify-center gap-8 text-sm font-bold text-slate-400">
            <span className="hover:text-orange-600 cursor-pointer transition-colors">Privacy</span>
            <span className="hover:text-orange-600 cursor-pointer transition-colors">Terms</span>
            <span className="hover:text-orange-600 cursor-pointer transition-colors">Contact</span>
          </div>
        </div>
      </footer>
    </div >
  );
};

// --- COMPONENTS ---

const NavIcon = ({ label, icon }) => (
  <div className="flex items-center gap-2 text-slate-500 hover:text-orange-600 cursor-pointer transition-colors group">
    <span className="group-hover:animate-bounce">{icon}</span>
    <span className="text-sm font-semibold">{label}</span>
  </div>
);

const FeatureStep = ({ icon, title, desc, delay }) => (
  <div className="bg-white/60 backdrop-blur-sm p-6 rounded-2xl border border-stone-100 hover:bg-white hover:shadow-xl transition-all duration-300 animate-slide-up" style={{ animationDelay: delay }}>
    <div className="w-12 h-12 bg-white rounded-xl shadow-sm flex items-center justify-center mb-4 text-2xl">
      {icon}
    </div>
    <h3 className="text-lg font-bold text-slate-900 mb-2">{title}</h3>
    <p className="text-sm text-slate-500 font-medium">{desc}</p>
  </div>
);

const InputGroup = ({ label, value, onChange, unit, step = 1, icon: Icon }) => (
  <div className="group">
    <label className="block text-xs font-bold text-slate-500 uppercase tracking-wide mb-2 transition-colors group-focus-within:text-orange-600">{label}</label>
    <div className="relative">
      <div className="absolute left-4 top-4 text-slate-400 group-focus-within:text-orange-500 transition-colors">
        {Icon && <Icon size={20} />}
      </div>
      <input
        type="number"
        value={value}
        step={step}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full px-4 py-4 food-input font-bold text-lg pl-12"
      />
      <span className="absolute right-4 top-4 text-slate-400 text-xs font-bold pointer-events-none">{unit}</span>
    </div>
  </div>
);

const SelectGroup = ({ label, value, onChange, options, icon: Icon }) => (
  <div className="group">
    <label className="block text-xs font-bold text-slate-500 uppercase tracking-wide mb-2 transition-colors group-focus-within:text-orange-600">{label}</label>
    <div className="relative">
      <div className="absolute left-4 top-4 text-slate-400 group-focus-within:text-orange-500 transition-colors">
        {Icon && <Icon size={20} />}
      </div>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full px-4 py-4 food-input appearance-none cursor-pointer font-bold text-lg pl-12"
      >
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
      <ChevronDown className="absolute right-4 top-5 w-5 h-5 text-slate-400 pointer-events-none" />
    </div>
  </div>
);

const MetricCard = ({ title, value, sub, subColor, icon, gradient, isHighlight, delay }) => (
  <div className={`food-card p-6 relative overflow-hidden group hover:-translate-y-1 transition-transform duration-300 ${isHighlight ? 'bg-slate-900 text-white border-none' : ''}`}>
    <div className={`absolute top-0 right-0 p-4 rounded-bl-3xl bg-gradient-to-br ${gradient} text-white shadow-lg animate-pop-in`} style={{ animationDelay: delay }}>
      {icon}
    </div>
    <p className={`text-xs font-extrabold uppercase tracking-widest mb-3 mt-1 ${isHighlight ? 'text-slate-400' : 'text-slate-400'}`}>{title}</p>
    <div className="flex items-baseline gap-2">
      <h2 className={`text-5xl font-black ${isHighlight ? 'text-white' : 'text-slate-900'} animate-pop-in`} style={{ animationDelay: delay }}>{value}</h2>
    </div>
    <p className={`${subColor} font-bold text-sm mt-2`}>{sub}</p>
  </div>
);

const MacroPill = ({ label, value, color }) => (
  <div className={`flex flex-col items-center justify-center ${color} w-24 py-3 rounded-xl shadow-sm`}>
    <span className="text-2xl font-black leading-none">{value}</span>
    <span className="text-xs font-bold uppercase opacity-70 mt-1">g {label}</span>
  </div>
);

export default App;