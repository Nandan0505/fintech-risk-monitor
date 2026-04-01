import joblib
import pandas as pd
import numpy as np
import warnings
import os

# 1. SILENCE WARNINGS
warnings.filterwarnings("ignore")

def run_test():
    print("\n" + "="*50)
    print("🛡️  PROJECT AEGIS: ML LOGIC VALIDATION")
    print("="*50)
    
    # 2. DYNAMIC PATH RESOLUTION (The Fix)
    # Get the directory of this script (tests folder)
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the PROJECT ROOT
    project_root = os.path.abspath(os.path.join(current_script_path, ".."))
    
    # Define absolute paths to the models in the 'ml' folder
    model_path = os.path.join(project_root, 'ml', 'models', 'fraud_model.pkl')
    scaler_path = os.path.join(project_root, 'ml', 'models', 'scaler.pkl')

    # 3. LOAD ARTIFACTS
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"✅ SUCCESS: Models loaded from: \n   {model_path}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not find model files.")
        print(f"   Expected Location: {model_path}")
        print(f"   Actual Error: {e}")
        return

    # 4. TEST SCENARIOS
    test_scenarios = [
        {"name": "High-Value Late Night", "data": [120000, 3, 115000]},
        {"name": "Small Afternoon Coffee", "data": [15, 14, 2]},
        {"name": "Medium Unusual Spike", "data": [5000, 11, 4800]}
    ]

    # 5. EXECUTE & DISPLAY
    print(f"\n{'Test Case':<25} | {'Risk %':<10} | {'Action'}")
    print("-" * 55)

    for test in test_scenarios:
        # Prepare Data
        df = pd.DataFrame([test['data']], columns=['amount', 'hour', 'amount_diff_from_avg'])
        
        # Scale Data
        scaled_data = scaler.transform(df)
        
        # Predict
        risk_score = model.predict_proba(scaled_data)[0][1]
        
        # Business Logic Thresholds
        if risk_score > 0.70:
            status = "🚨 BLOCK"
        elif risk_score > 0.30:
            status = "⚠️  MFA (OTP)"
        else:
            status = "🟢 PASS"
            
        print(f"{test['name']:<25} | {risk_score*100:>7.2f}% | {status}")
    print("-" * 55 + "\n")

if __name__ == "__main__":
    run_test()