import joblib
import pandas as pd
import os
import warnings

# Suppress the "Feature Names" warning for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def run_test():
    print("\n--- 🛡️ Starting ML Logic Validation ---")
    
    # 1. Load BOTH Model and Scaler
    # Using relative paths so it works regardless of where you run it
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'fraud_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ Model and Scaler loaded successfully.")
    except FileNotFoundError:
        print(f"❌ ERROR: Model files not found. Looking in: {model_path}")
        return

    # 2. Define Scenarios (Must match the 3 features used in training)
    # We remove 'location_enc' and 'device_enc' because the model doesn't expect them
    high_risk_tx = {
        'amount': 99999,
        'hour': 3,
        'amount_diff_from_avg': 95000
    }
    
    low_risk_tx = {
        'amount': 20,
        'hour': 14,
        'amount_diff_from_avg': 2
    }

    # 3. Run predictions
    for name, tx in [("High Risk", high_risk_tx), ("Low Risk", low_risk_tx)]:
        # Create DataFrame
        df = pd.DataFrame([tx])
        
        # CRITICAL: Scale the data before predicting
        scaled_data = scaler.transform(df)
        
        # Predict probability
        prob = model.predict_proba(scaled_data)[0][1]
        risk_score = prob * 100
        
        print(f"\nTest [{name}]: Risk Score = {risk_score:.2f}%")
        
        # Logic Validation
        if name == "High Risk":
            if risk_score > 70:
                print("✅ PASSED: Correctly identified high risk.")
            else:
                print("❌ FAILED: High risk transaction was ignored.")
        
        elif name == "Low Risk":
            if risk_score < 30:
                print("✅ PASSED: Correctly identified low risk.")
            else:
                print("❌ FAILED: Normal transaction was flagged too high.")

if __name__ == "__main__":
    run_test()