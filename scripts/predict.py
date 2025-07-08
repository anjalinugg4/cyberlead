import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/xgb_model.joblib")

def predict_lead_score(lead_dict):
    """
    Given a dictionary with lead features, return a predicted probability of conversion.
    """
    df = pd.DataFrame([lead_dict])  # Convert single lead to 1-row DataFrame
    proba = model.predict_proba(df)[0][1]  # Probability of class 1 (converted)
    return round(proba, 3)  # Return probability rounded to 3 decimal places

# Optional test
if __name__ == "__main__":
    sample_lead = {
        "company_name": "Lester-Howard",
        "industry": "Education",
        "company_size": 1919,
        "region": "EMEA",
        "deal_stage": "Closed Won",
        "rep_assigned": "Morgan",
        "last_touch_days": 11,
        "email_opens": 1,
        "meetings": 2,
        "deal_value": 35265,
        "threat_alerts_detected": 2
    }

    score = predict_lead_score(sample_lead)
    print(f"🔮 Predicted conversion probability: {score}")
