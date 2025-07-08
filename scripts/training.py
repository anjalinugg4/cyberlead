import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
import os


# Step 1: Load the dataset
df = pd.read_csv("data/sales_pipeline.csv")

# Step 2: Define features (X) and target (y)
X = df.drop("converted", axis=1)
y = df["converted"]

# Step 3: Specify preprocessing columns
categorical = ["industry", "region", "deal_stage", "rep_assigned"]
numerical = ["company_size", "last_touch_days", "email_opens", "meetings", "deal_value", "threat_alerts_detected"]

# Step 4: Build preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numerical)
])

# Step 5: Initialize the XGBoost classifier
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

# Step 6: Create the full pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

# Step 7: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Train the model
model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = model.predict(X_test)
print("✅ Model Evaluation Report:")
print(classification_report(y_test, y_pred))

# Step 10: Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.joblib")
print("✅ Trained model saved to models/xgb_model.joblib")
