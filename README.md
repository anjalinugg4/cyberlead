# CyberLead: Smart Sales Pipeline for Cybersecurity Teams

CyberLead is a data-driven dashboard designed to help cybersecurity sales teams prioritize high-value leads. By combining machine learning with lead metadata, CyberLead predicts conversion likelihood, assesses threat risk, and recommends next steps for each lead. This tool enables better decision-making across the sales pipeline.

## Goals and Objectives

- Predict the probability that a sales lead will convert
- Identify high-risk leads based on threat intelligence
- Recommend next steps for sales representatives based on scoring logic
- Enable uploading and filtering of new lead data in an interactive dashboard

## Tools and Technologies Used

- **Python**
- **Streamlit** – Interactive dashboard
- **scikit-learn** – Preprocessing and pipeline management
- **XGBoost** – Model training and inference
- **Faker** – Synthetic data generation
- **Plotly & Seaborn** – Data visualizations
- **Hugging Face Transformers** – NLP-based lead summaries
- **dotenv** – Environment variable management
- **joblib** – Saving and loading the trained model

## Machine Learning Pipeline

- **Model**: XGBoost Classifier
- **Target**: `converted` (binary classification)
- **Categorical Features**: `industry`, `region`, `deal_stage`, `rep_assigned`
- **Numerical Features**: `company_size`, `last_touch_days`, `email_opens`, `meetings`, `deal_value`, `threat_alerts_detected`
- **Pipeline**: `ColumnTransformer` for preprocessing + classifier using `Pipeline`
- **Evaluation**: Accuracy and classification report on a held-out test set

## Directory Structure

cyberlead/
├── app/
│ └── dashboard.py
├── data/
│ └── sales_pipeline.csv
├── models/
│ └── xgb_model.joblib
├── scripts/
│ ├── generate_data.py
│ ├── training.py
│ └── predict.py
├── .env
├── .gitignore
└── README.md


## Setup Instructions

## 1.Clone the Repository

```bash
git clone https://github.com/anjalinugg4/cyberlead.git
cd cyberlead
```

### 2. Create Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add Environment Variables
```
HF_TOKEN=your_huggingface_token_here
```

### 4. Generate Synethetic Lead Data
```
python scripts/generate_data.py
```

### 5. Train the Model
```
python scripts/training.py
```

### 6. Launch the Dashboard
```
streamlit run app/dashboard.py
```


## Dashboard Features

- **Lead Scoring**: Predicts the conversion probability for each lead using an XGBoost model.
- **Threat Risk Assessment**: Calculates a composite risk score based on threat alerts, industry-specific risk, and regional impact.
- **Actionable Recommendations**: Provides next-step guidance such as "Follow Up", "Send Proposal", or "Ignore" based on lead characteristics and predicted performance.
- **Interactive Visualizations**:
  - Histograms of conversion scores
  - Deal stage funnel charts (grouped by rep or industry)
  - Scatter plots of threat alerts vs. conversion scores
- **Advanced Filtering**:
  - Deal stage
  - Sales representative
  - Industry
  - Region
  - Threat risk score threshold
  - Recommendation type
- **CSV Export**: Allows download of filtered lead lists directly from the dashboard.

