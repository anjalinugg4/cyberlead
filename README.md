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


Dashboard Features
Lead Scoring: Predict conversion probability for each lead
Threat Risk Assessment: Score leads using threat alerts, industry risk, and region
Actionable Recommendations: Dynamic logic to recommend actions like “Follow Up” or “Ignore”
Interactive Visualizations: Histograms, bar charts, and scatter plots for insights
Filtering: By deal stage, rep, industry, region, risk score, and recommendation
CSV Export: Download filtered results directly from the dashboard



