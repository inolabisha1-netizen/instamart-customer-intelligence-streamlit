# Instamart Customer Intelligence & Support Copilot

**An end-to-end AI/ML project combining customer analytics, churn-risk prediction, support automation, and policy-based retrieval.**

## Overview

A customer intelligence and support system built using Python, Machine Learning, SQLite, and Streamlit. It transforms transactional data into actionable customer insights and demonstrates automated customer-support workflows.

## Key Features

- **Customer Segmentation:** RFM analysis and clustering to identify customer groups and premium customers.
- **Retention Analytics:** Random Forest classification to estimate customer lapse risk and prioritize retention.
- **Support Copilot:** Automated ticket classification, prioritization, routing, and human-review fallback.
- **Policy Assistant (RAG):** TF-IDF retrieval and grounded policy responses, with fallback for unsupported questions.
- **Refund Agent (Backend):** Rule-based refund validation, escalation, and checkpoint-based recovery.

## Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `SQLite` · `Streamlit` · `TF-IDF` · `Git`

## Model & System Highlights

| Component | Result |
|---|---|
| Customer segmentation | 4 customer segments |
| Clustering silhouette score | 0.526 |
| Retention model test F1-score | 0.637 |
| Support ticket processing | 164 tickets |
| Policy retrieval evaluation | 12/12 supported queries passed |

*The AI support workflows use a deterministic mock LLM for reproducible demonstrations; no paid LLM API is required.*


## Run Locally

### 1. Clone the repository
```bash
git clone <https://github.com/inolabisha1-netizen/instamart-customer-intelligence-streamlit.git>
cd instamart-customer-intelligence-streamlit
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

Activate it on Windows:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Generate synthetic data
```bash
python generate_data.py
```

### 5. Prepare the database and analytics outputs
```bash
python -m src.data_pipeline
python -m src.customer_models
python -m src.triage
```

### 6. Launch the Streamlit application
```bash
python -m streamlit run streamlit_app.py
```

The application opens locally in your browser, typically at
`http://localhost:8501`.

All datasets are synthetic and generated locally. No paid LLM API
is required for the deterministic demonstration.

## Project Scope

Developed as an end-to-end AI/ML capstone, demonstrating data engineering, customer intelligence, predictive modeling, retrieval-augmented responses, and tool-based support automation.

**Note:** The Streamlit application presents analytics and policy retrieval. The refund agent is a separate backend component, not an interactive dashboard feature.