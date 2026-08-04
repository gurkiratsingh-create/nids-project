# AI-Based Network Intrusion Detection System (NIDS)

An enterprise-style cybersecurity portfolio project that combines machine learning, explainability, incident logging, and a SOC-style dashboard to detect and analyze suspicious network activity.

## Project Overview

This project is designed to simulate a production-ready Network Intrusion Detection System (NIDS) using the CICIDS2017 dataset. It applies a multi-stage pipeline that includes:

- Data preprocessing and feature engineering
- Anomaly detection using Isolation Forest
- Known attack classification using XGBoost
- Risk scoring and attack explanation
- Incident logging to SQLite
- Interactive visualization in a Streamlit dashboard
- PDF report generation for security investigations

The system is intended to demonstrate how machine learning can support modern Security Operations Center (SOC) workflows.

## Features

- Detects unknown or anomalous network behavior with Isolation Forest
- Classifies known attack types with XGBoost
- Calculates a risk score for each detected event
- Uses SHAP-based explainability to highlight influential features
- Stores incidents in a structured SQLite database
- Presents findings through an interactive Streamlit dashboard
- Generates professional PDF security reports
- Designed with clean architecture, logging, and modular code organization

## System Architecture

```text
Raw Network Dataset
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Isolation Forest (anomaly detection)
    ↓
XGBoost (attack classification)
    ↓
Risk Score Calculation
    ↓
SHAP Explainability
    ↓
SQLite Incident Logging
    ↓
SOC Dashboard
    ↓
PDF Security Report
```

## Folder Structure

```text
network_intrusion_detection/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── database/
├── reports/
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── anomaly.py
│   ├── classifier.py
│   ├── explainability.py
│   ├── database.py
│   ├── recommendations.py
│   └── report_generator.py
└── dashboard/
    ├── pages/
    ├── charts.py
    └── utils.py
```

## Technologies Used

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Plotly
- Streamlit
- SQLite
- Joblib
- ReportLab
- Logging

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd network_intrusion_detection
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Train the models

```bash
python train.py
```

### Run the prediction pipeline

```bash
python predict.py
```

### Launch the dashboard

```bash
streamlit run app.py
```

## Dataset

This project uses the CICIDS2017 dataset, which contains labeled network traffic data suitable for intrusion detection research and model development.

> Ensure the dataset is placed in the data/raw directory before training.

## Dashboard Overview

The Streamlit dashboard provides a SOC-style view of:

- Detected anomalies and attack events
- Model confidence and risk levels
- Incident summaries from the SQLite database
- Visual analytics for network activity
- Explainability insights for model decisions

## Future Enhancements

- Add real-time streaming ingestion
- Integrate with SIEM platforms such as Splunk or Elastic
- Support model retraining pipelines
- Add alert escalation and notification workflows
- Improve explainability with richer attack narratives
- Expand reporting features with executive summaries

## License

This project is intended for educational and portfolio purposes. It may be adapted for internal or research use with appropriate attribution.
