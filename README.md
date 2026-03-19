# Customer Churn Prediction Pipeline (Local MLOps)

A reproducible machine learning pipeline for customer churn prediction using ZenML, scikit-learn, MLflow, and Streamlit.

This repository is designed to run on a local machine with local experiment tracking and local model artifacts.

## Project Overview

This project implements an end-to-end churn workflow:

1. Ingest data from CSV
2. Clean and preprocess features
3. Train a classification model
4. Evaluate model metrics
5. Deploy only if a quality threshold is met
6. Run inference and serve predictions through a Streamlit app

## What This Project Demonstrates

- Building modular ML steps and pipelines with ZenML
- Running repeatable training and deployment workflows
- Tracking experiments and runs with local MLflow
- Registering and loading models through a local MLflow registry
- Applying a quality gate before deployment
- Serving predictions through an interactive Streamlit interface
- Structuring an ML project for readability and maintainability

## Pipeline Flow

The core flow in code is:

Ingest -> Clean -> Train -> Evaluate -> Deploy

- Training pipeline logs runs and metrics
- Deployment pipeline applies an accuracy threshold (default 0.85)
- Inference pipeline loads a model and writes predictions to CSV

## Tech Stack

| Area                  | Tools          |
| --------------------- | -------------- |
| Language              | Python         |
| Data                  | pandas, numpy  |
| Modeling              | scikit-learn   |
| Orchestration         | ZenML          |
| Tracking and Registry | MLflow (local) |
| UI                    | Streamlit      |

## Repository Structure

```text
churn-pipeline-main/
├── app.py
├── run_pipeline.py
├── run_experiments.py
├── requirements.txt
├── requirements-streamlit.txt
├── extracted_data/
│   └── customer_churn_dataset-testing-master.csv
├── pipelines/
│   ├── trainning_pipeline.py
│   ├── deployement_pipeline.py
│   └── inference_pipeline.py
├── steps/
│   ├── ingest_data.py
│   ├── clean_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── deployment_steps.py
│   └── config.py
├── src/
│   ├── ingest_util.py
│   ├── clean_util.py
│   ├── model_util.py
│   └── evaluation_util.py
└── analysis/
    └── churn_prediction.ipynb
```

## Local Setup

### Prerequisites

- Python 3.10+
- Git

### 1) Clone and enter project

```bash
git clone https://github.com/asmiverma/churn-pipeline.git
cd churn-pipeline
```

### 2) Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv env
source env/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Initialize ZenML

```bash
zenml init
```

## Important Local Configuration

Two scripts currently use hardcoded dataset paths and should be updated before running:

- run_pipeline.py -> DATA_PATH
- run_experiments.py -> FILE_PATH

Set each path to your local CSV file, for example:

- extracted_data/customer_churn_dataset-testing-master.csv

## How to Run Locally

### 1) Start MLflow UI (local tracking)

In a separate terminal, from the project root:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then open:

- http://127.0.0.1:5000

### 2) Run training pipeline

```bash
python run_pipeline.py --mode train --model GradientBoosting
```

Other model names available in code:

- RandomForest
- LogisticRegression
- GradientBoosting
- SVMS

### 3) Run deployment pipeline (quality-gated)

```bash
python run_pipeline.py --mode deploy --model GradientBoosting --min-accuracy 0.85
```

If the model passes the threshold:

- A local model artifact is saved under models/deployed_YYYYMMDD_HHMMSS/
- Metadata is saved alongside the model
- The run is recorded in MLflow

### 4) Run inference pipeline

```bash
python run_pipeline.py --mode inference
```

Predictions are written to:

- predictions/predictions_YYYYMMDD_HHMMSS.csv

### 5) Run experiment suite

```bash
python run_experiments.py
```

This runs multiple predefined model configurations for comparison.

### 6) Launch Streamlit app

```bash
streamlit run app.py
```

The app supports:

- Single-customer prediction
- Batch CSV prediction
- Downloadable prediction output

## Local MLflow and Model Registry Notes

- MLflow artifacts and run metadata are stored locally in mlruns/
- Deployment uses a registered model name churn_predictor in pipeline code
- Inference loads the latest registered model by default
- Streamlit can also load from local model artifact folders as fallback

## Reproducibility Notes

- Data split uses a fixed random_state (42) in preprocessing utilities
- Pipeline steps are separated and version-controlled for consistent reruns
- Local artifacts and metrics are saved so runs can be inspected and compared

## Current Limitations

- Dataset paths are hardcoded in two entry scripts and should be edited locally
- Test coverage is currently minimal
- Error handling can be strengthened for path and schema validation

## Future Improvements

- Replace hardcoded data paths with CLI arguments or environment variables
- Add unit and integration tests for all pipeline steps
- Add data validation contracts before training/inference
- Add model drift monitoring on new inference batches
- Add CI checks for linting, tests, and reproducibility checks

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Maintainer

Asmi Verma
