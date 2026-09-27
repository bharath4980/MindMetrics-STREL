# MindMetrics — Stress Prediction System 🧠

> A capstone application for comparing stress-classification models on the STREL dataset  
> React experiment dashboard + FastAPI backend

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-green?style=flat&logo=fastapi)
![React](https://img.shields.io/badge/React-frontend-blue?style=flat&logo=react)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest%20%7C%20SVM-orange?style=flat)

**Course:** CSCI 6838 – Spring 2026 Capstone &nbsp;|&nbsp; **Team:** Mind Metrics  
**University:** University of Houston – Clear Lake

MindMetrics explores stress classification using physiological and activity-based data from the [STREL](https://osf.io/qshv7/) dataset of crisis leaders in naturalistic settings. Users select features and models, run training and evaluation, and compare metrics and visualizations through a React interface backed by FastAPI.

## My Contribution

This is my fork of our UHCL team capstone. My primary contribution was the **Logistic Regression workflow**, including preprocessing and model evaluation.

- Developed a Logistic Regression model in PyTorch using a linear layer, binary cross-entropy loss, and the Adam optimizer.
- Worked on numeric scaling and categorical encoding for the model inputs.
- Evaluated the model with five-fold GroupKFold, grouping records by participant.
- Reviewed accuracy, precision, recall, F1, ROC-AUC, and confusion matrices across folds.

See the [Logistic Regression implementation](src/models/logreg_model.py) and [earlier experiments](src/test/). The full application and four-model comparison are shared team work; the [upstream repository](https://github.com/harishcmuthyala/MindMetrics-STREL) retains the project history.

---

## 📊 Evaluation and Saved Results

The model implementations use **five-fold GroupKFold by participant**. Records from a participant stay within one fold, avoiding participant overlap between training and validation sets. The Logistic Regression preprocessing pipeline is fitted on each training fold before transforming its validation fold.

The committed [model metrics](results/metrics/model_metrics.csv) and [fold metrics](results/metrics/fold_metrics.csv) contain this saved experiment snapshot:

| Model | Mean fold accuracy | Mean fold F1 |
|---|---:|---:|
| Logistic Regression | 71.13% | 0.606 |
| Random Forest | 67.18% | 0.610 |

These values describe the stored run, not a fixed score for every feature selection. The current CSV snapshot contains these two models; the application also supports SVM and XGBoost. New runs regenerate the metrics files.

**Target:** `NHR_Stress` (binary: Stressed = 1, Not-Stressed = 0). Feature selection and model-specific preprocessing determine the inputs used for a run.

---

## 🖥️ UI Features

- Select from 32 features across 9 categories
- Choose any combination of the 4 ML models for training and evaluation
- View metrics, confusion matrices, ROC curves and feature importance charts
- Results auto-save to `results/runs/`

---

## 📁 Project Structure

```
MindMetrics-STREL/
├── data/
│   ├── raw/                 # Original CSV (STREL_raw.csv)
│   └── processed/           # Per-fold train/test splits
├── docs/                    # Feature processing guide
├── notebooks/
│   ├── eda/                 # Exploratory data analysis
│   └── prototyping/         # Model experiments (XGBoost + PyTorch hybrid)
├── src/
│   ├── data/                # Data preprocessing pipeline
│   ├── validation/          # Person-based CV splits
│   ├── models/              # Model implementations
│   └── evaluation/          # Metrics and comparison
├── results/
│   ├── eda/                 # Feature analysis outputs
│   ├── models/              # Saved trained models
│   ├── plots/               # Visualizations
│   ├── metrics/             # Performance tables
│   └── runs/                # Saved run history (JSON + plots)
├── UI/
│   ├── backend/             # FastAPI server
│   └── frontend/            # React + Vite interface
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

### Prerequisites
- Python 3.8+ · Node.js 16+ · Git · pip · npm

### 1. Clone the repository

```bash
git clone https://github.com/bharath4980/MindMetrics-STREL.git
cd MindMetrics-STREL
```

> Dataset is already included at `data/raw/STREL_raw.csv`

### 2. Start the backend

```bash
cd UI/backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Backend runs at: **http://127.0.0.1:8000**

### 3. Start the frontend (open a new terminal)

```bash
cd UI/frontend
npm install
npm run dev
```

Frontend runs at: **http://127.0.0.1:5173**

---

## 🔬 Algorithms

| Model               | Type         |
|---------------------|--------------|
| Logistic Regression | Linear       |
| Random Forest       | Ensemble     |
| XGBoost             | Ensemble     |
| SVM                 | Kernel-based |

---

## 📖 References

- STREL Paper: *"STREL – Naturalistic Dataset and Methods for Studying Mental Stress and Relaxation Patterns in Critical Leading Roles"*, IEEE Transactions on Affective Computing, 2025.  
  https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11185201
- Dataset: https://github.com/UH-ACDC/STREL/blob/main/data/Activity_Stress_data_N24.csv
- Original R code: https://github.com/UH-ACDC/STREL/
