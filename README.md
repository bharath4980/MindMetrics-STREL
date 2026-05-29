# MindMetrics — Stress Prediction System 🧠

> Automated stress detection from physiological signals using machine learning  
> Full-stack system: FastAPI backend + React frontend

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-green?style=flat&logo=fastapi)
![React](https://img.shields.io/badge/React-frontend-blue?style=flat&logo=react)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest%20%7C%20SVM-orange?style=flat)

**Course:** CSCI 6838 – Spring 2026 Capstone &nbsp;|&nbsp; **Team:** Mind Metrics  
**University:** University of Houston – Clear Lake

Traditional stress assessment relies on subjective surveys. MindMetrics replaces that with an objective, automated system — taking physiological and activity-based data as input and predicting whether a person is stressed or not in real time, using the [STREL](https://osf.io/qshv7/) dataset of crisis leaders in naturalistic settings.

---

## 📊 Results

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| XGBoost             | ~70%     | 0.75    |
| Random Forest       | ~70%     | 0.75    |
| SVM                 | ~69%     | ~0.73   |
| Logistic Regression | ~68%     | ~0.72   |

**Validation:** Person-based 3-fold GroupKFold CV — no participant appears in both train and test sets, preventing data leakage across individuals

**Features:** 26 total — 5 physiological, 2 motion, 3 NASA-TLX, 4 Big Five personality, 3 demographic, 3 daily patterns, 4 categorical (encoded), 2 temporal (Hour, Minute)

**Top predictor:** Heart rate-related features had the highest importance score  
**Target:** NHR_Stress (binary: Stressed = 1, Not-Stressed = 0)

---

## 🖥️ UI Features

- Select from 32 features across 9 categories
- Choose any of the 4 ML models for prediction
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
