# MindMetrics — ML Stress Prediction on STREL

**Course:** CSCI 6838 – Spring 2026 Capstone  
**Team:** Mind Metrics  

Predicts binary mental-stress states (Stressed / Not-Stressed) for crisis
leaders using the [STREL](https://osf.io/qshv7/) dataset. Perform
data preprocessing and implements 4 ML algorithms: Logistic Regression,
Random Forest, XGBoost, and SVM.

---

## 📁 Repo Layout

```
MindMetrics-STREL/
├── data/
│   ├── raw/                 # Original CSV (STREL_raw.csv)
│   └── processed/           # Per-fold train/test splits
├── docs/                    # Feature processing guide
├── notebooks/
│   ├── eda/                 # Exploratory data analysis
│   └── prototyping/         # Model experiments (XGBoost+PyTorch hybrid)
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

## 📋 Prerequisites

- **Python 3.8+** - For backend and ML models
- **Node.js 16+** - For frontend development
- **Git** - For cloning the repository
- **pip** - Python package manager
- **npm** - Node package manager

---

## ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/harishcmuthyala/MindMetrics-STREL.git
cd MindMetrics-STREL
```

> **Note:** The dataset is already included in `data/raw/STREL_raw.csv`

---

## 🔬 Algorithms

| Model               |
|---------------------|
| Logistic Regression |
| Random Forest       |
| XGBoost             |
| SVM                 |


---

## 📊 Evaluation

**Metrics:** Accuracy · Precision · Recall · F1-Score · ROC-AUC

**Validation:** Person-based 3-fold CV using GroupKFold on Participant column
(ensures no participant appears in both train and test sets).

**Features:** 26 total — 5 physiological, 2 motion, 3 NASA-TLX, 4 Big Five
personality, 3 demographic, 3 daily patterns, 4 categorical (encoded), 2
temporal (Hour, Minute).

**Target:** NHR_Stress (binary: S=1, NS=0)

---

## 🖥️ UI Setup

### Backend

```bash
cd UI/backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Backend runs at: **http://127.0.0.1:8000**

### Frontend (Open NEW terminal)

```bash
cd UI/frontend
npm install
npm run dev
```

Frontend runs at: **http://127.0.0.1:5173**

### Using the UI

- Select from 32 features across 9 categories
- Choose models: XGBoost, Random Forest, SVM, Logistic Regression
- View metrics, confusion matrices, ROC curves, feature importance
- Results auto-save to `results/runs/`

---

## 📖 Key References

- STREL Paper: *"STREL – Naturalistic Dataset and Methods for Studying
  Mental Stress and Relaxation Patterns in Critical Leading Roles"*,
  IEEE Transactions on Affective Computing, 2025.
  https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11185201
- Dataset: https://github.com/UH-ACDC/STREL/blob/main/data/Activity_Stress_data_N24.csv
- Original R code: https://github.com/UH-ACDC/STREL/
