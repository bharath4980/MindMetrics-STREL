# MindMetrics — Stress Classification

> A capstone application for comparing stress-classification models on the STREL dataset  
> React experiment dashboard + FastAPI backend

**Python · PyTorch · scikit-learn · FastAPI · React**

**Team:** Mind Metrics · **University:** University of Houston–Clear Lake

MindMetrics explores stress classification using physiological and activity-based data from the [STREL](https://osf.io/qshv7/) dataset of crisis leaders in naturalistic settings. Users select features and models, run training and evaluation, and compare metrics and visualizations through a React interface backed by FastAPI.

## My Contribution

This is my fork of our UHCL team capstone. My primary contribution was the **Logistic Regression workflow**, including preprocessing and model evaluation.

- Developed a Logistic Regression model in PyTorch using a linear layer, binary cross-entropy loss, and the Adam optimizer.
- Built preprocessing for missing values, numeric scaling, and categorical encoding, fitted within each training fold.
- Evaluated the model with five-fold GroupKFold, grouping records by participant.
- Returned fold-level classification metrics and aggregated confusion matrices for the team's comparison dashboard.

See the [Logistic Regression implementation](src/models/logreg_model.py) and [earlier experiments](src/test/). The full application and four-model comparison are shared team work; the [upstream repository](https://github.com/harishcmuthyala/MindMetrics-STREL) retains the project history.

---

## Evaluation and Saved Results

The model implementations use **five-fold GroupKFold by participant**. Records from a participant stay within one fold, avoiding participant overlap between training and validation sets. The Logistic Regression preprocessing pipeline is fitted on each training fold before transforming its validation fold.

The committed [model metrics](results/metrics/model_metrics.csv) and [fold metrics](results/metrics/fold_metrics.csv) contain this saved experiment snapshot:

| Model | Mean fold accuracy | Mean fold F1 |
|---|---:|---:|
| Logistic Regression | 71.13% | 0.606 |
| Random Forest | 67.18% | 0.610 |

These values describe the stored run, not a fixed score for every feature selection. The current CSV snapshot contains these two models; the application also supports SVM and XGBoost. New runs regenerate the metrics files.

**Target:** `NHR_Stress` (binary: Stressed = 1, Not-Stressed = 0). Feature selection and model-specific preprocessing determine the inputs used for a run.

---

## UI Features

- Select from 32 features across 9 categories
- Choose any combination of the 4 ML models for training and evaluation
- View metrics, confusion matrices, ROC curves and feature importance charts
- Save experiment results to `results/runs/`

---

## Project Structure

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

## How to Run Locally

### Prerequisites
- Python 3.11 and Node.js 22 for the setup below, plus Git, pip, and npm
- Backend dependencies are listed in `UI/backend/requirements.txt`; the frontend uses Vite 5. The old Python 3.8 / Node.js 16 setup is not compatible with these dependencies.

### 1. Clone the repository

```bash
git clone https://github.com/bharath4980/MindMetrics-STREL.git
cd MindMetrics-STREL
```

> Dataset is already included at `data/raw/STREL_raw.csv`

### 2. Start the backend

From the repository root on macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r UI/backend/requirements.txt
cd UI/backend
python -m uvicorn main:app --reload
```

Backend runs at: **http://127.0.0.1:8000**

### 3. Start the frontend (open a new terminal)

From the repository root:

```bash
cd UI/frontend
npm install
npm run dev
```

Frontend runs at: **http://127.0.0.1:5173**

---

## Algorithms

| Model               | Type         |
|---------------------|--------------|
| Logistic Regression | Linear       |
| Random Forest       | Ensemble     |
| XGBoost             | Ensemble     |
| SVM                 | Kernel-based |

---

## Scope and limitations

This is a local research and comparison application. The API has no authentication and uses permissive CORS, so it should not be exposed publicly without additional security work.

- Scores depend on feature selection, preprocessing, and the particular run. The saved table above is an experiment snapshot.
- The Logistic Regression ROC curve returned to the UI is from the final fold; the accuracy and F1 summaries average all five folds.
- Training does not set a fixed random seed, and several Python dependencies are unpinned, so reruns can vary.
- Grouped cross-validation prevents participant overlap. It does not by itself establish performance on new populations.

## References

- STREL Paper: *"STREL – Naturalistic Dataset and Methods for Studying Mental Stress and Relaxation Patterns in Critical Leading Roles"*, IEEE Transactions on Affective Computing, 2025.  
  https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11185201
- Dataset: https://github.com/UH-ACDC/STREL/blob/main/data/Activity_Stress_data_N24.csv
- Original R code: https://github.com/UH-ACDC/STREL/
