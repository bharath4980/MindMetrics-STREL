# MindMetrics — Feature Runner UI

Interactive web tool for running stress prediction models with selected feature subsets and model selection, saving results.

## Structure

```
ui/
├── backend/
│   ├── main.py          # FastAPI app — /features, /run, /save endpoints
│   ├── features.py      # Reads features.json and returns categories
│   ├── features.json    # Feature list organized by category with tooltips
│   ├── runner.py        # Calls model_comparison.run_all() and reads result CSVs
│   └── requirements.txt # Python dependencies
└── frontend/
    └── src/
        ├── LandingPage.jsx   # Screen 1 — project intro
        ├── App.jsx           # Screen 2 — S3 URL verification
        └── MainUI.jsx        # Screen 3 — feature & model selection, run, results
```

## Setup & Running

### Backend

From `UI/backend/`:

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload
```

Runs at `http://127.0.0.1:8000`

### Frontend

From `UI/frontend/`:

```bash
# Install dependencies
npm install

# Run dev server
npm run dev
```

Runs at `http://127.0.0.1:5173`

## Features

### Feature Selection
- 32 features organized in 9 categories:
  - Temporal (Hour, Minute)
  - Context (Day, Period, Profession)
  - Demographics (Age, Gender, BMI)
  - Heart Rate (HR, HR_Baseline, HR_Normalized)
  - Physiological (RR, Cadence, Speed, PNSindex, SNSindex, Stressindex)
  - Activity (Activity4)
  - Sleep & Physical Activity (Sleep_Time, PA_Percent, PA_Intensity)
  - NASA-TLX Workload (N_MD, N_PD, N_TD, N_P, N_E, N_F)
  - Big Five Personality (Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness)

### Model Selection
- XGBoost - Gradient boosting with decision trees
- Random Forest - Ensemble of decision trees
- SVM - Support Vector Machine classifier
- Logistic Regression - Linear classification model

Select any combination of models (minimum 1 required)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/features` | Returns feature categories with tooltips from `features.json` |
| POST | `/run` | Trains selected models with selected features, returns metrics |
| POST | `/save` | Saves run JSON to `results/runs/<user>_<timestamp>.json` |
| GET | `/results/<file>` | Serves static plot images from `results/metrics/` |

## Run Output

```json
{
  "selected_features": ["RR", "PNSindex", "Cadence"],
  "selected_models": ["XGBoost", "Random Forest"],
  "model_metrics": [
    { "Model": "XGBoost", "Accuracy": 0.86, "Precision": 0.84, "Recall": 0.83, "F1-Score": 0.83, "TN": 5458, "FP": 799, "FN": 747, "TP": 3951 }
  ],
  "fold_metrics": [
    { "model": "XGBoost", "fold": 1, "accuracy": 0.87, "precision": 0.90, "recall": 0.74, "f1_score": 0.81, "roc_auc": 0.96 }
  ],
  "roc_data": [
    { "model": "XGBoost", "fpr": [0.0, 0.001, ...], "tpr": [0.0, 0.123, ...] }
  ]
}
```

## Saved Run Format

Each run auto-saves to `results/runs/<user>_<timestamp>.json`:

```json
{
  "user": "Harish",
  "timestamp": "2026-04-05T12:00:00.000Z",
  "notes": "Testing HRV features only",
  "features_used": ["RR", "PNSindex", "SNSindex"],
  "features_excluded": ["Cadence", "Speed", "BMI"],
  "output": { 
    "model_metrics": [...], 
    "fold_metrics": [...],
    "roc_data": [...]
  }
}
```

## Regenerating Plots from Saved Runs

You can regenerate plots from any saved run JSON:

```bash
# Single run
python src/evaluation/plots_from_json.py results/runs/Harish_2026-04-05T12-00-00.json

# All runs
python src/evaluation/plots_from_json.py --all
```

Plots are saved to `results/runs/<run_name>/` folder.

## Notes

- Runs take several minutes — a live timer shows elapsed time on the Run button
- Results auto-save locally and attempt S3 upload simultaneously after each run
- S3 upload shows a Retry button if it fails
- Plot images in the UI are always from the latest run (`results/metrics/`)
- Full run history with ROC curve data is preserved in `results/runs/`
- Charts automatically adjust based on number of selected models
