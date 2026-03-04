"""
NHR_Stress Classification using XGBoost
Trains XGBoost model and evaluates on test data
Target: NHR_Stress (S = Stress, NS = No Stress)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PREPROCESSING FUNCTION
# ─────────────────────────────────────────────
DROP_COLS = ["Participant", "PA_Activity", "SNS_Stress"]
CAT_COLS = ["Day", "Period", "Profession", "Gender", "Activity4"]
TARGET = "NHR_Stress"

def preprocess(df, scaler=None, fit=True):
    """
    Preprocess data: drop columns, one-hot encode categoricals, scale features
    
    Args:
        df: Input dataframe
        scaler: StandardScaler for feature scaling
        fit: Whether to fit scaler (True for train, False for test)
    
    Returns:
        X: Processed features
        y: Target labels (0=No Stress, 1=Stress)
        scaler: Fitted scaler
    """
    df = df.copy()
    df.drop(columns=DROP_COLS, inplace=True)

    # Encode target
    y = (df[TARGET] == "S").astype(int).values
    df.drop(columns=[TARGET], inplace=True)

    # One-hot encode categoricals
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    df = pd.get_dummies(df, columns=[c for c in CAT_COLS if c in df.columns], drop_first=False)
    
    # Impute missing values
    df = df.fillna(df.median(numeric_only=True))

    X = df.values.astype(np.float32)

    # Scale features
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, scaler


def train_and_evaluate_xgboost():
    """
    Train XGBoost model with hyperparameter tuning and evaluate on test data
    
    Returns:
        dict: Model results including metrics and ROC curve data
    """

    # ─────────────────────────────────────────────
    # 1. LOAD DATA
    # ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    train_df = pd.read_excel("../../data/processed/train.xlsx")
    test_df = pd.read_excel("../../data/processed/test.xlsx")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # ─────────────────────────────────────────────
    # 2. PREPROCESSING
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing")
    print("=" * 60)

    X_train, y_train, scaler = preprocess(train_df, fit=True)
    X_test, y_test, _ = preprocess(test_df, scaler=scaler, fit=False)

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Train class balance — S: {y_train.sum()}, NS: {(y_train==0).sum()}")

    # ─────────────────────────────────────────────
    # 3. TRAINING
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Training XGBoost")
    print("=" * 60)

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
    }

    xgb_base = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    print("Running GridSearchCV (3-fold)...")
    grid_search = GridSearchCV(xgb_base, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    # ─────────────────────────────────────────────
    # 4. EVALUATION
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation")
    print("=" * 60)

    model = grid_search.best_estimator_
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)
    auc = roc_auc_score(y_test, y_probs)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

    label_names = ["NS (No Stress)", "S (Stress)"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_preds, target_names=label_names))
    print("\nConfusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, y_preds), index=label_names, columns=label_names))

    # Calculate TPR and FPR for ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    
    # Return results for comparison
    results = {
        'model_name': 'XGBoost',
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }
    
    return results


if __name__ == "__main__":
    train_and_evaluate_xgboost()