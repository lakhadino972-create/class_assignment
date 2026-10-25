"""
Train the dialysis-need model.

Usage (from project root):
    python backend/train.py

Outputs:
    model/dialysis_model.joblib   -> fitted sklearn Pipeline
    model/metadata.json           -> feature list, input ranges, metrics
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "kidney_disease_dataset.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Dialysis_Needed"
FEATURES = [
    "Age",
    "Creatinine_Level",
    "BUN",
    "Diabetes",
    "Hypertension",
    "GFR",
    "Urine_Output",
    "CKD_Status",
]
SEED = 42


def build_candidates():
    # class_weight="balanced" is important: only ~1.3% of rows need dialysis.
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("clf", RandomForestClassifier(
                    n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                    random_state=SEED, n_jobs=-1)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [("clf", GradientBoostingClassifier(random_state=SEED))]
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    X, y = df[FEATURES], df[TARGET]
    print(f"Rows: {len(df)} | Dialysis cases: {int(y.sum())} ({y.mean():.2%})")

    # Keep a stratified hold-out set for final, honest reporting.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Model selection with stratified CV on the training split only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for name, pipe in build_candidates().items():
        proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        scores[name] = {
            "cv_average_precision": float(average_precision_score(y_train, proba)),
            "cv_recall": float(recall_score(y_train, pred)),
            "cv_precision": float(precision_score(y_train, pred, zero_division=0)),
            "cv_f1": float(f1_score(y_train, pred)),
        }
        print(f"[CV] {name}: {scores[name]}")

    # F1 balances missed cases (recall) against false alarms (precision);
    # average precision breaks ties.
    best_name = max(scores, key=lambda n: (scores[n]["cv_f1"], scores[n]["cv_average_precision"]))
    print(f"\nSelected model: {best_name}")

    best = build_candidates()[best_name]
    best.fit(X_train, y_train)

    proba_test = best.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    print("\nHold-out test report:")
    print(classification_report(y_test, pred_test, digits=3, zero_division=0))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred_test))

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "average_precision": float(average_precision_score(y_test, proba_test)),
        "recall": float(recall_score(y_test, pred_test)),
        "precision": float(precision_score(y_test, pred_test, zero_division=0)),
        "test_rows": int(len(y_test)),
        "test_positive_cases": int(y_test.sum()),
    }

    # Refit on ALL data for the deployed model.
    final_model = build_candidates()[best_name].fit(X, y)
    joblib.dump(final_model, MODEL_DIR / "dialysis_model.joblib")

    metadata = {
        "model_name": best_name,
        "features": FEATURES,
        "ranges": {
            c: {"min": float(df[c].min()), "max": float(df[c].max()), "mean": float(df[c].mean())}
            for c in ["Age", "Creatinine_Level", "BUN", "GFR", "Urine_Output"]
        },
        "training_rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "cv_scores": scores,
        "test_metrics": test_metrics,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved model + metadata to {MODEL_DIR}")


if __name__ == "__main__":
    main()
