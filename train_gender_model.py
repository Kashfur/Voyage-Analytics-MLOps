"""
JourneyIQ — Traveller Gender Classification Model
==================================================
Dataset : users.csv  (1,340 rows)
Columns : code, company, name, gender, age

Target  : gender  (male | female) — 'none' rows excluded
Features: age (int), company_code (label-encoded, 5 classes)

Trains two candidates (Random Forest vs Gradient Boosting),
logs BOTH to MLflow, and saves the winner.

Run:
    python train_gender_model.py --data data/users.csv
Then open:
    mlflow ui --backend-store-uri sqlite:///mlflow_journeyiq.db
    → http://localhost:5000
"""

import argparse
import os
import logging
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("journeyiq.train_gender")

parser = argparse.ArgumentParser()
parser.add_argument("--data",       default="data/users.csv")
parser.add_argument("--experiment", default="journeyiq_gender_classification")
args = parser.parse_args()

FEATURE_COLS = ["age", "company_code"]


def main():
    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow_journeyiq.db")
    mlflow.set_experiment(args.experiment)

    log.info("Reading: %s", args.data)
    df = pd.read_csv(args.data)
    log.info("Shape: %s | Columns: %s", df.shape, df.columns.tolist())

    # Remove 'none' gender — not a valid classification target
    df = df[df["gender"].str.strip().str.lower() != "none"].copy()
    log.info("After removing 'none': %d rows", len(df))

    # ── Encoding ───────────────────────────────────────────────────────────────
    le_company = LabelEncoder()
    df["company_code"] = le_company.fit_transform(
        df["company"].astype(str).str.strip()
    )

    le_gender = LabelEncoder()   # female=0, male=1
    y = le_gender.fit_transform(df["gender"].str.strip().str.lower())
    X = df[FEATURE_COLS].values

    log.info("Classes: %s | label map: %s",
             le_gender.classes_,
             dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=7
    )

    # ── Two candidate models ───────────────────────────────────────────────────
    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8,
            class_weight="balanced", random_state=7, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.10,
            max_depth=4, random_state=7
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    best_model = None
    best_f1    = -1
    best_name  = None

    for model_name, clf in candidates.items():
        log.info("Training: %s", model_name)

        cv_f1 = cross_val_score(
            clf, X_train, y_train, scoring="f1_weighted", cv=skf
        )

        clf.fit(X_train, y_train)
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc  = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec  = float(recall_score(y_test, y_pred,    average="weighted", zero_division=0))
        f1   = float(f1_score(y_test, y_pred,        average="weighted", zero_division=0))
        auc  = float(roc_auc_score(y_test, y_proba))

        log.info("%s → Acc=%.4f | F1=%.4f | AUC=%.4f", model_name, acc, f1, auc)
        print(f"\n--- {model_name} ---")
        print(classification_report(y_test, y_pred, target_names=le_gender.classes_))

        # ── Log this run to MLflow ─────────────────────────────────────────────
        with mlflow.start_run(run_name=f"gender_{model_name}"):
            mlflow.log_param("model_type",       model_name)
            mlflow.log_param("features",         str(FEATURE_COLS))
            mlflow.log_param("n_train",          len(X_train))
            mlflow.log_param("n_test",           len(X_test))
            mlflow.log_param("classes",          str(list(le_gender.classes_)))
            mlflow.log_param("class_weight",
                             "balanced" if model_name == "RandomForest" else "none")

            mlflow.log_metric("cv_f1_mean",  float(cv_f1.mean()))
            mlflow.log_metric("cv_f1_std",   float(cv_f1.std()))
            mlflow.log_metric("test_accuracy",  acc)
            mlflow.log_metric("test_precision", prec)
            mlflow.log_metric("test_recall",    rec)
            mlflow.log_metric("test_f1",        f1)
            mlflow.log_metric("test_roc_auc",   auc)
            for i, v in enumerate(cv_f1):
                mlflow.log_metric("cv_f1_fold", float(v), step=i)

            # Tag the winning run
            if f1 > best_f1:
                mlflow.set_tag("champion", "true")
            else:
                mlflow.set_tag("champion", "false")

            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path=f"gender_model_{model_name}",
                registered_model_name="JourneyIQ_GenderClassifier"
            )

        if f1 > best_f1:
            best_f1    = f1
            best_model = clf
            best_name  = model_name

    log.info("Best model: %s (F1=%.4f)", best_name, best_f1)

    # ── Save artefacts ─────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model,   "models/gender_clf_model.joblib")
    joblib.dump(le_gender,    "models/gender_label_encoder.joblib")
    joblib.dump(le_company,   "models/company_label_encoder.joblib")
    joblib.dump(FEATURE_COLS, "models/gender_feature_cols.joblib")

    print("\n" + "="*62)
    print("  JourneyIQ Gender Classifier — Training Complete")
    print("="*62)
    print(f"  Best model : {best_name}")
    print(f"  Test F1    : {best_f1:.4f}")
    print(f"  MLflow exp : {args.experiment}")
    print("="*62)
    print("  View in MLflow UI:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow_journeyiq.db")
    print("  → http://localhost:5000")
    print("="*62)


if __name__ == "__main__":
    main()
