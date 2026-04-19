import argparse
import os
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("journeyiq.train_fare")

#XGBoost with sklearn fallback
try:
    from xgboost import XGBRegressor
    def get_model():
        return XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.80,
            random_state=42, n_jobs=-1, verbosity=0
        )
    MODEL_BACKEND = "XGBRegressor"
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    def get_model():
        return GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.85, random_state=42
        )
    MODEL_BACKEND = "GradientBoostingRegressor"

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--data",      default="data/flights.csv")
parser.add_argument("--test_size", type=float, default=0.20)
parser.add_argument("--cv_folds",  type=int,   default=5)
parser.add_argument("--experiment",default="journeyiq_fare_regression")
args = parser.parse_args()

# Column definitions
CATEGORICAL_COLS = ["from", "to", "flightType", "agency"]
NUMERIC_COLS     = ["time", "distance", "speed_proxy"]
FEATURE_COLS     = CATEGORICAL_COLS + NUMERIC_COLS
TARGET_COL       = "price"


def encode_categoricals(df, fit=True, encoders=None):
    if fit:
        encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder() if fit else encoders[col]
        if fit:
            df[col] = le.fit_transform(df[col].astype(str).str.strip())
            encoders[col] = le
        else:
            df[col] = df[col].astype(str).str.strip().map(
                lambda x, le=le: int(le.transform([x])[0]) if x in le.classes_ else -1
            )
    return df, encoders


def main():
    #MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow_journeyiq.db")
    mlflow.set_experiment(args.experiment)

    log.info("Model backend  : %s", MODEL_BACKEND)
    log.info("MLflow experiment: %s", args.experiment)
    log.info("Reading: %s", args.data)

    df = pd.read_csv(args.data)
    log.info("Raw shape: %s", df.shape)

    #Cleaning
    df.dropna(subset=[TARGET_COL, "time", "distance"], inplace=True)
    df = df[(df[TARGET_COL] > 0) & (df["time"] > 0) & (df["distance"] > 0)]
    upper = df[TARGET_COL].quantile(0.99)
    df    = df[df[TARGET_COL] <= upper]
    n_rows = len(df)
    log.info("After cleaning: %d rows", n_rows)

    #Feature engineering
    df["speed_proxy"] = df["distance"] / df["time"].clip(lower=0.1)

    df, label_encoders = encode_categoricals(df, fit=True)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    model = get_model()

    #Cross-validation
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    rmse_cv = np.sqrt(-cross_val_score(
        model, X_train, y_train,
        scoring="neg_mean_squared_error", cv=cv, n_jobs=-1
    ))
    log.info("CV RMSE scores: %s", np.round(rmse_cv, 2))

    #Final fit
    if MODEL_BACKEND == "XGBRegressor":
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    else:
        model.fit(X_train, y_train)

    y_pred    = model.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_mae  = float(mean_absolute_error(y_test, y_pred))
    test_r2   = float(r2_score(y_test, y_pred))

    #MLflow logging
    with mlflow.start_run(run_name=f"fare_{MODEL_BACKEND}") as run:
        # Parameters
        mlflow.log_param("model_type",   MODEL_BACKEND)
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth",    5)
        mlflow.log_param("learning_rate",0.08)
        mlflow.log_param("test_size",    args.test_size)
        mlflow.log_param("cv_folds",     args.cv_folds)
        mlflow.log_param("n_features",   len(FEATURE_COLS))
        mlflow.log_param("features",     str(FEATURE_COLS))
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows",  len(X_test))
        mlflow.log_param("dataset_rows", n_rows)

        # Cross-validation metrics
        mlflow.log_metric("cv_rmse_mean", float(rmse_cv.mean()))
        mlflow.log_metric("cv_rmse_std",  float(rmse_cv.std()))
        for i, v in enumerate(rmse_cv):
            mlflow.log_metric("cv_rmse_fold", float(v), step=i)

        # Test-set metrics
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae",  test_mae)
        mlflow.log_metric("test_r2",   test_r2)

        # Feature importances as artifact
        os.makedirs("models", exist_ok=True)
        imp_df = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        imp_path = "models/feature_importances.csv"
        imp_df.to_csv(imp_path, index=False)
        mlflow.log_artifact(imp_path)

        # Log the model itself into MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="fare_regression_model",
            registered_model_name="JourneyIQ_FarePredictor"
        )

        run_id = run.info.run_id
        log.info("MLflow run_id: %s", run_id)

    #Save joblib artefacts for Flask API
    joblib.dump(model,          "models/flight_price_model.joblib")
    joblib.dump(label_encoders, "models/label_encoders.joblib")
    joblib.dump(FEATURE_COLS,   "models/fare_feature_cols.joblib")

    print("\n" + "="*62)
    print("  JourneyIQ Fare Model — Training Complete")
    print("="*62)
    print(f"  Backend    : {MODEL_BACKEND}")
    print(f"  Test R²    : {test_r2:.4f}")
    print(f"  Test RMSE  : {test_rmse:.2f}")
    print(f"  Test MAE   : {test_mae:.2f}")
    print(f"  CV RMSE    : {rmse_cv.mean():.2f} ± {rmse_cv.std():.2f}")
    print(f"  MLflow run : {run_id}")
    print(f"  Experiment : {args.experiment}")
    print("="*62)
    print("  View in MLflow UI:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow_journeyiq.db")
    print("  → http://localhost:5000")
    print("="*62)


if __name__ == "__main__":
    main()
