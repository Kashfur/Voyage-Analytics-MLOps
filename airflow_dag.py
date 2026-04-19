
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule
import os
import logging
import pandas as pd

log = logging.getLogger("journeyiq.dag")

#DAG-level defaults
DEFAULT_ARGS = {
    "owner":             "journeyiq-mlops",
    "depends_on_past":   False,
    "start_date":        datetime(2024, 1, 1),
    "retries":           2,
    "retry_delay":       timedelta(minutes=5),
    "email_on_failure":  False,
    "email_on_retry":    False,
}

DATA_DIR   = os.environ.get("JOURNEYIQ_DATA_DIR",   "/opt/journeyiq/data")
MODELS_DIR = os.environ.get("JOURNEYIQ_MODELS_DIR", "/opt/journeyiq/models")
TRAIN_DIR  = os.environ.get("JOURNEYIQ_TRAIN_DIR",  "/opt/journeyiq")



# Python callables

def check_data_freshness(**ctx):
   
    required_files = ["flights.csv", "users.csv", "hotels.csv"]
    now_ts = datetime.utcnow().timestamp()
    stale  = []

    for fname in required_files:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            stale.append(f"{fname} (missing)")
            continue
        age_hours = (now_ts - os.path.getmtime(fpath)) / 3600
        if age_hours > 25:
            stale.append(f"{fname} (age={age_hours:.1f}h)")

    if stale:
        raise ValueError(f"Stale or missing data files: {stale}")

    log.info("Data freshness check passed for all required files.")


def preprocess_datasets(**ctx):
   
    for fname in ["flights.csv", "users.csv", "hotels.csv"]:
        fpath = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(fpath)
        original_len = len(df)

        # Standardise strings
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Price outlier capping (only for flights)
        if "price" in df.columns:
            cap = df["price"].quantile(0.99)
            df["price"] = df["price"].clip(upper=cap)

        cleaned_path = fpath.replace(".csv", "_clean.csv")
        df.to_csv(cleaned_path, index=False)
        log.info(
            "Preprocessed %s: %d → %d rows. Saved to %s",
            fname, original_len, len(df), cleaned_path
        )


def validate_new_model(**ctx):
  
    import json

    new_metrics_path  = os.path.join(MODELS_DIR, "latest_metrics.json")
    base_metrics_path = os.path.join(MODELS_DIR, "baseline_metrics.json")

    if not os.path.exists(new_metrics_path):
        log.warning("No new metrics file found — skipping promotion.")
        return "skip_promotion"

    with open(new_metrics_path) as f:
        new_metrics = json.load(f)

    if not os.path.exists(base_metrics_path):
        # No baseline yet — promote by default
        log.info("No baseline found. Promoting new model as first production version.")
        return "promote_if_improved"

    with open(base_metrics_path) as f:
        base_metrics = json.load(f)

    new_r2  = new_metrics.get("test_r2", 0)
    base_r2 = base_metrics.get("test_r2", 0)

    log.info("New R²: %.4f | Baseline R²: %.4f", new_r2, base_r2)
    if new_r2 >= base_r2 - 0.005:          # allow a 0.5% tolerance
        return "promote_if_improved"
    return "skip_promotion"


def promote_model(**ctx):
   
    import json, shutil

    log.info("Promoting new model to production...")
    for fname in ["flight_price_model.joblib", "label_encoders.joblib",
                  "gender_clf_model.joblib", "hotel_tfidf_matrix.joblib",
                  "hotels_metadata.joblib"]:
        src  = os.path.join(MODELS_DIR, f"new_{fname}")
        dest = os.path.join(MODELS_DIR, fname)
        if os.path.exists(src):
            shutil.move(src, dest)
            log.info("Promoted: %s → %s", src, dest)

    # Update baseline
    new_metrics_path = os.path.join(MODELS_DIR, "latest_metrics.json")
    base_path        = os.path.join(MODELS_DIR, "baseline_metrics.json")
    if os.path.exists(new_metrics_path):
        shutil.copy(new_metrics_path, base_path)
    log.info("Baseline metrics updated.")


def send_pipeline_notification(**ctx):
    """
    Placeholder for a Slack/email notification task.
    In production, replace the log.info call with an actual HTTP
    request to the Slack Incoming Webhooks endpoint or smtp.sendmail().
    """
    dag_run   = ctx["dag_run"]
    run_state = dag_run.get_state() if dag_run else "unknown"
    log.info(
        "NOTIFICATION → JourneyIQ pipeline '%s' completed with state: %s",
        dag_run.run_id if dag_run else "N/A", run_state
    )


# DAG definition


with DAG(
    dag_id="journeyiq_ml_pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="30 2 * * *",      # 02:30 UTC daily
    catchup=False,
    max_active_runs=1,
    tags=["journeyiq", "mlops", "regression", "classification", "recommender"],
    doc_md=__doc__,
) as dag:

    #Data freshness gate
    t_check_data = PythonOperator(
        task_id="data_freshness_check",
        python_callable=check_data_freshness,
    )

    #Preprocessing
    t_preprocess = PythonOperator(
        task_id="preprocess_datasets",
        python_callable=preprocess_datasets,
    )

    #Model training
    t_train_fare = BashOperator(
        task_id="train_fare_model",
        bash_command=(
            f"cd {TRAIN_DIR} && "
            "python train_fare_model.py "
            f"--data {DATA_DIR}/flights_clean.csv "
            "--experiment journeyiq_fare_regression"
        ),
    )

    t_train_gender = BashOperator(
        task_id="train_gender_model",
        bash_command=(
            f"cd {TRAIN_DIR} && "
            "python train_gender_model.py "
            f"--data {DATA_DIR}/users_clean.csv "
            "--experiment journeyiq_gender_classification"
        ),
    )

    t_train_recommender = BashOperator(
        task_id="train_recommender",
        bash_command=(
            f"cd {TRAIN_DIR} && "
            "python train_recommender.py "
            f"--data {DATA_DIR}/hotels_clean.csv"
        ),
    )

    #Validation branch
    t_validate = BranchPythonOperator(
        task_id="run_model_validation",
        python_callable=validate_new_model,
    )

    #Promote
    t_promote = PythonOperator(
        task_id="promote_if_improved",
        python_callable=promote_model,
    )

    t_skip = DummyOperator(task_id="skip_promotion")

    #Notify 
    t_notify = PythonOperator(
        task_id="notify_team",
        python_callable=send_pipeline_notification,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    #Dependency graph
    t_check_data >> t_preprocess >> [t_train_fare, t_train_gender, t_train_recommender]
    [t_train_fare, t_train_gender, t_train_recommender] >> t_validate
    t_validate >> [t_promote, t_skip]
    [t_promote, t_skip] >> t_notify
