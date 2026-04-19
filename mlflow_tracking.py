import os
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

log = logging.getLogger("journeyiq.mlflow")

# MLflow tracking URI — can point to a remote server or local sqlite store
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///C:/Users/yourname/project/mlflow_journeyiq.db")
mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()


#Context manager
@contextmanager
def run_context(
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
):
   
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        log.info(
            "MLflow run started | experiment=%s | run_id=%s",
            experiment_name, run.info.run_id
        )
        try:
            yield run
        except Exception as exc:
            mlflow.set_tag("run_status", "FAILED")
            mlflow.set_tag("failure_reason", str(exc))
            raise
        else:
            mlflow.set_tag("run_status", "COMPLETED")


#Model logging
def log_sklearn_model(
    model: Any,
    sample_input: np.ndarray,
    artifact_path: str,
    extra_pip: Optional[list] = None
) -> str:
   
    sample_output = model.predict(sample_input)
    signature     = infer_signature(sample_input, sample_output)

    pip_requirements = ["scikit-learn", "numpy", "pandas"]
    if extra_pip:
        pip_requirements += extra_pip

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        signature=signature,
        pip_requirements=pip_requirements,
        registered_model_name=f"journeyiq_{artifact_path}"
    )
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
    log.info("Model logged at: %s", model_uri)
    return model_uri


#Run comparison
def compare_runs(experiment_name: str, metric: str = "test_r2", top_n: int = 10):
    """
    Retrieve the top-N runs from an experiment and print a formatted
    comparison table sorted by the chosen metric (descending).

    Args:
        experiment_name : name of the MLflow experiment
        metric          : metric name to sort by
        top_n           : number of runs to display
    """
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        log.warning("Experiment '%s' not found.", experiment_name)
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n
    )

    records = []
    for r in runs:
        records.append({
            "run_id":    r.info.run_id[:8],
            "run_name":  r.data.tags.get("mlflow.runName", "—"),
            "status":    r.info.status,
            metric:      r.data.metrics.get(metric, float("nan")),
            "start_time": pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
        })

    df = pd.DataFrame(records)
    print(f"\n── Top {len(df)} runs for experiment: {experiment_name} ──")
    print(df.to_string(index=False))
    return df


#Champion registration
def register_champion(
    experiment_name: str,
    model_artifact_path: str,
    registry_model_name: str,
    metric: str = "test_r2",
    higher_is_better: bool = True
):
   
    order = "DESC" if higher_is_better else "ASC"
    exp   = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    best_run = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1
    )
    if not best_run:
        log.warning("No completed runs found for experiment '%s'", experiment_name)
        return

    best = best_run[0]
    model_uri = f"runs:/{best.info.run_id}/{model_artifact_path}"
    best_metric_val = best.data.metrics.get(metric, "N/A")

    log.info(
        "Registering champion: run=%s | %s=%.4f | uri=%s",
        best.info.run_id[:8], metric, best_metric_val or 0, model_uri
    )

    # Register a new model version
    mv = mlflow.register_model(model_uri, registry_model_name)

    # Archive current Production version
    try:
        prod_versions = client.get_latest_versions(registry_model_name, stages=["Production"])
        for pv in prod_versions:
            if pv.version != mv.version:
                client.transition_model_version_stage(
                    name=registry_model_name,
                    version=pv.version,
                    stage="Archived"
                )
                log.info("Archived previous production version: %s", pv.version)
    except Exception as e:
        log.warning("Could not archive old production version: %s", e)

    # Promote new version to Production
    client.transition_model_version_stage(
        name=registry_model_name,
        version=mv.version,
        stage="Production"
    )
    log.info("Version %s promoted to Production for model '%s'", mv.version, registry_model_name)


#Load production model
def load_production(registry_model_name: str) -> Any:
    model_uri = f"models:/{registry_model_name}/Production"
    log.info("Loading production model from: %s", model_uri)
    return mlflow.sklearn.load_model(model_uri)
