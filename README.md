# ✈️Intelligent Travel Analytics Platform

## Table of Contents

1. [Project Summary](#project-summary)  
2. [Problem Statement](#problem-statement)  
3. [Tech Stack](#tech-stack)  
4. [Repository Structure](#repository-structure)  
5. [Models Built](#models-built)  
6. [MLOps Architecture](#mlops-architecture)  
7. [How to Run This Project](#how-to-run-this-project)  
8. [API Reference](#api-reference)  
9. [Evaluation Metrics](#evaluation-metrics)  
10. [Challenges & Decisions](#challenges--decisions)  
11. [Future Improvements](#future-improvements)  

---

## Project Summary

The travel industry generates enormous volumes of transactional and behavioural data every day — flight bookings, user demographics, hotel stays, and more. Despite this richness, most travel platforms still rely on static pricing rules and manual recommendation lists. JourneyIQ was built to change that.

This platform takes three interrelated datasets — flights, users, and hotels — and uses them to power three distinct machine learning capabilities: predicting flight fares in real time, inferring traveller profiles from booking patterns, and surfacing personalised hotel recommendations. Rather than stopping at notebook-level experimentation, the project goes further to package every component for production: the fare model is served through a Flask REST API, containerised with Docker, and deployed on Kubernetes with horizontal autoscaling. An Apache Airflow DAG retrains and validates all three models on a daily schedule, while a Jenkins pipeline handles continuous integration and delivery from a code commit all the way to a production rollout. MLflow ties the experiment lifecycle together, tracking every hyperparameter, metric, and model version so no result is ever lost.

The Streamlit dashboard consolidates all three capabilities into a tabbed, interactive interface — fare estimation with a confidence band, traveller classification with a probability chart, and hotel discovery with a similarity score for each recommendation.

Working through this project surfaced a genuine appreciation for why "deploy the model" is far harder than "train the model." Every layer — validation logic, error handling, container security, pod readiness probes, pipeline gating — exists to make the system reliable under conditions the training environment never faces.

## Problem Statement

The travel and tourism sector faces three compounding challenges:

**1. Opaque flight pricing** makes it difficult for travellers to judge whether a quoted fare is fair or whether waiting a few more days will yield a better deal. Airlines use dynamic pricing models internally, but consumers have no equivalent tool. A regression model trained on historical flight data can fill this gap, giving travellers an independent estimate and a confidence range.

**2. One-size-fits-all experiences** erode engagement. Travel platforms that cannot distinguish a frequent business traveller from an occasional leisure tourist will surface irrelevant promotions and lose user trust. A classification model that infers basic demographic and behavioural attributes from booking history enables more targeted personalisation.

**3. Hotel discovery is fragmented.** Most recommendation engines rely on collaborative filtering, which fails for new users and new properties. A content-based approach — representing hotels by their metadata and ranking candidates by textual similarity to a user's stated preferences — sidesteps the cold-start problem and scales to arbitrarily large catalogues.

Beyond the modelling challenges, there is a systemic problem: the industry builds models that live in notebooks and never reach production. This project addresses that gap directly by implementing a full MLOps lifecycle.



---

## Models Built

### 1. Flight Fare Regression (XGBoost)

Predicts the ticket price for a given flight based on origin, destination, cabin class, flight duration, days until departure, and number of stops. Two derived features are added: a booking-urgency proxy (duration / days_ahead) and a binary direct-flight indicator. The model is trained with 5-fold cross-validation and evaluated on a held-out 20% test set.

**Key metrics:** R², RMSE, MAE

### 2. Traveller Gender Classification (Random Forest / Gradient Boosting)

Infers traveller gender from behavioural attributes — travel frequency, average trip spend, preferred cabin class, loyalty programme membership, and typical trip purpose. Two candidate models are trained and compared via stratified 5-fold CV; the winner is saved as the production classifier.

**Key metrics:** Accuracy, Precision, Recall, F1-Score (weighted), ROC-AUC

### 3. Hotel Recommendation Engine (TF-IDF + Cosine Similarity)

Builds a TF-IDF matrix over a concatenated representation of hotel name, city, category, amenities, and description text. At inference time, a free-text preference query is vectorised using the same vocabulary and ranked against all hotels by cosine similarity. Self-retrieval Precision@K is used as the offline evaluation metric.

**Key metric:** Precision@K (self-retrieval)

## How to Run This Project

> **Prerequisites:** Python 3.11+, Docker Desktop, kubectl (configured), Airflow 2.x, Jenkins, MLflow server (or local SQLite store).

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/journeyiq.git
cd journeyiq
```

### Step 2 — Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Add your datasets

Place `flights.csv`, `users.csv`, and `hotels.csv` inside the `data/` directory. The expected columns are described in each training script's docstring.

### Step 4 — Train the models

Each script can be run independently. Run all three to populate the `models/` directory:

```bash
python train_fare_model.py     --data data/flights.csv
python train_gender_model.py   --data data/users.csv
python train_recommender.py    --data data/hotels.csv
```

### Step 5 — Launch the Flask API

```bash
python api_server.py
# API is now live at http://localhost:5050/api/v1/
```

Test it with curl:

```bash
curl -X POST http://localhost:5050/api/v1/predict/fare \
  -H "Content-Type: application/json" \
  -d '{"origin":"DEL","destination":"BOM","travel_class":"economy","duration_hours":2.5,"days_until_flight":14,"num_stops":0}'
```

### Step 6 — Launch the Streamlit dashboard

```bash
streamlit run streamlit_dashboard.py
# Dashboard opens at http://localhost:8501
```

### Step 7 — View MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_journeyiq.db
# MLflow UI at http://localhost:5000
```

### Step 8 — Build and run with Docker

```bash
# Build the container
docker build -t journeyiq-fare-api:local .

# Run it locally
docker run -p 5050:5050 \
  -v $(pwd)/models:/app/models \
  journeyiq-fare-api:local
```

### Step 9 — Deploy on Kubernetes

```bash
# Apply all manifests (namespace, deployment, services, HPA, PVC)
kubectl apply -f deployment.yaml

# Watch the rollout
kubectl rollout status deployment/journeyiq-fare-api -n journeyiq

# Check pod status
kubectl get pods -n journeyiq
```

### Step 10 — Configure Apache Airflow

Copy `airflow_dag.py` into your Airflow `dags/` directory and set the following environment variables before starting the scheduler:

```bash
export JOURNEYIQ_DATA_DIR=/path/to/data
export JOURNEYIQ_MODELS_DIR=/path/to/models
export JOURNEYIQ_TRAIN_DIR=/path/to/project
airflow scheduler &
airflow webserver --port 8080
```

The DAG `journeyiq_ml_pipeline` will appear in the Airflow UI and run daily at 02:30 UTC.

### Step 11 — Configure Jenkins

Create a new Pipeline job in Jenkins and point it at this repository's `Jenkinsfile`. Add the following credentials in Jenkins Credential Store:

| Credential ID | Type | Purpose |
|---|---|---|
| `DOCKER_CREDENTIALS_ID` | Username/Password | Docker Hub push access |
| `KUBECONFIG_CREDENTIAL` | Secret File | kubectl cluster access |
| `SLACK_WEBHOOK_URL` | Secret Text | Slack notification webhook |

---

## API Reference

### `GET /api/v1/health`

Returns the service liveness status.

**Response:**
```json
{ "status": "ready", "service": "journeyiq-fare-api" }
```

### `POST /api/v1/predict/fare`

Predicts the estimated flight fare.

**Request body:**
```json
{
  "origin": "DEL",
  "destination": "BOM",
  "travel_class": "economy",
  "duration_hours": 2.5,
  "days_until_flight": 14,
  "num_stops": 0
}
```

**Response:**
```json
{
  "predicted_fare_inr": 4823.75,
  "confidence_band": { "low": 4245.00, "high": 5402.60 },
  "model_version": "v1.0"
}
```

---

## Evaluation Metrics

| Model | Primary Metric | Secondary Metrics |
|---|---|---|
| Fare Regression | R² Score | RMSE, MAE |
| Gender Classification | Weighted F1 | Accuracy, Precision, Recall, ROC-AUC |
| Hotel Recommendation | Precision@K | Cosine similarity distribution |

---

## Challenges & Decisions

**Multiple datasets, one coherent system.** Flights, users, and hotels are related by `travelCode` and `userCode` but serve completely different models. Keeping the training pipelines modular and independent was deliberate — it avoids a cascading failure where a bug in the recommender breaks the fare model's preprocessing.

**API design trade-offs.** Using Flask Blueprints adds some boilerplate compared to a minimal single-file Flask app, but the separation pays off as the service grows. Validation happens before the model is ever touched, so the model code stays clean.

**Container security.** Running the container as a non-root user and scanning with Trivy in the Jenkins pipeline are steps that are often skipped in academic projects but are non-negotiable in production. Adding them here reflects real production thinking.

**Airflow branching.** The `BranchPythonOperator` in the DAG means the pipeline can make a data-driven decision about whether to promote a new model without human intervention — unless the model regresses, in which case it simply archives the new version and keeps the old one live.

---

## Future Improvements

- Add a **model monitoring** step that detects data drift using the Evidently library and triggers retraining when drift crosses a threshold.
- Implement **JWT authentication** on the Flask API to prevent unauthorised access in a multi-tenant deployment.
- Replace the SQLite MLflow backend with a **PostgreSQL + S3 artefact store** for team-scale usage.
- Extend the hotel recommender with a **collaborative filtering layer** that blends content similarity with user-item interaction signals once enough booking history accumulates.
- Add **A/B testing** infrastructure so two model versions can serve live traffic simultaneously and be compared by downstream business metrics (e.g., booking conversion rate).
- Explore **cloud-native deployment** on GKE or EKS with Terraform-managed infrastructure as code.

---

*JourneyIQ — Masters Project, MLOps Specialisation*
