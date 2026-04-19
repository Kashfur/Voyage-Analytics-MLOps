"""
JourneyIQ — Hotel Recommendation Engine
========================================
Dataset : hotels.csv  (40,552 rows)
Columns : travelCode, userCode, name, place, days, price, total, date

9 unique hotels across 9 Brazilian cities.
Builds a TF-IDF representation and ranks by cosine similarity.
Self-retrieval Precision@K is logged to MLflow as the offline metric.

Run:
    python train_recommender.py --data data/hotels.csv
Then open:
    mlflow ui --backend-store-uri sqlite:///mlflow_journeyiq.db
    → http://localhost:5000
"""

import argparse
import os
import logging
import pandas as pd
import numpy as np
import joblib
import mlflow

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("journeyiq.train_recommender")

parser = argparse.ArgumentParser()
parser.add_argument("--data",       default="data/hotels.csv")
parser.add_argument("--top_k",      type=int, default=5)
parser.add_argument("--experiment", default="journeyiq_hotel_recommender")
args = parser.parse_args()


def price_tier(price: float) -> str:
    """Map numeric price/night to descriptive text for TF-IDF vocabulary."""
    if price < 100:
        return "budget affordable cheap economy"
    elif price < 200:
        return "mid-range moderate comfortable"
    elif price < 270:
        return "premium comfortable upscale"
    else:
        return "luxury expensive high-end exclusive"


def build_hotel_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse 40k booking rows into 9 hotel-level profiles.
    Each profile gets a combined_text field for TF-IDF.
    """
    hotel_df = df.groupby("name").agg(
        place=("place",      "first"),
        avg_price=("price",  "mean"),
        avg_days=("days",    "mean"),
        booking_count=("travelCode", "count"),
        total_revenue=("total", "sum")
    ).reset_index()

    hotel_df["price_tier_text"] = hotel_df["avg_price"].apply(price_tier)

    # City name repeated so city-based queries rank correctly
    hotel_df["combined_text"] = hotel_df.apply(
        lambda r: (
            f"{r['name'].lower()} "
            f"{r['place'].lower()} {r['place'].lower()} "
            f"{r['price_tier_text']} "
            f"stay {int(round(r['avg_days']))} nights"
        ), axis=1
    )

    log.info("Hotel profiles:\n%s",
             hotel_df[["name","place","avg_price","booking_count"]].to_string(index=False))
    return hotel_df


def precision_at_k(tfidf_matrix, k: int) -> float:
    n = tfidf_matrix.shape[0]
    hits = 0
    for i in range(n):
        scores = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
        top_k  = scores.argsort()[::-1][:k]
        if i in top_k:
            hits += 1
    return hits / n


def main():
    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow_journeyiq.db")
    mlflow.set_experiment(args.experiment)

    log.info("Reading: %s", args.data)
    df = pd.read_csv(args.data)
    log.info("Shape: %s | Columns: %s", df.shape, df.columns.tolist())

    hotel_df = build_hotel_profiles(df)

    # ── TF-IDF vectorisation ───────────────────────────────────────────────────
    vectoriser = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    tfidf_matrix = vectoriser.fit_transform(hotel_df["combined_text"])
    log.info("TF-IDF matrix: %s", tfidf_matrix.shape)

    # ── Offline evaluation ─────────────────────────────────────────────────────
    p_at_k   = precision_at_k(tfidf_matrix, args.top_k)
    p_at_3   = precision_at_k(tfidf_matrix, 3)
    vocab_sz = len(vectoriser.vocabulary_)

    log.info("Precision@%d: %.4f", args.top_k, p_at_k)
    log.info("Precision@3 : %.4f", p_at_3)

    # ── Sample query demo ──────────────────────────────────────────────────────
    sample_queries = [
        "florianopolis luxury",
        "sao paulo budget",
        "rio de janeiro"
    ]
    demo_results = {}
    for q in sample_queries:
        q_vec  = vectoriser.transform([q])
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top1   = int(scores.argsort()[::-1][0])
        demo_results[q] = {
            "top_hotel": hotel_df.iloc[top1]["name"],
            "city":      hotel_df.iloc[top1]["place"],
            "score":     float(scores[top1])
        }
        print(f"  Query '{q}' → {hotel_df.iloc[top1]['name']} "
              f"({hotel_df.iloc[top1]['place']}) score={scores[top1]:.4f}")

    # ── MLflow logging ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="hotel_recommender_tfidf") as run:
        mlflow.log_param("model_type",     "TF-IDF + cosine similarity")
        mlflow.log_param("ngram_range",    "(1, 2)")
        mlflow.log_param("sublinear_tf",   True)
        mlflow.log_param("stop_words",     "english")
        mlflow.log_param("n_hotels",       len(hotel_df))
        mlflow.log_param("n_raw_rows",     len(df))
        mlflow.log_param("vocabulary_size",vocab_sz)
        mlflow.log_param("top_k",         args.top_k)

        mlflow.log_metric(f"precision_at_{args.top_k}", p_at_k)
        mlflow.log_metric("precision_at_3",              p_at_3)
        mlflow.log_metric("vocab_size",                  vocab_sz)

        # Log per-hotel booking stats as metrics
        for _, row in hotel_df.iterrows():
            clean_name = row["name"].replace(" ", "_")
            mlflow.log_metric(f"bookings_{clean_name}", int(row["booking_count"]))
            mlflow.log_metric(f"avg_price_{clean_name}", round(row["avg_price"], 2))

        # Save hotel metadata as an artifact
        os.makedirs("models", exist_ok=True)
        hotel_df.to_csv("models/hotel_profiles.csv", index=False)
        mlflow.log_artifact("models/hotel_profiles.csv")

        run_id = run.info.run_id
        log.info("MLflow run_id: %s", run_id)

    # ── Save joblib artefacts ──────────────────────────────────────────────────
    joblib.dump(tfidf_matrix, "models/hotel_tfidf_matrix.joblib")
    joblib.dump(hotel_df,     "models/hotels_metadata.joblib")
    joblib.dump(vectoriser,   "models/hotel_vectoriser.joblib")

    print("\n" + "="*62)
    print("  JourneyIQ Hotel Recommender — Training Complete")
    print("="*62)
    print(f"  Hotels indexed  : {len(hotel_df)}")
    print(f"  Vocab size      : {vocab_sz}")
    print(f"  Precision@{args.top_k}     : {p_at_k:.4f}")
    print(f"  Precision@3     : {p_at_3:.4f}")
    print(f"  MLflow run      : {run_id}")
    print("="*62)
    print("  View in MLflow UI:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow_journeyiq.db")
    print("  → http://localhost:5000")
    print("="*62)


if __name__ == "__main__":
    main()
