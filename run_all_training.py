import subprocess
import sys
import os

scripts = [
    ("Fare Regression Model",        ["python", "train_fare_model.py",     "--data", "data/flights.csv"]),
    ("Gender Classification Model",  ["python", "train_gender_model.py",   "--data", "data/users.csv"]),
    ("Hotel Recommendation Engine",  ["python", "train_recommender.py",    "--data", "data/hotels.csv"]),
]

print("="*62)
print("  JourneyIQ — Full Training Pipeline")
print("="*62)

all_ok = True
for name, cmd in scripts:
    print(f"\n[STARTING] {name}")
    print("-"*50)
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[FAILED] {name} — exit code {result.returncode}")
        all_ok = False
    else:
        print(f"[DONE] {name}")

print("\n" + "="*62)
if all_ok:
    print("  ALL MODELS TRAINED SUCCESSFULLY")
    print("="*62)
    print()
    print("  Next steps:")
    print()
    print("  1. View MLflow runs:")
    print("     mlflow ui --backend-store-uri sqlite:///mlflow_journeyiq.db")
    print("     → Open http://localhost:5000")
    print()
    print("  2. Start the Flask API (new terminal):")
    print("     python api_server.py")
    print()
    print("  3. Start the Streamlit dashboard (new terminal):")
    print("     streamlit run streamlit_dashboard.py")
    print("     → Open http://localhost:8501")
else:
    print("  SOME MODELS FAILED — check output above")
print("="*62)
