

from flask import Flask, Blueprint, request, jsonify
import joblib
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("journeyiq.api")

predict_bp = Blueprint("predict", __name__, url_prefix="/api/v1")

MODEL_PATH   = os.environ.get("MODEL_PATH",   "models/flight_price_model.joblib")
ENCODER_PATH = os.environ.get("ENCODER_PATH", "models/label_encoders.joblib")
FEATURES_PATH = "models/fare_feature_cols.joblib"

try:
    fare_model    = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODER_PATH)
    FEATURE_COLS  = joblib.load(FEATURES_PATH)
    logger.info("Models loaded. Features: %s", FEATURE_COLS)
except FileNotFoundError as e:
    logger.warning("Model file not found: %s. Run train_fare_model.py first.", e)
    fare_model     = None
    label_encoders = {}
    FEATURE_COLS   = ["from", "to", "flightType", "agency", "time", "distance", "speed_proxy"]

# Valid categorical values
VALID_FLIGHT_TYPES = ["firstclass", "economic", "premium"]
VALID_AGENCIES     = ["flyingdrops", "cloudfy", "rainbow"]

REQUIRED_FIELDS = {
    "from":       str,
    "to":         str,
    "flightType": str,
    "time":       float,
    "distance":   float,
    "agency":     str,
}


def validate_payload(payload: dict) -> list:
    errors = []
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in payload:
            errors.append(f"Missing field: '{field}'")
        else:
            try:
                expected_type(payload[field])
            except (ValueError, TypeError):
                errors.append(f"'{field}' must be {expected_type.__name__}")
    return errors


def build_feature_vector(payload: dict) -> np.ndarray:
    """
    Build the feature vector in exactly the same order as FEATURE_COLS:
    [from, to, flightType, agency, time, distance, speed_proxy]
    """
    categorical_cols = ["from", "to", "flightType", "agency"]
    row = {}

    for col in categorical_cols:
        enc = label_encoders.get(col)
        val = str(payload[col]).strip()
        if enc:
            try:
                row[col] = int(enc.transform([val])[0])
            except ValueError:
                row[col] = -1
        else:
            row[col] = -1

    time_val     = float(payload["time"])
    distance_val = float(payload["distance"])
    row["time"]        = time_val
    row["distance"]    = distance_val
    row["speed_proxy"] = distance_val / max(time_val, 0.1)

    return np.array([[row[f] for f in FEATURE_COLS]])


@predict_bp.route("/health", methods=["GET"])
def health_check():
    status = "ready" if fare_model is not None else "degraded (run train_fare_model.py)"
    return jsonify({"status": status, "service": "journeyiq-fare-api"}), 200


@predict_bp.route("/predict/fare", methods=["POST"])
def predict_fare():
    """
    Predict estimated flight fare.

    Request JSON example:
    {
        "from": "Recife (PE)",
        "to": "Sao Paulo (SP)",
        "flightType": "economic",
        "time": 1.76,
        "distance": 676.53,
        "agency": "FlyingDrops"
    }
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    errors = validate_payload(payload)
    if errors:
        return jsonify({"errors": errors}), 422

    if fare_model is None:
        return jsonify({
            "predicted_price": 957.0,
            "note": "Model not loaded — run train_fare_model.py first"
        }), 200

    try:
        fv         = build_feature_vector(payload)
        prediction = float(fare_model.predict(fv)[0])
        low        = round(prediction * 0.90, 2)
        high       = round(prediction * 1.10, 2)

        logger.info("Fare: %.2f | %s → %s | %s",
                    prediction, payload["from"], payload["to"], payload["flightType"])

        return jsonify({
            "predicted_price":   round(prediction, 2),
            "confidence_band":   {"low": low, "high": high},
            "currency":          "local",
            "model_version":     os.environ.get("MODEL_VERSION", "v1.0")
        }), 200

    except Exception as exc:
        logger.error("Prediction error: %s", str(exc))
        return jsonify({"error": "Prediction failed", "detail": str(exc)}), 500


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(predict_bp)

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "hint": "Try GET /api/v1/health"}), 404

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app  = create_app()
    logger.info("Starting JourneyIQ API on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
