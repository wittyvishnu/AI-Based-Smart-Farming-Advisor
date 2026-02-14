import os
import logging
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from werkzeug.exceptions import BadRequest

# ---------------------------
# App Configuration
# ---------------------------

app = Flask(__name__)
CORS(app)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------
# Load ML Artifacts
# ---------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

try:
    model = load_model(os.path.join(MODEL_DIR, "irrigation_ann_model.h5"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    logger.info("Model and preprocessors loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model files: {str(e)}")
    raise e


# ---------------------------
# Health Check Endpoint
# ---------------------------

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


# ---------------------------
# Prediction Endpoint
# ---------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            raise BadRequest("Request must be JSON")

        data = request.get_json()

        required_fields = [
            "CropType",
            "CropDays",
            "SoilMoisture",
            "temperature",
            "Humidity"
        ]

        for field in required_fields:
            if field not in data:
                raise BadRequest(f"Missing field: {field}")

        # Encode crop type
        crop_type = data["CropType"]

        if crop_type not in label_encoder.classes_:
            raise BadRequest(f"Invalid CropType. Allowed: {list(label_encoder.classes_)}")

        crop_type_encoded = label_encoder.transform([crop_type])[0]

        features = np.array([[
            crop_type_encoded,
            float(data["CropDays"]),
            float(data["SoilMoisture"]),
            float(data["temperature"]),
            float(data["Humidity"])
        ]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled, verbose=0)
        probability = float(prediction[0][0])

        irrigation = int(probability > 0.6)

        return jsonify({
            "Irrigation": irrigation,
            "Probability": round(probability, 4)
        }), 200

    except BadRequest as e:
        logger.warning(f"Client error: {str(e)}")
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        logger.exception("Server error during prediction")
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------
# Error Handlers
# ---------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500
