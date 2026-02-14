from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import joblib
import numpy as np

app = Flask(__name__)

# Allow all origins (you can restrict later if needed)
CORS(app)

# Load model and preprocessors at startup
model = load_model("irrigation_ann_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ML Service Running"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

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
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Encode CropType
        crop_type_encoded = label_encoder.transform([data["CropType"]])[0]

        # Prepare input features
        features = np.array([[
            crop_type_encoded,
            float(data["CropDays"]),
            float(data["SoilMoisture"]),
            float(data["temperature"]),
            float(data["Humidity"])
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)

        probability = float(prediction[0][0])
        irrigation = 1 if probability > 0.6 else 0

        return jsonify({
            "Irrigation": irrigation,
            "Probability": round(probability, 4)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
