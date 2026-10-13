import logging
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load unified model pipeline (prioritize compressed model for speed and lower slug size)
COMPRESSED_MODEL_PATH = BASE_DIR / "model" / "model_compressed.pkl"
RAW_MODEL_PATH = BASE_DIR / "model" / "model.pkl"

if COMPRESSED_MODEL_PATH.exists():
    MODEL_PATH = COMPRESSED_MODEL_PATH
elif RAW_MODEL_PATH.exists():
    MODEL_PATH = RAW_MODEL_PATH
else:
    MODEL_PATH = None

model = None
model_load_error = None

if MODEL_PATH:
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model pipeline successfully loaded from: {MODEL_PATH}")
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"Failed to load model from {MODEL_PATH}: {e}", exc_info=True)
else:
    model_load_error = (
        f"Model file not found. Expected {COMPRESSED_MODEL_PATH} or {RAW_MODEL_PATH}."
    )
    logger.error(model_load_error)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH else None,
        "error": model_load_error,
    }), (200 if model is not None else 503)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    prediction_units = 2.45
    has_predicted = False
    form_data = {
        "longitude": "-122.23",
        "latitude": "37.88",
        "housing_median_age": "15",
        "total_rooms": "880",
        "total_bedrooms": "129",
        "population": "322",
        "households": "126",
        "median_income": "8.3252",
        "ocean_proximity": "<1H OCEAN",
    }

    if request.method == "POST":
        data = request.get_json(silent=True) or request.form

        try:
            if model is None:
                raise RuntimeError(
                    f"Prediction model is not available: {model_load_error or 'Model not loaded'}"
                )

            longitude = float(data.get("longitude", -122.23))
            latitude = float(data.get("latitude", 37.88))
            housing_median_age = float(data.get("housing_median_age", 15))
            total_rooms = float(data.get("total_rooms", 880))
            total_bedrooms = float(data.get("total_bedrooms", 129))
            population = float(data.get("population", 322))
            households = float(data.get("households", 126))
            median_income = float(data.get("median_income", 8.3252))
            ocean_proximity = str(data.get("ocean_proximity", "<1H OCEAN"))

            form_data = {
                "longitude": str(longitude),
                "latitude": str(latitude),
                "housing_median_age": str(housing_median_age),
                "total_rooms": str(total_rooms),
                "total_bedrooms": str(total_bedrooms),
                "population": str(population),
                "households": str(households),
                "median_income": str(median_income),
                "ocean_proximity": ocean_proximity,
            }

            input_data = pd.DataFrame([{
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity
            }])

            raw_pred = model.predict(input_data)[0]
            prediction = round(float(raw_pred), 2)
            prediction_units = round(prediction / 100000, 2)
            has_predicted = True

            if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({
                    "success": True,
                    "prediction": prediction,
                    "prediction_units": prediction_units,
                    "formatted": f"${prediction:,.2f}",
                })

        except Exception as e:
            logger.error(f"Prediction handling error: {e}", exc_info=True)
            if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"success": False, "error": str(e)}), 400

    return render_template(
        "index.html",
        prediction=prediction,
        prediction_units=prediction_units,
        has_predicted=has_predicted,
        form_data=form_data,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)