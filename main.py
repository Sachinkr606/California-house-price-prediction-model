import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load unified model pipeline (prioritize compressed model for speed)
MODEL_PATH = (
    Path("./model/model_compressed.pkl")
    if Path("./model/model_compressed.pkl").exists()
    else Path("./model/model.pkl")
)
model = joblib.load(MODEL_PATH)


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
    app.run(debug=True)