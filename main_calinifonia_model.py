import os
from pathlib import Path
import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_FILE = Path("./model/model.pkl")
COMPRESSED_MODEL_FILE = Path("./model/model_compressed.pkl")


def build_pipeline(num_attribute, cat_attribute):

    # Numerical pipeline
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    # Categorical pipeline
    cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        [("num", num_pipeline, num_attribute), ("cat", cat_pipeline, cat_attribute)]
    )

    # Full unified pipeline: preprocessing + model
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):

    # 1. Load dataset

    housing = pd.read_csv("./data/housing.csv")

    # 2. Create income category

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # 3. Stratified train/test split

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):

        strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)

        strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

    # 4. Save test data for inference

    strat_test_set.to_csv("./data/input.csv", index=False)

    # 5. Work on training data

    housing = strat_train_set.copy()

    # 6. Separate labels

    housing_labels = housing["median_house_value"].copy()

    housing_features = housing.drop("median_house_value", axis=1)

    # 7. Separate columns

    num_attribute = housing_features.drop("ocean_proximity", axis=1).columns.tolist()

    cat_attribute = ["ocean_proximity"]

    # 8. Build unified pipeline

    model = build_pipeline(num_attribute, cat_attribute)

    # 9. Fit full pipeline (preprocessing + model)

    model.fit(housing_features, housing_labels)

    # 10. Save unified model pipeline

    joblib.dump(model, MODEL_FILE)

    print("Model pipeline is trained and saved.")

    # 11. Save compressed model

    joblib.dump(model, COMPRESSED_MODEL_FILE, compress=3)

    print(
        "Original size:",
        round(MODEL_FILE.stat().st_size / (1024**2), 2),
        "MB",
    )

    print(
        "Compressed size:",
        round(
            COMPRESSED_MODEL_FILE.stat().st_size / (1024**2),
            2,
        ),
        "MB",
    )


else:

    # INFERENCE

    model = joblib.load(MODEL_FILE)

    # Load test/input data
    input_data = pd.read_csv("./data/input.csv")

    # Remove target column if it exists
    if "median_house_value" in input_data.columns:
        input_features = input_data.drop("median_house_value", axis=1)
    else:
        input_features = input_data

    # Make prediction
    prediction = model.predict(input_features)

    # Add prediction to input data
    input_data["prediction"] = prediction

    # Save output
    input_data.to_csv("./data/output.csv", index=False)

    # =========================
    # CHECK MODEL ACCURACY
    # =========================

    if "median_house_value" in input_data.columns:

        actual = input_data["median_house_value"]

        # MAE
        mae = mean_absolute_error(actual, prediction)

        # RMSE
        rmse = np.sqrt(mean_squared_error(actual, prediction))

        # R² Score
        r2 = r2_score(actual, prediction)

        print("\nModel Performance:")
        print("------------------")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R² Score : {r2:.4f}")
        print(f"R² Percentage : {r2 * 100:.2f}%")

    else:
        print("Actual target values are not available, so accuracy cannot be calculated.")

    print("\nInference complete.")
    print(prediction)

    # =========================
    # MODEL COMPRESSION
    # =========================

    model_path = MODEL_FILE
    compressed_path = COMPRESSED_MODEL_FILE

    if not compressed_path.exists() or model_path.stat().st_mtime > compressed_path.stat().st_mtime:
        joblib.dump(
            model,
            compressed_path,
            compress=3,
        )

    print(
        "\nOriginal size:",
        round(model_path.stat().st_size / (1024**2), 2),
        "MB",
    )

    print(
        "Compressed size:",
        round(
            compressed_path.stat().st_size / (1024**2),
            2,
        ),
        "MB",
    )