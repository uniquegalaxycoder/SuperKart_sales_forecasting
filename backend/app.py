import io
import json

import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL = joblib.load("deployment_files/superkart_best_model.joblib")
with open("deployment_files/feature_schema.json") as f:
    SCHEMA = json.load(f)

REQUIRED_FEATURES = SCHEMA["numeric_features"] + SCHEMA["categorical_features"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


@app.route("/predict", methods=["POST"])
def predict_single():
    """
    Expects JSON body like:
    {
        "Product_Weight": 12.66,
        "Product_Sugar_Content": "Low Sugar",
        "Product_Allocated_Area": 0.027,
        "Product_MRP": 117.08,
        "Store_Size": "Medium",
        "Store_Location_City_Type": "Tier 2",
        "Store_Type": "Supermarket Type2",
        "Product_Id_char": "FD",
        "Store_Age_Years": 17,
        "Product_Type_Category": "Perishables"
    }
    """
    payload = request.get_json(force=True)

    missing = [f for f in REQUIRED_FEATURES if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    row = pd.DataFrame([{f: payload[f] for f in REQUIRED_FEATURES}])
    try:
        prediction = MODEL.predict(row)[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"predicted_sales": round(float(prediction), 2)})


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Expects a multipart/form-data request with a CSV file under key 'file'.
    CSV must contain all REQUIRED_FEATURES columns.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form key 'file'."}), 400

    file = request.files["file"]
    try:
        df = pd.read_csv(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {missing}"}), 400

    try:
        preds = MODEL.predict(df[REQUIRED_FEATURES])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    df["Predicted_Sales"] = preds.round(2)
    return jsonify({"predictions": df.to_dict(orient="records")})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
