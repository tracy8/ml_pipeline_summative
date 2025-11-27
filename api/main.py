from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.model import load_trained_model, build_cnn_model
from src.prediction import predict_single_image


app = FastAPI(
    title="Grade Classifier API",
    description="Predict A–F grades + Retraining system",
    version="1.0"
)

# -------------------------------
# GLOBAL PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # /api
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # project root
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.h5")
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploaded")

os.makedirs(UPLOAD_DIR, exist_ok=True)


# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.get("/uptime")
async def uptime():
    return {"status": "running"}


# -------------------------------
# PREDICT SINGLE IMAGE
# -------------------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        img_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image file")

        # EMNIST preprocessing
        img = cv2.resize(img, (28, 28))
        img = img.astype("float32") / 255.0

        img = np.rot90(img, 1)
        img = np.fliplr(img)

        img = img.reshape(1, 28, 28, 1)

        model = load_trained_model(MODEL_PATH)

        encoder = LabelEncoder()
        encoder.fit(["A", "B", "C", "D", "E", "F"])

        prediction = predict_single_image(model, img, encoder)

        return {"predicted_grade": prediction}

    except Exception as e:
        print("PREDICTION ERROR:", e)
        return JSONResponse(status_code=400, content={"error": str(e)})


# -------------------------------
# RETRAINING ENDPOINT
# -------------------------------
@app.post("/retrain")
async def retrain_model():
    try:
        # Find uploaded CSV
        csv_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")]
        if not csv_files:
            raise ValueError("No CSV file uploaded for retraining")

        latest_csv = os.path.join(UPLOAD_DIR, csv_files[-1])
        print("Retraining using:", latest_csv)

        # Load CSV (no header)
        df = pd.read_csv(latest_csv, header=None)

        # Rename EMNIST columns
        df.rename(columns={0: "label"}, inplace=True)

        # 🔥 Filter only A–F (labels 1–6)
        df = df[df["label"].isin([1, 2, 3, 4, 5, 6])]

        if df.empty:
            raise ValueError("Filtered dataset is empty. CSV must contain labels 1–6 only.")

        # Prepare X and y
        y = df["label"].values
        X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0

        # Encode labels 1–6 → 0–5
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        # Build model
        model = build_cnn_model()

        # Train model
        model.fit(X, y_encoded, epochs=5, batch_size=32, verbose=1)

        # Save
        model.save(MODEL_PATH)

        return {"status": "success", "message": "Model retrained successfully"}

    except Exception as e:
        print("RETRAIN ERROR:", e)
        return JSONResponse(status_code=400, content={"error": str(e)})

