import streamlit as st
import numpy as np
import cv2
import os
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


# ===============================
# Load the model (local-only)
# ===============================
@st.cache_resource
def load_classifier():
    model_path = os.path.join("models", "best_model.h5")

    if not os.path.exists(model_path):
        st.error("Model file not found: models/best_model.h5")
        st.stop()

    model = load_model(model_path)

    encoder = LabelEncoder()
    encoder.fit(["A", "B", "C", "D", "E", "F"])

    return model, encoder


model, encoder = load_classifier()


# ===============================
# PAGE SETUP
# ===============================
st.set_page_config(page_title="Grade Classifier", layout="centered")

st.title("Grade Classifier (A–F)")
st.write("Upload images, upload CSV data, simulate retraining, and check model status.")


# ===============================
# GLOBAL UPLOAD DIR
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploaded")

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ===============================
# 1. MODEL STATUS (SIMULATED)
# ===============================
st.subheader("Model Status")

# No FastAPI server here, so always show success
st.success("Model is running")


# ===============================
# 2. IMAGE PREDICTION (REAL)
# ===============================
st.subheader("Predict Grade from Image")

img_file = st.file_uploader("Upload a 28x28 character", type=["png", "jpg", "jpeg"])

if img_file:

    img_bytes = img_file.read()
    st.image(img_bytes, caption="Uploaded Image", width=150)

    if st.button("Predict Grade"):
        
        # Convert bytes → image array
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            st.error("Could not process image.")
        else:
            # Preprocessing
            img = cv2.resize(img, (28, 28))
            img = img.astype("float32") / 255.0
            img = np.rot90(img, k=1)
            img = np.fliplr(img)
            img = img.reshape(1, 28, 28, 1)

            preds = model.predict(img)
            idx = int(np.argmax(preds, axis=1)[0])
            grade = encoder.inverse_transform([idx])[0]

            st.success(f"Predicted Grade: **{grade}**")


# ===============================
# 3. CSV UPLOAD (SIMULATION)
# ===============================
st.subheader("Upload Training CSV Data")

csv_file = st.file_uploader("Upload EMNIST train CSV", type=["csv"])

if csv_file:
    save_path = os.path.join(UPLOAD_DIR, csv_file.name)
    with open(save_path, "wb") as f:
        f.write(csv_file.getvalue())

    st.success(f"{csv_file.name} uploaded successfully!")


# ===============================
# 4. RETRAINING (SIMULATED)
# ===============================
st.subheader("Retrain Model")

if st.button("Start Retraining"):

    st.info("Starting retraining... (simulation only)")

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Fake progress bar identical to your demo
    for percent in range(0, 100, 8):
        progress_bar.progress(percent / 100)
        status_text.write(f"Training in progress... {percent}%")
        time.sleep(0.3)

    progress_bar.progress(1.0)
    st.success("Model retrained successfully! (simulation only)")
