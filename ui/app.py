import streamlit as st
import requests
import os
import time


API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title=" Grade Classifier", layout="centered")

st.title(" Grade Classifier (A–F) ")
st.write("Upload images, upload CSV data, retrain the model, and check model status.")

# -------------------------------
# GLOBAL UPLOAD DIR (ROOT LEVEL)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploaded")

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ===============================
# 1. MODEL STATUS
# ===============================
try:
    response = requests.get(f"{API_URL}/uptime")
    if response.status_code == 200:
        st.success("Model is running")
    else:
        st.error("Model reachable but returned error")
except:
    st.error("API is offline — start FastAPI server.")


# ===============================
# 2. IMAGE PREDICTION
# ===============================
st.subheader("🔍 Predict Grade from Image")

img_file = st.file_uploader("Upload a 28x28 character", type=["png", "jpg", "jpeg"])

if img_file:
    img_bytes = img_file.read()
    st.image(img_bytes, caption="Uploaded Image", width=150)

    if st.button("Predict Grade"):
        response = requests.post(
            f"{API_URL}/predict-image",
            files={"file": ("image.png", img_bytes, "image/png")}
        )

        if response.status_code == 200:
            pred = response.json()["predicted_grade"]
            st.success(f"Predicted Grade: **{pred}**")
        else:
            st.error("Prediction failed.")


# ===============================
# 3. UPLOAD CSV FOR RETRAINING
# ===============================
st.subheader("📤 Upload Training CSV Data")

csv_file = st.file_uploader("Upload EMNIST train CSV", type=["csv"])

if csv_file:
    save_path = os.path.join(UPLOAD_DIR, csv_file.name)
    with open(save_path, "wb") as f:
        f.write(csv_file.getvalue())

    st.success(f"{csv_file.name} uploaded successfully!")



# ===============================
# 4. TRIGGER RETRAINING
# ===============================

st.subheader("🔄 Retrain Model")

if st.button("Start Retraining"):
    
    st.info("Starting retraining...")

    # Show progress bar + status text
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Make the progress bar simulate work WHILE backend is training
    for percent in range(0, 100, 8):
        progress_bar.progress(percent / 100)
        status_text.write(f"Training in progress... {percent}%")
        time.sleep(0.3)    # smooth animation

    # Call the backend retrain endpoint
    response = requests.post(f"{API_URL}/retrain")

    if response.status_code == 200:
        progress_bar.progress(100)
        st.success("Model retrained successfully!")
    else:
        progress_bar.progress(100)
        st.error("Retraining failed")
        status_text.write("❌ Error occurred during retraining")

