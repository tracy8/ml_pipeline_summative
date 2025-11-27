import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time


# --------------------------------------
# Load classifier
# --------------------------------------
@st.cache_resource
def load_classifier():
    model_path = os.path.join("models", "best_model.h5")

    if not os.path.exists(model_path):
        st.error("Error: best_model.h5 not found inside /models/")
        st.stop()

    model = load_model(model_path)

    encoder = LabelEncoder()
    encoder.fit(["A", "B", "C", "D", "E", "F"])

    return model, encoder


model, encoder = load_classifier()

# --------------------------------------
# Streamlit Page Setup
# --------------------------------------
st.set_page_config(page_title="Grade Classifier (A–F)", layout="centered")

st.title("Grade Classifier (A–F)")
st.write("This application classifies characters into grades A–F using a trained deep learning model.")

# --------------------------------------
# MODEL STATUS
# --------------------------------------
st.subheader("Model Status")
st.success("Model loaded successfully and ready for prediction.")


# --------------------------------------
# IMAGE PREDICTION
# --------------------------------------
st.subheader("Image Prediction")
st.write("Upload a 28×28 character image to predict its grade.")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded:
    file_bytes = uploaded.read()

    st.image(file_bytes, width=150, caption="Uploaded Image")

    # Preprocess
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("Invalid image format")
        st.stop()

    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0

    # EMNIST orientation fix
    img = np.rot90(img, k=1)
    img = np.fliplr(img)

    img = img.reshape(1, 28, 28, 1)

    # Predict
    preds = model.predict(img)
    idx = int(np.argmax(preds, axis=1)[0])
    grade = encoder.inverse_transform([idx])[0]

    st.success(f"Predicted Grade: **{grade}**")


# --------------------------------------
# DATASET SECTION
# --------------------------------------
st.subheader("Dataset Information")
st.write("""
This model uses the **EMNIST Letters** dataset.  
For this project, only the letters **A to F** were used.
Each image is 28×28 pixels and grayscale.
""")


# --------------------------------------
# MODEL ARCHITECTURE
# --------------------------------------
st.subheader("Model Architecture")
st.write("""
The classifier is a Convolutional Neural Network (CNN) with:

- Conv2D + MaxPooling layers  
- Flatten layer  
- Dense layers  
- Softmax output for 6 classes (A–F)
""")


# --------------------------------------
# EVALUATION SECTION
# --------------------------------------
st.subheader("Model Evaluation")
st.write("""
The model was evaluated using accuracy and loss metrics.  
It achieved strong classification performance on the filtered EMNIST subset.
""")


# --------------------------------------
# SIMULATED RETRAINING (UI ONLY)
# --------------------------------------
st.subheader("Retraining (Simulation Only)")
st.write("This simulates retraining just for demo purposes.")

if st.button("Start Simulated Retraining"):
    st.info("Retraining started...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    for percent in range(0, 101, 10):
        progress_bar.progress(percent)
        status_text.write(f"Retraining in progress... {percent}%")
        time.sleep(0.25)

    st.success("Retraining completed (simulation only).")
