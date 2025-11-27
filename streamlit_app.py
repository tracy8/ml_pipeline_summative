import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------
# PAGE SETUP
# -----------------------------------------
st.set_page_config(page_title="Grade Classifier", layout="centered")

st.title("Grade Classifier (A–F)")
st.write("Upload a character image and the model will classify it.")

# -----------------------------------------
# LOAD MODEL & LABEL ENCODER
# -----------------------------------------
@st.cache_resource
def load_classifier():
    model_path = os.path.join("models", "best_model.h5")
    model = load_model(model_path)

    encoder = LabelEncoder()
    encoder.fit(["A", "B", "C", "D", "E", "F"])

    return model, encoder


model, encoder = load_classifier()


# -----------------------------------------
# PREDICTION FUNCTION
# -----------------------------------------
def preprocess_image(image_bytes):
    img_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0

    # EMNIST orientation fix
    img = np.rot90(img, 1)
    img = np.fliplr(img)

    img = img.reshape(1, 28, 28, 1)
    return img


def predict_grade(image):
    preds = model.predict(image)
    class_index = int(np.argmax(preds))
    grade = encoder.inverse_transform([class_index])[0]
    return grade


# -----------------------------------------
# IMAGE UPLOAD
# -----------------------------------------
st.subheader("Upload an image")

uploaded_img = st.file_uploader("Upload a 28x28 character", type=["png", "jpg", "jpeg"])

if uploaded_img:
    img_bytes = uploaded_img.read()
    st.image(img_bytes, caption="Uploaded Image", width=180)

    if st.button("Predict Grade"):
        processed = preprocess_image(img_bytes)

        if processed is None:
            st.error("Error reading image.")
        else:
            result = predict_grade(processed)
            st.success(f"Predicted Grade: {result}")
