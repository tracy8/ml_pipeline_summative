import numpy as np

def predict_single_image(model, image, encoder):
    """
    Predict grade for a single EMNIST preprocessed image.
    Input must be (1, 28, 28, 1) or (28, 28, 1)
    """
    # If input has no batch dimension, add it
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    class_idx = int(np.argmax(preds, axis=1)[0])
    return encoder.inverse_transform([class_idx])[0]
