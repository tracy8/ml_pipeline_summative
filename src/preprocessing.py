import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_filter_emnist(csv_path):
    """
    Loads EMNIST CSV file and filters labels 1–6.
    Returns DataFrame with added grade column.
    """

    df = pd.read_csv(csv_path, header=None)

    allowed_labels = [1, 2, 3, 4, 5, 6]
    df = df[df[0].isin(allowed_labels)].copy()

    label_map = {1:"A", 2:"B", 3:"C", 4:"D", 5:"E", 6:"F"}
    df["grade"] = df[0].map(label_map)

    return df


def preprocess_images(df):
    """
    Extract pixel columns, normalize, reshape, rotate, and add channel dimension.
    Returns numpy array of processed images.
    """

    pixel_cols = list(range(1, 785))  # columns 1–784

    X = df.iloc[:, pixel_cols].values.astype("float32")
    X /= 255.0  # normalize

    # reshape
    X = X.reshape((-1, 28, 28))

    # fix rotation
    X = np.rot90(X, k=1, axes=(1, 2))
    X = np.fliplr(X)

    # add channel dim
    X = X[..., np.newaxis]

    return X


def encode_labels(df):
    """
    Encodes A–F labels into numeric 0–5.
    Returns encoded labels and encoder.
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["grade"])
    return y, encoder
