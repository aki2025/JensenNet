import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from config import MAINTENANCE_MODEL_PATH, SEQUENCE_LENGTH
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def build_maintenance_model(input_dim):
    """Build a lightweight maintenance prediction model."""
    model = models.Sequential([
        layers.Dense(32, activation="relu", input_shape=(input_dim,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    try:
        data = pd.read_csv("data/maintenance_data.csv", compression="gzip")
        features = data[["temperature", "error_rate"]].values.astype(np.float32)
        labels = data["failure"].values.astype(np.int8)
        
        X, y = [], []
        for i in range(SEQUENCE_LENGTH, len(data)):
            seq = features[i-SEQUENCE_LENGTH:i].flatten()
            if len(seq) != SEQUENCE_LENGTH * 2:
                raise ValueError(f"Invalid sequence length at index {i}")
            X.append(seq)
            y.append(labels[i])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int8)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = build_maintenance_model(X.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(MAINTENANCE_MODEL_PATH, "wb") as f:
            f.write(tflite_model)
        
        y_pred = model.predict(X_test, batch_size=64)
        roc_auc = roc_auc_score(y_test, y_pred)
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
    except Exception as e:
        logger.error(f"Error during training: {e}") 
