import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import logging
from config import ANOMALY_MODEL_PATH, SCALER_PATH, THRESHOLD_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def build_autoencoder(input_dim):
    """Build a lightweight autoencoder for anomaly detection."""
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation="relu")(input_layer)
    encoded = layers.Dense(16, activation="relu")(encoded)
    decoded = layers.Dense(32, activation="relu")(encoded)
    decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)
    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder

def evaluate_model(model, X_test, y_test, threshold):
    """Evaluate anomaly detection performance with metrics."""
    predictions = model.predict(X_test, batch_size=32)
    mse = np.mean(np.square(X_test - predictions), axis=1)
    y_pred = (mse > threshold).astype(np.int8)
    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    logger.info(f"Evaluation Metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    try:
        data = pd.read_csv("data/historical_data.csv", compression="gzip")
        features = data[["rf_signal", "traffic_load", "num_users", "latency", "throughput"]].values.astype(np.float32)
        labels = data["anomaly"].values.astype(np.int8)
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
        
        autoencoder = build_autoencoder(features.shape[1])
        autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, validation_split=0.2, verbose=1)
        
        # Convert to TFLite for edge deployment
        converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(ANOMALY_MODEL_PATH, "wb") as f:
            f.write(tflite_model)
        
        train_predictions = autoencoder.predict(X_train, batch_size=64)
        train_mse = np.mean(np.square(X_train - train_predictions), axis=1)
        threshold = np.percentile(train_mse, 95)
        with open(THRESHOLD_PATH, "w") as f:
            f.write(str(threshold))
        
        evaluate_model(autoencoder, X_test, y_test, threshold)
    except Exception as e:
        logger.error(f"Error during training: {e}") 
