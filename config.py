# Configuration file for project settings
import os

# Data generation settings
NUM_SAMPLES = 10000
SEQUENCE_LENGTH = 10

# Model settings
ANOMALY_MODEL_PATH = os.path.join("models", "anomaly_detection_model.tflite")
MAINTENANCE_MODEL_PATH = os.path.join("models", "predictive_maintenance_model.tflite")
SCALER_PATH = os.path.join("models", "scaler.pkl")
THRESHOLD_PATH = os.path.join("models", "threshold.txt")

# Application settings
SLEEP_INTERVAL = 0.1  # in seconds
LOG_LEVEL = "INFO" 
