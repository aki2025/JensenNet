import unittest
import numpy as np
import tensorflow as tf
import asyncio
import time
from utils.data_preprocessing import generate_historical_data, generate_maintenance_data
from utils.real_time_data_stream import real_time_data_stream
from applications.network_optimization_app import NetworkOptimizer
from applications.predictive_maintenance_app import MaintenancePredictor
from config import ANOMALY_MODEL_PATH, SCALER_PATH, THRESHOLD_PATH, SEQUENCE_LENGTH

class TestNetworkOptimization(unittest.TestCase):
    def setUp(self):
        """Setup common resources for all tests."""
        self.interpreter_anomaly = tf.lite.Interpreter(model_path=ANOMALY_MODEL_PATH)
        self.interpreter_anomaly.allocate_tensors()
        self.interpreter_maintenance = tf.lite.Interpreter(model_path=MAINTENANCE_MODEL_PATH)
        self.interpreter_maintenance.allocate_tensors()
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        with open(THRESHOLD_PATH, "r") as f:
            self.threshold = float(f.read())
        self.anomaly_input_details = self.interpreter_anomaly.get_input_details()
        self.anomaly_output_details = self.interpreter_anomaly.get_output_details()
        self.maintenance_input_details = self.interpreter_maintenance.get_input_details()
        self.maintenance_output_details = self.interpreter_maintenance.get_output_details()

    def test_generate_historical_data(self):
        """Test historical data generation with validation."""
        data = generate_historical_data(num_samples=100)
        self.assertEqual(len(data), 100)
        self.assertTrue(all(col in data.columns for col in ["rf_signal", "latency", "anomaly"]))
        self.assertFalse(data.isnull().values.any())

    def test_generate_maintenance_data(self):
        """Test maintenance data generation with validation."""
        data = generate_maintenance_data(num_samples=100)
        self.assertEqual(len(data), 100)
        self.assertTrue(all(col in data.columns for col in ["temperature", "error_rate", "failure"]))
        self.assertFalse(data.isnull().values.any())

    async def async_stream_test(self):
        """Helper for testing real-time stream."""
        stream = real_time_data_stream(sleep_interval=0.1)
        data = await stream.__anext__()
        return data

    def test_real_time_data_stream(self):
        """Test real-time data stream output."""
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(self.async_stream_test())
        self.assertTrue(all(key in data for key in ["rf_signal", "latency", "anomaly"]))

    def test_anomaly_detection_normal(self):
        """Test anomaly detection with normal data."""
        normal_data = np.array([[50, 50, 100, 10, 100]], dtype=np.float32)
        scaled_data = self.scaler.transform(normal_data)
        self.interpreter_anomaly.set_tensor(self.anomaly_input_details[0]["index"], scaled_data)
        self.interpreter_anomaly.invoke()
        reconstruction = self.interpreter_anomaly.get_tensor(self.anomaly_output_details[0]["index"])
        mse = np.mean(np.square(scaled_data - reconstruction))
        self.assertLess(mse, self.threshold)

    def test_anomaly_detection_anomaly(self):
        """Test anomaly detection with anomalous data."""
        anomaly_data = np.array([[30, 70, 150, 30, 70]], dtype=np.float32)
        scaled_data = self.scaler.transform(anomaly_data)
        self.interpreter_anomaly.set_tensor(self.anomaly_input_details[0]["index"], scaled_data)
        self.interpreter_anomaly.invoke()
        reconstruction = self.interpreter_anomaly.get_tensor(self.anomaly_output_details[0]["index"])
        mse = np.mean(np.square(scaled_data - reconstruction))
        self.assertGreater(mse, self.threshold)

    def test_anomaly_inference_time(self):
        """Test anomaly detection inference performance."""
        data = np.random.rand(100, 5).astype(np.float32)
        scaled_data = self.scaler.transform(data)
        start_time = time.time()
        for i in range(100):
            self.interpreter_anomaly.set_tensor(self.anomaly_input_details[0]["index"], scaled_data[i:i+1])
            self.interpreter_anomaly.invoke()
        avg_time = (time.time() - start_time) / 100
        print(f"Anomaly inference time: {avg_time:.4f} seconds")
        self.assertLess(avg_time, 0.01)

    def test_maintenance_prediction(self):
        """Test maintenance prediction with sample sequence."""
        sequence = np.random.rand(SEQUENCE_LENGTH, 2).astype(np.float32).flatten().reshape(1, -1)
        self.interpreter_maintenance.set_tensor(self.maintenance_input_details[0]["index"], sequence)
        self.interpreter_maintenance.invoke()
        prediction = self.interpreter_maintenance.get_tensor(self.maintenance_output_details[0]["index"])[0][0]
        self.assertTrue(0 <= prediction <= 1)

    def test_maintenance_inference_time(self):
        """Test maintenance prediction inference performance."""
        data = np.random.rand(100, SEQUENCE_LENGTH * 2).astype(np.float32)
        start_time = time.time()
        for i in range(100):
            self.interpreter_maintenance.set_tensor(self.maintenance_input_details[0]["index"], data[i:i+1])
            self.interpreter_maintenance.invoke()
        avg_time = (time.time() - start_time) / 100
        print(f"Maintenance inference time: {avg_time:.4f} seconds")
        self.assertLess(avg_time, 0.01)

    async def async_optimizer_test(self):
        """Helper for testing network optimizer."""
        optimizer = NetworkOptimizer(self.threshold)
        await asyncio.sleep(1)  # Run for 1 second
        return True

    def test_network_optimizer(self):
        """Test network optimization app."""
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.async_optimizer_test())
        self.assertTrue(result)

    async def async_predictor_test(self):
        """Helper for testing maintenance predictor."""
        predictor = MaintenancePredictor(sequence_length=SEQUENCE_LENGTH)
        await asyncio.sleep(1)  # Run for 1 second
        return True

    def test_maintenance_predictor(self):
        """Test predictive maintenance app."""
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.async_predictor_test())
        self.assertTrue(result)

    def test_missing_model_file(self):
        """Test handling of missing model file."""
        with self.assertRaises(FileNotFoundError):
            tf.lite.Interpreter(model_path="nonexistent.tflite")

if __name__ == "__main__":
    unittest.main() 
