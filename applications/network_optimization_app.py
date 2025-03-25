import asyncio
import numpy as np
import tensorflow as tf
from utils.real_time_data_stream import real_time_data_stream
import pickle
import logging
from config import ANOMALY_MODEL_PATH, SCALER_PATH, THRESHOLD_PATH, SLEEP_INTERVAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class NetworkOptimizer:
    def __init__(self, threshold):
        try:
            self.interpreter = tf.lite.Interpreter(model_path=ANOMALY_MODEL_PATH)
            self.interpreter.allocate_tensors()
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
            self.threshold = threshold
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.anomaly_count = 0
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def optimize(self):
        """Monitor and optimize network asynchronously with realistic adjustments."""
        stream = real_time_data_stream(sleep_interval=SLEEP_INTERVAL)
        async for data_point in stream:
            try:
                features = np.array([[data_point["rf_signal"], data_point["traffic_load"], 
                                      data_point["num_users"], data_point["latency"], data_point["throughput"]]], 
                                    dtype=np.float32)
                features_scaled = self.scaler.transform(features)
                self.interpreter.set_tensor(self.input_details[0]["index"], features_scaled)
                self.interpreter.invoke()
                reconstruction = self.interpreter.get_tensor(self.output_details[0]["index"])
                mse = np.mean(np.square(features_scaled - reconstruction))
                
                if mse > self.threshold:
                    self.anomaly_count += 1
                    logger.info(f"Anomaly detected (MSE: {mse:.4f}, Count: {self.anomaly_count})")
                    await self._adjust_parameters(data_point)
                else:
                    logger.info(f"Network normal (MSE: {mse:.4f})")
            except Exception as e:
                logger.error(f"Error during optimization: {e}")

    async def _adjust_parameters(self, data_point):
        """Simulate realistic network adjustments based on data."""
        if data_point["rf_signal"] < 40:
            logger.info("Adjusting beamforming: increasing angle by 5° to improve signal strength")
        if data_point["traffic_load"] > 60:
            logger.info("Increasing spectrum allocation by 10MHz to handle high traffic load")
        await asyncio.sleep(0.01)

async def main():
    try:
        with open(THRESHOLD_PATH, "r") as f:
            threshold = float(f.read())
        optimizer = NetworkOptimizer(threshold)
        await optimizer.optimize()
    except Exception as e:
        logger.error(f"Main loop failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 
