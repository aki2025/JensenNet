 import asyncio
import numpy as np
import tensorflow as tf
from config import MAINTENANCE_MODEL_PATH, SEQUENCE_LENGTH, SLEEP_INTERVAL
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

async def hardware_metrics_stream(sleep_interval=SLEEP_INTERVAL):
    """Asynchronous hardware metrics stream with configurable interval."""
    time_step = 0
    while True:
        try:
            temperature = 30 + 5 * np.sin(time_step / 100) + np.random.normal(0, 1).astype(np.float32)
            error_rate = np.random.exponential(0.01).astype(np.float32)
            yield {"temperature": temperature, "error_rate": error_rate}
            time_step += 1
            await asyncio.sleep(sleep_interval)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in metrics stream: {e}")
            await asyncio.sleep(sleep_interval)

class MaintenancePredictor:
    def __init__(self, sequence_length=SEQUENCE_LENGTH):
        try:
            self.interpreter = tf.lite.Interpreter(model_path=MAINTENANCE_MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.sequence = []
            self.sequence_length = sequence_length
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def predict(self):
        """Predict hardware failures asynchronously with severity scoring."""
        stream = hardware_metrics_stream()
        async for data_point in stream:
            try:
                self.sequence.append([data_point["temperature"], data_point["error_rate"]])
                if len(self.sequence) > self.sequence_length:
                    self.sequence.pop(0)
                if len(self.sequence) == self.sequence_length:
                    input_data = np.array(self.sequence, dtype=np.float32).flatten().reshape(1, -1)
                    self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
                    self.interpreter.invoke()
                    prediction = self.interpreter.get_tensor(self.output_details[0]["index"])[0][0]
                    severity = prediction * 100  # Simple severity score
                    if prediction > 0.5:
                        logger.info(f"Failure predicted (Prob: {prediction:.4f}, Severity: {severity:.2f}%)! Scheduling maintenance.")
                    else:
                        logger.info(f"Hardware normal (Prob: {prediction:.4f}, Severity: {severity:.2f}%)")
            except Exception as e:
                logger.error(f"Error during prediction: {e}")

async def main():
    try:
        predictor = MaintenancePredictor()
        await predictor.predict()
    except Exception as e:
        logger.error(f"Main loop failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
