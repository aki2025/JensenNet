import asyncio
import numpy as np
from config import SLEEP_INTERVAL

async def real_time_data_stream(sleep_interval=SLEEP_INTERVAL):
    """Asynchronous real-time network data stream with configurable interval."""
    time_step = 0
    while True:
        try:
            rf_signal = 50 + 10 * np.sin(time_step / 100) + np.random.normal(0, 2).astype(np.float32)
            traffic_load = np.random.normal(50, 10).astype(np.float32)
            num_users = np.random.poisson(100).astype(np.int32)
            latency = 10 + 0.1 * traffic_load - 0.05 * rf_signal + np.random.normal(0, 1)
            throughput = 100 + 0.5 * rf_signal - 0.1 * num_users + np.random.normal(0, 5)
            anomaly = 1 if np.random.rand() < 0.05 else 0
            if anomaly:
                latency += np.random.normal(20, 5)
                throughput -= np.random.normal(30, 10)
            yield {
                "time_step": time_step, "rf_signal": rf_signal, "traffic_load": traffic_load,
                "num_users": num_users, "latency": latency, "throughput": throughput, "anomaly": anomaly
            }
            time_step += 1
            await asyncio.sleep(sleep_interval)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in data stream: {e}")
            await asyncio.sleep(sleep_interval)

if __name__ == "__main__":
    async def test_stream():
        stream = real_time_data_stream(sleep_interval=1)
        for _ in range(5):
            print(await stream.__anext__())
    asyncio.run(test_stream()) 
