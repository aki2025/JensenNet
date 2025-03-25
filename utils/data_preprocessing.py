import numpy as np
import pandas as pd
import argparse
from config import NUM_SAMPLES

def generate_historical_data(num_samples=NUM_SAMPLES):
    """Generate synthetic network data with validation."""
    np.random.seed(42)
    time_steps = np.arange(num_samples, dtype=np.int32)
    rf_signal = 50 + 10 * np.sin(time_steps / 100) + np.random.normal(0, 2, num_samples).astype(np.float32)
    traffic_load = np.cumsum(np.random.normal(0, 1, num_samples)) + 50
    num_users = np.random.poisson(100, num_samples).astype(np.int32)
    latency = 10 + 0.1 * traffic_load - 0.05 * rf_signal + np.random.normal(0, 1, num_samples)
    throughput = 100 + 0.5 * rf_signal - 0.1 * num_users + np.random.normal(0, 5, num_samples)
    anomaly = np.zeros(num_samples, dtype=np.int8)
    anomaly_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
    anomaly[anomaly_indices] = 1
    latency[anomaly_indices] += np.random.normal(20, 5, len(anomaly_indices))
    throughput[anomaly_indices] -= np.random.normal(30, 10, len(anomaly_indices))
    df = pd.DataFrame({
        "time_step": time_steps, "rf_signal": rf_signal, "traffic_load": traffic_load,
        "num_users": num_users, "latency": latency, "throughput": throughput, "anomaly": anomaly
    })
    if df.isnull().values.any():
        raise ValueError("Generated data contains NaN values")
    return df

def generate_maintenance_data(num_samples=NUM_SAMPLES):
    """Generate synthetic maintenance data with validation."""
    np.random.seed(42)
    time_steps = np.arange(num_samples, dtype=np.int32)
    temperature = 30 + 5 * np.sin(time_steps / 100) + np.random.normal(0, 1, num_samples).astype(np.float32)
    error_rate = np.random.exponential(0.01, num_samples).astype(np.float32)
    failure = np.zeros(num_samples, dtype=np.int8)
    failure_indices = np.random.choice(num_samples, int(num_samples * 0.01), replace=False)
    failure[failure_indices] = 1
    df = pd.DataFrame({
        "time_step": time_steps, "temperature": temperature, "error_rate": error_rate, "failure": failure
    })
    if df.isnull().values.any():
        raise ValueError("Generated maintenance data contains NaN values")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for 5G network optimization.")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES, help="Number of samples to generate")
    args = parser.parse_args()
    
    historical_data = generate_historical_data(args.num_samples)
    historical_data.to_csv("data/historical_data.csv", index=False, compression="gzip")
    
    maintenance_data = generate_maintenance_data(args.num_samples)
    maintenance_data.to_csv("data/maintenance_data.csv", index=False, compression="gzip")
    
    print(f"Generated {args.num_samples} samples for historical and maintenance data.") 
