# Intelligent Edge-Based Network Optimization in Smart Cities

## Project Overview
This project harnesses artificial intelligence (AI) and edge computing to optimize 5G networks in smart cities. Leveraging a simulated NVIDIA Aerial SDK environment, it provides real-time monitoring, anomaly detection, and predictive maintenance for network infrastructure. The system dynamically adjusts network parameters to maintain performance and reliability, making it ideal for latency-sensitive, high-demand applications.

## Detailed Features
- **Real-Time Anomaly Detection**: Uses a lightweight autoencoder to identify network anomalies (e.g., signal interference, traffic overload) with low latency.
- **Predictive Maintenance**: Employs a sequence-based neural network to forecast hardware failures, enabling proactive maintenance scheduling.
- **Dynamic Network Optimization**: Adjusts parameters such as beamforming angles and spectrum allocation based on real-time conditions.
- **Edge Deployment**: Models are converted to TensorFlow Lite for efficient execution on resource-constrained edge devices.
- **Asynchronous Processing**: Utilizes Python's asyncio for non-blocking, concurrent data processing and optimization.
- **Configurable Design**: Centralized settings in `config.py` allow easy customization of data generation, model paths, and runtime behavior.
- **Comprehensive Testing**: Unit tests cover data generation, model inference, and application logic.

## In-Depth Use Case Examples Across Domains
1. **Telecommunications (Smart Cities)**:
   Scenario: A 5G network in a smart city experiences sudden latency spikes due to a festival increasing user density.
   Solution: The anomaly detection model identifies high traffic load and latency. The optimizer increases spectrum allocation by 10MHz and adjusts beamforming to focus signals on crowded areas, restoring performance within seconds.
   Impact: Ensures uninterrupted connectivity for emergency services and public Wi-Fi.

2. **Smart Manufacturing**:
   Scenario: A factory's 5G-connected robotic arms report rising error rates due to overheating base station components.
   Solution: The predictive maintenance model detects a failure probability exceeding 50% and schedules cooling system checks. Network optimization reduces load on affected hardware until repairs are completed.
   Impact: Prevents production halts, saving thousands in downtime costs.

3. **Autonomous Vehicles**:
   Scenario: A fleet of self-driving cars experiences signal drops in a tunnel, disrupting vehicle-to-everything (V2X) communication.
   Solution: The system detects low RF signal strength as an anomaly and adjusts beamforming angles to penetrate the tunnel, maintaining low-latency communication.
   Impact: Enhances road safety by ensuring continuous data exchange between vehicles and infrastructure.

4. **Healthcare (Remote Surgery)**:
   Scenario: During a remote surgery, network latency increases due to interference, risking patient safety.
   Solution: The anomaly detection system flags the issue, and the optimizer reallocates spectrum to prioritize surgical data streams, reducing latency to below 10ms.
   Impact: Guarantees reliable connectivity, critical for life-saving procedures.

## Advantages
1. **Ultra-Low Latency**: Edge-based processing delivers decisions in milliseconds, crucial for real-time applications.
2. **Cost Efficiency**: Reduces operational costs by minimizing manual interventions and preventing hardware failures.
3. **Scalability**: Modular architecture supports deployment across multiple edge devices with minimal adjustments.
4. **Energy Efficiency**: Lightweight TFLite models optimize resource use on edge hardware.
5. **Robustness**: Comprehensive error handling and logging ensure system reliability.

## Limitations
1. **Synthetic Data Dependency**: Currently relies on simulated data; real-world data integration is pending.
2. **Simulated SDK**: Uses a mock NVIDIA Aerial SDK; full integration requires hardware-specific adaptations.
3. **Single-Device Limitation**: Optimized for one edge device; multi-device setups need orchestration tools like Kubernetes.
4. **Static Thresholds**: Anomaly detection uses a fixed threshold, which may not adapt to all scenarios.

## Installation and Dependencies

### Prerequisites
- Python 3.8+

### Git Steps
1. Clone the Repository:
   ```
   git clone https://github.com/your-repo/smart_city_5g_optimization.git
   cd smart_city_5g_optimization
   ```
2. Install Dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Generate Synthetic Data:
   ```
   python utils/data_preprocessing.py --num_samples 10000
   ```
4. Train Models:
   ```
   python models/anomaly_detection_model.py
   python models/predictive_maintenance_model.py
   ```
5. Run the Application:
   ```
   python main.py
   ```
6. Execute Tests:
   ```
   python tests/test_network_optimization.py
   ```

### Dependencies
1. `numpy`: Numerical computations
2. `pandas`: Data manipulation
3. `tensorflow`: Model training and TFLite conversion
4. `scikit-learn`: Data preprocessing and evaluation metrics

### Expected Output
This section outlines the outputs you can expect when running various components of the project, providing insight into the system's behavior during execution.
Running the Entire Project (main.py)

**OUTPUT**: A continuous stream of log messages in the console reflecting real-time network and hardware status:
	
 	**Network Optimization:**
	"Network normal (MSE: 0.0012)" when no anomalies are detected.
	"Anomaly detected (MSE: 0.1234, Count: 5)" followed by adjustment messages like "Adjusting beamforming: increasing angle by 5°" when anomalies occur.
 	
  	**Predictive Maintenance:**
	"Hardware normal (Prob: 0.1234, Severity: 12.34%)" when no failures are predicted.
	"Failure predicted (Prob: 0.6789, Severity: 67.89%)! Scheduling maintenance." when a potential failure is detected.
 
**Note**: The applications run indefinitely until manually stopped (e.g., with Ctrl+C).

### Running Individual Components


1. **Data Generation** (utils/data_preprocessing.py)
	**Output**:
	Creates historical_data.csv and maintenance_data.csv in the data/ directory.
	Prints a confirmation message:
	Generated 10000 samples for historical and maintenance data.

2. **Model Training**

**Anomaly Detection (models/anomaly_detection_model.py):**
	**Output:**
		Training logs showing loss per epoch:
		Epoch 1/20 - Loss: 0.0456
		Epoch 2/20 - Loss: 0.0234
		...
		Saves anomaly_detection_model.tflite, scaler.pkl, and threshold.txt.
		Prints evaluation metrics:
		Evaluation Metrics: {'precision': 0.95, 'recall': 0.92, 'f1': 0.93}

**Predictive Maintenance (models/predictive_maintenance_model.py):**
	**Output:**
		Training logs showing loss and accuracy:
		Epoch 1/10 - Loss: 0.5678 - Accuracy: 0.7234
		Epoch 2/10 - Loss: 0.3456 - Accuracy: 0.8456
		...
		Saves predictive_maintenance_model.tflite.
		Prints the ROC-AUC score:
		ROC-AUC: 0.89

**3. Real-Time Applications**

**Network Optimization (applications/network_optimization_app.py):**
	**Output: **Continuous log messages such as:
		Network normal (MSE: 0.0012)
		Anomaly detected (MSE: 0.1234, Count: 5)
		Adjusting beamforming: increasing angle by 5°

	
 **Predictive Maintenance (models/predictive_maintenance_model.py):**
	Output: Continuous log messages such as:
		Hardware normal (Prob: 0.1234, Severity: 12.34%)
		Failure predicted (Prob: 0.6789, Severity: 67.89%)! Scheduling maintenance.

**4. Testing (tests/test_network_optimization.py)**
	**Output**:
	Test results indicating pass or fail:
	test_data_generation ... OK
	test_model_inference ... OK
	test_application_logic ... OK
	Performance metrics:
	Average inference time: 0.023 seconds

 ## Future Work and Enhancements
1. **Real-World Data Integration**: Incorporate actual 5G network logs and hardware telemetry.
2. **Multi-Device Orchestration:** Use containerization (Docker) and orchestration (Kubernetes) for large-scale deployments.
3. **Advanced Models:** Explore LSTMs or Transformers for improved time-series prediction.
4. **Adaptive Thresholding:** Implement dynamic anomaly detection thresholds based on historical trends.
5. **Hardware Integration:** Fully integrate with NVIDIA Aerial SDK and edge devices like Jetson Nano.

### Deployment Methods in Real-Time Production

 ## Edge Device Deployments
Setup Hardware: Use an edge device (e.g., NVIDIA Jetson Nano) with Python and TensorFlow Lite installed.
Transfer Files: Copy the project directory, including trained TFLite models, scaler.pkl, and threshold.txt.
Install Dependencies:
pip install -r requirements.txt
Run the Application:
python main.py
Monitoring: Configure logging to a file or remote server for real-time performance tracking.

 ## Cloud-Augmented Deployment
Hybrid Approach: Run lightweight inference on edge devices and offload heavy training or analytics to the cloud.
Steps:
Deploy edge components as above.
Set up a cloud server (e.g., AWS EC2) to periodically retrain models with new data.
Use a message queue (e.g., MQTT) to sync edge devices with cloud updates.


 ## Scaling with Containers

**Dockerize**:
Create a Dockerfile:
dockerfile

FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]

Build and run:

docker build -t 5g-optimization .
docker run -d 5g-optimization

**Orchestration**: Use Kubernetes to manage multiple edge containers, ensuring high availability and load balancing.

### References
NVIDIA Aerial SDK Official Documentation: NVIDIA Aerial SDK - Provides tools for 5G network simulation and optimization.
Developer Guide: Explore baseband processing and radio resource management.

### Study Resources
1. TensorFlow Lite: TensorFlow Lite Guide - Learn model optimization for edge deployment.
2. Asyncio: Python Asyncio Documentation - Master asynchronous programming in Python.
3. 5G Network Optimization: IEEE Paper - "AI-Based Network Management in 5G Systems" (example reference).
4. Edge Computing: [Book] "Edge Computing: A Primer" by W. Shi et al. - Understand edge deployment principles.
