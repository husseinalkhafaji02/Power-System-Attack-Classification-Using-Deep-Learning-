# Power-System-Attack-Classification-Using-Deep-Learning-
A Federated Learning framework for Cyber-Physical System security. Uses CNNs to classify power grid events as either Natural Failures (weather/equipment) or Cyberattacks (data manipulation) while preserving data privacy.
# Federated Learning for Power System Security ⚡🛡️

## 📌 Project Overview
This research project explores the application of **Deep Learning (CNNs)** and **Federated Learning (FL)** to secure Cyber-Physical Systems (CPS). 

The core objective is to build a robust detection system capable of distinguishing between routine power grid failures and malicious cyberattacks, without requiring utility companies to share sensitive raw operational data.

## 🎯 Key Research Objectives
As outlined in the project scope, this repository focuses on three main goals:

1.  **Event Classification:**
    * Build a machine learning model to accurately classify power system events into two categories:
        * **Natural Occurrences:** Equipment failures, weather-related outages, and line faults.
        * **Cyberattacks:** Data manipulation, false data injection attacks (FDIA), and spoofed signals.
2.  **Model Comparison:**
    * Evaluate whether Deep Learning architectures—specifically **Convolutional Neural Networks (CNNs)**—can outperform traditional statistical models in detecting complex, non-linear attack patterns.
3.  **Federated Implementation:**
    * Deploy these models in a **Federated Learning** environment. This allows multiple grid nodes (or distinct utility providers) to collaboratively train a global security model without centralizing or exposing their private sensor data.

## 📂 Repository Structure
* **`networkmodel.py`**: Defines the 1D-CNN architecture optimized for processing power system time-series data.
* **`FL.py`**: Simulates a local grid node training on private event logs (Natural vs. Attack).
* **`main.py`**: The central server logic that aggregates model weights from different nodes to improve global detection accuracy.
* **`preprocess.py`**: Handles data cleaning and correctness 
* **`Final_Presentation.pdf`**: A slide deck summarizing the research methodology, confusion matrices, and accuracy results.

## 🚀 Why This Matters
Modern power grids are increasingly digitized, making them vulnerable to cyber threats. Traditional protection systems often fail to distinguish between a tree falling on a line (Natural) and a hacker manipulating sensor values (Cyberattack). 

By using **Federated Learning**, this project proposes a solution where security insights are shared across the grid, but sensitive operational data remains local—solving both the **security** and **privacy** challenges of the modern smart grid.

## 🛠️ Usage
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt` (PyTorch, NumPy, Scikit-Learn).
3.  Run the classification demo:
    ```bash
    python main.py --mode federated --epochs 50
    ```
