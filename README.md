# Dynamic Price Optimization Engine with Reinforcement Learning 🤖📈

This repository contains an end-to-end Machine Learning system designed to optimize e-commerce product pricing dynamically. The project combines **XGBoost** for demand forecasting and **Reinforcement Learning (Q-Learning)** for autonomous decision-making.

## 🚀 Overview
The goal of this project is to move away from static pricing and manual rule-based adjustments. Instead, an AI agent learns to maximize **Net Profit** by analyzing historical sales data, seasonality, and price elasticity.

### ✅ Key Strengths
* **End-to-End Pipeline:** Integrates raw data processing, demand modeling, and RL optimization in a single modular workflow.
* **Profit-Oriented Reward:** Unlike basic models that only chase revenue, this agent optimizes for **Net Profit** by incorporating Unit Economics (COGS).
* **Modular Architecture:** Clean separation of concerns between the environment, the forecaster, and the agent logic.
* **Model Persistence:** Includes built-in capabilities to save and load trained Q-tables for instant deployment.

### ⚠️ Current Scope & Limitations
* **State Space:** Currently utilizes a focused feature set `[price, is_weekend, month]`.
* **Methodology:** Employs tabular Q-Learning, which is ideal for understanding RL fundamentals before transitioning to Deep Learning.
* **Environment:** Operates in a single-agent simulated market based on historical demand patterns.

## 🛠 Tech Stack
* **Language:** Python 3.9+
* **ML Frameworks:** XGBoost (Regression), NumPy (RL logic)
* **Data Processing:** Pandas
* **Visualization:** Matplotlib
* **Environment:** Custom OpenAI-Gym style environment

## 📁 Project Structure
```text
├── data/
│   └── raw/               # Olist E-commerce dataset
├── models/
│   └── q_table.pkl        # Saved "brain" of the trained agent
├── src/
│   ├── data_loader.py     # Data ingestion and merging
│   ├── feature_eng.py     # Feature engineering & seasonality
│   ├── demand_model.py    # XGBoost training & forecasting
│   ├── environment.py     # Pricing simulation environment
│   └── agent.py           # RL Agent training and logic (Main Entry)
├── docs/
│   └── training_graph.png # Training progress visualization
├── .gitignore
└── README.md

## 📊 Result
The agent successfully learns to identify optimal price points. During training, we observe a steady increase in cumulative reward as the agent transitions from exploration to exploitation.

## ⚙️ How to Run
1. **Clone the repository:**
git clone [https://github.com/Arseniiiii-ai/dynamic-price-optimization-rl.git](https://github.com/Arseniiiii-ai/dynamic-price-optimization-rl.git)

2.**Install dependencies:**
pip install pandas numpy xgboost matplotlib

3.**Run the training and inference:**
python src/agent.py

## Roadmap
[ ] Phase 2: Implement Deep Q-Networks (DQN) using PyTorch for more complex state handling.

[ ] Phase 3: Add competitor price tracking and inventory constraints.

[ ] Phase 4: Build a Streamlit dashboard for business stakeholders.

Developed by Arsen — Focused on High-Performance AI Solutions.

## 📈 Future Roadmap

The project is currently in its **MVP (Phase 1)**. I am actively working on the following milestones to transition it from a laboratory experiment to a production-grade pricing engine:

### 🟢 Phase 1: Foundation (Completed)
- [x] Integration of historical e-commerce datasets.
- [x] Demand forecasting using **XGBoost**.
- [x] Basic **Q-Learning** agent implementation.
- [x] Unit economics-based reward function (Profit over Revenue).

### 🟡 Phase 2: Deep Learning Upgrade (In Progress)
- [ ] Transition from Q-Table to **Deep Q-Networks (DQN)** using **PyTorch**.
- [ ] Implementation of **Experience Replay** to stabilize training graphs.
- [ ] Hyperparameter tuning for the neural network architecture.

### 🔴 Phase 3: -

### 🔵 Phase 4: -
