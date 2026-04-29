# Dynamic Price Optimization Engine with Reinforcement Learning 🤖📈

This repository contains an end-to-end Machine Learning system designed to optimize e-commerce product pricing dynamically. The project combines **XGBoost** for demand forecasting and **Reinforcement Learning (Q-Learning)** for autonomous decision-making.

## 🚀 Overview
The goal of this project is to move away from static pricing and manual rule-based adjustments. Instead, an AI agent learns to maximize **Net Profit** by analyzing historical sales data, seasonality, and price elasticity.

### ✅ Key Strengths (Phase 2 Upgrade)
* **Deep Q-Learning (DQN):** Transitioned from tabular methods to neural network approximation using **PyTorch**.
* **Experience Replay:** Implemented a replay buffer to stabilize training and break temporal data correlation.
* **XGBoost Demand Forecaster:** High-precision base demand prediction with an average error of ~5.39 units.
* **Profit-Oriented Reward:** The agent optimizes for **Net Profit** (Revenue - COGS), ensuring sustainable business logic.
* **Advanced Visualization:** High-resolution training logs with Moving Average smoothing for trend analysis.

## ⚠️ Current Weaknesses & Limitations

While the DQN agent outperforms static pricing, the current system has technical constraints that are being addressed in upcoming phases:

* **Static Competitor Modeling:** The environment currently assumes a stable market. It does not yet account for "predatory pricing" or aggressive retaliatory moves from rival AI agents.
* **Cold Start Problem:** The agent requires historical demand data from the XGBoost forecaster. It currently struggles to price entirely new products with zero sales history.
* **Discrete Action Space:** The agent is limited to fixed percentage changes (e.g., -10%, 0, +10%). Moving to a continuous action space (PPO/DDPG) would allow for more granular optimization.
* **Delayed Reward Signal:** Profit is calculated per transaction, but long-term Customer Lifetime Value (CLV) is not yet incorporated into the reward function.

## 🛠 Tech Stack
* **Language:** Python 3.9+
* **ML Frameworks:** XGBoost (Regression), NumPy (RL logic)
* **Data Processing:** Pandas
* **Visualization:** Matplotlib
* **Environment:** Custom OpenAI-Gym style environment

## 📁 Project Structure
```text
├── data/               # Raw and processed e-commerce datasets
├── models/             # Compiled model weights (.pth) for DQN
├── src/                # Core source code
│   ├── data_loader.py  # Data cleaning and ingestion pipeline
│   ├── environment.py  # Custom RL market simulation environment
│   ├── agent.py        # DQN Architecture & Experience Replay logic
│   ├── demand_model.py # XGBoost demand forecasting engine
│   └── feature_eng.py  # Seasonality and price elasticity engineering
├── docs/               # Visual assets and training analytics
├── .gitignore          # Environment and cache exclusion
└── README.md           # Project documentation
```

## 📊 Result of each Phase~

##  Phase 1
The agent successfully learns to identify optimal price points. During training, we observe a steady increase in cumulative reward as the agent transitions from exploration to exploitation.

##  Phase 2
In Phase 2, the agent demonstrates a significant ability to adapt to market fluctuations. By utilizing Experience Replay, we achieved a more stable learning curve compared to standard Q-learning. Model weights are saved in models/dqn_model.pth for immediate inference.

## ⚙️ How to Run
1. **Clone the repository:**
git clone [https://github.com/Arseniiiii-ai/dynamic-price-optimization-rl.git](https://github.com/Arseniiiii-ai/dynamic-price-optimization-rl.git)

2. **Install dependencies:**
pip install pandas numpy xgboost matplotlib

3. **Run the training and inference:**
python src/agent.py

## Roadmap
[ ] Phase 2: Implement Deep Q-Networks (DQN) using PyTorch for more complex state handling.

[ ] Phase 3: Add competitor price tracking and inventory constraints.

[ ] Phase 4: Build a Streamlit dashboard for business stakeholders.

Developed by Arsen — Focused on High-Performance AI Solutions.

## 📈 Future Roadmap

The project is currently in its **Phase 2 (DQN Implementation)**. I am actively working on the following milestones to transition it from a laboratory experiment to a production-grade pricing engine:

### 🟢 Phase 1: Foundation (Completed)
- [x] Integration of historical e-commerce datasets.
- [x] Demand forecasting using **XGBoost**.
- [x] Basic **Q-Learning** agent implementation.
- [x] Unit economics-based reward function (Profit over Revenue).

### 🟢 Phase 2: Deep Learning (Completed)

- [x] Implementation of DQN architecture with PyTorch.
- [x] Integration of Experience Replay buffer.
- [x] Enhanced training visualization (300 DPI graphs).
- [x] Model persistence (.pth saving/loading).

### 🟡 Phase 3: Market Complexity (In Progress)

- [ ] Competitor Dynamics: Adding rival agents to the environment.
- [ ] Inventory Constraints: Stock-level aware pricing strategies.
- [ ] Multi-Agent Simulation: Competitive market scenarios.

### 🔵 Phase 4: -

Soon!


---
### 👤 About the Author

**Arsen** *Building High-Performance AI Solutions & Data science*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](www.linkedin.com/in/arsen-baktygaliyev-53474631a)
[![Researchpapers](https://img.shields.io/badge/Researchpapers-Explore-green?style=flat-square&logo=vercel)](https://t.me/build_ai_notes)
