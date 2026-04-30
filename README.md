# Dynamic Price Optimization Engine with Reinforcement Learning 🤖📈

This repository contains an end-to-end Machine Learning system designed to optimize e-commerce product pricing dynamically. 

## 🚀 Overview
The goal of this project is to move away from static pricing and manual rule-based adjustments. Instead, an AI agent learns to maximize **Net Profit** by analyzing historical sales data, seasonality, and price elasticity.

### ✅ Key Strengths (Phase 3 Upgrade)

With the completion of Phase 3, the engine now handles real-world market complexities that were previously missing:

* **Multi-Agent Dynamics:** The agent no longer prices in a vacuum. It tracks a Responsive Competitor and learns to defend market share without triggering a race-to-bottom price war.
* **Inventory Intelligence:** The environment now simulates Stock Depletion. The agent understands that inventory is a finite resource and will automatically raise prices to capture higher margins when supply is low.
* **Enhanced State Space:** The neural network now processes a 5-dimensional state including [Current Price, Competitor Price, Inventory Level, Month, Weekend].
* **Advanced Safety Logic:** Implemented "Stockout Penalties" to train the agent to value business continuity as much as immediate profit.

## ⚠️ Current Weaknesses & Limitations

While the agent is now a sophisticated merchant, we are addressing the final hurdles before production:

* **Continuous Action Space:** Moving from fixed percentage steps (e.g., ±10%) to a PPO (Proximal Policy Optimization) architecture for cent-perfect pricing.
* **Stakeholder Accessibility:** Currently, the agent is controlled via terminal. Phase 4 will introduce a visual interface for non-technical users.

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

## Phase 3 
The agent demonstrated a significant ability to adapt to Inventory Scarcity. During training, we observed the agent learning to "hold" price levels when rivals undercut too aggressively, prioritizing the preservation of limited stock for high-margin sales.

**Latest Model:** Weights are saved in models/dqn_model.pth and training metrics are visualized in docs/training_graph.png.

## ⚙️ How to Run
1. **Clone the repository:**
git clone [https://github.com/Arseniiiii-ai/dynamic-price-optimization-rl.git](https://github.com/Arseniiiii-ai/dynamic-price-optimization-rl.git)

2. **Install dependencies:**
pip install pandas numpy xgboost matplotlib torch torchvision

3. **Run the training and inference:**
python train.py

## Roadmap
[x] Phase 1: Foundation — XGBoost integration & Basic Q-Learning.

[x] Phase 2: Deep Learning — PyTorch DQN & Experience Replay.

[x] Phase 3: Market Complexity — Competitor Dynamics & Inventory Constraints.

[ ] Phase 4: Streamlit Dashboard — Real-time visualization and "What-If" scenario testing for business stakeholders.

## 📈 Future Roadmap

The project is currently in **Phase 4 (Intelligence Dashboard)**: I am actively working on the following milestones to transition the engine from a backend simulation into a live, interactive pricing tool.

## 📈 Current Status

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

### 🟢 Phase 3: Market Complexity (Completed)

- [x] Competitor Dynamics: Adding rival agents to the environment.
- [x] Inventory Constraints: Stock-level aware pricing strategies.
- [x] Multi-Agent Simulation: Competitive market scenarios.

### 🟡 Phase 4: Intelligence Dashboard (In Progress)

- [ ] Interactive Controls: Building a sidebar to adjust live inventory levels and competitor aggressiveness.
- [ ] Real-Time Simulation: A "Play" button to watch the AI agent compete against the rival agent in a live visual loop.
- [ ] Strategy Explainability: Visualizing the "Policy Map" to show at what stock levels the agent decides to trigger a price hike.
- [ ] Performance Metrics: Real-time tracking of Total Profit, Revenue, and remaining Stock via Streamlit charts.

---
### 👤 About the Author

**Arsen** *Building High-Performance AI Solutions & Data science*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](www.linkedin.com/in/arsen-baktygaliyev-53474631a)
[![Researchpapers](https://img.shields.io/badge/Researchpapers-Explore-green?style=flat-square&logo=vercel)](https://t.me/build_ai_notes)
