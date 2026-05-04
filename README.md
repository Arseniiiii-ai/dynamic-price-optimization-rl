# Dynamic Price Optimization Engine with Reinforcement Learning 🤖📈

This repository contains an end-to-end Machine Learning system designed to optimize e-commerce product pricing dynamically. 

## Overview
The goal of this project is to move away from static pricing and manual rule-based adjustments. Instead, an AI agent learns to maximize **Net Profit** by analyzing historical sales data, seasonality, and price elasticity.

### ✅ Key Strengths (Phase 4 Upgrade)

With the completion of Phase 4, the system has gained a professional "face" and deep analytical transparency, evolving into a production-ready prototype:

* **Real-Time Interactive Dashboard**: Implemented a full-scale Streamlit interface, allowing users to manage simulations via UI sliders. Market parameters like initial inventory and competitor aggression can now be adjusted "on the fly" without touching the source code.
* **High-Fidelity Visual Analytics**: Powered by Plotly, data is no longer just static numbers. Interactive charts allow stakeholders to track "Price Wars," Cumulative Profit, and Stock Depletion dynamics in real-time.
* **AI Strategy Insights**: Added an automated interpretation layer that analyzes the neural network's behavior. The dashboard provides business insights, identifying whether the model is acting aggressively to capture market share or conservatively to protect margins.
* **Financial Guardrails Transparency**: Visualized "Loss Zones" (Below COGS). It is now clearly visible when the agent chooses to sacrifice short-term profit for long-term dominance or inventory protection.

## ⚠️ Current Weaknesses & Limitations

While the system now looks like a finished product, we are continuing to refine the "engine" to meet industrial standards:

* **Discrete Action Bottleneck**: The agent is still limited to choosing from fixed percentage steps (e.g., ±5%). For cent-perfect pricing precision, we must transition to a continuous control architecture.
* **Execution Latency**: The current model assumes instantaneous price updates. In the real world, there is a delay between the algorithm's command and the price changing on the storefront; this will be addressed in the next iteration.

---

## 🛠 Tech Stack
* **Language:** Python 3.9+
* **Deep Learning:** PyTorch (DQN Architecture, Experience Replay)
* **Machine Learning:** XGBoost (Demand Forecasting)
* **Interactive UI:** Streamlit (Phase 4 Dashboard)
* **Data Visualization:** Plotly (Dynamic Analytics), Matplotlib
* **Data Processing:** Pandas, NumPy
* **Environment:** Custom OpenAI Gym-style market simulator

## 📁 Project Structure
```text
├── data/                   # Raw and processed e-commerce datasets
├── models/                 # Compiled model weights (.pth) for DQN
├── src/                    # Core source code
│   ├── data_loader.py      # Data cleaning and ingestion pipeline
│   ├── environment.py      # Custom RL market simulation environment
│   ├── agent.py            # DQN Architecture & Experience Replay logic
│   ├── demand_model.py     # XGBoost demand forecasting engine
│   ├── feature_eng.py      # Seasonality and price elasticity engineering
│   └── model.py            # Neural Network definitions
├── docs/                   # UI screenshots and training analytics
├── app.py                  # Streamlit Dashboard (Main Entry Point)
├── requirements.txt        # Project dependencies
├── .gitignore              # Environment and cache exclusion
└── README.md               # Project documentation
```

---

## 📊 Result of each Phase~

##  Phase 1
The agent successfully learns to identify optimal price points. During training, we observe a steady increase in cumulative reward as the agent transitions from exploration to exploitation.

##  Phase 2
In Phase 2, the agent demonstrates a significant ability to adapt to market fluctuations. By utilizing Experience Replay, we achieved a more stable learning curve compared to standard Q-learning. Model weights are saved in models/dqn_model.pth for immediate inference.

## Phase 3 
The agent demonstrated a significant ability to adapt to Inventory Scarcity. During training, we observed the agent learning to "hold" price levels when rivals undercut too aggressively, prioritizing the preservation of limited stock for high-margin sales.

## Phase 4
Phase 4 successfully transitioned the engine from a back-end script to an interactive Strategic Dashboard. The integration of Streamlit and Plotly allowed for real-time stress testing of the trained model against various market scenarios.

---

## ⚙️ How to Run
1. **Clone the repository**
In Bash:
git clone https://github.com/Arseniiiii-ai/dynamic-price-optimization-rl.git
cd dynamic-price-optimization-rl

2. **Install dependencies**
It is recommended to use a virtual environment. Install all required libraries (PyTorch, Streamlit, Plotly, etc.) using:
In Bash:
pip install -r requirements.txt

3. **Launch the Intelligence Dashboard**
To run the Phase 4 interactive market simulation, use the following command:
In Bash:
streamlit run app.py

4. **(Optional) Re-train the Agent**
If you wish to re-train the DQN model from scratch using the historical data pipeline:
In Bash:
python src/train.py

## Roadmap
[x] Phase 1: Foundation — XGBoost integration & Basic Q-Learning.

[x] Phase 2: Deep Learning — PyTorch DQN & Experience Replay.

[x] Phase 3: Market Complexity — Competitor Dynamics & Inventory Constraints.

[x] Phase 4: Streamlit Dashboard — Real-time visualization and "What-If" scenario testing for business stakeholders.

[ ] Phase 5: Continuous Control & Real-World Latency

---

## Future Roadmap

The project is currently transitioning into Phase 5 (Continuous Control & High-Frequency Pricing). Having successfully deployed an interactive dashboard, I am now focused on elevating the engine to industrial standards of precision and scalability.

---

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

### 🟢 Phase 4: Intelligence Dashboard (Completed)

- [x] Interactive Controls: Building a sidebar to adjust live inventory levels and competitor aggressiveness.
- [x] Real-Time Simulation: A "Play" button to watch the AI agent compete against the rival agent in a live visual loop.
- [x] Strategy Explainability: Visualizing the "Policy Map" to show at what stock levels the agent decides to trigger a price hike.
- [x] Performance Metrics: Real-time tracking of Total Profit, Revenue, and remaining Stock via Streamlit charts.

### 🟡 Phase 5: Production-Grade Optimization (In Progress)

- [ ] Continuous Action Space: Transitioning from DQN to PPO (Proximal Policy Optimization) to enable cent-perfect pricing precision instead of fixed percentage steps.
- [ ] Price Elasticity Layer: Integrating an elasticity-modeling block within the neural network to predict demand sensitivity in real-time.
- [ ] Market Latency Simulation: Training the agent to handle "Execution Lag"—the real-world delay between an AI price decision and its update on the storefront.
- [ ] Multi-Objective Rewards: Implementing a "Strategy Toggle" to allow users to switch the agent's priority between Market Share (GMV) and Profit Margin.
- [ ] Advanced Stress Testing: Simulating extreme market events (e.g., Black Friday demand spikes) to verify agent stability during high-volatility periods.

### 🔵 Phase 6: Multi-Agent Intelligence & Elastic Scaling (Upcoming)

# Soon!

---
### 👤 About the Author

**Arsen** *Building High-Performance AI Solutions & Data science*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](www.linkedin.com/in/arsen-baktygaliyev-53474631a)
[![Researchpapers](https://img.shields.io/badge/Researchpapers-Explore-green?style=flat-square&logo=vercel)](https://t.me/build_ai_notes)
