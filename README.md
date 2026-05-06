# Dynamic Price Optimization Engine (v2.0) 🤖📈

## Overview

A high-precision automated pricing engine powered by **Deep Reinforcement Learning (DRL)**. This system navigates complex market dynamics, competitor aggression, and inventory constraints to find the optimal price point in real-time.

---

## Architectural Evolution: From DQN to PPO

The core of this project evolved from a discrete "step-based" model to a continuous "precision-based" engine. This transition was critical to solving the **Discretization Trap**.

| Feature | Base Version (DQN) | Production Engine (PPO) |
|---|---|---|
| **Action Space** | Discrete (Fixed steps like ±5%) | **Continuous** (Any value, e.g., +1.34%) |
| **Algorithm** | Deep Q-Network | **Proximal Policy Optimization** |
| **Precision** | Coarse & "Steppy" | **Cent-perfect & Smooth** |
| **Stability** | Prone to sudden "policy shifts" | **Robust** (via clipping mechanism) |

---

## 🛠 Technical Stack

- **Framework:** PyTorch (Neural Networks)
- **Environment:** Gymnasium (Custom Market Simulation)
- **Algorithm:** PPO (Actor-Critic Architecture)
- **Dashboard:** Streamlit (Real-time Inference & Visualization)
- **Data Science:** NumPy, Pandas, Matplotlib

---

## 📁 Project Structure

```text
├── data/               # Historical market data and processed datasets
├── models/             # Production weights for PPO (.pth) and legacy DQN backups
├── docs/               # UI screenshots and PPO training performance analytics
├── src/                # Core simulation and AI logic
│   ├── environment.py  # High-fidelity Market Simulation (Gymnasium-based)
│   ├── agent.py        # PPO Actor-Critic Architecture & Continuous Action logic
│   └── utils.py        # Helper functions for normalization and logging
├── app.py              # Streamlit Dashboard (Interactive UI & Live Inference)
├── train.py            # Training pipeline for Robust PPO (Market Stress Tests)
├── requirements.txt    # Project dependencies (Torch, Gymnasium, Streamlit)
├── .gitignore          # Exclusion of virtual envs, caches, and raw models
└── README.md           # Project documentation and technical specs
```

---

## 🖥 Interactive Dashboard

The system includes a Streamlit dashboard for real-time market testing. Users can manually adjust competitor aggression, trigger market shocks, and watch the agent's strategy adapt instantly.

```bash
streamlit run app.py
```

---

## ⚙️ How to Run

**1. Clone the repository**

```bash
git clone https://github.com/Arseniiiii-ai/dynamic-price-optimization-rl.git
cd dynamic-price-optimization-rl
```

**2. Install dependencies**

> It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

**3. Launch the Intelligence Dashboard**

```bash
streamlit run app.py
```

**4. (Optional) Re-train the Agent**

```bash
python train.py
```

---

## 📊 Phase Results

### Phase 1
The agent successfully learns to identify optimal price points. During training, we observe a steady increase in cumulative reward as the agent transitions from exploration to exploitation.

### Phase 2
The agent demonstrates a significant ability to adapt to market fluctuations. By utilizing Experience Replay, we achieved a more stable learning curve compared to standard Q-learning. Model weights are saved in `models/dqn_model.pth` for immediate inference.

### Phase 3
The agent demonstrated a significant ability to adapt to Inventory Scarcity. During training, we observed the agent learning to "hold" price levels when rivals undercut too aggressively, prioritizing the preservation of limited stock for high-margin sales.

### Phase 4
Successfully transitioned the engine from a back-end script to an interactive Strategic Dashboard. The integration of Streamlit and Plotly allowed for real-time stress testing of the trained model against various market scenarios.

### Phase 5
Successfully transitioned the engine from a discrete action space to a continuous action space. The agent is now able to make cent-perfect pricing decisions and has been trained on a variety of market scenarios including Black Friday stress tests.

---

## 📈 Current Status

### 🟢 Phase 1: Foundation (Completed)
- [x] Integration of historical e-commerce datasets
- [x] Demand forecasting using **XGBoost**
- [x] Basic **Q-Learning** agent implementation
- [x] Unit economics-based reward function (Profit over Revenue)

### 🟢 Phase 2: Deep Learning (Completed)
- [x] Implementation of DQN architecture with PyTorch
- [x] Integration of Experience Replay buffer
- [x] Enhanced training visualization (300 DPI graphs)
- [x] Model persistence (`.pth` saving/loading)

### 🟢 Phase 3: Market Complexity (Completed)
- [x] Competitor Dynamics: Adding rival agents to the environment
- [x] Inventory Constraints: Stock-level aware pricing strategies
- [x] Multi-Agent Simulation: Competitive market scenarios

### 🟢 Phase 4: Intelligence Dashboard (Completed)
- [x] Interactive Controls: Live sidebar for inventory and competitor aggression
- [x] Real-Time Simulation: "Play" button to watch the AI compete against rivals visually
- [x] Strategy Explainability: "Policy Map" showing when the agent triggers price hikes
- [x] Performance Metrics: Real-time tracking of Profit, Revenue, and Stock via Streamlit

### 🟢 Phase 5: Production-Grade Optimization (Completed)
- [x] Continuous Action Space: DQN → PPO for cent-perfect pricing precision
- [x] Price Elasticity Layer: Real-time demand sensitivity modeling in the neural network
- [x] Market Latency Simulation: Training for real-world execution lag handling
- [x] Multi-Objective Rewards: Strategy toggle between Market Share (GMV) and Profit Margin
- [x] Advanced Stress Testing: Black Friday demand spikes and high-volatility scenarios

### 🟡 Phase 6: Multi-Agent Intelligence & Elastic Scaling (In Progress)

- [ ] Adversarial Self-Play: Implementing a training loop where the PPO agent competes against past versions of itself to discover advanced counter-pricing strategies.
- [ ] Multi-Agent Ecosystem: Expanding the environment to support 5+ simultaneous AI agents (e.g., Aggressive, Conservative, and Trend-Following) to simulate a crowded marketplace.
- [ ] Explainable AI (XAI) Integration: Adding a SHAP-based interpretability layer to the dashboard to visualize which features (inventory vs. competitor price) are driving the agent's decisions.
- [ ] Market Regime Switching: Training the agent to detect and adapt to different macro-economic "regimes" (e.g., Inflationary, Recessionary, or Holiday Peaks) in real-time.
- [ ] Dynamic Portfolio Optimization: Moving beyond single-product pricing to manage "Cannibalization" across multiple related SKUs to maximize total category revenue.

### 🔵 Phase 7: Portfolio Orchestration & Global Autonomy (Upcoming)

Soon!

---

## 👤 About the Author

**Arsen** — *Building High-Performance AI Solutions & Data Science*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/arsen-baktygaliyev-53474631a)
[![Research Notes](https://img.shields.io/badge/Research_Notes-Explore-green?style=flat-square&logo=telegram)](https://t.me/build_ai_notes)
