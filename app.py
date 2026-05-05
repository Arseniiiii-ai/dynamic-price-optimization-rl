import plotly.graph_objects as go
import plotly.express as px
import streamlit as st 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.environment import MarketEnv
from src.agent import PPOAgent
from src.data_loader import PricingDataLoader
from src.feature_engineering import create_features

# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI Pricing Dashboard", layout="wide")
st.title("🤖 Dynamic Price Optimization Dashboard")
st.markdown("### Phase 4: Real-Time Market Simulation (Continuous PPO)")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Market Parameters")
initial_inventory = st.sidebar.slider("Starting Inventory", 10, 500, 100)
competitor_aggression = st.sidebar.select_slider(
    "Competitor Aggression", 
    options=["Passive", "Balanced", "Aggressive"]
)

# --- LOAD CONTINUOUS PPO BRAIN ---
@st.cache_resource
def load_model():
    state_size = 5 
    action_size = 1 
    agent = PPOAgent(state_size, action_size)
    
    # Check if model exists, else just return untrained for now
    if os.path.exists("models/ppo_model.pth"):
        agent.model.load_state_dict(torch.load("models/ppo_model.pth"))
    return agent

agent = load_model()
if os.path.exists("models/ppo_model.pth"):
    st.success("✅ Continuous PPO Model Loaded Successfully!")
else:
    st.warning("⚠️ PPO Model not found. Running with untrained agent. Please run `python train.py` first.")

# --- SIMULATION LOGIC ---
results = []
if st.button("▶️ Run Market Simulation"):
    env = MarketEnv(inventory=initial_inventory) 
    
    state = env.reset()
    state = state / [100, 100, 12, 1, 500] 

    # Run a 30-day simulation
    for day in range(30):
        # 1. Get action from the loaded AI agent (Deterministic for inference)
        # Use a single variable first to see what the agent is actually giving back
        result = agent.get_action(state, deterministic=True)

        # If the result is a tuple (action, log_prob), take the first part
        if isinstance(result, tuple):
            action = result[0]
        else:
            action = result
        
        # 2. Step the environment
        next_state, reward, done, _ = env.step(action)
        
        # Normalize next_state
        next_state = next_state / [100, 100, 12, 1, 500]
        
        # 3. Save data for plotting
        results.append({
            "Day": day + 1,
            "Price": env.our_price,
            "Comp_Price": env.comp_price,
            "Inventory": env.inventory, 
            "Profit": reward
        })
        state = next_state
        if done:
            break

# --- DISPLAY DATA ---
df_sim = pd.DataFrame(results)

if not df_sim.empty:
    # 1. Расширенные метрики
    total_profit = df_sim['Profit'].sum()
    final_stock = df_sim['Inventory'].iloc[-1]
    stock_utilization = ((initial_inventory - final_stock) / initial_inventory) * 100
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Profit", f"${total_profit:,.2f}", delta=f"{df_sim['Profit'].iloc[-1]:.2f} last day")
    col_m2.metric("Final Inventory", f"{int(final_stock)} units")
    col_m3.metric("Stock Sold", f"{stock_utilization:.1f}%")
    col_m4.metric("Avg Margin", f"${(df_sim['Price'] - 50).mean():.2f}") # Допустим COGS=50

    # 2. Главный график: Price Battle + Зона прибыли
    st.subheader("⚔️ Competitive Strategy Analysis")
    fig_price = go.Figure()
    
    # Добавляем зону себестоимости (COGS)
    fig_price.add_hrect(y0=0, y1=50, fillcolor="red", opacity=0.1, annotation_text="Loss Zone (Below COGS)", annotation_position="bottom left")
    
    fig_price.add_trace(go.Scatter(x=df_sim['Day'], y=df_sim['Price'], name='AI Agent', line=dict(color='#1f77b4', width=4)))
    fig_price.add_trace(go.Scatter(x=df_sim['Day'], y=df_sim['Comp_Price'], name='Competitor', line=dict(color='#ff7f0e', dash='dot')))
    
    fig_price.update_layout(
        hovermode="x unified", 
        template="plotly_white", 
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # 3. Второй ряд графиков
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Cumulative Profit Growth")
        df_sim['Cum_Profit'] = df_sim['Profit'].cumsum()
        fig_cum = px.line(df_sim, x='Day', y='Cum_Profit', markers=True, line_shape="spline")
        fig_cum.update_traces(line_color='#2ca02c')
        st.plotly_chart(fig_cum, use_container_width=True)
        
    with col2:
        st.subheader("📦 Inventory & Demand")
        fig_stock = go.Figure()
        fig_stock.add_trace(go.Scatter(x=df_sim['Day'], y=df_sim['Inventory'], fill='tozeroy', name='Stock', line_color='#9467bd'))
        fig_stock.update_layout(template="plotly_white")
        st.plotly_chart(fig_stock, use_container_width=True)

    # 4. Аналитический вывод (AI Insight)
    st.info("💡 **Strategy Insight:** " + 
            ("The agent is aggressively undercutting to clear stock." if stock_utilization > 80 else 
             "The agent is maintaining high margins with conservative pricing."))