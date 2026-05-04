import plotly.graph_objects as go
import plotly.express as px
import streamlit as st 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.environment import CompetitiveMarketEnv
from src.agent import DQNAgent
from src.data_loader import PricingDataLoader
from src.feature_engineering import create_features

# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI Pricing Dashboard", layout="wide")
st.title("🤖 Dynamic Price Optimization Dashboard")
st.markdown("### Phase 4: Real-Time Market Simulation")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Market Parameters")
initial_inventory = st.sidebar.slider("Starting Inventory", 10, 500, 100)
competitor_aggression = st.sidebar.select_slider(
    "Competitor Aggression", 
    options=["Passive", "Balanced", "Aggressive"]
)

# --- LOAD PHASE 3 BRAIN ---
@st.cache_resource
def load_model():
    # Make sure action_size matches your training (looks like it's 3 in your agent.py)
    state_size = 5 
    action_size = 3 
    agent = DQNAgent(state_size, action_size)
    
    # Use .model here
    agent.model.load_state_dict(torch.load("models/dqn_model.pth"))
    return agent

agent = load_model()
st.success("✅ Phase 3 Model Loaded Successfully!")

# --- SIMULATION LOGIC ---
results = []
if st.button("▶️ Run Market Simulation"):
    # This is the line you asked about, placed correctly within the button trigger
    env = CompetitiveMarketEnv(inventory=initial_inventory) 
    
    state = env.reset()
    

    # Run a 30-day simulation
    for day in range(30):
        # 1. Get action from the loaded AI agent
        action = agent.get_action(state)
        
        # 2. Step the environment
        next_state, reward, done, _ = env.step(action)
        
        # 3. Save data for plotting (Mapping state indices to names)
        results.append({
            "Day": day + 1,
            "Price": state[0],
            "Comp_Price": state[1],
            "Inventory": state[4], # Check if inventory is index 3 or 4 in your env
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