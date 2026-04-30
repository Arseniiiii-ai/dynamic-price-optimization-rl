import os
import matplotlib.pyplot as plt
import numpy as np
from src.environment import CompetitiveMarketEnv
from src.agent import DQNAgent

env = CompetitiveMarketEnv()
agent = DQNAgent(state_size=5, action_size=3)
history = []

for e in range(50): # 50 Episodes for Phase 3
    state = env.reset()
    # Normalize: Prices / 100, Month / 12, Weekend / 1, Inventory / 500
    state = state / [100, 100, 12, 1, 500] 
    total_reward = 0
    
    for time in range(30):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Normalize next_state
        next_state = next_state / [100, 100, 12, 1, 500]
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if done:
            print(f"Episode: {e+1}/50, Total Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Final Inventory: {env.inventory:.1f}")
            break
            
    history.append(total_reward)
    agent.replay()

# Save the trained model
agent.save()

# Plot and save results
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        pass # Fallback to default style

plt.figure(figsize=(10, 5))
plt.plot(history, color='royalblue', label='Episode Reward')
if len(history) >= 5:
    moving_avg = np.convolve(history, np.ones(5)/5, mode='valid')
    plt.plot(range(4, len(history)), moving_avg, color='red', label='Moving Avg (5 ep)')
plt.title('Competitive Environment Training (with Inventory)', fontsize=14, fontweight='bold')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Profit', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

os.makedirs('docs', exist_ok=True)
plt.savefig('docs/training_graph.png', dpi=300, bbox_inches='tight')
print("Model saved to models/dqn_model.pth")
print("Training graph saved to docs/training_graph.png")