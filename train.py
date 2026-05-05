import os
import matplotlib.pyplot as plt
import numpy as np
from src.environment import MarketEnv
from src.agent import PPOAgent

env = MarketEnv()
agent = PPOAgent(state_size=5, action_size=1)
history = []

print("Starting PPO Continuous Training...")

for e in range(50): # 50 Episodes
    state = env.reset()
    # Normalize state: Prices / 100, Month / 12, Weekend / 1, Inventory / 500
    state = state / [100, 100, 12, 1, 500] 
    
    states, actions, log_probs, rewards, next_states, dones = [], [], [], [], [], []
    total_reward = 0
    
    for time in range(30):
        action, log_prob = agent.get_action(state, deterministic=False)
        next_state, reward, done, _ = env.step(action)
        
        # Normalize next_state
        next_state = next_state / [100, 100, 12, 1, 500]
        
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
            
    # Perform PPO update at the end of the episode
    agent.update(states, actions, log_probs, rewards, next_states, dones)
    
    history.append(total_reward)
    print(f"Episode: {e+1}/50, Total Score: {total_reward:.2f}, Final Inventory: {env.inventory:.1f}")

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
plt.title('Continuous PPO Environment Training', fontsize=14, fontweight='bold')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Profit', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

os.makedirs('docs', exist_ok=True)
plt.savefig('docs/ppo_training_graph.png', dpi=300, bbox_inches='tight')
print("Model saved to models/ppo_model.pth")
print("Training graph saved to docs/ppo_training_graph.png")