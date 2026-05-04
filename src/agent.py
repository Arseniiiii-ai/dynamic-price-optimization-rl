import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
import os
import matplotlib.pyplot as plt
from collections import deque

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- NEURAL NETWORK ARCHITECTURE ---
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

        

# --- DEEP Q-LEARNING AGENT ---
class DQNAgent:
    def __init__(self, state_size=5, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters - MAKE SURE THESE ALL HAVE 'self.'
        self.gamma = 0.95            
        self.epsilon = 1.0           
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001  
        self.batch_size = 32
        
        # Networks - Use 'self.model' to be consistent
        self.model = DQNetwork(state_size, action_size)
        
        # This line will work now because self.learning_rate exists!
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state_t)
        return torch.argmax(action_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
            
            target = reward
            if not done:
                # Bellman equation with Neural Network
                target = (reward + self.gamma * torch.max(self.model(next_state_t)).item())
            
            target_f = self.model(state_t)
            # We want the network to predict 'target' for the specific action taken
            with torch.no_grad():
                actual_target = target_f.clone()
                actual_target[0][action] = target
            
            # Gradient Descent
            self.optimizer.zero_grad()
            output = self.model(state_t)
            loss = self.criterion(output, actual_target)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path="models/dqn_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model weights saved to {path}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    from data_loader import PricingDataLoader
    from feature_engineering import create_features
    from demand_model import DemandForecaster
    from environment import PricingEnv

    # 1. Setup Environment
    loader = PricingDataLoader()
    df = create_features(loader.load_data())
    forecaster = DemandForecaster()
    train_data = forecaster.prepare_training_data(df)
    forecaster.train(train_data)

    env = PricingEnv(train_data.head(500), forecaster.model) # Using subset for speed
    agent = DQNAgent()

    logging.info("Starting Deep Q-Learning training...")
    history = []

    for episode in range(25):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Train the network on a batch from memory
            agent.replay()
        
        history.append(total_reward)
        logging.info(f"Episode {episode+1}/25 - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

    # Save the trained brain
    agent.save()

    # Plot results
    # --- VISUALIZATION ---
    plt.style.use('seaborn-v0_8-darkgrid') # Modern professional style
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.plot(history, alpha=0.3, color='royalblue', label='Raw Episode Reward')
    
    # Plot Moving Average (Window = 5)
    if len(history) >= 5:
        moving_avg = np.convolve(history, np.ones(5)/5, mode='valid')
        plt.plot(range(4, len(history)), moving_avg, color='red', linewidth=2, label='Moving Average (5 ep)')

    plt.title('Deep Q-Network: Profit Optimization Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Profit (Currency)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Auto-save to docs
    os.makedirs('docs', exist_ok=True)
    plt.savefig('docs/training_graph.png', dpi=300, bbox_inches='tight')
    logging.info("Enhanced graph saved to docs/training_graph.png")
    
    plt.show()
