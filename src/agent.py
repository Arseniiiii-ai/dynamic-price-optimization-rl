import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor can be used, but separate is often more stable
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Tanh() # Constrain action mean to [-1, 1]
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Action standard deviation as a learnable parameter
        self.log_std = nn.Parameter(torch.zeros(1, action_size))
        
    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        return mu, std, value

class PPOAgent:
    def __init__(self, state_size=5, action_size=1):
        self.state_size = state_size
        self.action_size = action_size
        
        # PPO Hyperparameters
        self.gamma = 0.99            # Discount factor
        self.eps_clip = 0.2          # PPO clip parameter
        self.k_epochs = 4            # Number of epochs to update policy
        self.lr = 0.0003             # Learning rate
        
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def get_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu, std, _ = self.model(state_t)
            
            if deterministic:
                return mu.numpy()[0]
                
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            
            # Clip action to valid range [-1, 1]
            action = torch.clamp(action, -1.0, 1.0)
            
            log_prob = dist.log_prob(action)
            
        return action.numpy()[0], log_prob.numpy()[0]
        
    def update(self, states, actions, log_probs, rewards, next_states, dones):
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions)).unsqueeze(-1)
        old_log_probs_t = torch.FloatTensor(np.array(log_probs)).unsqueeze(-1)
        
        # Compute discounted returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns_t = torch.FloatTensor(returns).unsqueeze(1)
        
        # Normalize returns
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-7)
        
        for _ in range(self.k_epochs):
            # Evaluate current policy
            mu, std, state_values = self.model(states_t)
            dist = torch.distributions.Normal(mu, std)
            curr_log_probs = dist.log_prob(actions_t)
            dist_entropy = dist.entropy()
            
            # Advantage estimation
            advantages = returns_t - state_values.detach()
            
            # Importance ratio
            ratios = torch.exp(curr_log_probs - old_log_probs_t)
            
            # Surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Loss calculation
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, returns_t)
            entropy_bonus = dist_entropy.mean() * 0.01
            
            loss = actor_loss + 0.5 * critic_loss - entropy_bonus
            
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, path="models/ppo_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logging.info(f"PPO Model saved to {path}")
