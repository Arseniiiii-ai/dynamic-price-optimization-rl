import numpy as np
import gym
from gym import spaces

class MarketEnv(gym.Env):
    def __init__(self, cogs=50.0, inventory=500.0):
        super(MarketEnv, self).__init__()
        self.cogs = cogs
        self.initial_inventory = inventory 
        
        # Continuous action space between -1.0 and 1.0
        # -1.0 = Max Price Decrease (-15%)
        #  1.0 = Max Price Increase (+15%)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # State space remains 5D: [Price, Comp_Price, Month, Weekend, Inventory]
        # To match the DQN state space from earlier
        self.observation_space = spaces.Box(low=0, high=1000, shape=(5,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.our_price = 100.0
        self.comp_price = 102.0
        self.step_count = 0
        self.inventory = self.initial_inventory 
        return self._get_state()
        
    def _get_state(self):
        return np.array([self.our_price, self.comp_price, 5, 0, self.inventory], dtype=np.float32)

    def step(self, action):
        # Ensure action is unpacked correctly (can be array or float)
        if isinstance(action, np.ndarray):
            action_val = action[0]
        else:
            action_val = action
            
        # 1. Map continuous action to a precise percentage change
        # Action is [-1, 1], so multiplying by 0.15 gives us a ±15% range
        price_change_pct = action_val * 0.15
        
        # 2. Update price with cent-level precision
        old_price = self.our_price
        self.our_price = np.round(self.our_price * (1 + price_change_pct), 2)
        
        # 3. Apply a "Smoothing Penalty" to prevent erratic price flickering
        jitter_penalty = -0.01 * abs(self.our_price - old_price)

        # 4. Competitor Logic (Heuristic: Undercuts us by 2%)
        self.comp_price = max(self.our_price * 0.98, self.cogs * 1.05)

        # 5. Calculate Demand based on Relative Price
        price_ratio = self.comp_price / self.our_price
        base_demand = 50 
        demand = base_demand * (price_ratio ** 1.2) 

        # Limit demand to available inventory
        actual_sales = min(demand, self.inventory)
        self.inventory -= actual_sales

        # 6. Profit Calculation
        current_step_profit = (self.our_price - self.cogs) * actual_sales
        
        # Penalty for selling below cost
        reward = current_step_profit if self.our_price > self.cogs else -200
        
        # Penalty for running out of stock
        if self.inventory <= 0:
            reward -= 100
            
        # Final Reward calculation incorporating jitter penalty
        reward += jitter_penalty
            
        self.step_count += 1
        done = self.step_count >= 30 or self.inventory <= 0
        
        return self._get_state(), float(reward), done, {}