import numpy as np

class CompetitiveMarketEnv:
    def __init__(self, cogs=50.0):
        self.cogs = cogs
        self.reset()

    def reset(self):
        self.our_price = 100.0
        self.comp_price = 102.0 # Competitor starts near us
        self.step_count = 0
        self.inventory = 500.0 # Starting inventory
        # Return state with 5 features
        return np.array([self.our_price, self.comp_price, 5, 0, self.inventory], dtype=np.float32)

    def step(self, action_idx):
        # 1. Map Action (0: -5%, 1: 0%, 2: +5%)
        multipliers = [0.95, 1.0, 1.05]
        self.our_price *= multipliers[action_idx]

        # 2. Competitor Logic (Heuristic: Undercluts us by 2%)
        self.comp_price = max(self.our_price * 0.98, self.cogs * 1.05)

        # 3. Calculate Demand based on Relative Price
        # If our_price < comp_price, demand increases significantly
        price_ratio = self.comp_price / self.our_price
        base_demand = 50 # Simplified for Phase 3
        demand = base_demand * (price_ratio ** 1.2) 

        # Limit demand to available inventory
        actual_sales = min(demand, self.inventory)
        self.inventory -= actual_sales

        # 4. Profit Calculation
        profit = (self.our_price - self.cogs) * actual_sales
        
        # Penalty for selling below cost
        reward = profit if self.our_price > self.cogs else -200
        
        # Penalty for running out of stock
        if self.inventory <= 0:
            reward -= 100
            
        self.step_count += 1
        done = self.step_count >= 30 or self.inventory <= 0
        
        next_state = np.array([self.our_price, self.comp_price, 5, 0, self.inventory], dtype=np.float32)
        return next_state, reward, done, {}