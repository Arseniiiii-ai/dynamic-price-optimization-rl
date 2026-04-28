import numpy as np
import pandas as pd

class PricingEnv:
    def __init__(self, data, demand_model):
        self.data = data
        self.demand_model = demand_model
        self.current_step = 0
        self.categories = data['product_category_name'].unique()
        
    def reset(self):
        """Сброс окружения в начало (начало новой игровой сессии)."""
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        """Возвращает текущую ситуацию на рынке (что видит ИИ)."""
        row = self.data.iloc[self.current_step]
        # Состояние: [Средняя цена, выходной ли сейчас, месяц]
        return np.array([row['price'], row['is_weekend'], row['month']], dtype=float)

    def step(self, action):
        # Настройка цены
        price_adjustment = {0: 0.9, 1: 1.0, 2: 1.1}
        current_row = self.data.iloc[self.current_step]
        new_price = current_row['price'] * price_adjustment[action]
        
        # Предсказание продаж
        input_data = pd.DataFrame([[new_price, current_row['is_weekend'], current_row['month']]], 
                                 columns=['price', 'is_weekend', 'month'])
        predicted_sales = self.demand_model.predict(input_data)[0]
        
        # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
        # Предположим, себестоимость (закупка + логистика) — это 70% от оригинальной цены
        cost_price = current_row['price'] * 0.7 
        
        # Награда теперь — это чистая ПРИБЫЛЬ
        reward = (new_price - cost_price) * predicted_sales
        # -----------------------

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._get_state(), reward, done