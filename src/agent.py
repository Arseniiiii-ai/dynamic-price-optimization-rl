import numpy as np
import random
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')

class QLearningAgent:
    def __init__(self, action_size=3):
        self.action_size = action_size
        self.q_table = {} # Здесь хранится "опыт" ИИ
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0 # Шанс того, что агент решит "рискнуть" и попробовать новое
        self.epsilon_decay = 0.995

    def get_action(self, state):
        state_str = str(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(self.action_size)
        
        # Исследование vs Эксплуатация
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state_str])

    def learn(self, state, action, reward, next_state):
        state_str = str(state)
        next_state_str = str(next_state)
        
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = np.zeros(self.action_size)
            
        # Формула Q-обучения (Обновляем опыт на основе награды)
        old_value = self.q_table[state_str][action]
        next_max = np.max(self.q_table[next_state_str])
        
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state_str][action] = new_value

if __name__ == "__main__":
    from data_loader import PricingDataLoader
    from feature_engineering import create_features
    from demand_model import DemandForecaster
    from environment import PricingEnv

    # 1. Готовим данные и модель спроса
    loader = PricingDataLoader()
    df = create_features(loader.load_data())
    forecaster = DemandForecaster()
    train_data = forecaster.prepare_training_data(df)
    forecaster.train(train_data)

    # 2. Инициализируем Мир и Агента
    # Для теста возьмем первые 1000 строк данных
    env = PricingEnv(train_data.head(1000), forecaster.model)
    agent = QLearningAgent()

    logging.info("Начинаем обучение ИИ...")
    
    for episode in range(10): # Прогоним симуляцию 10 раз
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        agent.epsilon *= agent.epsilon_decay # С каждым разом агент "рискует" всё меньше
        logging.info(f"Эпизод {episode+1}: Общая выручка = {total_reward:.2f}")

history = [] # Список для хранения результатов

for episode in range(20): # Увеличим до 20 для наглядности
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        
    history.append(total_reward)
    agent.epsilon *= agent.epsilon_decay
    print(f"Эпизод {episode+1}: Выручка = {total_reward:.2f}")

# СТРОИМ ГРАФИК
plt.figure(figsize=(10, 5))
plt.plot(history, marker='o', linestyle='-', color='b')
plt.title('Прогресс обучения ИИ-агента')
plt.xlabel('Эпизод')
plt.ylabel('Общая выручка')
plt.grid(True)
plt.show()

def suggest_action(agent, price, is_weekend, month):
    state = np.array([price, is_weekend, month], dtype=float)
    # Мы временно отключаем случайные действия (epsilon = 0), чтобы получить лучший совет
    old_epsilon = agent.epsilon
    agent.epsilon = 0 
    action = agent.get_action(state)
    agent.epsilon = old_epsilon
    
    mapping = {0: "Снизить цену на 10% 📉", 1: "Оставить без изменений ⚖️", 2: "Повысить цену на 10% 📈"}
    return mapping[action]

# ПРОВЕРКА:
print("\n--- ТЕСТ РЕКОМЕНДАЦИИ ---")
test_price = 150.0
test_weekend = 1 # Суббота
test_month = 5   # Май
advice = suggest_action(agent, test_price, test_weekend, test_month)
print(f"Для цены {test_price} в выходной дня мая ИИ советует: {advice}")