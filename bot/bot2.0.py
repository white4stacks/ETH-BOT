import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import alpaca_trade_api as tradeapi
import gym
from gym import spaces
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import ta  # Technical Analysis library
from flask import Flask, request
import alpaca_trade_api as tradeapi

app = Flask(__name__)
ALPACA_API_KEY = "PK5BN0WDWQ9JGRZPLHP1"
ALPACA_SECRET_KEY = "Tr2M2zvTdPRMsL9foulZa5vp2AeUMgaKwuxJJ6qz"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    message = data['message']

    # Execute trading action based on message
    if message == 'BUY':
        # Execute buy order logic here
        pass
    elif message == 'SELL':
        # Execute sell order logic here
        pass

    return 'OK'

if __name__ == '__main__':
    app.run(debug=True)

# Define the Trading Environment
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.done = False
        self.total_profit = 0
        self.positions = []
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        self.done = False
        self.total_profit = 0
        self.positions = []
        return self._next_observation()
    
    def _next_observation(self):
        return self.df.iloc[self.current_step].values
    
    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
        
        reward = 0
        if action == 1:  # buy
            self.positions.append(self.df.iloc[self.current_step]['close'])
        elif action == 2 and self.positions:  # sell
            bought_price = self.positions.pop(0)
            reward = self.df.iloc[self.current_step]['close'] - bought_price
            self.total_profit += reward
        
        obs = self._next_observation()
        return obs, reward, self.done, {}
    
    def render(self, mode='human', close=False):
        profit = self.total_profit
        print(f'Step: {self.current_step}, Profit: {profit}')

# Define the DQN Model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(np.array(state, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(np.array(state, dtype=np.float32)).to(self.device)
            next_state = torch.FloatTensor(np.array(next_state, dtype=np.float32)).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)[0])
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

# Setup Alpaca API and Fetch Data
ALPACA_API_KEY = 'PK5BN0WDWQ9JGRZPLHP1'
ALPACA_SECRET_KEY = 'Tr2M2zvTdPRMsL9foulZa5vp2AeUMgaKwuxJJ6qz'
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

# Fetch historical data for Ethereum (replace 'ETH/USD' with the correct symbol if needed)
start_date = '2023-01-01'
end_date = '2024-01-01'
timeframe = '1Day'

bars = api.get_crypto_bars('ETH/USD'.upper(), timeframe, start=start_date, end=end_date).df

# Convert to DataFrame and preprocess
df = pd.DataFrame(bars)
df.index = pd.to_datetime(df.index)

# Add technical indicators
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['sma'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
df['ema'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
df['macd'] = ta.trend.MACD(df['close']).macd()
df = df.dropna()

# Normalize the data
df = (df - df.mean()) / df.std()

# Define the environment
env = TradingEnv(df)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the DQN agent
agent = DQNAgent(state_size, action_size)

batch_size = 32
save_interval = 100  # Save the model every 100 episodes
save_path = "dqn_model.pth"

try:
    episode = 0
    while True:
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_profit = 0
        
        for time_step in range(500):  # adjust this range based on your data
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_profit += reward
            if done:
                print(f"episode: {episode}, profit: {total_profit:.2f}, e: {agent.epsilon:.2f}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        if episode % save_interval == 0:
            agent.save(save_path)
            print(f"Model saved at episode {episode}")
            
        episode += 1

except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    agent.save(save_path)
    print(f"Model saved at episode {episode}")

# Save the model one last time after the loop ends
agent.save(save_path)
print(f"Final model saved at episode {episode}")
