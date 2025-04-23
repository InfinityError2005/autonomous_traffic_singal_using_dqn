import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.learning_rate = 0.001  # Learning rate

        # Neural network to approximate Q-values
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        # Simple neural network architecture
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def update_target(self):
        # Sync the target model with the current model
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Decide on an action (exploration or exploitation)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action (exploration)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()  # Choose action with highest Q-value (exploitation)

    def train(self):
        # Train the model using a batch of experiences from the memory
        if len(self.memory) < 32:
            return  # Not enough experiences to train

        batch = random.sample(self.memory, 32)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            target = torch.tensor(reward, dtype=torch.float32)  # Ensure target is a tensor

            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state))

            current_q_values = self.model(state)
            current_q_value = current_q_values[0][action]

            # Ensure current_q_value and target are both tensors
            loss = nn.MSELoss()(current_q_value, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Decay the epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay