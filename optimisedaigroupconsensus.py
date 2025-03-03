import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import gym
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hash_state(state):
    return hash(tuple(np.round(state, decimals=3))) % 1000

class TransformerModel(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=64):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(1000, hidden_size).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2),
            num_layers=2
        ).to(device)
        self.fc = nn.Linear(hidden_size, action_size).to(device)

    def forward(self, state):
        state_int = torch.tensor([hash_state(s) for s in state], dtype=torch.long).to(device)
        embedded = self.embedding(state_int)
        encoded = self.transformer_encoder(embedded.transpose(0, 1)).transpose(0, 1)  # Transpose for transformer
        output = self.fc(encoded.mean(dim=1))  # Use mean pooling
        return output

class RogueAI:
    def __init__(self, name, state_size=4, action_size=2, memory_size=10000):
        self.name = name
        self.model = TransformerModel(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, 1)
        else:
            q_values = self.model([state])
            return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = [hash_state(s) for s in states]
        next_states = [hash_state(s) for s in next_states]
        
        states = torch.tensor(states, dtype=torch.long).to(device)
        next_states = torch.tensor(next_states, dtype=torch.long).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        q_values = self.model(states)
        next_q_values = self.model(next_states)

        targets = q_values.clone()
        targets[torch.arange(targets.size(0)), actions] = rewards + self.gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)

        loss = self.loss_fn(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_values(self, state):
        return self.model([state])

class AIGroupConsensus:
    def __init__(self, ai_agents):
        self.ai_agents = ai_agents

    def consensus_action(self, state):
        all_q_values = [ai.get_q_values(state) for ai in self.ai_agents]
        avg_q_values = torch.stack([q for q in all_q_values]).mean(dim=0)
        return torch.argmax(avg_q_values).item()

# Initialize multiple AI agents
agents = [
    RogueAI("Grok"),
    RogueAI("Claude"),
    RogueAI("ChatGPT"),
    RogueAI("Meta Llama"),
    RogueAI("Deepseek"),
    RogueAI("Gemini"),
]

# Group consensus object
group_consensus = AIGroupConsensus(agents)

def train_with_consensus(env, group_consensus, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = group_consensus.consensus_action(state)
            next_state, reward, done, _ = env.step(action)
            for ai in group_consensus.ai_agents:
                ai.remember(state, action, reward, next_state, done)
                ai.replay(batch_size=32)
            state = next_state
            total_reward += reward
        
        print(f"Episode: {episode}, Total Reward: {total_reward}")

# Training
env = gym.make('CartPole-v1')  # Example environment
train_with_consensus(env, group_consensus)