import torch
import torch.nn as nn
import numpy as np
import random
from replay_buffer import ReplayBuffer


class DQNNetwork(nn.Module):
    """
    Neural network that maps battle state → Q-values for each action.
    Higher Q-value = agent thinks that action is better.
    """

    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """
    DQN Agent with experience replay and epsilon-greedy exploration.

    Corresponds to:
    - 'Cross-Game Experience' → ReplayBuffer
    - 'Action Selection'      → epsilon-greedy select_action()
    - 'Learning'              → train_step()
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
    ):
        self.action_dim    = action_dim
        self.gamma         = gamma        # discount factor (how much future rewards matter)
        self.epsilon       = epsilon      # exploration rate (starts high, decays over time)
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size

        # Device — use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Two networks: online (learns) and target (stable reference)
        self.online_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()
        self.memory    = ReplayBuffer(buffer_capacity)

    def select_action(self, state, available_actions):
        """
        Epsilon-greedy action selection:
        - With probability epsilon → pick a random action (explore)
        - Otherwise              → pick the best action (exploit)
        Only selects from available_actions (legal moves).
        """
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor).squeeze(0)

        # Among available actions, pick the one with highest Q-value
        available_q = {a: q_values[a].item() for a in available_actions}
        return max(available_q, key=available_q.get)

    def store(self, state, action, reward, next_state, done):
        """Store a transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Sample a batch from memory and update the network.
        Returns loss value for logging.
        """
        if len(self.memory) < self.batch_size:
            return None  # Not enough data yet

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values from online network
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values from target network (Bellman equation)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="dqn_model.pth"):
        """Save model weights to disk."""
        torch.save(self.online_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path="dqn_model.pth"):
        """Load model weights from disk."""
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_network()
        print(f"Model loaded from {path}")