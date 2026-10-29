import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer — stores past game transitions
    so the DQN agent can learn from previous experiences.

    Corresponds to 'Cross-Game Experience' in the project synopsis.
    """

    def __init__(self, capacity=10000):
        # deque automatically discards oldest memories when full
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a single transition (one move in a battle)."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions for training."""
        transitions = random.sample(self.buffer, batch_size)

        # Unpack and convert to numpy arrays
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)