import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity):
        self.capacity = capacity
        self.index = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.costs = np.zeros((capacity, 1), dtype=np.float32)
        self.not_dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, cost, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.next_states[self.index] = next_state
        self.rewards[self.index] = reward
        self.costs[self.index] = cost
        self.not_dones[self.index] = 1.0 - float(done)

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.next_states[indices],
            self.rewards[indices],
            self.costs[indices],
            self.not_dones[indices],
        )

    def state_dict(self):
        return {
            "size": self.size,
            "index": self.index,
            "states": self.states[:self.size],
            "actions": self.actions[:self.size],
            "next_states": self.next_states[:self.size],
            "rewards": self.rewards[:self.size],
            "costs": self.costs[:self.size],
            "not_dones": self.not_dones[:self.size],
        }

    def load_state_dict(self, saved):
        self.size = int(saved["size"])
        self.index = int(saved["index"])
        self.states[:self.size] = saved["states"]
        self.actions[:self.size] = saved["actions"]
        self.next_states[:self.size] = saved["next_states"]
        self.rewards[:self.size] = saved["rewards"]
        self.costs[:self.size] = saved["costs"]
        self.not_dones[:self.size] = saved["not_dones"]
