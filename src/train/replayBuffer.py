import numpy as np
import torch


import numpy as np
import torch


class ReplayBuffer(object):
    """
    saves transitions (state, action, next_state, reward, cost, done) in a circular buffer
    and returns random batches for training

    """

    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        '''
        Initializes the replay buffer.

        Args:
            state_dim (int): Dimension of the observation/state space.
            action_dim (int): Dimension of the action space.
            max_size (int): Maximum number of transitions to store in the buffer.
        '''
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)

        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.cost = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, cost, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state

        self.reward[self.ptr] = reward
        self.cost[self.ptr] = cost
        self.not_done[self.ptr] = 1.0 - float(done)

        # Circular buffer:
        self.ptr = (self.ptr + 1) % self.max_size

        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # get a random batch of transitions 
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[indices]).to(self.device),
            torch.FloatTensor(self.action[indices]).to(self.device),
            torch.FloatTensor(self.next_state[indices]).to(self.device),
            torch.FloatTensor(self.reward[indices]).to(self.device),
            torch.FloatTensor(self.cost[indices]).to(self.device),
            torch.FloatTensor(self.not_done[indices]).to(self.device),
        )