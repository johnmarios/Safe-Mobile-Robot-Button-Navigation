
import numpy as np 

import torch 




class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.idx = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim),dtype=np.float32)
		self.action = np.zeros((max_size, action_dim),dtype=np.float32)
		self.next_state = np.zeros((max_size, state_dim),dtype=np.float32)
		self.reward = np.zeros((max_size, 1),dtype=np.float32)
		self.cost = np.zeros((max_size, 1), dtype=np.float32)
		self.not_done = np.zeros((max_size, 1),dtype=np.float32)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, cost, done):

		# Store transition in buffer
		self.state[self.idx, :] = state
		self.action[self.idx, :] = action
		self.next_state[self.idx, :] = next_state
		self.reward[self.idx, 0] = reward
		self.cost[self.idx, 0] = cost
		self.not_done[self.idx, 0] = 1.0 - done

		# Update index 
		self.idx = (self.idx + 1) % self.max_size

		# Update current size
		if self.size < self.max_size:
			self.size += 1


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.cost[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)