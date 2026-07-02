import numpy as np


class MixedObservationNormalizer:
    """Finite observations go to [-1, 1] 
    unbounded ones use running z-score """

    def __init__(self, observation_space, clip=5.0):
        self.low = np.asarray(observation_space.low, dtype=np.float32)
        self.high = np.asarray(observation_space.high, dtype=np.float32)

        # create masks for finite and unbounded observation dimensions
        self.finite_mask = (np.isfinite(self.low) & np.isfinite(self.high) & (self.high > self.low))
        self.unbounded_mask = ~self.finite_mask

        self.finite_low = self.low[self.finite_mask]
        self.finite_span = self.high[self.finite_mask] - self.finite_low

        self.clip = clip
        self.running_mean = np.zeros(self.unbounded_mask.sum(), dtype=np.float64)
        self.running_var = np.ones(self.unbounded_mask.sum(), dtype=np.float64)

    def load_state_dict(self, saved):
        self.running_mean = np.asarray(saved["running_mean"], dtype=np.float64)
        self.running_var = np.asarray(saved["running_var"], dtype=np.float64)

    def normalize(self, observation):
        values = np.asarray(observation, dtype=np.float32).copy()
        
        # normalize finite dimensions to [-1, 1]
        finite_values = values[self.finite_mask]
        values[self.finite_mask] = np.clip(2.0 * (finite_values - self.finite_low) / self.finite_span - 1.0, -1.0,1.0,)

        # normalize unbounded dimensions using running statistics
        unbounded_values = values[self.unbounded_mask]
        unbounded_values = (unbounded_values - self.running_mean) / np.sqrt(self.running_var + 1e-8)

        values[self.unbounded_mask] = np.clip(unbounded_values,-self.clip,self.clip,)

        return values
