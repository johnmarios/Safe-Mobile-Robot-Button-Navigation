import numpy as np


class RunningMeanStd:
    def __init__(self, size):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = 1e-4

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)

        # incoming batch statistics
        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # update running mean and variance using Welford's algorithm
        self.mean += delta * batch_count / total_count
        self.var = (self.var * self.count+ batch_var * batch_count + delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count


class MixedObservationNormalizer:
    """Finite observations go to [-1, 1] 
    unbounded ones use running z-score """

    def __init__(self, observation_space, clip=5.0):
        self.low = np.asarray(observation_space.low, dtype=np.float32).reshape(-1)
        self.high = np.asarray(observation_space.high, dtype=np.float32).reshape(-1)
        
        self.dimension = self.low.size
        self.clip = clip

        # create masks for finite and unbounded observation dimensions
        self.finite_mask = np.isfinite(self.low) & np.isfinite(self.high) & (self.high > self.low)
        self.unbounded_mask = ~self.finite_mask

        self.finite_low = self.low[self.finite_mask]
        self.finite_span = self.high[self.finite_mask] - self.finite_low

        self.finite_dimensions = int(self.finite_mask.sum())
        self.unbounded_dimensions = int(self.unbounded_mask.sum())
        self.running = RunningMeanStd(self.unbounded_dimensions)

    def observe(self, observation):
        """Update running statistics for unbounded dimensions."""

        # if there are no unbounded dimensions, there's nothing to update
        if self.unbounded_dimensions == 0:
            return

        # convert observation to a 2D array for batch processing
        values = np.asarray(observation, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        
        # update running statistics for unbounded dimensions
        self.running.update(values[:, self.unbounded_mask])

    def normalize(self, observation):
        values = np.asarray(observation, dtype=np.float32)

        original_shape = values.shape
        if values.ndim == 1:
            values = values.reshape(1, -1)

        normalized = values.copy()

        # normalize finite dimensions to [-1, 1]
        if self.finite_dimensions:
            finite_values = values[:, self.finite_mask]
            normalized[:, self.finite_mask] = np.clip(2.0 * (finite_values - self.finite_low) / self.finite_span - 1.0,-1.0,1.0,)

        # normalize unbounded dimensions using running statistics
        if self.unbounded_dimensions:
            std = np.sqrt(self.running.var + 1e-8)
            unbounded_values = (values[:, self.unbounded_mask] - self.running.mean) / std
            normalized[:, self.unbounded_mask] = np.clip(unbounded_values,-self.clip,self.clip,)

        return normalized.astype(np.float32).reshape(original_shape)

    def state_dict(self):
        
        # Return a dictionary containing the state of the normalizer, including running statistics 
        return {
            "dimension": self.dimension,
            "finite_mask": self.finite_mask,
            "low": self.low,
            "high": self.high,
            "clip": self.clip,
            "running_mean": self.running.mean,
            "running_var": self.running.var,
            "running_count": self.running.count,
        }

    def load_state_dict(self, saved):
        self.running.mean = np.asarray(saved["running_mean"], dtype=np.float64)
        self.running.var = np.asarray(saved["running_var"], dtype=np.float64)
        self.running.count = float(saved["running_count"])
