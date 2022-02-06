import numpy as np
from .statsBase import statsBase

class statsJack(statsBase):

    def __init__(self, num_config):
        super().__init__(num_config, num_config)
        self.prefactor = num_config -1
        self.ID = 'jack'

    def generate_bins(self, array_raw_in):
        """It generates resampled bins from raw data using jackknife."""
        bins = np.array([np.delete(np.arange(self.num_bins), b, 0) for b in range(self.num_bins)])
        bins = np.mean(array_raw_in[bins], axis=1)
        return bins


