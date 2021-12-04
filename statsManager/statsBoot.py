import numpy as np
from .statsBase import statsBase

class statsBoot(statsBase):

    def __init__(self, num_config, num_bins, seed=0):
        super().__init__(num_config, num_bins)
        self.prefactor = 1
        self.seed = seed
        self.ID = 'boot'

    def generate_bins(self, array_raw_in):
        """It generates resampled bins from raw data using jackknife."""
        num_config = np.size(array_raw_in, 0)
        T = np.size(array_raw_in, 1)

        bins = np.random.randint(0, num_config, size=(num_config, self.num_bins)).transpose() 
        bins = np.mean(array_raw_in[bins], axis=1)
        return bins
