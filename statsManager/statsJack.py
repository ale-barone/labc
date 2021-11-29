import numpy as np
from .statsBase import statsBase

class statsJack(statsBase):

    def __init__(self, num_config):
        super().__init__(num_config, num_config)
        self.prefactor = num_config -1
        self.ID = 'jack'

    def generate_bins(self, array_raw_in):
        """It generates resampled bins from raw data using jackknife."""
        num_bins = np.size(array_raw_in, 0)
        T = np.size(array_raw_in, 1)

        bins = np.array([])
        for j in range(self.num_bins):
            array_raw_in_delete1 = np.delete(array_raw_in, j, 0)
            bin_j = np.mean(array_raw_in_delete1, 0)
            bins = np.append(bins, bin_j)   
        bins = np.reshape(bins, (num_bins, T))
        return bins



