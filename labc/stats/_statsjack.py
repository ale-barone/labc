import numpy as np
from ._statsbase import StatsBase


class StatsJack(StatsBase):

    def __init__(self, num_config):
        super().__init__(num_config, num_config)
        self.prefactor = num_config -1
        self.ID = 'Jack'

    def generate_bins(self, array_raw_in):
        """It generates resampled bins from raw data using jackknife."""
        num_config = np.size(array_raw_in, 0)
        assert(num_config==self.num_bins),\
            "num_bins of the object does not agree with StatsType.Jack"
        bins = np.array([np.delete(np.arange(self.num_bins), b, 0) for b in range(self.num_bins)])
        bins = np.mean(array_raw_in[bins], axis=1)
        return bins
