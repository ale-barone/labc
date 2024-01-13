import numpy as np
from ._statsbase import StatsBase


class StatsJack(StatsBase):

    def __init__(self, num_config, num_bins):
        super().__init__(num_config, num_bins)
        if num_bins!=None:
            self.prefactor = num_bins -1
        self.ID = 'Jack'

    def generate_bins(self, array_raw_in):
        """It generates resampled bins from raw data using jackknife."""
        num_config = np.size(array_raw_in, 0)
        # FIXME to allow rebinning
        # assert(num_config==self.num_bins),\
        #     "num_bins of the object does not agree with StatsType.Jack"
        bins = np.array([np.delete(np.arange(num_config), b, 0) for b in range(num_config)])
        bins = np.mean(array_raw_in[bins], axis=1)
        return bins
