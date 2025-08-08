import numpy as np
from ._statsbase import StatsBase


class StatsBoot(StatsBase):

    @staticmethod
    def _prefactor_func(_):
        return 1

    def __init__(self, num_config, num_bins, seed=0):
        super().__init__(num_config, num_bins)
        self._prefactor_func = StatsBoot._prefactor_func
        self.seed = seed
        self.ID = 'Boot'

    def generate_bins(self, array_raw_in):
        """It generates resampled bins from raw data using bootstrap."""
        num_config = np.size(array_raw_in, 0)
        if num_config != self.num_config:
            raise ValueError(
                f"data has {num_config} configurations but this object "
                f"was created with num_config={self.num_config}"
            )
        
        rng = np.random.default_rng(self.seed)
        bins = rng.integers(0, num_config, size=(self.num_bins, num_config))
        bins = np.mean(array_raw_in[bins], axis=1)
        return bins
