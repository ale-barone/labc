import math
import warnings
import numpy as np
from ._statsbase import StatsBase


class StatsJack(StatsBase):

    _MAX_REBIN = 3

    @staticmethod
    def _prefactor_func(num_bins):
        return num_bins - 1

    def __init__(self, num_config, rebin=1):
        if rebin > self._MAX_REBIN:
            raise ValueError(
                f"rebin={rebin} is too large; maximum allowed is {self._MAX_REBIN}. "
                f"Consider pre-averaging your configurations manually."
            )
        if rebin == self._MAX_REBIN:
            warnings.warn(
                f"rebin={rebin} is unusual and may indicate autocorrelation issues. "
                f"Make sure this is intentional.",
                UserWarning, stacklevel=2
            )
        # round up
        num_bins = math.ceil(num_config / rebin) if num_config is not None else None
        super().__init__(num_config, num_bins)
        self._prefactor_func = StatsJack._prefactor_func
        self.rebin = rebin
        self.ID = 'Jack'

    def generate_bins(self, array_raw_in):
        """Generate jackknife bins from raw data."""
        num_config = np.size(array_raw_in, 0)

        # pad if needed so num_config is divisible by rebin
        remainder = num_config % self.rebin
        if remainder != 0:
            n_pad = self.rebin - remainder
            padding = np.repeat(array_raw_in[[-1]], n_pad, axis=0)
            array_raw_in = np.concatenate([array_raw_in, padding], axis=0)

        # average blocks of `rebin` neighbouring configs
        array_rebinned = np.mean(
            array_raw_in.reshape(self.num_bins, self.rebin, -1), axis=1
        )

        # jackknife leave-one-out on the rebinned configs
        bins = np.array([
            np.delete(np.arange(self.num_bins), b, 0) for b in range(self.num_bins)
        ])
        bins = np.mean(array_rebinned[bins], axis=1)
        return bins