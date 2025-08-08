import numpy as np 
from abc import ABC, abstractmethod
from .. import data as dM
from scipy.linalg import block_diag


class Istats(ABC):
    """Base class for managing statistics."""

    @abstractmethod
    def generate_bins(self, array_raw_in):
        """It generates resampled bins from raw data."""

    @abstractmethod
    def err_func(self, array_bins_in):
        """It computes the error from resampled bins."""

    @abstractmethod
    def generate_stats(self, array_raw_in):
        """It computes the 'stats' version returning (array_mean, array_err, array_bins)."""

    @abstractmethod
    def cov(self, *arrays_in):
        """General covariance matrix possibly for different input arrays."""

    @abstractmethod
    def corr(self, *arrays_in):
        """General correlation matrix possibly for different input arrays."""


class StatsBase(Istats):
    
    def __init__(self, num_config=None, num_bins=None, seed=None):
        self.num_config = num_config
        self.num_bins = num_bins
        self.seed = seed
        self._prefactor_func = None
        self.ID = None
    
    def __str__(self):
        out = (
            f"StatsType = '{self.ID}':" 
            f"\n -num_config = {self.num_config}"
            f"\n -num_bins = {self.num_bins}"
            f"\n -seed = {self.seed}"
        )
        return out
    
    def __repr__(self):
        out = (
            f"StatsType.{self.ID}(" 
            f"num_config={self.num_config}, "
            f"num_bins={self.num_bins}, "
            f"seed={self.seed})"
        )
        return out

    def generate_bins(self, array_raw_in):
        raise NotImplementedError

    def _get_num_bins(self, array_bins_in: np.ndarray):
        assert(array_bins_in.ndim==2)
        num_bins = len(array_bins_in)
        return num_bins

    def _get_prefactor(self, array_bins_in: np.ndarray):
        num_bins = self._get_num_bins(array_bins_in)
        return self._prefactor_func(num_bins)

    def _assert_bins_size(self, array_in):
        if self.num_bins is not None:
            if self.num_bins != len(array_in):
                raise ValueError(
                    f"array length {len(array_in)} does not match "
                    f"num_bins={self.num_bins} of this StatsType object"
                )

    def err_func(self, array_mean_in, array_bins_in):
        # error 
        diff2 = (array_bins_in - array_mean_in)**2
        self._assert_bins_size(diff2)
        prefactor = self._get_prefactor(diff2)
        err2 = prefactor * np.mean(diff2, 0)
        err = np.sqrt(err2)
        return err

    def generate_stats(self, array_raw_in):
        mean = np.mean(array_raw_in, 0)
        bins = self.generate_bins(array_raw_in)
        err = self.err_func(mean, bins)  
        return mean, err, bins

    def cov(self, data_x_in, data_y_in=None):
        """Compute the covariance matrix of one or two DataStats objects.

        If only data_x_in is provided, computes the auto-covariance matrix.
        If data_y_in is also provided, computes the cross-covariance matrix.
        Slicing (rangefit, thinning) should be applied to the inputs beforehand.
        """
        if data_y_in is None:
            data_y_in = data_x_in
        num_bins = data_x_in.num_bins()
        cov = np.empty(shape=(num_bins, len(data_x_in), len(data_y_in)))
        for b in range(num_bins):
            vec_x = data_x_in.bins[b] - data_x_in.mean
            vec_y = data_y_in.bins[b] - data_y_in.mean
            cov[b] = np.outer(vec_x, vec_y)
        prefactor = self._get_prefactor(data_x_in.bins)
        return prefactor * np.mean(cov, 0)

    def cov_blocks(self, *data_in):
        """Covariance matrix of multiple DataStats objects treated as one dataset."""
        return self.cov(dM.merge(*data_in))

    def cov_blocks_diag(self, *data_in):
        """Block diagonal covariance matrix of multiple DataStats objects."""
        return block_diag(*[self.cov(data) for data in data_in])

    def corr(self, data_x_in, data_y_in=None):
        """Compute the correlation matrix of one or two DataStats objects.

        If only data_x_in is provided, computes the auto-correlation matrix.
        If data_y_in is also provided, computes the cross-correlation matrix.
        Slicing (rangefit, thinning) should be applied to the inputs beforehand.
        """
        if data_y_in is None:
            data_y_in = data_x_in
        cov = self.cov(data_x_in, data_y_in)
        corr = np.diag(1/data_x_in.err) @ cov @ np.diag(1/data_y_in.err)
        return corr
