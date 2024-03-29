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
        self.prefactor = None
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
        pass

    def _get_num_bins(self, array_bins_in: np.ndarray):
        assert(array_bins_in.ndim==2)
        num_bins = len(array_bins_in)
        return num_bins

    def _get_prefactor(self, array_bins_in: np.ndarray):
        if self.prefactor!=None:
            prefactor = self.prefactor
        else:
            num_bins = self._get_num_bins(array_bins_in)
            if self.ID=='Jack':
                prefactor = num_bins-1
            elif self.ID=='Boot':
                prefactor=1
        return prefactor

    def _assert_bins_size(self, array_in):
        if self.num_bins!=None:
            assert(self.num_bins==len(array_in)),\
            "size of array is different from the number of bins"\
            " 'num_bins' of the object StatsType"

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

    def cov2(self, data_x_in, data_y_in, *, num_bins=None, rangefit=None, thin=1):
        """Compute the covariance matrix of two dataStats objects."""
        
        T = len(data_x_in)
        num_bins = data_x_in.num_bins() if num_bins is None else num_bins
        xmin, xmax = (0, T) if rangefit is None else (rangefit[0], rangefit[1])
        
        data_x_cut_mean = data_x_in.mean[xmin:xmax:thin]
        data_y_cut_mean = data_y_in.mean[xmin:xmax:thin]
        num_points = len(data_x_cut_mean)
        
        cov = np.empty(shape=(num_bins, num_points, num_points))
        for b in range(num_bins):
            bins_x_cut_aux = data_x_in.bins[b][xmin:xmax:thin]
            bins_y_cut_aux = data_y_in.bins[b][xmin:xmax:thin]
            
            # Covariance (already applying cuts)
            vec_x = bins_x_cut_aux - data_x_cut_mean
            vec_y = bins_y_cut_aux - data_y_cut_mean
            cov[b] = np.outer(vec_x, vec_y)
  
        prefactor = self._get_prefactor(data_x_in.bins)
        cov = prefactor * np.mean(cov, 0)
        return cov

    def cov(self, data_x_in, *, num_bins=None, rangefit=None, thin=1):
        return self.cov2(data_x_in, data_x_in, num_bins=num_bins, rangefit=rangefit, thin=thin)

    def cov_blocks(self, *data_in, num_bins=None, rangefit=None, thin=1):
        """General covariance matrix possibly for different input arrays."""
        data_in_merged = dM.merge(*data_in)
        return self.cov(data_in_merged, num_bins=num_bins, rangefit=rangefit, thin=thin)

    def cov_blocks_diag(self, *data_in, num_bins=None, rangefit=None, thin=1):
        """Block diagonal covariance matrix"""
        cov_list = []
        for data in data_in:
            cov_list.append(self.cov(data))
        return block_diag(*cov_list)


    def corr2(self, data_x_in, data_y_in, *, num_bins=None, rangefit=None, thin=1):
        """Compute the correlation matrix of two dataStats objects."""
        cov = self.cov2(data_x_in, data_y_in, 
            num_bins=num_bins, rangefit=rangefit, thin=thin
        )
        xmin, xmax = (0, len(data_x_in)) if rangefit is None else (rangefit[0], rangefit[1])
        err_x = data_x_in.err[xmin:xmax:thin]
        err_y = data_y_in.err[xmin:xmax:thin]
        corr = np.diag(1/err_x)@cov@np.diag(1/err_y)
        return corr

    def corr(self, data_x_in, *, num_bins=None, rangefit=None, thin=1):
        """
        Compute the correlation matrix of the time slices of 
        one dataStats objects.
        """
        corr = self.corr2(data_x_in, data_x_in,
            num_bins=num_bins, rangefit=rangefit, thin=thin
        )
        return corr
