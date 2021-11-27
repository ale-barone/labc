import numpy as np 
from abc import ABC, abstractmethod

class Istats(ABC):
    """Base class for managing statistics."""

    @abstractmethod
    def generate_bins(self, array_raw_in):
        """It generates resampled bins from raw data."""

    @abstractmethod
    def errFun(self, array_bins_in):
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


class statsBase(Istats):
    
    def __init__(self, *, num_config, num_bins):
        self.num_config = num_config
        self.num_bins = num_bins
        self.prefactor = None

    def generate_bins(self, array_raw_in):
        pass

    def errFun(self, array_mean_in, array_bins_in):
        # error 
        diff = array_bins_in - array_mean_in
        diff2 = diff**2
        err2 = self.prefactor * np.mean(diff2, 0)
        err = np.sqrt(err2)
        return err

    def generate_stats(self, array_raw_in):
        mean = np.mean(array_raw_in, 0)
        bins = self.generate_bins(mean)
        err = self.errFun(mean, bins)  
        return mean, err, bins

    @staticmethod
    def cov2(data_x_in, data_y_in, *, num_bins=None, rangefit=None, thin=1):
        """Compute the covariance of two dataStats objects."""
        
        T = data_x_in.T()
        num_bins = data_x_in.num_bins() if num_bins is None else num_bins

        xmin, xmax = (0, T) if rangefit is None else (rangefit[0], rangefit[1])
        num_points = xmax-xmin
        
        mean_x_cut = data_x_in.mean[xmin:xmax:thin]
        mean_y_cut = data_y_in.mean[xmin:xmax:thin]
        
        Cov = np.array([])
        for j in range(num_bins):
            bins_x_cut_aux = data_x_in.bins[j][xmin:xmax:thin]
            bins_y_cut_aux = data_y_in.bins[j][xmin:xmax:thin]
            
            # Covariance (already applying cuts)
            vec_x = bins_x_cut_aux - mean_x_cut
            vec_y = bins_y_cut_aux - mean_y_cut
            Cov_aux = np.outer(vec_x, vec_y)
            Cov = np.append(Cov, Cov_aux)
    
        Cov = np.reshape(Cov, (num_bins, num_points, num_points))
        Cov = (num_bins-1) * np.mean(Cov, 0)
        return Cov

    def cov(self, data_x_in, *, num_bins=None, rangefit=None, thin=1):
        return self.cov2(data_x_in, data_x_in, num_bins=None, rangefit=None, thin=1)

    def cov_blocks(self, data_array_in, num_bins=None, rangefit=None, thin=1):
        """General covariance matrix possibly for different input arrays."""
        data_array_in = tuple(data_array_in)
        
        n = len(data_array_in) * data_array_in[0].T()
        cov = np.array([])
        for data_x_in in data_array_in:
            cov_block_row = self.cov2(data_x_in, data_array_in[0],  num_bins=num_bins, rangefit=rangefit, thin=thin)
            for data_y_in in data_array_in[1:]:
                cov_block = self.cov2(data_x_in, data_y_in,  num_bins=num_bins, rangefit=rangefit, thin=thin) 
                cov_block_row = np.append(cov_block_row, cov_block, axis=1)
                
            cov = np.append(cov, cov_block_row)
        cov = np.reshape(cov, (n,n))

        return cov

    def corr(self, *arrays_in):
        """General correlation matrix possibly for different input arrays."""
        return NotImplementedError

