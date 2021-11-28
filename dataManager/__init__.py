import h5py
import numpy as np

import pathlib
from LatticeAB.data import dataContainer as dC
from LatticeAB.data.dataContainer.HDF5.HDF5Utilities import get_statsID
import matplotlib.pyplot as plt



# READER WITH STATS CONNOTATION
class dataStats:    
    """Basic class for data manipulation."""

    def __init__(self, data, statsType):
        self.data = data
        self.mean = np.asarray(data[0])
        self.err = np.asarray(data[1])
        self.bins = np.asarray(data[2:])
        # stats
        self.statsType = statsType
        self.errFun = statsType.errFun

    def T(self):
        return len(self.mean)

    def num_bins(self):
        return len(self.bins)

    # # TODO: change this into a more efficient factory
    # def save(self, file_out):
    #     ext = get_extension(file_out)
    #     if ext=='.h5':
    #         writer = dC.writer(file_out, 'stats')
    #         writer.init_groupStructure(self.statsType, self.stats_par)
    #         writer.add_mean(self.mean)
    #         writer.add_err(self.err)
    #         writer.add_bins(self.bins)
    #     else:
    #         raise NotImplementedError(f"File extension '{ext}' not implemented!")

    # I need a method to pass statsType in a nice way...or I need to avoid giving a defaul statsType='jack'
    # or I can wrap everything/ initialize something globally somewhere (see maybe Ryan, he probably did something of this kind)
    # I need to make sure that a function of dataStats object copies the correct info 


    # need method to print information nicely
    
    def concatenate_dataStats(self, mean, err, bins):
        out = np.array([mean, err])
        out = np.concatenate([out, bins], axis=0)
        return dataStats(out, self.statsType)

    # OVERLOAD OF MATH OPERATIONS
    def __mul__(self, other):
        if isinstance(other, dataStats):
            out_mean = self.mean * other.mean
            out_bins = self.bins * other.bins
            out_err = self.errFun(out_mean, out_bins)
            
            out = self.concatenate_dataStats(out_mean, out_err, out_bins)
        elif isinstance(other, (int, float)):
            out_mean = self.mean * other
            out_bins = self.bins * other
            out_err = self.err * other #self.errFun(out_mean, out_bins)
            
            out = self.concatenate_dataStats(out_mean, out_err, out_bins)
        return out
    
    def __rmul__(self, other):
        if isinstance(other, dataStats):
            out_mean = self.mean * other.mean
            out_bins = self.bins * other.bins
            out_err = self.errFun(out_mean, out_bins)
            
            out = self.concatenate_dataStats(out_mean, out_err, out_bins)
        elif isinstance(other, (int, float)):
            out_mean = self.mean * other
            out_bins = self.bins * other
            out_err  = self.err * other #self.errFun(out_mean, out_bins) #!!
            
            out = self.concatenate_dataStats(out_mean, out_err, out_bins)
        return out
        

    def __truediv__(self, other):
        if isinstance(other, dataStats):
            out_mean = self.mean / other.mean
            out_bins = self.bins / other.bins
            out_err  = self.errFun(out_mean, out_bins)
            
            out = self.concatenate_dataStats(out_mean, out_err, out_bins)
        elif isinstance(other, (int, float)):
            out_mean = self.mean / other
            out_bins = self.bins / other
            out_err  = self.errFun(out_mean, out_bins)
            
            out = self.concatenate_dataStats(out_mean, out_err, out_bins)
        return out


    def __add__(self, other):
        if isinstance(other, dataStats):
            out_mean = self.mean + other.mean
            out_bins = self.bins + other.bins
            out_err  = self.errFun(out_mean, out_bins)
            
            out = self.concatenate_dataStats(out_mean, out_err, out_bins)
            return out 
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, dataStats):
            out_mean = self.mean - other.mean
            out_bins = self.bins - other.bins
            out_err  = self.errFun(out_mean, out_bins)
            
            out = self.concatenate_dataStats(out_mean, out_err, out_bins)
            return out   
        else:
            raise NotImplementedError
    
    def __eq__(self, other):
        if isinstance(other, dataStats):
            # add some printing
            if np.allclose(self.mean, other.mean, atol=1e-15) and np.allclose(self.bins, other.bins, atol=1e-15):
                return True
            else:
                return False

    def __getitem__(self, key):
        out_mean = np.asarray(self.mean[key])
        out_err = np.asarray(self.err[key])
        new_size = out_mean.size
        if new_size==1:
            #out_bins = self.bins[:,key]
            out_mean = np.array([out_mean])
            out_err = np.array([out_err])

        out_bins = np.reshape(self.bins[:,key], (self.num_bins(), new_size))  
        out = self.concatenate_dataStats(out_mean, out_err, out_bins)
        return out

    # TODO some append method?


    def rel_diff(self, other):
        assert(isinstance(other, dataStats))
        out_mean = np.abs((self.mean - other.mean) / self.mean)
        out_bins = np.abs((self.bins - other.bins) / self.bins)
        out_err  = self.errFun(out_mean, out_bins)

        out = np.array([out_mean, out_err])
        out = dataStats(np.concatenate([out, out_bins], axis=0), self.statsType)
        return out 


# class corr(dataStats):

#     def meff(self):
#         meff_mean = np.log( self.mean[:-1]/ self.mean[1:] )
#         meff_bins = np.log( np.apply_along_axis(lambda x : x[:-1], 1, self.bins ) / np.apply_along_axis(lambda x : x[1:], 1, self.bins ))
#         meff_err = self.err_fun(meff_mean, meff_bins)
        
#         meff_stats = self.data_stats_concatenate(meff_mean, meff_err, meff_bins)
#         return meff_stats
        
    
#     def plot_meff(self, xmin, xmax, shift=0, err=True, *args, **kwargs):
#         x = np.arange(xmin, xmax)
#         y = self.meff().mean[xmin:xmax]
        
#         if err == False:
#             y_err = None
#         elif err == True:
#             y_err = self.meff().err[xmin:xmax]
            
#         plt.errorbar(x + shift, y, yerr = y_err, *args, **kwargs)
#         plt.xlabel(r'$t$', fontsize = 14)
#         plt.ylabel(r'$m_{eff}$', fontsize=14)
#         plt.title(r'$\log ( \, C(t) \, / \, C(t+1) \, )$', fontsize = 14)
