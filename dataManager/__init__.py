import h5py
import numpy as np

from math import log10, floor
from LatticeABC.dataManager import dataContainer as dC
from .Utilities import _get_extension
from LatticeABC.dataManager.dataContainer.HDF5.HDF5Utilities import get_statsID
import matplotlib.pyplot as plt

# TODO: need to decide how I want pass data into dataStats
def concatenate_stats(mean, err, bins):
    out = np.array([mean, err])
    out = np.concatenate([out, bins], axis=0)
    return out

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

    # def __repr__(self) -> str:
    #     def get_power(num):
    #         return floor(log10(num))    

    #     def count_significantDigits(num):
    #         double = 15
    #         power_num = floor(log10(num))
    #         num_exp = f"{num:.{double}e}"
    #         print(num_exp)
    #         digits = str( float(num_exp) / 10**power_num ).replace('.', '')
    #         return len(digits)

    #     def print_single(mean, err):
    #         rel_err = err/mean
    #         power_rel = floor(log10(rel_err))  
    #         power_err = floor(log10(err))
            
    #         if power_rel<-4: 
    #             out = f'{mean:.4e} +- {err:.4e}' 
                
    #         elif power_rel>0:
    #             out = 'large error!'  
                
    #         else:
    #             precision = np.abs(floor(log10(err))-1)
    #             err_out = err / (10**(power_err-1))
    #             if 
    #             out = f'{mean:.{precision}f}({err_out:.0f})'
         
    #         return out

    #     print('[')
    #     for m, e in zip(self.mean, self.err):
    #         print(print_single(m, e), ',')
    #     print(']')

    def T(self):
        return len(self.mean)

    def num_bins(self):
        return len(self.bins)

    # TODO: change this into a more efficient factory
    def save(self, file_out):
        ext = _get_extension(file_out)
        if ext=='.h5':
            writer = dC.writer(file_out, 'stats')
            writer.init_groupStructure(self.statsType)
            writer.add_mean(self.mean)
            writer.add_err(self.err)
            writer.add_bins(self.bins)
        else:
            raise NotImplementedError(f"File extension '{ext}' not implemented!")


    # TODO: need method to print information nicely
    
    def concatenate_dataStats(self, mean, err, bins):
        out = np.array([mean, err])
        out = np.concatenate([out, bins], axis=0)
        return dataStats(out, self.statsType)
    
    # not sure if these methods (NOT the dunders) should create a new object...
    def append(self, other):
        out_mean = np.append([other], self.mean)
        out_bins = np.array([ np.append([other], self.bins[b]) for b in range(self.num_bins()) ])
        out_err = self.errFun(out_mean, out_bins)
        
        out = self.concatenate_dataStats(out_mean, out_err, out_bins)
        return out
    
    def roll(self, shift):
        out_mean = np.roll(self.mean, shift)
        out_bins = np.roll(self.bins, shift, axis=1)
        out_err = np.roll(self.err, shift)

        out = self.concatenate_dataStats(out_mean, out_err, out_bins)
        return out

  

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
            out_err = self.err * other #self.errFun(out_mean, out_bins) #!!
            
            out = self.concatenate_dataStats(out_mean, out_err, out_bins)
        return out  

    def __truediv__(self, other):
        if isinstance(other, dataStats):
            out_mean = self.mean / other.mean
            out_bins = self.bins / other.bins
            out_err = self.errFun(out_mean, out_bins)
            
            out = self.concatenate_dataStats(out_mean, out_err, out_bins)
        elif isinstance(other, (int, float)):
            out_mean = self.mean / other
            out_bins = self.bins / other
            out_err = self.errFun(out_mean, out_bins)
            
            out = self.concatenate_dataStats(out_mean, out_err, out_bins)
        return out


    def __add__(self, other):
        if isinstance(other, dataStats):
            out_mean = self.mean + other.mean
            out_bins = self.bins + other.bins
            out_err = self.errFun(out_mean, out_bins)
            
            out = self.concatenate_dataStats(out_mean, out_err, out_bins)
            return out 
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, dataStats):
            out_mean = self.mean - other.mean
            out_bins = self.bins - other.bins
            out_err = self.errFun(out_mean, out_bins)
            
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

# need to move this from here!
class corrStats(dataStats):

    def meff(self):
        meff_mean = np.log( self.mean[:-1]/ self.mean[1:] )
        meff_bins = np.log( np.apply_along_axis(lambda x : x[:-1], 1, self.bins ) / np.apply_along_axis(lambda x : x[1:], 1, self.bins ))
        meff_err = self.errFun(meff_mean, meff_bins)
        
        meff_stats = self.concatenate_dataStats(meff_mean, meff_err, meff_bins)
        return meff_stats
        
    
    def plot_meff(self, xmin, xmax, shift=0, err=True, *args, **kwargs):
        x = np.arange(xmin, xmax)
        y = self.meff().mean[xmin:xmax]
        
        if err == False:
            y_err = None
        elif err == True:
            y_err = self.meff().err[xmin:xmax]
            
        plt.errorbar(x + shift, y, yerr = y_err, *args, **kwargs)
        plt.xlabel(r'$t$', fontsize = 14)
        plt.ylabel(r'$m_{eff}$', fontsize=14)
        plt.title(r'$\log ( \, C(t) \, / \, C(t+1) \, )$', fontsize = 14)
