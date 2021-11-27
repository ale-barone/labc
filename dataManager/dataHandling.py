#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:29:13 2021

@author: abarone
"""

import h5py
import numpy as np
from LatticeAB import stats
import sys
import os.path


# =============================================================================
# DATA GENERATION
# =============================================================================

# DATA GENERATION FROM HDF5
def generate_datah5(file, t_roll, meson, z):

    f   = h5py.File(file,'r')
    key = list(f.keys())
    dset1 = f[key[0]]
    dset2 = dset1[meson]
    dset3 = dset2['corr']
    
    y  = np.array([])  
    T = len(dset3)
    for k in range(T):
        if z == "re":
            y = np.append(y, dset3[k][0])
        elif z == "im":
            y = np.append(y, dset3[k][1])

    y = np.roll(y, -t_roll)
    return y

# SAVING DATA
def savedata(data_mean, data_err, data_bins, file):
    tosave = np.concatenate([[data_mean, data_err], data_bins])
    with open(file,'w') as new: 
        if data_mean.size==1:
            np.savetxt(new, tosave, newline=" ")
        else:    
            np.savetxt(new, np.transpose(tosave))

def savedata_stats(data_stats, file):
    savedata(data_stats.mean(), data_stats.err(), data_stats.bins(), file)


# =============================================================================
# DATA READING
# =============================================================================

   

# BASIC READER of file with columns without any operation or statistics
# I could wrap the reader from dataContainer to specialize it for data_stats structure (as an initiliazer?)
class data_reader(object):
    def __init__(self, data, unpack=True, folded=False, *args, **kwargs):
        # initialization with file name
        if type(data) == str:
            extension = os.path.splitext(data)[1]
            if extension=='.dat':
                self.data = np.loadtxt(data, unpack=unpack, *args, **kwargs)
        # initialization with numpy array
        elif type(data) == np.ndarray:
            self.data = data
        else:
            sys.exit("Object of the class " + str(self) + 
                     "must be a file_name or "
                     "a numpy.ndarray with shape (num_bins, T)") 
        if folded == True:
            #self.data = np.apply_along_axis(lambda x: (x + np.roll(x[::-1], 1))/2, 1, self.data)
            # error??
            raise NotImplementedError
            
    
# READER WITH STATS CONNOTATION
class data_stats(data_reader):      
    def __init__(self, data, stats_type='jack', unpack=True, folded=False, *args, **kwargs):
        super().__init__(data, unpack, folded, *args, **kwargs)
        
        self.stats_type = stats_type
        if stats_type=='jack':
            self.err_fun = stats.jackknife_err
        elif stats_type=='boot':
            self.err_fun = stats.bootstrap_err
        
    def mean(self):
        return np.asarray(self.data[0])
    
    def bins(self, binn=None):
        binns = self.data[2:]
        if binn is None:
            return np.asarray(binns) 
        else:
            return np.asarray(binns[binn])
        
    def err(self):
        return np.asarray(self.data[1])

    def T(self):
        return len(self.mean())

    def num_bins(self):
        return len(self.bins())

    # TMP! push a value in front everywhere
    def push(self, other):
        out_mean = np.append( [other], self.mean())
        out_bins = np.array([ np.append([other], self.bins()[b]) for b in range(self.num_bins()) ])
        out_err  = np.append([0], self.err())
        
        out = np.array([out_mean, out_err])
        out = data_stats(np.concatenate([out, out_bins], axis=0), self.stats_type)
        return out

    # I need a method to pass stats_type in a nice way...or I need to avoid giving a defaul stats_type='jack'
    # or I can wrap everything/ initialize something globally somewhere (see maybe Ryan, he probably did something of this kind)
    # I need to make sure that a function of data_stats object copies the correct info 


    # need method to print information nicely
    

    # OVERLOAD OF MATH OPERATIONS
    def __mul__(self, other):
        if isinstance(other, data_stats):
            out_mean = self.mean() * other.mean()
            out_bins = self.bins() * other.bins()
            out_err  = self.err_fun(out_mean, out_bins)
            
            out = np.array([out_mean, out_err])
            out = data_stats(np.concatenate([out, out_bins], axis=0), self.stats_type)
        elif isinstance(other, (int, float)):
            out_mean = self.mean() * other
            out_bins = self.bins() * other
            out_err  = self.err() * other #self.err_fun(out_mean, out_bins)
            
            out = np.array([out_mean, out_err])
            out = data_stats(np.concatenate([out, out_bins], axis=0), self.stats_type)
        return out
    
    def __rmul__(self, other):
        if isinstance(other, data_stats):
            out_mean = self.mean() * other.mean()
            out_bins = self.bins() * other.bins()
            out_err  = self.err_fun(out_mean, out_bins)
            
            out = np.array([out_mean, out_err])
            out = data_stats(np.concatenate([out, out_bins], axis=0), self.stats_type)
        elif isinstance(other, (int,float)):
            out_mean = self.mean() * other
            out_bins = self.bins() * other
            out_err  = self.err() * other #self.err_fun(out_mean, out_bins) #!!
            
            out = np.array([out_mean, out_err])
            out = data_stats(np.concatenate([out, out_bins], axis=0), self.stats_type)
        return out
        

    def __truediv__(self, other):
        if isinstance(other, data_stats):
            out_mean = self.mean() / other.mean()
            out_bins = self.bins() / other.bins()
            out_err  = self.err_fun(out_mean, out_bins)
            
            out = np.array([out_mean, out_err])
            out = data_stats(np.concatenate([out, out_bins], axis=0), self.stats_type)
        elif isinstance(other, (int, float)):
            out_mean = self.mean() / other
            out_bins = self.bins() / other
            out_err  = self.err_fun(out_mean, out_bins)
            
            out = np.array([out_mean, out_err])
            out = data_stats(np.concatenate([out, out_bins], axis=0), self.stats_type)
        return out

    
    def __eq__(self, other):
        if isinstance(other, data_stats):
            # add some printing
            if np.allclose(self.mean(), other.mean(), atol=1e-15) and np.allclose(self.bins(), other.bins(), atol=1e-15):
                return True
            else:
                return False

    def __add__(self, other):
        if isinstance(other, data_stats):
            out_mean = self.mean() + other.mean()
            out_bins = self.bins() + other.bins()
            out_err  = self.err_fun(out_mean, out_bins)
            
            out = np.array([out_mean, out_err])
            out = data_stats(np.concatenate([out, out_bins], axis=0), self.stats_type)
            return out 

    def __sub__(self, other):
        if isinstance(other, data_stats):
            out_mean = self.mean() - other.mean()
            out_bins = self.bins() - other.bins()
            out_err  = self.err_fun(out_mean, out_bins)
            
            out = np.array([out_mean, out_err])
            out = data_stats(np.concatenate([out, out_bins], axis=0), self.stats_type)
            return out   
    
    def __getitem__(self, key):
        out_mean = np.asarray(self.mean()[key])
        out_err  = np.asarray(self.err()[key])
        ran  = int(len(out_mean))
        out_bins = np.reshape(self.bins()[:,key], (self.num_bins(), ran))  
        #out_bins  = self.bins()[:,key],
        out = data_stats(np.concatenate( ([out_mean, out_err], out_bins), axis=0 ))
        return out

    # TODO some append method?


    def rel_diff(self, other):
        assert(isinstance(other, data_stats))
        out_mean = np.abs((self.mean() - other.mean()) / self.mean())
        out_bins = np.abs((self.bins() - other.bins()) / self.bins())
        out_err  = self.err_fun(out_mean, out_bins)

        out = np.array([out_mean, out_err])
        out = data_stats(np.concatenate([out, out_bins], axis=0), self.stats_type)
        return out 


# STATS DATA FOR JACKKNIFE
class data_jack(data_stats):
        
    def __init__(self, data, unpack=True, folded=False, *args, **kwargs):
        super().__init__(data, 'jack', unpack, folded, *args, **kwargs)
    
    
    def jack(self, binn=None):
        jack = self.data[2:]
        if binn is None:
            return jack 
        else:
            return jack[binn]
    
    
    # def num_config(self):
    #     return self.data.shape[0]-2
    
    # def num_tslices(self):
    #     return self.data.shape[1]

    # def save(self, file):
    #     savedata_stats(self, file)

# STATS DATA FOR BOOTSTRAP
class data_boot(data_stats):
        
    def __init__(self, data, unpack=True, folded=False, *args, **kwargs):
        super().__init__(data, 'boot', unpack, folded, *args, **kwargs)
    
    def boot(self, binn=None):
        boot = self.data[2:]
        if binn is None:
            return boot 
        else:
            return boot[binn]



#
class data_stats_row(data_stats):

    def __init__(self, data, row=0, stats_type='jack', unpack=False, folded=False, *args, **kwargs):
        super().__init__(data, stats_type, unpack, folded, *args, **kwargs)
        self.data = self.data[row]
        
    # def __init__(self, data, row, *args, **kwargs):
    #     if type(data) == str:
    #         self.data = np.loadtxt(data, *args, **kwargs)[row]

    #     elif type(data) == np.ndarray:
    #         if data.ndim>1:
    #             sys.exit("Wrong structure of the input 'data'\n" + 
    #                       "Object of the class " + str(self) +
    #                       " is expected to contain a single row")
    #         else:
    #             self.data = data
    #     else:
    #         sys.exit("Object of the class " + str(self) + 
    #                  " must be a file_name or "
    #                  "a numpy.ndarray with shape (num_config, t)")   


class data_gauge_raw(data_reader):

    def __init__(self, data, unpack=True, folded=False, *args, **kwargs):
        super().__init__(data, unpack, folded, *args, **kwargs)

    def mean(self):
        return np.mean(self.data, 0)
  
    def jack(self): 
        return stats.jackknife_bin(self.data)

    def boot(self, boot_par):
        return stats.bootstrap_bin(self.data, boot_par)

    def err(self, stats_type, boot_par):
        if   stats_type=='jack': err = stats.jackknife_err(self.mean(), self.jack())
        elif stats_type=='boot': err = stats.bootstrap_err(self.mean(), self.boot(boot_par))
        return err
    
    def config(self, c):
        return self.data[c]
    
    def num_config(self):
        return self.data.shape[0]
    
    def num_tslices(self):
        return self.data.shape[1]
    
    # could actually put this as an init and derive this from data_stats
    def make_stats(self):
        data = np.concatenate([[self._mean, self._err], self._bins])
        data = data_stats(data, self.stats_type)
        return data

# GAUGE DATA READING # TO BE CHECKED!!
class data_gauge(data_stats):

    def __init__(self, data, stats_type='jack', stats_par=None, unpack=True, row=None, folded=False, *args, **kwargs):
        super().__init__(data, stats_type, unpack, folded, *args, **kwargs)
        self.stats_par  = stats_par

        if stats_type=='jack':
            self.err_fun   = stats.jackknife_err
            self.stats_fun = stats.jackknife
        elif stats_type=='boot':
            self.err_fun   = stats.bootstrap_err
            self.stats_fun = stats.bootstrap
        self._mean, self._err, self._bins = self.stats_fun(self.data, self.stats_par)

    def mean(self):
        return self._mean
    
    def bins(self):
        return self._bins

    def err(self):
        return self._err

    
    # def jack(self):
    #     jack = stats.jackknife_bin(self.data)
    #     return jack

    # def boot(self, boot_par=None):
    #     if boot_par==None:
    #         boot = stats.bootstrap_bin(self.data, self.stats_par)
    #     else:
    #         boot = stats.bootstrap_bin(self.data, boot_par)     
    #     return boot

    
    def config(self, c):
        return self.data[c]
    
    def num_config(self):
        return self.data.shape[0]
    
    def num_tslices(self):
        return self.data.shape[1]
    
    # could actually put this as an init and derive this from data_stats
    def make_stats(self):
        data = np.concatenate([[self._mean, self._err], self._bins])
        data = data_stats(data, self.stats_type)
        return data
    
    
def cutrange(data_in, range):
    tmin, tmax = range
    assert isinstance(data_in, data_stats), TypeError
    err_fun = data_in.err_fun
    out_mean = data_in.mean()[tmin:tmax]
    out_bins = data_in.bins()[:,tmin:tmax]
    out_err  = err_fun(out_mean, out_bins)

    out = data_stats(np.concatenate([[out_mean, out_err], out_bins]))
    return out
            
        
    
    
    