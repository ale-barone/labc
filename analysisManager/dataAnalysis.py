#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:29:13 2021

@author: abarone
"""

import h5py
import numpy as np
from LatticeAB import stats
import pathlib
from LatticeAB.data import dataContainer as dC
from LatticeAB.data.dataContainer.HDF5.HDF5Utilities import get_statsID
import matplotlib.pyplot as plt


# =============================================================================
# DATA GENERATION
# =============================================================================

def get_extension(file):
    path = pathlib.Path(file)
    return path.suffix


# =============================================================================
# DATA READING
# =============================================================================

def concatenate(mean, err, bins):
    out = np.array([mean, err])
    out = np.concatenate([out, bins], axis=0)
    return out


def quick_gauge_h5reader(file, stats_type, stats_par, tsrc_list):
    with h5py.File(file, 'r') as hf:
        all_config = np.asarray([np.asarray(hf[f'corr/t{tsrc}/t{tsrc}']) for tsrc in tsrc_list])
        mean = np.mean(all_config, 0)
        if stats_type=='jack':
            stats_fun = stats.jackknife
        elif stats_type=='boot':
            stats_fun = stats.bootstrap
        mean, err, bins = stats_fun(mean, stats_par)
        out = concatenate(mean, err, bins)
    return out

def quick_stats_h5reader(file, stats_type, stats_par):
    with h5py.File(file, 'r') as hf:
        assert(hf.attrs['stats_type']==stats_type), "Different stats_type!"
        if stats_type=='jack':
            assert(int(hf['bins'].attrs['stats_par'])==stats_par), "Different stats_par!"
        elif stats_type=='boot':
            par0, par1 = list(map(int,(hf['bins'].attrs['stats_par'].split(','))))   #hf.attrs['stats_type'].split(',')
            assert(par0==stats_par[0]), "Par0 (num_bins) is different!"
            assert(par1==stats_par[1]), "Par1 (seed) is different!"
        mean = np.asarray(hf['mean/mean'])
        err = np.asarray(hf['err/err'])
        bins = np.asarray(hf['bins/bins'])
        out = concatenate(mean, err, bins)
    return out


def get_stats_fun(stats_type, stats_par):
    if stats_type=='jack':
        stats_fun = stats.jackknife
    elif stats_type=='boot':
        def wrapped_bootstrap(array_in):
            return stats.bootstrap(array_in, boot_par=stats_par)    
        stats_fun = wrapped_bootstrap
    else:
        raise ValueError(f"stats_type '{stats_type}' unknown.")
    return stats_fun

def get_err_fun(stats_type):
    if stats_type=='jack':
        err_fun = stats.jackknife_err
    elif stats_type=='boot':  
        err_fun = stats.bootstrap_err
    else:
        raise ValueError(f"stats_type '{stats_type}' unknown.")
    return err_fun



class analysis:
    """Initializer for the analysis."""

    def __init__(self, *, stats_type, stats_par, tsrc_list): # I need also config list in principle
        self.tsrc_list = tsrc_list
        self.stats_type = stats_type
        self.stats_par = stats_par
        
        # self.stats_fun = get_stats_fun(stats_type, stats_par)
        # self.err_fun = get_err_fun(stats_type)
    
    def dataStats(self, file):
        statsID = get_statsID(file)
        if statsID=='gauge':
            data = quick_gauge_h5reader(file, self.stats_type, self.stats_par, self.tsrc_list)
        elif statsID=='stats':
            data = quick_stats_h5reader(file, self.stats_type, self.stats_par)
        return data_stats(data, self.stats_type, self.stats_par)

    def corrStats(self, file):
        statsID = get_statsID(file)
        if statsID=='gauge':
            data = quick_gauge_h5reader(file, self.stats_type, self.stats_par, self.tsrc_list)
        elif statsID=='stats':
            data = quick_stats_h5reader(file, self.stats_type, self.stats_par)
        return corr(data, self.stats_type, self.stats_par)

    
    # I NEED A METHOD TO EASILY APPLY A GENERIC FUNCTION
    # TO dataStats object


    
# READER WITH STATS CONNOTATION
class data_stats:      
    def __init__(self, data, stats_type, stats_par):
        self.data = data
        self.mean = np.asarray(data[0])
        self.err =  np.asarray(data[1])
        self.bins =  np.asarray(data[2:])
        self.stats_type = stats_type
        self.stats_par = stats_par
        
        self.stats_fun = get_stats_fun(stats_type, stats_par) # carefull here, this makes sense only for 'gauge' files!
        self.err_fun = get_err_fun(stats_type)

    def T(self):
        return len(self.mean)

    def num_bins(self):
        return len(self.bins)


    def save(self, file_out):
        ext = get_extension(file_out)
        if ext=='.h5':
            writer = dC.writer(file_out, 'stats')
            writer.init_groupStructure(self.stats_type, self.stats_par)
            writer.add_mean(self.mean)
            writer.add_err(self.err)
            writer.add_bins(self.bins)
        else:
            raise NotImplementedError(f"File extension '{ext}' not implemented!")

    def push(self, other):
        out_mean = np.append( [other], self.mean)
        out_bins = np.array([ np.append([other], self.bins[b]) for b in range(self.num_bins()) ])
        out_err  = np.append([0], self.err)
        
        out = np.array([out_mean, out_err])
        out = self.data_stats_concatenate(out_mean, out_err, out_bins)
        return out

    # I need a method to pass stats_type in a nice way...or I need to avoid giving a defaul stats_type='jack'
    # or I can wrap everything/ initialize something globally somewhere (see maybe Ryan, he probably did something of this kind)
    # I need to make sure that a function of data_stats object copies the correct info 


    # need method to print information nicely
    
    def data_stats_concatenate(self, mean, err, bins):
        out = np.array([mean, err])
        out = np.concatenate([out, bins], axis=0)
        return data_stats(out, self.stats_type, self.stats_par)

    # OVERLOAD OF MATH OPERATIONS
    def __mul__(self, other):
        if isinstance(other, data_stats):
            out_mean = self.mean * other.mean
            out_bins = self.bins * other.bins
            out_err  = self.err_fun(out_mean, out_bins)
            
            out = self.data_stats_concatenate(out_mean, out_err, out_bins)
        elif isinstance(other, (int, float)):
            out_mean = self.mean * other
            out_bins = self.bins * other
            out_err  = self.err * other #self.err_fun(out_mean, out_bins)
            
            out = self.data_stats_concatenate(out_mean, out_err, out_bins)
        return out
    
    def __rmul__(self, other):
        if isinstance(other, data_stats):
            out_mean = self.mean * other.mean
            out_bins = self.bins * other.bins
            out_err  = self.err_fun(out_mean, out_bins)
            
            out = self.data_stats_concatenate(out_mean, out_err, out_bins)
        elif isinstance(other, (int, float)):
            out_mean = self.mean * other
            out_bins = self.bins * other
            out_err  = self.err * other #self.err_fun(out_mean, out_bins) #!!
            
            out = self.data_stats_concatenate(out_mean, out_err, out_bins)
        return out
        

    def __truediv__(self, other):
        if isinstance(other, data_stats):
            out_mean = self.mean / other.mean
            out_bins = self.bins / other.bins
            out_err  = self.err_fun(out_mean, out_bins)
            
            out = self.data_stats_concatenate(out_mean, out_err, out_bins)
        elif isinstance(other, (int, float)):
            out_mean = self.mean / other
            out_bins = self.bins / other
            out_err  = self.err_fun(out_mean, out_bins)
            
            out = self.data_stats_concatenate(out_mean, out_err, out_bins)
        return out



    def __add__(self, other):
        if isinstance(other, data_stats):
            out_mean = self.mean + other.mean
            out_bins = self.bins + other.bins
            out_err  = self.err_fun(out_mean, out_bins)
            
            out = self.data_stats_concatenate(out_mean, out_err, out_bins)
            return out 
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, data_stats):
            out_mean = self.mean - other.mean
            out_bins = self.bins - other.bins
            out_err  = self.err_fun(out_mean, out_bins)
            
            out = self.data_stats_concatenate(out_mean, out_err, out_bins)
            return out   
        else:
            raise NotImplementedError
    
    def __eq__(self, other):
        if isinstance(other, data_stats):
            # add some printing
            if np.allclose(self.mean, other.mean, atol=1e-15) and np.allclose(self.bins, other.bins, atol=1e-15):
                return True
            else:
                return False

    def __getitem__(self, key):
        out_mean = np.asarray(self.mean[key])
        out_err  = np.asarray(self.err[key])
        new_size  = out_mean.size
        if new_size==1:
            #out_bins = self.bins[:,key]
            out_mean = np.array([out_mean])
            out_err = np.array([out_err])

        out_bins = np.reshape(self.bins[:,key], (self.num_bins(), new_size))  
        out = self.data_stats_concatenate(out_mean, out_err, out_bins)
        return out

    # TODO some append method?


    def rel_diff(self, other):
        assert(isinstance(other, data_stats))
        out_mean = np.abs((self.mean - other.mean) / self.mean)
        out_bins = np.abs((self.bins - other.bins) / self.bins)
        out_err  = self.err_fun(out_mean, out_bins)

        out = np.array([out_mean, out_err])
        out = data_stats(np.concatenate([out, out_bins], axis=0), self.stats_type)
        return out 

    
    
class corr(data_stats):

    def meff(self):
        meff_mean = np.log( self.mean[:-1]/ self.mean[1:] )
        meff_bins = np.log( np.apply_along_axis(lambda x : x[:-1], 1, self.bins ) / np.apply_along_axis(lambda x : x[1:], 1, self.bins ))
        meff_err  = self.err_fun(meff_mean, meff_bins)
        
        meff_stats = self.data_stats_concatenate(meff_mean, meff_err, meff_bins)
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
        
            
def cov_jackknife_full_v1(data_x_in, data_y_in,  num_bins=None, rangefit=None, thin=1):
    assert(isinstance(data_x_in, data_stats))
    
    T        = data_x_in.T()
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
        vec_x     = bins_x_cut_aux - mean_x_cut
        vec_y     = bins_y_cut_aux - mean_y_cut
        Cov_aux = np.outer(vec_x, vec_y)
        Cov     = np.append(Cov, Cov_aux)
   
    Cov = np.reshape(Cov, (num_bins, num_points, num_points))
    Cov = (num_bins -1 ) * np.mean(Cov, 0)
    return Cov 
    
def cov_jackknife_general(data_array_in, num_bins=None, rangefit=None, thin=1):
    
    n = len(data_array_in) * data_array_in[0].T()

    cov = np.array([])
    for data_x_in in data_array_in:
    
        for data_y_in in data_array_in:
            if data_y_in == data_array_in[0]: 
                cov_block_row = cov_jackknife_full_v1(data_x_in, data_y_in,  num_bins=num_bins, rangefit=rangefit, thin=thin)
            else:
                cov_block = cov_jackknife_full_v1(data_x_in, data_y_in,  num_bins=num_bins, rangefit=rangefit, thin=thin) 
                cov_block_row = np.append(cov_block_row, cov_block, axis=1)

        cov = np.append(cov, cov_block_row)
    
    cov = np.reshape(cov, (n,n))

    return cov
