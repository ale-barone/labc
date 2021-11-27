#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:20:24 2021

@author: abarone
"""

import numpy as np
from LatticeAB.data import dataHandling as dt
import sys


# GENERAL
#def cov_matrix(array_in_bins_x, array_in_bins_x):


# =============================================================================
#  JACKKNIFE
# =============================================================================

# JACKKNIFE BIN
def jackknife_bin(array_in, *args):
    array_in_jack = np.array([])
    num_config = np.size(array_in, 0)
    T          = np.size(array_in, 1)
    for j in range(num_config):
        array_in_delete1      = np.delete(array_in,j,0)
        array_in_delete1_mean = np.mean(array_in_delete1, 0)
        array_in_jack         = np.append(array_in_jack, array_in_delete1_mean)   
    array_in_jack = np.reshape(array_in_jack, (num_config, T) )
    return array_in_jack

# JACKKNIFE ERROR
def jackknife_err(array_in_mean, array_in_jack):
    num_config = np.size(array_in_jack, 0)
    # error 
    diff  =  array_in_jack - array_in_mean
    diff2 = diff**2
    err2  = ( num_config -1 ) * np.mean(diff2,0)
    err   = err2**0.5
    return err

# JACKNIFE COMPLETE
def jackknife(array_in, *args):
    array_in_mean = np.mean(array_in, 0)
    array_in_jack = jackknife_bin(array_in)
    err = jackknife_err(array_in_mean, array_in_jack)  
    return array_in_mean, err, array_in_jack


# COVARIANCE FOR JACKKNIFE ARRAY
def cov_jackknife(array_in_jack_x, array_in_jack_y):
    """ Covariance of two variables (i.e. correlators) evaluated at the same time slice.
    For example, given two correlators C1 and C2, the output of cov_jackknife is
    cov[t] = cov(C1[t], C2[t]).
    """
    dim_x = array_in_jack_x.ndim
    dim_y = array_in_jack_y.ndim
    if not dim_x == dim_y and not dim_x ==2:
        sys.exit("Closing program. Wrong array dimension in the usage of " + "'cov_jackknife()'" )
    N = array_in_jack_x.shape[0]
    t = array_in_jack_x.shape[1]
    f = (N-1)**2 / N
    cov = np.array([])
    for i in range(t):
        array_in_jack_x_T = np.transpose(array_in_jack_x)
        array_in_jack_y_T = np.transpose(array_in_jack_y)
        cov_aux = f * np.cov(array_in_jack_x_T[i], array_in_jack_y_T[i])[0][1]
        cov     = np.append(cov, cov_aux)
    return cov

# this works with data_stats object
def cov_jackknife_full_v1(data_x_in, data_y_in,  num_bins=None, rangefit=None, thin=1):
    assert(isinstance(data_x_in, dt.data_stats))
    
    T        = data_x_in.T()
    num_bins = data_x_in.num_bins() if num_bins is None else num_bins

    xmin, xmax = (0, T) if rangefit is None else (rangefit[0], rangefit[1])
    num_points = xmax-xmin
    
    mean_x_cut = data_x_in.mean()[xmin:xmax:thin]
    mean_y_cut = data_y_in.mean()[xmin:xmax:thin]
    
    Cov = np.array([])
    for j in range(num_bins):
        bins_x_cut_aux = data_x_in.bins()[j][xmin:xmax:thin]
        bins_y_cut_aux = data_y_in.bins()[j][xmin:xmax:thin]
        
        # Covariance (already applying cuts)
        vec_x     = bins_x_cut_aux - mean_x_cut
        vec_y     = bins_y_cut_aux - mean_y_cut
        Cov_aux = np.outer(vec_x, vec_y)
        Cov     = np.append(Cov, Cov_aux)
   
    Cov = np.reshape(Cov, (num_bins, num_points, num_points))
    Cov = (num_bins -1 ) * np.mean(Cov, 0)
    return Cov

def cov_jackknife_full(data_x_in, data_y_in,  num_bins=None, rangefit=None, thin=1):
    
    cov_xx = cov_jackknife_full_v1(data_x_in, data_x_in,  num_bins=num_bins, rangefit=rangefit, thin=thin)
    cov_xy = cov_jackknife_full_v1(data_x_in, data_y_in,  num_bins=num_bins, rangefit=rangefit, thin=thin)
    cov_yx = cov_jackknife_full_v1(data_y_in, data_x_in,  num_bins=num_bins, rangefit=rangefit, thin=thin)
    cov_yy = cov_jackknife_full_v1(data_y_in, data_y_in,  num_bins=num_bins, rangefit=rangefit, thin=thin)

    cov_1 = np.append(cov_xx, cov_xy, axis=1)
    cov_2 = np.append(cov_yx, cov_yy, axis=1)
    cov   = np.append(cov_1, cov_2, axis=0)

    return cov

def cov_jackknife_general(data_array_in, num_bins=None, rangefit=None, thin=1):
    
    n = len(data_array_in) * len(data_array_in[0].mean())

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

# CORRELATIONS     
def corr_jackknife(array_in_jack_x, array_in_jack_y):
    mean_x = np.mean(array_in_jack_x, 0) 
    mean_y = np.mean(array_in_jack_y, 0) 
    err_x  = jackknife_err(mean_x, array_in_jack_x)
    err_y  = jackknife_err(mean_y, array_in_jack_y)
    cov = cov_jackknife(array_in_jack_x, array_in_jack_y)
    return cov / (err_x * err_y )



# =============================================================================
# to be checked
# =============================================================================

def bootstrap_bin(array_in, boot_par):
    num_config = np.size(array_in, 0)
    # default_boot_par = (num_config, None)
    # num_samples, seed = tuple(map(lambda x, y: x if y is None else y, default_boot_par, boot_par))
    num_samples, seed = boot_par
    # to match random numbers in Ryan's code
    
    np.random.seed(seed)  
    bins          = np.random.randint(0, num_config, size=(num_config, num_samples)).transpose()
    array_in_boot = np.mean(array_in[bins], axis=1)
    return array_in_boot

def bootstrap_err(array_in_mean, array_in_boot):
    # error 
    diff  = array_in_boot - array_in_mean
    diff2 = diff**2
    err2  = np.mean(diff2,0)
    err   = err2**0.5
    return err

def bootstrap(array_in, boot_par):
    array_in_mean = np.mean(array_in, 0)
    array_in_boot = bootstrap_bin(array_in, boot_par)
    array_in_err  = bootstrap_err(array_in_mean, array_in_boot)
    return array_in_mean, array_in_err, array_in_boot

def cov_bootstrap(array_in_boot_x, array_in_boot_y):
    dim_x = array_in_boot_x.ndim
    dim_y = array_in_boot_y.ndim
    if not dim_x == dim_y and not dim_x ==2:
        sys.exit("Closing program. Wrong array dimension in the usage of " + "'cov_bootstrap()'" )
    N = array_in_boot_x.shape[0]
    t = array_in_boot_x.shape[1]
    f = (N-1) / N
    cov = np.array([])
    for i in range(t):
        array_in_boot_x_T = np.transpose(array_in_boot_x)
        array_in_boot_y_T = np.transpose(array_in_boot_y)
        cov_aux = f * np.cov(array_in_boot_x_T[i], array_in_boot_y_T[i])[0][1]
        cov     = np.append(cov, cov_aux)
    return cov


# =============================================================================
# General bins-error
# =============================================================================

#def stats_bin(array_in, boot_par):

def bins_err(array_in_mean, array_in_bins, stats_type):
    if   stats_type=='jack': err = jackknife_err(array_in_mean, array_in_bins)
    elif stats_type=='boot': err = bootstrap_err(array_in_mean, array_in_bins)
    return err

def cov_matrix(array_in_bins_x, array_in_bins_y, stats_type):
    if   stats_type=='jack': cov = cov_jackknife(array_in_bins_x, array_in_bins_y)
    elif stats_type=='boot': cov = cov_bootstrap(array_in_bins_x, array_in_bins_y)
    return cov

     
def cov_inv_tmp(data_in, num_config, rangefit, thin=1):
    xmin = rangefit[0]
    xmax = rangefit[1]+1
    num_points = xmax-xmin
    
    mean_cut = data_in.mean()[xmin:xmax:thin]
    num_points = len(mean_cut)
    
    bins_cut = np.array([])
    Cov = np.array([])
    for j in range(num_config):
        bins_cut_aux = data_in.bins()[j][xmin:xmax:thin]
        
        # Covariance (already applying cuts)
        vec     = bins_cut_aux - mean_cut
        Cov_aux = np.outer( vec, vec )
        Cov     = np.append(Cov, Cov_aux)
        
        bins_cut     = np.append(bins_cut, bins_cut_aux)      
        
    bins_cut = np.reshape(bins_cut, (num_config, num_points))
    
    Cov = np.reshape(Cov, (num_config, num_points, num_points))
    Cov = (num_config-1) * np.mean(Cov, 0)
    return Cov

# # PSEUDOINVERSE WITH SINGLE VALUE DECOMPOSITION
# def pseudoinv(M):
#     _U, _s, _VT = scipy.linalg.svd(M, full_matrices=False)
#     _threshold = np.finfo(float).eps * max(M.shape) * _s[0]
#     #_s = _s[_s > _threshold]
#     _VT = _VT[:_s.size]
#     cov = np.dot(_U / _s, _VT)
    
#     return cov