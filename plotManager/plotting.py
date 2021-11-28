#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:03:53 2021

@author: abarone
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy as sci
import scipy.optimize as opt
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.linalg import cholesky

import gvar as gv
import lsqfit

from LatticeAB import stats
from LatticeAB.data import dataHandling as dt

# =============================================================================
# BASIC PLOT
# =============================================================================

FONTSIZE = {'S' : 14, 'M': 14, 'B': 16, 'BB': 20}
FIGSIZE  = (10,7)
LABELS   = {'x' : 't', 'y' : 'y', 'title' : 'Title'}

# INITIALIZATION FOR PLOT 
def plot_init(*args, **kwargs):
    # see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html
    
    plt.rc('figure', titlesize=FONTSIZE['BB'], figsize = FIGSIZE)  # fontsize of the figure    
    plt.rc('axes',   titlesize=FONTSIZE['B'], labelsize = FONTSIZE['M'])    # fontsize of the axes (created with subplots)
    plt.rc('legend', fontsize=FONTSIZE['S'])    # legend fontsize
    
    #TO DO: find a way to initialize plt.tight_layout() that holds for every plot
        
    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('xtick', labelsize=45)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

    
# INITIALIZATION FOR AXES
def axes_init(nrows=1, ncols=1, figsize=FIGSIZE, *args, **kwargs):
    plot_init()
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols , figsize[1]*nrows),  *args, **kwargs)
    fig.tight_layout(pad=5.0)
    # if (not nrows==1) or (not ncols==1):
    #     fig.suptitle('Title', y= 0)
    return fig, ax

# IT WOULD BE NICE TO HAVE A TRACK OF THE AXIS I DEFINED, LIKE FOR EX axes_init
# THIS WOULD ALLOW ME TO APPLY COMMON CHANGES TO ALL OF THEM WITHOUT HAVING TO
# DO IT MANUALLY FOR EACH ONE...maybe set attributes of the object (should I define a class?)

# PLOT ON AXES  
def plot(axes, x, y, *args, xlim=None, **kwargs):
        
    if xlim is None:
        xmin = 0
        xmax = len(x)
    else:
        xmin = xlim[0]
        xmax = xlim[1]
    
    xcut = x[xmin:xmax]
    if issubclass(type(y), dt.data_stats):
        ycut = y.mean()[xmin:xmax]
    else:
        ycut = y[xmin:xmax]

    ret = axes.plot(xcut, ycut, *args, **kwargs)
    axes.set_title(LABELS['title'])#fontsize['title'])
    axes.set_xlabel(LABELS['x'])
    axes.set_ylabel(LABELS['y'])

    return ret

    
def errplot(axes, x, y, *args, xlim=None, yerr = None, **kwargs):
        
    if xlim is None:
        xmin = 0
        xmax = len(x)
    else:
        xmin = xlim[0]
        xmax = xlim[1]
    
    xcut = x[xmin:xmax]
    
    if yerr is None and issubclass(type(y), dt.data_stats):
        ycut = y.mean()[xmin:xmax]
        yerr = y.err()[xmin:xmax]
    else:
        ycut = y[xmin:xmax]
        yerr = yerr[xmin:xmax]
    
    
    axes.errorbar(xcut, ycut, yerr = yerr, *args, **kwargs, markersize='10', capsize = 2, elinewidth=0.9)
    axes.set_title(LABELS['title'])#fontsize['title'])
    axes.set_xlabel(LABELS['x'])
    axes.set_ylabel(LABELS['y'])
    
    
def print_axes(axb):
    fig = plt.figure(figsize=(2,2))
    ax=axb
    ax.figure=fig
    fig.axes.append(ax)
    fig.add_axes(ax)


    # dummy = fig.add_subplot(111)
    # ax.set_position(dummy.get_position())
    # dummy.remove()
    # plt.show()
    #return fig

# =============================================================================
# FIT
# =============================================================================

def const(param, t):
    return param[0]
    
def exp(param, t):
    return param[0] * np.exp(-param[1]*t)

def cosh(param, t, T):
    Thalf = T/2
    return ( 2 * param[0] ) * np.exp(- param[1] * Thalf ) * np.cosh( param[1] * ( Thalf - t) ) 
    #return param[0] * (np.exp(-param[1] * t) + np.exp(-param[1] * (T - t)))

def pole(param, x, M):
    return param[0] / (M-x)

# P-VALUE (from Lattice_Analysis)
def p_value(k, x):
    return sci.special.gammaincc(k / 2., x / 2.)

def chi_sq(res):
    """
     takes result of leastsq as input and returns
         p-value,
         chi^2/Ndof
         Ndof
    """
    Ndof = len(res[2]['fvec']) - len(res[0])
    chisq = sum(res[2]['fvec'] ** 2.0)
    pv = p_value(Ndof, chisq)
    return pv, chisq / Ndof, Ndof


# _TODO: CHECK FOR BOOTSTRAP! THIS SEEMS TO BE DEFINED ONLY FOR JACK!
def cov(data_in, num_config, rangefit, thin=1):
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
    Cov = (num_config -1 ) * np.mean(Cov, 0)
    return Cov

# build covariance matrix
def cov_inv_sqrt(data_in, num_config, rangefit, thin=1):
    Cov = cov(data_in, num_config, rangefit, thin=thin)
    
    Cov_inv = np.linalg.inv(Cov)
    Cov_inv_sqrt = cholesky(Cov_inv)
 
    return Cov_inv_sqrt

# fit function
def fit(x, data_in, fit_function, func_args=None, rangefit=None, thin=1, guess = [1, 1], correlated=False, savepath=None):
        num_config = data_in.bins().shape[0]
        xmin = rangefit[0]
        xmax = rangefit[1]+1
        num_points = xmax-xmin
        err_fun = data_in.err_fun
        
        xcut    = x[xmin:xmax:thin]
        mean_cut = data_in.mean()[xmin:xmax:thin] #array_in_mean, cutf, cutb, thin)
        bins_cut = np.apply_along_axis(lambda x: x[xmin:xmax:thin], 1, data_in.bins())
        num_points = len(xcut)
        
        if correlated == True:
            Cov_inv_sqrt = cov_inv_sqrt(data_in, num_config, rangefit, thin)
        elif correlated == False:
            Cov_inv_sqrt = np.diag( 1 / data_in.err()[xmin:xmax:thin] )
  
                    
        def res(param, data):
            if func_args == None:
                func = fit_function(param, xcut)
                return np.dot( Cov_inv_sqrt, func - data)            
            else:
                func = fit_function(param, xcut, func_args)
                return np.dot( Cov_inv_sqrt, func - data)    

             
        sol    = leastsq( res, guess, args = (mean_cut),  maxfev=2000, ftol=1e-10, xtol=1e-10, full_output=True)
        chisq  = chi_sq(sol)

        fit_mean = sol[0]
        num_param = len(sol[0])
         
        # bins
        fit_bins = np.array([])
        for k in range(num_config):
            sol    = leastsq( res, guess, args = (bins_cut[k]), maxfev=2000, ftol=1e-10, xtol=1e-10, full_output=True ) 
            fit_bins = np.append(fit_bins, sol[0])  
            

        fit_bins  = np.transpose(np.reshape(fit_bins, (num_config, len(fit_mean))))
        
        fit_err = np.array([])
        for p in range(num_param):
            fit_err_aux = err_fun(fit_mean[p], fit_bins[p]) 
            fit_err     = np.append(fit_err, fit_err_aux)
         


        print("\n################ FIT RESULTS #######################")
        print("# Correlated =", correlated)
        print("# IndexRange = [" + str(xmin) + ", " + str(rangefit[1]) + "]")
        print("# Range      = [" + str(xcut[0])+ ", " + str(xcut[-1]) + "]")
        print("# Thinning   = " + str(thin))
        for p in range(num_param):
            print('# param_' + str(p) + ' = ', fit_mean[p], "  err =", fit_err[p] )
        print('# CHISQ   = ', chisq[1], "  pvalue = ", chisq[0] )
        print("####################################################\n")
        

        out = []
        if num_param==1:
            out_aux = np.concatenate( (np.array([fit_mean[p]]), np.array([fit_err[p]]), fit_bins[p]))
            out = dt.data_stats(out_aux, data_in.stats_type)
        else:
            for p in range(num_param):
                out_aux = np.concatenate([np.array([fit_mean[p], fit_err[p]]), fit_bins[p]])
                out_aux = dt.data_stats(out_aux, data_in.stats_type)
                out.append(out_aux)     
            

        if not savepath is None:
            fit_param_a = np.array([])
            for fit_param in out:
                fit_param_a = np.append(fit_param_a, fit_param.data)
            fit_param_a = np.reshape(fit_param_a, (num_param, num_config+2))

            with open(savepath, 'w') as new: 
                np.savetxt(new, np.vstack([fit_param_a]))
        
           
        return out


# fit function
def fit_raw(n, data_in, fit_function, func_args):
        assert(n+1==len(data_in.mean()))
        num_bins = data_in.bins().shape[0]
        err_fun = data_in.err_fun
        guess=np.append(1,[0 for i in range(n)])

        # # if correlated == True:
        #Cov_inv_sqrt = cov_inv_sqrt(data_in, num_config, rangefit)
        # if correlated == False:
        #     Cov_inv_sqrt = np.diag( 1 / data_in.err())
  
        def sigmoid(param):
            l = 1
            out = 0
            for p in param:
                out+=  1/(1+np.exp(-p+l)) * 1/(1+np.exp(p-l))
            return out

        

        # def chisq_fun(param, data):
        #     out = fit_function(param, 0, func_args) - data[0]
        #     for d in range(1, n):
        #         def res(param, data):
        #             func = fit_function(param, d, func_args)  
        #             return (func - data[d])/data_in.err()[d] 
        #         out += res(param, data)**2 + sigmoid(param)
        #     return out

        ###### if correlated...tmp
        Cov_inv_sqrt = cov_inv_sqrt(data_in, num_bins, (1,n+1))
        def chisq_fun(param, data):
            #out = fit_function(param, 0, func_args) - data[0]
            out = np.array([])
            for d in range(1, n+1):
                def res(param, data):
                    func = fit_function(param, d, func_args)  
                    return (func - data[d])
                out = np.append(out, res(param, data) )#+ sigmoid(param))
            return np.sum(np.dot(Cov_inv_sqrt, out)**2)
        ########
        
        
        sol = minimize(chisq_fun, guess, args=data_in.mean(), tol=1e-6,
                        bounds=[(1,1), *[(-1,1) for i in range(n)]], method='Powell')
        fit_mean = sol.x 
        print(sol.x)

        fit_bins = np.apply_along_axis(lambda bin: minimize(chisq_fun, guess, args=bin, tol=1e-6, 
                                        bounds=[(1,1), *[(-1,1) for i in range(n)]], method='Powell').x, 1, data_in.bins()) 
        fit_err  = err_fun(fit_mean, fit_bins) 

        print("\n################ FIT RESULTS #######################")
        for p in range(n+1):
            print('# Tn_' + str(p) + ' = ', fit_mean[p], "  err =", fit_err[p] )
        print("####################################################\n")


        out = dt.data_stats(np.concatenate([np.array([fit_mean, fit_err]), fit_bins]))
        return out





# fit function using least_squares
def fit_squares(n, data_in, fit_function, func_args):
        num_bins = data_in.bins().shape[0]
        err_fun  = data_in.err_fun  
        guess = np.zeros(n+1)

        Cov_inv_sqrt = cov_inv_sqrt(data_in, num_bins, (1,n+1))
        def res_mean(param):
            func = fit_function(param, np.arange(1, n+1), func_args)
            return np.dot(Cov_inv_sqrt, func - data_in.mean()[1:])        

        sol_mean = least_squares(res_mean, guess, bounds=(-1,1), ftol=1e-10, xtol=1e-10, method='trf')
        fit_mean = sol_mean.x
        print(sol_mean)

        # fit_bins = np.array([])
        # for b in range(num_bins):
        #     def res_bins(param):
        #         func = fit_function(param, np.arange(1, n+1), func_args)
        #         return np.dot(Cov_inv_sqrt, func - data_in.bins(b)[1:]) 
        #     #print("Bins number:", b)
        #     sol_bins = least_squares(res_bins, guess, bounds=(-1,1), ftol=1e-6, xtol=1e-6, method='trf')
        #     fit_bins = np.append(fit_bins, sol_bins.x)
        #     print(sol_bins.x)

        # fit_bins = np.reshape(fit_bins, (num_bins, n+1))
            
        # fit_err = err_fun(fit_mean, fit_bins)
        # print("\n################ FIT RESULTS least_squares #######################")
        # for p in range(n+1):
        #     print('# Tn_' + str(p) + ' = ', fit_mean[p], "  err =", fit_err[p] )
        # print("###################################################################\n")

        # fit = dt.data_stats(np.concatenate([[fit_mean, fit_err], fit_bins]))

        # return fit

