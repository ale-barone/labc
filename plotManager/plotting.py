
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy as sci
import scipy.optimize as opt
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.linalg import cholesky

import LatticeABC.dataManager as dM

# =============================================================================
# BASIC PLOT
# =============================================================================

FONTSIZE = {'S' : 16, 'M': 18, 'B': 20, 'BB': 22}
FIGSIZE  = (10,7)
LABELS   = {'x' : 't', 'y' : 'y', 'title' : 'Title'}

# INITIALIZATION FOR PLOT 
def init_plot(*args, **kwargs):
    # see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html
    # see https://matplotlib.org/stable/api/matplotlib_configuration_api.html
    
    plt.rc('figure', titlesize=FONTSIZE['BB'], figsize = FIGSIZE)  # fontsize of the figure    
    plt.rc('axes',   titlesize=FONTSIZE['BB'], labelsize = FONTSIZE['B'])    # fontsize of the axes (created with subplots)
    plt.rc('legend', fontsize=FONTSIZE['S'])    # legend fontsize

    # plt.rc('text', usetex=True)
    # plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    
    #TO DO: find a way to initialize plt.tight_layout() that holds for every plot
        
    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('xtick', labelsize=FONTSIZE['S'])    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONTSIZE['S'])    # fontsize of the tick labels



    
# INITIALIZATION FOR AXES
def init_axes(nrows=1, ncols=1, figsize=FIGSIZE, *args, **kwargs):
    init_plot()
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
    if issubclass(type(y), dM.dataStats):
        ycut = y.mean[xmin:xmax]
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
    
    if yerr is None and issubclass(type(y), dM.dataStats):
        ycut = y.mean[xmin:xmax]
        yerr = y.err[xmin:xmax]
    else:
        ycut = y[xmin:xmax]
        yerr = yerr[xmin:xmax]
    
    
    axes.errorbar(xcut, ycut, yerr = yerr, *args, **kwargs, markersize='10', capsize = 2, elinewidth=0.9)
    axes.set_title(LABELS['title'])#fontsize['title'])
    axes.set_xlabel(LABELS['x'])
    axes.set_ylabel(LABELS['y'])


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
    
    mean_cut = data_in.mean[xmin:xmax:thin]
    num_points = len(mean_cut)
    
    bins_cut = np.array([])
    Cov = np.array([])
    for j in range(num_config):
        bins_cut_aux = data_in.bins[j][xmin:xmax:thin]
        
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

# def cholesky(cov):
#     cov_inv = np.linalg.inv(cov)
#     cov_inv_sqrt = cholesky(cov_inv)
#     return cov_inv_sqrt

# fit function
def fit(x, data_in, fit_function, func_args=None, rangefit=None, thin=1, guess = [1, 1], correlated=False):
    num_bins = data_in.num_bins()
    
    xmin = rangefit[0]
    xmax = rangefit[1]

    statsType = data_in.statsType
    errFun = data_in.errFun
    
    xcut    = x[xmin:xmax:thin]
    mean_cut = data_in.mean[xmin:xmax:thin] #array_in_mean, cutf, cutb, thin)
    bins_cut = np.apply_along_axis(lambda x: x[xmin:xmax:thin], 1, data_in.bins)
    num_points = len(xcut)
    
    if correlated == True:
        cov = statsType.cov(data_in, num_bins=num_bins, rangefit=rangefit, thin=1)
        Cov_inv_sqrt = cholesky(cov)
    elif correlated == False:
        Cov_inv_sqrt = np.diag( 1 / data_in.err[xmin:xmax:thin] )

                
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
    for k in range(num_bins):
        sol    = leastsq( res, guess, args = (bins_cut[k]), maxfev=2000, ftol=1e-10, xtol=1e-10, full_output=True ) 
        fit_bins = np.append(fit_bins, sol[0])  
        

    fit_bins  = np.transpose(np.reshape(fit_bins, (num_bins, len(fit_mean))))
    
    fit_err = np.array([])
    for p in range(num_param):
        fit_err_aux = errFun(fit_mean[p], fit_bins[p]) 
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
        mean_p = np.array([fit_mean[p]])
        err_p = np.array([fit_err[p]])
        bins_p = np.reshape(fit_bins[p], (len(fit_bins[p]), 1) )
        out = np.array([mean_p, err_p])
        out = np.concatenate([out, bins_p], axis=0)
        return dM.dataStats(out, data_in.statsType)
    else:
        for p in range(num_param):
            mean_p = np.array([fit_mean[p]])
            err_p = np.array([fit_err[p]])
            bins_p = np.reshape(fit_bins[p], (len(fit_bins[p]), 1) )
            out_aux = np.array([mean_p, err_p])
            out_aux = np.concatenate([out_aux, bins_p], axis=0)
            out_aux = dM.dataStats(out_aux, data_in.statsType)
            out.append(out_aux)     
    
    return out
