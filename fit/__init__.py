import numpy as np
import scipy as sci
import scipy.optimize as opt
from scipy.optimize import leastsq
from scipy.linalg import cholesky
from ..data  import DataStats


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
    err_func = data_in.statsType.err_func
    
    xcut    = x[xmin:xmax:thin]
    mean_cut = data_in.mean[xmin:xmax:thin] #array_in_mean, cutf, cutb, thin)
    bins_cut = np.apply_along_axis(lambda x: x[xmin:xmax:thin], 1, data_in.bins)
    num_points = len(xcut)
    
    if correlated == True:
        cov = statsType.cov(data_in, num_bins=num_bins, rangefit=rangefit, thin=1)
        Cov_inv_sqrt = cholesky(np.linalg.inv(cov))
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
    fit_bins = np.empty(shape=(num_bins, len(fit_mean)))
    for k in range(num_bins):
        sol    = leastsq( res, guess, args = (bins_cut[k]), maxfev=2000, ftol=1e-10, xtol=1e-10, full_output=True ) 
        fit_bins[k] = sol[0]
        

    fit_bins  = np.transpose(np.reshape(fit_bins, (num_bins, len(fit_mean))))
    
    fit_err = np.array([])
    for p in range(num_param):
        fit_err_aux = err_func(fit_mean[p], fit_bins[p]) 
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
        #err_p = np.array([fit_err[p]])
        bins_p = np.reshape(fit_bins[p], (len(fit_bins[p]), 1) )
        # out = np.array([mean_p, err_p])
        # out = np.concatenate([out, bins_p], axis=0)
        out = DataStats(mean_p, bins_p, data_in.statsType)
    else:
        for p in range(num_param):
            mean_p = np.array([fit_mean[p]])
            #err_p = np.array([fit_err[p]])
            bins_p = np.reshape(fit_bins[p], (len(fit_bins[p]), 1) )

            out_aux = DataStats(mean_p, bins_p, data_in.statsType)
            out.append(out_aux)     
    
    return out


def fit_cosh(param, t, T):
    Thalf = T/2
    return 2*param['A'] * np.exp(-param['E']*Thalf) * np.cosh(param['E']*( Thalf - t)) 


class Fitter:

    def __init__(self, range_fit, data_in, correlated=True):
        self.range_fit = range_fit
        self.data_in = data_in[range_fit]
        self.correlated = correlated

        self.statsType = data_in.statsType
        self.cov_inv_sqrt = self._cov_inv_sqrt()
    
    def _cov_inv_sqrt(self):
        if self.correlated==True:
            cov = self.statsType.cov(self.data_in)
            out = cholesky(np.linalg.inv(cov))
        elif self.correlated==False:
            out = np.diag(1/self.data_in.err)
        return out
    
    # def make_fit_func(self, func, *args, **kwargs):
    #     def internal_func(param):
    #         return func(param, *args, **kwargs)

    def assign_fit_func(self, ID, *args):
        if ID=='cosh':
            def func(param):
                return cosh(param, self.x, 64)
            self.fit_func = func
        
    def _residual(self):
        def func(param):
            func = self.fit_func(param)
            return np.dot(self.cov_inv_sqrt, func - self.data_in.mean)   
        return func

    def eval_mean(self, guess):  
        res = self._residual()
        sol = leastsq(res, guess, maxfev=2000, ftol=1e-10, xtol=1e-10, full_output=True)
        chisq = chi_sq(sol)
        return sol
    





# def fit(x, data_in, fit_function, func_args=None, rangefit=None, thin=1, guess = [1, 1], correlated=False):
#     num_bins = data_in.num_bins()
    
#     xmin = rangefit[0]
#     xmax = rangefit[1]

#     statsType = data_in.statsType
#     err_func = data_in.statsType.err_func
    
#     xcut    = x[xmin:xmax:thin]
#     mean_cut = data_in.mean[xmin:xmax:thin] #array_in_mean, cutf, cutb, thin)
#     bins_cut = np.apply_along_axis(lambda x: x[xmin:xmax:thin], 1, data_in.bins)
#     num_points = len(xcut)
    
#     if correlated == True:
#         cov = statsType.cov(data_in, num_bins=num_bins, rangefit=rangefit, thin=1)
#         Cov_inv_sqrt = cholesky(cov)
#     elif correlated == False:
#         Cov_inv_sqrt = np.diag( 1 / data_in.err[xmin:xmax:thin] )

                
#     def res(param, data):
#         if func_args == None:
#             func = fit_function(param, xcut)
#             return np.dot( Cov_inv_sqrt, func - data)            
#         else:
#             func = fit_function(param, xcut, func_args)
#             return np.dot( Cov_inv_sqrt, func - data)    

#     sol    = leastsq( res, guess, args = (mean_cut),  maxfev=2000, ftol=1e-10, xtol=1e-10, full_output=True)
#     chisq  = chi_sq(sol)

#     fit_mean = sol[0]
#     num_param = len(sol[0])
        
#     # bins
#     fit_bins = np.empty(shape=(num_bins, len(fit_mean)))
#     for k in range(num_bins):
#         sol    = leastsq( res, guess, args = (bins_cut[k]), maxfev=2000, ftol=1e-10, xtol=1e-10, full_output=True ) 
#         fit_bins[k] = sol[0]
        

#     fit_bins  = np.transpose(np.reshape(fit_bins, (num_bins, len(fit_mean))))
    
#     fit_err = np.array([])
#     for p in range(num_param):
#         fit_err_aux = err_func(fit_mean[p], fit_bins[p]) 
#         fit_err     = np.append(fit_err, fit_err_aux)
        


#     print("\n################ FIT RESULTS #######################")
#     print("# Correlated =", correlated)
#     print("# IndexRange = [" + str(xmin) + ", " + str(rangefit[1]) + "]")
#     print("# Range      = [" + str(xcut[0])+ ", " + str(xcut[-1]) + "]")
#     print("# Thinning   = " + str(thin))
#     for p in range(num_param):
#         print('# param_' + str(p) + ' = ', fit_mean[p], "  err =", fit_err[p] )
#     print('# CHISQ   = ', chisq[1], "  pvalue = ", chisq[0] )
#     print("####################################################\n")
    

#     out = []
#     if num_param==1:        
#         mean_p = np.array([fit_mean[p]])
#         #err_p = np.array([fit_err[p]])
#         bins_p = np.reshape(fit_bins[p], (len(fit_bins[p]), 1) )
#         # out = np.array([mean_p, err_p])
#         # out = np.concatenate([out, bins_p], axis=0)
#         out = dM.dataStats(mean_p, bins_p, data_in.statsType)
#     else:
#         for p in range(num_param):
#             mean_p = np.array([fit_mean[p]])
#             #err_p = np.array([fit_err[p]])
#             bins_p = np.reshape(fit_bins[p], (len(fit_bins[p]), 1) )

#             out_aux = DataStats(mean_p, bins_p, data_in.statsType)
#             out.append(out_aux)     
    
#     return out
