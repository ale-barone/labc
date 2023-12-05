import numpy as np
import scipy as sci
import scipy.optimize as opt
from scipy.optimize import least_squares, leastsq
from scipy.linalg import cholesky
import pandas as pd
from .. import data as dM
from  ..data import constant as dMconstant
from  ..data import zeros as dMzeros
from ..data import DataStats, dataStats_args
from ..data import merge as dm_merge
from . import functions as lib
from .functions import const, exp, cosh, pole
from  .. import plot as plt

# =============================================================================
# utilities for fit
# =============================================================================

# P-VALUE (from Lattice_Analysis)
def p_value(k, x):
    return sci.special.gammaincc(k / 2., x / 2.)

# def chi_sq(fit):
#     """
#      takes result of leastsq fit as input and returns
#          p-value,
#          chi^2/Ndof
#          Ndof
#     """
#     Ndof = len(fit[2]['fvec']) - len(fit[0])
#     chisq = sum(fit[2]['fvec'] ** 2.0)
#     pv = p_value(Ndof, chisq)
#     return pv, chisq / Ndof, Ndof

def chi_sq(fit):
    """
     takes result of least_squares fit as input and returns
         p-value,
         chi^2/Ndof
         Ndof
    """
    # FIXME don't count the priors if zero!!
    Ndof = len(fit.fun) - len(fit.x)
    chisq = sum(fit.fun**2.0)
    pv = p_value(Ndof, chisq)
    return pv, chisq / Ndof, Ndof

################################################################################
# simple fitter
################################################################################

# FIXME
def fit(x, data_in, fit_function, *func_args, rangefit=None, thin=1, guess=[1, 1], correlated=False):
    num_bins = data_in.num_bins()
    
    xmin = rangefit[0]
    xmax = rangefit[1]

    statsType = data_in.statsType
    err_func = data_in.statsType.err_func
    
    xcut = x[xmin:xmax:thin]
    mean_cut = data_in.mean[xmin:xmax:thin] #array_in_mean, cutf, cutb, thin)
    bins_cut = np.apply_along_axis(lambda x: x[xmin:xmax:thin], 1, data_in.bins)
    num_points = len(xcut)
    
    if correlated == True:
        cov = statsType.cov(data_in, num_bins=num_bins, rangefit=rangefit, thin=thin)
        cov_inv_sqrt = cholesky(np.linalg.inv(cov))
    elif correlated == False:
        cov_inv_sqrt = np.diag( 1 / data_in.err[xmin:xmax:thin] )

                
    def res(param, data):   
        func = fit_function(param, xcut, *func_args)
        return np.dot(cov_inv_sqrt, func - data) 

    sol    = leastsq( res, guess, args = (mean_cut),  maxfev=2000, ftol=1e-10, xtol=1e-10, full_output=True)
    chisq  = chi_sq(sol)

    fit_mean = sol[0]
    num_param = len(sol[0])
        
    # bins
    fit_bins = np.empty(shape=(num_bins, len(fit_mean)))
    for k in range(num_bins):
        sol    = leastsq( res, guess, args = (bins_cut[k]), maxfev=2000, ftol=1e-10, xtol=1e-10, full_output=True ) 
        fit_bins[k] = sol[0]
        

    fit_bins = np.transpose(np.reshape(fit_bins, (num_bins, len(fit_mean))))
    
    fit_err = np.array([])
    for p in range(num_param):
        fit_err_aux = err_func(fit_mean[p], fit_bins[p]) 
        fit_err = np.append(fit_err, fit_err_aux)
        


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

################################################################################
# Fit results
################################################################################

# Could be nice to build such a class that takes care of collecting
# the results (including cov, priors,...)
# and do the relevant stuff (chisq, pval...) and pretty print

#class FitResult:


################################################################################
# Class Fitter
################################################################################

class Fitter:

    def __init__(self, x, y, fit_func=None,
                 *fit_func_args, **fit_func_kwargs):
        self.x = x
        if isinstance(y, list):
            self.y = y
            self.statsType = y[0].statsType
            self.num_bins = y[0].num_bins()
        else: 
            self.y = y
            self.statsType = y.statsType
            self.num_bins = y.num_bins()

        if fit_func is not None:
            libfunc = getattr(lib, fit_func)
            self.func_param = libfunc.PARAM
            self.funcstr = libfunc.STRING
            def func(param, x):
                return libfunc()(param, x, *fit_func_args, **fit_func_kwargs)
            self._fit_func = func
        else:
            self._fit_func = None
        

        # # initialize null priors
        # def null_prior(*args):
        #     return 0
        # null_prior_data = 0*y #dMzeros(1, self.statsType)
        
        # self.prior = {v: null_prior for v in self.param.values()}
        # self.prior_resampling = None
        # self.prior_data = {v: null_prior_data[vi] for vi, v in enumerate(self.param.values())}

        self.prior = None

        # print('\n_________________________________________')
        # print('| --- Initialise fitter for function ')
        # print('|  ' + self.funcstr)
        # print('|')
        # print('_______________________________\n')
    
    @property
    def fit_func(self):
        return self._fit_func
    
    @fit_func.setter
    def fit_func(self, func):
        self._fit_func = func

    def set_prior(self, param, mu, sigma, resampling=True):
        # assert(param in self.param.values())
        self.prior = {}
        self.prior_data = {}
        self.prior_resampling = resampling
        if resampling==True:
            bins = np.random.normal(mu, sigma, size=self.num_bins)
            self.prior_data[param] = DataStats(mu, bins, self.statsType)
        else:
            self.prior_data[param] = dMconstant(mu, self.statsType)
        
        def out(p, x):
            return (x-p)/sigma
        self.prior[param] = out

        # def prior_func(p, *args):
        #     for k, v in self.param.items():
        #         if v in param:
        #             prior = prior[v](param[k])
        #             out = np.append(out, prior)   
        # self.prior_func = prior_func
    
    def _cov(self, data, correlated):
        if correlated==True:
            cov = self.statsType.cov(data)
        elif correlated==False:
            cov = np.diag(data.err**2)
        return cov

    def _cov_inv(self, data, correlated):
        cov = self._cov(data, correlated)
        cov_inv = np.linalg.inv(cov)
        return cov_inv
        
    def _cov_inv_sqrt(self, data, correlated):
        cov_inv = self._cov_inv(data, correlated)
        cov_inv_sqrt = cholesky(cov_inv)
        return cov_inv_sqrt
         
    def _residual(self, x, y, cov_inv_sqrt):
        def func(param):
            fit_func = self.fit_func(param, x)
            out = np.dot(cov_inv_sqrt, fit_func-y)
            if self.prior is not None:
                prior_func = np.array([])
                for idx, pr in enumerate(self.prior_data):
                    prior_func = self.prior[self.param[idx]]
                    print(prior_func)
                    out = np.append(out, prior_func(param[idx], pr))
            return out
            
        return func

    def _parse_guess(self, guess_dict):
        # FIXME put here the collection of array param (or dict param)
        assert(set(guess_dict.keys())==set(self.func_param.values()))
        param = {}

        param_keys = list(self.func_param.values())
        param_values = list(guess_dict.values())

        out = []
        counter = 0
        for p in param_keys:
            pv = guess_dict[p]
            if isinstance(pv, dict):
                for idxpk, pk in enumerate(pv.keys()):
                    param[counter] = f'{p}{pk}'
                    out.append(list(pv.values())[idxpk])
                    counter += 1
            else:
                param[counter] = f'{p}'
                out.append(param_values[counter])
                counter += 1

        self.param = param
        self.num_param = len(param)
        self.guess = out

        return out
    
    def _collect_fit_param(self, fitted_param):
        out = {self.param[i]: fitted_param[i] for i in range(len(self.param))}
        return out
    
    def _collect_fit_param_str(self, fitted_param):
        out = {self.param[i]: str(fitted_param[i]) for i in range(len(self.param))}
        return out
    
    def _collect_fit_quality(self, fit):
        pv, chisq_Ndof, Ndof = chi_sq(fit)
        out = {
            'pvalue': f'{pv:.3f}',
            'chisq/Ndof': f'{chisq_Ndof:.3f}',
            'Ndof': Ndof, 
        }
        return out

    def _collect_prior_data(self):
        prior_data = []
        for k, v in self.param.items():
            prior_data.append(self.prior_data[v])
        prior_data = [self.prior_data[i] for i in range(self.num_param)]
        return dM.merge(prior_data)

    def _collect_output(self, fit, fitted_param):
        quality = self._collect_fit_quality(fit)
        param = self._collect_fit_param(fitted_param)
        out = {**quality, **param}
        return out

    # def _collect_fit_results(self, fit):
    #     sol = {self.param[i]: f'{fit[0][i]:.4f}' for i in range(len(fit[0]))}
    #     pv, chisq_Ndof, Ndof = chi_sq(fit)
    #     summary = {
    #         'pvalue': f'{pv:.3f}',
    #         'chisq/Ndof': f'{chisq_Ndof:.3f}',
    #         'Ndof': Ndof, 
    #         'fit_param': sol
    #     }
    #     return summary, sol

    def _fitter(self, x, y, guess, cov_inv_sqrt):
        res = self._residual(x, y, cov_inv_sqrt)
        #sol = leastsq(res, guess, maxfev=2000, ftol=1e-10, xtol=1e-10, full_output=True)
        sol = least_squares(res, guess)
        return sol

    @dataStats_args
    def _eval(self, x, y, guess, cov_inv_sqrt): # prior data must be a list of DataStats
        sol = self._fitter(x, y, guess, cov_inv_sqrt)
        return sol.x

    
    def eval(self, fit_points, guess, correlated):
        x = self.x[fit_points]
        y = self.y[fit_points]
        self.fit_points = x, y

        guess = self._parse_guess(guess)
        cov_inv_sqrt = self._cov_inv_sqrt(y, correlated)
        #prior = self._collect_prior_data()

        # deal with prior_data
        fit = self._fitter(x, y.mean, guess, cov_inv_sqrt)
        sol = self._eval(x, y, guess, cov_inv_sqrt)
        out = self._collect_output(fit, sol)
        return out
    
    # this must just be a wrapper around least squares
    def eval_mean(self, fit_range, guess, correlated):
        x = self.x[fit_range]
        y = self.y[fit_range]
        guess = self._parse_guess(guess)
        cov_inv_sqrt = self._cov_inv_sqrt(y, correlated)

        prior = self._collect_prior_data()
        res = self._residual(x, y.mean, cov_inv_sqrt, prior.mean)
        
        sol = leastsq(res, guess, maxfev=2000, ftol=1e-10, xtol=1e-10, full_output=True)
        pv, chisq_Ndof, Ndof = chi_sq(sol)
        return sol

    # this must scan ranges and provide a report (pandas dataframe)
    def _scan(self, fit_range: list, guess: list, *, correlated, min_num_points=None, max_num_points=None, thin=1):
        if min_num_points is None:
            min_num_points = len(self.param)+1
        elif min_num_points<=len(self.param):
            param = list(self.param.values())
            raise ValueError(
                "'min_num_points' must be > than number of parameters, "
                f"here {len(self.param)}, i.e. {param} "
            )
        
        lower, upper = fit_range
        tot_lenght = upper - lower

        if max_num_points is None:
            max_num_points = tot_lenght
        elif max_num_points>tot_lenght:
            raise ValueError(
                "'max_num_points' must be < than the lenght of the fit range, "
                f"here {tot_lenght}"
            )

        x = self.x[lower:upper]
        y = self.y[lower:upper]
        guess = self._parse_guess(guess)
        #cov_inv_sqrt = self._cov_inv_sqrt(y, correlated)

        cov = self._cov(y, correlated)

        fit_points = []
        fit_quality = []
        for l in range(tot_lenght):
            for u in range(min_num_points+l, l+max_num_points+1):
                x_cut = x[l:u:thin]
                if len(x_cut)<min_num_points:
                    continue
                y_cut = y[l:u:thin]
                cov_inv_sqrt_cut = cholesky(np.linalg.inv(cov[l:u:thin,l:u:thin]))

                #fit = self._eval_mean(x_cut, y_cut.mean, guess, cov_inv_sqrt_cut)
                fit = self._fitter(x_cut, y_cut.mean, guess, cov_inv_sqrt_cut)
                param = self._eval(x_cut, y_cut.mean, guess, cov_inv_sqrt_cut)
                out = {**self._collect_fit_quality(fit), **self._collect_fit_param(param)}
                
                # collect
                fit_points.append(str(slice(lower+l, lower+u, thin)))
                fit_quality.append(out)

        data = pd.DataFrame(fit_quality, index=fit_points)
        data = data.sort_values('pvalue', ascending=False)
        return data
    
    def full_scan(self, fit_range: list, guess: dict, *, correlated,
                  min_num_points=None, max_num_points=None, thin=1):
        if min_num_points is None:
            min_num_points = len(self.param)+1
        elif min_num_points<=len(self.param):
            param = list(self.param.values())
            raise ValueError(
                "'min_num_points' must be > than number of parameters, "
                f"here {len(self.param)}, i.e. {param} "
            )
        
        lower, upper = fit_range
        tot_lenght = upper - lower

        if max_num_points is None:
            max_num_points = tot_lenght
        elif max_num_points>tot_lenght:
            raise ValueError(
                "'max_num_points' must be < than the lenght of the fit range, "
                f"here {tot_lenght}"
            )

        x = self.x[lower:upper]
        y = self.y[lower:upper]
        guess = self._parse_guess(guess)
        #cov_inv_sqrt = self._cov_inv_sqrt(y, correlated)

        cov = self._cov(y, correlated)

        fit_points = []
        fit_quality = []
        print(x)
        for l in range(tot_lenght-min_num_points+1):
            u_min = l+min_num_points
            u_max = l+max_num_points+1
            if u_max>tot_lenght:
                u_max = tot_lenght+1
            for u in range(u_min, u_max):
                x_cut = x[l:u:thin]
                if len(x_cut)<min_num_points:
                    continue
                y_cut = y[l:u:thin]
                cov_inv_sqrt_cut = cholesky(np.linalg.inv(cov[l:u:thin,l:u:thin]))

                #fit = self._eval_mean(x_cut, y_cut.mean, guess, cov_inv_sqrt_cut)
                prior = self._collect_prior_data()
                fit = self._fitter(x_cut, y_cut.mean, guess, cov_inv_sqrt_cut, prior.mean)
                quality = self._collect_fit_quality(fit)
                pv = float(quality['pvalue'])
                # if pv<0.05 or pv>0.95:
                #     continue
                param = self._eval(x_cut, y_cut, guess, cov_inv_sqrt_cut, prior)
                out = {**quality, **self._collect_fit_param_str(param)}
                
                # collect
                fit_points.append(str(slice(lower+l, lower+u, thin)))
                fit_quality.append(out)

        data = pd.DataFrame(fit_quality, index=fit_points)
        data = data.sort_values('pvalue', ascending=False)
        return data


    # combine the info from scan and tell me the best option
    def _optimize(self):
        pass

    # combine _scan and _optimize
    def finder(self):
        pass

    # take output from leastsq and print results nicely
    def print(self):
        pass
