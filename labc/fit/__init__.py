import numpy as np
import scipy as sci
import scipy.optimize as opt
from scipy.optimize import least_squares, leastsq, curve_fit
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
import h5py

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
    print('# CHISQ   = ', chisq[1], "  pval = ", chisq[0] )
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
# Fitter wrapper
################################################################################

# I need to create a class that picks the minimizer (e.g. least_squares,
# minimize, leastsq) and that contains all the standard operation for each case
# (e.g. collect the fit parameters, fit quality) and then pass this into my
# Fitter class

def _collect_fit_quality(fit):
    pv, chisq_Ndof, Ndof = chi_sq(fit)
    out = {
        'pval': np.round(pv, 3), #f'{pv:.3f}',
        'chisq/Ndof': np.round(chisq_Ndof, 3), #f'{chisq_Ndof:.3f}',
        'Ndof': Ndof, 
    }
    return out

################################################################################
# Fit results
################################################################################

# Could be nice to build such a class that takes care of collecting
# the results (including cov, priors,...)
# and do the relevant stuff (chisq, pval...) and pretty print

class FitResult:

    def __init__(self, fit_output, param, param_map, fitted_param):
        self.fit = fit_output
        self.param = param
        self.param_dict = param_map

        self.result_full = self._get_result_full(fitted_param)
        self.result = self._get_result(self.result_full)
        self.quality = _collect_fit_quality(fit_output)


    def _get_result_full(self, fitted_param):
        param_dict_flatten = self._flatten_param_dict(self.param_dict)
        out = {param_dict_flatten[i]: res 
               for i, res in enumerate(fitted_param)}
        return out
    
    def _flatten_param_dict(self, param_dict):
        param_list = []
        for k, v in param_dict.items():
            if isinstance(v, list):
                for l in v:
                    param_list.append(l)
            else:
                param_list.append(v)
        
        out = {i: par for i, par in enumerate(param_list)}
        return out

    def _get_result(self, full_result_dict):
        out = {}

        for param in self.param:
            param_list = [p for p in list(full_result_dict.keys()) if param in p]
            out[param] = dM.merge([full_result_dict[pl] for pl in param_list])
        return out
        # for param, param_list in self.param_dict.items():
        #     if isinstance(param_list, list):
        #         out[param] = dM.merge([full_result_dict[pl] for pl in param_list])
        #     else:
        #         out[param] = full_result_dict[param_list]
        # return out
    
    # TODO: make it more general
    def save(self, file, group, rename_par=None):
        if rename_par is None:
            rename_par = {p: p for p in self.param}
        with h5py.File(file, 'a') as hf:
            G = hf.create_group(group)
            for p, pr in rename_par.items():
                G.create_dataset(f'{pr}/mean', data=self.result[p].mean)
                G.create_dataset(f'{pr}/err', data=self.result[p].err)
                G.create_dataset(f'{pr}/bins', data=self.result[p].bins)

            G.create_dataset('pval', data=self.quality['pval'])
            G.create_dataset('chisq/Ndof', data=self.quality['chisq/Ndof'])
            G.create_dataset('Ndof', data=self.quality['Ndof'])


################################################################################
# Class for dealing with covariance
################################################################################

class FitCov:

    def __init__(self, cov):
        self.cov = cov

    def __repr__(self):
        return self.cov.__repr__()

    def offdiagdamp(self, damp):
        N = len(self.cov)
        out = (1-damp)*np.diag(np.diag(self.cov)) \
            + np.full((N,N), damp)*self.cov
        return out
    
    def correlated(self):
        return self.cov

    def uncorrelated(self):
        return np.diag(np.diag(self.cov))

    def svd(self, rcond=None, cut_back=None, full_output=False):
        U, s, VT  = np.linalg.svd(self.cov, full_matrices=True, hermitian=True)
        
        if rcond is not None and cut_back is not None:
            raise ValueError("'rcond' and 'cut_back' cannot be both assigned")
            
        if (rcond is None and cut_back is not None) or \
           (rcond is not None and cut_back is None):
            if rcond is None:
                rcond = np.linalg.cond(self.cov)
                s_last = s[-cut_back]
                s[-cut_back:] = s_last 
            elif cut_back is None:
                covrcond = np.linalg.cond(self.cov)
                if not rcond>covrcond:
                    indx = np.where(s[0]/s > rcond)[0][0]
                    s_last = s[indx]
                    s[indx:] = s_last 
        
        if full_output:
            out = U, s, VT
        else:
            out = U@np.diag(s)@VT
        return out

# # FIXME this is rcond
# def svd_inv(cov, rcond=None, cut_back=None):
#     U, s, VT = np.linalg.svd(cov, full_matrices=True, hermitian=True)
#     # cov = U@np.diag(s)@VT

#     if (rcond is not None) and (rcond is not None):
#         raise ValueError("'rcond' and 'cut_back' cannot be both not None")
        
#     if (rcond is None) and (cut_back is None):
#         sinv = 1/s
#     elif rcond is None:
#         rcond = np.linalg.cond(cov)

#         s_last = s[-cut_back]
#         s[-cut_back:] = s_last 
#         sinv = 1/s
#     elif cut_back is None:
#         covrcond = np.linalg.cond(cov)
#         if rcond>covrcond:
#             sinv = 1/s
#         else:
#             indx = np.where(s[0]/s > rcond)[0][0]
#             s_last = s[indx]
#             s[indx:] = s_last 
#             sinv = 1/s
    
#     inv = VT.T@np.diag(sinv)@U.T
#     return inv



################################################################################
# Class Fitter
################################################################################


class Fitter:

    def __init__(self, x, y, fit_func,
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

        if isinstance(fit_func, str):
            libfunc = getattr(lib, fit_func)
        else:
            libfunc = fit_func
        self.func_param = list(libfunc.PARAM.values())
        self.funcstr = libfunc.STRING
        def func(param, x):
            return libfunc()(param, x, *fit_func_args, **fit_func_kwargs)
        self._fit_func = func
        
        self.prior = None
    
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
         
    def _residual(self, x, y, cov_inv_sqrt):
        def func(param):
            fit_func = self.fit_func(param, x).flatten() # FIXME check if it's fine always
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
        assert(set(guess_dict.keys())==set(self.func_param))
        param_dict = {}

        out_guess = []
        for idxp, param in enumerate(self.func_param):
            guess_param = guess_dict[param] 
            if isinstance(guess_param, dict):
                guess_param_list = []
                for guess_param_k, guess_param_v in guess_param.items():
                    guess_param_list.append(f'{param}{guess_param_k}')
                    out_guess.append(guess_param_v)
                param_dict[idxp] = guess_param_list 
            elif isinstance(guess_param, (list, np.ndarray)):
                guess_param_list = []
                for idxg, guess_param_v in enumerate(guess_param):
                    guess_param_list.append(f'{param}_{idxg}')
                    out_guess.append(guess_param_v)
                param_dict[idxp] = guess_param_list
            else:
                param_dict[idxp] = param 
                out_guess.append(guess_param) 
        out_guess = np.asarray(out_guess)
        
        return param_dict, out_guess


    def _collect_prior_data(self):
        prior_data = []
        for k, v in self.param.items():
            prior_data.append(self.prior_data[v])
        prior_data = [self.prior_data[i] for i in range(self.num_param)]
        return dM.merge(prior_data)

    def _fitter(self, x, y, guess, cov_inv_sqrt):
        res = self._residual(x, y, cov_inv_sqrt)
        #sol = leastsq(res, guess, maxfev=2000, ftol=1e-10, xtol=1e-10, full_output=True)
        #sol = least_squares(res, guess)#, jac='3-point')

        sol = least_squares(
            fun=res, x0=guess,
            xtol=1e-10, gtol=1e-10, ftol=1e-10,
            max_nfev=2000,
        )

        return sol

    @dataStats_args
    def _eval(self, x, y, guess, cov_inv_sqrt): # prior data must be a list of DataStats
        sol = self._fitter(x, y, guess, cov_inv_sqrt)
        return sol.x

    def _set_fit_points(self, fit_points):
        x = self.x[fit_points]
        y = self.y[fit_points]
        return x, y

    # # COVARIANCE
    # def _cov(self, data, correlated, offdiagdamp=1):
    #     if correlated==True:
    #         cov = data.cov
    #         N = len(cov)
    #         cov = (1-offdiagdamp)*np.diag(np.diag(cov)) + np.full((N,N), offdiagdamp)*cov
    #     elif correlated==False:
    #         cov = np.diag(data.err**2)
    #     return cov

    # def _cov_inv(self, cov, cut_back=None, set_equal=True):
    #     if cut_back is not None:
    #         cov_inv = svd_inv(cov, cut_back=cut_back, set_equal=set_equal)
    #     else:
    #         cov_inv = np.linalg.inv(cov)
    #     return cov_inv

    def _set_cov(self, y, method=None, **method_kwargs):
        _cov = y.cov
        cov = getattr(FitCov(_cov), method)(**method_kwargs)              
        return cov
        
    # FIT EVALUATION
    def eval_fit_quality(self, fit_points, guess, *,
             cov_inv=None, cov=None,
             method='correlated', **method_kwargs):
        
        # set fit_points
        x, y = self._set_fit_points(fit_points)

        # parse the guess
        _, guess = self._parse_guess(guess)

        # set cov
        if cov_inv is None:
            if cov is None: 
                _cov = y.cov
                cov = self._set_cov(y, method, **method_kwargs)     
            cov_inv = np.linalg.inv(cov)
        cov_inv_sqrt = cholesky(cov_inv)

        # fit
        fit = self._fitter(x, y.mean, guess, cov_inv_sqrt)
        out = _collect_fit_quality(fit)
        return out
    
    def eval(self, fit_points, guess, *,
             cov_inv=None, cov=None,
             method='correlated', **method_kwargs):
        
        x, y = self._set_fit_points(fit_points)

        # parse the guess
        param_dict, guess = self._parse_guess(guess)

        # set covariance and Cholesky decomposition
        if cov_inv is None:
            if cov is None: 
                _cov = y.cov
                cov = self._set_cov(y, method, **method_kwargs)     
            cov_inv = np.linalg.inv(cov)
        cov_inv_sqrt = cholesky(cov_inv)

        # fit
        fit = self._fitter(x, y.mean, guess, cov_inv_sqrt)
        sol = self._eval(x, y, guess, cov_inv_sqrt)
        #out = self._collect_output(fit, param_dict, sol)

        fitres = FitResult(fit, self.func_param, param_dict, sol)
        return fitres #out
    
    # def eval_gauss(self, fit_range, guess, *,
    #          cov=None, correlated=True, offdiagdamp=1):
        
    #     x = self.x[fit_range]
    #     y = self.y[fit_range]
    #     # set covariance
    #     if cov is None:
    #         cov = self._cov(y, correlated, offdiagdamp)   
        
    #     # parse the guess
    #     guess = self._parse_guess(guess)

    #     # redefine fit_func to match curve_fit conventions
    #     def fit_func(x, *param):
    #         return self.fit_func(param, x).flatten()

    #     fit = curve_fit(fit_func, x, y.mean, sigma=cov, p0=guess,
    #             absolute_sigma=True, check_finite=True, method='trf',
    #             ftol=1e-08,xtol=1e-10,gtol=1e-10, full_output=True)
        
    #     fit_mean = fit[0]
    #     fit_cov = fit[1]
    #     fit_out = dM.DataErr(fit_mean, fit_cov)
    #     fit_out = fit_out.to_dataStats(None, self.statsType)

    #     return fit_out

    # # this must scan ranges and provide a report (pandas dataframe)
    # def _scan(self, fit_range: list, guess: list, *, correlated, min_num_points=None, max_num_points=None, thin=1):
    #     if min_num_points is None:
    #         min_num_points = len(self.param)+1
    #     elif min_num_points<=len(self.param):
    #         param = list(self.param.values())
    #         raise ValueError(
    #             "'min_num_points' must be > than number of parameters, "
    #             f"here {len(self.param)}, i.e. {param} "
    #         )
        
    #     lower, upper = fit_range
    #     tot_lenght = upper - lower

    #     if max_num_points is None:
    #         max_num_points = tot_lenght
    #     elif max_num_points>tot_lenght:
    #         raise ValueError(
    #             "'max_num_points' must be < than the lenght of the fit range, "
    #             f"here {tot_lenght}"
    #         )

    #     x = self.x[lower:upper]
    #     y = self.y[lower:upper]
    #     guess = self._parse_guess(guess)
    #     #cov_inv_sqrt = self._cov_inv_sqrt(y, correlated)

    #     cov = self._cov(y, correlated)

    #     fit_points = []
    #     fit_quality = []
    #     for l in range(tot_lenght):
    #         for u in range(min_num_points+l, l+max_num_points+1):
    #             x_cut = x[l:u:thin]
    #             if len(x_cut)<min_num_points:
    #                 continue
    #             y_cut = y[l:u:thin]
    #             cov_inv_sqrt_cut = cholesky(np.linalg.inv(cov[l:u:thin,l:u:thin]))

    #             #fit = self._eval_mean(x_cut, y_cut.mean, guess, cov_inv_sqrt_cut)
    #             fit = self._fitter(x_cut, y_cut.mean, guess, cov_inv_sqrt_cut)
    #             param = self._eval(x_cut, y_cut.mean, guess, cov_inv_sqrt_cut)
    #             out = {**_collect_fit_quality(fit), **self._collect_fit_param(param)}
                
    #             # collect
    #             fit_points.append(str(slice(lower+l, lower+u, thin)))
    #             fit_quality.append(out)

    #     data = pd.DataFrame(fit_quality, index=fit_points)
    #     data = data.sort_values('pval', ascending=False)
    #     return data
    
    # def full_scan(self, fit_range: list, guess: dict, *, correlated,
    #               min_num_points=None, max_num_points=None, thin=1):
    #     if min_num_points is None:
    #         min_num_points = len(self.param)+1
    #     elif min_num_points<=len(self.param):
    #         param = list(self.param.values())
    #         raise ValueError(
    #             "'min_num_points' must be > than number of parameters, "
    #             f"here {len(self.param)}, i.e. {param} "
    #         )
        
    #     lower, upper = fit_range
    #     tot_lenght = upper - lower

    #     if max_num_points is None:
    #         max_num_points = tot_lenght
    #     elif max_num_points>tot_lenght:
    #         raise ValueError(
    #             "'max_num_points' must be < than the lenght of the fit range, "
    #             f"here {tot_lenght}"
    #         )

    #     x = self.x[lower:upper]
    #     y = self.y[lower:upper]
    #     guess = self._parse_guess(guess)
    #     #cov_inv_sqrt = self._cov_inv_sqrt(y, correlated)

    #     cov = self._cov(y, correlated)

    #     fit_points = []
    #     fit_quality = []
    #     print(x)
    #     for l in range(tot_lenght-min_num_points+1):
    #         u_min = l+min_num_points
    #         u_max = l+max_num_points+1
    #         if u_max>tot_lenght:
    #             u_max = tot_lenght+1
    #         for u in range(u_min, u_max):
    #             x_cut = x[l:u:thin]
    #             if len(x_cut)<min_num_points:
    #                 continue
    #             y_cut = y[l:u:thin]
    #             cov_inv_sqrt_cut = cholesky(np.linalg.inv(cov[l:u:thin,l:u:thin]))

    #             #fit = self._eval_mean(x_cut, y_cut.mean, guess, cov_inv_sqrt_cut)
    #             prior = self._collect_prior_data()
    #             fit = self._fitter(x_cut, y_cut.mean, guess, cov_inv_sqrt_cut, prior.mean)
    #             quality = _collect_fit_quality(fit)
    #             pv = float(quality['pval'])
    #             # if pv<0.05 or pv>0.95:
    #             #     continue
    #             param = self._eval(x_cut, y_cut, guess, cov_inv_sqrt_cut, prior)
    #             out = {**quality, **self._collect_fit_param_str(param)}
                
    #             # collect
    #             fit_points.append(str(slice(lower+l, lower+u, thin)))
    #             fit_quality.append(out)

    #     data = pd.DataFrame(fit_quality, index=fit_points)
    #     data = data.sort_values('pval', ascending=False)
    #     return data


    # combine the info from scan and tell me the best option
    def _optimize(self):
        pass

    # combine _scan and _optimize
    def finder(self):
        pass

    # take output from leastsq and print results nicely
    def print(self):
        pass
