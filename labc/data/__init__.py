"""Data containers for statistical analysis in lattice QCD.

Provides :class:`DataBins`, :class:`DataStats`, and :class:`DataErr` —
containers that store mean values alongside resampled bins and propagate
errors transparently through arithmetic operations.  Standard NumPy
functions work directly on these objects via the ``__array_ufunc__`` and
``__array_function__`` hooks.
"""
from __future__ import annotations

import numpy as np
from math import floor, log10
from typing import TYPE_CHECKING
from .container import Writer as _Writer
from .utilities import _get_extension
from scipy.linalg import block_diag

if TYPE_CHECKING:
    from ..stats._statsbase import StatsBase


# print methods
def _print_dataStats(mean, err, prec):
    """Print mean and error in the form (mean +- err)e+xx """
    power_err = floor(log10(np.abs(err))) if not err==0 else 0 
    power_mean = floor(log10(np.abs(mean))) if not mean==0 else 0
    power_rel = power_err-power_mean
    
    power_str = f"{10**power_mean:.0e}".replace('1e', 'e')
    mean_str = f"{mean/10**power_mean: .{prec}f}"
    
    if power_rel<-3 or power_rel>0:
        err_str = f"{err * 10**(-power_mean):.1e}"
    else:
        err_str = f"{err * 10**(-power_mean):.{prec}f}"
    
    out = f"({mean_str} +- {err_str} ){power_str}"
    return out


# notation like 3.244(12)e-01 or 
def _print_dataStats_bis(mean, err, num_digits=2, scientific=False):
    """Print mean and error in the form (mean(err))e+xx"""   
    power_err = floor(log10(np.abs(err))) if not err==0 else 0 
    power_mean = floor(log10(np.abs(mean))) if not mean==0 else 0
    power_rel = power_err-power_mean
    
    power_str = f"{10**power_mean:.0e}".replace('1e', 'e')

    # notation like 0.3244(12)
    if scientific==False:
      if -5<power_rel<=0:
        # mean value < 1 
        if power_mean<0:
          num_zero_after_comma = np.abs(power_mean) - 1
          num_significant_digits = num_digits + np.abs(power_rel)      
          mean_prec = num_significant_digits + num_zero_after_comma 
          mean_str = f"{mean:.{mean_prec}f}"

          err_prec = -power_err + (num_digits-1) 
          err_digits = round(err * 10**(err_prec))
          err_str = f"{err_digits}"
        
        # mean value >= 1
        elif power_mean>=0:
          num_digits_before_comma = np.abs(power_mean) + 1
          num_significant_digits = num_digits + np.abs(power_rel)       
          mean_prec = num_significant_digits - num_digits_before_comma
          mean_str = f"{mean:.{mean_prec}f}"

          if power_err>=0:
            err_prec = -power_err + (num_digits-1) 
            err_str = f"{err:.{err_prec}f}"
          else:
            err_prec = -power_err + (num_digits-1) 
            err_digits = round(err * 10**(err_prec))
            err_str = f"{err_digits}"

        out = f"{mean_str}({err_str})"

      elif power_rel > 0:
        # error >= mean: show both mean and error to the same decimal places,
        # determined by rounding mean to (num_digits-1) significant figures.
        # The matching decimal in the error avoids ambiguity, e.g.
        # 0.3(121.0) is unambiguous, while 0.3(121) could be read as +-0.121
        d = max(0, num_digits - 2 - power_mean)
        mean_str = f"{mean:.{d}f}"
        err_str  = f"{err:.{d}f}"
        out = f"{mean_str}({err_str})"
      else:
        # power_rel <= -5: very small error, use standard compact notation
        # with as many decimal places as needed to correctly place the error digit
        err_prec = -power_err + (num_digits - 1)
        err_digits = round(err * 10**err_prec)
        mean_str = f"{mean:.{err_prec}f}"
        err_str = f"{err_digits}"
        out = f"{mean_str}({err_str})"
    
    # notation like 3.244(12)e-01
    if scientific==True:
      if -5<power_rel<0:
        mean_num_digits = power_mean-power_err + (num_digits-1)
        mean_str = f"{mean/10**power_mean:.{mean_num_digits}f}"

        err_prec = -power_err + (num_digits-1)
        err_digits = round(err * 10**(err_prec))
        err_str = f"{err_digits}"
        out = f"{mean_str}({err_str}){power_str}"
        
      elif power_rel==0:
        mean_num_digits = power_mean-power_err + (num_digits-1)
        mean_str = f"{mean/10**power_mean:.{mean_num_digits}f}"

        err_prec = -power_err #+ (num_digits-1)
        err_digits = err*10**(err_prec)
        err_str = f"{err_digits:.{num_digits-1}f}"
        out = f"{mean_str}({err_str}){power_str}"
      else:        
        mean_str = f"{mean/10**power_mean:.{num_digits-1}f}"
        err_str = f"{err * 10**(-power_mean):.{num_digits-1}e}"
        out = f"({mean_str} +- {err_str}){power_str}"    

    return out


################################################################################
# DataBins
################################################################################

class DataBins:
    """Container for binned data supporting arithmetic error propagation.
    
    Essentially a thin wrapper around a ``(1+num_bins, T)`` array: 
    the first row stores the mean and the remaining rows store the resampled bins.
    It defines arithmetic operations (``+``, ``-``, ``*``, ``/``) that act
    element-wise on both the mean and all bins simultaneously.

    Parameters
    ----------
    mean : np.ndarray or float
        Central value(s).  A scalar is promoted to a 1-element array.
    bins : np.ndarray
        Resampled bins, shape ``(num_bins, T)``.
    *args, **kwargs
        Forwarded to :meth:`_make_class` when constructing derived objects.

    Attributes
    ----------
    mean : np.ndarray
        Central value(s), shape ``(T,)``.
    bins : np.ndarray
        Resampled bins, shape ``(num_bins, T)``.
    """

    def __init__(self, mean: np.ndarray | float, bins: np.ndarray,
                 *args, **kwargs) -> None:
        # make sure we always deal with numpy array
        if not isinstance(mean, (np.ndarray, list)):
            mean = np.array([mean])
            bins = np.reshape(bins, (len(bins), len(mean)))

        self._args = args
        self._kwargs = kwargs
        self._data_vectorized = np.concatenate((np.array([mean]), bins), axis=0)
        self.mean = self._data_vectorized[0]
        self.bins = self._data_vectorized[1:]

    def num_bins(self) -> int:
        """Return the number of resampled bins."""
        return len(self.bins)

    def __len__(self) -> int:
        """Return the number of observables ``T``."""
        return len(self.mean)
    
    ############################################################################
    # MATH
    ############################################################################
    
    def _make_class(self, mean, bins):
        return self.__class__(mean, bins, *self._args, **self._kwargs)

    # generic overload for mathematical operations among 2 DataStats objects
    def _overload_math_class(self, other, operation):
        if type(self)==type(other):
            out_data = getattr(self._data_vectorized, operation)(other._data_vectorized)
            out = self._make_class(out_data[0], out_data[1:])
        else:
            out = NotImplemented
        return out
    
    # generic overload for mathematical operations (following numpy)
    def _overload_math_numpy(self, other, operation):
        out_data = getattr(self._data_vectorized, operation)(other)
        out = self._make_class(out_data[0], out_data[1:])
        return out
    
    # math overload
    def _overload_math(self, other, operation):
        if isinstance(other, DataBins):
            out = self._overload_math_class(other, operation)  
        elif isinstance(other, (int, float, np.ndarray)):
            out = self._overload_math_numpy(other, operation)
        else:
            out = NotImplemented
        return out

    # explicit (slow) check for overload of math operations
    def _check_math(self, other, operation):
        out_mean = getattr(self.mean, operation)(other)
        out_bins = np.apply_along_axis(
            lambda bin: getattr(bin, operation)(other), 1,
            self.bins
        )
        out = self._make_class(out_mean, out_bins)
        return out

    # OVERLOAD OF MATH OPERATIONS
    def __mul__(self, other):
        return self._overload_math(other, '__mul__')
    
    def __rmul__(self, other):
        return self._overload_math(other, '__rmul__')
            
    def __truediv__(self, other):
        return self._overload_math(other, '__truediv__')
    
    def __rtruediv__(self, other):
        return self._overload_math(other, '__rtruediv__')
    
    def __add__(self, other):
        return self._overload_math(other, '__add__')

    def __radd__(self, other):
        return self._overload_math(other, '__radd__')

    def __sub__(self, other):
        return self._overload_math(other, '__sub__')

    def __rsub__(self, other):
        return self._overload_math(other, '__rsub__')

    def __pow__(self, other):
        return self._overload_math(other, '__pow__')

    def __neg__(self):
        return -1*self
    
    def __pos__(self):
        return +1*self
    
    def __eq__(self, other):
        # np.array_equal ?
        if issubclass(other.__class__, DataBins):
            # add some printing
            if np.allclose(self.mean, other.mean, atol=1e-15) \
               and np.allclose(self.bins, other.bins, atol=1e-15):
                return True
            else:
                return False  

    # HOOK NUMPY
    def __array__(self, dtype=None):  # dtype required by numpy API, always ignored
        out = np.empty((), dtype=object)
        out[()] = self
        return out


################################################################################
# DataStats
################################################################################

class DataStats(DataBins):
    """Binned data with an associated statistical resampling strategy.

    Extends :class:`DataBins` with error, covariance, and correlation
    estimates computed via the attached :class:`~labc.stats.StatsBase`
    resampling object.  All three quantities are cached on first access.

    Parameters
    ----------
    mean : np.ndarray or float
        Central value(s), shape ``(T,)`` or scalar.
    bins : np.ndarray
        Jackknife or bootstrap bins, shape ``(num_bins, T)``.
    statsType : StatsBase
        Resampling strategy used to compute errors and covariances.

    Attributes
    ----------
    mean : np.ndarray
        Central value(s), shape ``(T,)``.
    bins : np.ndarray
        Resampled bins, shape ``(num_bins, T)``.
    statsType : StatsBase
        Resampling strategy attached to this dataset.
    err : np.ndarray
        Statistical error, shape ``(T,)``.  Computed and cached on first access.
    cov : np.ndarray
        Covariance matrix, shape ``(T, T)``.  Computed and cached on first access.
    corr : np.ndarray
        Correlation matrix, shape ``(T, T)``.  Computed and cached on first access.
    """

    def __init__(self, mean: np.ndarray | float, bins: np.ndarray,
                 statsType: StatsBase) -> None:
        super().__init__(mean, bins, statsType)
        self._err = None
        self._cov = None
        self._corr = None

        self.statsType = statsType

    def _overload_math_class(self, other, operation):
        if isinstance(other, DataErr):
            other_as_ds = other._to_datastats(self)
            return getattr(self, operation)(other_as_ds)
        if isinstance(other, DataBins):
            out_data = getattr(self._data_vectorized, operation)(other._data_vectorized)
            return self._make_class(out_data[0], out_data[1:])
        return NotImplemented

    # ERROR
    @property
    def err(self) -> np.ndarray:
        """Statistical error, shape ``(T,)``.

        Computed by :meth:`~labc.stats.StatsBase.err_func` of the attached
        ``statsType`` and cached after the first call.
        """
        if self._err is None:
            self._err = self.statsType.err_func(self.mean, self.bins)
        return self._err

    def rel_err(self) -> np.ndarray:
        """Absolute relative error ``|err / mean|``, shape ``(T,)``."""
        return np.abs(self.err/self.mean)

    def rel_diff(self, other: DataStats) -> DataStats:
        """Element-wise relative difference ``(self - other) / self``.

        Parameters
        ----------
        other : DataStats
            Dataset to compare against.  Must have the same ``T`` and
            compatible ``num_bins``.

        Returns
        -------
        DataStats
            Relative difference as a new :class:`DataStats`.
        """
        assert(isinstance(other, DataStats))
        out_mean = (self.mean - other.mean) / self.mean
        out_bins = (self.bins - other.bins) / self.bins
        out = self._make_class(out_mean, out_bins)
        return out

    # COVARIANCE MATRIX
    @property
    def cov(self) -> np.ndarray:
        """Covariance matrix, shape ``(T, T)``.

        Computed by :meth:`~labc.stats.StatsBase.cov` of the attached
        ``statsType`` and cached after the first call.
        """
        if self._cov is None:
            self._cov = self.statsType.cov(self)
        return self._cov

    # CORRELATION MATRIX
    @property
    def corr(self) -> np.ndarray:
        """Correlation matrix, shape ``(T, T)``.

        Computed by :meth:`~labc.stats.StatsBase.corr` of the attached
        ``statsType`` and cached after the first call.
        """
        if self._corr is None:
            self._corr = self.statsType.corr(self)
        return self._corr
    
    # OUTPUT
    def __repr__(self):
        prec = 4 # precision
        space = len('DataStats[')*" "
        out = f"DataStats["
        if len(self)>1:
            out += f"{self.mean[0]: .{prec}e} +- {self.err[0]:.{prec}e},\n" + space
            for mean, err in zip(self.mean[1:-1], self.err[1:-1]):
                out += f"{mean: .{prec}e} +- {err:.{prec}e},\n" + space
        out += f"{self.mean[-1]: .{prec}e} +- {self.err[-1]:.{prec}e}]"      
        return out

    def __str__(self):
        prec = 5 # precision
        space = len('DataStats[')*" "
        out = f"DataStats["
        if len(self)>1:
            out += _print_dataStats(self.mean[0], self.err[0], prec) + ",\n"
            for mean, err in zip(self.mean[1:-1], self.err[1:-1]):
                out += space + _print_dataStats(mean, err, prec) + ",\n"
            out += space + _print_dataStats(self.mean[-1], self.err[-1], prec) + "]"
        else:
            out += _print_dataStats(self.mean[0], self.err[0], prec) + "]" 
        return out
    
    def print(self, num_digits: int = 2, scientific: bool = False) -> list[str]:
        """Return a list of compact ``mean(err)`` strings, one per observable.

        Parameters
        ----------
        num_digits : int, optional
            Number of significant digits in the error.  Default is 2.
        scientific : bool, optional
            If ``True``, use scientific notation.  Default is ``False``.

        Returns
        -------
        list[str]
            One formatted string per element of ``self``.
        """
        out = [_print_dataStats_bis(self.mean[0], self.err[0], num_digits, scientific)]
        if len(self)>1:
            for mean, err in zip(self.mean[1:-1], self.err[1:-1]):
                out.append(_print_dataStats_bis(mean, err, num_digits, scientific))
            out.append(_print_dataStats_bis(self.mean[-1], self.err[-1], num_digits, scientific))
        return out

    # SAVE
    # TODO: change this into a more efficient factory
    def save(self, file_out: str, group: str, *args, **kwargs) -> None:
        """Save mean, error, and bins to an HDF5 file.

        Parameters
        ----------
        file_out : str
            Path to the output file.  Only ``.h5`` is currently supported.
        group : str
            HDF5 group name under which the data are stored.
        *args, **kwargs
            Forwarded to the writer's ``add_mean`` / ``add_err`` / ``add_bins``
            methods.
        """
        ext = _get_extension(file_out)
        if ext=='.h5':
            writer = _Writer(file_out, 'stats')
            writer.add_stats_group(self.statsType, group)
            writer.add_mean(group, self.mean, *args, **kwargs)
            writer.add_err(group, self.err, *args, **kwargs)
            writer.add_bins(group, self.bins, *args, **kwargs)
        else:
            raise NotImplementedError(f"File extension '{ext}' not implemented!")


    @staticmethod
    def _has_dataStats(args):
        # check if there are DataStats object in args
        out = False
        for arg in args:
            if isinstance(arg, DataStats):
                out = True
                break
        return out 

    @staticmethod
    def _get_statsType(args):
        for arg in args:
            if isinstance(arg, DataStats):
                out = arg.statsType
                break
        return out                

    @staticmethod
    def _get_num_bins(args):
        if not type(args)==tuple:
            args = list([args])
        for arg in args:
            if isinstance(arg, DataStats):
                num_bins = arg.num_bins()
        return num_bins

    @staticmethod
    def _collect_data_args(args):
        args_data = []
        for arg in args:
            if isinstance(arg, DataStats):
                arg = arg._data_vectorized
            args_data.append(arg)
        args_data = tuple(args_data)
        return args_data

    @staticmethod
    def _collect_mean_args(args):
        args_mean = []
        for arg in args:
            if isinstance(arg, DataStats):
                arg = arg.mean
            args_mean.append(arg)
        args_mean = tuple(args_mean)
        return args_mean

    @staticmethod
    def _collect_bins_args(args, num_bins):
        args_bins = []
        for b in range(num_bins):
            args_bin = []
            for arg in args:
                if isinstance(arg, DataStats):
                    arg = arg.bins[b]
                args_bin.append(arg)
            args_bins.append(tuple(args_bin))
        
        return args_bins

    @staticmethod
    def _collect_bin_args(args, b):
        args_bin = []
        for arg in args:
            if isinstance(arg, DataStats):
                arg = arg.bins[b]
            args_bin.append(arg)
        args_bin = tuple(args_bin)
        return args_bin
    
    @staticmethod
    def _collect_mean_kwargs(kwargs):
        dict_mean = {}
        for key, value in kwargs.items():
            if isinstance(value, DataStats):
                value = value.mean
            dict_mean[key] = value        
        return dict_mean
    
    #-----HOOK ON NUMPY FUNCTIONS (REDEFINE NUMPY BEHAVIOUR)--------------------
    
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # print('ufunc', ufunc)
        # print('method', method)
        # print('args', args)
        # print('kwargs', kwargs)
        if method=='__call__':
            args_data = self._collect_data_args(args)
            try:
                out_data = ufunc(*args_data, axis=1, **kwargs)
            except TypeError:
                out_data = ufunc(*args_data, **kwargs)
            out = self._make_class(out_data[0], out_data[1:])
            return out
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        # print('func', func)
        # print('types', types)
        # print('args', args)
        # print('kwargs', kwargs)       

        # safe and pedantic implementation
        # args_mean = self._collect_mean_args(args)  
        # out_mean = func(*args_mean, **kwargs)
        
        # num_bins = self._get_num_bins(args)
        # args_bins = self._collect_bins_args(args, num_bins)  
        # out_bins = []
        # for b in range(num_bins):
        #     out_bins.append(func(*args_bins[b], **kwargs))
        # out_bins = np.asarray(out_bins)
        # out = self._make_dataStats(out_mean, out_bins)
            
        args_data = self._collect_data_args(args)
        try:
            out_data = func(*args_data, axis=1, **kwargs)
        except TypeError:
            out_data = func(*args_data, **kwargs)
        out = self._make_class(out_data[0], out_data[1:])
        return out

    def __getitem__(self, key):
        out_data = self._data_vectorized[:, key]     
        # TODO: implement this and get rid of bins reshape in __init__ 
        # if isinstance(key, int):
        #     out_data = np.reshape(out_data, (len(out_data)))
        out = self._make_class(out_data[0], out_data[1:])
        return out 
    
    def __setitem__(self, index, data):
        self._data_vectorized[:, index] = data._data_vectorized.flatten()


################################################################################
# DataErr
################################################################################


class DataErr(DataBins):
    """Observable with Gaussian uncertainty defined by a mean and covariance.

    Stores a central value and a covariance matrix and propagates errors
    analytically through arithmetic operations.  When combined with a
    :class:`DataStats` object, the covariance is converted on-the-fly to
    compatible resampled bins.

    Parameters
    ----------
    mean : np.ndarray or float
        Central value(s), shape ``(T,)`` or scalar.
    err_or_cov : np.ndarray
        Either a 1-D error array of shape ``(T,)`` (sqrt of diagonal covariance)
        or a full 2-D covariance matrix of shape ``(T, T)``.
    seed : int or None, optional
        Seed for the random number generator used when sampling bins.
        Fixing the seed makes bin generation reproducible.

    Attributes
    ----------
    mean : np.ndarray
        Central value(s), shape ``(T,)``.
    err : np.ndarray
        Diagonal errors ``sqrt(diag(cov))``, shape ``(T,)``.
    cov : np.ndarray
        Covariance matrix, shape ``(T, T)``.
    corr : np.ndarray
        Correlation matrix, shape ``(T, T)``.
    seed : int or None
        Random seed for bin generation.
    """

    NUM_BINS: int = 5000

    def __init__(self, mean: np.ndarray | float,
                 err_or_cov: np.ndarray, *, seed: int | None = None) -> None:
        if not isinstance(mean, (np.ndarray, list)):
            mean = np.array([mean])
        
        if not isinstance(err_or_cov, (np.ndarray)):
            err_or_cov = np.array([err_or_cov])

        if err_or_cov.ndim==1:
            self.err = err_or_cov 
            self.cov = np.diag(self.err**2)
        elif err_or_cov.ndim==2:
            self.cov = err_or_cov
            self.err = np.sqrt(np.diag(self.cov))
        self.corr = np.diag(1/self.err)@self.cov@np.diag(1/self.err)

        self._args = ()
        self._kwargs = {'seed': seed}
        self.seed = seed

        self.mean = mean
        self._num_bins = self.NUM_BINS


    @property
    def num_bins(self) -> int:
        """Default number of bins used when sampling without a target DataStats."""
        return self._num_bins

    @num_bins.setter
    def num_bins(self, value: int) -> None:
        self._num_bins = value
    
    def print(self, num_digits=2, scientific=False):
        # fill outputstring
        out = [_print_dataStats_bis(self.mean[0], self.err[0], num_digits, scientific)]
        if len(self)>1:
            #out.append(_print_dataStats_bis(self.mean[0], self.err[0], prec))
            for mean, err in zip(self.mean[1:-1], self.err[1:-1]):
                out.append(_print_dataStats_bis(mean, err, num_digits, scientific))
            out.append(_print_dataStats_bis(self.mean[-1], self.err[-1], num_digits, scientific))
        return out


    def _resample(self, num_bins: int | None = None,
                  statsType: StatsBase | None = None) -> np.ndarray:
        """Sample bins from the Gaussian distribution ``N(mean, cov)``.

        Parameters
        ----------
        num_bins : int, optional
            Number of bins to generate.  Defaults to ``self.num_bins`` when
            ``statsType`` is ``None``, or to ``statsType.num_bins`` otherwise.
        statsType : StatsBase, optional
            If provided, the raw samples are passed through
            ``statsType.generate_bins`` to produce jackknife/bootstrap bins.

        Returns
        -------
        np.ndarray
            Resampled bins, shape ``(num_bins, T)``.

        .. warning::
            Slicing a :class:`DataErr` before arithmetic (e.g.
            ``de[0] + de[1]``) samples each component independently and loses
            off-diagonal covariance information.  Operate on the full object
            and convert with :meth:`to_dataStats` when correlations matter.
        """
        rng = np.random.default_rng(self.seed)

        if statsType is None:
            raw_bins = rng.multivariate_normal(
                self.mean, self.cov, num_bins
            )
            bias = np.mean(raw_bins, 0)-self.mean
            bins = raw_bins-bias
        else:
            if num_bins is None and statsType.num_bins is not None:
                num_bins = statsType.num_bins
            raw_bins = rng.multivariate_normal(
                self.mean, num_bins*self.cov, num_bins
            )
            bias = np.mean(raw_bins, 0)-self.mean
            raw_bins = raw_bins-bias
            # FIXME: add also correction for bias on the error estimate, 
            # which fluctuates by ~1/sqrt(2*num_bins) around self.err
            bins = statsType.generate_bins(raw_bins)

        return bins

    def bins(self, num_bins: int | None = None,
             statsType: StatsBase | None = None) -> np.ndarray:
        """Return sampled bins, shape ``(num_bins, T)``.

        Parameters
        ----------
        num_bins : int, optional
            Number of bins.  Defaults to ``self.num_bins``.
        statsType : StatsBase, optional
            If provided, bins are structured as jackknife/bootstrap samples
            via ``statsType.generate_bins``.

        Returns
        -------
        np.ndarray
            Resampled bins, shape ``(num_bins, T)``.
        """
        if statsType is None:
            if num_bins==None:
                num_bins = self.num_bins
        out = self._resample(num_bins, statsType)
        out = np.reshape(out, (len(out), len(self.mean)))
        return out

    @property
    def _data_vectorized(self):
        bins = self.bins(self._num_bins)
        return np.concatenate((np.array([self.mean]), bins), axis=0)

    def _data_vectorized_with(self, num_bins, statsType):
        bins = self.bins(num_bins, statsType)
        return np.concatenate((np.array([self.mean]), bins), axis=0)

    # def err_func(self):
    #     bins = self.bins
    #     err = np.sqrt(np.var(bins, axis=0))
    #     return err 

    # FIXME build it inside stastType classes
    def _cov_func(self, bins, statsType=None):
        if statsType is None:
            N = bins.shape[1]
            cov = np.cov(bins, rowvar=False)
            cov = np.reshape(cov, (N,N))
        return cov
    

    def __repr__(self):
        prec = 4 # precision
        space = len('DataErr[')*" "
        out = f"DataErr["
        if len(self)>1:
            out += f"{self.mean[0]: .{prec}e} +- {self.err[0]:.{prec}e},\n" + space
            for mean, err in zip(self.mean[1:-1], self.err[1:-1]):
                out += f"{mean: .{prec}e} +- {err:.{prec}e},\n" + space
        out += f"{self.mean[-1]: .{prec}e} +- {self.err[-1]:.{prec}e}]"      
        return out

    def __str__(self):
        prec = 5 # precision
        space = len('DataErr[')*" "
        out = f"DataErr["
        if len(self)>1:
            out += _print_dataStats(self.mean[0], self.err[0], prec) + ",\n"
            for mean, err in zip(self.mean[1:-1], self.err[1:-1]):
                out += space + _print_dataStats(mean, err, prec) + ",\n"
            out += space + _print_dataStats(self.mean[-1], self.err[-1], prec) + "]"
        else:
            out += _print_dataStats(self.mean[0], self.err[0], prec) + "]" 
        return out
    
    # FIXME: consolidate with _resample so _to_datastats simply calls it
    def _to_datastats(self, other: DataStats) -> DataStats:
        """Convert to a :class:`DataStats` compatible with *other*.

        Bins are sampled from :math:`\\mathcal{N}(\\mu,\\,\\Sigma/f)` where
        :math:`f` is the prefactor of *other*'s ``statsType``, so that
        ``statsType.err_func`` recovers ``self.err`` up to finite-sample noise.

        Parameters
        ----------
        other : DataStats
            Target dataset whose ``statsType`` and bin count define the
            output format.

        Returns
        -------
        DataStats
            New :class:`DataStats` with the same mean and (approximate)
            errors as this object.
        """
        statsType = other.statsType
        # prefactor and num_bins inferred from the target DataStats bins array
        prefactor = statsType._get_prefactor(other.bins)
        num_bins = other.num_bins()
        rng = np.random.default_rng(self.seed)
        # sample from N(mean, cov/prefactor) so err_func returns self.err
        bins = rng.multivariate_normal(
            self.mean, self.cov / prefactor, num_bins
        )
        # correct finite-sample bias in the mean
        bins += self.mean - np.mean(bins, 0)
        return DataStats(self.mean, bins, statsType)

    def to_dataStats(self, num_bins: int, statsType: StatsBase) -> DataStats:
        """Convert to a :class:`DataStats` with *num_bins* bins. If *statsType*
        is provided with *num_bins!=None*, it must have *num_bins=statsType.num_bins*. 

        Parameters
        ----------
        num_bins : int
            Number of bins to generate.
        statsType : StatsBase
            Resampling strategy for the output :class:`DataStats`.

        Returns
        -------
        DataStats
            New :class:`DataStats` with mean and binned representation of
            this object's Gaussian uncertainty.
        """
        bins = self.bins(num_bins, statsType)
        return DataStats(self.mean, bins, statsType)
    
    
    def _make_class(self, mean, bins):
        cov = self._cov_func(bins)
        out = self.__class__(mean, cov, *self._args, **self._kwargs)
        out.num_bins = len(bins)
        return out
    
    def _overload_math_class(self, other, operation):
        if isinstance(other, DataErr):
            if self.num_bins==other.num_bins:
                out_data = getattr(
                    self._data_vectorized, operation
                )(other._data_vectorized)
            elif self.num_bins>other.num_bins:
                other_num_bins = other.num_bins
                other.num_bins = self.num_bins
                out_data = getattr(
                    self._data_vectorized, operation
                )(other._data_vectorized)
                other.num_bins = other_num_bins
            elif self.num_bins<other.num_bins:
                self_num_bins = self.num_bins
                self.num_bins = other.num_bins
                out_data = getattr(
                    self._data_vectorized, operation
                )(other._data_vectorized)
                self.num_bins = self_num_bins
            
            out = self._make_class(out_data[0], out_data[1:])
        elif isinstance(other, DataStats):
            self_as_ds = self._to_datastats(other)
            out = getattr(self_as_ds, operation)(other)
        return out
    
    def __getitem__(self, key):
        key_cov = key
        if isinstance(key, int):
            key_cov = slice(key, key+1, None)
        out = DataErr(
            self.mean[key], self.cov[key_cov,key_cov], seed=self.seed
        )
        return out


################################################################################
# UTILITIES
################################################################################

# def merge(*data_in):
#     if isinstance(data_in[0], list):
#         data_in = tuple(data_in[0])
#     statsType = data_in[0].statsType

#     data_vectorized = np.concatenate([data._data_vectorized for data in data_in], axis=1)
#     out = DataStats(data_vectorized[0], data_vectorized[1:], statsType)
#     return out

def merge(*data_in: DataStats | DataErr) -> DataStats | DataErr:
    """Concatenate multiple objects along the observable axis.

    All inputs must be of the same type (:class:`DataStats` or
    :class:`DataErr`).  For :class:`DataStats` the ``statsType`` is taken
    from the first element; for :class:`DataErr` the covariance is assembled
    as a block-diagonal matrix (no cross-correlations between inputs).

    Parameters
    ----------
    *data_in : DataStats or DataErr
        Objects to concatenate.  May also be passed as a single list or
        ``np.ndarray`` of objects.

    Returns
    -------
    DataStats or DataErr
        Concatenated object of the same type as the inputs.
    """
    if isinstance(data_in[0], list):
        data_in = tuple(data_in[0])
    elif isinstance(data_in[0], np.ndarray):
        data_in = tuple(list(data_in[0]))

    if isinstance(data_in[0], DataStats):
        statsType = data_in[0].statsType
        data_vectorized = np.concatenate([data._data_vectorized for data in data_in], axis=1)
        out = DataStats(data_vectorized[0], data_vectorized[1:], statsType)
    elif isinstance(data_in[0], DataErr):
        mean = np.concatenate([data.mean for data in data_in])
        cov = block_diag(*[data.cov for data in data_in])
        out = DataErr(mean, err_or_cov=cov)
    return out


def _parse_bins_arg(num_bins_or_statstype: int | StatsBase,
                    ) -> tuple[int | None, StatsBase | None]:
    """Parse a ``num_bins_or_statstype`` argument into ``(num_bins, statsType)``."""
    if isinstance(num_bins_or_statstype, int):
        return num_bins_or_statstype, None
    statsType = num_bins_or_statstype
    return statsType.num_bins, statsType


def zeros(T: int, num_bins_or_statstype: int | StatsBase) -> DataBins | DataStats:
    """Return a :class:`DataBins` or :class:`DataStats` filled with zeros.

    Parameters
    ----------
    T : int
        Number of observables.
    num_bins_or_statstype : int or StatsBase
        Number of bins (returns :class:`DataBins`) or a ``StatsBase`` object
        (returns :class:`DataStats`).
    """
    num_bins, statsType = _parse_bins_arg(num_bins_or_statstype)
    mean = np.zeros(T)
    bins = np.zeros(shape=(num_bins, T))
    if statsType is None:
        return DataBins(mean, bins)
    return DataStats(mean, bins, statsType)


def ones(T: int, num_bins_or_statstype: int | StatsBase) -> DataBins | DataStats:
    """Return a :class:`DataBins` or :class:`DataStats` filled with ones.

    Parameters
    ----------
    T : int
        Number of observables.
    num_bins_or_statstype : int or StatsBase
        Number of bins (returns :class:`DataBins`) or a ``StatsBase`` object
        (returns :class:`DataStats`).
    """
    num_bins, statsType = _parse_bins_arg(num_bins_or_statstype)
    mean = np.ones(T)
    bins = np.ones(shape=(num_bins, T))
    if statsType is None:
        return DataBins(mean, bins)
    return DataStats(mean, bins, statsType)


def empty(T: int, num_bins_or_statstype: int | StatsBase) -> DataBins | DataStats:
    """Return a :class:`DataBins` or :class:`DataStats` with uninitialized values.

    Parameters
    ----------
    T : int
        Number of observables.
    num_bins_or_statstype : int or StatsBase
        Number of bins (returns :class:`DataBins`) or a ``StatsBase`` object
        (returns :class:`DataStats`).
    """
    num_bins, statsType = _parse_bins_arg(num_bins_or_statstype)
    mean = np.empty(T)
    bins = np.empty(shape=(num_bins, T))
    if statsType is None:
        return DataBins(mean, bins)
    return DataStats(mean, bins, statsType)


def constant(const: float,
             num_bins_or_statstype: int | StatsBase) -> DataBins | DataStats:
    """Return a length-1 object with all values equal to *const*.

    Parameters
    ----------
    const : float
        Constant value for both mean and bins.
    num_bins_or_statstype : int or StatsBase
        Passed to :func:`ones`.
    """
    return const * ones(1, num_bins_or_statstype)

def gaussian(T: int, statsType: StatsBase,
             mu: float = 0.0, sigma: float = 1.0) -> DataStats:
    """Generate a :class:`DataStats` with Gaussian bins, mean=*mu*, err=*sigma*.

    Bins are drawn from :math:`\\mathcal{N}(\\mu,\\,(\\sigma/\\sqrt{f})^2)` where
    :math:`f` is the ``statsType`` prefactor, so that ``err_func`` returns
    *sigma* regardless of the resampling strategy.

    Parameters
    ----------
    T : int
        Number of observables.
    statsType : StatsBase
        Resampling strategy (jackknife or bootstrap).
    mu : float, optional
        Mean value.  Default is ``0.0``.
    sigma : float, optional
        Target error.  Default is ``1.0``.

    Returns
    -------
    DataStats
        Dataset with ``mean = mu`` and ``err = sigma``.
    """
    num_bins = statsType.num_bins
    prefactor = statsType._prefactor_func(num_bins)
    bin_sigma = sigma / np.sqrt(prefactor)
    bins = np.random.normal(mu, bin_sigma, size=(num_bins, T))
    # correct finite-sample bias in the mean
    mean = np.full(T, mu)
    bins += mean - np.mean(bins, axis=0)
    return DataStats(mean, bins, statsType)


def uniform(T: int, statsType: StatsBase,
            low: float = 0.0, high: float = 1.0) -> DataStats:
    """Generate a :class:`DataStats` with uniform bins and err=(high-low)/sqrt(12).

    Bins are drawn from :math:`\\mathrm{Uniform}(\\mathrm{low},\\mathrm{high})`
    and then divided by :math:`\\sqrt{f}` (the ``statsType`` prefactor) so that
    ``err_func`` returns ``(high-low)/sqrt(12)`` for any resampling strategy.

    Parameters
    ----------
    T : int
        Number of observables.
    statsType : StatsBase
        Resampling strategy (jackknife or bootstrap).
    low : float, optional
        Lower bound of the uniform distribution.  Default is ``0.0``.
    high : float, optional
        Upper bound of the uniform distribution.  Default is ``1.0``.

    Returns
    -------
    DataStats
        Dataset with ``mean = (low+high)/2`` and ``err ≈ (high-low)/sqrt(12)``.
    """
    num_bins = statsType.num_bins
    prefactor = statsType._prefactor_func(num_bins)
    bins = np.random.uniform(low, high, size=(num_bins, T))/np.sqrt(prefactor)
    mean = (low + high) / 2.0
    bias = mean - np.mean(bins, axis=0)
    bins += bias
    return DataStats(mean, bins, statsType)


def Z2(T: int, statsType: StatsBase) -> DataStats:
    """Generate a :class:`DataStats` with :math:`\\mathbb{Z}_2` bins, mean=0, err≈1.

    Each bin entry is drawn from ``{-1/sqrt(f), +1/sqrt(f)}`` with equal
    probability, where :math:`f` is the ``statsType`` prefactor.

    Parameters
    ----------
    T : int
        Number of observables.
    statsType : StatsBase
        Resampling strategy (jackknife or bootstrap).

    Returns
    -------
    DataStats
        Dataset with ``mean = 0`` and ``err ≈ 1``.
    """
    num_bins = statsType.num_bins
    prefactor = statsType._prefactor_func(num_bins)
    bin_sigma = 1.0 / np.sqrt(prefactor)
    bins = np.random.choice([-bin_sigma, bin_sigma], size=(num_bins, T))
    mean = np.zeros(T)
    return DataStats(mean, bins, statsType)

################################################################################
# DECORATORS
################################################################################

def dataStats_args(func):
    """Decorator that lifts a scalar function to accept :class:`DataStats` arguments.

    The decorated function is called once on the means and once per bin,
    then the results are assembled into a new :class:`DataStats`.  Use
    :func:`dataStats_vectorized_args` when the function can broadcast over
    the stacked ``(1+num_bins, T)`` array for better performance.
    """

    def wrapper(*args, **kwargs):
        is_data_stats = DataStats._has_dataStats(args)

        if is_data_stats:
            statsType = DataStats._get_statsType(args) 
            num_bins = DataStats._get_num_bins(args) #statsType.num_bins

            args_mean = DataStats._collect_mean_args(args)
            mean = func(*args_mean, **kwargs)    

            args_bins = DataStats._collect_bins_args(args, num_bins)
            bins = []
            for b in range(num_bins):
                bins.append(func(*args_bins[b], **kwargs))
            bins = np.asarray(bins)

            out = DataStats(mean, bins, statsType)
        else:
            out = func(*args, **kwargs)
        return out
    return wrapper


def dataStats_vectorized_args(func):
    """Decorator that lifts a vectorized function to accept :class:`DataStats` arguments.

    The decorated function is called once on the stacked
    ``(1+num_bins, T)`` data array (mean in row 0, bins in rows 1…).
    This is faster than :func:`dataStats_args` when the underlying function
    supports broadcasting over the extra leading axis.
    """

    def wrapper(*args, **kwargs):
        is_data_stats = DataStats._has_dataStats(args)

        if is_data_stats:
            statsType = DataStats._get_statsType(args) 

            args_data = DataStats._collect_data_args(args)
            data = func(*args_data, **kwargs)    

            out = DataStats(data[0], data[1:], statsType)
        else:
            out = func(*args, **kwargs)
        return out
    return wrapper


# FIXME: it needs to be revisited, at the moment it feels a bit ad hoc
def dataStats_func(func):
    """Decorator for functions that return functions."""

    def wrapper(*args, **kwargs):
        # check if there is a DataStats object in args
        is_data_stats = DataStats._has_dataStats(args)

        if is_data_stats:
            statsType = DataStats._get_statsType(args) 
            num_bins =  DataStats._get_num_bins(args) #statsType.num_bins

            args_mean = DataStats._collect_mean_args(args)
            func_mean = func(*args_mean, **kwargs) 

            args_bins = DataStats._collect_bins_args(args, num_bins)
            def func_bins(*args_func_bins, **kwargs_func_bins):      
                out = []
                for b in range(num_bins):
                    func_bin = func(*args_bins[b], **kwargs)
                    out.append(func_bin(*args_func_bins, **kwargs_func_bins))
                return np.asarray(out)
            
            # final output function
            def func_out(*args, **kwargs):
                mean = func_mean(*args, **kwargs)
                bins = func_bins(*args, **kwargs)

                out = DataStats(mean, bins, statsType)
                return out
            
            out = func_out
        else:
            out = func(*args, **kwargs)
        return out  
    return wrapper

################################################################################
# NEW DECORATORS TMP
################################################################################


def _has_dataStats(*args, **kwargs):
    # check if there are DataStats object in args
    out = False
    for arg in tuple(args + tuple(kwargs.values())):
        if isinstance(arg, DataStats):
            out = True
            break
    return out 

def _get_statsType(*args, **kwargs):
    for arg in tuple(args + tuple(kwargs.values())):
        if isinstance(arg, DataStats):
            out = arg.statsType
            break
    return out                


def _get_num_bins(*args, **kwargs):
    for arg in tuple(args + tuple(kwargs.values())):
        if isinstance(arg, DataStats):
            out = arg.num_bins()
            break
    return out    

def _collect_data_args(*args, **kwargs):
    args_data = []
    for arg in args:
        if isinstance(arg, DataStats):
            arg = arg._data_vectorized
        args_data.append(arg)
    args_data = tuple(args_data)

    kwargs_data = {}
    for k, v, in kwargs.items():
        if isinstance(v, DataStats):
            v = v._data_vectorized
        kwargs_data[k] = v
    
    return args_data, kwargs_data
    

def dataStats_args_tmp(func):
    """Decorator to extend a generic function 'func' to allow DataStats
    arguments with vectorization."""

    def wrapper(*args, **kwargs):
        is_data_stats = _has_dataStats(*args, **kwargs)

        #print(args)

        if is_data_stats:
            statsType = _get_statsType(*args, **kwargs)
            num_data = _get_num_bins(*args, **kwargs)+1 # + mean

            data_args = []
            for d in range(num_data):
              _data_args = []
              for arg in args:
                if isinstance(arg, DataStats):
                    _data_args.append(arg._data_vectorized[d])
                else:
                    _data_args.append(arg)
              data_args.append(_data_args)
            
            if bool(kwargs):
              data_kwargs = []
              for d in range(num_data):
                _data_kwargs = {}
                for karg, varg in kwargs.items():
                  if isinstance(varg, DataStats):
                      _data_kwargs[karg] = varg._data_vectorized[d]
                  else:
                      _data_kwargs[karg] = varg
                data_kwargs.append(_data_kwargs)
            else:
              data_kwargs = [{} for d in range(num_data)]
            
            data = []
            for d in range(num_data):
                data.append(func(*data_args[d], **data_kwargs[d]))
              
            data = np.asarray(data)
                  
            out = DataStats(data[0], data[1:], statsType)
        else:
            out = func(*args, **kwargs)
        return out
    return wrapper