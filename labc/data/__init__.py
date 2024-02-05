import numpy as np
from math import floor, log10 
from .container import Writer as _Writer
from .utilities import _get_extension
from scipy.linalg import block_diag


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

# notation like 3.244(12)
def _print_dataStats_bis(mean, err, num_digits=2):
    """Print mean and error in the form (mean(err))e+xx"""   
    power_err = floor(log10(np.abs(err))) if not err==0 else 0 
    power_mean = floor(log10(np.abs(mean))) if not mean==0 else 0
    power_rel = power_err-power_mean
    
    power_str = f"{10**power_mean:.0e}".replace('1e', 'e')

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
    """Basic class for manipulation of binned data."""

    def __init__(self, mean, bins, *args, **kwargs):
        # make sure we always deal with numpy array
        if not isinstance(mean, (np.ndarray, list)):
            mean = np.array([mean])
            bins = np.reshape(bins, (len(bins), len(mean)))

        self._args = args
        self._kwargs = kwargs
        self._data_vectorized = np.concatenate((np.array([mean]), bins), axis=0)
        self.mean = self._data_vectorized[0]
        self.bins = self._data_vectorized[1:]
    
    def num_bins(self):
        return len(self.bins)

    def __len__(self):
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
    # FIXME: I don't particularly like this trick, it's a bit fishy...
    def __array__(self):
        class ObjWrapper:
            def __init__(self, data):
                self.data = data
        out = ObjWrapper(self)
        return np.asarray(out)  


################################################################################
# DataStats
################################################################################

class DataStats(DataBins):    
    """Basic class for data manipulation."""

    def __init__(self, mean, bins, statsType):
        super().__init__(mean, bins, statsType)
        self._err = None
        self._cov = None
        self._corr = None

        self.statsType = statsType

    # ERROR
    @property
    def err(self):
        if self._err is None:
            self._err = self.statsType.err_func(self.mean, self.bins)
        return self._err
    
    def rel_err(self):
        return np.abs(self.err/self.mean)

    def rel_diff(self, other):
        assert(isinstance(other, DataStats))
        out_mean = np.abs((self.mean - other.mean) / self.mean)
        out_bins = np.abs((self.bins - other.bins) / self.bins)
        out = self._make_class(out_mean, out_bins)
        return out 
    
    # COVARIANCE MATRIX
    @property
    def cov(self):
        """Compute the covariance matrix."""
        if self._cov is None:
            self._cov = self.statsType.cov(self)
        return self._cov
    
    # CORRELATION MATRIX
    @property
    def corr(self):
        """Compute the correlation matrix."""
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
    
    def print(self, num_digits=2):
        out = [_print_dataStats_bis(self.mean[0], self.err[0], num_digits)]
        if len(self)>1:
            #out.append(_print_dataStats_bis(self.mean[0], self.err[0], prec))
            for mean, err in zip(self.mean[1:-1], self.err[1:-1]):
                out.append(_print_dataStats_bis(mean, err, num_digits))
            out.append(_print_dataStats_bis(self.mean[-1], self.err[-1], num_digits))
        return out

    # SAVE
    # TODO: change this into a more efficient factory
    def save(self, file_out, group, *args, **kwargs):
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
    
    ############################################################################
    # HOOK ON NUMPY FUNCTIONS (REDEFINE NUMPY BEHAVIOUR)
    ############################################################################


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
    """Class for error propagation."""

    NUM_BINS = 5000

    def __init__(self, mean, err_or_cov, *, seed=None):
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

        self._args = ()
        self._kwargs = {'seed': None}
        self.seed = seed

        self.mean = mean
        self._num_bins = self.NUM_BINS
        self._statsType = None


    @property
    def num_bins(self):
        return self._num_bins
    
    @num_bins.setter
    def num_bins(self, value):
        self._num_bins = value 
        

    def _resample(self, num_bins=None, statsType=None):
        np.random.seed(self.seed)

        if statsType is None:
            raw_bins = np.random.multivariate_normal(
                self.mean, self.cov, num_bins
            )
            bias = np.mean(raw_bins, 0)-self.mean
            bins = raw_bins-bias
        else:
            if num_bins is None:
                num_bins = statsType.num_bins
            else:
                assert(num_bins==statsType.num_bins)
            raw_bins = np.random.multivariate_normal(
                self.mean, num_bins*self.cov, num_bins
            )
            bias = np.mean(raw_bins, 0)-self.mean
            raw_bins = raw_bins-bias
            bins = statsType.generate_bins(raw_bins)

        return bins

    def bins(self, num_bins=None, statsType=None):
        if statsType is None:
            if num_bins==None:
                num_bins = self.num_bins
        out = self._resample(num_bins, statsType)
        out = np.reshape(out, (len(out), len(self.mean)))
        return out

    @property
    def _data_vectorized(self):
        bins = self.bins(self._num_bins, self._statsType)
        out = np.concatenate(
            (np.array([self.mean]), bins), axis=0
        )
        return out

    # def err_func(self):
    #     bins = self.bins
    #     err = np.sqrt(np.var(bins, axis=0))
    #     return err 

    # FIXME build it inside stastType classes
    def cov_func(self, bins, statsType=None):
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
    

    def to_dataStats(self, num_bins, statsType):
        bins = self.bins(num_bins, statsType)
        out = DataStats(self.mean, bins, statsType)
        return out
    
    
    def _make_class(self, mean, bins):
        cov = self.cov_func(bins)
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
            self._statsType = other.statsType
            self.num_bins = other.num_bins()
            out_data = getattr(
                self._data_vectorized, operation
            )(other._data_vectorized)
            self._statsType = None
            out = DataStats(out_data[0], out_data[1:], other.statsType)
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
# DataErr
################################################################################


# class DataErr:
#     """Class for error propagation."""

#     def __init__(self, mean, err=None, *, cov=None, num_bins=2000, seed=None):
#         if not isinstance(mean, (np.ndarray, list)):
#             mean = np.array([mean])

#         if cov is None:
#             if not isinstance(err, np.ndarray):
#                 err = np.array([err])
#             self.cov = np.diag(err**2)
#         elif cov is not None:
#             #assert(np.allclose(err**2, np.diag(cov), atol=1e-15))
#             assert(err is None), "'err' must be 'None' if cov is specified"
#             assert(cov.ndim==2), f"'cov' has to be a 2D array, " \
#                                   f" here ndim={cov.ndim}" \
#                                   f" with shape={cov.shape}"
#             self.cov = cov
#             err = np.sqrt(np.diag(cov))

#         self.seed = seed
#         self.num_resampled_bins = num_bins

#         self.mean = np.asarray(mean)
#         self._bins = None
#         self.err = np.asarray(err)
        
    
#     def num_bins(self):
#         return self.num_resampled_bins

#     def resample(self, num_bins):
#         if num_bins is None:
#             num_bins = self.num_bins()
#         np.random.seed(self.seed)
#         raw_bins = np.random.multivariate_normal(
#             self.mean, self.cov, num_bins
#         )
#         bias = np.mean(raw_bins, 0)-self.mean
#         bins = raw_bins-bias
#         return bins
    
#     @property
#     def bins(self):
#         if self._bins is None:
#             self._bins = self.resample(self.num_bins())
#         return self._bins
    
#     def err_func(self):
#         bins = self.bins
#         err = np.sqrt(np.var(bins, axis=0))
#         return err 

#     def cov_func(self):
#         cov = np.cov(self.bins, rowvar=False)
#         return cov
    

#     # def make_class_from_bins(self, mean, bins):
#     #     dataStats = DataBins(mean, bins)
#     #     err = 



#     def __len__(self):
#         return (len(self.mean))

#     def __repr__(self):
#         prec = 4 # precision
#         space = len('DataErr[')*" "
#         out = f"DataErr["
#         if len(self)>1:
#             out += f"{self.mean[0]: .{prec}e} +- {self.err[0]:.{prec}e},\n" + space
#             for mean, err in zip(self.mean[1:-1], self.err[1:-1]):
#                 out += f"{mean: .{prec}e} +- {err:.{prec}e},\n" + space
#         out += f"{self.mean[-1]: .{prec}e} +- {self.err[-1]:.{prec}e}]"      
#         return out

#     def __str__(self):
#         prec = 5 # precision
#         space = len('DataErr[')*" "
#         out = f"DataErr["
#         if len(self)>1:
#             out += _print_dataStats(self.mean[0], self.err[0], prec) + ",\n"
#             for mean, err in zip(self.mean[1:-1], self.err[1:-1]):
#                 out += space + _print_dataStats(mean, err, prec) + ",\n"
#             out += space + _print_dataStats(self.mean[-1], self.err[-1], prec) + "]"
#         else:
#             out += _print_dataStats(self.mean[0], self.err[0], prec) + "]" 
#         return out
    

#     def to_dataStats(self, num_bins, statsType):
#         bins = self.resample_statsType(num_bins, statsType)
#         out = DataStats(self.mean, bins, statsType)
#         return out
    
#     def resample_statsType(self, num_bins, statsType):
#         np.random.seed(self.seed)
#         raw_bins = np.random.multivariate_normal(self.mean, num_bins*self.cov, num_bins)
#         bias = np.mean(raw_bins, 0)-self.mean
#         raw_bins = raw_bins-bias
#         bins = statsType.generate_bins(raw_bins)
#         return bins

    
#     # generic overload for mathematical operations among 2 DataStats objects
#     def _overload_math_dataStats(self, other, operation):
#         statsType = other.statsType
#         num_bins = other.num_bins()

#         bins = self.resample_statsType(num_bins, statsType)

#         data = DataStats(self.mean, bins, statsType)
#         out_data = getattr(other, operation)(data)
#         return out_data
    
#     def _overload_math_dataErr(self, other, operation):
#         num_bins = max(self.num_bins(), other.num_bins())
#         bins = self.resample(num_bins)
#         bins_other = other.resample(num_bins)

#         out_mean = getattr(self.mean, operation)(other.mean)
#         out_bins = getattr(bins, operation)(bins_other)
#         out_err = self.err_func()
#         out = DataErr(out_mean, err=out_err, num_bins=num_bins)
#         return out
    
#     # generic overload for mathematical operations (following numpy)
#     def _overload_math_numpy(self, other, operation):
#         out_mean = getattr(self.mean, operation)(other)
#         bins = self.resample()
#         out_bins = getattr(bins, operation)(other)
#         #out_err = getattr(self.err, operation)(other)
#         out_err = np.sqrt(np.var(out_bins, axis=0))
#         #out_cov = getattr(self.cov, operation)(other)
#         # recompute covariance with np.cov?
#         out = DataErr(out_mean, err=out_err)
#         return out
    
#     # # math overload

#     def _overload_math(self, other, operation):
#         if isinstance(other, DataErr):
#             out = self._overload_math_dataErr(other, operation)    
#         elif isinstance(other, DataStats):
#             out = self._overload_math_dataStats(other, operation)      
#         else:
#             try:
#                 out = self._overload_math_numpy(other, operation)
#                 return out
#             except:
#                 return NotImplemented
#         return out
    
#     # OVERLOAD OF MATH OPERATIONS
#     def __mul__(self, other):
#         return self._overload_math(other, '__mul__')
    
#     def __rmul__(self, other):
#         return self._overload_math(other, '__rmul__')
            
#     def __truediv__(self, other):
#         return self._overload_math(other, '__truediv__')
    
#     def __rtruediv__(self, other):
#         return self._overload_math(other, '__rtruediv__')
    
#     def __add__(self, other):
#         return self._overload_math(other, '__add__')

#     def __radd__(self, other):
#         return self._overload_math(other, '__radd__')

#     def __sub__(self, other):
#         return self._overload_math(other, '__sub__')

#     def __rsub__(self, other):
#         return self._overload_math(other, '__rsub__')

#     def __pow__(self, other):
#         return self._overload_math(other, '__pow__')

#     def __neg__(self):
#         return -1*self
    
#     def __pos__(self):
#         return +1*self
    
#     def __getitem__(self, key):
#         key_cov = key
#         if isinstance(key, int):
#             key_cov = slice(key, key+1, None)
#         out = DataErr(
#             self.mean[key], cov=self.cov[key_cov,key_cov],
#             num_bins=self.num_resampled_bins, seed=self.seed
#         )
#         return out

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

def merge(*data_in):
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
        out = DataErr(mean, cov=cov)
    return out

def zeros(T, statsType):
    num_bins = statsType.num_bins
    mean = np.zeros(T)
    bins = np.zeros(shape=(num_bins, T))
    out = DataStats(mean, bins, statsType)
    return out

def ones(T, statsType):
    num_bins = statsType.num_bins
    mean = np.ones(T)
    bins = np.ones(shape=(num_bins, T))
    out = DataStats(mean, bins, statsType)
    return out

def empty(T, statsType):
    num_bins = statsType.num_bins
    mean = np.empty(T)
    bins = np.empty(shape=(num_bins, T))
    out = DataStats(mean, bins, statsType)
    return out

def constant(const, statsType):
    return const * ones(1, statsType)

def random(T, statsType):
    num_bins = statsType.num_bins
    mean = np.random.normal(0, 1, T)
    bins = np.random.normal(0, 1, size=(num_bins, T))
    out = DataStats(mean, bins, statsType)
    return out

def uniform(T, statsType, low=0.0, high=1.0):   
    num_bins = statsType.num_bins
    bins = np.random.uniform(low=low, high=high, size=(num_bins, T))
    mean = np.mean(bins, 0)
    out = DataStats(mean, bins, statsType)
    return out

def Z2(T, statsType):   
    num_bins = statsType.num_bins
    bins = np.random.randint(low=-1, high=1, size=(num_bins, T))
    bins = np.where(bins<0, bins, +1)
    mean = np.mean(bins, 0)
    out = DataStats(mean, bins, statsType)
    return out

################################################################################
# DECORATORS
################################################################################

def dataStats_args(func):
    """Decorator to extend a generic function 'func' to allow DataStats
    arguments."""

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
    """Decorator to extend a generic function 'func' to allow DataStats
    arguments with vectorization."""

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
