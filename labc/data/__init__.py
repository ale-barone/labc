import numpy as np
from math import floor, log10 
from .container import Writer as _Writer
from .utilities import _get_extension


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
def _print_dataStats_bis(mean, err, num_digits=0):
    """Print mean and error in the form (mean(err))e+xx"""   
    power_err = floor(log10(np.abs(err))) if not err==0 else 0 
    power_mean = floor(log10(np.abs(mean))) if not mean==0 else 0
    power_rel = power_err-power_mean
    
    power_str = f"{10**power_mean:.0e}".replace('1e', 'e')

    mean_prec = power_mean-power_err + (num_digits-1)
    err_prec = -power_err + (num_digits-1)

    if -5<power_rel<=0:
        mean_num_digits = mean_prec#power_mean-power_err+1+extra_digits
        mean_str = f"{mean/10**power_mean:.{mean_num_digits}f}"
        err_digits = round(err * 10**(err_prec))
        err_str = f"{err_digits}"
        out = f"{mean_str}({err_str}){power_str}"     
    else:
        err_str = f"{err * 10**(-power_mean): .1e}"
        out = f"({mean_str} +- {err_str} ){power_str}"
    return out


# READER WITH STATS CONNOTATION
class DataStats:    
    """Basic class for data manipulation."""

    def __init__(self, mean, bins, statsType):
        # make sure we always deal with numpy array
        if not isinstance(mean, (np.ndarray, list)):
            mean = np.array([mean])
            bins = np.reshape(bins, (len(bins), len(mean)))

        self._data_vectorized = np.concatenate((np.array([mean]), bins), axis=0)
        self.mean = self._data_vectorized[0]
        self.bins = self._data_vectorized[1:]
        self._err = None

        # stats
        self.statsType = statsType

    @property
    def err(self):
        if self._err is None:
            self._err = self.statsType.err_func(self.mean, self.bins)
        return self._err 
    
    def __repr__(self):
        prec = 4 # precision
        space = len('DataStats[')*" "
        out = f"DataStats["
        if len(self)>1:
            out += f"{self.mean[0]: .4e} +- {self.err[0]:.4e},\n" + space
            for mean, err in zip(self.mean[1:-1], self.err[1:-1]):
                out += f"{mean: .{prec}e} +- {err:.{prec}e},\n" + space
        out += f"{self.mean[-1]: .4e} +- {self.err[-1]:.4e}]"      
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
        
    def num_bins(self):
        return len(self.bins)

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

    
    def _make_dataStats(self, mean, bins):
        return DataStats(mean, bins, self.statsType)
    
    def _make_dataStats_from_data(self, data):
        mean = data[0]
        bins = data[1:]
        return DataStats(mean, bins, self.statsType)

    # not sure if these methods (NOT the dunders) should create a new object...
    def push_front(self, other):
        out_mean = np.append([other], self.mean)
        out_bins = np.array([ np.append([other], self.bins[b]) for b in range(self.num_bins()) ])        
        out = self._make_dataStats(out_mean, out_bins)
        return out
    
    def roll(self, shift):
        out_mean = np.roll(self.mean, shift)
        out_bins = np.roll(self.bins, shift, axis=1)
        out = self._make_dataStats(out_mean, out_bins)
        return out

    def __len__(self):
        return len(self.mean)


    # explicit (slow) check for overload of math operations
    def _check_math(self, other, operation):
        out_mean = getattr(self.mean, operation)(other)
        out_bins = np.apply_along_axis(
            lambda bin: getattr(bin, operation)(other), 1,
            self.bins
        )
        out = self._make_dataStats(out_mean, out_bins)
        return out

    # generic overload for mathematical operations among 2 DataStats objects
    def _overload_math_dataStats(self, other, operation):
        out_data = getattr(self._data_vectorized, operation)(other._data_vectorized)
        out = self._make_dataStats_from_data(out_data)
        return out
    
    # generic overload for mathematical operations (following numpy)
    def _overload_math_numpy(self, other, operation):
        out_data = getattr(self._data_vectorized, operation)(other)
        out = self._make_dataStats_from_data(out_data)
        return out
    
    # math overload
    def _overload_math(self, other, operation):
        if isinstance(other, DataStats):
            out = self._overload_math_dataStats(other, operation)      
        else:
            out = self._overload_math_numpy(other, operation) 
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
        if isinstance(other, DataStats):
            # add some printing
            if np.allclose(self.mean, other.mean, atol=1e-15) \
               and np.allclose(self.bins, other.bins, atol=1e-15):
                return True
            else:
                return False    

    # HOOK ON NUMPY FUNCTIONS -> REDEFINE WHAT HAPPENS WHEN A NUMPY METHOD IS CALLED 
    # UTILITIES

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
    
    # def __array__(self, dtype=object):
    #     return None

    # FIXME: I don't particularly like this trick, it's a bit fishy...
    def __array__(self):
        class ObjWrapper:
            def __init__(self, data):
                self.data = data
        out = ObjWrapper(self)
        return np.asarray(out)

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
            out = self._make_dataStats_from_data(out_data)
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
        out = self._make_dataStats_from_data(out_data)
        return out

    def __getitem__(self, key):
        out_data = self._data_vectorized[:, key]      
        out = self._make_dataStats(out_data[0], out_data[1:])
        return out 
    
    def __setitem__(self, index, data):
        self._data_vectorized[:, index] = data._data_vectorized.flatten()

    def rel_err(self):
        return np.abs(self.err/self.mean)

    def rel_diff(self, other):
        assert(isinstance(other, DataStats))
        out_mean = np.abs((self.mean - other.mean) / self.mean)
        out_bins = np.abs((self.bins - other.bins) / self.bins)
        out = self._make_dataStats(out_mean, out_bins)
        return out 

################################################################################
# UTILITIES
################################################################################

def merge(*data_in):
    if isinstance(data_in[0], list):
        data_in = tuple(data_in[0])
    statsType = data_in[0].statsType

    data_vectorized = np.concatenate([data._data_vectorized for data in data_in], axis=1)
    out = DataStats(data_vectorized[0], data_vectorized[1:], statsType)
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
