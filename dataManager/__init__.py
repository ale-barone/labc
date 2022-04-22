from tkinter.messagebox import NO
import numpy as np
from math import floor, log10 

from LatticeABC.dataManager import dataContainer as dC
from .Utilities import _get_extension
from LatticeABC.dataManager.dataContainer.HDF5.HDF5Utilities import get_statsID
import matplotlib.pyplot as plt


# print methods
def _print_dataStats(mean, err, prec):
    """Print mean and error in the form (mean +- err)e+xx """
    power_err = floor(log10(np.abs(err)))
    power_mean = floor(log10(np.abs(mean)))
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
def _print_dataStats_bis(mean, err, prec):
    """Print mean and error in the form (mean(err))e+xx"""   
    power_err = floor(log10(np.abs(err)))
    power_mean = floor(log10(np.abs(mean)))
    power_rel = power_err-power_mean
    
    power_str = f"{10**power_mean:.0e}".replace('1e', 'e')
    mean_str = f"{mean/10**power_mean: .{prec}f}"

    if -5<power_rel<0:
        mean_str = f"{mean/10**power_mean: .{power_mean-power_err+1}f}"
        err_digits = int(err * 10**(-power_err+1))
        err_str = f"{err_digits}"
        out = f"{mean_str}({err_str}){power_str}"     
    else:
        err_str = f"{err * 10**(-power_mean): .1e}"
        out = f"({mean_str} +- {err_str} ){power_str}"
    return out


# READER WITH STATS CONNOTATION
class dataStats:    
    """Basic class for data manipulation."""

    def __init__(self, mean, bins, statsType):
        self.mean = mean
        self.bins = bins
        self._data_vectorized = np.concatenate((np.array([mean]), bins), axis=0)
        self._err = None

        # stats
        if statsType is not None:
            self.statsType = statsType
            self.err_func = statsType.errFun

    @property
    def err(self):
        if self._err is None:
            self._err = self.err_func(self.mean, self.bins)
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
        


    def num_bins(self):
        return len(self.bins)

    # TODO: change this into a more efficient factory
    def save(self, file_out):
        ext = _get_extension(file_out)
        if ext=='.h5':
            writer = dC.writer(file_out, 'stats')
            writer.init_groupStructure(self.statsType)
            writer.add_mean(self.mean)
            writer.add_err(self.err)
            writer.add_bins(self.bins)
        else:
            raise NotImplementedError(f"File extension '{ext}' not implemented!")


    
    def _make_dataStats(self, mean, bins):
        return dataStats(mean, bins, self.statsType)
    
    def _make_dataStats_from_data(self, data):
        mean = data[0]
        bins = data[1:]
        return dataStats(mean, bins, self.statsType)

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
        if isinstance(other, dataStats):
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
    
    def __add__(self, other):
        return self._overload_math(other, '__add__')

    def __radd__(self, other):
        return self._overload_math(other, '__radd__')

    def __sub__(self, other):
        return self._overload_math(other, '__sub__')
    
    def __eq__(self, other):
        # np.array_equal ?
        if isinstance(other, dataStats):
            # add some printing
            if np.allclose(self.mean, other.mean, atol=1e-15) and np.allclose(self.bins, other.bins, atol=1e-15):
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
            if isinstance(arg, dataStats):
                out = True
                break
        return out 

    @staticmethod
    def _get_statsType(args):
        for arg in args:
            if isinstance(arg, dataStats):
                out = arg.statsType
                break
        return out                

    @staticmethod
    def _get_num_bins(args):
        if not type(args)==tuple:
            args = list([args])
        for arg in args:
            if isinstance(arg, dataStats):
                num_bins = arg.num_bins()
        return num_bins

    @staticmethod
    def _collect_data_args(args):
        args_mean = []
        for arg in args:
            if isinstance(arg, dataStats):
                arg = arg._data_vectorized
            args_mean.append(arg)
        args_mean = tuple(args_mean)
        return args_mean

    @staticmethod
    def _collect_mean_args(args):
        args_mean = []
        for arg in args:
            if isinstance(arg, dataStats):
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
                if isinstance(arg, dataStats):
                    arg = arg.bins[b]
                args_bin.append(arg)
            args_bins.append(tuple(args_bin))
        
        return args_bins

    @staticmethod
    def _collect_bin_args(args, b):
        args_bin = []
        for arg in args:
            if isinstance(arg, dataStats):
                arg = arg.bins[b]
            args_bin.append(arg)
        args_bin = tuple(args_bin)
        return args_bin
    
    @staticmethod
    def _collect_mean_kwargs(kwargs):
        dict_mean = {}
        for key, value in kwargs.items():
            if isinstance(value, dataStats):
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


    # def __getitem__(self, key):
    #     if self.mean.ndim==0:
    #         if key==0:
    #             out_mean = np.array([self.mean])
    #             out_bins = self.bins
    #             out = self._make_dataStats(out_mean, out_bins)
    #         else:
    #             raise ValueError #TODO: raise appropriate error
    #     else:
    #         out_mean = self.mean[key]
    #         new_size = out_mean.size

    #         out_bins = np.reshape(self.bins[:,key], (self.num_bins(), new_size))  
    #         out = self._make_dataStats(out_mean, out_bins)
    #     return out

    def __getitem__(self, key):
        out_data = self._data_vectorized[:, key]        
        out_mean = out_data[0]
        if not isinstance(out_mean, np.ndarray):
            out_mean = np.array([out_data[0]])
        out_bins = np.reshape(out_data[1:], (self.num_bins(), len(out_mean)))
        out = self._make_dataStats(out_mean, out_bins)
        return out 
    
    # FIXME: define a setitem
    def __setitem__(self, index, data):
        assert(len(data)==1)
        out_mean = self.mean
        out_mean[index] = data.mean
        out_bins = self.bins
        for b in range(self.num_bins()):
            out_bins[b][index] = data.bins[b]
        # out_bins = np.apply_along_axis(
        #     lambda b: b[index],
        #     1, self.bins
        # )
        out = self._make_dataStats(out_mean, out_bins)
        return out


    def rel_diff(self, other):
        assert(isinstance(other, dataStats))
        out_mean = np.abs((self.mean - other.mean) / self.mean)
        out_bins = np.abs((self.bins - other.bins) / self.bins)
        out = self._make_dataStats(out_mean, out_bins)
        return out 


def merge(*data_in):    
    if isinstance(data_in[0], list):
        data_in = tuple(data_in[0])
    if isinstance(data_in[0], np.ndarray): 
        # TODO: implement properly numpy array
        raise TypeError("np.ndarray not yet implemented.")

    n = len(data_in)
    statsType = data_in[0].statsType
    num_bins = data_in[0].num_bins()
    T = len(data_in[0])

    data_in_mean = np.array([data.mean for data in data_in]).flatten()
    data_in_bins = np.empty(shape=(num_bins, n*T))
    for b in range(num_bins):
        data_in_bins[b] = np.array([data_in.bins[b] for data_in in data_in]).flatten()

    out = dataStats(data_in_mean, data_in_bins, statsType)
    return out




def dataStats_args(func):
    """Decorator to extend a generic function 'func' to allow DataStats
    arguments."""

    def wrapper(*args, **kwargs):
        is_data_stats = dataStats._has_dataStats(args)

        if is_data_stats:
            statsType = dataStats._get_statsType(args) 
            num_bins = statsType.num_bins

            args_mean = dataStats._collect_mean_args(args)
            mean = func(*args_mean, **kwargs)    

            args_bins = dataStats._collect_bins_args(args, num_bins)
            bins = []
            for b in range(num_bins):
                bins.append(func(*args_bins[b], **kwargs))
            bins = np.asarray(bins)

            out = dataStats(mean, bins, statsType)
        else:
            out = func(*args, **kwargs)
        return out
    return wrapper


def dataStats_vectorized_args(func):
    """Decorator to extend a generic function 'func' to allow DataStats
    arguments with vectorization."""

    def wrapper(*args, **kwargs):
        is_data_stats = dataStats._has_dataStats(args)

        if is_data_stats:
            statsType = dataStats._get_statsType(args) 

            args_data = dataStats._collect_data_args(args)
            data = func(*args_data, **kwargs)    

            out = dataStats(data[0], data[1:], statsType)
        else:
            out = func(*args, **kwargs)
        return out
    return wrapper


# FIXME: it needs to be revisited, at the moment it feels a bit ad hoc
def dataStats_func(func):
    """Decorator for functions that return functions."""

    def wrapper(*args, **kwargs):
        # check if there is a DataStats object in args
        is_data_stats = dataStats._has_dataStats(args)

        if is_data_stats:
            statsType = dataStats._get_statsType(args) 
            num_bins = statsType.num_bins

            args_mean = dataStats._collect_mean_args(args)
            func_mean = func(*args_mean, **kwargs) 

            args_bins = dataStats._collect_bins_args(args, num_bins)
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

                out = dataStats(mean, bins, statsType)
                return out
            
            out = func_out
        else:
            out = func(*args, **kwargs)
        return out  
    return wrapper



def zeros(T, statsType):
    num_bins = statsType.num_bins
    mean = np.zeros(T)
    bins = np.zeros(shape=(num_bins, T))
    out = dataStats(mean, bins, statsType)
    return out

def ones(T, statsType):
    num_bins = statsType.num_bins
    mean = np.ones(T)
    bins = np.ones(shape=(num_bins, T))
    out = dataStats(mean, bins, statsType)
    return out

def empty(T, statsType):
    num_bins = statsType.num_bins
    mean = np.empty(T)
    bins = np.empty(shape=(num_bins, T))
    out = dataStats(mean, bins, statsType)
    return out

def random(T, statsType):
    num_bins = statsType.num_bins
    mean = np.random.normal(0, 1, T)
    bins = np.random.normal(0, 1, size=(num_bins, T))
    out = dataStats(mean, bins, statsType)
    return out


# need to move this from here!
class corrStats(dataStats):

    def meff(self):
        meff_mean = np.log( self.mean[:-1]/ self.mean[1:] )
        meff_bins = np.log( np.apply_along_axis(lambda x : x[:-1], 1, self.bins ) / np.apply_along_axis(lambda x : x[1:], 1, self.bins ))
              
        meff_stats = self._make_dataStats(meff_mean, meff_bins)
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
