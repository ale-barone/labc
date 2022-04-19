from tkinter.messagebox import NO
import h5py
import numpy as np
from numba import jit, njit

from math import log10, floor
from LatticeABC.dataManager import dataContainer as dC
from .Utilities import _get_extension
from LatticeABC.dataManager.dataContainer.HDF5.HDF5Utilities import get_statsID
import matplotlib.pyplot as plt


# READER WITH STATS CONNOTATION
class dataStats:    
    """Basic class for data manipulation."""

    def __init__(self, mean, bins, statsType):
        self.mean = mean
        self.bins = bins
        self._err = None

        # stats
        if statsType is not None:
            self.statsType = statsType
            self.err_func = statsType.errFun

    @property
    def err(self):
        if self._err is not None:
            return self._err
        elif self._err is None:
            out_err = self.err_func(self.mean, self.bins)
            self._err = out_err
        return self._err 

    # def __repr__(self) -> str:
    #     def get_power(num):
    #         return floor(log10(num))    

    #     def count_significantDigits(num):
    #         double = 15
    #         power_num = floor(log10(num))
    #         num_exp = f"{num:.{double}e}"
    #         print(num_exp)
    #         digits = str( float(num_exp) / 10**power_num ).replace('.', '')
    #         return len(digits)

    #     def print_single(mean, err):
    #         rel_err = err/mean
    #         power_rel = floor(log10(rel_err))  
    #         power_err = floor(log10(err))
            
    #         if power_rel<-4: 
    #             out = f'{mean:.4e} +- {err:.4e}' 
                
    #         elif power_rel>0:
    #             out = 'large error!'  
                
    #         else:
    #             precision = np.abs(floor(log10(err))-1)
    #             err_out = err / (10**(power_err-1))
    #             if 
    #             out = f'{mean:.{precision}f}({err_out:.0f})'
         
    #         return out

    #     print('[')
    #     for m, e in zip(self.mean, self.err):
    #         print(print_single(m, e), ',')
    #     print(']')

    # TODO: need method to print information nicely

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
        out_mean = getattr(self.mean, operation)(other.mean)
        out_bins = getattr(self.bins, operation)(other.bins)
        out = self._make_dataStats(out_mean, out_bins)
        return out

    # generic overload for mathematical operations (following numpy)
    def _overload_math_numpy(self, other, operation):
        out_mean = getattr(self.mean, operation)(other)
        out_bins = getattr(self.bins, operation)(other)
        out = self._make_dataStats(out_mean, out_bins)
        return out #out_mean, out_bins
    
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
    def __array__(self, dtype=None):
        return None

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        print('ufunc', ufunc)
        # print('method', method)
        print('args', args)
        # print('kwargs', kwargs)
        if method=='__call__' and not ufunc.__name__=='multiply':
            # FIXME: fix order of args as in the decorator
            # args_mean = []
            # for arg in args:
            #     if isinstance(arg, dataStats):
            #         arg = arg.mean
            #     args_mean.append(arg)
            # args_mean = tuple(args_mean)

            # for arg in args:
            #     if isinstance(arg, dataStats):
            #         bins = arg.bins

            # def arg_bins(bin):
            #     new_args_bin = []
            #     for arg in args:
            #         if isinstance(arg, dataStats):
            #             arg = bin
            #         new_args_bin.append(arg)
            #     new_args_bin = tuple(new_args_bin)
            #     return new_args_bin

            # out_mean = ufunc(*args_mean, **kwargs)
            # out_bins = np.apply_along_axis(
            #     lambda bin: ufunc(*arg_bins(bin), **kwargs),
            #     1, bins
            # )

            arg_np = args[:-1]
            arg_dataStats = args[-1]
            out_mean = ufunc(*arg_np, arg_dataStats.mean, **kwargs)
            out_bins = np.apply_along_axis(
                lambda b: ufunc(*arg_np, b, **kwargs),
                1, arg_dataStats.bins
            )
            out = self._make_dataStats(out_mean, out_bins)
            return out
        else:
            return NotImplemented
    
    def __array_function__(self, func, types, args, kwargs):
        # FIXME: consider case of multiple dataStats/speed up (avoid apply_along)
        # print('func', func)
        # print('types', types)
        # print('args', args)
        # print('kwargs', kwargs)           

        new_args_mean = []
        for arg in args:
            if isinstance(arg, dataStats):
                arg = arg.mean
            new_args_mean.append(arg)
        new_args_mean = tuple(new_args_mean)
        
        for arg in args:
            if isinstance(arg, dataStats):
                bins = arg.bins

        def arg_bins(bin):
            new_args_bin = []
            for arg in args:
                if isinstance(arg, dataStats):
                    arg = bin
                new_args_bin.append(arg)
            new_args_bin = tuple(new_args_bin)
            return new_args_bin

        out_mean = func(*new_args_mean, **kwargs)
        
        out_bins = np.apply_along_axis(
            lambda b: func(*arg_bins(b), **kwargs),
            1, bins
        )
        out = self._make_dataStats(out_mean, out_bins)
        return out


    def __getitem__(self, key):
        if self.mean.ndim==0:
            if key==0:
                out_mean = np.array([self.mean])
                out_bins = self.bins
                out = self._make_dataStats(out_mean, out_bins)
            else:
                raise ValueError #TODO: raise appropriate error
        else:
            out_mean = self.mean[key]
            new_size = out_mean.size

            out_bins = np.reshape(self.bins[:,key], (self.num_bins(), new_size))  
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


# FIXME: it needs to be revisited, at the moment it feels a bit ad hoc
def func_of_data_stats(func):
    """Decorator for functions that return functions."""

    def wrapper(*args, **kwargs):
        # check if there is a DataStats object in args
        is_data_stats = False
        for arg in args:
            if isinstance(arg, dataStats):
                data_stats = arg
                num_bins = data_stats.num_bins()
                is_data_stats = True

        if is_data_stats:
            # mean
            args_mean = []
            for arg in args:
                if isinstance(arg, dataStats):
                    arg = arg.mean
                args_mean.append(arg)
            args_mean = tuple(args_mean)  

            func_mean = func(*args_mean, **kwargs)
            
            # bins
            def arg_bins(b):
                args_bin = []
                for arg in args:
                    if isinstance(arg, dataStats):
                        arg = arg.bins[b]
                    args_bin.append(arg)
                return tuple(args_bin)

            def func_bins(*args_bins, **kwargs_bins):      
                out = []
                for b in range(num_bins):
                    out.append(func(*arg_bins(b), **kwargs)(*args_bins, **kwargs_bins))
                return np.asarray(out)
            
            # final output function
            def func_out(*args, **kwargs):
                mean = func_mean(*args, **kwargs)
                bins = func_bins(*args, **kwargs)

                out = dataStats(mean, bins, data_stats.statsType)
                return out
            
            out = func_out
        else:
            out = func(*args, **kwargs)
        return out  
    return wrapper




def dataStats_func(func):

    def wrapper(*args, **kwargs):
        is_data_stats = False
        for arg in args:
            if isinstance(arg, dataStats):
                data_stats = arg
                num_bins = data_stats.num_bins()
                is_data_stats = True

        if is_data_stats:
            args_mean = []
            for arg in args:
                if isinstance(arg, dataStats):
                    arg = arg.mean
                args_mean.append(arg)
            args_mean = tuple(args_mean)   

            mean = func(*args_mean, **kwargs)     

            def arg_bins(b):
                args_bin = []
                for arg in args:
                    if isinstance(arg, dataStats):
                        arg = arg.bins[b]
                    args_bin.append(arg)
                return tuple(args_bin)

            bins = []
            for b in range(num_bins):
                bins.append(func(*arg_bins(b), **kwargs))
            bins = np.asarray(bins)

            out = dataStats(mean, bins, data_stats.statsType)
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
