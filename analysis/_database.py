import numpy as np
from LatticeABC import dataManager as _dM
from LatticeABC.dataManager import dataContainer as _dC
import os
from os.path import dirname, abspath 
from functools import wraps



class Getter:
    
    def dump(self):
        attr = vars(self).keys()
        return attr



class DatabaseFiles:
    
    def __init__(self):
        self.register = {}
    
    def add_file(self, file_func, path):
        """Add file as an attribute to the data base."""
        key = file_func.__name__
        if not key in self.register.keys():

            def file_func_path(file_func):
                @wraps(file_func)
                def wrapper(*args, **kwargs):
                    return path + file_func(*args, **kwargs)
                return wrapper

            self.register[key] = file_func
            setattr(self, file_func.__name__, file_func_path(file_func))
        else:
            raise ValueError(
                message = f"Database file '{file_func}' already defined."
            )

class DatabaseData:
    
    def __init__(self, databaseFiles, *, tsrc_list=None, statsType=None):
        self.databaseFiles = databaseFiles
        self.tsrc_list = tsrc_list
        self.statsType = statsType
        self.register = {}
        
        
        for key in databaseFiles.register.keys():
            file_func = getattr(databaseFiles, key)
            self._add_data(file_func)
            
        
    def _add_data(self, file_func):
        """Add data as an attribute to the data base."""
        if not file_func.__name__ in self.register.keys():
            def data_func(file_func):
                def wrapper(tsrc_list=self.tsrc_list, statsType=self.statsType):
                    @wraps(file_func)
                    def _data_func(*args, **kwargs):
                        formatter = _dC.formatter(file_func(*args, **kwargs), statsType)
                        data = formatter.format(tsrc_list)
                        return _dM.dataStats(*data, statsType)
                    return _data_func
                return wrapper
        
            self.register[file_func.__name__] = data_func(file_func)
            setattr(self, file_func.__name__, data_func(file_func)())
        else:
            raise ValueError(
                message = f"Database function '{file_func}' already defined."
            )
        

class DatabaseFunc:
    
    def __init__(self, databaseData):
        self.databaseData = databaseData
        self.register = {}
        self.get = Getter()

    @staticmethod
    def assign_keywords(_func=None, **kwargs):
        """Decorator to map arguments of a generic function to the desired data."""
        def decorator(func):
            @wraps(func)
            def wrapper(**kwargs_wrapper):
                new_kwargs = {}
                for key in kwargs.keys():
                    if key in kwargs_wrapper.keys():
                        new_kwargs[key] = kwargs_wrapper[key]
                    else:
                        new_kwargs[key] = kwargs[key]
                return func(**new_kwargs)
            return wrapper
        if _func is None:
            return decorator
        else:
            return decorator(_func)
    
    def _register(self, func):
        if not func.__name__ in self.register.keys():
            setattr(self.get, func.__name__, func)
            self.register[func.__name__] = func
        else:
            raise ValueError(
                message = f"Database function '{func}' already defined."
            )

    def add_func(self, _func=None, **kwargs):
        """Add function as an attribute to the data base."""
        if _func is not None:
            self._register(_func)
        else:
            def wrap_decorator(func):
                new_func = self.assign_keywords(_func, **kwargs)(func)
                self._register(new_func)
                return new_func #self._keyword_decorator(_func, **kwargs)(func)
            return wrap_decorator    



