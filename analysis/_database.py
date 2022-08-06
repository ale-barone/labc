import numpy as np
from functools import wraps
from LatticeABC import data as _dM
from ..data import container as _dC



class Getter:
    
    def dump(self):
        attr = vars(self).keys()
        return attr

# This classes are intended for the following firs steps of the analysis
# - define the naming convention based on the raw data output exluding 
#   any info on {source, config, extension}
# - add function for the raw data with {source, config, extension} through
#   the help of a Database object that does the job for us
# - finally build the database for the data. This will be the final key object,
#   i.e. the one that knows how retrieve/format/combine data adn will be
#   directly used in the analysis. Steps here are:
#   - define files that stores the raw data as 'gauge' data all together,
#     i.e. a single file for all sources/config
#   - define files that stores data from fits ('stats' type)
#   - define files/functions that define new quantity built from the gauge data
#     and fits. NB: it is not necessary to store every data, for some it's 
#     enough to register the functions that combines the data


class Database:
    """Class for naming convention of raw files, without config ID and
    extension"""

    def __init__(self):
        self._register = {}
    
    def add_file(self, file_func):
        """Add function that returns the file name as an attribute to 
        the data base."""

        key = file_func.__name__
        if not key in self._register.keys():
            self._register[key] = file_func
            setattr(self, file_func.__name__, file_func)
        else:
            raise ValueError(
                message = f"Database file {file_func.__name__} already defined."
            )
        return file_func

class DatabaseWrapped(Database):

    def __init__(self, databaseName, wrapper):
        self._wrapper = wrapper

        register = {}
        if databaseName is not None:
            for name, func in databaseName._register.items():
                register[name] = self._wrapper(func)
                setattr(self, name, self._wrapper(func))
        self._register = register

    def add_file(self, file_func):
        """Add function that returns the file name as an attribute to 
        the data base."""

        key = file_func.__name__
        if not key in self._register.keys():
            self._register[key] = self._wrapper(file_func)
            setattr(self, file_func.__name__, self._wrapper(file_func))
        else:
            raise ValueError(
                message = f"Database file {file_func.__name__} already defined."
            )
        return self._wrapper(file_func)
    
    def add_data(self, group, complex):
        """Add function that returns the file name as an attribute to 
        the data base."""

        out = self._wrapper(group, complex)
        self._register[out] = out
        setattr(self, out.__name__, out)
        return out



# class DatabaseData:
    
#     def __init__(self, database, statsType, tsrc_list):
#         self.database = database
#         self.tsrc_list = tsrc_list
#         self.statsType = statsType
#         self._register = {}
        
        
#         for key in database._register.keys():
#             file_func = getattr(database, key)(self.statsType, self.tsrc_list)
#             self._add_data(file_func)
            
        
#     def _add_data(self, file_func):
#         """Add data as an attribute to the data base."""
#         if not file_func.__name__ in self._register.keys():
#             def data_func(file_func):
#                 def wrapper(tsrc_list=self.tsrc_list, statsType=self.statsType):
#                     @wraps(file_func)
#                     def _data_func(*args, **kwargs):
#                         formatter = _dC.Formatter(file_func(*args, **kwargs), statsType)
#                         data = formatter.format(statsType, tsrc_list)
#                         return _dM.DataStats(*data, statsType)
#                     return _data_func
#                 return wrapper
        
#             self._register[file_func.__name__] = data_func(file_func)
#             setattr(self, file_func.__name__, data_func(file_func))
#         else:
#             raise ValueError(
#                 message = f"Database function '{file_func}' already defined."
#             )
        

class DatabaseAnalysis:
    
    def __init__(self):
        self.tsrc_list = None
        self.statsType = None
        self._register = {}


    def set_stats(self, statsType, tsrc_list):
        self.statsType = statsType
        self.tsrc_list = tsrc_list
        self._apply_stats()
    
    def _apply_stats(self):
        for func in self._register.values():
            setattr(self, func.__name__, func(self.statsType, self.tsrc_list))
            
    def add_func_stats(self, file_func):
        if not file_func.__name__ in self._register.keys():
            self._register[file_func.__name__] = file_func
            # setattr(self, file_func.__name__, file_func)
        else:
            raise ValueError(
                message = f"Database function '{file_func}' already defined."
            )
    
    def add_func(self, file_func):
        if not file_func.__name__ in self._register.keys():
            #self._register_func[file_func.__name__] = file_func
            setattr(self, file_func.__name__, file_func)
        else:
            raise ValueError(
                message = f"Database function '{file_func}' already defined."
            )


# class DatabaseFunc:
    
#     def __init__(self, databaseData):
#         self.databaseData = databaseData
#         self.register = {}
#         self.get = Getter()

#     @staticmethod
#     def assign_keywords(_func=None, **kwargs):
#         """Decorator to map arguments of a generic function to the desired data."""
#         def decorator(func):
#             @wraps(func)
#             def wrapper(**kwargs_wrapper):
#                 new_kwargs = {}
#                 for key in kwargs.keys():
#                     if key in kwargs_wrapper.keys():
#                         new_kwargs[key] = kwargs_wrapper[key]
#                     else:
#                         new_kwargs[key] = kwargs[key]
#                 return func(**new_kwargs)
#             return wrapper
#         if _func is None:
#             return decorator
#         else:
#             return decorator(_func)
    
#     def _register(self, func):
#         if not func.__name__ in self.register.keys():
#             setattr(self.get, func.__name__, func)
#             self.register[func.__name__] = func
#         else:
#             raise ValueError(
#                 message = f"Database function '{func}' already defined."
#             )

#     def add_func(self, _func=None, **kwargs):
#         """Add function as an attribute to the data base."""
#         if _func is not None:
#             self._register(_func)
#         else:
#             def wrap_decorator(func):
#                 new_func = self.assign_keywords(_func, **kwargs)(func)
#                 self._register(new_func)
#                 return new_func #self._keyword_decorator(_func, **kwargs)(func)
#             return wrap_decorator    



