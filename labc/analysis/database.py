import numpy as np
from functools import wraps
from ..data import container as _dC
import copy

class DatabaseFile:
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
    
    
class DatabaseEnsemble:
    
    def __init__(self):
        self._register_file = {}
        self._register_func_raw = {}
        self._register_func_stats = {}
        self._register_func = {}

        self.func_stats = {}

    
    def _apply_stats(self, ensemble):
        # for func in self._register_func_raw.values():
        #     setattr(self, func.__name__, func(self.statsType, self.tsrc_list))
        for func in self._register_func_stats.values():
            setattr(self, func.__name__, func(ensemble))
            self.func_stats[func.__name__] = func(ensemble)

    # def __init__(self):
    #     self._register = {}
    
    # def add_file(self, file_func):
    #     """Add function that returns the file name as an attribute to 
    #     the data base."""

    #     key = file_func.__name__
    #     if not key in self._register.keys():
    #         self._register_file[key] = file_func
    #         setattr(self, file_func.__name__, file_func)
    #     else:
    #         raise ValueError(
    #             message = f"Database file {file_func.__name__} already defined."
    #         )
    #     return file_func
            
    def add_func_raw(self, file_func):
        if not file_func.__name__ in self._register.keys():
            self._register_func_raw[file_func.__name__] = file_func
        else:
            raise ValueError(
                message = f"Database function '{file_func}' already defined."
            )
    
    def add_func_stats(self, file_func):
        if not file_func.__name__ in self._register_func_stats.keys():
            self._register_func_stats[file_func.__name__] = copy.deepcopy(file_func)
        else:
            raise ValueError(
                message = f"Database function '{file_func}' already defined."
            )
    
    def add_func(self, file_func):
        if not file_func.__name__ in self._register.keys():
            self._register_func[file_func.__name__] = file_func
        else:
            raise ValueError(
                message = f"Database function '{file_func}' already defined."
            )
