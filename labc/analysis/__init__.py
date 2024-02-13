# I need to initialize an analysis
# I need to know:
# - dealing with 'gauge' files or 'stats' (is this a problem for the reader? I don't think so, 
#   the reader can just take proper outerClass based on the file extension and assign the 'gauge' or 'stats' according to the analysisInitialization)
#   ? but is this going to undermine the flexibility? I will have to stick with either 'gauge' or 'stats' data.. this is not going to be feasible if I
#     have to store data from fits or similar (those are going to be always 'stats' file) 
# - choosing which statistics to use, either 'jack' or 'boot' and set their paramters, if any
# - (choosing the file format we are going to work with? )

import numpy as np
from labc import data as _dM
from labc.data import container as _dC
import os
from os.path import dirname, abspath 

class _AlreadyDefined(Exception):
    """Custom error to avoid overriding global variables."""
    def __init__(self, message):
        super().__init__(message)

def _print_bar() -> None:
    print("==========================================================")

def _print_title(title: str) -> None:
    _print_bar()
    print(f"# {title}")
    _print_bar()

# it would be nice to be have a function (built-in method of class "database")
# for a sort of "analysis_function" which
# - define a quantity of interest (i.e. fit, combination of correlators,...)
# - check if already exist and if it corresponds to the analysis object currently in use
# - if not, it recomputes the quantity on the fly and store the results it into a cache
#   in case it needs to be recomputed somewhere else in the same run
# - (if told so) write actual data from the cache
    
    
class Globals:
    # so far stupid wrapper of a dictionary... maybe we can make a smart use
    # of this

    def __init__(self):
        self._register = {}

    def add_global(self, key, value):
        self._register[key] = value 
    
    def dump(self):
        print(self._register)

    def get(self, key):
        return self._register[key]

        


class Analysis:
    """Initializer for the analysis."""

    def __init__(self, statsType, tsrc_list, config_list):
        self.tsrc_list = tsrc_list
        self.config_list = config_list
        self.statsType = statsType
        self._register_ensemble = {}
        self.globals = {}
        
        # dictionary for global variables of the analysis
        
        # # database for files (saved data)
        # self.database_file = database_file()
        # self.files = database_file()

        # # database for useful function in the analysis
        # # i.e. fits
        # self.database_eval = database_eval()
        # self.eval = database_eval()
        
        # # database that combines files and useful functions
        # # (and allow me to switch from the 2 with self.onthefly)
        # self.database = database()
        # self.get = database()
        # # TODO: think about this a bit better
        # self.onthefly = False
        
        # # database for object that depends on the analysis object
        # self.objdatabase = database()
        # self.obj = database()
    
    # def set_globals(self, globs):
    #     self.globs = globs
    
    def add_global(self, key, value):
        self.globals[key] = value
        

    # def copy(self):
    #     new = type(self)(self.statsType, self.tsrc_list, self.config_list)
    #     new.globals = self.globals
    #     new.database = self.database
    #     new._update_database()
    #     return new

    def print_setup(self):
        _print_title("ANALYSIS SETUP")
        print("# num_config  = ", len(self.config_list))
        print("# num_sources = ", len(self.tsrc_list))
        print("# statsType   = ", 'to be implemented')
        _print_bar()
    
    # def dataStats(self, file: str, tsrc_list: list=None) -> _dM.DataStats:
    #     tsrc_list = self.tsrc_list if tsrc_list is None else tuple(tsrc_list)
    #     formatter = _dC.formatter(file, self.statsType)
    #     data = formatter.format(tsrc_list)
    #     return _dM.DataStats(*data, self.statsType)
    
    # def dataStats_data(self, data: np.ndarray, tsrc_list: list=None) -> _dM.DataStats:
    #     tsrc_list = self.tsrc_list if tsrc_list is None else tuple(tsrc_list)
    #     return _dM.DataStats(data, self.statsType)


    # def add_global(self, name: str, value) -> None:
    #     """Add global variables to the analysis."""
    #     if not name in self.globals: # FIXME: is that correct? Or should be over the keys?
    #         self.globals[name] = value
    #     else:
    #         raise _AlreadyDefined(
    #             message = f"Global variable '{name}' already defined with value: {value}"
    #         )
       
    # def _update_database(self):
    #     self.get = database() # clear the 'get' operation

    #     for an_func in tuple(self.database.__dict__.values()):
    #         func = an_func(self)
    #         if not hasattr(self.get, func.__name__):
    #             setattr(self.get, func.__name__, func)
    #         else:
    #             raise _AlreadyDefined(
    #                 message = f"Database function '{func}' already defined."
    #             )



    def add_ensemble(self, ensemble):
        self._register_ensemble[ensemble.ID] = ensemble
        setattr(self, ensemble.ID, ensemble)
        
    def ensemble(self, ensembleID):
        return self._register_ensemble[ensembleID]

    def list_ensembles(self):
        return list(self._register_ensemble.keys())

    def data(self, ensembleID: str):
        att = getattr(self, ensembleID)
        att2 = getattr(att, 'data')
        return att2

    def add_database_files(self, database):
        self.files = database
    
    # def add_database_analysis(self, database):
    #     self.get = database
    #     self.get.set_stats(self.statsType, self.tsrc_list)

    # def add_database_gauge(self, database):
    #     self.db_gauge = database
    
    # def add_database_data(self, database):
    #     self.db_data = database



    # TODO: use a common method to add_* and see if I can use try..except instead of if-else+raise
    # def add_func(self, an_func):
    #     """Add functions that retrieves the desired data."""
    #     if not hasattr(self.database, an_func.__name__):
    #         setattr(self.database, an_func.__name__, an_func)
    #     else:
    #         raise _AlreadyDefined(
    #             message = f"Database raw function '{an_func}' already defined."
    #         )

    #     func = an_func(self)
    #     if not hasattr(self.get, func.__name__):
    #         setattr(self.get, func.__name__, func)
    #     else:
    #         raise _AlreadyDefined(
    #             message = f"Database function '{func}' already defined."
    #         )
    
    # def add_file(self, file):
    #     """Add functions that retrieves the desired data."""
    #     if not hasattr(self.files, file.__name__):
    #         setattr(self.files, file.__name__, file)
    #     else:
    #         raise _AlreadyDefined(
    #             message = f"Database function '{file}' already defined."
    #         )
    
    # def add_obj(self, an_obj):
    #     """Add object (class)."""
    #     if not hasattr(self.objdatabase, an_obj.__name__):
    #         setattr(self.objdatabase, an_obj.__name__, an_obj)
    #     else:
    #         raise _AlreadyDefined(
    #             message = f"Class/object '{an_obj}' already defined."
    #         )

    #     obj = an_obj(self)
    #     if not hasattr(self.obj, obj.__name__):
    #         setattr(self.obj, obj.__name__, obj)
    #     else:
    #         raise _AlreadyDefined(
    #             message = f"Class/obj '{obj}' already defined."
    #         )
  
    # def print_globals(self):
    #     _print_title("GLOBAL VARIABLES")
    #     for key, value in self.globals.items():
    #         print(f"# {key} =", value)
    #     _print_bar()

  






