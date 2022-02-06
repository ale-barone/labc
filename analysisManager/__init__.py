# I need to initialize an analysis
# I need to know:
# - dealing with 'gauge' files or 'stats' (is this a problem for the reader? I don't think so, 
#   the reader can just take proper outerClass based on the file extension and assign the 'gauge' or 'stats' according to the analysisInitialization)
#   ? but is this going to undermine the flexibility? I will have to stick with either 'gauge' or 'stats' data.. this is not going to be feasible if I
#     have to store data from fits or similar (those are going to be always 'stats' file) 
# - choosing which statistics to use, either 'jack' or 'boot' and set their paramters, if any
# - (choosing the file format we are going to work with? )

import numpy as np
from LatticeABC import dataManager as _dM
from LatticeABC.dataManager import dataContainer as _dC


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

# I could implement this nicely
class database:
    pass

class analysis:
    """Initializer for the analysis."""

    def __init__(self, statsType, tsrc_list, config_list):
        self.tsrc_list = tsrc_list
        self.config_list = config_list
        self.statsType = statsType
        self.globals = {}
        # self.files = {} # I don't like this solution
        self.database = database()
        self.get = database()
        self.objdatabase = database()
        self.obj = database()
    
    def copy(self):
        new = type(self)(self.statsType, self.tsrc_list, self.config_list)
        new.globals = self.globals
        new.database = self.database
        new._update_database()
        return new

    def print_setup(self):
        _print_title("ANALYSIS SETUP")
        print("# num_config  = ", len(self.config_list))
        print("# num_sources = ", len(self.tsrc_list))
        print("# statsType   = ", 'to be implemented')
        _print_bar()
    
    def dataStats(self, file: str, tsrc_list: list=None) -> _dM.dataStats:
        tsrc_list = self.tsrc_list if tsrc_list is None else tuple(tsrc_list)
        formatter = _dC.formatter(file, self.statsType)
        data = formatter.format(tsrc_list)
        return _dM.dataStats(data, self.statsType)
    
    def dataStats_data(self, data: np.ndarray, tsrc_list: list=None) -> _dM.dataStats:
        tsrc_list = self.tsrc_list if tsrc_list is None else tuple(tsrc_list)
        return _dM.dataStats(data, self.statsType)

    # TODO at the moment it's the same as dataStats
    def corrStats(self, file: str, tsrc_list: list=None) -> _dM.dataStats:
        tsrc_list = self.tsrc_list if tsrc_list is None else tuple(tsrc_list)
        formatter = _dC.formatter(file, self.statsType)
        data = formatter.format(tsrc_list)
        return _dM.corrStats(data, self.statsType)
    
    def add_global(self, name: str, value) -> None:
        """Add global variables to the analysis."""
        if not name in self.globals:
            self.globals[name] = value
        else:
            raise _AlreadyDefined(
                message = f"Global variable '{name}' already defined with value: {value}"
            )
       
    def _update_database(self):
        self.get = database() # clear the 'get' operation

        for an_func in tuple(self.database.__dict__.values()):
            func = an_func(self)
            if not hasattr(self.get, func.__name__):
                setattr(self.get, func.__name__, func)
            else:
                raise _AlreadyDefined(
                    message = f"Database function '{func}' already defined."
                )


    def add_func(self, an_func):
        """Add functions that retrieves the desired data."""
        if not hasattr(self.database, an_func.__name__):
            setattr(self.database, an_func.__name__, an_func)
        else:
            raise _AlreadyDefined(
                message = f"Database raw function '{an_func}' already defined."
            )

        func = an_func(self)
        if not hasattr(self.get, func.__name__):
            setattr(self.get, func.__name__, func)
        else:
            raise _AlreadyDefined(
                message = f"Database function '{func}' already defined."
            )
    
    def add_obj(self, an_obj):
        """Add object (class)."""
        if not hasattr(self.objdatabase, an_obj.__name__):
            setattr(self.objdatabase, an_obj.__name__, an_obj)
        else:
            raise _AlreadyDefined(
                message = f"Class/object '{an_obj}' already defined."
            )

        obj = an_obj(self)
        if not hasattr(self.obj, obj.__name__):
            setattr(self.obj, obj.__name__, obj)
        else:
            raise _AlreadyDefined(
                message = f"Class/obj '{obj}' already defined."
            )
  

    def print_globals(self):
        _print_title("GLOBAL VARIABLES")
        for key, value in self.globals.items():
            print(f"# {key} =", value)
        _print_bar()

  






