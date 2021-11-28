# I need to initialize an analysis
# I need to know:
# - dealing with 'gauge' files or 'stats' (is this a problem for the reader? I don't think so, 
#   the reader can just take proper outerClass based on the file extension and assign the 'gauge' or 'stats' according to the analysisInitialization)
#   ? but is this going to undermine the flexibility? I will have to stick with either 'gauge' or 'stats' data.. this is not going to be feasible if I
#     have to store data from fits or similar (those are going to be always 'stats' file) 
# - choosing which statistics to use, either 'jack' or 'boot' and set their paramters, if any
# - (choosing the file format we are going to work with? )

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



class analysis:
    """Initializer for the analysis."""

    def __init__(self, statsType, tsrc_list, config_list): # I need also config list in principle
        self.tsrc_list = tsrc_list
        self.config_list = config_list
        self.statsType = statsType
        self.globals = {}
        self.files = [] # I don't like this solution

    def print_setup(self):
        _print_title("ANALYSIS SETUP")
        print("# num_config  = ", len(self.config_list))
        print("# num_sources = ", len(self.tsrc_list))
        print("# statsType   = ", 'to be implemented')
        _print_bar()
    
    def dataStats(self, file: str, tsrc_list: list=None) -> _dM.dataStats:
        tsrc_list = self.tsrc_list if tsrc_list is None else tuple(tsrc_list)
        formatter = _dC.formatter(file, self.statsType)
        data = formatter.format(self.tsrc_list)
        return _dM.dataStats(data, self.statsType)
    
    def add_global(self, name: str, value) -> None:
        """Add global variables to the analysis."""
        if not name in self.globals:
            self.globals[name] = value
        else:
            raise _AlreadyDefined(
                message = f"Global variable '{name}' already defined with value: {value}"
            )
    
    def print_globals(self):
        _print_title("GLOBAL VARIABLES")
        for key, value in self.globals.items():
            print(f"# {key} =", value)
        _print_bar()

    # TODO: I don't like this, needs change
    def add_file(self, file_fun):
        self.files.append(file_fun)






