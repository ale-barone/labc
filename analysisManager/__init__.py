# I need to initialize an analysis
# I need to know:
# - dealing with 'gauge' files or 'stats' (is this a problem for the reader? I don't think so, 
#   the reader can just take proper outerClass based on the file extension and assign the 'gauge' or 'stats' according to the analysisInitialization)
#   ? but is this going to undermine the flexibility? I will have to stick with either 'gauge' or 'stats' data.. this is not going to be feasible if I
#     have to store data from fits or similar (those are going to be always 'stats' file) 
# - choosing which statistics to use, either 'jack' or 'boot' and set their paramters, if any
# - (choosing the file format we are going to work with? )

from LatticeABC import statsManager as sM
from LatticeABC import dataManager as dM
from LatticeABC.dataManager import dataContainer as dC


class analysis:
    """Initializer for the analysis."""

    def __init__(self, statsType, tsrc_list, config_list): # I need also config list in principle
        self.tsrc_list = tsrc_list
        self.config_list = config_list
        self.statsType = statsType
    
    def dataStats(self, file, tsrc_list=None):
        tsrc_list = self.tsrc_list if tsrc_list is None else tsrc_list
        formatter = dC.formatter(file, self.statsType)
        data = formatter.format(self.tsrc_list)
        return dM.dataStats(data, self.statsType)


