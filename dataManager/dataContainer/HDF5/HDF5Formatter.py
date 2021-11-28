import numpy as np
import h5py
from  LatticeABC.dataManager.Utilities import concatenate_stats
from .HDF5Utilities import get_statsID
from LatticeABC import statsManager as stats


# I should write something more general independent of the specific extension
# (and I should put some 'check' for the statsID)

class HDF5Formatter:
    class gauge:
        def __init__(self, file, statsType):
            self.file = file
            self.statsType = statsType
            self.statsID = get_statsID(file)
        
        def format(self, tsrc_list): # TODO: add config_list
            with h5py.File(self.file, 'r') as hf:
                all_config_tsrc = np.asarray([np.asarray(hf[f'corr/t{tsrc}/t{tsrc}']) for tsrc in tsrc_list])
                all_config = np.mean(all_config_tsrc, 0)
                mean, err, bins = self.statsType.generate_stats(all_config)
                out = concatenate_stats(mean, err, bins)
            return out
    
    class stats:
        def __init__(self, file, *args):
            self.file = file
            self.statsType = "TODO" # TODO NEED TO IMPLEMENT THIS (reads it from file)
            self.statsID = get_statsID(file)


        def format(self):
            with h5py.File(self.file, 'r') as hf:
                mean = np.asarray(hf['mean/mean'])
                err = np.asarray(hf['err/err'])
                bins = np.asarray(hf['bins/bins'])
                out = concatenate_stats(mean, err, bins)
            return out