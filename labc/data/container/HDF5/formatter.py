import numpy as np
import h5py
from .utilities import get_fileID


# I should write something more general independent of the specific extension
# (and I should put some 'check' for the fileID)

class Formatter:

    class gauge:
        def __init__(self, file, statsType):
            self.file = file
            self.statsType = statsType
            self.fileID = get_fileID(file)

        def extract_raw(self, group, tsrc_list):
            with h5py.File(self.file, 'r') as hf:
                all_config_tsrc = np.asarray(
                    [np.asarray(hf[f'{group}/tsrc{tsrc}']) 
                    for tsrc in tsrc_list]
                )
                # shape is (tsrc, config, T)
                all_config = np.mean(all_config_tsrc, 0)
            return all_config
        
        def format(self, group, tsrc_list): # TODO: add config_list
            with h5py.File(self.file, 'r') as hf:
                all_config_tsrc = np.asarray(
                    [np.asarray(hf[f'{group}/tsrc{tsrc}']) 
                    for tsrc in tsrc_list]
                )
                # shape is (tsrc, config, T)
                all_config = np.mean(all_config_tsrc, 0)
                mean, _, bins = self.statsType.generate_stats(all_config)
                out = [mean, bins]
            return tuple(out)
    
    class stats:
        def __init__(self, file, *args):
            self.file = file
            self.statsType = "TODO" # TODO NEED TO IMPLEMENT THIS (reads it from file)
            self.fileID = get_fileID(file)

        # TODO I don't like having *args as a placeholder...
        def format(self, group):
            with h5py.File(self.file, 'r') as hf:
                mean = np.asarray(hf[f'{group}/mean'])
                bins = np.asarray(hf[f'{group}/bins'])
                out = [mean, bins] #_dM.concatenate_stats(mean, err, bins)
            return tuple(out)
