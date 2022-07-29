import h5py
import numpy as np
from .utilities import *
from .._objectInterface import IReader


class _ReaderBase(IReader):
    def __init__(self, file):
        super().__init__(file)
        self.paths = get_paths(file)
        self.groups = get_groups(file)
        self.datasets = get_datasets(file)   
        self.fileID = get_fileID(file)
    
    def read(self, dset):
        with h5py.File(self.file, 'r') as hf:
            out = np.asarray(hf[dset])
        return out

    def dump(self):
        with h5py.File(self.file, 'r') as hf:
            out_dict = {d: np.array(hf[d]) for d in self.datasets}
        return out_dict 

    def get_structure(self):
        # TODO: I should implement the tree representation here! It would be super nice
        return self.paths


class Reader: # just as a namespace, for organization
    class generic(_ReaderBase):
        """Generic reader for HDF5 files."""
        pass    

    class gauge(_ReaderBase):
        """
        Reader for HDF5 files containing raw gauge data structured as
        /corr/{t_0/{config_0, config_1, ...}, t_1/{config_0, config_1 ...}, ...}
        """
        pass

                   
    class stats(_ReaderBase):
        """
        Reader for HDF5 files containing resampled data structured as
        /group/mean, /group/err, /group/bins.
        """
        pass
