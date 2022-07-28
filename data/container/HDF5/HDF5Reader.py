import h5py
import numpy as np
from abc import ABC, abstractmethod
import inspect
from .HDF5Utilities import *
from ..objectInterface import IReader



class _HDF5ReaderBase(IReader):
    def __init__(self, file):
        super().__init__(file)
        self.paths = get_paths(file)
        self.groups = get_groups(file)
        self.datasets = get_datasets(file)   
        self.statsID = get_statsID(file)
    
    def read(self, dset):
        assert(dset in self.datasets), f"dataset '{dset}' must be in datasets list (check 'datasets' attribute)"
        with h5py.File(self.file, 'r') as hf:
            out = np.asarray(hf[dset])
        return out

    def dump(self, re=None):
        num_dsets = len(self.datasets)
        with h5py.File(self.file, 'r') as hf:
            dsets = np.array([hf[self.datasets[d]] for d in range(num_dsets)])
        out_dict = {self.datasets[d]: np.asarray(dsets[d])[re] for d in range(num_dsets)}
        return out_dict 

    def get_structure(self):
        # TODO: I should implement the tree representation here! It would be super nice
        return self.paths


class HDF5Reader: # just as a namespace, for organization
    class generic(_HDF5ReaderBase):
        """Generic reader for HDF5 files."""
        pass    

    class gauge(_HDF5ReaderBase):
        """
        Reader for HDF5 files containing raw gauge data structured as
        /corr/{t_0/{config_0, config_1, ...}, t_1/{config_0, config_1 ...}, ...}
        """

        # TODO: Not good, I need to keep also the generic reader!
        def read_config(self, tsrc, config, re=None):
            path = f'corr/t{tsrc}/config{config}'
            #assert(dset in self.datasets), "'dset' must be in datasets list (check 'datasets' attribute)"
            with h5py.File(self.file, 'r') as hf:
                out = np.asarray(hf[path])[re]
            return out

                   
    class stats(_HDF5ReaderBase):
        """
        Reader for HDF5 files containing resampled data structured as
        /corr/{mean{mean}, err{err}, bins{bin_0, bin_2, ...}}
        """
        
        AVAIL_STATS = {'mean', 'err', 'bins'}

        def read(self, stats):
            if stats in __class__.AVAIL_STATS:
                path = f'/{stats}/{stats}'
                with h5py.File(self.file, 'r') as hf:
                    out = np.asarray(hf[path])
            else:
                raise ValueError(f"stats value '{stats}' is not defined. Avaiblable stats are {__class__.AVAIL_STATS}")
            return out

