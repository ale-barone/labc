import h5py
import numpy as np
from .utilities import get_fileID
from .._objectInterface import IReader


class _ReaderBase(IReader):
    
    def __init__(self, file):
        super().__init__(file)
        self._groups = None
        self._datasets = None
        self._paths = None
        self._fileID = None
        
    def _assign_paths(self):
        groups = []
        datasets = []
        with h5py.File(self.file, 'r') as hf:
            def append_path(name, obj):
                if isinstance(obj, h5py._hl.group.Group):
                    groups.append(name) 
                elif isinstance(obj, h5py._hl.dataset.Dataset):
                    datasets.append(name)
            hf.visititems(append_path)

        self._groups = groups
        self._datasets = datasets
        self._paths = [*groups, *datasets]


    @property
    def datasets(self):
        if self._datasets is None:
            self._assign_paths()
        return self._datasets
    
    @property
    def groups(self):
        if self._groups is None:
            self._assign_paths()
        return self._groups
    
    @property
    def paths(self):
        if self._paths is None:
            self._assign_paths()
        return self._paths
    
    @property
    def fileID(self):
        if self._fileID is None:
            self._fileID = get_fileID(self.file)
        return self._fileID
    
    @fileID.setter
    def fileID(self, value):
        if self._fileID is None:
            self._fileID = value
        else:
            raise ValueError(
                "Attribute fileID has been already assigned with value "
                f"'{self._fileID}'"
            )
    
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
