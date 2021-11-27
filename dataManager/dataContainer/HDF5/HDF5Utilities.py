import h5py
import numpy as np
import os

def get_groups(h5file):
    groups = []
    with h5py.File(h5file, 'r') as hf:
        def append_groups(name, obj):
            if isinstance(obj, h5py._hl.group.Group):
                groups.append(name) 
        hf.visititems(append_groups)
    return groups

def get_datasets(h5file):
    datasets = []
    with h5py.File(h5file, 'r') as hf:
        def append_datasets(name, obj):
            if isinstance(obj, h5py._hl.dataset.Dataset):
                datasets.append(name)
        hf.visititems(append_datasets)
    return datasets

def get_paths(h5file):
    groups = get_groups(h5file)
    datasets = get_datasets(h5file)
    out = np.append(groups, datasets)
    return out



class NonStatsIDFileError(Exception):
    """Custom error to avoid overriding external files."""

    def __init__(self, file, message):
        self.file = file
        super().__init__(message)




def get_statsID(h5file):
    with h5py.File(h5file, 'r') as hf:
        try:
            statsID = hf.attrs['statsID']
            return statsID
        except:
            raise NonStatsIDFileError(
                file=h5file,
                message=f"File '{h5file}' has no attribute 'statsID'."
            )

def assert_statsID(h5file):
    with h5py.File(h5file, 'r') as hf:
        statsID_bool = 'statsID' in hf.attrs
        return statsID_bool

def assign_statsID(h5file):
    if assert_statsID(h5file):
        statsID = get_statsID(h5file)
    else:
        statsID = 'generic'
    return statsID

def check_statsID(h5file):
    if os.path.isfile(h5file):
        if not assert_statsID(h5file):
            raise NonStatsIDFileError(
                file=h5file,
                message=f"File '{h5file}' has no attribute 'statsID'."
            )
    else:
        return True
     
    
