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

# TODO: the following are probably more general functions.. I may consider
# defining them generally and specialize them
# for every file format

class NonStatsIDFileError(Exception):
    """Custom error to avoid overriding external files."""

    def __init__(self, file, message):
        self.file = file
        super().__init__(message)


# def get_fileID(h5file):
#     with h5py.File(h5file, 'r') as hf:
#         try:
#             fileID = hf.attrs['fileID']
#             return fileID
#         except:
#             raise NonStatsIDFileError(
#                 file=h5file,
#                 message=f"File '{h5file}' has no attribute 'fileID'."
#             )

def has_fileID(h5file):
    with h5py.File(h5file, 'r') as hf:
        fileID_bool = 'fileID' in hf.attrs
        return fileID_bool

def get_fileID(h5file):
    if has_fileID(h5file):
        with h5py.File(h5file, 'r') as hf:
            fileID = hf.attrs['fileID']            
    else:
        fileID = 'generic'
    return fileID

def check_fileID(h5file):
    if os.path.isfile(h5file):
        if not has_fileID(h5file):
            raise NonStatsIDFileError(
                file=h5file,
                message=f"File '{h5file}' has no attribute 'fileID'."
            )
    else:
        return True
